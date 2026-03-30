"""
实验数据导入器
统一接口导入各种来源的实验数据
"""
import os
import json
import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from pathlib import Path
import numpy as np
import logging

from .data_formats import (
    CrystalStructure, ExperimentalProperty, ExperimentalDataset,
    DataFormatHandler, CIFHandler, POSCARHandler, JSONHandler,
    read_structure, read_properties
)
from .data_sources import (
    DatabaseConnector, DatabaseManager, SearchQuery, SearchResult,
    DatabaseType
)


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImportConfig:
    """导入配置"""
    data_dir: Optional[str] = None
    cache_enabled: bool = True
    validate_on_import: bool = True
    auto_convert_units: bool = True
    missing_value_strategy: str = 'warn'  # 'ignore', 'warn', 'error'
    property_mapping: Dict[str, str] = field(default_factory=dict)
    unit_conversions: Dict[str, Callable[[float], float]] = field(default_factory=dict)


@dataclass
class ImportResult:
    """导入结果"""
    success: bool
    dataset: Optional[ExperimentalDataset] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        status = "✓ 成功" if self.success else "✗ 失败"
        return f"ImportResult({status}, errors={len(self.errors)}, warnings={len(self.warnings)})"


class DataImporter(ABC):
    """数据导入器抽象基类"""
    
    def __init__(self, config: ImportConfig):
        self.config = config
    
    @abstractmethod
    def import_data(self, source: Union[str, Path, Dict], **kwargs) -> ImportResult:
        """导入数据"""
        pass
    
    @abstractmethod
    def can_import(self, source: Union[str, Path]) -> bool:
        """检查是否可以导入"""
        pass
    
    def validate_dataset(self, dataset: ExperimentalDataset) -> Tuple[bool, List[str]]:
        """验证数据集"""
        errors = []
        
        if dataset.structure is None:
            errors.append("缺少晶体结构数据")
        else:
            if dataset.structure.num_atoms == 0:
                errors.append("结构中没有原子")
            if dataset.structure.lattice.volume <= 0:
                errors.append("晶格体积无效")
        
        for prop in dataset.properties:
            if np.isnan(prop.value) or np.isinf(prop.value):
                errors.append(f"属性 {prop.name} 包含无效数值")
        
        return len(errors) == 0, errors


class FileImporter(DataImporter):
    """文件数据导入器"""
    
    def __init__(self, config: ImportConfig):
        super().__init__(config)
        self._handlers: Dict[str, DataFormatHandler] = {
            '.cif': CIFHandler(),
            '.poscar': POSCARHandler(),
            '.vasp': POSCARHandler(),
            '.json': JSONHandler()
        }
    
    def can_import(self, source: Union[str, Path]) -> bool:
        """检查文件是否可导入"""
        if isinstance(source, str):
            source = Path(source)
        
        if not source.exists():
            return False
        
        ext = source.suffix.lower()
        return ext in self._handlers
    
    def import_data(self, source: Union[str, Path], **kwargs) -> ImportResult:
        """从文件导入数据"""
        if isinstance(source, str):
            source = Path(source)
        
        if not source.exists():
            return ImportResult(
                success=False,
                errors=[f"文件不存在: {source}"]
            )
        
        ext = source.suffix.lower()
        handler = self._handlers.get(ext)
        
        if handler is None:
            # 尝试通过文件名检测
            if source.name.upper() in ['POSCAR', 'CONTCAR']:
                handler = POSCARHandler()
            else:
                return ImportResult(
                    success=False,
                    errors=[f"不支持的文件格式: {ext}"]
                )
        
        try:
            # 读取结构
            structure = handler.read_structure(str(source))
            
            # 读取属性
            properties = handler.read_properties(str(source))
            
            # 创建数据集
            dataset = ExperimentalDataset(
                structure=structure,
                properties=properties,
                source_database='local_file',
                entry_id=str(source),
                metadata={'filepath': str(source), 'import_time': str(np.datetime64('now'))}
            )
            
            # 验证
            errors = []
            if self.config.validate_on_import:
                valid, errors = self.validate_dataset(dataset)
                if not valid:
                    return ImportResult(success=False, errors=errors)
            
            return ImportResult(
                success=True,
                dataset=dataset,
                warnings=errors,
                metadata={'format': ext, 'filename': source.name}
            )
            
        except Exception as e:
            logger.error(f"导入文件失败 {source}: {e}")
            return ImportResult(
                success=False,
                errors=[str(e)]
            )


class CSVImporter(DataImporter):
    """CSV表格数据导入器"""
    
    def __init__(self, config: ImportConfig):
        super().__init__(config)
    
    def can_import(self, source: Union[str, Path]) -> bool:
        if isinstance(source, str):
            source = Path(source)
        return source.suffix.lower() == '.csv'
    
    def import_data(self, source: Union[str, Path], 
                   formula_column: str = 'formula',
                   property_columns: Optional[List[str]] = None,
                   unit_columns: Optional[Dict[str, str]] = None,
                   **kwargs) -> ImportResult:
        """从CSV导入数据"""
        if isinstance(source, str):
            source = Path(source)
        
        try:
            with open(source, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if not rows:
                return ImportResult(success=False, errors=["CSV文件为空"])
            
            # 使用第一行创建数据集
            row = rows[0]
            
            properties = []
            columns = property_columns or [k for k in row.keys() if k != formula_column]
            
            for col in columns:
                if col in row:
                    try:
                        value = float(row[col])
                        unit = unit_columns.get(col, '') if unit_columns else ''
                        
                        prop = ExperimentalProperty(
                            name=col,
                            value=value,
                            unit=unit
                        )
                        properties.append(prop)
                    except (ValueError, TypeError):
                        continue
            
            dataset = ExperimentalDataset(
                properties=properties,
                source_database='csv',
                entry_id=str(source),
                metadata={'row_count': len(rows), 'columns': list(row.keys())}
            )
            
            return ImportResult(success=True, dataset=dataset)
            
        except Exception as e:
            return ImportResult(success=False, errors=[str(e)])


class DatabaseImporter(DataImporter):
    """数据库导入器"""
    
    def __init__(self, config: ImportConfig, connector: DatabaseConnector):
        super().__init__(config)
        self.connector = connector
    
    def can_import(self, source: Union[str, Path]) -> bool:
        """数据库导入总是可用"""
        return True
    
    def import_data(self, entry_id: str, 
                   fetch_structure: bool = True,
                   fetch_properties: bool = True,
                   **kwargs) -> ImportResult:
        """从数据库导入条目"""
        errors = []
        warnings = []
        
        structure = None
        properties = []
        
        # 获取结构
        if fetch_structure:
            try:
                structure = self.connector.get_structure(entry_id)
                if structure is None:
                    warnings.append(f"无法获取条目 {entry_id} 的结构")
            except Exception as e:
                errors.append(f"获取结构失败: {e}")
        
        # 获取属性
        if fetch_properties:
            try:
                properties = self.connector.get_properties(entry_id)
            except Exception as e:
                errors.append(f"获取属性失败: {e}")
        
        if not structure and not properties:
            return ImportResult(
                success=False,
                errors=errors + ["未获取到任何数据"]
            )
        
        dataset = ExperimentalDataset(
            structure=structure,
            properties=properties,
            source_database=self.connector.database_type.value,
            entry_id=entry_id
        )
        
        # 验证
        if self.config.validate_on_import:
            valid, val_errors = self.validate_dataset(dataset)
            if not valid:
                errors.extend(val_errors)
        
        return ImportResult(
            success=len(errors) == 0,
            dataset=dataset,
            errors=errors,
            warnings=warnings,
            metadata={'database': self.connector.database_type.value}
        )
    
    def search_and_import(self, query: SearchQuery, limit: int = 10) -> List[ImportResult]:
        """搜索并导入多个条目"""
        results = []
        search_results = self.connector.search(query, limit)
        
        for sr in search_results:
            result = self.import_data(sr.entry_id)
            results.append(result)
        
        return results


class BatchImporter:
    """批量导入器"""
    
    def __init__(self, config: ImportConfig):
        self.config = config
        self.importers: List[DataImporter] = [
            FileImporter(config),
            CSVImporter(config)
        ]
        self._imported: List[ImportResult] = []
    
    def add_importer(self, importer: DataImporter):
        """添加导入器"""
        self.importers.append(importer)
    
    def import_file(self, filepath: Union[str, Path]) -> ImportResult:
        """导入单个文件"""
        for importer in self.importers:
            if importer.can_import(filepath):
                result = importer.import_data(filepath)
                self._imported.append(result)
                return result
        
        return ImportResult(
            success=False,
            errors=[f"没有可用的导入器处理: {filepath}"]
        )
    
    def import_directory(self, directory: Union[str, Path], 
                        pattern: str = "*",
                        recursive: bool = True) -> List[ImportResult]:
        """导入整个目录"""
        if isinstance(directory, str):
            directory = Path(directory)
        
        results = []
        
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))
        
        for filepath in files:
            if filepath.is_file():
                result = self.import_file(filepath)
                results.append(result)
        
        return results
    
    def get_successful(self) -> List[ExperimentalDataset]:
        """获取成功导入的数据集"""
        return [r.dataset for r in self._imported if r.success and r.dataset]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取导入统计"""
        total = len(self._imported)
        successful = sum(1 for r in self._imported if r.success)
        failed = total - successful
        
        all_errors = []
        for r in self._imported:
            all_errors.extend(r.errors)
        
        error_types = {}
        for error in all_errors:
            error_type = error.split(':')[0] if ':' in error else 'unknown'
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'error_summary': error_types
        }


class UnitConverter:
    """单位转换器"""
    
    # 单位转换因子（转换为SI单位）
    CONVERSIONS = {
        # 长度
        'angstrom': 1e-10, 'a': 1e-10, 'å': 1e-10,  # lowercase å
        'nm': 1e-9, 'um': 1e-6, 'mm': 1e-3, 'cm': 1e-2, 'm': 1.0,
        'pm': 1e-12, 'fm': 1e-15,
        # 能量
        'ev': 1.60218e-19, 
        'rydberg': 2.17987e-18, 'ry': 2.17987e-18,
        'hartree': 4.35974e-18, 'ha': 4.35974e-18,
        'j': 1.0, 'kj/mol': 1.66054e-21, 'kj': 1000.0,
        # 压力
        'gpa': 1e9, 
        'mpa': 1e6, 
        'kpa': 1e3, 
        'pa': 1.0, 
        'bar': 1e5, 'atm': 1.01325e5,
        'mbar': 100.0, 'kbar': 1e8,
        # 温度 (特殊处理)
        'k': 1.0, 
        # 时间
        'fs': 1e-15, 'ps': 1e-12, 'ns': 1e-9, 'us': 1e-6, 'ms': 1e-3, 's': 1.0,
        # 密度
        'g/cm^3': 1000.0, 'g/cm3': 1000.0, 'kg/m^3': 1.0, 'kg/m3': 1.0,
    }
    
    @classmethod
    def convert(cls, value: float, from_unit: str, to_unit: str) -> float:
        """单位转换"""
        if from_unit == to_unit:
            return value
        
        # 标准化单位名称
        from_unit_norm = cls.normalize_unit(from_unit)
        to_unit_norm = cls.normalize_unit(to_unit)
        
        if from_unit_norm == to_unit_norm:
            return value
        
        # 特殊处理温度转换
        if from_unit_norm.lower() == 'k' and to_unit_norm.lower() == 'c':
            return value - 273.15
        if from_unit_norm.lower() == 'c' and to_unit_norm.lower() == 'k':
            return value + 273.15
        if from_unit_norm.lower() == 'f' and to_unit_norm.lower() == 'c':
            return (value - 32) * 5/9
        if from_unit_norm.lower() == 'c' and to_unit_norm.lower() == 'f':
            return value * 9/5 + 32
        if from_unit_norm.lower() == 'k' and to_unit_norm.lower() == 'f':
            return (value - 273.15) * 9/5 + 32
        if from_unit_norm.lower() == 'f' and to_unit_norm.lower() == 'k':
            return (value - 32) * 5/9 + 273.15
        
        from_conv = cls.CONVERSIONS.get(from_unit_norm.lower())
        to_conv = cls.CONVERSIONS.get(to_unit_norm.lower())
        
        if from_conv is None or to_conv is None:
            raise ValueError(f"不支持的单位转换: {from_unit} -> {to_unit}")
        
        # 转换到SI，再转换到目标单位
        si_value = value * from_conv
        return si_value / to_conv
    
    @classmethod
    def normalize_unit(cls, unit: str) -> str:
        """标准化单位名称"""
        unit_lower = unit.lower()
        unit_map = {
            'angstrom': 'Å', 'a': 'Å', 'ang': 'Å', 'å': 'Å',
            'ev': 'eV',
            'gpa': 'GPa',
            'mpa': 'MPa',
            'kpa': 'KPa',
            'pa': 'Pa',
            'k': 'K', 'kelvin': 'K',
            'c': 'C', 'celsius': 'C',
            'f': 'F', 'fahrenheit': 'F',
            'nm': 'nm', 'nanometer': 'nm',
            'um': 'um', 'micrometer': 'um', 'micron': 'um',
            'mm': 'mm', 'millimeter': 'mm',
            'cm': 'cm', 'centimeter': 'cm',
            'm': 'm', 'meter': 'm',
            's': 's', 'second': 's',
            'fs': 'fs', 'femtosecond': 'fs',
            'ps': 'ps', 'picosecond': 'ps',
            'ns': 'ns', 'nanosecond': 'ns',
        }
        normalized = unit_map.get(unit_lower)
        if normalized:
            return normalized
        # 如果包含Å字符，直接返回
        if 'Å' in unit:
            return 'Å'
        return unit


class PropertyNormalizer:
    """属性标准化器"""
    
    # 属性名称映射
    PROPERTY_ALIASES = {
        'band_gap': ['bandgap', 'band gap', 'eg', 'e_g', 'e_gap', 'gap'],
        'formation_energy': ['e_form', 'formation', 'delta_e', 'delta_e_form'],
        'lattice_constant': ['a0', 'a_0', 'lattice', 'lattice_parameter'],
        'bulk_modulus': ['b0', 'b_0', 'bulk', 'k0'],
        'shear_modulus': ['g0', 'g', 'shear'],
        'youngs_modulus': ['e', 'young', 'elastic_modulus'],
        'density': ['rho', 'mass_density'],
        'energy_per_atom': ['energy/atom', 'e/atom'],
    }
    
    # 标准单位
    STANDARD_UNITS = {
        'band_gap': 'eV',
        'formation_energy': 'eV/atom',
        'lattice_constant': 'Å',
        'bulk_modulus': 'GPa',
        'shear_modulus': 'GPa',
        'youngs_modulus': 'GPa',
        'density': 'g/cm^3',
        'energy_per_atom': 'eV/atom',
    }
    
    @classmethod
    def normalize_name(cls, name: str) -> str:
        """标准化属性名称"""
        name_lower = name.lower().replace(' ', '_').replace('-', '_')
        
        for standard, aliases in cls.PROPERTY_ALIASES.items():
            if name_lower in [a.lower() for a in aliases] or name_lower == standard:
                return standard
        
        return name_lower
    
    @classmethod
    def get_standard_unit(cls, property_name: str) -> Optional[str]:
        """获取标准单位"""
        normalized = cls.normalize_name(property_name)
        return cls.STANDARD_UNITS.get(normalized)
    
    @classmethod
    def normalize_property(cls, prop: ExperimentalProperty) -> ExperimentalProperty:
        """标准化属性"""
        normalized_name = cls.normalize_name(prop.name)
        standard_unit = cls.get_standard_unit(normalized_name)
        
        if standard_unit and prop.unit != standard_unit:
            try:
                converted_value = UnitConverter.convert(
                    prop.value, prop.unit, standard_unit
                )
                return ExperimentalProperty(
                    name=normalized_name,
                    value=converted_value,
                    unit=standard_unit,
                    uncertainty=prop.uncertainty,
                    temperature=prop.temperature,
                    pressure=prop.pressure,
                    method=prop.method,
                    reference=prop.reference
                )
            except (ValueError, KeyError):
                # 转换失败，保留原值
                pass
        
        return ExperimentalProperty(
            name=normalized_name,
            value=prop.value,
            unit=prop.unit,
            uncertainty=prop.uncertainty,
            temperature=prop.temperature,
            pressure=prop.pressure,
            method=prop.method,
            reference=prop.reference
        )


# =============================================================================
# 便捷函数
# =============================================================================

def import_from_file(filepath: str, validate: bool = True) -> ImportResult:
    """从文件导入实验数据"""
    config = ImportConfig(validate_on_import=validate)
    importer = FileImporter(config)
    return importer.import_data(filepath)


def import_from_database(entry_id: str, 
                        db_type: DatabaseType = DatabaseType.MATERIALS_PROJECT,
                        api_key: Optional[str] = None) -> ImportResult:
    """从数据库导入实验数据"""
    from .data_sources import create_mp_connector
    
    config = ImportConfig()
    
    if db_type == DatabaseType.MATERIALS_PROJECT:
        connector = create_mp_connector(api_key)
    else:
        raise ValueError(f"不支持的数据库类型: {db_type}")
    
    importer = DatabaseImporter(config, connector)
    return importer.import_data(entry_id)


def batch_import(directory: str, pattern: str = "*") -> List[ImportResult]:
    """批量导入目录中的文件"""
    config = ImportConfig()
    batch = BatchImporter(config)
    return batch.import_directory(directory, pattern)


def normalize_experimental_properties(properties: List[ExperimentalProperty]) -> List[ExperimentalProperty]:
    """标准化实验属性列表"""
    return [PropertyNormalizer.normalize_property(p) for p in properties]


# =============================================================================
# 演示
# =============================================================================

def demo():
    """演示数据导入功能"""
    print("=" * 80)
    print("📥 实验数据导入器演示")
    print("=" * 80)
    
    # 创建导入配置
    config = ImportConfig(
        validate_on_import=True,
        auto_convert_units=True
    )
    
    print("\n🔹 导入配置:")
    print(f"   验证导入: {config.validate_on_import}")
    print(f"   自动单位转换: {config.auto_convert_units}")
    
    # 文件导入器
    print("\n🔹 文件导入器:")
    file_importer = FileImporter(config)
    
    # 检查支持的格式
    test_files = [
        'structure.cif',
        'POSCAR',
        'data.json',
        'results.csv',
        'unknown.xyz'
    ]
    for f in test_files:
        can_import = file_importer.can_import(f)
        status = "✓" if can_import else "✗"
        print(f"   {status} {f}")
    
    # 批量导入器
    print("\n🔹 批量导入器:")
    batch = BatchImporter(config)
    print(f"   已注册导入器: {len(batch.importers)}")
    
    # 单位转换演示
    print("\n🔹 单位转换:")
    conversions = [
        (5.0, 'GPa', 'Pa'),
        (300, 'K', 'C'),
        (3.5, 'eV', 'J'),
        (4.2, 'Å', 'nm'),
    ]
    for value, from_u, to_u in conversions:
        result = UnitConverter.convert(value, from_u, to_u)
        print(f"   {value} {from_u} = {result:.4e} {to_u}")
    
    # 属性标准化
    print("\n🔹 属性名称标准化:")
    test_names = [
        'bandgap', 'Band Gap', 'E_g', 'EG',
        'formation energy', 'E_form', 'DeltaE',
        'lattice parameter', 'a0'
    ]
    for name in test_names:
        normalized = PropertyNormalizer.normalize_name(name)
        standard_unit = PropertyNormalizer.get_standard_unit(normalized)
        print(f"   {name:20s} → {normalized:20s} (单位: {standard_unit or 'N/A'})")
    
    # 模拟属性标准化
    print("\n🔹 属性标准化示例:")
    prop = ExperimentalProperty(
        name='bandgap',
        value=2.5,
        unit='eV',
        uncertainty=0.1
    )
    normalized = PropertyNormalizer.normalize_property(prop)
    print(f"   原始: {prop.name} = {prop.value} {prop.unit}")
    print(f"   标准化: {normalized.name} = {normalized.value} {normalized.unit}")
    
    print("\n" + "=" * 80)
    print("✅ 数据导入器演示完成!")
    print("=" * 80)


if __name__ == '__main__':
    demo()
