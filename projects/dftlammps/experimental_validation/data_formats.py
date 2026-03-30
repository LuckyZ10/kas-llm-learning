"""
实验数据验证模块 - 数据格式处理
支持CIF、POSCAR、JSON等多种晶体结构格式
"""
import os
import re
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import numpy as np
from datetime import datetime


class DataFormat(Enum):
    """支持的数据格式"""
    CIF = "cif"
    POSCAR = "poscar"
    JSON = "json"
    YAML = "yaml"
    CSV = "csv"
    EXCEL = "xlsx"
    HDF5 = "h5"
    UNKNOWN = "unknown"


@dataclass
class Lattice:
    """晶格参数"""
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    
    def to_matrix(self) -> np.ndarray:
        """转换为3x3矩阵"""
        a, b, c = self.a, self.b, self.c
        alpha, beta, gamma = np.radians([self.alpha, self.beta, self.gamma])
        
        # 构建晶格矩阵
        v = np.sqrt(1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 
                    + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
        
        lattice_matrix = np.array([
            [a, b * np.cos(gamma), c * np.cos(beta)],
            [0, b * np.sin(gamma), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)],
            [0, 0, c * v / np.sin(gamma)]
        ])
        return lattice_matrix
    
    @property
    def volume(self) -> float:
        """计算晶胞体积"""
        matrix = self.to_matrix()
        return abs(np.linalg.det(matrix))
    
    def __str__(self) -> str:
        return (f"Lattice(a={self.a:.4f}, b={self.b:.4f}, c={self.c:.4f}, "
                f"α={self.alpha:.2f}°, β={self.beta:.2f}°, γ={self.gamma:.2f}°)")


@dataclass
class AtomSite:
    """原子位点"""
    element: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    wyckoff: Optional[str] = None
    label: Optional[str] = None
    b_iso: Optional[float] = None  # 各向同性温度因子
    
    @property
    def fractional_coords(self) -> np.ndarray:
        """分数坐标"""
        return np.array([self.x, self.y, self.z])
    
    def to_cartesian(self, lattice: Lattice) -> np.ndarray:
        """转换为笛卡尔坐标"""
        return lattice.to_matrix() @ self.fractional_coords


@dataclass
class CrystalStructure:
    """晶体结构数据"""
    formula: str
    lattice: Lattice
    sites: List[AtomSite] = field(default_factory=list)
    space_group: Optional[str] = None
    space_group_number: Optional[int] = None
    source: Optional[str] = None
    source_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_atoms(self) -> int:
        return len(self.sites)
    
    @property
    def elements(self) -> List[str]:
        """获取所有元素（去重）"""
        return list(set(site.element for site in self.sites))
    
    @property
    def composition(self) -> Dict[str, int]:
        """化学组成计数"""
        comp = {}
        for site in self.sites:
            comp[site.element] = comp.get(site.element, 0) + 1
        return comp
    
    def get_site_by_element(self, element: str) -> List[AtomSite]:
        """按元素筛选位点"""
        return [site for site in self.sites if site.element == element]
    
    def __str__(self) -> str:
        return (f"CrystalStructure({self.formula}, {self.lattice}, "
                f"{self.num_atoms} atoms, space_group={self.space_group})")


@dataclass
class ExperimentalProperty:
    """实验测量属性"""
    name: str
    value: float
    unit: str
    uncertainty: Optional[float] = None
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    method: Optional[str] = None  # 测量方法
    instrument: Optional[str] = None  # 仪器信息
    reference: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def relative_uncertainty(self) -> Optional[float]:
        """相对不确定度"""
        if self.uncertainty is not None and self.value != 0:
            return abs(self.uncertainty / self.value)
        return None
    
    def __str__(self) -> str:
        unc = f" ± {self.uncertainty}" if self.uncertainty else ""
        return f"{self.name}: {self.value}{unc} {self.unit}"


@dataclass
class ExperimentalDataset:
    """实验数据集"""
    structure: Optional[CrystalStructure] = None
    properties: List[ExperimentalProperty] = field(default_factory=list)
    source_database: Optional[str] = None
    entry_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_property(self, prop: ExperimentalProperty):
        self.properties.append(prop)
    
    def get_property(self, name: str) -> Optional[ExperimentalProperty]:
        """获取特定属性"""
        for prop in self.properties:
            if prop.name.lower() == name.lower():
                return prop
        return None
    
    def get_properties_by_temperature(self, temp: float, tolerance: float = 1.0) -> List[ExperimentalProperty]:
        """按温度筛选属性"""
        return [p for p in self.properties 
                if p.temperature is not None and abs(p.temperature - temp) <= tolerance]
    
    def __str__(self) -> str:
        props = ", ".join([p.name for p in self.properties[:3]])
        if len(self.properties) > 3:
            props += f" ... (+{len(self.properties)-3} more)"
        return f"ExperimentalDataset({self.source_database}:{self.entry_id}, properties=[{props}])"


class DataFormatHandler(ABC):
    """数据格式处理抽象基类"""
    
    @abstractmethod
    def detect_format(self, filepath: str) -> bool:
        """检测文件格式是否匹配"""
        pass
    
    @abstractmethod
    def read_structure(self, filepath: str) -> CrystalStructure:
        """读取晶体结构"""
        pass
    
    @abstractmethod
    def read_properties(self, filepath: str) -> List[ExperimentalProperty]:
        """读取属性数据"""
        pass
    
    @abstractmethod
    def write_structure(self, structure: CrystalStructure, filepath: str) -> None:
        """写入晶体结构"""
        pass


class CIFHandler(DataFormatHandler):
    """CIF格式处理器"""
    
    def detect_format(self, filepath: str) -> bool:
        return filepath.lower().endswith('.cif')
    
    def read_structure(self, filepath: str) -> CrystalStructure:
        """读取CIF文件"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # 解析CIF数据
        data = self._parse_cif(content)
        
        # 提取晶格参数
        lattice = Lattice(
            a=self._get_float(data, '_cell_length_a', 1.0),
            b=self._get_float(data, '_cell_length_b', 1.0),
            c=self._get_float(data, '_cell_length_c', 1.0),
            alpha=self._get_float(data, '_cell_angle_alpha', 90.0),
            beta=self._get_float(data, '_cell_angle_beta', 90.0),
            gamma=self._get_float(data, '_cell_angle_gamma', 90.0)
        )
        
        # 提取原子位点
        sites = self._parse_atom_sites(data)
        
        # 提取化学式
        formula = data.get('_chemical_formula_sum', 'Unknown')
        
        # 提取空间群
        space_group = data.get('_symmetry_space_group_name_H-M')
        sg_number = self._get_int(data, '_symmetry_Int_Tables_number', None)
        
        return CrystalStructure(
            formula=formula,
            lattice=lattice,
            sites=sites,
            space_group=space_group,
            space_group_number=sg_number,
            source='CIF',
            metadata=data
        )
    
    def _parse_cif(self, content: str) -> Dict[str, str]:
        """解析CIF内容"""
        data = {}
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 匹配 key value 格式
            match = re.match(r'(_\S+)\s+(.+)', line)
            if match:
                key, value = match.groups()
                data[key] = value.strip().strip('"\'')
        
        return data
    
    def _get_float(self, data: Dict, key: str, default: float) -> float:
        """安全获取浮点数"""
        try:
            val = data.get(key, str(default))
            # 处理括号内的不确定度
            val = re.sub(r'\([^)]+\)', '', val)
            return float(val)
        except (ValueError, TypeError):
            return default
    
    def _get_int(self, data: Dict, key: str, default: Optional[int]) -> Optional[int]:
        """安全获取整数"""
        try:
            return int(data.get(key, default))
        except (ValueError, TypeError):
            return default
    
    def _parse_atom_sites(self, data: Dict) -> List[AtomSite]:
        """解析原子位点（简化版）"""
        sites = []
        # 尝试从 _atom_site 数据解析
        # 实际实现需要处理更复杂的CIF循环结构
        
        # 基础实现：查找单一位点
        elements = data.get('_atom_site_type_symbol', '').split()
        xs = data.get('_atom_site_fract_x', '').split()
        ys = data.get('_atom_site_fract_y', '').split()
        zs = data.get('_atom_site_fract_z', '').split()
        
        for i, elem in enumerate(elements):
            if i < len(xs) and i < len(ys) and i < len(zs):
                try:
                    x = float(re.sub(r'\([^)]+\)', '', xs[i]))
                    y = float(re.sub(r'\([^)]+\)', '', ys[i]))
                    z = float(re.sub(r'\([^)]+\)', '', zs[i]))
                    
                    sites.append(AtomSite(
                        element=elem.strip(),
                        x=x, y=y, z=z,
                        occupancy=1.0
                    ))
                except (ValueError, IndexError):
                    continue
        
        return sites
    
    def read_properties(self, filepath: str) -> List[ExperimentalProperty]:
        """从CIF读取属性（如可用）"""
        properties = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        data = self._parse_cif(content)
        
        # 提取常见的实验属性
        prop_mapping = {
            '_cell_measurement_temperature': ('temperature', 'K'),
            '_exptl_crystal_density_meas': ('density', 'g/cm^3'),
            '_exptl_crystal_density_diffrn': ('density_calculated', 'g/cm^3'),
        }
        
        for key, (name, unit) in prop_mapping.items():
            if key in data:
                try:
                    value = float(re.sub(r'\([^)]+\)', '', data[key]))
                    properties.append(ExperimentalProperty(
                        name=name,
                        value=value,
                        unit=unit,
                        method='X-ray diffraction'
                    ))
                except ValueError:
                    continue
        
        return properties
    
    def write_structure(self, structure: CrystalStructure, filepath: str) -> None:
        """写入CIF文件"""
        lines = [
            "data_",
            "_symmetry_space_group_name_H-M 'P 1'",
            f"_cell_length_a {structure.lattice.a:.6f}",
            f"_cell_length_b {structure.lattice.b:.6f}",
            f"_cell_length_c {structure.lattice.c:.6f}",
            f"_cell_angle_alpha {structure.lattice.alpha:.4f}",
            f"_cell_angle_beta {structure.lattice.beta:.4f}",
            f"_cell_angle_gamma {structure.lattice.gamma:.4f}",
            f"_chemical_formula_sum '{structure.formula}'",
            "",
            "loop_",
            "_atom_site_label",
            "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
            "_atom_site_occupancy"
        ]
        
        for i, site in enumerate(structure.sites):
            label = f"{site.element}{i+1}"
            lines.append(f"{label} {site.element} {site.x:.6f} {site.y:.6f} {site.z:.6f} {site.occupancy:.4f}")
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))


class POSCARHandler(DataFormatHandler):
    """POSCAR/VASP格式处理器"""
    
    def detect_format(self, filepath: str) -> bool:
        name = os.path.basename(filepath).lower()
        return name in ['poscar', 'contcar'] or filepath.lower().endswith(('.vasp', '.poscar'))
    
    def read_structure(self, filepath: str) -> CrystalStructure:
        """读取POSCAR文件"""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # 解析POSCAR
        comment = lines[0].strip()
        scale = float(lines[1].strip())
        
        # 晶格向量
        lattice_vectors = []
        for i in range(2, 5):
            vec = [float(x) for x in lines[i].split()]
            lattice_vectors.append(np.array(vec) * scale)
        
        # 从晶格向量计算晶格参数
        lattice = self._vectors_to_lattice(lattice_vectors)
        
        # 元素和数量
        elements = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        
        # 坐标类型
        coord_type = lines[7].strip()[0].lower()
        
        # 读取坐标
        sites = []
        line_idx = 8
        elem_idx = 0
        count_idx = 0
        
        for count in counts:
            elem = elements[elem_idx] if elem_idx < len(elements) else 'X'
            for _ in range(count):
                if line_idx < len(lines):
                    coords = [float(x) for x in lines[line_idx].split()[:3]]
                    sites.append(AtomSite(
                        element=elem,
                        x=coords[0], y=coords[1], z=coords[2]
                    ))
                    line_idx += 1
            elem_idx += 1
        
        formula = ''.join([f"{e}{c}" for e, c in zip(elements, counts)])
        
        return CrystalStructure(
            formula=formula,
            lattice=lattice,
            sites=sites,
            source='POSCAR',
            metadata={'comment': comment, 'scale': scale}
        )
    
    def _vectors_to_lattice(self, vectors: List[np.ndarray]) -> Lattice:
        """从晶格向量计算晶格参数"""
        a_vec, b_vec, c_vec = vectors
        
        a = np.linalg.norm(a_vec)
        b = np.linalg.norm(b_vec)
        c = np.linalg.norm(c_vec)
        
        alpha = np.degrees(np.arccos(np.dot(b_vec, c_vec) / (b * c)))
        beta = np.degrees(np.arccos(np.dot(a_vec, c_vec) / (a * c)))
        gamma = np.degrees(np.arccos(np.dot(a_vec, b_vec) / (a * b)))
        
        return Lattice(a, b, c, alpha, beta, gamma)
    
    def read_properties(self, filepath: str) -> List[ExperimentalProperty]:
        """POSCAR不包含属性数据"""
        return []
    
    def write_structure(self, structure: CrystalStructure, filepath: str) -> None:
        """写入POSCAR文件"""
        lines = [
            f"{structure.formula}",
            "1.0"
        ]
        
        # 晶格向量
        matrix = structure.lattice.to_matrix()
        for i in range(3):
            lines.append(f"{matrix[i,0]:.10f} {matrix[i,1]:.10f} {matrix[i,2]:.10f}")
        
        # 元素和计数
        elements = structure.elements
        counts = [len(structure.get_site_by_element(e)) for e in elements]
        
        lines.append(' '.join(elements))
        lines.append(' '.join(map(str, counts)))
        lines.append("Direct")
        
        # 坐标
        for elem in elements:
            for site in structure.get_site_by_element(elem):
                lines.append(f"{site.x:.10f} {site.y:.10f} {site.z:.10f}")
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))


class JSONHandler(DataFormatHandler):
    """JSON格式处理器"""
    
    def detect_format(self, filepath: str) -> bool:
        return filepath.lower().endswith('.json')
    
    def read_structure(self, filepath: str) -> CrystalStructure:
        """从JSON读取结构"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # 解析晶格
        lattice_data = data.get('lattice', data.get('cell', {}))
        lattice = Lattice(
            a=lattice_data.get('a', 1.0),
            b=lattice_data.get('b', 1.0),
            c=lattice_data.get('c', 1.0),
            alpha=lattice_data.get('alpha', 90.0),
            beta=lattice_data.get('beta', 90.0),
            gamma=lattice_data.get('gamma', 90.0)
        )
        
        # 解析位点
        sites = []
        for site_data in data.get('sites', data.get('atoms', [])):
            sites.append(AtomSite(
                element=site_data['element'],
                x=site_data['x'],
                y=site_data['y'],
                z=site_data['z'],
                occupancy=site_data.get('occupancy', 1.0),
                label=site_data.get('label')
            ))
        
        return CrystalStructure(
            formula=data.get('formula', 'Unknown'),
            lattice=lattice,
            sites=sites,
            space_group=data.get('space_group'),
            source='JSON',
            source_id=data.get('id'),
            metadata=data.get('metadata', {})
        )
    
    def read_properties(self, filepath: str) -> List[ExperimentalProperty]:
        """从JSON读取属性"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        properties = []
        props_data = data.get('properties', data.get('data', []))
        
        for prop in props_data:
            properties.append(ExperimentalProperty(
                name=prop['name'],
                value=prop['value'],
                unit=prop.get('unit', ''),
                uncertainty=prop.get('uncertainty'),
                temperature=prop.get('temperature'),
                pressure=prop.get('pressure'),
                method=prop.get('method'),
                reference=prop.get('reference')
            ))
        
        return properties
    
    def write_structure(self, structure: CrystalStructure, filepath: str) -> None:
        """写入JSON"""
        data = {
            'formula': structure.formula,
            'lattice': {
                'a': structure.lattice.a,
                'b': structure.lattice.b,
                'c': structure.lattice.c,
                'alpha': structure.lattice.alpha,
                'beta': structure.lattice.beta,
                'gamma': structure.lattice.gamma
            },
            'sites': [
                {
                    'element': s.element,
                    'x': s.x, 'y': s.y, 'z': s.z,
                    'occupancy': s.occupancy
                }
                for s in structure.sites
            ],
            'space_group': structure.space_group,
            'metadata': structure.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class FormatRegistry:
    """格式处理器注册表"""
    
    def __init__(self):
        self._handlers: List[DataFormatHandler] = [
            CIFHandler(),
            POSCARHandler(),
            JSONHandler()
        ]
    
    def get_handler(self, filepath: str) -> Optional[DataFormatHandler]:
        """获取合适的处理器"""
        for handler in self._handlers:
            if handler.detect_format(filepath):
                return handler
        return None
    
    def register(self, handler: DataFormatHandler):
        """注册新处理器"""
        self._handlers.append(handler)
    
    def detect_format(self, filepath: str) -> DataFormat:
        """检测文件格式"""
        ext = os.path.splitext(filepath)[1].lower()
        format_map = {
            '.cif': DataFormat.CIF,
            '.poscar': DataFormat.POSCAR,
            '.vasp': DataFormat.POSCAR,
            '.json': DataFormat.JSON,
            '.yaml': DataFormat.YAML,
            '.yml': DataFormat.YAML,
            '.csv': DataFormat.CSV,
            '.xlsx': DataFormat.EXCEL,
            '.h5': DataFormat.HDF5,
            '.hdf5': DataFormat.HDF5
        }
        return format_map.get(ext, DataFormat.UNKNOWN)


# 全局注册表
_format_registry = FormatRegistry()


def read_structure(filepath: str) -> CrystalStructure:
    """读取晶体结构（自动检测格式）"""
    handler = _format_registry.get_handler(filepath)
    if handler is None:
        raise ValueError(f"Unsupported file format: {filepath}")
    return handler.read_structure(filepath)


def read_properties(filepath: str) -> List[ExperimentalProperty]:
    """读取实验属性（自动检测格式）"""
    handler = _format_registry.get_handler(filepath)
    if handler is None:
        raise ValueError(f"Unsupported file format: {filepath}")
    return handler.read_properties(filepath)


def write_structure(structure: CrystalStructure, filepath: str, format: Optional[str] = None) -> None:
    """写入晶体结构"""
    if format:
        # 根据格式选择处理器
        format_map = {
            'cif': CIFHandler(),
            'poscar': POSCARHandler(),
            'json': JSONHandler()
        }
        handler = format_map.get(format.lower())
        if handler is None:
            raise ValueError(f"Unknown format: {format}")
    else:
        handler = _format_registry.get_handler(filepath)
        if handler is None:
            raise ValueError(f"Cannot determine format for: {filepath}")
    
    handler.write_structure(structure, filepath)


def detect_format(filepath: str) -> DataFormat:
    """检测文件格式"""
    return _format_registry.detect_format(filepath)


# =============================================================================
# 演示代码
# =============================================================================

def demo():
    """演示数据格式处理功能"""
    print("=" * 80)
    print("📊 实验数据格式处理演示")
    print("=" * 80)
    
    # 创建示例晶体结构
    lattice = Lattice(4.2, 4.2, 4.2, 90, 90, 90)
    
    sites = [
        AtomSite('Na', 0.0, 0.0, 0.0),
        AtomSite('Na', 0.5, 0.5, 0.5),
        AtomSite('Cl', 0.5, 0.0, 0.0),
        AtomSite('Cl', 0.0, 0.5, 0.5),
        AtomSite('Cl', 0.0, 0.0, 0.5),
        AtomSite('Cl', 0.5, 0.5, 0.0),
    ]
    
    structure = CrystalStructure(
        formula='Na2Cl4',
        lattice=lattice,
        sites=sites,
        space_group='Fm-3m',
        space_group_number=225
    )
    
    print(f"\n🔹 示例晶体结构:")
    print(f"   化学式: {structure.formula}")
    print(f"   晶胞体积: {structure.lattice.volume:.4f} Å³")
    print(f"   原子数: {structure.num_atoms}")
    print(f"   元素: {structure.elements}")
    print(f"   空间群: {structure.space_group} (#{structure.space_group_number})")
    
    # 测试导出
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n🔹 导出结构到不同格式:")
        
        # CIF格式
        cif_path = os.path.join(tmpdir, 'test.cif')
        write_structure(structure, cif_path, 'cif')
        print(f"   ✓ CIF: {cif_path}")
        
        # 读取验证
        cif_structure = read_structure(cif_path)
        print(f"     读取验证: {cif_structure.formula}")
        
        # POSCAR格式
        poscar_path = os.path.join(tmpdir, 'POSCAR')
        write_structure(structure, poscar_path, 'poscar')
        print(f"   ✓ POSCAR: {poscar_path}")
        
        # 读取验证
        poscar_structure = read_structure(poscar_path)
        print(f"     读取验证: {poscar_structure.formula}")
        
        # JSON格式
        json_path = os.path.join(tmpdir, 'test.json')
        write_structure(structure, json_path, 'json')
        print(f"   ✓ JSON: {json_path}")
        
        # 读取验证
        json_structure = read_structure(json_path)
        print(f"     读取验证: {json_structure.formula}")
    
    # 实验属性示例
    print("\n🔹 实验属性示例:")
    props = [
        ExperimentalProperty('band_gap', 3.2, 'eV', uncertainty=0.1, temperature=300, method='UV-Vis'),
        ExperimentalProperty('lattice_constant', 4.2, 'Å', uncertainty=0.01, temperature=298, method='XRD'),
        ExperimentalProperty('bulk_modulus', 24.5, 'GPa', uncertainty=1.2, temperature=300, method='Ultrasonic')
    ]
    
    for prop in props:
        print(f"   • {prop}")
        if prop.relative_uncertainty:
            print(f"     相对不确定度: {prop.relative_uncertainty*100:.2f}%")
    
    print("\n" + "=" * 80)
    print("✅ 数据格式处理演示完成!")
    print("=" * 80)


if __name__ == '__main__':
    demo()
