"""
DFT-LAMMPS 跨模块桥接系统
=========================
自动适配不同接口

解决模块间接口不兼容问题，实现无缝数据流动。

Author: DFT-LAMMPS Team
Phase: 56 - Orchestration System
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Set, Type, TypeVar,
    Union, get_type_hints, get_origin, get_args
)
from enum import Enum, auto


logger = logging.getLogger("cross_module_bridge")


class CompatibilityLevel(Enum):
    """兼容性级别"""
    NATIVE = "native"          # 原生兼容
    ADAPTER = "adapter"        # 需要适配器
    CONVERTER = "converter"    # 需要转换器
    INCOMPATIBLE = "incompatible"  # 不兼容


@dataclass
class InterfaceSpec:
    """接口规范"""
    name: str                           # 接口名称
    version: str = "1.0.0"              # 版本
    
    # 输入输出定义
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据
    description: str = ""
    required: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)
    
    def is_compatible_with(self, other: InterfaceSpec) -> CompatibilityLevel:
        """检查与另一接口的兼容性"""
        # 检查输出是否能匹配输入
        for key, spec in other.input_schema.items():
            if key not in self.output_schema:
                if key in other.required:
                    return CompatibilityLevel.INCOMPATIBLE
                continue
            
            if not self._schema_compatible(self.output_schema[key], spec):
                return CompatibilityLevel.CONVERTER
        
        return CompatibilityLevel.NATIVE
    
    def _schema_compatible(self, schema1: Any, schema2: Any) -> bool:
        """检查schema兼容性"""
        if isinstance(schema1, dict) and isinstance(schema2, dict):
            for k, v in schema2.items():
                if k not in schema1:
                    return False
                if not self._schema_compatible(schema1[k], v):
                    return False
            return True
        return schema1 == schema2


@dataclass
class DataMapping:
    """数据映射定义"""
    source_path: str                    # 源路径（如 "output.energy"）
    target_path: str                    # 目标路径（如 "input.energy"）
    transform: Optional[Callable] = None  # 转换函数
    condition: Optional[Callable] = None  # 条件函数
    default: Any = None                 # 默认值


class DataConverter(ABC):
    """数据转换器基类"""
    
    @property
    @abstractmethod
    def source_type(self) -> str:
        """源数据类型"""
        pass
    
    @property
    @abstractmethod
    def target_type(self) -> str:
        """目标数据类型"""
        pass
    
    @abstractmethod
    def convert(self, data: Any, **kwargs) -> Any:
        """执行转换"""
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """验证数据"""
        pass


class StructureConverter(DataConverter):
    """结构数据转换器"""
    
    SUPPORTED_FORMATS = ["pymatgen", "ase", "cif", "poscar", "xyz", "lammps_data"]
    
    def __init__(self, source_format: str, target_format: str):
        self.source_format = source_format
        self.target_format = target_format
    
    @property
    def source_type(self) -> str:
        return f"structure_{self.source_format}"
    
    @property
    def target_type(self) -> str:
        return f"structure_{self.target_format}"
    
    def convert(self, data: Any, **kwargs) -> Any:
        """转换结构数据"""
        try:
            # 使用pymatgen作为中间格式
            structure = self._to_pymatgen(data, self.source_format)
            return self._from_pymatgen(structure, self.target_format)
        except Exception as e:
            logger.error(f"Structure conversion failed: {e}")
            raise
    
    def validate(self, data: Any) -> bool:
        """验证结构数据"""
        try:
            self._to_pymatgen(data, self.source_format)
            return True
        except:
            return False
    
    def _to_pymatgen(self, data: Any, format: str) -> Any:
        """转换为pymatgen Structure"""
        from pymatgen.core import Structure
        
        if format == "pymatgen":
            return data
        elif format == "cif":
            return Structure.from_str(data, fmt="cif")
        elif format == "poscar":
            return Structure.from_str(data, fmt="poscar")
        elif format == "ase":
            return Structure.from_ase_atoms(data)
        else:
            raise ValueError(f"Unsupported source format: {format}")
    
    def _from_pymatgen(self, structure: Any, format: str) -> Any:
        """从pymatgen Structure转换"""
        if format == "pymatgen":
            return structure
        elif format == "cif":
            return structure.to(fmt="cif")
        elif format == "poscar":
            return structure.to(fmt="poscar")
        elif format == "ase":
            return structure.to_ase_atoms()
        elif format == "lammps_data":
            return self._to_lammps_data(structure)
        else:
            raise ValueError(f"Unsupported target format: {format}")
    
    def _to_lammps_data(self, structure: Any) -> str:
        """转换为LAMMPS data格式"""
        lines = [f"Generated from structure with {len(structure)} atoms"]
        lines.append(f"{len(structure)} atoms")
        lines.append(f"{len(structure.species)} atom types")
        
        # 盒子边界
        a, b, c = structure.lattice.abc
        lines.append(f"0.0 {a} xlo xhi")
        lines.append(f"0.0 {b} ylo yhi")
        lines.append(f"0.0 {c} zlo zhi")
        
        # 原子
        lines.append("\nAtoms")
        for i, site in enumerate(structure, 1):
            x, y, z = site.coords
            elem_id = list(structure.species).index(site.specie) + 1
            lines.append(f"{i} {elem_id} {x} {y} {z}")
        
        return "\n".join(lines)


class EnergyConverter(DataConverter):
    """能量数据转换器"""
    
    UNIT_CONVERSIONS = {
        ("eV", "eV"): 1.0,
        ("eV", "kJ/mol"): 96.485,
        ("eV", "kcal/mol"): 23.061,
        ("Ha", "eV"): 27.2114,
        ("Ry", "eV"): 13.6057,
    }
    
    def __init__(self, source_unit: str = "eV", target_unit: str = "eV"):
        self.source_unit = source_unit
        self.target_unit = target_unit
    
    @property
    def source_type(self) -> str:
        return f"energy_{self.source_unit}"
    
    @property
    def target_type(self) -> str:
        return f"energy_{self.target_unit}"
    
    def convert(self, data: Any, **kwargs) -> Any:
        """转换能量单位"""
        if isinstance(data, dict):
            return {k: self._convert_value(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._convert_value(v) for v in data]
        else:
            return self._convert_value(data)
    
    def _convert_value(self, value: float) -> float:
        """转换单个值"""
        key = (self.source_unit, self.target_unit)
        if key in self.UNIT_CONVERSIONS:
            return value * self.UNIT_CONVERSIONS[key]
        
        # 尝试反向查找
        reverse_key = (self.target_unit, self.source_unit)
        if reverse_key in self.UNIT_CONVERSIONS:
            return value / self.UNIT_CONVERSIONS[reverse_key]
        
        raise ValueError(f"Cannot convert from {self.source_unit} to {self.target_unit}")
    
    def validate(self, data: Any) -> bool:
        """验证能量数据"""
        try:
            if isinstance(data, dict):
                all(isinstance(v, (int, float)) for v in data.values())
            elif isinstance(data, (list, tuple)):
                all(isinstance(v, (int, float)) for v in data)
            else:
                isinstance(data, (int, float))
            return True
        except:
            return False


class ForceConverter(DataConverter):
    """力数据转换器"""
    
    UNIT_CONVERSIONS = {
        ("eV/Ang", "eV/Ang"): 1.0,
        ("eV/Ang", "Ha/Bohr"): 0.0194469,
        ("Ry/Bohr", "eV/Ang"): 25.711,
    }
    
    def __init__(self, source_unit: str = "eV/Ang", target_unit: str = "eV/Ang"):
        self.source_unit = source_unit
        self.target_unit = target_unit
    
    @property
    def source_type(self) -> str:
        return f"force_{self.source_unit}"
    
    @property
    def target_type(self) -> str:
        return f"force_{self.target_unit}"
    
    def convert(self, data: Any, **kwargs) -> Any:
        """转换力数据"""
        import numpy as np
        
        key = (self.source_unit, self.target_unit)
        factor = self.UNIT_CONVERSIONS.get(key, 1.0)
        
        if isinstance(data, np.ndarray):
            return data * factor
        elif isinstance(data, (list, tuple)):
            return [[f * factor for f in atom] for atom in data]
        else:
            return data * factor
    
    def validate(self, data: Any) -> bool:
        """验证力数据"""
        import numpy as np
        return isinstance(data, (np.ndarray, list))


class TrajectoryConverter(DataConverter):
    """轨迹数据转换器"""
    
    SUPPORTED_FORMATS = ["ase", "pymatgen", "vasp_xdatcar", "lammps_dump", "extxyz"]
    
    def __init__(self, source_format: str, target_format: str):
        self.source_format = source_format
        self.target_format = target_format
    
    @property
    def source_type(self) -> str:
        return f"trajectory_{self.source_format}"
    
    @property
    def target_type(self) -> str:
        return f"trajectory_{self.target_format}"
    
    def convert(self, data: Any, **kwargs) -> Any:
        """转换轨迹数据"""
        # 使用ASE作为中间格式
        from ase.io import read, write
        import io
        
        # 读取
        if self.source_format == "ase":
            atoms_list = data
        else:
            buffer = io.StringIO(data)
            atoms_list = []
            while True:
                try:
                    atoms = read(buffer, format=self.source_format)
                    if atoms is None:
                        break
                    atoms_list.append(atoms)
                except:
                    break
        
        # 写出
        if self.target_format == "ase":
            return atoms_list
        else:
            buffer = io.StringIO()
            for atoms in atoms_list:
                write(buffer, atoms, format=self.target_format)
            return buffer.getvalue()
    
    def validate(self, data: Any) -> bool:
        """验证轨迹数据"""
        try:
            self.convert(data)
            return True
        except:
            return False


class CrossModuleBridge:
    """
    跨模块桥接器
    
    管理模块间的数据转换和接口适配
    
    Example:
        bridge = CrossModuleBridge()
        
        # 注册转换器
        bridge.register_converter(StructureConverter("cif", "poscar"))
        
        # 转换数据
        poscar_data = bridge.convert(structure_data, "structure_cif", "structure_poscar")
    """
    
    def __init__(self):
        self._converters: Dict[Tuple[str, str], DataConverter] = {}
        self._adapters: Dict[Tuple[str, str], Callable] = {}
        self._interface_specs: Dict[str, InterfaceSpec] = {}
        self._default_mappings: Dict[str, List[DataMapping]] = {}
        
        # 注册内置转换器
        self._register_builtin_converters()
    
    def register_converter(self, converter: DataConverter) -> None:
        """注册数据转换器"""
        key = (converter.source_type, converter.target_type)
        self._converters[key] = converter
        logger.info(f"Registered converter: {converter.source_type} -> {converter.target_type}")
    
    def register_adapter(
        self, 
        source_interface: str, 
        target_interface: str, 
        adapter_func: Callable
    ) -> None:
        """注册接口适配器"""
        key = (source_interface, target_interface)
        self._adapters[key] = adapter_func
        logger.info(f"Registered adapter: {source_interface} -> {target_interface}")
    
    def register_interface(self, name: str, spec: InterfaceSpec) -> None:
        """注册接口规范"""
        self._interface_specs[name] = spec
    
    def convert(
        self, 
        data: Any, 
        source_type: str, 
        target_type: str,
        **kwargs
    ) -> Any:
        """
        转换数据
        
        Args:
            data: 源数据
            source_type: 源数据类型
            target_type: 目标数据类型
            **kwargs: 额外参数
        """
        # 直接查找转换器
        key = (source_type, target_type)
        if key in self._converters:
            converter = self._converters[key]
            if converter.validate(data):
                return converter.convert(data, **kwargs)
            else:
                raise ValueError(f"Data validation failed for {source_type}")
        
        # 尝试通过中间格式转换
        converted = self._try_chain_conversion(data, source_type, target_type, **kwargs)
        if converted is not None:
            return converted
        
        raise ValueError(f"No converter found: {source_type} -> {target_type}")
    
    def adapt(
        self, 
        source_output: Dict[str, Any], 
        source_interface: str, 
        target_interface: str
    ) -> Dict[str, Any]:
        """
        适配接口输出
        
        将一个模块的输出适配为另一个模块的输入
        """
        # 查找适配器
        key = (source_interface, target_interface)
        if key in self._adapters:
            return self._adapters[key](source_output)
        
        # 尝试自动映射
        return self._auto_adapt(source_output, source_interface, target_interface)
    
    def check_compatibility(
        self, 
        source_interface: str, 
        target_interface: str
    ) -> CompatibilityLevel:
        """检查接口兼容性"""
        source_spec = self._interface_specs.get(source_interface)
        target_spec = self._interface_specs.get(target_interface)
        
        if not source_spec or not target_spec:
            return CompatibilityLevel.INCOMPATIBLE
        
        return source_spec.is_compatible_with(target_spec)
    
    def create_bridge(
        self, 
        source_module: str, 
        target_module: str
    ) -> Optional[Callable]:
        """
        创建模块间桥接函数
        
        返回一个函数，可以直接用于转换源模块输出到目标模块输入
        """
        # 获取模块的输入输出规范
        source_output_type = f"{source_module}_output"
        target_input_type = f"{target_module}_input"
        
        # 查找转换链
        converter_chain = self._find_converter_chain(source_output_type, target_input_type)
        
        if not converter_chain:
            return None
        
        def bridge_func(data: Any) -> Any:
            result = data
            for source, target in converter_chain:
                result = self.convert(result, source, target)
            return result
        
        return bridge_func
    
    def auto_bridge_modules(
        self, 
        modules: List[str]
    ) -> Dict[Tuple[str, str], Callable]:
        """自动为多组模块创建桥接"""
        bridges = {}
        
        for i, source in enumerate(modules):
            for target in modules[i+1:]:
                # 尝试创建双向桥接
                forward = self.create_bridge(source, target)
                if forward:
                    bridges[(source, target)] = forward
                
                backward = self.create_bridge(target, source)
                if backward:
                    bridges[(target, source)] = backward
        
        return bridges
    
    def _register_builtin_converters(self) -> None:
        """注册内置转换器"""
        # 结构转换器
        for source in StructureConverter.SUPPORTED_FORMATS:
            for target in StructureConverter.SUPPORTED_FORMATS:
                if source != target:
                    self.register_converter(StructureConverter(source, target))
        
        # 能量转换器（常用单位）
        energy_units = ["eV", "kJ/mol", "kcal/mol", "Ha", "Ry"]
        for source in energy_units:
            for target in energy_units:
                if source != target:
                    self.register_converter(EnergyConverter(source, target))
        
        # 力转换器
        force_units = ["eV/Ang", "Ha/Bohr", "Ry/Bohr"]
        for source in force_units:
            for target in force_units:
                if source != target:
                    self.register_converter(ForceConverter(source, target))
    
    def _try_chain_conversion(
        self, 
        data: Any, 
        source_type: str, 
        target_type: str,
        max_depth: int = 3,
        **kwargs
    ) -> Optional[Any]:
        """尝试链式转换"""
        # BFS查找转换链
        from collections import deque
        
        queue = deque([(source_type, [source_type], data)])
        visited = {source_type}
        
        while queue and max_depth > 0:
            current_type, chain, current_data = queue.popleft()
            
            if current_type == target_type:
                # 执行转换链
                result = data
                for i in range(len(chain) - 1):
                    key = (chain[i], chain[i + 1])
                    if key in self._converters:
                        result = self._converters[key].convert(result, **kwargs)
                return result
            
            # 扩展搜索
            for (s, t), converter in self._converters.items():
                if s == current_type and t not in visited:
                    visited.add(t)
                    queue.append((t, chain + [t], None))
            
            max_depth -= 1
        
        return None
    
    def _auto_adapt(
        self, 
        source_output: Dict[str, Any], 
        source_interface: str, 
        target_interface: str
    ) -> Dict[str, Any]:
        """自动适配接口"""
        # 简单的名称匹配
        result = {}
        
        for key, value in source_output.items():
            # 直接匹配
            result[key] = value
            
            # 常见别名
            aliases = {
                "energy": ["total_energy", "e_0", "final_energy"],
                "forces": ["force", "atom_forces"],
                "structure": ["final_structure", "optimized_structure", "atoms"],
                "stress": ["virial_stress", "pressure_tensor"]
            }
            
            for canonical, alias_list in aliases.items():
                if key in alias_list and canonical not in result:
                    result[canonical] = value
        
        return result
    
    def _find_converter_chain(
        self, 
        source_type: str, 
        target_type: str
    ) -> List[Tuple[str, str]]:
        """查找转换链"""
        # BFS
        from collections import deque
        
        queue = deque([(source_type, [])])
        visited = {source_type}
        
        while queue:
            current, chain = queue.popleft()
            
            if current == target_type:
                return chain
            
            for (s, t), _ in self._converters.items():
                if s == current and t not in visited:
                    visited.add(t)
                    queue.append((t, chain + [(s, t)]))
        
        return []


# 适配器装饰器
class Adapter:
    """适配器装饰器"""
    
    def __init__(
        self, 
        source_interface: str, 
        target_interface: str,
        bridge: Optional[CrossModuleBridge] = None
    ):
        self.source_interface = source_interface
        self.target_interface = target_interface
        self.bridge = bridge or CrossModuleBridge()
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 转换输入
            if args and isinstance(args[0], dict):
                adapted_input = self.bridge.adapt(
                    args[0], 
                    self.source_interface, 
                    self.target_interface
                )
                args = (adapted_input,) + args[1:]
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 转换输出
            if isinstance(result, dict):
                return self.bridge.adapt(
                    result,
                    self.target_interface,
                    self.source_interface
                )
            
            return result
        
        return wrapper


# 便捷函数
def get_global_bridge() -> CrossModuleBridge:
    """获取全局桥接器实例"""
    if not hasattr(get_global_bridge, '_instance'):
        get_global_bridge._instance = CrossModuleBridge()
    return get_global_bridge._instance


def convert_data(
    data: Any, 
    source_type: str, 
    target_type: str,
    **kwargs
) -> Any:
    """便捷函数：转换数据"""
    return get_global_bridge().convert(data, source_type, target_type, **kwargs)


def adapt_output(
    data: Dict[str, Any],
    source_module: str,
    target_module: str
) -> Dict[str, Any]:
    """便捷函数：适配模块输出"""
    return get_global_bridge().adapt(
        data,
        f"{source_module}_output",
        f"{target_module}_input"
    )


def register_structure_converter(source_fmt: str, target_fmt: str) -> None:
    """便捷函数：注册结构转换器"""
    get_global_bridge().register_converter(
        StructureConverter(source_fmt, target_fmt)
    )


# 类型转换辅助函数
def ensure_pymatgen_structure(data: Any) -> Any:
    """确保数据为pymatgen Structure"""
    from pymatgen.core import Structure
    
    if isinstance(data, Structure):
        return data
    
    bridge = get_global_bridge()
    
    # 尝试各种格式转换
    for fmt in ["cif", "poscar", "ase"]:
        try:
            return bridge.convert(data, f"structure_{fmt}", "structure_pymatgen")
        except:
            continue
    
    raise ValueError("Cannot convert to pymatgen Structure")


def ensure_ase_atoms(data: Any) -> Any:
    """确保数据为ASE Atoms"""
    from ase import Atoms
    
    if isinstance(data, Atoms):
        return data
    
    bridge = get_global_bridge()
    
    for fmt in ["cif", "poscar", "pymatgen"]:
        try:
            return bridge.convert(data, f"structure_{fmt}", "structure_ase")
        except:
            continue
    
    raise ValueError("Cannot convert to ASE Atoms")