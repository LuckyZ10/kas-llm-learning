"""
DFT-LAMMPS 统一数据模型
=======================
结构、性质、计算结果标准化

提供跨模块的数据标准化表示

Author: DFT-LAMMPS Team
Phase: 56 - Orchestration System
"""

from __future__ import annotations

import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union


# 类型变量
T = TypeVar('T')


class DataCategory(Enum):
    """数据类别"""
    STRUCTURE = "structure"          # 结构数据
    CALCULATION = "calculation"      # 计算结果
    PROPERTY = "property"            # 材料性质
    SIMULATION = "simulation"        # 模拟数据
    ANALYSIS = "analysis"            # 分析结果
    METADATA = "metadata"            # 元数据


class DataQuality(Enum):
    """数据质量等级"""
    RAW = "raw"                      # 原始数据
    VALIDATED = "validated"          # 已验证
    VERIFIED = "verified"            # 已核实
    PUBLISHED = "published"          # 已发表


class PropertyType(Enum):
    """性质类型"""
    # 电子性质
    BAND_GAP = "band_gap"
    ELECTRONIC_DOS = "electronic_dos"
    FERMI_LEVEL = "fermi_level"
    WORK_FUNCTION = "work_function"
    
    # 力学性质
    BULK_MODULUS = "bulk_modulus"
    SHEAR_MODULUS = "shear_modulus"
    YOUNG_MODULUS = "young_modulus"
    POISSON_RATIO = "poisson_ratio"
    ELASTIC_TENSOR = "elastic_tensor"
    
    # 热力学性质
    FORMATION_ENERGY = "formation_energy"
    COHESIVE_ENERGY = "cohesive_energy"
    SURFACE_ENERGY = "surface_energy"
    DEFECT_ENERGY = "defect_energy"
    
    # 输运性质
    IONIC_CONDUCTIVITY = "ionic_conductivity"
    ELECTRONIC_CONDUCTIVITY = "electronic_conductivity"
    DIFFUSION_COEFFICIENT = "diffusion_coefficient"
    
    # 光学性质
    DIELECTRIC_CONSTANT = "dielectric_constant"
    REFRACTIVE_INDEX = "refractive_index"
    ABSORPTION_SPECTRUM = "absorption_spectrum"
    
    # 磁学性质
    MAGNETIC_MOMENT = "magnetic_moment"
    CURIE_TEMPERATURE = "curie_temperature"
    
    # 化学性质
    ADSORPTION_ENERGY = "adsorption_energy"
    REACTION_BARRIER = "reaction_barrier"


@dataclass
class DataSource:
    """数据来源"""
    module: str                         # 产生数据的模块
    version: str                        # 模块版本
    calculation_id: Optional[str] = None  # 计算ID
    timestamp: datetime = field(default_factory=datetime.now)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "version": self.version,
            "calculation_id": self.calculation_id,
            "timestamp": self.timestamp.isoformat(),
            "parameters": self.parameters
        }


@dataclass
class DataProvenance:
    """数据血缘"""
    data_id: str                        # 数据ID
    parent_ids: List[str] = field(default_factory=list)  # 父数据ID
    derived_from: Optional[str] = None  # 派生自
    derivation_chain: List[str] = field(default_factory=list)  # 派生链
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_id": self.data_id,
            "parent_ids": self.parent_ids,
            "derived_from": self.derived_from,
            "derivation_chain": self.derivation_chain
        }


class UnifiedDataModel(ABC):
    """
    统一数据模型基类
    
    所有标准化数据类型的基类
    """
    
    def __init__(
        self,
        category: DataCategory,
        data_id: Optional[str] = None
    ):
        self._id = data_id or str(uuid.uuid4())
        self._category = category
        self._created_at = datetime.now()
        self._modified_at = datetime.now()
        self._quality = DataQuality.RAW
        self._source: Optional[DataSource] = None
        self._provenance: Optional[DataProvenance] = None
        self._metadata: Dict[str, Any] = {}
        self._tags: Set[str] = set()
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def category(self) -> DataCategory:
        return self._category
    
    @property
    def quality(self) -> DataQuality:
        return self._quality
    
    @quality.setter
    def quality(self, value: DataQuality) -> None:
        self._quality = value
        self._modified_at = datetime.now()
    
    @property
    def source(self) -> Optional[DataSource]:
        return self._source
    
    @source.setter
    def source(self, value: DataSource) -> None:
        self._source = value
        self._modified_at = datetime.now()
    
    @property
    def provenance(self) -> Optional[DataProvenance]:
        return self._provenance
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata
    
    def set_metadata(self, key: str, value: Any) -> None:
        """设置元数据"""
        self._metadata[key] = value
        self._modified_at = datetime.now()
    
    def add_tag(self, tag: str) -> None:
        """添加标签"""
        self._tags.add(tag)
    
    def remove_tag(self, tag: str) -> None:
        """移除标签"""
        self._tags.discard(tag)
    
    def has_tag(self, tag: str) -> bool:
        """检查是否有标签"""
        return tag in self._tags
    
    @abstractmethod
    def validate(self) -> Tuple[bool, List[str]]:
        """验证数据"""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> UnifiedDataModel:
        """从字典创建"""
        pass
    
    def compute_hash(self) -> str:
        """计算数据哈希"""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def derive(self, derivation_type: str) -> UnifiedDataModel:
        """创建派生数据"""
        new_data = self._create_copy()
        new_data._id = str(uuid.uuid4())
        new_data._provenance = DataProvenance(
            data_id=new_data._id,
            parent_ids=[self._id],
            derived_from=self._id,
            derivation_chain=(self._provenance.derivation_chain if self._provenance else []) + [derivation_type]
        )
        return new_data
    
    @abstractmethod
    def _create_copy(self) -> UnifiedDataModel:
        """创建副本（内部使用）"""
        pass


@dataclass
class LatticeData:
    """晶格数据"""
    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LatticeData:
        return cls(**data)


@dataclass
class SiteData:
    """原子位置数据"""
    species: str                        # 元素种类
    position: List[float]               # 位置（笛卡尔或分数坐标）
    coords_are_cartesian: bool = False  # 是否为笛卡尔坐标
    properties: Dict[str, Any] = field(default_factory=dict)  # 额外属性（如磁矩）
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "species": self.species,
            "position": self.position,
            "coords_are_cartesian": self.coords_are_cartesian,
            "properties": self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SiteData:
        return cls(
            species=data["species"],
            position=data["position"],
            coords_are_cartesian=data.get("coords_are_cartesian", False),
            properties=data.get("properties", {})
        )


class StructureData(UnifiedDataModel):
    """
    标准化结构数据
    
    统一的晶体结构表示，兼容pymatgen和ASE
    """
    
    def __init__(
        self,
        lattice: LatticeData,
        sites: List[SiteData],
        structure_id: Optional[str] = None
    ):
        super().__init__(DataCategory.STRUCTURE, structure_id)
        self._lattice = lattice
        self._sites = sites
        self._charge: Optional[float] = None
        self._spin_multiplicity: Optional[int] = None
    
    @property
    def lattice(self) -> LatticeData:
        return self._lattice
    
    @property
    def sites(self) -> List[SiteData]:
        return self._sites
    
    @property
    def num_sites(self) -> int:
        return len(self._sites)
    
    @property
    def formula(self) -> str:
        """计算化学式"""
        from collections import Counter
        elements = [site.species for site in self._sites]
        counts = Counter(elements)
        return "".join(f"{elem}{cnt if cnt > 1 else ''}" for elem, cnt in sorted(counts.items()))
    
    @property
    def charge(self) -> Optional[float]:
        return self._charge
    
    @charge.setter
    def charge(self, value: float) -> None:
        self._charge = value
        self._modified_at = datetime.now()
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证结构数据"""
        errors = []
        
        # 检查晶格参数
        if self._lattice.a <= 0 or self._lattice.b <= 0 or self._lattice.c <= 0:
            errors.append("Lattice parameters must be positive")
        
        # 检查原子位置
        if not self._sites:
            errors.append("Structure must have at least one site")
        
        for i, site in enumerate(self._sites):
            if len(site.position) != 3:
                errors.append(f"Site {i}: position must have 3 coordinates")
            
            # 检查分数坐标范围
            if not site.coords_are_cartesian:
                for j, coord in enumerate(site.position):
                    if not 0 <= coord < 1:
                        errors.append(f"Site {i}: fractional coordinate {j} out of range [0, 1)")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "category": self._category.value,
            "lattice": self._lattice.to_dict(),
            "sites": [site.to_dict() for site in self._sites],
            "formula": self.formula,
            "num_sites": self.num_sites,
            "charge": self._charge,
            "spin_multiplicity": self._spin_multiplicity,
            "quality": self._quality.value,
            "metadata": self._metadata,
            "tags": list(self._tags)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StructureData:
        lattice = LatticeData.from_dict(data["lattice"])
        sites = [SiteData.from_dict(s) for s in data["sites"]]
        
        structure = cls(lattice, sites, data.get("id"))
        if "quality" in data:
            structure._quality = DataQuality(data["quality"])
        if "metadata" in data:
            structure._metadata = data["metadata"]
        if "tags" in data:
            structure._tags = set(data["tags"])
        
        return structure
    
    def to_pymatgen(self):
        """转换为pymatgen Structure"""
        from pymatgen.core import Structure, Lattice
        
        lattice = Lattice.from_parameters(
            self._lattice.a, self._lattice.b, self._lattice.c,
            self._lattice.alpha, self._lattice.beta, self._lattice.gamma
        )
        
        species = [site.species for site in self._sites]
        coords = [site.position for site in self._sites]
        coords_are_cartesian = self._sites[0].coords_are_cartesian if self._sites else False
        
        return Structure(lattice, species, coords, coords_are_cartesian=coords_are_cartesian)
    
    @classmethod
    def from_pymatgen(cls, structure, structure_id: Optional[str] = None) -> StructureData:
        """从pymatgen Structure创建"""
        lattice = LatticeData(
            a=structure.lattice.a,
            b=structure.lattice.b,
            c=structure.lattice.c,
            alpha=structure.lattice.alpha,
            beta=structure.lattice.beta,
            gamma=structure.lattice.gamma
        )
        
        sites = []
        for site in structure:
            sites.append(SiteData(
                species=str(site.specie),
                position=site.coords.tolist(),
                coords_are_cartesian=True
            ))
        
        return cls(lattice, sites, structure_id)
    
    def to_ase(self):
        """转换为ASE Atoms"""
        from ase import Atoms
        
        symbols = [site.species for site in self._sites]
        positions = [site.position for site in self._sites]
        
        cell = [
            [self._lattice.a, 0, 0],
            [0, self._lattice.b, 0],
            [0, 0, self._lattice.c]
        ]
        
        return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    
    @classmethod
    def from_ase(cls, atoms, structure_id: Optional[str] = None) -> StructureData:
        """从ASE Atoms创建"""
        cell = atoms.get_cell()
        lattice = LatticeData(
            a=cell[0, 0],
            b=cell[1, 1],
            c=cell[2, 2]
        )
        
        sites = []
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        
        for symbol, position in zip(symbols, positions):
            sites.append(SiteData(
                species=symbol,
                position=position.tolist(),
                coords_are_cartesian=True
            ))
        
        return cls(lattice, sites, structure_id)
    
    def _create_copy(self) -> StructureData:
        """创建副本"""
        return StructureData(
            LatticeData(**self._lattice.to_dict()),
            [SiteData.from_dict(s.to_dict()) for s in self._sites],
            None
        )


class PropertyData(UnifiedDataModel):
    """
    标准化性质数据
    
    材料性质的统一表示
    """
    
    def __init__(
        self,
        property_type: PropertyType,
        value: Any,
        unit: str,
        structure_id: Optional[str] = None,
        data_id: Optional[str] = None
    ):
        super().__init__(DataCategory.PROPERTY, data_id)
        self._property_type = property_type
        self._value = value
        self._unit = unit
        self._structure_id = structure_id
        self._uncertainty: Optional[float] = None
        self._conditions: Dict[str, Any] = {}  # 测量条件（温度、压力等）
    
    @property
    def property_type(self) -> PropertyType:
        return self._property_type
    
    @property
    def value(self) -> Any:
        return self._value
    
    @value.setter
    def value(self, v: Any) -> None:
        self._value = v
        self._modified_at = datetime.now()
    
    @property
    def unit(self) -> str:
        return self._unit
    
    @property
    def structure_id(self) -> Optional[str]:
        return self._structure_id
    
    @property
    def uncertainty(self) -> Optional[float]:
        return self._uncertainty
    
    @uncertainty.setter
    def uncertainty(self, value: float) -> None:
        self._uncertainty = value
        self._modified_at = datetime.now()
    
    def set_condition(self, key: str, value: Any) -> None:
        """设置测量条件"""
        self._conditions[key] = value
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证性质数据"""
        errors = []
        
        if self._value is None:
            errors.append("Property value cannot be None")
        
        if not self._unit:
            errors.append("Property unit is required")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "category": self._category.value,
            "property_type": self._property_type.value,
            "value": self._value,
            "unit": self._unit,
            "structure_id": self._structure_id,
            "uncertainty": self._uncertainty,
            "conditions": self._conditions,
            "quality": self._quality.value,
            "metadata": self._metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PropertyData:
        prop = cls(
            property_type=PropertyType(data["property_type"]),
            value=data["value"],
            unit=data["unit"],
            structure_id=data.get("structure_id"),
            data_id=data.get("id")
        )
        
        if "uncertainty" in data:
            prop._uncertainty = data["uncertainty"]
        if "conditions" in data:
            prop._conditions = data["conditions"]
        if "quality" in data:
            prop._quality = DataQuality(data["quality"])
        if "metadata" in data:
            prop._metadata = data["metadata"]
        
        return prop
    
    def _create_copy(self) -> PropertyData:
        """创建副本"""
        return PropertyData(
            self._property_type,
            self._value,
            self._unit,
            self._structure_id
        )


class CalculationResultData(UnifiedDataModel):
    """
    标准化计算结果数据
    
    DFT/MD计算结果的统一表示
    """
    
    def __init__(
        self,
        calculation_type: str,
        structure_id: str,
        converged: bool,
        data_id: Optional[str] = None
    ):
        super().__init__(DataCategory.CALCULATION, data_id)
        self._calculation_type = calculation_type
        self._structure_id = structure_id
        self._converged = converged
        
        self._energy: Optional[float] = None
        self._forces: Optional[List[List[float]]] = None
        self._stress: Optional[List[float]] = None
        self._properties: Dict[str, Any] = {}
        self._elapsed_time: Optional[float] = None
        self._iterations: Optional[int] = None
    
    @property
    def calculation_type(self) -> str:
        return self._calculation_type
    
    @property
    def structure_id(self) -> str:
        return self._structure_id
    
    @property
    def converged(self) -> bool:
        return self._converged
    
    @property
    def energy(self) -> Optional[float]:
        return self._energy
    
    @energy.setter
    def energy(self, value: float) -> None:
        self._energy = value
        self._modified_at = datetime.now()
    
    @property
    def forces(self) -> Optional[List[List[float]]]:
        return self._forces
    
    @forces.setter
    def forces(self, value: List[List[float]]) -> None:
        self._forces = value
        self._modified_at = datetime.now()
    
    @property
    def stress(self) -> Optional[List[float]]:
        return self._stress
    
    @stress.setter
    def stress(self, value: List[float]) -> None:
        self._stress = value
        self._modified_at = datetime.now()
    
    def set_property(self, key: str, value: Any) -> None:
        """设置计算属性"""
        self._properties[key] = value
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """获取计算属性"""
        return self._properties.get(key, default)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证计算结果"""
        errors = []
        
        if not self._converged and self._energy is None:
            errors.append("Non-converged calculation must have at least energy")
        
        if self._forces:
            expected_len = len(self._forces)
            for i, force in enumerate(self._forces):
                if len(force) != 3:
                    errors.append(f"Force {i} must have 3 components")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "category": self._category.value,
            "calculation_type": self._calculation_type,
            "structure_id": self._structure_id,
            "converged": self._converged,
            "energy": self._energy,
            "forces": self._forces,
            "stress": self._stress,
            "properties": self._properties,
            "elapsed_time": self._elapsed_time,
            "iterations": self._iterations,
            "quality": self._quality.value,
            "metadata": self._metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CalculationResultData:
        result = cls(
            calculation_type=data["calculation_type"],
            structure_id=data["structure_id"],
            converged=data["converged"],
            data_id=data.get("id")
        )
        
        result._energy = data.get("energy")
        result._forces = data.get("forces")
        result._stress = data.get("stress")
        result._properties = data.get("properties", {})
        result._elapsed_time = data.get("elapsed_time")
        result._iterations = data.get("iterations")
        
        if "quality" in data:
            result._quality = DataQuality(data["quality"])
        if "metadata" in data:
            result._metadata = data["metadata"]
        
        return result
    
    def _create_copy(self) -> CalculationResultData:
        """创建副本"""
        return CalculationResultData(
            self._calculation_type,
            self._structure_id,
            self._converged
        )


class SimulationTrajectoryData(UnifiedDataModel):
    """
    标准化模拟轨迹数据
    
    MD/DFT轨迹的统一表示
    """
    
    def __init__(
        self,
        initial_structure_id: str,
        temperature: float,
        timestep: float,
        data_id: Optional[str] = None
    ):
        super().__init__(DataCategory.SIMULATION, data_id)
        self._initial_structure_id = initial_structure_id
        self._temperature = temperature
        self._timestep = timestep
        
        self._frames: List[Dict[str, Any]] = []
        self._ensemble: Optional[str] = None
        self._total_steps: int = 0
        self._dump_interval: int = 1
    
    @property
    def initial_structure_id(self) -> str:
        return self._initial_structure_id
    
    @property
    def temperature(self) -> float:
        return self._temperature
    
    @property
    def timestep(self) -> float:
        return self._timestep
    
    @property
    def num_frames(self) -> int:
        return len(self._frames)
    
    @property
    def total_time(self) -> float:
        """总模拟时间（ps）"""
        return self._total_steps * self._timestep / 1000.0
    
    def add_frame(
        self,
        structure_data: StructureData,
        step: int,
        energy: float,
        forces: Optional[List[List[float]]] = None,
        velocities: Optional[List[List[float]]] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """添加帧"""
        frame = {
            "step": step,
            "structure_id": structure_data.id,
            "energy": energy,
            "forces": forces,
            "velocities": velocities,
            "additional": additional_data or {}
        }
        self._frames.append(frame)
        self._total_steps = max(self._total_steps, step)
    
    def get_frame(self, index: int) -> Optional[Dict[str, Any]]:
        """获取特定帧"""
        if 0 <= index < len(self._frames):
            return self._frames[index]
        return None
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证轨迹数据"""
        errors = []
        
        if self._timestep <= 0:
            errors.append("Timestep must be positive")
        
        if not self._frames:
            errors.append("Trajectory must have at least one frame")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "category": self._category.value,
            "initial_structure_id": self._initial_structure_id,
            "temperature": self._temperature,
            "timestep": self._timestep,
            "ensemble": self._ensemble,
            "total_steps": self._total_steps,
            "dump_interval": self._dump_interval,
            "num_frames": self.num_frames,
            "total_time": self.total_time,
            "frames": self._frames,
            "quality": self._quality.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SimulationTrajectoryData:
        traj = cls(
            initial_structure_id=data["initial_structure_id"],
            temperature=data["temperature"],
            timestep=data["timestep"],
            data_id=data.get("id")
        )
        
        traj._ensemble = data.get("ensemble")
        traj._total_steps = data.get("total_steps", 0)
        traj._dump_interval = data.get("dump_interval", 1)
        traj._frames = data.get("frames", [])
        
        if "quality" in data:
            traj._quality = DataQuality(data["quality"])
        
        return traj
    
    def _create_copy(self) -> SimulationTrajectoryData:
        """创建副本"""
        return SimulationTrajectoryData(
            self._initial_structure_id,
            self._temperature,
            self._timestep
        )


class DataRepository:
    """
    数据仓库
    
    统一管理数据的存储和检索
    """
    
    def __init__(self):
        self._storage: Dict[str, UnifiedDataModel] = {}
        self._index_by_category: Dict[DataCategory, Set[str]] = {
            cat: set() for cat in DataCategory
        }
        self._index_by_tag: Dict[str, Set[str]] = {}
        self._index_by_structure: Dict[str, Set[str]] = {}
    
    def store(self, data: UnifiedDataModel) -> str:
        """存储数据"""
        self._storage[data.id] = data
        
        # 更新索引
        self._index_by_category[data.category].add(data.id)
        
        for tag in data._tags:
            if tag not in self._index_by_tag:
                self._index_by_tag[tag] = set()
            self._index_by_tag[tag].add(data.id)
        
        # 如果是性质或计算结果，建立结构索引
        if isinstance(data, PropertyData) and data.structure_id:
            if data.structure_id not in self._index_by_structure:
                self._index_by_structure[data.structure_id] = set()
            self._index_by_structure[data.structure_id].add(data.id)
        
        return data.id
    
    def retrieve(self, data_id: str) -> Optional[UnifiedDataModel]:
        """检索数据"""
        return self._storage.get(data_id)
    
    def find_by_category(self, category: DataCategory) -> List[UnifiedDataModel]:
        """按类别查找"""
        ids = self._index_by_category.get(category, set())
        return [self._storage[id] for id in ids if id in self._storage]
    
    def find_by_tag(self, tag: str) -> List[UnifiedDataModel]:
        """按标签查找"""
        ids = self._index_by_tag.get(tag, set())
        return [self._storage[id] for id in ids if id in self._storage]
    
    def find_by_structure(self, structure_id: str) -> List[UnifiedDataModel]:
        """按结构查找相关数据"""
        ids = self._index_by_structure.get(structure_id, set())
        return [self._storage[id] for id in ids if id in self._storage]
    
    def delete(self, data_id: str) -> bool:
        """删除数据"""
        if data_id not in self._storage:
            return False
        
        data = self._storage.pop(data_id)
        
        # 更新索引
        self._index_by_category[data.category].discard(data_id)
        
        for tag in data._tags:
            self._index_by_tag.get(tag, set()).discard(data_id)
        
        if isinstance(data, PropertyData) and data.structure_id:
            self._index_by_structure.get(data.structure_id, set()).discard(data_id)
        
        return True
    
    def export(self, data_id: str, format: str = "json") -> str:
        """导出数据"""
        data = self.retrieve(data_id)
        if not data:
            raise ValueError(f"Data not found: {data_id}")
        
        if format == "json":
            return json.dumps(data.to_dict(), indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_data(self, content: str, format: str = "json") -> UnifiedDataModel:
        """导入数据"""
        if format == "json":
            data = json.loads(content)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        category = DataCategory(data["category"])
        
        if category == DataCategory.STRUCTURE:
            obj = StructureData.from_dict(data)
        elif category == DataCategory.PROPERTY:
            obj = PropertyData.from_dict(data)
        elif category == DataCategory.CALCULATION:
            obj = CalculationResultData.from_dict(data)
        elif category == DataCategory.SIMULATION:
            obj = SimulationTrajectoryData.from_dict(data)
        else:
            raise ValueError(f"Unsupported category: {category}")
        
        self.store(obj)
        return obj


# 便捷函数
def create_structure_data(structure) -> StructureData:
    """从各种格式创建标准化结构数据"""
    try:
        # 尝试pymatgen
        from pymatgen.core import Structure
        if isinstance(structure, Structure):
            return StructureData.from_pymatgen(structure)
    except ImportError:
        pass
    
    try:
        # 尝试ASE
        from ase import Atoms
        if isinstance(structure, Atoms):
            return StructureData.from_ase(structure)
    except ImportError:
        pass
    
    raise ValueError("Cannot create StructureData from provided structure")


def get_repository() -> DataRepository:
    """获取全局数据仓库"""
    if not hasattr(get_repository, '_instance'):
        get_repository._instance = DataRepository()
    return get_repository._instance