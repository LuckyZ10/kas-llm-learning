#!/usr/bin/env python3
"""
Global Materials Database
全球材料数据库 - 分布式材料数据存储与查询系统

Features:
- Multi-region data replication
- Unified material identification
- Cross-site query federation
- Data provenance tracking
- Compliance with regional data regulations

Author: DFTLammps Platform
Version: 2.0.0
"""

import asyncio
import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any, Union, Set, Tuple
from enum import Enum, auto
from collections import defaultdict
from datetime import datetime, timedelta
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataRegion(Enum):
    """数据区域"""
    NORTH_AMERICA = "na"
    EUROPE = "eu"
    ASIA_PACIFIC = "apac"
    CHINA = "cn"
    SOUTH_AMERICA = "sa"
    MIDDLE_EAST = "me"
    AFRICA = "af"


class MaterialProperty(Enum):
    """材料属性类型"""
    STRUCTURAL = "structural"      # 结构属性
    ELECTRONIC = "electronic"      # 电子属性
    THERMAL = "thermal"            # 热力学属性
    MECHANICAL = "mechanical"      # 力学属性
    OPTICAL = "optical"            # 光学属性
    MAGNETIC = "magnetic"          # 磁性属性
    CATALYTIC = "catalytic"        # 催化属性


class DataClassification(Enum):
    """数据分类"""
    PUBLIC = "public"              # 公开数据
    ACADEMIC = "academic"          # 学术数据
    COMMERCIAL = "commercial"      # 商业数据
    PROPRIETARY = "proprietary"    # 专有数据
    RESTRICTED = "restricted"      # 受限数据


@dataclass
class ChemicalComposition:
    """化学成分"""
    formula: str
    elements: Dict[str, float]  # 元素 -> 比例
    
    def to_dict(self) -> Dict:
        return {"formula": self.formula, "elements": self.elements}


@dataclass
class CrystalStructure:
    """晶体结构"""
    space_group: str
    lattice_parameters: Dict[str, float]  # a, b, c, alpha, beta, gamma
    atomic_positions: List[Dict]  # 原子位置
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MaterialPropertyValue:
    """材料属性值"""
    property_name: str
    property_type: MaterialProperty
    value: Any
    unit: str
    
    # 数据质量
    uncertainty: Optional[float] = None
    confidence: float = 0.95
    
    # 元数据
    calculation_method: str = ""  # DFT, MD, Experiment, etc.
    software: str = ""
    parameters: Dict = field(default_factory=dict)
    
    # 时间戳
    calculated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            "property_name": self.property_name,
            "property_type": self.property_type.value,
            "value": self.value,
            "unit": self.unit,
            "uncertainty": self.uncertainty,
            "confidence": self.confidence,
            "calculation_method": self.calculation_method,
            "software": self.software
        }


@dataclass
class MaterialRecord:
    """材料记录"""
    material_id: str  # 全局唯一ID
    name: str
    description: str = ""
    
    # 化学信息
    composition: Optional[ChemicalComposition] = None
    structure: Optional[CrystalStructure] = None
    
    # 属性数据
    properties: List[MaterialPropertyValue] = field(default_factory=list)
    
    # 数据治理
    classification: DataClassification = DataClassification.PUBLIC
    owner: str = ""
    contributors: List[str] = field(default_factory=list)
    
    # 区域信息
    primary_region: DataRegion = DataRegion.NORTH_AMERICA
    replicated_regions: List[DataRegion] = field(default_factory=list)
    
    # 元数据
    tags: List[str] = field(default_factory=list)
    external_refs: Dict[str, str] = field(default_factory=dict)  # 外部数据库引用
    
    # 审计
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    
    def __post_init__(self):
        if not self.material_id:
            # 生成全局唯一ID
            self.material_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """生成材料ID"""
        if self.composition:
            base = f"{self.composition.formula}-{self.structure.space_group if self.structure else 'amorphous'}"
        else:
            base = f"{self.name}-{time.time()}"
        
        return f"MAT-{hashlib.sha256(base.encode()).hexdigest()[:16].upper()}"
    
    def get_fingerprint(self) -> str:
        """获取材料指纹 (用于去重)"""
        if self.composition and self.structure:
            return f"{self.composition.formula}_{self.structure.space_group}"
        return self.material_id
    
    def to_dict(self) -> Dict:
        return {
            "material_id": self.material_id,
            "name": self.name,
            "description": self.description,
            "composition": self.composition.to_dict() if self.composition else None,
            "structure": self.structure.to_dict() if self.structure else None,
            "properties": [p.to_dict() for p in self.properties],
            "classification": self.classification.value,
            "primary_region": self.primary_region.value,
            "tags": self.tags,
            "version": self.version
        }


@dataclass
class QueryFilter:
    """查询过滤器"""
    # 化学成分过滤
    elements_include: List[str] = field(default_factory=list)
    elements_exclude: List[str] = field(default_factory=list)
    formula_pattern: Optional[str] = None
    
    # 属性过滤
    property_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # 结构过滤
    space_groups: List[str] = field(default_factory=list)
    
    # 数据治理过滤
    classifications: List[DataClassification] = field(default_factory=list)
    
    # 区域过滤
    regions: List[DataRegion] = field(default_factory=list)
    
    # 文本搜索
    search_text: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class FederatedQuery:
    """联邦查询"""
    query_id: str
    filter: QueryFilter
    
    # 查询选项
    include_calculated: bool = True  # 包含计算数据
    include_experimental: bool = True  # 包含实验数据
    
    # 结果选项
    sort_by: str = "relevance"
    limit: int = 100
    offset: int = 0
    
    # 性能选项
    timeout_seconds: int = 30
    parallel_execution: bool = True


@dataclass
class QueryResult:
    """查询结果"""
    query_id: str
    materials: List[MaterialRecord]
    
    # 性能指标
    execution_time_ms: float
    regions_queried: List[DataRegion]
    total_matches: int
    
    # 统计
    by_region: Dict[str, int]
    by_property_type: Dict[str, int]


class RegionalDataNode:
    """区域数据节点"""
    
    def __init__(self, region: DataRegion, node_id: str):
        self.region = region
        self.node_id = node_id
        self.materials: Dict[str, MaterialRecord] = {}
        self.indexes: Dict[str, Dict] = defaultdict(dict)
        self._lock = asyncio.Lock()
    
    async def store(self, record: MaterialRecord) -> bool:
        """存储材料记录"""
        async with self._lock:
            self.materials[record.material_id] = record
            
            # 更新索引
            if record.composition:
                for element in record.composition.elements.keys():
                    self.indexes["element"][element] = record.material_id
            
            if record.structure:
                self.indexes["space_group"][record.structure.space_group] = record.material_id
            
            for tag in record.tags:
                self.indexes["tag"][tag] = record.material_id
            
            return True
    
    async def query(self, filter: QueryFilter, limit: int = 100) -> List[MaterialRecord]:
        """本地查询"""
        results = []
        
        # 使用索引快速过滤
        candidate_ids = None
        
        if filter.elements_include:
            for element in filter.elements_include:
                if element in self.indexes["element"]:
                    ids = {self.indexes["element"][element]}
                    candidate_ids = candidate_ids & ids if candidate_ids else ids
        
        if candidate_ids is None:
            candidate_ids = set(self.materials.keys())
        
        # 应用详细过滤
        for material_id in candidate_ids:
            record = self.materials.get(material_id)
            if not record:
                continue
            
            if self._matches_filter(record, filter):
                results.append(record)
            
            if len(results) >= limit:
                break
        
        return results
    
    def _matches_filter(self, record: MaterialRecord, filter: QueryFilter) -> bool:
        """检查记录是否匹配过滤器"""
        # 元素过滤
        if filter.elements_include and record.composition:
            if not all(
                el in record.composition.elements
                for el in filter.elements_include
            ):
                return False
        
        if filter.elements_exclude and record.composition:
            if any(
                el in record.composition.elements
                for el in filter.elements_exclude
            ):
                return False
        
        # 属性范围过滤
        if filter.property_ranges:
            for prop_name, (min_val, max_val) in filter.property_ranges.items():
                prop = next(
                    (p for p in record.properties if p.property_name == prop_name),
                    None
                )
                if not prop or not (min_val <= prop.value <= max_val):
                    return False
        
        # 分类过滤
        if filter.classifications:
            if record.classification not in filter.classifications:
                return False
        
        # 标签过滤
        if filter.tags:
            if not all(tag in record.tags for tag in filter.tags):
                return False
        
        return True
    
    def get_stats(self) -> Dict:
        """获取节点统计"""
        return {
            "region": self.region.value,
            "total_materials": len(self.materials),
            "elements_indexed": len(self.indexes["element"]),
            "space_groups_indexed": len(self.indexes["space_group"])
        }


class DataReplicationManager:
    """数据复制管理器"""
    
    def __init__(self):
        self.replication_policies: Dict[DataClassification, List[DataRegion]] = {
            DataClassification.PUBLIC: [
                DataRegion.NORTH_AMERICA, DataRegion.EUROPE, 
                DataRegion.ASIA_PACIFIC, DataRegion.CHINA
            ],
            DataClassification.ACADEMIC: [
                DataRegion.NORTH_AMERICA, DataRegion.EUROPE, DataRegion.ASIA_PACIFIC
            ],
            DataClassification.COMMERCIAL: [
                DataRegion.NORTH_AMERICA, DataRegion.EUROPE
            ],
            DataClassification.PROPRIETARY: [
                DataRegion.NORTH_AMERICA
            ],
            DataClassification.RESTRICTED: []  # 不复制
        }
        
        self.compliance_rules: Dict[DataRegion, List[str]] = {
            DataRegion.EUROPE: ["GDPR"],
            DataRegion.CHINA: ["CSL", "DSL"],
            DataRegion.NORTH_AMERICA: ["CCPA"]
        }
    
    def get_replication_targets(
        self,
        record: MaterialRecord
    ) -> List[DataRegion]:
        """获取复制目标区域"""
        return self.replication_policies.get(record.classification, [])
    
    def check_compliance(
        self,
        record: MaterialRecord,
        target_region: DataRegion
    ) -> Tuple[bool, List[str]]:
        """检查合规性"""
        requirements = self.compliance_rules.get(target_region, [])
        
        # 简化实现 - 实际应该检查具体的数据处理要求
        violations = []
        
        if "GDPR" in requirements and not record.owner:
            violations.append("Missing data owner for GDPR compliance")
        
        return (len(violations) == 0, violations)


class GlobalMaterialsDB:
    """全球材料数据库"""
    
    def __init__(self):
        self.nodes: Dict[DataRegion, RegionalDataNode] = {}
        self.replication_manager = DataReplicationManager()
        
        # 全局索引
        self.material_locations: Dict[str, List[DataRegion]] = defaultdict(list)
        self.fingerprints: Dict[str, str] = {}  # 指纹 -> material_id
        
        # 查询统计
        self.query_stats = {
            "total_queries": 0,
            "avg_response_time_ms": 0
        }
    
    def register_node(self, node: RegionalDataNode):
        """注册区域节点"""
        self.nodes[node.region] = node
        logger.info(f"Registered data node: {node.region.value}")
    
    async def insert_material(self, record: MaterialRecord) -> str:
        """
        插入材料记录
        
        流程：
        1. 生成/检查全局ID
        2. 存储到主区域
        3. 根据策略复制到其他区域
        """
        # 检查重复
        fingerprint = record.get_fingerprint()
        if fingerprint in self.fingerprints:
            existing_id = self.fingerprints[fingerprint]
            logger.info(f"Material with fingerprint {fingerprint} already exists: {existing_id}")
            return existing_id
        
        # 存储到主区域
        primary_node = self.nodes.get(record.primary_region)
        if not primary_node:
            raise ValueError(f"Primary region {record.primary_region} not available")
        
        await primary_node.store(record)
        self.fingerprints[fingerprint] = record.material_id
        self.material_locations[record.material_id].append(record.primary_region)
        
        # 复制到其他区域
        replication_targets = self.replication_manager.get_replication_targets(record)
        
        for target_region in replication_targets:
            if target_region == record.primary_region:
                continue
            
            # 检查合规性
            compliant, violations = self.replication_manager.check_compliance(
                record, target_region
            )
            
            if not compliant:
                logger.warning(
                    f"Cannot replicate {record.material_id} to {target_region}: {violations}"
                )
                continue
            
            # 执行复制
            target_node = self.nodes.get(target_region)
            if target_node:
                await target_node.store(record)
                record.replicated_regions.append(target_region)
                self.material_locations[record.material_id].append(target_region)
                logger.info(f"Replicated {record.material_id} to {target_region.value}")
        
        return record.material_id
    
    async def query(
        self,
        filter: QueryFilter,
        regions: Optional[List[DataRegion]] = None
    ) -> QueryResult:
        """
        跨区查询
        """
        query_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:16]
        start_time = asyncio.get_event_loop().time()
        
        # 确定查询区域
        target_regions = regions or list(self.nodes.keys())
        
        # 并行查询各区域
        tasks = [
            self._query_region(region, filter)
            for region in target_regions
            if region in self.nodes
        ]
        
        region_results = await asyncio.gather(*tasks)
        
        # 合并结果
        all_materials = []
        seen_ids = set()
        by_region = defaultdict(int)
        by_property_type = defaultdict(int)
        
        for region, materials in zip(target_regions, region_results):
            for record in materials:
                if record.material_id not in seen_ids:
                    all_materials.append(record)
                    seen_ids.add(record.material_id)
                    by_region[region.value] += 1
                    
                    for prop in record.properties:
                        by_property_type[prop.property_type.value] += 1
        
        execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # 更新统计
        self.query_stats["total_queries"] += 1
        self.query_stats["avg_response_time_ms"] = (
            (self.query_stats["avg_response_time_ms"] * (self.query_stats["total_queries"] - 1) +
             execution_time) / self.query_stats["total_queries"]
        )
        
        return QueryResult(
            query_id=query_id,
            materials=all_materials,
            execution_time_ms=execution_time,
            regions_queried=target_regions,
            total_matches=len(all_materials),
            by_region=dict(by_region),
            by_property_type=dict(by_property_type)
        )
    
    async def _query_region(
        self,
        region: DataRegion,
        filter: QueryFilter
    ) -> List[MaterialRecord]:
        """查询单个区域"""
        node = self.nodes.get(region)
        if not node:
            return []
        
        try:
            return await node.query(filter)
        except Exception as e:
            logger.error(f"Query failed for region {region}: {e}")
            return []
    
    async def get_material(self, material_id: str) -> Optional[MaterialRecord]:
        """获取材料记录 (从最近的位置)"""
        regions = self.material_locations.get(material_id, [])
        
        for region in regions:
            node = self.nodes.get(region)
            if node and material_id in node.materials:
                return node.materials[material_id]
        
        return None
    
    def get_database_stats(self) -> Dict:
        """获取数据库统计"""
        total_materials = len(self.fingerprints)
        
        region_stats = {}
        for region, node in self.nodes.items():
            region_stats[region.value] = node.get_stats()
        
        return {
            "total_unique_materials": total_materials,
            "total_replicas": sum(
                len(regions) for regions in self.material_locations.values()
            ),
            "regions": region_stats,
            "query_stats": self.query_stats
        }


# 示例使用
async def demo():
    """全球材料数据库演示"""
    
    db = GlobalMaterialsDB()
    
    # 创建区域节点
    print("=== 初始化区域节点 ===")
    regions = [
        (DataRegion.NORTH_AMERICA, "na-node-01"),
        (DataRegion.EUROPE, "eu-node-01"),
        (DataRegion.ASIA_PACIFIC, "apac-node-01"),
        (DataRegion.CHINA, "cn-node-01"),
    ]
    
    for region, node_id in regions:
        node = RegionalDataNode(region, node_id)
        db.register_node(node)
    
    # 插入材料数据
    print("\n=== 插入材料数据 ===")
    
    materials_data = [
        {
            "name": "Silicon Crystal",
            "composition": ChemicalComposition(
                formula="Si",
                elements={"Si": 1.0}
            ),
            "structure": CrystalStructure(
                space_group="Fd-3m",
                lattice_parameters={"a": 5.43, "b": 5.43, "c": 5.43, 
                                   "alpha": 90, "beta": 90, "gamma": 90},
                atomic_positions=[{"element": "Si", "x": 0, "y": 0, "z": 0}]
            ),
            "primary_region": DataRegion.NORTH_AMERICA,
            "classification": DataClassification.PUBLIC,
            "properties": [
                MaterialPropertyValue(
                    property_name="band_gap",
                    property_type=MaterialProperty.ELECTRONIC,
                    value=1.12,
                    unit="eV",
                    calculation_method="DFT-PBE",
                    software="VASP"
                ),
                MaterialPropertyValue(
                    property_name="bulk_modulus",
                    property_type=MaterialProperty.MECHANICAL,
                    value=97.8,
                    unit="GPa"
                )
            ],
            "tags": ["semiconductor", "elemental", "cubic"]
        },
        {
            "name": "Lithium Cobalt Oxide",
            "composition": ChemicalComposition(
                formula="LiCoO2",
                elements={"Li": 0.33, "Co": 0.33, "O": 0.33}
            ),
            "structure": CrystalStructure(
                space_group="R-3m",
                lattice_parameters={"a": 2.82, "b": 2.82, "c": 14.05,
                                   "alpha": 90, "beta": 90, "gamma": 120},
                atomic_positions=[]
            ),
            "primary_region": DataRegion.EUROPE,
            "classification": DataClassification.ACADEMIC,
            "properties": [
                MaterialPropertyValue(
                    property_name="voltage",
                    property_type=MaterialProperty.ELECTRONIC,
                    value=3.7,
                    unit="V"
                )
            ],
            "tags": ["battery", "cathode", "layered"]
        },
        {
            "name": "Graphene",
            "composition": ChemicalComposition(
                formula="C",
                elements={"C": 1.0}
            ),
            "structure": CrystalStructure(
                space_group="P6/mmm",
                lattice_parameters={"a": 2.46, "b": 2.46, "c": 6.7,
                                   "alpha": 90, "beta": 90, "gamma": 120},
                atomic_positions=[]
            ),
            "primary_region": DataRegion.CHINA,
            "classification": DataClassification.PUBLIC,
            "properties": [
                MaterialPropertyValue(
                    property_name="thermal_conductivity",
                    property_type=MaterialProperty.THERMAL,
                    value=5000,
                    unit="W/(m·K)"
                ),
                MaterialPropertyValue(
                    property_name="electron_mobility",
                    property_type=MaterialProperty.ELECTRONIC,
                    value=200000,
                    unit="cm²/(V·s)"
                )
            ],
            "tags": ["2d-material", "carbon", "high-mobility"]
        },
    ]
    
    for data in materials_data:
        record = MaterialRecord(
            material_id="",
            name=data["name"],
            composition=data.get("composition"),
            structure=data.get("structure"),
            primary_region=data["primary_region"],
            classification=data["classification"],
            properties=data.get("properties", []),
            tags=data.get("tags", [])
        )
        
        material_id = await db.insert_material(record)
        print(f"  Inserted: {record.name} ({material_id})")
        print(f"    Primary region: {record.primary_region.value}")
        print(f"    Replicated to: {[r.value for r in record.replicated_regions]}")
    
    # 查询测试
    print("\n=== 跨区查询测试 ===")
    
    # 按元素查询
    filter1 = QueryFilter(elements_include=["Si"])
    result1 = await db.query(filter1)
    print(f"  Query: elements_include=['Si']")
    print(f"    Found: {result1.total_matches} materials")
    print(f"    Response time: {result1.execution_time_ms:.2f}ms")
    print(f"    By region: {result1.by_region}")
    
    # 按属性范围查询
    filter2 = QueryFilter(
        property_ranges={"band_gap": (1.0, 2.0)},
        tags=["semiconductor"]
    )
    result2 = await db.query(filter2)
    print(f"\n  Query: band_gap in [1.0, 2.0], tags=['semiconductor']")
    print(f"    Found: {result2.total_matches} materials")
    for m in result2.materials:
        print(f"    - {m.name} ({m.material_id})")
    
    # 按空间群查询
    filter3 = QueryFilter(space_groups=["R-3m"])
    result3 = await db.query(filter3)
    print(f"\n  Query: space_groups=['R-3m']")
    print(f"    Found: {result3.total_matches} materials")
    
    # 数据库统计
    print("\n=== 数据库统计 ===")
    stats = db.get_database_stats()
    print(f"  Total unique materials: {stats['total_unique_materials']}")
    print(f"  Total replicas: {stats['total_replicas']}")
    print(f"  Query stats: {stats['query_stats']}")
    
    print("\n  Regional stats:")
    for region, node_stats in stats['regions'].items():
        print(f"    {region}: {node_stats['total_materials']} materials")


if __name__ == "__main__":
    import time
    asyncio.run(demo())
