"""
知识图谱模块 - Knowledge Graph for Materials Science
==================================================
提供材料实体抽取、关系建模和知识推理功能。

Author: DFT-LAMMPS Team
Date: 2025
"""

import os
import re
import json
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from collections import defaultdict
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityType(Enum):
    """实体类型枚举"""
    MATERIAL = "material"
    ELEMENT = "element"
    PROPERTY = "property"
    METHOD = "method"
    SOFTWARE = "software"
    RESEARCHER = "researcher"
    INSTITUTION = "institution"
    PUBLICATION = "publication"
    SYNTHESIS = "synthesis"
    APPLICATION = "application"
    PHENOMENON = "phenomenon"


class RelationType(Enum):
    """关系类型枚举"""
    HAS_PROPERTY = "has_property"
    HAS_ELEMENT = "has_element"
    USED_FOR = "used_for"
    SYNTHESIZED_BY = "synthesized_by"
    MEASURED_BY = "measured_by"
    SIMILAR_TO = "similar_to"
    DERIVED_FROM = "derived_from"
    CITED_IN = "cited_in"
    IMPLEMENTS = "implements"
    PRODUCES = "produces"
    INHIBITS = "inhibits"
    CATALYZES = "catalyzes"
    ALLOYS_WITH = "alloys_with"


@dataclass
class Entity:
    """知识图谱实体"""
    id: str
    name: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    confidence: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "properties": self.properties,
            "source": self.source,
            "confidence": self.confidence,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """从字典创建"""
        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=EntityType(data["entity_type"]),
            properties=data.get("properties", {}),
            source=data.get("source"),
            confidence=data.get("confidence", 1.0),
            created_at=data.get("created_at", datetime.now().isoformat())
        )


@dataclass
class Relation:
    """知识图谱关系"""
    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "properties": self.properties,
            "confidence": self.confidence,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        """从字典创建"""
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 1.0),
            created_at=data.get("created_at", datetime.now().isoformat())
        )


class EntityExtractor:
    """实体抽取器"""
    
    # 元素周期表
    ELEMENTS = {
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
        'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
        'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
        'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
        'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
        'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
        'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
        'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
    }
    
    # 材料性质关键词
    PROPERTY_KEYWORDS = {
        "band_gap": ["band gap", "bandgap", "eg", "e_g"],
        "formation_energy": ["formation energy", "e_form", "cohesive energy"],
        "bulk_modulus": ["bulk modulus", "b0", "incompressibility"],
        "shear_modulus": ["shear modulus", "g0", "rigidity"],
        "young_modulus": ["young's modulus", "young modulus", "elastic modulus"],
        "lattice_constant": ["lattice constant", "lattice parameter", "a0"],
        "density": ["density", "specific gravity"],
        "hardness": ["hardness", "vickers hardness", "mohs hardness"],
        "thermal_conductivity": ["thermal conductivity", "kappa", "heat conductivity"],
        "electrical_conductivity": ["electrical conductivity", "resistivity", "conductance"],
        "dielectric_constant": ["dielectric constant", "permittivity", "epsilon"],
        "magnetic_moment": ["magnetic moment", "magnetization", "magnetic susceptibility"],
        "superconducting_tc": ["critical temperature", "tc", "superconducting"],
        "melting_point": ["melting point", "fusion point"],
        "heat_capacity": ["heat capacity", "specific heat", "cp", "cv"],
        "ionization_energy": ["ionization energy", "ip", "ie"],
        "electron_affinity": ["electron affinity", "ea"],
        "work_function": ["work function", "wf"]
    }
    
    # 计算方法关键词
    METHOD_KEYWORDS = {
        "dft": ["density functional theory", "dft"],
        "lda": ["local density approximation", "lda"],
        "gga": ["generalized gradient approximation", "gga"],
        "pbe": ["pbe", "perdew-burke-ernzerhof"],
        "hse": ["hse", "hey-scuseria-ernzerhof", "hybrid functional"],
        "gw": ["gw approximation", "gw method"],
        "mp2": ["mp2", "møller-plesset"],
        "ccsd": ["ccsd", "coupled cluster"],
        "md": ["molecular dynamics", "md"],
        "aimd": ["ab initio md", "aimd"],
        "mc": ["monte carlo", "mc"],
        "neb": ["nudged elastic band", "neb"],
        "dft_u": ["dft+u", "hubbard u", "lda+u"],
        "vdw": ["van der waals", "dispersion correction", "dft-d"],
        "ml_potential": ["machine learning potential", "ml potential", "neural network potential"]
    }
    
    # 软件关键词
    SOFTWARE_KEYWORDS = {
        "vasp": ["vasp", "vienna ab initio simulation package"],
        "quantum_espresso": ["quantum espresso", "pwscf", "qe"],
        "lammps": ["lammps", "large-scale atomic/molecular massively parallel simulator"],
        "cp2k": ["cp2k"],
        "gaussian": ["gaussian", "g09", "g16"],
        "orca": ["orca"],
        "abinit": ["abinit"],
        "siesta": ["siesta"],
        "castep": ["castep"],
        "gamess": ["gamess", "gamess-us"],
        "nwchem": ["nwchem"],
        "phonopy": ["phonopy"],
        "pymatgen": ["pymatgen"],
        "ase": ["ase", "atomic simulation environment"],
        "ovito": ["ovito"],
        "vesta": ["vesta"],
        "xcrysden": ["xcrysden"]
    }
    
    # 合成方法关键词
    SYNTHESIS_KEYWORDS = {
        "sol_gel": ["sol-gel", "sol gel"],
        "hydrothermal": ["hydrothermal", "solvothermal"],
        "cvd": ["chemical vapor deposition", "cvd", "mocvd"],
        "pvd": ["physical vapor deposition", "pvd", "sputtering"],
        "mbbe": ["molecular beam epitaxy", "mbe"],
        "pld": ["pulsed laser deposition", "pld"],
        "electrodeposition": ["electrodeposition", "electroplating"],
        "ball_milling": ["ball milling", "mechanical alloying"],
        "solid_state": ["solid state reaction", "solid state synthesis"],
        "melt_synthesis": ["melt synthesis", "melt spinning"],
        "electrospinning": ["electrospinning"]
    }
    
    # 应用领域关键词
    APPLICATION_KEYWORDS = {
        "battery": ["battery", "lithium ion", "solid state battery"],
        "solar_cell": ["solar cell", "photovoltaic", "pv"],
        "catalyst": ["catalyst", "catalysis", "electrocatalysis"],
        "superconductor": ["superconductor", "superconducting"],
        "thermoelectric": ["thermoelectric", "seebeck"],
        "piezoelectric": ["piezoelectric", "piezoelectricity"],
        "magnetic_storage": ["magnetic storage", "hard disk"],
        "optoelectronic": ["optoelectronic", "led", "laser"],
        "sensor": ["sensor", "biosensor", "gas sensor"],
        "coating": ["coating", "protective coating", "thin film"],
        "biomedical": ["biomedical", "implant", "biocompatible"]
    }
    
    def __init__(self):
        self._entity_counter = 0
    
    def _generate_id(self, prefix: str = "ent") -> str:
        """生成唯一ID"""
        self._entity_counter += 1
        return f"{prefix}_{self._entity_counter:06d}"
    
    def extract_from_text(self, text: str, source: Optional[str] = None) -> Tuple[List[Entity], List[Relation]]:
        """
        从文本中提取实体和关系
        
        Args:
            text: 输入文本
            source: 来源标识
            
        Returns:
            (实体列表, 关系列表)
        """
        entities = []
        relations = []
        
        # 提取元素
        elements = self._extract_elements(text)
        for elem in elements:
            entity = Entity(
                id=self._generate_id("elem"),
                name=elem,
                entity_type=EntityType.ELEMENT,
                source=source
            )
            entities.append(entity)
        
        # 提取材料（化学式）
        materials = self._extract_materials(text)
        for mat in materials:
            entity = Entity(
                id=self._generate_id("mat"),
                name=mat,
                entity_type=EntityType.MATERIAL,
                properties={"formula": mat},
                source=source
            )
            entities.append(entity)
            
            # 建立材料-元素关系
            mat_elements = self._parse_formula_elements(mat)
            for elem in entities:
                if elem.entity_type == EntityType.ELEMENT and elem.name in mat_elements:
                    rel = Relation(
                        id=self._generate_id("rel"),
                        source_id=entity.id,
                        target_id=elem.id,
                        relation_type=RelationType.HAS_ELEMENT
                    )
                    relations.append(rel)
        
        # 提取性质
        properties = self._extract_properties(text)
        for prop_name, prop_info in properties.items():
            entity = Entity(
                id=self._generate_id("prop"),
                name=prop_name,
                entity_type=EntityType.PROPERTY,
                properties=prop_info,
                source=source
            )
            entities.append(entity)
            
            # 关联性质与材料
            for mat_entity in entities:
                if mat_entity.entity_type == EntityType.MATERIAL:
                    rel = Relation(
                        id=self._generate_id("rel"),
                        source_id=mat_entity.id,
                        target_id=entity.id,
                        relation_type=RelationType.HAS_PROPERTY,
                        properties={"value": prop_info.get("value")}
                    )
                    relations.append(rel)
        
        # 提取方法
        methods = self._extract_methods(text)
        for method in methods:
            entity = Entity(
                id=self._generate_id("meth"),
                name=method,
                entity_type=EntityType.METHOD,
                source=source
            )
            entities.append(entity)
        
        # 提取软件
        software = self._extract_software(text)
        for sw in software:
            entity = Entity(
                id=self._generate_id("sw"),
                name=sw,
                entity_type=EntityType.SOFTWARE,
                source=source
            )
            entities.append(entity)
        
        # 提取合成方法
        synthesis = self._extract_synthesis(text)
        for syn in synthesis:
            entity = Entity(
                id=self._generate_id("syn"),
                name=syn,
                entity_type=EntityType.SYNTHESIS,
                source=source
            )
            entities.append(entity)
        
        # 提取应用
        applications = self._extract_applications(text)
        for app in applications:
            entity = Entity(
                id=self._generate_id("app"),
                name=app,
                entity_type=EntityType.APPLICATION,
                source=source
            )
            entities.append(entity)
        
        return entities, relations
    
    def _extract_elements(self, text: str) -> Set[str]:
        """提取元素"""
        found = set()
        # 匹配独立的大写字母开头的单词
        for elem in self.ELEMENTS:
            pattern = r'\b' + elem + r'\b'
            if re.search(pattern, text):
                found.add(elem)
        return found
    
    def _extract_materials(self, text: str) -> Set[str]:
        """提取材料化学式"""
        materials = set()
        
        # 匹配常见的化学式模式
        # 简单模式: SiO2, Fe3O4, etc.
        pattern = r'\b([A-Z][a-z]?\d*[A-Z]?\d*)+\b'
        matches = re.findall(pattern, text)
        
        for match in matches:
            # 验证是否是有效的化学式
            if self._is_valid_formula(match):
                materials.add(match)
        
        return materials
    
    def _is_valid_formula(self, formula: str) -> bool:
        """验证是否是有效的化学式"""
        # 基本验证：包含至少一个元素符号
        element_pattern = r'[A-Z][a-z]?'
        elements = re.findall(element_pattern, formula)
        return len(elements) > 0 and any(e in self.ELEMENTS for e in elements)
    
    def _parse_formula_elements(self, formula: str) -> Set[str]:
        """解析化学式中的元素"""
        element_pattern = r'([A-Z][a-z]?)\d*'
        elements = re.findall(element_pattern, formula)
        return set(elements)
    
    def _extract_properties(self, text: str) -> Dict[str, Dict[str, Any]]:
        """提取性质及其数值"""
        properties = {}
        
        for prop_name, keywords in self.PROPERTY_KEYWORDS.items():
            for keyword in keywords:
                # 查找性质及其数值
                # 模式：property = value unit 或 property of X is value
                pattern = rf'{re.escape(keyword)}.*?[=\s]\s*([\d.]+)\s*(eV|GPa|g/cm³|K|J|Ω)?'
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    value = float(match.group(1))
                    unit = match.group(2) if match.group(2) else "unknown"
                    
                    properties[prop_name] = {
                        "value": value,
                        "unit": unit,
                        "keyword_found": keyword
                    }
        
        return properties
    
    def _extract_methods(self, text: str) -> Set[str]:
        """提取计算方法"""
        methods = set()
        
        for method_name, keywords in self.METHOD_KEYWORDS.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                    methods.add(method_name)
                    break
        
        return methods
    
    def _extract_software(self, text: str) -> Set[str]:
        """提取软件"""
        software = set()
        
        for sw_name, keywords in self.SOFTWARE_KEYWORDS.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                    software.add(sw_name)
                    break
        
        return software
    
    def _extract_synthesis(self, text: str) -> Set[str]:
        """提取合成方法"""
        synthesis = set()
        
        for syn_name, keywords in self.SYNTHESIS_KEYWORDS.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                    synthesis.add(syn_name)
                    break
        
        return synthesis
    
    def _extract_applications(self, text: str) -> Set[str]:
        """提取应用领域"""
        applications = set()
        
        for app_name, keywords in self.APPLICATION_KEYWORDS.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                    applications.add(app_name)
                    break
        
        return applications


class KnowledgeGraph:
    """
    材料科学知识图谱
    
    存储实体和关系，提供查询和推理功能。
    """
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self._indices: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.extractor = EntityExtractor()
        
        logger.info("KnowledgeGraph initialized")
    
    def add_entity(self, entity: Entity) -> str:
        """添加实体"""
        self.entities[entity.id] = entity
        
        # 更新索引
        self._indices["name"][entity.name.lower()].add(entity.id)
        self._indices["type"][entity.entity_type.value].add(entity.id)
        if entity.source:
            self._indices["source"][entity.source].add(entity.id)
        
        return entity.id
    
    def add_relation(self, relation: Relation) -> str:
        """添加关系"""
        self.relations[relation.id] = relation
        
        # 更新索引
        self._indices["source_entity"][relation.source_id].add(relation.id)
        self._indices["target_entity"][relation.target_id].add(relation.id)
        self._indices["relation_type"][relation.relation_type.value].add(relation.id)
        
        return relation.id
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        return self.entities.get(entity_id)
    
    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """获取关系"""
        return self.relations.get(relation_id)
    
    def find_entities_by_name(self, name: str) -> List[Entity]:
        """按名称查找实体"""
        ids = self._indices["name"].get(name.lower(), set())
        return [self.entities[id] for id in ids if id in self.entities]
    
    def find_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """按类型查找实体"""
        ids = self._indices["type"].get(entity_type.value, set())
        return [self.entities[id] for id in ids if id in self.entities]
    
    def find_relations_by_source(self, entity_id: str) -> List[Relation]:
        """查找源实体的所有关系"""
        ids = self._indices["source_entity"].get(entity_id, set())
        return [self.relations[id] for id in ids if id in self.relations]
    
    def find_relations_by_target(self, entity_id: str) -> List[Relation]:
        """查找目标实体的所有关系"""
        ids = self._indices["target_entity"].get(entity_id, set())
        return [self.relations[id] for id in ids if id in self.relations]
    
    def find_related_entities(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None
    ) -> List[Tuple[Entity, Relation]]:
        """查找相关实体"""
        results = []
        
        # 出向关系
        for rel in self.find_relations_by_source(entity_id):
            if relation_type is None or rel.relation_type == relation_type:
                target = self.get_entity(rel.target_id)
                if target:
                    results.append((target, rel))
        
        # 入向关系
        for rel in self.find_relations_by_target(entity_id):
            if relation_type is None or rel.relation_type == relation_type:
                source = self.get_entity(rel.source_id)
                if source:
                    results.append((source, rel))
        
        return results
    
    def ingest_text(self, text: str, source: Optional[str] = None) -> Dict[str, int]:
        """
        从文本中提取并添加知识
        
        Args:
            text: 输入文本
            source: 来源标识
            
        Returns:
            统计信息
        """
        entities, relations = self.extractor.extract_from_text(text, source)
        
        entity_count = 0
        relation_count = 0
        
        # 添加实体（去重）
        existing_names = {
            (e.name.lower(), e.entity_type): e.id 
            for e in self.entities.values()
        }
        
        entity_id_map = {}  # 临时映射，用于关系建立
        
        for entity in entities:
            key = (entity.name.lower(), entity.entity_type)
            if key in existing_names:
                entity_id_map[entity.id] = existing_names[key]
            else:
                new_id = self.add_entity(entity)
                entity_id_map[entity.id] = new_id
                entity_count += 1
                existing_names[key] = new_id
        
        # 添加关系
        for relation in relations:
            # 更新关系中的实体ID
            source_id = entity_id_map.get(relation.source_id, relation.source_id)
            target_id = entity_id_map.get(relation.target_id, relation.target_id)
            
            # 检查关系是否已存在
            exists = False
            for rel in self.relations.values():
                if (rel.source_id == source_id and 
                    rel.target_id == target_id and
                    rel.relation_type == relation.relation_type):
                    exists = True
                    break
            
            if not exists:
                relation.source_id = source_id
                relation.target_id = target_id
                self.add_relation(relation)
                relation_count += 1
        
        return {
            "entities_added": entity_count,
            "relations_added": relation_count,
            "total_entities": len(self.entities),
            "total_relations": len(self.relations)
        }
    
    def query(self, query_text: str) -> Dict[str, Any]:
        """
        查询知识图谱
        
        Args:
            query_text: 查询文本
            
        Returns:
            查询结果
        """
        results = {
            "entities": [],
            "relations": [],
            "suggestions": []
        }
        
        query_lower = query_text.lower()
        
        # 按名称搜索实体
        for entity in self.entities.values():
            if query_lower in entity.name.lower():
                results["entities"].append(entity.to_dict())
        
        # 搜索相关属性
        for entity in self.entities.values():
            for key, value in entity.properties.items():
                if query_lower in str(key).lower() or query_lower in str(value).lower():
                    if entity.to_dict() not in results["entities"]:
                        results["entities"].append(entity.to_dict())
        
        # 生成建议
        if not results["entities"]:
            results["suggestions"] = [
                "Try searching for specific material names or chemical formulas",
                "Search for elements (e.g., 'Fe', 'O')",
                "Search for properties (e.g., 'band gap')"
            ]
        
        return results
    
    def reason(
        self,
        premise: str,
        entity_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        知识推理
        
        Args:
            premise: 推理前提
            entity_id: 起始实体ID
            
        Returns:
            推理结果
        """
        inferences = []
        
        if entity_id:
            entity = self.get_entity(entity_id)
            if not entity:
                return inferences
            
            # 查找直接关系
            related = self.find_related_entities(entity_id)
            
            for rel_entity, relation in related:
                inference = {
                    "from": entity.name,
                    "relation": relation.relation_type.value,
                    "to": rel_entity.name,
                    "confidence": relation.confidence,
                    "explanation": f"{entity.name} {relation.relation_type.value.replace('_', ' ')} {rel_entity.name}"
                }
                inferences.append(inference)
            
            # 简单规则推理
            if entity.entity_type == EntityType.MATERIAL:
                # 如果材料有元素，查找包含相同元素的材料
                elements = self.find_related_entities(
                    entity_id, RelationType.HAS_ELEMENT
                )
                
                for elem, _ in elements:
                    similar_materials = self.find_related_entities(
                        elem.id, RelationType.HAS_ELEMENT
                    )
                    for mat, _ in similar_materials:
                        if mat.id != entity_id:
                            inferences.append({
                                "from": entity.name,
                                "inferred": "may be similar to",
                                "to": mat.name,
                                "explanation": f"Both contain {elem.name}",
                                "confidence": 0.6
                            })
        
        return inferences
    
    def find_similar_materials(
        self,
        material_name: str,
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        查找相似材料
        
        Args:
            material_name: 材料名称
            similarity_threshold: 相似度阈值
            
        Returns:
            相似材料列表
        """
        # 查找材料实体
        material_entities = self.find_entities_by_name(material_name)
        
        if not material_entities:
            return []
        
        material = material_entities[0]
        
        # 获取材料的元素
        elements = [
            e for e, _ in self.find_related_entities(material.id, RelationType.HAS_ELEMENT)
        ]
        
        element_ids = {e.id for e in elements}
        
        # 查找包含相同元素的材料
        similar = []
        
        for entity in self.find_entities_by_type(EntityType.MATERIAL):
            if entity.id == material.id:
                continue
            
            other_elements = [
                e for e, _ in self.find_related_entities(entity.id, RelationType.HAS_ELEMENT)
            ]
            other_element_ids = {e.id for e in other_elements}
            
            # 计算Jaccard相似度
            intersection = len(element_ids & other_element_ids)
            union = len(element_ids | other_element_ids)
            
            if union > 0:
                similarity = intersection / union
                if similarity >= similarity_threshold:
                    # 检查共同性质
                    common_properties = []
                    mat_props = [p for p, _ in self.find_related_entities(material.id, RelationType.HAS_PROPERTY)]
                    other_props = [p for p, _ in self.find_related_entities(entity.id, RelationType.HAS_PROPERTY)]
                    
                    for prop in mat_props:
                        if any(p.name == prop.name for p in other_props):
                            common_properties.append(prop.name)
                    
                    similar.append({
                        "material": entity.name,
                        "similarity": similarity,
                        "common_elements": list(element_ids & other_element_ids),
                        "common_properties": common_properties
                    })
        
        # 按相似度排序
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similar[:10]  # 返回前10个
    
    def get_property_statistics(
        self,
        property_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取性质统计信息
        
        Args:
            property_name: 特定性质名称，None表示所有性质
            
        Returns:
            统计信息
        """
        stats = defaultdict(list)
        
        for entity in self.find_entities_by_type(EntityType.PROPERTY):
            if property_name is None or entity.name == property_name:
                value = entity.properties.get("value")
                if value is not None and isinstance(value, (int, float)):
                    stats[entity.name].append(value)
        
        result = {}
        for prop_name, values in stats.items():
            if values:
                result[prop_name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "unit": "varies"  # 简化处理
                }
        
        return result
    
    def export_to_json(self, filepath: str):
        """导出到JSON文件"""
        data = {
            "entities": [e.to_dict() for e in self.entities.values()],
            "relations": [r.to_dict() for r in self.relations.values()],
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "entity_count": len(self.entities),
                "relation_count": len(self.relations)
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Knowledge graph exported to {filepath}")
    
    def import_from_json(self, filepath: str):
        """从JSON文件导入"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 导入实体
        for entity_data in data.get("entities", []):
            entity = Entity.from_dict(entity_data)
            self.add_entity(entity)
        
        # 导入关系
        for relation_data in data.get("relations", []):
            relation = Relation.from_dict(relation_data)
            self.add_relation(relation)
        
        logger.info(f"Knowledge graph imported from {filepath}")
    
    def export_to_neo4j_cypher(self, filepath: str):
        """导出为Neo4j Cypher脚本"""
        cypher_statements = []
        
        # 创建实体节点
        for entity in self.entities.values():
            props = json.dumps(entity.properties).replace('"', '\\"')
            cypher = (
                f'CREATE (e:{entity.entity_type.value} {{'
                f'id: "{entity.id}", '
                f'name: "{entity.name}", '
                f'properties: "{props}", '
                f'confidence: {entity.confidence}'
                f'}})'
            )
            cypher_statements.append(cypher)
        
        # 创建关系
        for relation in self.relations.values():
            cypher = (
                f'MATCH (a {{id: "{relation.source_id}"}}), (b {{id: "{relation.target_id}"}}) '
                f'CREATE (a)-[r:{relation.relation_type.value} {{'
                f'confidence: {relation.confidence}'
                f'}}]->(b)'
            )
            cypher_statements.append(cypher)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cypher_statements))
        
        logger.info(f"Cypher script exported to {filepath}")


class LiteratureMiningPipeline:
    """文献挖掘流水线"""
    
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
    
    def process_literature(
        self,
        texts: List[Dict[str, str]],
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        处理文献集
        
        Args:
            texts: 文献列表，每项包含text和source
            batch_size: 批处理大小
            
        Returns:
            处理统计
        """
        total_stats = {
            "documents_processed": 0,
            "entities_extracted": 0,
            "relations_extracted": 0
        }
        
        for i, item in enumerate(texts):
            logger.info(f"Processing document {i+1}/{len(texts)}: {item.get('source', 'unknown')}")
            
            stats = self.kg.ingest_text(item["text"], item.get("source"))
            
            total_stats["documents_processed"] += 1
            total_stats["entities_extracted"] += stats["entities_added"]
            total_stats["relations_extracted"] += stats["relations_added"]
        
        return total_stats
    
    def discover_materials_with_property(
        self,
        target_property: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        发现具有特定性质的材料
        
        Args:
            target_property: 目标性质
            min_value: 最小值
            max_value: 最大值
            
        Returns:
            材料列表
        """
        results = []
        
        # 查找所有具有该性质的材料
        for entity in self.kg.find_entities_by_type(EntityType.MATERIAL):
            properties = [
                p for p, _ in self.kg.find_related_entities(entity.id, RelationType.HAS_PROPERTY)
            ]
            
            for prop in properties:
                if prop.name == target_property or target_property.lower() in prop.name.lower():
                    value = prop.properties.get("value")
                    unit = prop.properties.get("unit", "")
                    
                    # 检查数值范围
                    if isinstance(value, (int, float)):
                        if min_value is not None and value < min_value:
                            continue
                        if max_value is not None and value > max_value:
                            continue
                    
                    results.append({
                        "material": entity.name,
                        "property": prop.name,
                        "value": value,
                        "unit": unit
                    })
        
        return results
    
    def identify_research_trends(self) -> Dict[str, Any]:
        """识别研究趋势"""
        trends = {
            "popular_methods": {},
            "popular_materials": {},
            "active_research_areas": {}
        }
        
        # 统计方法使用
        for entity in self.kg.find_entities_by_type(EntityType.METHOD):
            count = len(self.kg.find_relations_by_source(entity.id))
            trends["popular_methods"][entity.name] = count
        
        # 统计材料研究
        for entity in self.kg.find_entities_by_type(EntityType.MATERIAL):
            count = len(self.kg.find_related_entities(entity.id))
            trends["popular_materials"][entity.name] = count
        
        # 按次数排序
        trends["popular_methods"] = dict(
            sorted(trends["popular_methods"].items(), key=lambda x: x[1], reverse=True)[:10]
        )
        trends["popular_materials"] = dict(
            sorted(trends["popular_materials"].items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        return trends


# 便捷函数
def create_knowledge_graph() -> KnowledgeGraph:
    """创建知识图谱"""
    return KnowledgeGraph()


def extract_from_literature(text: str, source: Optional[str] = None) -> Dict[str, Any]:
    """从文献中提取知识"""
    kg = KnowledgeGraph()
    stats = kg.ingest_text(text, source)
    
    return {
        "stats": stats,
        "entities": [e.to_dict() for e in kg.entities.values()],
        "relations": [r.to_dict() for r in kg.relations.values()]
    }


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("KnowledgeGraph Test")
    print("=" * 60)
    
    # 测试1: 实体抽取
    print("\n1. Testing entity extraction...")
    sample_text = """
    We investigated the electronic properties of perovskite solar cell materials 
    using density functional theory with the VASP code. The band gap of CsPbI3 
    was calculated to be 1.73 eV using the HSE06 functional. The material was 
    synthesized using solution processing method. This makes it suitable for 
    photovoltaic applications.
    """
    
    kg = KnowledgeGraph()
    stats = kg.ingest_text(sample_text, "Test Paper 2024")
    print(f"Entities added: {stats['entities_added']}")
    print(f"Relations added: {stats['relations_added']}")
    
    # 显示提取的实体
    print("\nExtracted entities:")
    for entity in kg.entities.values():
        print(f"  - {entity.name} ({entity.entity_type.value})")
    
    # 测试2: 查询
    print("\n2. Testing query...")
    results = kg.query("CsPbI3")
    print(f"Query results: {len(results['entities'])} entities found")
    
    # 测试3: 推理
    print("\n3. Testing reasoning...")
    # 先找到CsPbI3的ID
    cs_entities = kg.find_entities_by_name("CsPbI3")
    if cs_entities:
        inferences = kg.reason("material properties", cs_entities[0].id)
        print(f"Inferences made: {len(inferences)}")
        for inf in inferences[:3]:
            print(f"  - {inf.get('explanation', 'N/A')}")
    
    # 测试4: 相似材料
    print("\n4. Testing similar materials...")
    similar = kg.find_similar_materials("CsPbI3")
    print(f"Found {len(similar)} similar materials")
    
    # 测试5: 性质统计
    print("\n5. Testing property statistics...")
    prop_stats = kg.get_property_statistics()
    print(f"Properties with statistics: {list(prop_stats.keys())}")
    
    # 测试6: 导出
    print("\n6. Testing export...")
    kg.export_to_json("/tmp/test_kg.json")
    print("Exported to /tmp/test_kg.json")
    
    # 测试7: 文献挖掘流水线
    print("\n7. Testing literature mining pipeline...")
    texts = [
        {
            "text": "Fe3O4 nanoparticles were synthesized by solvothermal method. "
                    "The saturation magnetization was 92 emu/g.",
            "source": "Paper A"
        },
        {
            "text": "TiO2 anatase has a band gap of 3.2 eV calculated using PBE functional.",
            "source": "Paper B"
        }
    ]
    
    pipeline = LiteratureMiningPipeline(kg)
    mining_stats = pipeline.process_literature(texts)
    print(f"Processed {mining_stats['documents_processed']} documents")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
