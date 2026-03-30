"""
Knowledge Builder - 知识构建器
=============================
提供知识图谱构建、实体标准化和关系抽取功能。
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """实体类型"""
    MATERIAL = "Material"
    ELEMENT = "Element"
    PROPERTY = "Property"
    STRUCTURE = "Structure"
    METHOD = "Method"
    CALCULATION = "Calculation"
    SIMULATION = "Simulation"
    EXPERIMENT = "Experiment"
    PUBLICATION = "Publication"
    RESEARCHER = "Researcher"
    INSTITUTION = "Institution"
    PHENOMENON = "Phenomenon"
    APPLICATION = "Application"


class RelationType(Enum):
    """关系类型"""
    HAS_PROPERTY = "HAS_PROPERTY"
    HAS_ELEMENT = "HAS_ELEMENT"
    HAS_STRUCTURE = "HAS_STRUCTURE"
    CALCULATED_BY = "CALCULATED_BY"
    MEASURED_BY = "MEASURED_BY"
    SIMULATED_BY = "SIMULATED_BY"
    IS_A = "IS_A"
    PART_OF = "PART_OF"
    DERIVED_FROM = "DERIVED_FROM"
    SIMILAR_TO = "SIMILAR_TO"
    CITED_IN = "CITED_IN"
    AUTHORED_BY = "AUTHORED_BY"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    CAUSES = "CAUSES"
    INHIBITS = "INHIBITS"
    ENABLES = "ENABLES"


@dataclass
class KnowledgeSchema:
    """知识图谱模式"""
    entity_types: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    relation_types: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        # 初始化默认实体类型
        if not self.entity_types:
            self.entity_types = self._default_entity_types()
        
        # 初始化默认关系类型
        if not self.relation_types:
            self.relation_types = self._default_relation_types()
    
    def _default_entity_types(self) -> Dict[str, Dict[str, Any]]:
        """默认实体类型定义"""
        return {
            "Material": {
                "properties": ["formula", "name", "structure_type", "space_group"],
                "required": ["formula"],
                "description": "Chemical material or compound"
            },
            "Element": {
                "properties": ["symbol", "atomic_number", "atomic_mass", "group", "period"],
                "required": ["symbol"],
                "description": "Chemical element"
            },
            "Property": {
                "properties": ["name", "value", "unit", "conditions", "uncertainty"],
                "required": ["name", "value"],
                "description": "Physical or chemical property"
            },
            "Structure": {
                "properties": ["type", "lattice_parameters", "space_group", "sites"],
                "required": ["type"],
                "description": "Crystal structure"
            },
            "Method": {
                "properties": ["name", "type", "parameters", "software", "accuracy"],
                "required": ["name", "type"],
                "description": "Computational or experimental method"
            },
            "Calculation": {
                "properties": ["type", "parameters", "results", "status", "runtime"],
                "required": ["type"],
                "description": "Simulation calculation"
            },
            "Publication": {
                "properties": ["title", "authors", "journal", "year", "doi", "abstract"],
                "required": ["title"],
                "description": "Scientific publication"
            }
        }
    
    def _default_relation_types(self) -> Dict[str, Dict[str, Any]]:
        """默认关系类型定义"""
        return {
            "HAS_PROPERTY": {
                "from": ["Material"],
                "to": ["Property"],
                "description": "Material has property"
            },
            "HAS_ELEMENT": {
                "from": ["Material"],
                "to": ["Element"],
                "description": "Material contains element"
            },
            "HAS_STRUCTURE": {
                "from": ["Material"],
                "to": ["Structure"],
                "description": "Material has structure"
            },
            "CALCULATED_BY": {
                "from": ["Property"],
                "to": ["Method"],
                "description": "Property calculated by method"
            },
            "MEASURED_BY": {
                "from": ["Property"],
                "to": ["Experiment"],
                "description": "Property measured by experiment"
            },
            "SIMULATED_BY": {
                "from": ["Material"],
                "to": ["Calculation"],
                "description": "Material simulated by calculation"
            },
            "IS_A": {
                "from": ["Material", "Method"],
                "to": ["Material", "Method"],
                "description": "Is-a relationship"
            },
            "DERIVED_FROM": {
                "from": ["Material", "Property"],
                "to": ["Material", "Property"],
                "description": "Derived from another entity"
            },
            "SIMILAR_TO": {
                "from": ["Material", "Structure"],
                "to": ["Material", "Structure"],
                "description": "Similar to another entity"
            },
            "CITED_IN": {
                "from": ["Publication"],
                "to": ["Publication"],
                "description": "Publication cited in another"
            },
            "AUTHORED_BY": {
                "from": ["Publication"],
                "to": ["Researcher"],
                "description": "Publication authored by researcher"
            }
        }
    
    def validate_entity(self, entity_type: str, properties: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证实体"""
        if entity_type not in self.entity_types:
            return False, [f"Unknown entity type: {entity_type}"]
        
        schema = self.entity_types[entity_type]
        errors = []
        
        # 检查必需属性
        for required in schema.get("required", []):
            if required not in properties:
                errors.append(f"Missing required property: {required}")
        
        return len(errors) == 0, errors
    
    def validate_relation(
        self,
        relation_type: str,
        from_type: str,
        to_type: str
    ) -> Tuple[bool, List[str]]:
        """验证关系"""
        if relation_type not in self.relation_types:
            return False, [f"Unknown relation type: {relation_type}"]
        
        schema = self.relation_types[relation_type]
        errors = []
        
        if from_type not in schema.get("from", []):
            errors.append(f"Invalid source type: {from_type}")
        
        if to_type not in schema.get("to", []):
            errors.append(f"Invalid target type: {to_type}")
        
        return len(errors) == 0, errors


class MaterialOntology:
    """材料科学本体"""
    
    def __init__(self):
        self._concepts: Dict[str, Dict[str, Any]] = {}
        self._hierarchy: Dict[str, List[str]] = {}  # parent -> children
        self._initialize()
    
    def _initialize(self):
        """初始化材料本体"""
        # 材料分类
        self.add_concept("Material", {
            "description": "Base concept for materials",
            "properties": ["formula", "name"]
        })
        
        self.add_concept("Inorganic", {"parent": "Material"})
        self.add_concept("Organic", {"parent": "Material"})
        self.add_concept("Composite", {"parent": "Material"})
        
        # 无机材料分类
        self.add_concept("Metal", {"parent": "Inorganic"})
        self.add_concept("Ceramic", {"parent": "Inorganic"})
        self.add_concept("Semiconductor", {"parent": "Inorganic"})
        self.add_concept("Superconductor", {"parent": "Inorganic"})
        
        # 结构分类
        self.add_concept("Crystal", {"parent": "Structure"})
        self.add_concept("Amorphous", {"parent": "Structure"})
        self.add_concept("Molecule", {"parent": "Structure"})
        
        # 性质分类
        self.add_concept("MechanicalProperty", {"parent": "Property"})
        self.add_concept("ElectronicProperty", {"parent": "Property"})
        self.add_concept("ThermalProperty", {"parent": "Property"})
        self.add_concept("OpticalProperty", {"parent": "Property"})
        
        # 具体性质
        self.add_concept("BandGap", {"parent": "ElectronicProperty"})
        self.add_concept("BulkModulus", {"parent": "MechanicalProperty"})
        self.add_concept("ThermalConductivity", {"parent": "ThermalProperty"})
    
    def add_concept(self, name: str, definition: Dict[str, Any]):
        """添加概念"""
        self._concepts[name] = definition
        
        # 更新层次结构
        parent = definition.get("parent")
        if parent:
            if parent not in self._hierarchy:
                self._hierarchy[parent] = []
            self._hierarchy[parent].append(name)
    
    def get_concept(self, name: str) -> Optional[Dict[str, Any]]:
        """获取概念定义"""
        return self._concepts.get(name)
    
    def get_children(self, concept: str) -> List[str]:
        """获取子概念"""
        return self._hierarchy.get(concept, [])
    
    def get_ancestors(self, concept: str) -> List[str]:
        """获取祖先概念"""
        ancestors = []
        current = concept
        
        while current:
            definition = self._concepts.get(current)
            if definition and definition.get("parent"):
                parent = definition["parent"]
                ancestors.append(parent)
                current = parent
            else:
                break
        
        return ancestors
    
    def is_a(self, concept: str, parent: str) -> bool:
        """检查概念继承关系"""
        if concept == parent:
            return True
        return parent in self.get_ancestors(concept)
    
    def infer_type(self, properties: Dict[str, Any]) -> List[str]:
        """根据属性推断类型"""
        types = []
        
        if "band_gap" in properties:
            types.append("Semiconductor")
        if "critical_temperature" in properties:
            types.append("Superconductor")
        if "bulk_modulus" in properties or "shear_modulus" in properties:
            types.append("MechanicalProperty")
        
        return types


class EntityNormalizer:
    """实体标准化器"""
    
    def __init__(self):
        self._synonyms: Dict[str, List[str]] = {}
        self._canonical: Dict[str, str] = {}
        self._initialize()
    
    def _initialize(self):
        """初始化同义词词典"""
        # 元素同义词
        self.add_synonyms("Lithium", ["Li", "lithium", "LITHIUM"])
        self.add_synonyms("Sodium", ["Na", "sodium", "SODIUM"])
        self.add_synonyms("Iron", ["Fe", "iron", "IRON"])
        self.add_synonyms("Oxygen", ["O", "oxygen", "OXYGEN"])
        self.add_synonyms("Silicon", ["Si", "silicon", "SILICON"])
        
        # 结构类型同义词
        self.add_synonyms("cubic", ["Cubic", "CUBIC", "bcc", "fcc"])
        self.add_synonyms("hexagonal", ["Hexagonal", "HEXAGONAL", "hcp"])
        self.add_synonyms("tetragonal", ["Tetragonal", "TETRAGONAL"])
        
        # 计算方法同义词
        self.add_synonyms("DFT", ["dft", "Density Functional Theory", "density functional theory"])
        self.add_synonyms("GGA", ["gga", "Generalized Gradient Approximation"])
        self.add_synonyms("LDA", ["lda", "Local Density Approximation"])
    
    def add_synonyms(self, canonical: str, synonyms: List[str]):
        """添加同义词"""
        self._synonyms[canonical] = synonyms
        for syn in synonyms:
            self._canonical[syn.lower()] = canonical
    
    def normalize(self, text: str) -> str:
        """标准化文本"""
        lower = text.lower().strip()
        return self._canonical.get(lower, text)
    
    def normalize_formula(self, formula: str) -> str:
        """标准化化学式"""
        # 移除空格
        formula = formula.replace(" ", "")
        
        # 统一数字格式 (如将下标数字转为正常数字)
        # 简化处理：保持原样
        
        return formula
    
    def normalize_structure(self, structure: str) -> str:
        """标准化结构类型"""
        return self.normalize(structure).lower()


class RelationExtractor:
    """关系抽取器"""
    
    def __init__(self, ontology: Optional[MaterialOntology] = None):
        self.ontology = ontology or MaterialOntology()
        self._patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """编译关系抽取模式"""
        return {
            "composition": re.compile(
                r'([A-Z][a-z]?\d*)+(?:\s*:\s*([A-Z][a-z]?\d*))*',
                re.IGNORECASE
            ),
            "property_value": re.compile(
                r'(\w+(?:\s+\w+)*)\s*[:=]\s*([\d.]+)\s*(\w+)',
                re.IGNORECASE
            ),
            "method_calculation": re.compile(
                r'(DFT|MD|MC|GW|MP2|CCSD)\s*(?:calculation|simulation)?',
                re.IGNORECASE
            )
        }
    
    def extract_from_text(self, text: str) -> List[Dict[str, Any]]:
        """从文本中抽取关系"""
        relations = []
        
        # 抽取组成关系
        comp_matches = self._patterns["composition"].findall(text)
        for match in comp_matches:
            relations.append({
                "type": "HAS_ELEMENT",
                "confidence": 0.8,
                "evidence": match[0] if isinstance(match, tuple) else match
            })
        
        # 抽取性质关系
        prop_matches = self._patterns["property_value"].findall(text)
        for prop_name, value, unit in prop_matches:
            relations.append({
                "type": "HAS_PROPERTY",
                "property": prop_name.strip(),
                "value": float(value),
                "unit": unit,
                "confidence": 0.9
            })
        
        # 抽取计算方法
        method_matches = self._patterns["method_calculation"].findall(text)
        for method in method_matches:
            relations.append({
                "type": "CALCULATED_BY",
                "method": method.upper(),
                "confidence": 0.85
            })
        
        return relations
    
    def extract_from_calculation(
        self,
        calculation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """从计算结果中抽取关系"""
        relations = []
        
        material = calculation.get("material", {})
        method = calculation.get("method", {})
        results = calculation.get("results", {})
        
        # 材料-计算关系
        if material and method:
            relations.append({
                "type": "SIMULATED_BY",
                "from": material.get("formula", ""),
                "to": method.get("name", ""),
                "confidence": 1.0
            })
        
        # 材料-性质关系
        if material and results:
            for prop_name, value in results.items():
                relations.append({
                    "type": "HAS_PROPERTY",
                    "from": material.get("formula", ""),
                    "property": prop_name,
                    "value": value,
                    "confidence": 0.95
                })
        
        # 性质-方法关系
        if method and results:
            for prop_name in results.keys():
                relations.append({
                    "type": "CALCULATED_BY",
                    "from": prop_name,
                    "to": method.get("name", ""),
                    "confidence": 0.95
                })
        
        return relations
    
    def extract_from_structure(self, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从结构数据抽取关系"""
        relations = []
        
        formula = structure.get("formula", "")
        structure_type = structure.get("structure_type", "")
        space_group = structure.get("space_group", "")
        
        if formula and structure_type:
            relations.append({
                "type": "HAS_STRUCTURE",
                "from": formula,
                "to": structure_type,
                "confidence": 1.0
            })
        
        if structure_type and space_group:
            relations.append({
                "type": "HAS_SPACE_GROUP",
                "from": structure_type,
                "to": space_group,
                "confidence": 1.0
            })
        
        return relations


class KnowledgeMerger:
    """知识合并器"""
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
    
    def merge_entities(
        self,
        entities1: List[Dict[str, Any]],
        entities2: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """合并实体列表"""
        merged = list(entities1)
        
        for e2 in entities2:
            is_duplicate = False
            for e1 in merged:
                if self._is_same_entity(e1, e2):
                    # 合并属性
                    self._merge_entity_properties(e1, e2)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(e2)
        
        return merged
    
    def merge_relations(
        self,
        relations1: List[Dict[str, Any]],
        relations2: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """合并关系列表"""
        merged = list(relations1)
        
        for r2 in relations2:
            is_duplicate = False
            for r1 in merged:
                if self._is_same_relation(r1, r2):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(r2)
        
        return merged
    
    def _is_same_entity(self, e1: Dict[str, Any], e2: Dict[str, Any]) -> bool:
        """判断是否为同一实体"""
        # 检查ID
        if e1.get("id") and e2.get("id") and e1["id"] == e2["id"]:
            return True
        
        # 检查名称和类型
        if e1.get("type") == e2.get("type"):
            name1 = e1.get("name", "").lower()
            name2 = e2.get("name", "").lower()
            
            # 完全匹配
            if name1 == name2:
                return True
            
            # 编辑距离
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, name1, name2).ratio()
            if similarity > self.threshold:
                return True
        
        return False
    
    def _is_same_relation(self, r1: Dict[str, Any], r2: Dict[str, Any]) -> bool:
        """判断是否为同一关系"""
        return (
            r1.get("type") == r2.get("type") and
            r1.get("from") == r2.get("from") and
            r1.get("to") == r2.get("to")
        )
    
    def _merge_entity_properties(
        self,
        e1: Dict[str, Any],
        e2: Dict[str, Any]
    ):
        """合并实体属性"""
        for key, value in e2.get("properties", {}).items():
            if key not in e1.get("properties", {}):
                if "properties" not in e1:
                    e1["properties"] = {}
                e1["properties"][key] = value
            elif e1["properties"][key] != value:
                # 属性冲突，保留更具体的值
                if value is not None:
                    e1["properties"][key] = value


class KnowledgeBuilder:
    """
    知识构建器
    
    整合各种组件，提供统一的知识图谱构建接口。
    """
    
    def __init__(
        self,
        schema: Optional[KnowledgeSchema] = None,
        ontology: Optional[MaterialOntology] = None,
        normalizer: Optional[EntityNormalizer] = None,
        extractor: Optional[RelationExtractor] = None,
        merger: Optional[KnowledgeMerger] = None
    ):
        self.schema = schema or KnowledgeSchema()
        self.ontology = ontology or MaterialOntology()
        self.normalizer = normalizer or EntityNormalizer()
        self.extractor = extractor or RelationExtractor(self.ontology)
        self.merger = merger or KnowledgeMerger()
        
        self._entities: List[Dict[str, Any]] = []
        self._relations: List[Dict[str, Any]] = []
    
    def add_entity(
        self,
        entity_type: str,
        name: str,
        properties: Optional[Dict[str, Any]] = None,
        source: str = ""
    ) -> Dict[str, Any]:
        """
        添加实体
        
        Args:
            entity_type: 实体类型
            name: 实体名称
            properties: 属性
            source: 来源
            
        Returns:
            创建的实体
        """
        # 标准化名称
        normalized_name = self.normalizer.normalize(name)
        
        entity = {
            "id": self._generate_id(entity_type, normalized_name),
            "type": entity_type,
            "name": normalized_name,
            "properties": properties or {},
            "source": source,
            "created_at": datetime.now().isoformat()
        }
        
        # 验证实体
        is_valid, errors = self.schema.validate_entity(entity_type, entity["properties"])
        if not is_valid:
            logger.warning(f"Entity validation failed: {errors}")
        
        self._entities.append(entity)
        return entity
    
    def add_relation(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        添加关系
        
        Args:
            relation_type: 关系类型
            from_entity: 源实体ID
            to_entity: 目标实体ID
            properties: 属性
            
        Returns:
            创建的关系
        """
        relation = {
            "id": self._generate_id("relation", f"{from_entity}-{relation_type}-{to_entity}"),
            "type": relation_type,
            "from": from_entity,
            "to": to_entity,
            "properties": properties or {},
            "created_at": datetime.now().isoformat()
        }
        
        # 验证关系
        from_type = self._get_entity_type(from_entity)
        to_type = self._get_entity_type(to_entity)
        is_valid, errors = self.schema.validate_relation(relation_type, from_type, to_type)
        if not is_valid:
            logger.warning(f"Relation validation failed: {errors}")
        
        self._relations.append(relation)
        return relation
    
    def build_from_calculation(self, calculation: Dict[str, Any]) -> Dict[str, Any]:
        """
        从计算结果构建知识
        
        Args:
            calculation: 计算结果
            
        Returns:
            构建的知识子图
        """
        subgraph = {
            "entities": [],
            "relations": []
        }
        
        # 提取材料实体
        material = calculation.get("material", {})
        if material:
            mat_entity = self.add_entity(
                entity_type="Material",
                name=material.get("formula", "Unknown"),
                properties={
                    "formula": material.get("formula"),
                    "structure_type": material.get("structure_type")
                },
                source="calculation"
            )
            subgraph["entities"].append(mat_entity)
            
            # 提取元素
            elements = self._extract_elements(material.get("formula", ""))
            for elem in elements:
                elem_entity = self.add_entity(
                    entity_type="Element",
                    name=elem,
                    source="calculation"
                )
                subgraph["entities"].append(elem_entity)
                
                # 添加组成关系
                rel = self.add_relation(
                    relation_type="HAS_ELEMENT",
                    from_entity=mat_entity["id"],
                    to_entity=elem_entity["id"]
                )
                subgraph["relations"].append(rel)
        
        # 提取方法实体
        method = calculation.get("method", {})
        if method:
            method_entity = self.add_entity(
                entity_type="Method",
                name=method.get("name", "Unknown"),
                properties={
                    "type": method.get("type"),
                    "software": method.get("software")
                },
                source="calculation"
            )
            subgraph["entities"].append(method_entity)
            
            # 添加计算关系
            if material:
                rel = self.add_relation(
                    relation_type="SIMULATED_BY",
                    from_entity=mat_entity["id"],
                    to_entity=method_entity["id"]
                )
                subgraph["relations"].append(rel)
        
        # 提取性质实体
        results = calculation.get("results", {})
        for prop_name, value in results.items():
            prop_entity = self.add_entity(
                entity_type="Property",
                name=prop_name,
                properties={"value": value},
                source="calculation"
            )
            subgraph["entities"].append(prop_entity)
            
            # 添加性质关系
            if material:
                rel1 = self.add_relation(
                    relation_type="HAS_PROPERTY",
                    from_entity=mat_entity["id"],
                    to_entity=prop_entity["id"]
                )
                subgraph["relations"].append(rel1)
            
            # 添加计算方法关系
            if method:
                rel2 = self.add_relation(
                    relation_type="CALCULATED_BY",
                    from_entity=prop_entity["id"],
                    to_entity=method_entity["id"]
                )
                subgraph["relations"].append(rel2)
        
        return subgraph
    
    def get_knowledge_graph(self) -> Dict[str, List[Dict[str, Any]]]:
        """获取完整知识图谱"""
        return {
            "entities": self._entities,
            "relations": self._relations,
            "schema": {
                "entity_types": self.schema.entity_types,
                "relation_types": self.schema.relation_types
            }
        }
    
    def export_to_neo4j(self, neo4j_db: Any) -> bool:
        """
        导出到Neo4j
        
        Args:
            neo4j_db: Neo4j数据库实例
            
        Returns:
            是否成功
        """
        try:
            from ..graph.neo4j_graph import NodeSpec, RelationSpec
            
            # 创建节点
            for entity in self._entities:
                node = NodeSpec(
                    labels=[entity["type"]],
                    properties={
                        "name": entity["name"],
                        **entity.get("properties", {})
                    }
                )
                neo4j_db.create_node(node)
            
            # 创建关系
            for relation in self._relations:
                rel = RelationSpec(
                    rel_type=relation["type"],
                    properties=relation.get("properties", {})
                )
                neo4j_db.create_relationship(
                    relation["from"],
                    relation["to"],
                    rel
                )
            
            logger.info(f"Exported {len(self._entities)} entities and {len(self._relations)} relations to Neo4j")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to Neo4j: {e}")
            return False
    
    def clear(self):
        """清空知识图谱"""
        self._entities = []
        self._relations = []
    
    def _generate_id(self, prefix: str, content: str) -> str:
        """生成唯一ID"""
        import hashlib
        content_hash = hashlib.md5(f"{prefix}:{content}".encode()).hexdigest()[:12]
        return f"{prefix}_{content_hash}"
    
    def _get_entity_type(self, entity_id: str) -> str:
        """获取实体类型"""
        for entity in self._entities:
            if entity["id"] == entity_id:
                return entity.get("type", "")
        return ""
    
    def _extract_elements(self, formula: str) -> List[str]:
        """从化学式提取元素"""
        import re
        # 匹配元素符号
        pattern = r'([A-Z][a-z]?)'
        elements = re.findall(pattern, formula)
        return list(set(elements))  # 去重


def create_knowledge_builder() -> KnowledgeBuilder:
    """工厂函数：创建知识构建器"""
    return KnowledgeBuilder()
