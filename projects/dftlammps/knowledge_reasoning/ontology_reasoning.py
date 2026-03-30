"""
本体推理模块 - Ontology Reasoning for Materials Science

实现基于描述逻辑的本体推理，支持概念分类、属性推理和一致性检查。
适用于材料科学领域的知识组织和推理。
"""

from typing import List, Dict, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn


class ConceptType(Enum):
    """概念类型"""
    PRIMITIVE = auto()      # 原始概念
    DEFINED = auto()        # 定义概念
    ANONYMOUS = auto()      # 匿名概念（限制）


class RoleType(Enum):
    """角色/关系类型"""
    OBJECT_PROPERTY = auto()    # 对象属性
    DATA_PROPERTY = auto()      # 数据属性
    ANNOTATION = auto()         # 注释属性


@dataclass
class Concept:
    """本体概念"""
    name: str
    concept_type: ConceptType
    parents: Set[str] = field(default_factory=set)
    equivalent_to: Set[str] = field(default_factory=set)
    disjoint_with: Set[str] = field(default_factory=set)
    restrictions: List[Dict[str, Any]] = field(default_factory=list)
    annotations: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if not isinstance(other, Concept):
            return False
        return self.name == other.name
    
    def is_subconcept_of(self, other: 'Concept', hierarchy: 'ConceptHierarchy') -> bool:
        """检查是否是指定概念的子概念"""
        if other.name in self.parents:
            return True
        # 递归检查父概念
        for parent_name in self.parents:
            parent = hierarchy.get_concept(parent_name)
            if parent and parent.is_subconcept_of(other, hierarchy):
                return True
        return False


@dataclass
class Role:
    """本体角色/关系"""
    name: str
    role_type: RoleType
    domain: Optional[str] = None
    range: Optional[str] = None
    inverse: Optional[str] = None
    transitive: bool = False
    symmetric: bool = False
    functional: bool = False
    subroles: Set[str] = field(default_factory=set)
    
    def __hash__(self):
        return hash(self.name)


@dataclass
class Individual:
    """本体实例/个体"""
    name: str
    types: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)
    relations: Dict[str, List[str]] = field(default_factory=dict)
    annotations: Dict[str, Any] = field(default_factory=dict)


class ConceptHierarchy:
    """概念层次结构管理"""
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.roles: Dict[str, Role] = {}
        self.individuals: Dict[str, Individual] = {}
        self.top_concept = Concept("Thing", ConceptType.PRIMITIVE)
        self.bottom_concept = Concept("Nothing", ConceptType.PRIMITIVE)
        self.concepts["Thing"] = self.top_concept
        self.concepts["Nothing"] = self.bottom_concept
    
    def add_concept(self, concept: Concept) -> bool:
        """添加概念"""
        if concept.name in self.concepts:
            return False
        self.concepts[concept.name] = concept
        # 默认父概念为Thing
        if not concept.parents and concept.name != "Thing":
            concept.parents.add("Thing")
        return True
    
    def get_concept(self, name: str) -> Optional[Concept]:
        """获取概念"""
        return self.concepts.get(name)
    
    def add_role(self, role: Role) -> bool:
        """添加角色"""
        if role.name in self.roles:
            return False
        self.roles[role.name] = role
        return True
    
    def add_individual(self, individual: Individual) -> bool:
        """添加实例"""
        if individual.name in self.individuals:
            return False
        self.individuals[individual.name] = individual
        return True
    
    def get_ancestors(self, concept_name: str) -> Set[str]:
        """获取所有祖先概念"""
        ancestors = set()
        concept = self.get_concept(concept_name)
        if not concept:
            return ancestors
        
        to_process = list(concept.parents)
        while to_process:
            parent_name = to_process.pop()
            if parent_name not in ancestors:
                ancestors.add(parent_name)
                parent = self.get_concept(parent_name)
                if parent:
                    to_process.extend(parent.parents)
        
        return ancestors
    
    def get_descendants(self, concept_name: str) -> Set[str]:
        """获取所有后代概念"""
        descendants = set()
        
        for name, concept in self.concepts.items():
            if concept_name in self.get_ancestors(name):
                descendants.add(name)
        
        return descendants
    
    def get_siblings(self, concept_name: str) -> Set[str]:
        """获取兄弟概念（同父概念）"""
        concept = self.get_concept(concept_name)
        if not concept:
            return set()
        
        siblings = set()
        for parent_name in concept.parents:
            parent = self.get_concept(parent_name)
            if parent:
                for child_name in self.get_descendants(parent_name):
                    if child_name != concept_name:
                        siblings.add(child_name)
        
        return siblings
    
    def compute_lca(self, concept1: str, concept2: str) -> Optional[str]:
        """计算最近公共祖先（LCA）"""
        ancestors1 = self.get_ancestors(concept1) | {concept1}
        ancestors2 = self.get_ancestors(concept2) | {concept2}
        
        common_ancestors = ancestors1 & ancestors2
        if not common_ancestors:
            return None
        
        # 找到最深的公共祖先
        deepest = None
        max_depth = -1
        for ancestor in common_ancestors:
            depth = len(self.get_ancestors(ancestor))
            if depth > max_depth:
                max_depth = depth
                deepest = ancestor
        
        return deepest
    
    def compute_similarity(self, concept1: str, concept2: str) -> float:
        """计算概念间的相似度（基于层次距离）"""
        if concept1 == concept2:
            return 1.0
        
        lca = self.compute_lca(concept1, concept2)
        if not lca:
            return 0.0
        
        # Wu-Palmer相似度
        depth1 = len(self.get_ancestors(concept1)) + 1
        depth2 = len(self.get_ancestors(concept2)) + 1
        depth_lca = len(self.get_ancestors(lca)) + 1
        
        return 2 * depth_lca / (depth1 + depth2)


class DescriptionLogicReasoner:
    """
    描述逻辑推理器
    
    实现基本的描述逻辑推理服务，包括：
    - 概念可满足性检查
    - 概念包含检查
    - 实例分类
    - 一致性检查
    """
    
    def __init__(self, hierarchy: ConceptHierarchy):
        self.hierarchy = hierarchy
        self.reasoning_cache: Dict[str, Any] = {}
    
    def check_subsumption(self, sub: str, sup: str) -> bool:
        """
        检查概念包含关系：sub ⊑ sup
        
        即sub概念是否是sup概念的子概念。
        """
        cache_key = f"subsumption:{sub}:{sup}"
        if cache_key in self.reasoning_cache:
            return self.reasoning_cache[cache_key]
        
        sub_concept = self.hierarchy.get_concept(sub)
        sup_concept = self.hierarchy.get_concept(sup)
        
        if not sub_concept or not sup_concept:
            return False
        
        # 直接检查
        result = sup in self.hierarchy.get_ancestors(sub) or sub == sup
        
        # 缓存结果
        self.reasoning_cache[cache_key] = result
        return result
    
    def check_equivalence(self, concept1: str, concept2: str) -> bool:
        """检查概念等价：C1 ≡ C2"""
        return (self.check_subsumption(concept1, concept2) and
                self.check_subsumption(concept2, concept1))
    
    def check_disjointness(self, concept1: str, concept2: str) -> bool:
        """检查概念是否不相交"""
        c1 = self.hierarchy.get_concept(concept1)
        c2 = self.hierarchy.get_concept(concept2)
        
        if not c1 or not c2:
            return False
        
        # 直接声明的不相交
        if concept2 in c1.disjoint_with or concept1 in c2.disjoint_with:
            return True
        
        # 继承的不相交
        for ancestor1 in self.hierarchy.get_ancestors(concept1) | {concept1}:
            anc1 = self.hierarchy.get_concept(ancestor1)
            if anc1 and concept2 in anc1.disjoint_with:
                return True
        
        return False
    
    def check_satisfiability(self, concept_name: str) -> bool:
        """
        检查概念是否可满足（非空）
        
        检查概念是否与Nothing不相交。
        """
        return not self.check_subsumption(concept_name, "Nothing")
    
    def classify_individual(self, individual: Individual) -> Set[str]:
        """
        对实例进行分类
        
        根据实例的属性和关系推断其类型。
        """
        inferred_types = set(individual.types)
        
        for concept_name, concept in self.hierarchy.concepts.items():
            if self._individual_satisfies_concept(individual, concept):
                inferred_types.add(concept_name)
        
        return inferred_types
    
    def _individual_satisfies_concept(self, 
                                     individual: Individual,
                                     concept: Concept) -> bool:
        """检查实例是否满足概念定义"""
        # 检查声明的类型
        if concept.name in individual.types:
            return True
        
        # 检查属性限制
        for restriction in concept.restrictions:
            if not self._check_restriction(individual, restriction):
                return False
        
        return True
    
    def _check_restriction(self, 
                          individual: Individual,
                          restriction: Dict[str, Any]) -> bool:
        """检查个体是否满足限制"""
        restriction_type = restriction.get('type')
        property_name = restriction.get('property')
        
        if restriction_type == 'some':
            # ∃R.C: 存在R关系指向C类型的个体
            target_type = restriction.get('target_type')
            if property_name in individual.relations:
                for target in individual.relations[property_name]:
                    target_ind = self.hierarchy.individuals.get(target)
                    if target_ind and target_type in target_ind.types:
                        return True
            return False
        
        elif restriction_type == 'only':
            # ∀R.C: 所有R关系都指向C类型的个体
            target_type = restriction.get('target_type')
            if property_name in individual.relations:
                for target in individual.relations[property_name]:
                    target_ind = self.hierarchy.individuals.get(target)
                    if target_ind and target_type not in target_ind.types:
                        return False
            return True
        
        elif restriction_type == 'min':
            # ≥n R: 至少有n个R关系
            min_count = restriction.get('count', 1)
            count = len(individual.relations.get(property_name, []))
            return count >= min_count
        
        elif restriction_type == 'max':
            # ≤n R: 最多有n个R关系
            max_count = restriction.get('count', 1)
            count = len(individual.relations.get(property_name, []))
            return count <= max_count
        
        elif restriction_type == 'exactly':
            # =n R: 恰好有n个R关系
            exact_count = restriction.get('count', 1)
            count = len(individual.relations.get(property_name, []))
            return count == exact_count
        
        elif restriction_type == 'value':
            # R value v: R关系的值等于v
            target_value = restriction.get('value')
            return target_value in individual.relations.get(property_name, [])
        
        return True
    
    def check_consistency(self) -> Tuple[bool, List[str]]:
        """
        检查本体的一致性
        
        Returns:
            (是否一致, 不一致原因列表)
        """
        inconsistencies = []
        
        # 检查概念不相交性
        for concept_name, concept in self.hierarchy.concepts.items():
            # 检查概念是否声明与自身不相交
            if concept_name in concept.disjoint_with:
                inconsistencies.append(
                    f"Concept {concept_name} is disjoint with itself"
                )
            
            # 检查不相交概念的公共子概念
            for disjoint in concept.disjoint_with:
                common_descendants = (
                    self.hierarchy.get_descendants(concept_name) &
                    self.hierarchy.get_descendants(disjoint)
                )
                if common_descendants:
                    inconsistencies.append(
                        f"Concepts {concept_name} and {disjoint} are disjoint "
                        f"but have common descendants: {common_descendants}"
                    )
        
        # 检查实例类型一致性
        for ind_name, individual in self.hierarchy.individuals.items():
            for type1 in individual.types:
                for type2 in individual.types:
                    if type1 != type2 and self.check_disjointness(type1, type2):
                        inconsistencies.append(
                            f"Individual {ind_name} has disjoint types "
                            f"{type1} and {type2}"
                        )
        
        return len(inconsistencies) == 0, inconsistencies
    
    def infer_properties(self, individual: Individual) -> Dict[str, Any]:
        """
        推断实例的隐含属性
        
        基于本体定义和实例已知信息推断隐含属性。
        """
        inferred = {}
        
        # 基于类型的属性继承
        for type_name in individual.types:
            concept = self.hierarchy.get_concept(type_name)
            if concept:
                # 继承注解属性
                for key, value in concept.annotations.items():
                    if key not in individual.annotations:
                        inferred[f"annotation_{key}"] = value
        
        # 基于角色的传递性推断
        for role_name, role in self.hierarchy.roles.items():
            if role.transitive and role_name in individual.relations:
                # 传递闭包
                all_targets = set()
                to_process = list(individual.relations[role_name])
                while to_process:
                    target = to_process.pop()
                    if target not in all_targets:
                        all_targets.add(target)
                        target_ind = self.hierarchy.individuals.get(target)
                        if target_ind and role_name in target_ind.relations:
                            to_process.extend(target_ind.relations[role_name])
                
                if all_targets > set(individual.relations.get(role_name, [])):
                    inferred[f"transitive_{role_name}"] = list(all_targets)
        
        # 基于逆角色的推断
        for role_name, role in self.hierarchy.roles.items():
            if role.inverse:
                for target in individual.relations.get(role_name, []):
                    target_ind = self.hierarchy.individuals.get(target)
                    if target_ind:
                        if individual.name not in target_ind.relations.get(role.inverse, []):
                            inferred.setdefault(f"inverse_{role.inverse}", []).append(target)
        
        return inferred


class NeuralOntologyReasoner(nn.Module):
    """
    神经本体推理器
    
    结合神经网络的本体推理模型，用于学习概念嵌入和
    进行模糊推理。
    """
    
    def __init__(self, hierarchy: ConceptHierarchy, embedding_dim: int = 128):
        super().__init__()
        self.hierarchy = hierarchy
        self.embedding_dim = embedding_dim
        
        num_concepts = len(hierarchy.concepts)
        num_roles = len(hierarchy.roles)
        
        # 概念嵌入
        self.concept_embeddings = nn.Embedding(num_concepts, embedding_dim)
        
        # 角色嵌入
        self.role_embeddings = nn.Embedding(num_roles, embedding_dim)
        
        # 推理网络
        self.subsumption_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 实例分类网络
        self.classification_net = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_concepts),
            nn.Sigmoid()
        )
        
        # 概念名称到索引的映射
        self.concept_to_idx = {name: i for i, name in enumerate(hierarchy.concepts.keys())}
        self.role_to_idx = {name: i for i, name in enumerate(hierarchy.roles.keys())}
    
    def compute_neural_subsumption(self, concept1: str, concept2: str) -> torch.Tensor:
        """
        使用神经网络计算概念包含概率
        
        作为符号推理的补充，处理模糊或不确定的情况。
        """
        idx1 = self.concept_to_idx.get(concept1, 0)
        idx2 = self.concept_to_idx.get(concept2, 0)
        
        emb1 = self.concept_embeddings(torch.tensor(idx1))
        emb2 = self.concept_embeddings(torch.tensor(idx2))
        
        combined = torch.cat([emb1, emb2])
        prob = self.subsumption_net(combined)
        
        return prob
    
    def classify_individual_neural(self, 
                                   individual_features: torch.Tensor) -> torch.Tensor:
        """
        使用神经网络对实例进行分类
        
        Args:
            individual_features: 实例特征向量
        
        Returns:
            每个概念的概率分布
        """
        return self.classification_net(individual_features)
    
    def forward(self, query_type: str, **kwargs) -> torch.Tensor:
        """
        神经推理前向传播
        
        Args:
            query_type: "subsumption", "classification", "similarity"
        """
        if query_type == "subsumption":
            return self.compute_neural_subsumption(
                kwargs['concept1'], kwargs['concept2']
            )
        elif query_type == "classification":
            return self.classify_individual_neural(kwargs['features'])
        elif query_type == "similarity":
            idx1 = self.concept_to_idx.get(kwargs['concept1'], 0)
            idx2 = self.concept_to_idx.get(kwargs['concept2'], 0)
            emb1 = self.concept_embeddings(torch.tensor(idx1))
            emb2 = self.concept_embeddings(torch.tensor(idx2))
            return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        else:
            raise ValueError(f"Unknown query type: {query_type}")


# ==================== 材料科学本体构建工具 ====================

def create_material_ontology() -> ConceptHierarchy:
    """
    创建材料科学领域本体
    
    构建一个包含材料分类、属性和关系的完整本体。
    """
    hierarchy = ConceptHierarchy()
    
    # 添加材料分类概念
    material_concepts = [
        # 按导电性分类
        Concept("Conductor", ConceptType.PRIMITIVE, 
                parents={"Material"},
                annotations={"conductivity": "high"}),
        Concept("Semiconductor", ConceptType.PRIMITIVE,
                parents={"Material"},
                annotations={"conductivity": "moderate"}),
        Concept("Insulator", ConceptType.PRIMITIVE,
                parents={"Material"},
                annotations={"conductivity": "low"}),
        
        # 按维度分类
        Concept("BulkMaterial", ConceptType.PRIMITIVE, parents={"Material"}),
        Concept("TwoDMaterial", ConceptType.PRIMITIVE, parents={"Material"}),
        Concept("OneDMaterial", ConceptType.PRIMITIVE, parents={"Material"}),
        Concept("ZeroDMaterial", ConceptType.PRIMITIVE, parents={"Material"}),
        
        # 具体材料
        Concept("Silicon", ConceptType.DEFINED,
                parents={"Semiconductor", "BulkMaterial"},
                annotations={"symbol": "Si", "atomic_number": 14}),
        Concept("Germanium", ConceptType.DEFINED,
                parents={"Semiconductor", "BulkMaterial"},
                annotations={"symbol": "Ge", "atomic_number": 32}),
        Concept("Graphene", ConceptType.DEFINED,
                parents={"Conductor", "TwoDMaterial"},
                annotations={"structure": "hexagonal"}),
        Concept("CNT", ConceptType.DEFINED,
                parents={"Conductor", "OneDMaterial"},
                annotations={"full_name": "Carbon Nanotube"}),
        
        # 晶体结构
        Concept("CrystallineMaterial", ConceptType.PRIMITIVE, parents={"Material"}),
        Concept("AmorphousMaterial", ConceptType.PRIMITIVE, parents={"Material"}),
        Concept("CubicStructure", ConceptType.PRIMITIVE, 
                parents={"CrystallineMaterial"}),
        Concept("HexagonalStructure", ConceptType.PRIMITIVE,
                parents={"CrystallineMaterial"}),
    ]
    
    # 不相交声明
    material_concepts[0].disjoint_with = {"Semiconductor", "Insulator"}
    material_concepts[1].disjoint_with = {"Conductor", "Insulator"}
    material_concepts[2].disjoint_with = {"Conductor", "Semiconductor"}
    
    for concept in material_concepts:
        hierarchy.add_concept(concept)
    
    # 添加角色
    roles = [
        Role("hasProperty", RoleType.OBJECT_PROPERTY,
             domain="Material", range="Property"),
        Role("hasStructure", RoleType.OBJECT_PROPERTY,
             domain="Material", range="CrystalStructure"),
        Role("isComposedOf", RoleType.OBJECT_PROPERTY,
             domain="Material", range="Element", transitive=True),
        Role("hasBandGap", RoleType.DATA_PROPERTY,
             domain="Material", range="float"),
        Role("hasConductivity", RoleType.DATA_PROPERTY,
             domain="Material", range="float"),
    ]
    
    for role in roles:
        hierarchy.add_role(role)
    
    return hierarchy


def perform_ontology_reasoning_example():
    """执行本体推理示例"""
    print("构建材料科学本体...")
    hierarchy = create_material_ontology()
    
    print(f"概念数量: {len(hierarchy.concepts)}")
    print(f"角色数量: {len(hierarchy.roles)}")
    
    # 创建推理器
    reasoner = DescriptionLogicReasoner(hierarchy)
    
    # 测试包含关系
    print("\n包含关系测试:")
    print(f"Silicon ⊑ Semiconductor: {reasoner.check_subsumption('Silicon', 'Semiconductor')}")
    print(f"Silicon ⊑ Conductor: {reasoner.check_subsumption('Silicon', 'Conductor')}")
    print(f"Semiconductor ⊑ Material: {reasoner.check_subsumption('Semiconductor', 'Material')}")
    
    # 测试不相交
    print("\n不相交测试:")
    print(f"Conductor disjoint with Insulator: {reasoner.check_disjointness('Conductor', 'Insulator')}")
    
    # 测试相似度
    print("\n概念相似度:")
    print(f"Sim(Silicon, Germanium): {hierarchy.compute_similarity('Silicon', 'Germanium'):.3f}")
    print(f"Sim(Silicon, Graphene): {hierarchy.compute_similarity('Silicon', 'Graphene'):.3f}")
    
    # 一致性检查
    print("\n一致性检查:")
    is_consistent, errors = reasoner.check_consistency()
    print(f"本体一致: {is_consistent}")
    if errors:
        print(f"不一致原因: {errors}")
    
    return hierarchy, reasoner


if __name__ == "__main__":
    print("=" * 60)
    print("本体推理模块测试")
    print("=" * 60)
    
    perform_ontology_reasoning_example()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
