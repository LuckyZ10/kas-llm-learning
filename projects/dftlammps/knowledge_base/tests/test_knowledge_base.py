"""
Tests for Knowledge Base Module
==============================
知识库模块的单元测试。
"""

import unittest
import json
import tempfile
import os
from datetime import datetime


class TestStorage(unittest.TestCase):
    """测试存储层"""
    
    def test_data_record(self):
        """测试数据记录"""
        from ..storage.base_storage import DataRecord
        
        record = DataRecord(
            id="test_123",
            data={"name": "Li3PS4", "band_gap": 2.5},
            tags=["battery", "solid_electrolyte"]
        )
        
        self.assertEqual(record.id, "test_123")
        self.assertEqual(record.data["name"], "Li3PS4")
        self.assertTrue(record.checksum)
        
        # 测试序列化
        record_dict = record.to_dict()
        self.assertIn("data", record_dict)
        
        # 测试反序列化
        record2 = DataRecord.from_dict(record_dict)
        self.assertEqual(record2.id, record.id)
    
    def test_query_filter(self):
        """测试查询过滤器"""
        from ..storage.base_storage import QueryFilter, create_filter
        
        f1 = QueryFilter("band_gap", "gt", 1.0)
        self.assertEqual(f1.field, "band_gap")
        self.assertEqual(f1.operator, "gt")
        
        f2 = create_filter("formula", "eq", "Li3PS4")
        self.assertEqual(f2.value, "Li3PS4")


class TestVersionControl(unittest.TestCase):
    """测试版本控制"""
    
    def test_version_creation(self):
        """测试版本创建"""
        from ..versioning.version_control import CalculationVersion, VersionStatus
        
        version = CalculationVersion(
            message="Initial calculation",
            author="test_user",
            data_snapshot={"energy": -100.5},
            calculation_type="DFT",
            parameters={"ecut": 500},
            results={"band_gap": 2.5}
        )
        
        self.assertTrue(version.id)
        self.assertEqual(version.status, VersionStatus.COMMITTED)
        self.assertTrue(version.data_hash)
    
    def test_diff(self):
        """测试版本比较"""
        from ..versioning.version_control import VersionComparator, CalculationVersion
        
        v1 = CalculationVersion(
            id="v1",
            data_snapshot={"energy": -100, "volume": 100}
        )
        
        v2 = CalculationVersion(
            id="v2",
            data_snapshot={"energy": -105, "volume": 100, "new_prop": 1.0}
        )
        
        diff = VersionComparator.compare(v1, v2)
        
        self.assertTrue(diff.has_changes())
        self.assertIn("energy", diff.modified)
        self.assertIn("new_prop", diff.added)
        self.assertIn("volume", diff.unchanged)
    
    def test_branch_manager(self):
        """测试分支管理"""
        from ..versioning.version_control import BranchManager
        
        bm = BranchManager()
        
        # 创建分支
        self.assertTrue(bm.create_branch("feature_branch", description="New feature"))
        
        # 列出分支
        branches = bm.list_branches()
        self.assertIn("main", branches)
        self.assertIn("feature_branch", branches)
        
        # 切换分支
        self.assertTrue(bm.switch_branch("feature_branch"))
        self.assertEqual(bm.get_current_branch(), "feature_branch")
    
    def test_version_control_system(self):
        """测试版本控制系统"""
        from ..versioning.version_control import VersionControl
        
        vc = VersionControl()
        
        # 提交版本
        v1 = vc.commit(
            data={"energy": -100},
            message="First calculation",
            author="user1"
        )
        
        v2 = vc.commit(
            data={"energy": -105},
            message="Second calculation",
            author="user1"
        )
        
        # 检查版本链
        self.assertEqual(v2.parent_id, v1.id)
        
        # 检出
        data = vc.checkout(v1.id)
        self.assertEqual(data["energy"], -100)
        
        # 日志
        log = vc.log(limit=10)
        self.assertEqual(len(log), 2)


class TestKnowledgeBuilder(unittest.TestCase):
    """测试知识构建器"""
    
    def test_entity_creation(self):
        """测试实体创建"""
        from ..knowledge_builder import KnowledgeBuilder, EntityType
        
        builder = KnowledgeBuilder()
        
        entity = builder.add_entity(
            entity_type="Material",
            name="Li3PS4",
            properties={"formula": "Li3PS4", "band_gap": 2.5}
        )
        
        self.assertEqual(entity["type"], "Material")
        self.assertEqual(entity["name"], "Li3PS4")
        self.assertIn("id", entity)
    
    def test_relation_creation(self):
        """测试关系创建"""
        from ..knowledge_builder import KnowledgeBuilder
        
        builder = KnowledgeBuilder()
        
        # 创建实体
        mat = builder.add_entity("Material", "Li3PS4")
        elem = builder.add_entity("Element", "Li")
        
        # 创建关系
        rel = builder.add_relation(
            relation_type="HAS_ELEMENT",
            from_entity=mat["id"],
            to_entity=elem["id"]
        )
        
        self.assertEqual(rel["type"], "HAS_ELEMENT")
        self.assertEqual(rel["from"], mat["id"])
    
    def test_calculation_to_knowledge(self):
        """测试从计算结果构建知识"""
        from ..knowledge_builder import KnowledgeBuilder
        
        builder = KnowledgeBuilder()
        
        calculation = {
            "material": {"formula": "Li3PS4", "structure_type": "orthorhombic"},
            "method": {"name": "VASP", "type": "DFT", "xc": "PBE"},
            "results": {"band_gap": 2.5, "formation_energy": -2.1}
        }
        
        subgraph = builder.build_from_calculation(calculation)
        
        self.assertTrue(len(subgraph["entities"]) > 0)
        self.assertTrue(len(subgraph["relations"]) > 0)
        
        # 验证实体类型
        entity_types = [e["type"] for e in subgraph["entities"]]
        self.assertIn("Material", entity_types)
        self.assertIn("Property", entity_types)
    
    def test_ontology(self):
        """测试本体"""
        from ..knowledge_builder import MaterialOntology
        
        onto = MaterialOntology()
        
        # 检查继承关系
        self.assertTrue(onto.is_a("Semiconductor", "Inorganic"))
        self.assertTrue(onto.is_a("Semiconductor", "Material"))
        self.assertFalse(onto.is_a("Semiconductor", "Organic"))
        
        # 获取祖先
        ancestors = onto.get_ancestors("Semiconductor")
        self.assertIn("Inorganic", ancestors)
        self.assertIn("Material", ancestors)


class TestVectorStore(unittest.TestCase):
    """测试向量存储"""
    
    def test_local_vector_store(self):
        """测试本地向量存储"""
        from ..embeddings.vector_store import LocalVectorStore, VectorConfig
        
        config = VectorConfig(provider="local", dimension=128)
        store = LocalVectorStore(config)
        
        store.connect()
        store.create_collection("test", 128)
        
        # 插入向量
        vectors = [[0.1] * 128, [0.2] * 128]
        ids = ["vec1", "vec2"]
        metadata = [{"content": "test1"}, {"content": "test2"}]
        
        self.assertTrue(store.upsert(vectors, ids, metadata))
        
        # 搜索
        results = store.search([0.1] * 128, top_k=2)
        self.assertEqual(len(results), 2)
        
        # 获取
        results = store.get(["vec1"])
        self.assertEqual(len(results), 1)


class TestSemanticSearch(unittest.TestCase):
    """测试语义搜索"""
    
    def test_search_query(self):
        """测试搜索查询"""
        from ..search.semantic_search import SearchQuery, SearchMode
        
        query = SearchQuery(
            query="lithium battery materials",
            mode=SearchMode.HYBRID,
            top_k=10,
            filters={"type": "material"}
        )
        
        self.assertEqual(query.query, "lithium battery materials")
        self.assertEqual(query.mode, SearchMode.HYBRID)


class TestKnowledgeAPI(unittest.TestCase):
    """测试知识库API"""
    
    def test_api_config(self):
        """测试API配置"""
        from ..knowledge_api import APIConfig
        
        config = APIConfig(
            mongodb_host="localhost",
            mongodb_port=27017,
            vector_provider="local"
        )
        
        self.assertEqual(config.mongodb_host, "localhost")
        self.assertEqual(config.vector_provider, "local")
    
    def test_config_from_dict(self):
        """测试从字典创建配置"""
        from ..knowledge_api import APIConfig
        
        config_dict = {
            "mongodb_host": "127.0.0.1",
            "mongodb_port": 27018,
            "vector_dimension": 768
        }
        
        config = APIConfig.from_dict(config_dict)
        self.assertEqual(config.mongodb_host, "127.0.0.1")
        self.assertEqual(config.vector_dimension, 768)


def run_tests():
    """运行所有测试"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()
