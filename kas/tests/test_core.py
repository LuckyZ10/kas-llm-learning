"""
KAS Tests - 核心功能测试
"""

import unittest
import tempfile
import shutil
from pathlib import Path

from kas.core.models import Agent, Capability, CapabilityType
from kas.core.fusion import FusionEngine, FusionStrategy
from kas.core.ingestion import IngestionEngine


class TestAgentFusion(unittest.TestCase):
    """测试 Agent 合体功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.fusion = FusionEngine()
        
        # 创建测试 Agent
        self.agent1 = Agent(
            name="Agent1",
            capabilities=[
                Capability("Code Review", CapabilityType.CODE_REVIEW, "审查代码", 0.9),
                Capability("Debug", CapabilityType.DEBUGGING, "调试", 0.8)
            ],
            system_prompt="You are Agent1"
        )
        
        self.agent2 = Agent(
            name="Agent2",
            capabilities=[
                Capability("Code Review", CapabilityType.CODE_REVIEW, "审查代码", 0.85),
                Capability("Docs", CapabilityType.DOCUMENTATION, "文档", 0.75)
            ],
            system_prompt="You are Agent2"
        )
    
    def test_union_strategy(self):
        """测试并集策略"""
        result = self.fusion.fuse([self.agent1, self.agent2], FusionStrategy.UNION)
        
        # 应该包含所有不重复的能力
        cap_types = [c.type for c in result.capabilities]
        self.assertIn(CapabilityType.CODE_REVIEW, cap_types)
        self.assertIn(CapabilityType.DEBUGGING, cap_types)
        self.assertIn(CapabilityType.DOCUMENTATION, cap_types)
        
        # 应该有3个能力（CODE_REVIEW去重后算1个）
        self.assertEqual(len(result.capabilities), 3)
    
    def test_intersection_strategy(self):
        """测试交集策略"""
        result = self.fusion.fuse([self.agent1, self.agent2], FusionStrategy.INTERSECTION)
        
        # 只应该包含共同的能力（CODE_REVIEW）
        cap_types = [c.type for c in result.capabilities]
        self.assertIn(CapabilityType.CODE_REVIEW, cap_types)
        self.assertNotIn(CapabilityType.DEBUGGING, cap_types)
        self.assertNotIn(CapabilityType.DOCUMENTATION, cap_types)
        
        # 应该只有1个能力
        self.assertEqual(len(result.capabilities), 1)
    
    def test_dominant_strategy(self):
        """测试主导策略"""
        result = self.fusion.fuse([self.agent1, self.agent2], FusionStrategy.DOMINANT)
        
        # 应该包含 agent1 的所有能力 + agent2 的独特能力
        cap_types = [c.type for c in result.capabilities]
        self.assertIn(CapabilityType.CODE_REVIEW, cap_types)
        self.assertIn(CapabilityType.DEBUGGING, cap_types)
        self.assertIn(CapabilityType.DOCUMENTATION, cap_types)
    
    def test_intersection_with_no_common(self):
        """测试没有共同能力的交集"""
        agent3 = Agent(
            name="Agent3",
            capabilities=[
                Capability("Test", CapabilityType.TEST_GENERATION, "测试", 0.9)
            ],
            system_prompt="You are Agent3"
        )
        
        result = self.fusion.fuse([self.agent1, agent3], FusionStrategy.INTERSECTION)
        
        # 应该没有共同能力
        self.assertEqual(len(result.capabilities), 0)


class TestIngestion(unittest.TestCase):
    """测试代码吞食功能"""
    
    def setUp(self):
        """创建临时项目目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir) / "test_project"
        self.project_path.mkdir()
        
        # 创建测试文件
        (self.project_path / "main.py").write_text("""
def hello():
    print("Hello World")

class MyClass:
    def method(self):
        try:
            result = 1 / 0
        except ZeroDivisionError:
            pass
""")
        
        (self.project_path / "README.md").write_text("# Test Project")
        
        self.engine = IngestionEngine()
    
    def tearDown(self):
        """清理临时目录"""
        shutil.rmtree(self.temp_dir)
    
    def test_analyze_structure(self):
        """测试项目结构分析"""
        info = self.engine._analyze_structure(self.project_path)
        
        self.assertEqual(info['file_types']['.py'], 1)
        self.assertEqual(info['file_types']['.md'], 1)
        self.assertTrue(info['has_docs'])
        self.assertEqual(info['total_files'], 2)
    
    def test_extract_code_samples(self):
        """测试代码样本提取"""
        samples = self.engine._extract_code_samples(self.project_path)
        
        self.assertGreater(len(samples), 0)
        self.assertIn("main.py", samples[0])
    
    def test_detect_capabilities_rule_based(self):
        """测试基于规则的能力检测"""
        project_info = {
            'file_types': {'.py': 2},
            'has_tests': False,
            'has_docs': True,
            'has_config': False
        }
        code_samples = ["def hello(): pass", "class MyClass: pass"]
        
        capabilities = self.engine._detect_capabilities_rule_based(project_info, code_samples)
        
        cap_types = [c.type for c in capabilities]
        self.assertIn(CapabilityType.CODE_REVIEW, cap_types)
        self.assertIn(CapabilityType.DOCUMENTATION, cap_types)
    
    def test_ingest(self):
        """测试完整吞食流程"""
        agent = self.engine.ingest(str(self.project_path), "TestAgent")
        
        self.assertEqual(agent.name, "TestAgent")
        self.assertGreater(len(agent.capabilities), 0)
        self.assertIsNotNone(agent.system_prompt)


class TestCapabilityModel(unittest.TestCase):
    """测试 Capability 模型"""
    
    def test_capability_creation(self):
        """测试创建 Capability"""
        cap = Capability(
            name="Test",
            type=CapabilityType.CODE_REVIEW,
            description="Test capability",
            confidence=0.8
        )
        
        self.assertEqual(cap.name, "Test")
        self.assertEqual(cap.type, CapabilityType.CODE_REVIEW)
        self.assertEqual(cap.confidence, 0.8)
    
    def test_agent_to_dict(self):
        """测试 Agent 序列化"""
        agent = Agent(
            name="TestAgent",
            capabilities=[
                Capability("Code Review", CapabilityType.CODE_REVIEW, "审查代码", 0.9)
            ],
            system_prompt="You are a test agent"
        )
        
        data = agent.to_dict()
        
        self.assertEqual(data['name'], "TestAgent")
        self.assertEqual(len(data['capabilities']), 1)
        self.assertEqual(data['system_prompt'], "You are a test agent")
    
    def test_agent_from_dict(self):
        """测试 Agent 反序列化"""
        data = {
            'name': 'TestAgent',
            'version': '1.0.0',
            'description': 'Test',
            'capabilities': [
                {
                    'name': 'Code Review',
                    'type': 'code_review',
                    'description': '审查代码',
                    'confidence': 0.9
                }
            ],
            'system_prompt': 'You are a test agent',
            'model_config': {},
            'created_from': '/test'
        }
        
        agent = Agent.from_dict(data)
        
        self.assertEqual(agent.name, "TestAgent")
        self.assertEqual(len(agent.capabilities), 1)
        self.assertEqual(agent.capabilities[0].type, CapabilityType.CODE_REVIEW)


if __name__ == '__main__':
    unittest.main()
