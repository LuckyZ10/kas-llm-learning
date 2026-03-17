"""
KAS Tests - 扩展功能测试
测试 stats, market, versioning 模块
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta

from kas.core.stats import AnalyticsDatabase, StatsDashboard, DailyStats
from kas.core.market import PackagePacker, LocalMarket, PackageInfo
from kas.core.versioning import VersionManager, VersionInfo
from kas.core.models import Agent, Capability, CapabilityType


class TestAnalyticsDatabase(unittest.TestCase):
    """测试统计数据库"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        # 重置单例
        import kas.core.stats as stats_module
        stats_module.AnalyticsDatabase._instance = None
        stats_module.AnalyticsDatabase._conn = None
        stats_module.STATS_DIR = Path(self.temp_dir) / 'stats'
        stats_module.STATS_DB = stats_module.STATS_DIR / 'analytics.db'
        self.db = AnalyticsDatabase()
    
    def tearDown(self):
        """清理"""
        # 关闭连接
        if self.db._conn:
            self.db._conn.close()
        # 重置单例
        import kas.core.stats as stats_module
        stats_module.AnalyticsDatabase._instance = None
        stats_module.AnalyticsDatabase._conn = None
        shutil.rmtree(self.temp_dir)
    
    def test_record_conversation(self):
        """测试记录对话"""
        from kas.core.stats import ConversationMetrics
        
        metrics = ConversationMetrics(
            conversation_id="test_001",
            agent_name="TestAgent",
            timestamp=datetime.now().isoformat(),
            message_count=10,
            response_time_avg=1.5,
            token_input=100,
            token_output=200,
            user_rating=4
        )
        
        self.db.record_conversation(metrics)
        
        # 验证
        stats = self.db.get_agent_stats("TestAgent", days=1)
        self.assertEqual(stats['conversations']['total_conversations'], 1)
        self.assertEqual(stats['conversations']['avg_messages'], 10)
    
    def test_get_daily_stats(self):
        """测试获取每日统计"""
        from kas.core.stats import ConversationMetrics
        
        # 添加测试数据
        for i in range(3):
            metrics = ConversationMetrics(
                conversation_id=f"test_{i}",
                agent_name="TestAgent",
                timestamp=datetime.now().isoformat(),
                message_count=5,
                response_time_avg=1.0,
                token_input=50,
                token_output=100,
                user_rating=5
            )
            self.db.record_conversation(metrics)
        
        daily_stats = self.db.get_daily_stats(days=7)
        self.assertEqual(len(daily_stats), 7)
        self.assertGreaterEqual(daily_stats[-1].total_conversations, 3)


class TestVersionManager(unittest.TestCase):
    """测试版本管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        import kas.core.versioning as versioning_module
        versioning_module.VERSIONS_DIR = Path(self.temp_dir) / 'versions'
        self.vm = VersionManager("TestAgent")
        
        # 创建测试 Agent
        self.agent = Agent(
            name="TestAgent",
            capabilities=[
                Capability("Code Review", CapabilityType.CODE_REVIEW, "审查代码", 0.9)
            ],
            system_prompt="你是一个测试 Agent"
        )
    
    def tearDown(self):
        """清理"""
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_version(self):
        """测试保存和加载版本"""
        # 保存版本
        version_id = self.vm.save_version(
            self.agent,
            description="初始版本",
            tags=["stable"],
            quality_score=80.0
        )
        
        self.assertIsNotNone(version_id)
        
        # 加载版本
        loaded_agent = self.vm.load_version(version_id)
        self.assertIsNotNone(loaded_agent)
        self.assertEqual(loaded_agent.name, "TestAgent")
    
    def test_list_versions(self):
        """测试列出版本"""
        # 保存多个版本
        for i in range(3):
            self.vm.save_version(
                self.agent,
                description=f"版本 {i}",
                quality_score=70.0 + i * 10
            )
        
        versions = self.vm.list_versions()
        self.assertEqual(len(versions), 3)
    
    def test_delete_version_with_children(self):
        """测试删除有子版本的版本"""
        # 保存父版本
        v1 = self.vm.save_version(self.agent, description="父版本", quality_score=80.0)
        
        # 保存子版本
        v2 = self.vm.save_version(self.agent, description="子版本", quality_score=85.0)
        
        # 尝试删除父版本（不带force）
        success, msg = self.vm.delete_version(v1)
        self.assertFalse(success)
        self.assertIn("子版本", msg)
        
        # 使用force删除
        success, msg = self.vm.delete_version(v1, force=True)
        self.assertTrue(success)


class TestPackagePacker(unittest.TestCase):
    """测试包打包器"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        import kas.core.market as market_module
        market_module.MARKET_DIR = Path(self.temp_dir) / 'market'
        
        self.agent = Agent(
            name="TestPackageAgent",
            version="1.0.0",
            description="测试包",
            capabilities=[
                Capability("Test", CapabilityType.CODE_REVIEW, "测试能力", 0.9)
            ],
            system_prompt="你是一个测试 Agent"
        )
    
    def tearDown(self):
        """清理"""
        shutil.rmtree(self.temp_dir)
    
    def test_pack_and_unpack(self):
        """测试打包和解包"""
        packer = PackagePacker()
        
        # 打包
        package_path = packer.pack(
            self.agent,
            output_path=f"{self.temp_dir}/test.kas-agent",
            author="TestAuthor",
            tags=["test"]
        )
        
        self.assertTrue(Path(package_path).exists())
        
        # 解包
        unpacked_agent = packer.unpack(package_path)
        self.assertIsNotNone(unpacked_agent)
        self.assertEqual(unpacked_agent.name, "TestPackageAgent")


class TestLocalMarket(unittest.TestCase):
    """测试本地市场"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        import kas.core.market as market_module
        market_module.MARKET_DIR = Path(self.temp_dir) / 'market'
        market_module.MARKET_INDEX = market_module.MARKET_DIR / 'index.json'
        
        self.market = LocalMarket()
        self.packer = PackagePacker()
        
        # 使用唯一名称避免冲突
        import uuid
        self.unique_name = f"MarketTestAgent_{uuid.uuid4().hex[:8]}"
        self.agent = Agent(
            name=self.unique_name,
            version="1.0.0",
            capabilities=[
                Capability("Test", CapabilityType.CODE_REVIEW, "测试", 0.9)
            ],
            system_prompt="测试 Agent"
        )
    
    def tearDown(self):
        """清理"""
        shutil.rmtree(self.temp_dir)
    
    def test_publish_and_search(self):
        """测试发布和搜索"""
        # 打包
        package_path = self.packer.pack(
            self.agent,
            output_path=f"{self.temp_dir}/test.kas-agent"
        )
        
        # 发布
        result = self.market.publish(package_path)
        self.assertTrue(result)
        
        # 搜索
        results = self.market.search(self.unique_name[:10])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, self.unique_name)
    
    def test_publish_duplicate_without_force(self):
        """测试重复发布不带force"""
        # 首次发布
        package_path = self.packer.pack(
            self.agent,
            output_path=f"{self.temp_dir}/test.kas-agent"
        )
        self.market.publish(package_path)
        
        # 重复发布（不带force）
        result = self.market.publish(package_path, force=False)
        self.assertFalse(result)
    
    def test_publish_duplicate_with_force(self):
        """测试重复发布带force"""
        # 首次发布
        package_path = self.packer.pack(
            self.agent,
            output_path=f"{self.temp_dir}/test.kas-agent"
        )
        self.market.publish(package_path)
        
        # 模拟下载
        info = self.market.get_info(self.unique_name)
        info.downloads = 10
        
        # 更新版本并强制发布
        self.agent.version = "1.1.0"
        package_path = self.packer.pack(
            self.agent,
            output_path=f"{self.temp_dir}/test2.kas-agent"
        )
        result = self.market.publish(package_path, force=True)
        self.assertTrue(result)
        
        # 验证下载计数保留
        info = self.market.get_info(self.unique_name)
        self.assertEqual(info.downloads, 10)


if __name__ == '__main__':
    unittest.main()
