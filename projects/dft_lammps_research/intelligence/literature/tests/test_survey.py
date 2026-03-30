"""
测试套件
"""

import unittest
from datetime import datetime
from pathlib import Path
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from literature_survey.config.models import Paper, Author
from literature_survey.config.database import DatabaseManager


class TestModels(unittest.TestCase):
    """测试数据模型"""
    
    def test_paper_creation(self):
        """测试论文创建"""
        paper = Paper(
            id="test:123",
            title="Test Paper",
            authors=[Author(name="John Doe")],
            abstract="This is a test abstract.",
            publication_date=datetime.now()
        )
        
        self.assertEqual(paper.id, "test:123")
        self.assertEqual(paper.title, "Test Paper")
        self.assertEqual(len(paper.authors), 1)
    
    def test_author_names(self):
        """测试作者名字获取"""
        paper = Paper(
            id="test:123",
            title="Test",
            authors=[
                Author(name="Alice Smith"),
                Author(name="Bob Jones")
            ],
            abstract="Test",
            publication_date=datetime.now()
        )
        
        self.assertEqual(paper.get_author_names(), "Alice Smith, Bob Jones")


class TestDatabase(unittest.TestCase):
    """测试数据库"""
    
    def setUp(self):
        """设置测试数据库"""
        self.db = DatabaseManager(db_path=Path("/tmp/test_literature.db"))
    
    def test_save_and_get_paper(self):
        """测试保存和获取论文"""
        paper = Paper(
            id="test:456",
            title="Database Test Paper",
            authors=[Author(name="Test Author")],
            abstract="Test abstract for database",
            publication_date=datetime.now(),
            source="test"
        )
        
        # 保存
        result = self.db.save_paper(paper)
        self.assertTrue(result)
        
        # 获取
        retrieved = self.db.get_paper("test:456")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.title, "Database Test Paper")
    
    def test_search_papers(self):
        """测试搜索论文"""
        # 搜索应该返回列表
        results = self.db.search_papers(query="test", limit=10)
        self.assertIsInstance(results, list)


class TestFetchers(unittest.TestCase):
    """测试文献抓取器"""
    
    def test_arxiv_fetcher_initialization(self):
        """测试arXiv抓取器初始化"""
        from literature_survey.fetcher.arxiv_fetcher import ArxivFetcher
        
        fetcher = ArxivFetcher()
        self.assertIsNotNone(fetcher.base_url)
    
    def test_query_building(self):
        """测试查询构建"""
        from literature_survey.fetcher import LiteratureFetcher
        
        fetcher = LiteratureFetcher()
        # 这里可以测试查询构建逻辑
        self.assertIsNotNone(fetcher)


class TestAnalysis(unittest.TestCase):
    """测试分析模块"""
    
    def test_topic_modeler(self):
        """测试主题建模器"""
        from literature_survey.analysis.topic_modeling import TopicModeler
        
        modeler = TopicModeler(method="simple", n_topics=5)
        self.assertEqual(modeler.n_topics, 5)
    
    def test_trend_analyzer(self):
        """测试趋势分析器"""
        from literature_survey.analysis.trend_analysis import TrendAnalyzer
        
        analyzer = TrendAnalyzer(window_size=2)
        self.assertEqual(analyzer.window_size, 2)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestModels))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabase))
    suite.addTests(loader.loadTestsFromTestCase(TestFetchers))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalysis))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
