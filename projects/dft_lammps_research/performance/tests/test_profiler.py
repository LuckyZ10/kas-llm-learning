#!/usr/bin/env python3
"""
tests/test_profiler.py
======================
性能分析器测试
"""

import time
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from performance.profiler import (
    Profiler, 
    profile_function, 
    PerformanceContext,
    FunctionStats,
    BottleneckAnalyzer
)


class TestProfiler:
    """测试Profiler类"""
    
    def setup_method(self):
        """每个测试前重置"""
        Profiler._instance = None
    
    def test_singleton(self):
        """测试单例模式"""
        p1 = Profiler()
        p2 = Profiler()
        assert p1 is p2
    
    def test_profile_decorator(self):
        """测试分析装饰器"""
        profiler = Profiler()
        
        @profiler.profile
        def test_func():
            time.sleep(0.01)
            return 42
        
        result = test_func()
        assert result == 42
        
        stats = profiler.get_stats()
        assert len(stats) > 0
    
    def test_measure_context(self):
        """测试上下文管理器"""
        profiler = Profiler()
        
        with profiler.measure("test_block"):
            time.sleep(0.01)
        
        stats = profiler.get_stats()
        assert any("test_block" in k for k in stats.keys())
    
    def test_get_stats(self):
        """测试获取统计"""
        profiler = Profiler()
        
        @profiler.profile
        def func_a():
            time.sleep(0.001)
        
        func_a()
        func_a()
        
        stats = profiler.get_stats()
        assert len(stats) >= 1
    
    def test_get_slowest_functions(self):
        """测试获取最慢函数"""
        profiler = Profiler()
        
        @profiler.profile
        def slow_func():
            time.sleep(0.05)
        
        @profiler.profile
        def fast_func():
            pass
        
        slow_func()
        fast_func()
        
        slowest = profiler.get_slowest_functions()
        assert len(slowest) >= 1
    
    def test_generate_report(self):
        """测试生成报告"""
        profiler = Profiler()
        
        @profiler.profile
        def test_func():
            time.sleep(0.01)
        
        test_func()
        
        report = profiler.generate_report()
        assert "PERFORMANCE PROFILE REPORT" in report
    
    def test_export_json(self, tmp_path):
        """测试导出JSON"""
        profiler = Profiler()
        
        @profiler.profile
        def test_func():
            time.sleep(0.01)
        
        test_func()
        
        output_path = tmp_path / "stats.json"
        profiler.export_json(output_path)
        assert output_path.exists()


class TestFunctionStats:
    """测试FunctionStats类"""
    
    def test_basic_stats(self):
        """测试基本统计"""
        stat = FunctionStats(name="test", module="test_module")
        stat.call_count = 10
        stat.total_time = 1.0
        
        assert stat.avg_time == 0.1
        assert stat.to_dict()["name"] == "test"
    
    def test_percentiles(self):
        """测试百分位数"""
        stat = FunctionStats(name="test", module="test_module")
        
        for t in [0.01, 0.02, 0.03, 0.04, 0.05]:
            stat.times.append(t)
        
        assert stat.p50_time > 0
        assert stat.p95_time >= stat.p50_time


class TestBottleneckAnalyzer:
    """测试瓶颈分析器"""
    
    def test_analyze(self):
        """测试分析功能"""
        profiler = Profiler()
        analyzer = BottleneckAnalyzer(profiler)
        
        @profiler.profile
        def heavy_func():
            time.sleep(0.05)
        
        heavy_func()
        
        result = analyzer.analyze()
        assert "bottlenecks" in result


class TestDecorators:
    """测试便捷装饰器"""
    
    def test_profile_function(self):
        """测试profile_function装饰器"""
        
        @profile_function
        def test_func():
            time.sleep(0.01)
            return "done"
        
        result = test_func()
        assert result == "done"
    
    def test_performance_context(self):
        """测试PerformanceContext"""
        
        with PerformanceContext("my_block"):
            time.sleep(0.01)
        
        # 如果运行到这里没有异常，测试通过
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
