#!/usr/bin/env python3
"""
tests/test_optimizer.py
=======================
优化器测试
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from performance.optimizer import (
    Optimizer,
    JITCompiler,
    Vectorizer,
    OptimizationLevel,
    optimize,
    jit_compile
)


class TestOptimizer:
    """测试Optimizer类"""
    
    def setup_method(self):
        """每个测试前重置"""
        Optimizer._instance = None
    
    def test_singleton(self):
        """测试单例模式"""
        o1 = Optimizer()
        o2 = Optimizer()
        assert o1 is o2
    
    def test_optimize_decorator(self):
        """测试优化装饰器"""
        optimizer = Optimizer()
        
        @optimizer.optimize(level="basic")
        def test_func(x):
            return x ** 2
        
        result = test_func(5)
        assert result == 25
    
    def test_benchmark(self):
        """测试基准测试"""
        optimizer = Optimizer()
        
        def test_func(n):
            return sum(range(n))
        
        results = optimizer.benchmark(test_func, 1000)
        assert "original_time" in results
        assert "optimized_time" in results
    
    def test_optimization_levels(self):
        """测试不同优化级别"""
        levels = ["none", "basic", "aggressive"]
        
        for level in levels:
            optimizer = Optimizer()
            
            @optimizer.optimize(level=level)
            def func():
                return 42
            
            assert func() == 42


class TestJITCompiler:
    """测试JIT编译器"""
    
    def test_jit_decorator(self):
        """测试JIT装饰器"""
        compiler = JITCompiler()
        
        @compiler.jit
        def compute(x, y):
            return x ** 2 + y ** 2
        
        result = compute(3.0, 4.0)
        assert result == 25.0
    
    def test_vectorize(self):
        """测试向量化"""
        compiler = JITCompiler()
        
        @compiler.vectorize
        def square(x):
            return x ** 2
        
        arr = np.array([1.0, 2.0, 3.0])
        result = square(arr)
        expected = np.array([1.0, 4.0, 9.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_list_compiled(self):
        """测试列出编译函数"""
        compiler = JITCompiler()
        
        @compiler.jit
        def func1(x):
            return x + 1
        
        @compiler.jit
        def func2(x):
            return x * 2
        
        compiled = compiler.list_compiled()
        assert "func1" in compiled
        assert "func2" in compiled


class TestVectorizer:
    """测试向量化器"""
    
    def test_broadcast_arrays(self):
        """测试数组广播"""
        vectorizer = Vectorizer()
        
        a = np.array([[1], [2], [3]])
        b = np.array([1, 2, 3])
        
        result = vectorizer.broadcast_arrays(a, b)
        assert len(result) == 2
        assert result[0].shape == result[1].shape
    
    def test_einsum(self):
        """测试爱因斯坦求和"""
        vectorizer = Vectorizer()
        
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        
        # 矩阵乘法
        result = vectorizer.einsum('ij,jk->ik', a, b)
        expected = np.dot(a, b)
        np.testing.assert_array_almost_equal(result, expected)


class TestConvenienceFunctions:
    """测试便捷函数"""
    
    def test_optimize_decorator(self):
        """测试optimize装饰器"""
        
        @optimize(level="basic")
        def test_func(x):
            return x * 2
        
        assert test_func(5) == 10
    
    def test_jit_compile_decorator(self):
        """测试jit_compile装饰器"""
        
        @jit_compile
        def compute(x):
            return x ** 2 + 1
        
        assert compute(3) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
