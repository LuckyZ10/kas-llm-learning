"""
感知机运行示例
================

本脚本演示感知机在各种场景下的应用
"""

import numpy as np
import matplotlib.pyplot as plt
from perceptron import (
    Perceptron, PerceptronDual,
    plot_decision_boundary, plot_learning_curve,
    plot_xor_problem, plot_perceptron_architecture,
    test_logic_gates, demo_and_gate, demo_xor_failure
)


def example_1_basic_usage():
    """
    示例1: 感知机基本用法
    """
    print("\n" + "="*60)
    print("示例1: 感知机基本用法")
    print("="*60)
    
    # 创建简单的二分类数据
    X = np.array([
        [2, 3],   # 类别 +1
        [1, 1],   # 类别 +1
        [-1, -1], # 类别 -1
        [-2, -3]  # 类别 -1
    ])
    y = np.array([1, 1, -1, -1])
    
    # 创建并训练感知机
    p = Perceptron(learning_rate=0.1, n_iterations=100)
    p.fit(X, y)
    
    # 预测
    predictions = p.predict(X)
    print(f"预测结果: {predictions}")
    print(f"真实标签: {y}")
    print(f"准确率: {np.mean(predictions == y) * 100}%")
    
    # 预测新样本
    new_samples = np.array([[3, 3], [-3, -2]])
    new_predictions = p.predict(new_samples)
    print(f"\n新样本 {new_samples[0]} 预测为: {new_predictions[0]}")
    print(f"新样本 {new_samples[1]} 预测为: {new_predictions[1]}")


def example_2_and_gate():
    """
    示例2: AND门实现
    """
    print("\n" + "="*60)
    print("示例2: AND门实现")
    print("="*60)
    
    # AND真值表
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([-1, -1, -1, 1])  # 0, 0, 0, 1
    
    p = Perceptron(learning_rate=0.1, n_iterations=100)
    p.fit(X, y)
    
    print("\nAND门真值表:")
    print("x1  x2  |  预测  |  真实")
    print("-" * 25)
    for xi, yi in zip(X, y):
        pred = p.predict_single(xi)
        print(f" {xi[0]}   {xi[1]}  |   {pred if pred == 1 else 0}   |   {yi if yi == 1 else 0}")
    
    print(f"\n决策边界方程: {p.weights_[0]:.3f}*x1 + {p.weights_[1]:.3f}*x2 + {p.bias_:.3f} = 0")


def example_3_xor_limitation():
    """
    示例3: XOR问题 - 展示感知机的局限性
    """
    print("\n" + "="*60)
    print("示例3: XOR问题（感知机无法解决）")
    print("="*60)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([-1, 1, 1, -1])  # XOR
    
    p = Perceptron(learning_rate=0.1, n_iterations=100)
    p.fit(X, y)
    
    print("\nXOR真值表:")
    print("x1  x2  |  预测  |  真实  |  正确?")
    print("-" * 35)
    for xi, yi in zip(X, y):
        pred = p.predict_single(xi)
        correct = "✓" if pred == yi else "✗"
        print(f" {xi[0]}   {xi[1]}  |   {pred if pred == 1 else 0}   |   {yi if yi == 1 else 0}   |   {correct}")
    
    print("\n说明: XOR问题不是线性可分的，单层感知机无法正确分类")


def example_4_2d_visualization():
    """
    示例4: 二维数据可视化
    """
    print("\n" + "="*60)
    print("示例4: 二维数据可视化")
    print("="*60)
    
    # 生成线性可分的二维数据
    np.random.seed(42)
    
    # 类别 +1: 围绕 (2, 2) 的点
    X_pos = np.random.randn(50, 2) + np.array([2, 2])
    
    # 类别 -1: 围绕 (-2, -2) 的点
    X_neg = np.random.randn(50, 2) + np.array([-2, -2])
    
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 50 + [-1] * 50)
    
    # 训练
    p = Perceptron(learning_rate=0.01, n_iterations=200)
    p.fit(X, y)
    
    # 可视化
    plot_decision_boundary(X, y, p, title="线性可分数据 - 感知机分类")
    plot_learning_curve(p, title="学习曲线")


def example_5_dual_form():
    """
    示例5: 对偶形式
    """
    print("\n" + "="*60)
    print("示例5: 对偶形式感知机")
    print("="*60)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([-1, -1, -1, 1])  # AND
    
    p_dual = PerceptronDual(learning_rate=0.1, n_iterations=100)
    p_dual.fit(X, y)
    
    print(f"对偶变量 alpha: {p_dual.alpha_}")
    print(f"偏置 b: {p_dual.b_:.4f}")
    print(f"转换后的权重 w: [{p_dual.weights_[0]:.4f}, {p_dual.weights_[1]:.4f}]")
    
    predictions = p_dual.predict(X)
    print(f"预测结果: {predictions}")


def example_6_convergence_analysis():
    """
    示例6: 收敛性分析
    """
    print("\n" + "="*60)
    print("示例6: 收敛性分析")
    print("="*60)
    
    # 测试不同难度的数据集（不同的间隔）
    difficulties = [
        ("简单 (大间隔)", 3.0),
        ("中等 (中间隔)", 1.5),
        ("困难 (小间隔)", 0.5)
    ]
    
    for name, margin in difficulties:
        np.random.seed(42)
        
        # 生成数据，控制间隔
        X_pos = np.random.randn(20, 2) * 0.3 + np.array([margin, margin])
        X_neg = np.random.randn(20, 2) * 0.3 + np.array([-margin, -margin])
        
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * 20 + [-1] * 20)
        
        p = Perceptron(learning_rate=0.01, n_iterations=500)
        p.fit(X, y)
        
        print(f"\n{name}:")
        print(f"  收敛所需迭代: {len(p.errors_)}")
        print(f"  总更新次数: {p.n_updates_}")


if __name__ == "__main__":
    import os
    os.makedirs('output', exist_ok=True)
    
    print("\n" + "="*70)
    print("感知机算法 - 运行示例")
    print("="*70)
    
    # 运行所有示例
    example_1_basic_usage()
    example_2_and_gate()
    example_3_xor_limitation()
    example_4_2d_visualization()
    example_5_dual_form()
    example_6_convergence_analysis()
    
    print("\n" + "="*70)
    print("所有示例运行完成！")
    print("="*70)
