# Active Learning V2 Benchmark Report

## 实验设置

- **Dataset**: Synthetic regression problem
- **Samples**: 300
- **Features**: 10
- **Batch Size**: 5
- **Max Iterations**: 20
- **Random State**: 42

## 结果摘要

### 最终性能对比

| Strategy | Final Accuracy | Total Cost | AUC | Total Time (s) |
|----------|---------------|------------|-----|----------------|
| random | 0.8386 | 1150.00 | 65.9846 | 0.15 |
| bayesian_optimization | 0.7911 | 1150.00 | 72.5462 | 0.19 |
| dpp_diversity | 0.8693 | 1150.00 | 74.0223 | 0.22 |
| adaptive_hybrid | 0.8627 | 1150.00 | 73.1101 | 0.30 |

### 策略排名

#### 按最终准确率排名
1. dpp_diversity
1. adaptive_hybrid
1. random
1. bayesian_optimization

#### 按AUC排名
1. dpp_diversity
1. adaptive_hybrid
1. bayesian_optimization
1. random

#### 按成本效率排名
1. dpp_diversity
1. adaptive_hybrid
1. bayesian_optimization
1. random

## 结论


- **最佳准确率策略**: dpp_diversity
- **最佳效率策略**: dpp_diversity

- **自适应混合策略相对贝叶斯优化的提升**: 0.78%

## 图表

### 性能对比图
![Comparison Plots](comparison_plots.png)

### 雷达图
![Radar Chart](radar_chart.png)
