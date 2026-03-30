"""
Process Parameter Optimization Example
======================================

演示如何使用RL进行工艺参数优化
与贝叶斯优化进行对比
"""

import numpy as np
import matplotlib.pyplot as plt
from dftlammps.rl_optimization.environments.process_env import (
    ProcessEnvConfig, SynthesisEnv, ParameterEnv, BayesianOptimizer
)
from dftlammps.rl_optimization.training.process_trainer import (
    ProcessOptimizationTrainer, ProcessTrainingConfig,
    compare_optimizers, MultiFidelityOptimizer
)


def synthesis_optimization_example():
    """合成工艺优化示例"""
    print("=" * 70)
    print("Synthesis Process Optimization Example")
    print("=" * 70)
    
    # 创建环境
    config = ProcessEnvConfig(
        param_bounds={
            'temperature': (300.0, 1200.0),
            'pressure': (0.1, 5.0),
            'time': (1.0, 24.0),
            'concentration': (0.05, 0.5),
        },
        discrete_params={
            'method': ['sol-gel', 'hydrothermal', 'solid-state'],
            'atmosphere': ['air', 'nitrogen', 'argon'],
        },
        max_steps=30,
        noise_level=0.02
    )
    
    env = SynthesisEnv(
        target_property='ion_conductivity',
        target_value=0.5,
        config=config
    )
    
    print(f"\nEnvironment created:")
    print(f"  Target: {env.target_property} = {env.target_value}")
    print(f"  Continuous params: {list(config.param_bounds.keys())}")
    print(f"  Discrete params: {list(config.discrete_params.keys())}")
    
    # 贝叶斯优化
    print("\n" + "-" * 50)
    print("Bayesian Optimization")
    print("-" * 50)
    
    bo = BayesianOptimizer(
        bounds=[(0, 1)] * env.action_dim,
        acquisition='ei'
    )
    
    bo_rewards = []
    state = env.reset()
    
    for step in range(30):
        # 归一化动作
        action_normalized = bo.suggest()
        action = action_normalized * 2 - 1  # 转换到[-1, 1]
        
        state, reward, done, info = env.step(action)
        bo.observe(action_normalized, reward)
        bo_rewards.append(reward)
        
        if done:
            break
    
    print(f"Steps: {len(bo_rewards)}")
    print(f"Best reward: {max(bo_rewards):.3f}")
    print(f"Final reward: {bo_rewards[-1]:.3f}")
    
    # 随机搜索对比
    print("\n" + "-" * 50)
    print("Random Search (Baseline)")
    print("-" * 50)
    
    random_rewards = []
    state = env.reset()
    
    for step in range(30):
        action = np.random.randn(env.action_dim) * 0.5
        state, reward, done, info = env.step(action)
        random_rewards.append(reward)
        
        if done:
            break
    
    print(f"Best reward: {max(random_rewards):.3f}")
    print(f"Final reward: {random_rewards[-1]:.3f}")
    
    return bo_rewards, random_rewards


def parameter_optimization_example():
    """通用参数优化示例"""
    print("\n" + "=" * 70)
    print("General Parameter Optimization Example")
    print("=" * 70)
    
    # 定义目标函数 (Ackley function)
    def ackley(x):
        a, b, c = 20, 0.2, 2 * np.pi
        d = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        return -(-a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e)
    
    print(f"\nTarget function: Ackley function (3D)")
    print(f"Global minimum at x = [0, 0, 0]")
    
    # 比较优化器
    print("\nComparing optimizers...")
    results = compare_optimizers(
        ackley,
        [(-5, 5)] * 3,
        budget=50,
        num_runs=3
    )
    
    # 计算统计
    for method, histories in results.items():
        final_values = [h[-1] for h in histories]
        print(f"\n{method}:")
        print(f"  Mean final value: {np.mean(final_values):.3f} ± {np.std(final_values):.3f}")
        print(f"  Best: {max(final_values):.3f}")
    
    return results


def multifidelity_optimization_example():
    """多保真度优化示例"""
    print("\n" + "=" * 70)
    print("Multi-Fidelity Optimization Example")
    print("=" * 70)
    
    # 定义不同保真度的目标函数
    def high_fidelity(x):
        """高保真度: DFT计算模拟"""
        # 模拟真实的DFT计算 (更准确但更慢)
        true_value = -np.sum(x**2)
        noise = np.random.randn() * 0.05  # 低噪声
        return true_value + noise
    
    def low_fidelity(x):
        """低保真度: 力场计算模拟"""
        # 模拟力场计算 (更快但噪声更大)
        true_value = -np.sum(x**2)
        # 添加系统偏差和噪声
        bias = 0.1 * np.sin(np.sum(x))
        noise = np.random.randn() * 0.3  # 高噪声
        return true_value + bias + noise
    
    print(f"\nHigh-fidelity: Accurate but expensive (DFT-like)")
    print(f"Low-fidelity: Fast but noisy (Force-field-like)")
    
    # 运行多保真度优化
    mf_optimizer = MultiFidelityOptimizer(
        low_fidelity_fn=low_fidelity,
        high_fidelity_fn=high_fidelity,
        cost_ratio=0.1  # 低保真度成本是高保真度的10%
    )
    
    print("\nRunning multi-fidelity optimization...")
    best_params, best_value = mf_optimizer.optimize(
        [(-2, 2)] * 3,
        budget=40
    )
    
    print(f"\nResults:")
    print(f"  Best value: {best_value:.3f}")
    print(f"  Best params: {best_params}")
    print(f"  True optimum: [0, 0, 0], value = 0")
    
    # 对比: 只使用高保真度
    print("\nComparison: High-fidelity only (20 evaluations)")
    
    best_hf_value = float('-inf')
    best_hf_params = None
    
    for _ in range(20):  # 只能运行20次高保真度
        params = np.array([np.random.uniform(-2, 2) for _ in range(3)])
        value = high_fidelity(params)
        
        if value > best_hf_value:
            best_hf_value = value
            best_hf_params = params
    
    print(f"  Best value: {best_hf_value:.3f}")
    print(f"  Best params: {best_hf_params}")
    
    return best_params, best_value


def plot_comparison(results: dict):
    """绘制优化器比较图"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        for method, histories in results.items():
            # 计算平均和置信区间
            max_len = max(len(h) for h in histories)
            
            # 对齐历史
            aligned = []
            for h in histories:
                aligned.append(np.array(h + [h[-1]] * (max_len - len(h))))
            
            aligned = np.array(aligned)
            mean = np.mean(aligned, axis=0)
            std = np.std(aligned, axis=0)
            
            x = range(max_len)
            plt.plot(x, mean, label=method)
            plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        
        plt.xlabel('Iteration')
        plt.ylabel('Best Value Found')
        plt.title('Optimizer Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('optimizer_comparison.png', dpi=150)
        print("\nPlot saved to optimizer_comparison.png")
        
    except ImportError:
        print("\nMatplotlib not available, skipping plot")


def main():
    print("=" * 70)
    print("Process Parameter Optimization Examples")
    print("=" * 70)
    
    # 运行示例
    synthesis_optimization_example()
    results = parameter_optimization_example()
    multifidelity_optimization_example()
    
    # 绘图
    print("\nGenerating comparison plot...")
    plot_comparison(results)
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
