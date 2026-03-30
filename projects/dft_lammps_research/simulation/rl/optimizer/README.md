# 强化学习材料优化引擎 (RL Optimizer for Materials Design)

一个基于强化学习(RL)的材料设计和优化引擎，用于晶体结构优化、化学组成设计和材料性质预测。

## 功能特性

### 1. RL环境设计
- **晶体结构操作空间**: 添加/删除/移动/替换原子
- **化学组成调整空间**: 元素比例优化
- **状态表示**: 图神经网络编码
- **奖励函数**: 能量、稳定性、目标性质组合

### 2. RL算法实现
- **PPO** (Proximal Policy Optimization)
- **SAC** (Soft Actor-Critic)
- **DQN** 及其变体 (Dueling DQN, Rainbow DQN)
- **多目标RL** (NSGA-III, MOEA/D)
- **离线RL** (CQL, Decision Transformer)

### 3. 材料优化场景
- **电池材料**: 优化离子电导率、电压窗口
- **催化剂**: 优化活性位点、选择性
- **合金**: 优化强度-延展性平衡
- **拓扑材料**: 发现新奇拓扑态

### 4. DFT/MD耦合
- RL代理调用DFT计算奖励
- 使用ML势快速评估
- 主动学习迭代改进
- 人机协作优化

### 5. 可解释性
- 注意力可视化
- 优化轨迹分析
- 化学直觉提取
- 反事实解释

## 安装

```bash
# 克隆仓库
git clone <repository-url>
cd dft_lammps_research

# 安装依赖
pip install numpy scipy torch matplotlib
```

## 快速开始

### 电池材料优化

```python
from rl_optimizer import BatteryOptimizer, BatteryConfig

# 创建配置
config = BatteryConfig(
    target_ionic_conductivity=1e-3,
    n_episodes=100
)

# 创建优化器
optimizer = BatteryOptimizer(config=config)

# 训练
results = optimizer.train()

# 查看最佳结构
for struct in results['best_structures'][:5]:
    print(f"组成: {struct['formula']}, 奖励: {struct['reward']:.4f}")
```

### 合金多目标优化

```python
from rl_optimizer import AlloyOptimizer, AlloyConfig

# 创建配置
config = AlloyConfig(
    alloy_type='lightweight',
    population_size=100,
    n_generations=200
)

# 创建优化器
optimizer = AlloyOptimizer(config=config)

# 优化
pareto_front = optimizer.optimize()

# 获取最佳合金
best_alloys = optimizer.get_best_alloys(n=5)
for alloy in best_alloys:
    print(f"强度: {alloy['strength']:.1f} MPa")
    print(f"延展性: {alloy['ductility']:.3f}")
    print(f"密度: {alloy['density']:.2f} g/cm³")
```

### 基础RL使用

```python
from rl_optimizer import CrystalStructureEnv, SACAgent

# 创建环境
env = CrystalStructureEnv()

# 创建智能体
agent = SACAgent()

# 训练
for episode in range(100):
    state = env.reset()
    episode_reward = 0
    
    for step in range(50):
        action = agent.select_action(state)
        result = env.step(action)
        
        agent.store_transition(
            state, action, result.reward,
            result.state, result.done
        )
        agent.update()
        
        episode_reward += result.reward
        
        if result.done:
            break
        
        state = result.state
    
    print(f"Episode {episode}: Reward = {episode_reward:.4f}")
```

## 模块结构

```
rl_optimizer/
├── __init__.py              # 主模块入口
├── environment/             # RL环境
│   ├── base_env.py         # 基础环境类
│   ├── crystal_env.py      # 晶体结构环境
│   └── composition_env.py  # 化学组成环境
├── algorithms/              # RL算法
│   ├── ppo.py              # PPO算法
│   ├── sac.py              # SAC算法
│   ├── dqn.py              # DQN及其变体
│   ├── multi_objective.py  # 多目标RL
│   └── offline_rl.py       # 离线RL
├── representations/         # 状态表示
│   └── __init__.py         # GNN编码器
├── rewards/                 # 奖励函数
│   └── __init__.py         # 各种奖励函数
├── scenarios/               # 材料优化场景
│   ├── battery.py          # 电池材料
│   ├── catalyst.py         # 催化剂
│   ├── alloy.py            # 合金
│   └── topological.py      # 拓扑材料
├── coupling/                # DFT/MD耦合
│   └── __init__.py         # 耦合接口
├── explainability/          # 可解释性
│   └── __init__.py         # 解释工具
├── visualization/           # 可视化
│   └── __init__.py         # 绘图工具
└── examples/                # 应用示例
    ├── example_battery.py
    ├── example_catalyst.py
    ├── example_alloy.py
    └── example_topological.py
```

## 算法说明

### PPO (Proximal Policy Optimization)
- 稳定的策略梯度方法
- 适用于连续和离散动作空间
- 通过裁剪防止策略更新过大

### SAC (Soft Actor-Critic)
- Off-policy最大熵算法
- 高样本效率
- 适合连续动作空间的材料优化

### DQN及其变体
- Dueling DQN: 分离价值流和优势流
- Rainbow DQN: 组合多种改进技术
- 适合离散动作空间

### 多目标RL
- NSGA-III: 基于参考点的非支配排序
- MOEA/D: 基于分解的多目标进化算法
- 适用于强度-延展性等多目标平衡

### 离线RL
- CQL: 保守Q学习，防止价值高估
- Decision Transformer: 序列建模方法
- 适合利用已有材料数据库

## DFT/MD集成

### 主动学习循环

```python
from rl_optimizer import ActiveLearningCoupling, DFTCoupling, MLCoupling

# 创建耦合器
dft = DFTCoupling(calculator='vasp')
ml = MLCoupling(potential_type='nep')

al_coupling = ActiveLearningCoupling(
    dft_coupling=dft,
    ml_coupling=ml,
    uncertainty_threshold=0.2
)

# 智能选择计算方法
energy, method = al_coupling.calculate_energy(structure)
print(f"使用 {method} 计算，能量 = {energy:.4f} eV")
```

## 可解释性工具

### 注意力可视化

```python
from rl_optimizer import AttentionVisualizer

viz = AttentionVisualizer()
attention = viz.extract_attention(model, state)
important_sites = viz.identify_important_sites(attention, structure)
```

### 轨迹分析

```python
from rl_optimizer import TrajectoryAnalyzer

analyzer = TrajectoryAnalyzer()
analyzer.add_trajectory(trajectory)

# 分析动作分布
action_dist = analyzer.analyze_action_distribution()

# 识别常见模式
patterns = analyzer.identify_common_patterns()
```

### 化学直觉提取

```python
from rl_optimizer import ChemicalIntuitionExtractor

extractor = ChemicalIntuitionExtractor()
correlations = extractor.extract_element_correlations(structures, rewards)
guidelines = extractor.generate_design_guidelines()
```

## 可视化

```python
from rl_optimizer.visualization import OptimizationPlotter

plotter = OptimizationPlotter()

# 绘制奖励曲线
plotter.plot_reward_curve('reward_curve.png')

# 绘制帕累托前沿
plotter.plot_multi_objective_front(pareto_front, save_path='pareto.png')
```

## 示例

### 运行电池优化示例

```bash
python -m rl_optimizer.examples.example_battery
```

### 运行合金优化示例

```bash
python -m rl_optimizer.examples.example_alloy
```

### 运行综合示例

```bash
python -m rl_optimizer.examples
```

## 性能优化

### GPU加速

```python
# SAC配置中使用GPU
config = SACConfig(device='cuda')
agent = SACAgent(config)
```

### 并行训练

```python
from rl_optimizer.algorithms import ParallelOptimizer

optimizer = ParallelOptimizer(n_workers=4)
optimizer.train(env, agent)
```

## 引用

如果您使用本工具进行研究，请引用：

```bibtex
@software{rl_optimizer_materials,
  title={RL Optimizer for Materials Design},
  author={DFT-ML Research Team},
  year={2025}
}
```

## 参考文献

1. Schulman et al. "Proximal Policy Optimization Algorithms", 2017
2. Haarnoja et al. "Soft Actor-Critic", 2018
3. Deb & Jain "NSGA-III", IEEE TEC 2014
4. Kumar et al. "Conservative Q-Learning", NeurIPS 2020
5. Chen et al. "Decision Transformer", NeurIPS 2021
6. Xie & Grossman "Crystal Graph CNN", 2018

## 许可证

MIT License

## 联系我们

- 邮箱: dft-ml@example.com
- 项目地址: https://github.com/example/rl-optimizer
