# KAS DRL 训练指南

## 快速开始

### 安装依赖

```bash
cd kas_drl
pip install -r requirements.txt
```

### 运行演示

```bash
python main.py demo
```

### 训练PPO Agent

```bash
python main.py train --config configs/ppo_default.yaml --seed 42
```

### 训练SAC Agent

```bash
python main.py train --config configs/sac_default.yaml
```

### 元学习训练

```bash
python main.py meta-train --config configs/maml_meta.yaml --method maml
```

## 训练配置

### 修改训练参数

编辑配置文件 `configs/ppo_default.yaml`:

```yaml
agent:
  lr: 3.0e-4          # 学习率
  gamma: 0.99         # 折扣因子
  clip_epsilon: 0.2   # PPO裁剪参数

training:
  num_episodes: 2000  # 训练回合数
  eval_interval: 50   # 评估间隔
  device: "cuda"      # 训练设备
```

### 命令行覆盖

可以通过命令行参数覆盖配置:

```bash
python train.py train --config configs/ppo_default.yaml \
    --num-episodes 3000 \
    --device cpu
```

## 监控训练

### TensorBoard (可选)

```bash
tensorboard --logdir ./logs
```

### 查看训练历史

```python
import json

with open('./logs/kas_ppo_default/training_history.json', 'r') as f:
    history = json.load(f)

print(f"Mean reward: {sum(history['episode_rewards'][-100:]) / 100}")
```

## 评估模型

### 基础评估

```bash
python main.py eval --config configs/ppo_default.yaml \
    --checkpoint ./logs/kas_ppo_default/best_model.pt \
    --episodes 10
```

### 编程评估

```python
from train import create_agent, create_env
from training.trainer import Trainer

# 加载配置
config = load_config('configs/ppo_default.yaml')

# 创建环境和Agent
env = create_env(config)
agent = create_agent(config, 'cuda')

# 加载模型
agent.load('path/to/model.pt')

# 评估
trainer = Trainer(agent, env)
metrics = trainer.evaluate(num_episodes=10)
print(f"Mean reward: {metrics['mean_reward']:.2f}")
```

## 超参数调优

### 学习率

- **太小 (1e-5)**: 收敛慢
- **适中 (3e-4)**: 推荐值
- **太大 (1e-2)**: 不稳定

### 折扣因子 (gamma)

- **0.99**: 推荐值，考虑长期回报
- **0.95**: 更注重近期回报

### 熵系数

- **0.01**: 适度探索
- **0.001**: 较少探索（策略更确定性）
- **0.1**: 更多探索

## 故障排查

### 训练不稳定

**症状**: 奖励波动大

**解决方案**:
- 降低学习率
- 增加batch size
- 减小clip_epsilon

### 收敛慢

**症状**: 奖励增长缓慢

**解决方案**:
- 增加熵系数
- 调整GAE lambda
- 检查奖励函数设计

### 过拟合

**症状**: 训练奖励高，评估奖励低

**解决方案**:
- 增加环境多样性
- 使用早停
- 增加评估频率

## 高级训练

### 课程学习

编辑环境配置:

```yaml
environment:
  use_curriculum: true
  task_complexity_range: [0.1, 1.0]
```

### 多任务训练

```python
from training.environment import MultiTaskEnv

env = MultiTaskEnv(config)
env.tasks = [task1, task2, task3]
```

### 在线学习

```python
from training.online_learning import OnlineLearner

online_learner = OnlineLearner(agent)

# 部署后收集反馈
online_learner.observe(state, action, reward, next_state, done)
```

## 性能优化

### GPU训练

确保CUDA可用:

```python
import torch
print(torch.cuda.is_available())
```

配置文件中设置:

```yaml
training:
  device: "cuda"
```

### 混合精度 (可选)

```yaml
training:
  use_mixed_precision: true
```

### 并行训练 (高级)

```python
from training.trainer import DistributedTrainer

trainer = DistributedTrainer(
    agents=[agent1, agent2, agent3, agent4],
    env_factory=lambda: create_env(config),
    num_workers=4
)
```

## 最佳实践

### 1. 从小规模开始

```bash
# 先用100回合测试
python main.py train --config configs/ppo_default.yaml \
    --num-episodes 100
```

### 2. 保存检查点

```python
# 每100回合保存
training:
  save_interval: 100
```

### 3. 早停

```python
# 如果50回合没有改善则停止
training:
  use_early_stopping: true
  patience: 50
  min_delta: 0.01
```

### 4. 复现实验

```bash
# 设置随机种子
python main.py train --config configs/ppo_default.yaml --seed 42
```

## 模型部署

### 导出模型

```python
agent.save('production_model.pt')
```

### 加载生产模型

```python
agent.load('production_model.pt')
```

### 与LLMClient集成

```python
from integration.llm_client import LLMClientAdapter

adapter = LLMClientAdapter(
    llm_client=existing_client,
    drl_agent=agent,
    state_encoder=encoder
)

result = adapter.generate(prompt, use_drl=True)
```
