# KAS DRL API Reference

## Core Module

### StateSpace

```python
from core.state_space import StateSpace

state_space = StateSpace(device="cuda")
state = state_space.encode(agent_state, task_features, user_feedback)
```

**Methods**:
- `encode(agent_state, task_features, user_feedback) -> torch.Tensor`: 编码状态
- `get_state_dim() -> int`: 获取状态维度
- `reset()`: 重置状态
- `get_temporal_context(window_size) -> torch.Tensor`: 获取时序上下文

### ActionSpace

```python
from core.action_space import ActionSpace

action_space = ActionSpace()
prompt_action, template_action, param_action = action_space.decode(action_vector)
```

**Methods**:
- `decode(action_vector) -> Tuple[PromptAction, TemplateAction, ParameterAction]`: 解码动作
- `encode(prompt_action, template_action, param_action) -> np.ndarray`: 编码动作
- `sample_random() -> np.ndarray`: 随机采样
- `get_dimensions() -> Dict[str, int]`: 获取维度信息

### RewardFunction

```python
from core.reward import RewardFunction, RewardConfig, InteractionOutcome

config = RewardConfig(
    user_satisfaction_weight=0.35,
    capability_retention_weight=0.20,
    convergence_speed_weight=0.20,
    response_quality_weight=0.15,
    efficiency_weight=0.10
)
reward_fn = RewardFunction(config)
reward = reward_fn.compute(outcome)
```

**Methods**:
- `compute(outcome) -> float`: 计算奖励
- `get_statistics() -> Dict[str, float]`: 获取统计信息
- `reset()`: 重置历史

## Algorithms Module

### PPOAgent

```python
from algorithms.ppo import PPOAgent, PPOConfig

config = PPOConfig(
    lr=3e-4,
    gamma=0.99,
    clip_epsilon=0.2
)
agent = PPOAgent(state_dim=128, action_dim=21, config=config, device="cuda")

# 选择动作
action = agent.select_action(state)

# 存储转移
agent.store_transition(reward, next_state, done)

# 更新
losses = agent.update()

# 保存/加载
agent.save("model.pt")
agent.load("model.pt")
```

### SACAgent

```python
from algorithms.sac import SACAgent, SACConfig

config = SACConfig(
    lr=3e-4,
    alpha=0.2,
    automatic_entropy_tuning=True
)
agent = SACAgent(state_dim=128, action_dim=21, config=config, device="cuda")

# 选择动作
action = agent.select_action(state, deterministic=False)

# 添加到回放缓冲区
agent.replay_buffer.push(state, action, reward, next_state, done)

# 更新
losses = agent.update()
```

### DDPGAgent

```python
from algorithms.ddpg import DDPGAgent, DDPGConfig

config = DDPGConfig(
    actor_lr=1e-4,
    critic_lr=1e-3
)
agent = DDPGAgent(state_dim=128, action_dim=21, config=config, device="cuda")

action = agent.select_action(state, add_noise=True)
```

## Meta-Learning Module

### MAML

```python
from meta_learning.maml import MAML, MAMLConfig

config = MAMLConfig(
    inner_lr=0.01,
    outer_lr=0.001,
    num_inner_steps=5
)

maml = MAML(model, config)

# 元训练
metrics = maml.outer_step(tasks)

# 适应新任务
adapted_model = maml.adapt_to_new_task(support_data, num_steps=10)
```

### Reptile

```python
from meta_learning.reptile import Reptile, ReptileConfig

config = ReptileConfig(
    inner_lr=0.01,
    meta_lr=0.1,
    inner_steps=5
)

reptile = Reptile(model, config)

# 元训练
history = reptile.train(task_sampler, num_iterations=1000)

# 适应新任务
adapted_model = reptile.adapt(support_data, num_steps=10)
```

### Feature Encoders

```python
from meta_learning.encoders import (
    ProjectFeatureEncoder,
    TelemetryLSTMEncoder,
    MultiModalProjectEncoder
)

# 项目特征编码器
code_encoder = ProjectFeatureEncoder(
    vocab_size=10000,
    embedding_dim=256,
    output_dim=128
)
features = code_encoder(token_ids, attention_mask)

# 遥测数据编码器
telemetry_encoder = TelemetryLSTMEncoder(
    input_dim=10,
    hidden_dim=128,
    output_dim=64
)
features = telemetry_encoder(telemetry_sequence)

# 多模态编码器
multimodal_encoder = MultiModalProjectEncoder(
    code_vocab_size=10000,
    code_dim=128,
    telemetry_dim=64,
    output_dim=128
)
features = multimodal_encoder(code_tokens, telemetry_sequence)
```

## Training Module

### Trainer

```python
from training.trainer import Trainer, TrainingConfig

config = TrainingConfig(
    num_episodes=1000,
    max_steps_per_episode=500,
    eval_interval=50,
    log_dir="./logs"
)

trainer = Trainer(agent, env, config)
history = trainer.train()

# 评估
eval_metrics = trainer.evaluate(num_episodes=10)

# 保存/加载
trainer.save_checkpoint("model.pt")
trainer.load_checkpoint("model.pt")
```

### Environment

```python
from training.environment import KASAgentEnv, CurriculumEnv

# 基础环境
env = KASAgentEnv(config)
state = env.reset()
next_state, reward, done, info = env.step(action)

# 课程学习环境
env = CurriculumEnv(config)
# 难度会自动根据成功率调整
```

### Online Learning

```python
from training.online_learning import OnlineLearner, OnlineLearningConfig

config = OnlineLearningConfig(
    buffer_size=1000,
    update_interval=50,
    batch_size=32
)

online_learner = OnlineLearner(agent, config)

# 观察交互结果
online_learner.observe(state, action, reward, next_state, done)

# 获取统计信息
stats = online_learner.get_learning_statistics()
```

## Integration Module

### LLMClientAdapter

```python
from integration.llm_client import LLMClientAdapter, LLMConfig

config = LLMConfig(
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=2000
)

adapter = LLMClientAdapter(
    llm_client=existing_client,
    drl_agent=trained_agent,
    state_encoder=state_encoder,
    config=config
)

# 生成响应
result = adapter.generate(prompt, context, use_drl=True)

# 接收用户反馈
adapter.feedback(rating=4.5, metadata={})
```

### Fallback Manager

```python
from integration.fallback import FallbackManager, FallbackConfig

config = FallbackConfig(
    error_threshold=5,
    latency_threshold=10.0
)

fallback = FallbackManager(config)

# 记录错误
fallback.on_error(exception)

# 记录延迟
fallback.on_latency(latency)

# 获取当前降级参数
params = fallback.get_current_params()

# 注册回调
fallback.register_fallback_callback(on_fallback)
```

### Deployment

```python
from integration.deployment import (
    CanaryDeployment,
    ABTestDeployment,
    ModelRegistry
)

# 金丝雀部署
canary = CanaryDeployment(old_agent, new_agent, config)
canary.start()
action = canary.route_request(state)
canary.record_metric(is_canary=True, metric={'reward': 1.0})
result = canary.evaluate()

# A/B测试
ab_test = ABTestDeployment(control_agent, treatment_agent, config)
group = ab_test.assign_user(user_id)
agent = ab_test.get_agent(user_id)
ab_test.record_outcome(user_id, outcome)
analysis = ab_test.analyze()

# 模型注册表
registry = ModelRegistry("./model_registry")
registry.register("kas_agent", "v1.0", "path/to/model.pt", metrics)
registry.promote("kas_agent", "v1.0", "production")
model_path = registry.get_model("kas_agent", "production")
```

## Command Line Interface

### Training

```bash
# 训练PPO
python main.py train --config configs/ppo_default.yaml --seed 42

# 训练SAC
python main.py train --config configs/sac_default.yaml

# 评估
python main.py eval --config configs/ppo_default.yaml --checkpoint path/to/model.pt --episodes 10
```

### Meta-Learning

```bash
# 训练MAML
python main.py meta-train --config configs/maml_meta.yaml --method maml

# 训练Reptile
python main.py meta-train --config configs/maml_meta.yaml --method reptile
```

### Demo

```bash
python main.py demo
```

## Configuration Files

### PPO Config (configs/ppo_default.yaml)

```yaml
agent:
  type: "ppo"
  state_dim: 128
  action_dim: 21
  lr: 3.0e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  value_coef: 0.5
  entropy_coef: 0.01

training:
  num_episodes: 2000
  eval_interval: 50
  device: "cuda"
```

### SAC Config (configs/sac_default.yaml)

```yaml
agent:
  type: "sac"
  state_dim: 128
  action_dim: 21
  lr: 3.0e-4
  alpha: 0.2
  automatic_entropy_tuning: true
  buffer_size: 100000
```

### MAML Config (configs/maml_meta.yaml)

```yaml
meta_learning:
  type: "maml"
  inner_lr: 0.01
  outer_lr: 0.001
  num_inner_steps: 5
  k_shot: 5
  q_query: 15

encoder:
  vocab_size: 10000
  embedding_dim: 256
  output_dim: 128
```
