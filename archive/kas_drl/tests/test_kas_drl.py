"""
KAS DRL - Unit Tests
"""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.state_space import StateSpace, AgentState, TaskFeatures, UserFeedback
from core.action_space import ActionSpace, PromptAction, TemplateAction, ParameterAction
from core.reward import RewardFunction, RewardConfig, InteractionOutcome


class TestStateSpace:
    """测试状态空间"""
    
    def test_agent_state_to_vector(self):
        """测试Agent状态转向量"""
        from core.state_space import AgentCapability
        
        agent_state = AgentState(
            capabilities={cap: 0.5 for cap in AgentCapability},
            confidence=0.8,
            success_rate=0.75,
            experience_count=100
        )
        
        vec = agent_state.to_vector()
        assert len(vec) == 11  # 8 capabilities + 3 meta
        assert all(0 <= v <= 1 for v in vec)
    
    def test_task_features_to_vector(self):
        """测试任务特征转向量"""
        task = TaskFeatures(
            task_type="code_review",
            complexity=0.5,
            domain="backend",
            language="python",
            file_count=10,
            line_count=1000
        )
        
        vec = task.to_vector()
        assert len(vec) == 15  # 10 one-hot + 5 features
        assert 0 <= vec[10] <= 1  # complexity
    
    def test_state_encoder(self):
        """测试状态编码器"""
        encoder = StateSpace(device="cpu")
        
        from core.state_space import AgentCapability
        agent_state = AgentState(
            capabilities={cap: 0.5 for cap in AgentCapability},
            confidence=0.8,
            success_rate=0.75,
            experience_count=100
        )
        task = TaskFeatures(
            task_type="code_review",
            complexity=0.5,
            domain="backend",
            language="python",
            file_count=10,
            line_count=1000
        )
        
        state = encoder.encode(agent_state, task)
        assert state.shape == (1, 128)  # batch=1, output_dim=128


class TestActionSpace:
    """测试动作空间"""
    
    def test_action_dimensions(self):
        """测试动作维度"""
        action_space = ActionSpace()
        dims = action_space.get_dimensions()
        
        assert dims['prompt'] == 9  # 8 adjustments + strength
        assert dims['template'] == 7  # 6 templates + confidence
        assert dims['param'] == 5  # 5 parameters
        assert dims['total'] == 21
    
    def test_action_encode_decode(self):
        """测试动作编解码"""
        action_space = ActionSpace()
        
        prompt_action = PromptAction(
            adjustment=list(PromptAdjustment)[0],
            strength=0.5
        )
        template_action = TemplateAction(
            template=list(TemplateType)[0],
            confidence=0.8
        )
        param_action = ParameterAction(
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # 编码
        vec = action_space.encode(prompt_action, template_action, param_action)
        assert len(vec) == 21
        
        # 解码
        p, t, param = action_space.decode(vec)
        assert isinstance(p, PromptAction)
        assert isinstance(t, TemplateAction)
        assert isinstance(param, ParameterAction)


class TestRewardFunction:
    """测试奖励函数"""
    
    def test_reward_computation(self):
        """测试奖励计算"""
        config = RewardConfig()
        reward_fn = RewardFunction(config)
        
        outcome = InteractionOutcome(
            user_rating=4.5,
            response_time=10.0,
            token_usage=1000,
            iteration_count=1,
            accuracy=0.9,
            completeness=0.85,
            relevance=0.9,
            accepted=True,
            capability_changes={'code_review': 0.1}
        )
        
        reward = reward_fn.compute(outcome)
        assert isinstance(reward, float)
        assert -10 <= reward <= 10  # 奖励裁剪范围
    
    def test_reward_statistics(self):
        """测试奖励统计"""
        config = RewardConfig()
        reward_fn = RewardFunction(config)
        
        # 添加一些奖励
        for i in range(10):
            outcome = InteractionOutcome(
                user_rating=3.0 + i * 0.2,
                response_time=10.0,
                token_usage=1000,
                iteration_count=1,
                accuracy=0.8,
                completeness=0.8,
                relevance=0.8,
                accepted=True
            )
            reward_fn.compute(outcome)
        
        stats = reward_fn.get_statistics()
        assert 'mean_reward' in stats
        assert 'recent_mean' in stats


class TestAlgorithms:
    """测试算法"""
    
    def test_ppo_agent_creation(self):
        """测试PPO Agent创建"""
        from algorithms.ppo import PPOAgent, PPOConfig
        
        config = PPOConfig()
        agent = PPOAgent(state_dim=28, action_dim=21, config=config, device='cpu')
        
        assert agent is not None
        assert agent.state_dim == 28
        assert agent.action_dim == 21
    
    def test_sac_agent_creation(self):
        """测试SAC Agent创建"""
        from algorithms.sac import SACAgent, SACConfig
        
        config = SACConfig()
        agent = SACAgent(state_dim=28, action_dim=21, config=config, device='cpu')
        
        assert agent is not None
        assert hasattr(agent, 'actor')
        assert hasattr(agent, 'critic1')
    
    def test_ddpg_agent_creation(self):
        """测试DDPG Agent创建"""
        from algorithms.ddpg import DDPGAgent, DDPGConfig
        
        config = DDPGConfig()
        agent = DDPGAgent(state_dim=28, action_dim=21, config=config, device='cpu')
        
        assert agent is not None
        assert hasattr(agent, 'actor')
        assert hasattr(agent, 'critic')


class TestEnvironment:
    """测试环境"""
    
    def test_env_reset(self):
        """测试环境重置"""
        from training.environment import KASAgentEnv
        
        env = KASAgentEnv()
        state = env.reset()
        
        assert isinstance(state, np.ndarray)
        assert len(state) == 28  # 状态维度
    
    def test_env_step(self):
        """测试环境步进"""
        from training.environment import KASAgentEnv
        
        env = KASAgentEnv()
        state = env.reset()
        
        action = np.random.rand(21) * 2 - 1  # [-1, 1]
        next_state, reward, done, info = env.step(action)
        
        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)


class TestMetaLearning:
    """测试元学习"""
    
    def test_maml_creation(self):
        """测试MAML创建"""
        from meta_learning.maml import MAML, MAMLConfig
        from algorithms.ppo import ActorNetwork
        
        model = ActorNetwork(state_dim=128, action_dim=21)
        config = MAMLConfig()
        maml = MAML(model, config)
        
        assert maml is not None
    
    def test_reptile_creation(self):
        """测试Reptile创建"""
        from meta_learning.reptile import Reptile, ReptileConfig
        from algorithms.ppo import ActorNetwork
        
        model = ActorNetwork(state_dim=128, action_dim=21)
        config = ReptileConfig()
        reptile = Reptile(model, config)
        
        assert reptile is not None


class TestEncoders:
    """测试编码器"""
    
    def test_project_feature_encoder(self):
        """测试项目特征编码器"""
        from meta_learning.encoders import ProjectFeatureEncoder
        
        encoder = ProjectFeatureEncoder(vocab_size=1000, output_dim=64)
        
        # 模拟token输入
        tokens = torch.randint(0, 1000, (2, 50))  # batch=2, seq_len=50
        features = encoder(tokens)
        
        assert features.shape == (2, 64)
    
    def test_telemetry_lstm_encoder(self):
        """测试遥测LSTM编码器"""
        from meta_learning.encoders import TelemetryLSTMEncoder
        
        encoder = TelemetryLSTMEncoder(input_dim=10, output_dim=32)
        
        # 模拟遥测序列
        telemetry = torch.randn(2, 100, 10)  # batch=2, seq=100, features=10
        features = encoder(telemetry)
        
        assert features.shape == (2, 32)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
