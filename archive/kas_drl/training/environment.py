"""
KAS Training - Simulation Environment
模拟环境用于离线训练
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import gym
from gym import spaces


@dataclass
class KASEnvConfig:
    """KAS环境配置"""
    max_steps: int = 100
    num_tasks: int = 10
    task_complexity_range: Tuple[float, float] = (0.1, 1.0)
    user_satisfaction_threshold: float = 0.7
    capability_decay: float = 0.01
    reward_scale: float = 1.0


class KASAgentEnv(gym.Env):
    """KAS Agent模拟环境"""
    
    def __init__(self, config: Optional[KASEnvConfig] = None):
        super().__init__()
        
        self.config = config or KASEnvConfig()
        self.current_step = 0
        
        # 状态空间
        # [agent_capabilities(8), task_features(5), user_feedback(5), telemetry(10)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(28,),
            dtype=np.float32
        )
        
        # 动作空间
        # [prompt_adjustment(9), template_selection(7), parameters(5)]
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(21,),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_step = 0
        
        # 初始化Agent能力
        self.agent_capabilities = np.random.rand(8) * 0.5 + 0.3  # 0.3-0.8
        
        # 生成初始任务
        self.current_task = self._generate_task()
        
        # 初始化用户反馈
        self.user_feedback = np.zeros(5)
        
        # 初始化遥测数据
        self.telemetry = np.zeros(10)
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作
        
        Args:
            action: [21] 动作向量
        
        Returns:
            observation: 新状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        self.current_step += 1
        
        # 解析动作
        prompt_action = self._parse_prompt_action(action[:9])
        template_action = self._parse_template_action(action[9:16])
        param_action = self._parse_param_action(action[16:])
        
        # 模拟交互结果
        interaction_result = self._simulate_interaction(
            prompt_action, template_action, param_action
        )
        
        # 更新状态
        self._update_state(interaction_result)
        
        # 计算奖励
        reward = self._compute_reward(interaction_result)
        
        # 检查是否结束
        done = (
            self.current_step >= self.config.max_steps or
            np.mean(self.agent_capabilities) < 0.1
        )
        
        info = {
            'interaction_result': interaction_result,
            'agent_capabilities': self.agent_capabilities.copy(),
            'step': self.current_step
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """获取观察"""
        return np.concatenate([
            self.agent_capabilities,
            self.current_task,
            self.user_feedback,
            self.telemetry
        ])
    
    def _generate_task(self) -> np.ndarray:
        """生成任务特征"""
        complexity = np.random.uniform(*self.config.task_complexity_range)
        task_type = np.random.randint(0, 6)  # 6种任务类型
        
        return np.array([
            complexity,
            task_type / 5.0,  # 归一化
            np.random.rand(),  # 领域
            np.random.rand(),  # 语言
            np.random.rand()   # 紧急程度
        ])
    
    def _parse_prompt_action(self, action: np.ndarray) -> Dict:
        """解析Prompt调整动作"""
        adj_type = np.argmax(action[:8])
        strength = np.clip(action[8], 0, 1)
        return {'type': adj_type, 'strength': strength}
    
    def _parse_template_action(self, action: np.ndarray) -> Dict:
        """解析模板选择动作"""
        template = np.argmax(action[:6])
        confidence = np.clip(action[6], 0, 1)
        return {'template': template, 'confidence': confidence}
    
    def _parse_param_action(self, action: np.ndarray) -> Dict:
        """解析参数动作"""
        return {
            'temperature': np.clip(action[0], 0, 1) * 2,
            'max_tokens': int(np.clip(action[1], 0, 1) * 4000),
            'top_p': np.clip(action[2], 0, 1),
            'frequency_penalty': np.clip(action[3], -1, 1) * 2,
            'presence_penalty': np.clip(action[4], -1, 1) * 2
        }
    
    def _simulate_interaction(
        self,
        prompt_action: Dict,
        template_action: Dict,
        param_action: Dict
    ) -> Dict:
        """模拟用户交互"""
        # 基于Agent能力和任务复杂度计算成功率
        task_complexity = self.current_task[0]
        avg_capability = np.mean(self.agent_capabilities)
        
        # 动作质量影响
        action_quality = (
            prompt_action['strength'] * 0.3 +
            template_action['confidence'] * 0.3 +
            (1 - abs(param_action['temperature'] - 0.7)) * 0.4
        )
        
        # 计算成功概率
        success_prob = avg_capability * (1 - task_complexity * 0.5) * action_quality
        success = np.random.rand() < success_prob
        
        # 用户满意度
        if success:
            satisfaction = np.random.uniform(0.6, 1.0)
        else:
            satisfaction = np.random.uniform(0.1, 0.5)
        
        # 响应时间（与复杂度和参数相关）
        response_time = task_complexity * 30 + np.random.rand() * 10
        
        return {
            'success': success,
            'satisfaction': satisfaction,
            'response_time': response_time,
            'iterations': 1 if success else np.random.randint(2, 5),
            'action_quality': action_quality
        }
    
    def _update_state(self, interaction: Dict):
        """更新状态"""
        # 更新能力
        if interaction['success']:
            self.agent_capabilities += 0.02 * (1 - self.agent_capabilities)
        else:
            self.agent_capabilities -= self.config.capability_decay
        
        self.agent_capabilities = np.clip(self.agent_capabilities, 0, 1)
        
        # 更新用户反馈
        self.user_feedback = np.array([
            interaction['satisfaction'],
            interaction['satisfaction'] > self.config.user_satisfaction_threshold,
            interaction['iterations'] / 5.0,
            1.0 / (1 + interaction['response_time'] / 60),
            interaction['action_quality']
        ])
        
        # 更新遥测
        self.telemetry = np.roll(self.telemetry, -1)
        self.telemetry[-1] = interaction['satisfaction']
        
        # 生成新任务
        if np.random.rand() < 0.3:  # 30%概率切换任务
            self.current_task = self._generate_task()
    
    def _compute_reward(self, interaction: Dict) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 成功奖励
        if interaction['success']:
            reward += 2.0
        else:
            reward -= 1.0
        
        # 满意度奖励
        reward += interaction['satisfaction'] * 2.0 - 1.0
        
        # 效率奖励
        efficiency = 1.0 / (1 + interaction['response_time'] / 60)
        reward += efficiency * 0.5
        
        # 迭代次数惩罚
        reward -= (interaction['iterations'] - 1) * 0.2
        
        return reward * self.config.reward_scale
    
    def render(self, mode='human'):
        """渲染环境"""
        print(f"Step: {self.current_step}")
        print(f"Capabilities: {self.agent_capabilities.mean():.3f}")
        print(f"Task Complexity: {self.current_task[0]:.3f}")
        print(f"User Satisfaction: {self.user_feedback[0]:.3f}")


class CurriculumEnv(KASAgentEnv):
    """课程学习环境"""
    
    def __init__(self, config: Optional[KASEnvConfig] = None):
        super().__init__(config)
        self.difficulty_level = 0
        self.success_history = []
    
    def _generate_task(self) -> np.ndarray:
        """根据当前难度生成任务"""
        # 难度随等级增加
        min_complexity = 0.1 + self.difficulty_level * 0.15
        max_complexity = min(0.3 + self.difficulty_level * 0.2, 1.0)
        
        complexity = np.random.uniform(min_complexity, max_complexity)
        task_type = np.random.randint(0, 6)
        
        return np.array([
            complexity,
            task_type / 5.0,
            np.random.rand(),
            np.random.rand(),
            np.random.rand()
        ])
    
    def _compute_reward(self, interaction: Dict) -> float:
        """计算课程奖励"""
        reward = super()._compute_reward(interaction)
        
        # 记录成功
        self.success_history.append(interaction['success'])
        if len(self.success_history) > 20:
            self.success_history.pop(0)
        
        # 升级检查
        if len(self.success_history) >= 20:
            recent_success_rate = np.mean(self.success_history[-20:])
            if recent_success_rate > 0.7 and self.difficulty_level < 5:
                self.difficulty_level += 1
                reward += 5.0  # 升级奖励
                print(f"Level Up! New difficulty: {self.difficulty_level}")
        
        return reward


class MultiTaskEnv(KASAgentEnv):
    """多任务环境"""
    
    def __init__(self, config: Optional[KASEnvConfig] = None):
        super().__init__(config)
        self.tasks = []
        self.current_task_idx = 0
        self.task_performance = {}
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.tasks = [self._generate_task() for _ in range(self.config.num_tasks)]
        self.current_task_idx = 0
        self.current_task = self.tasks[0]
        self.task_performance = {i: [] for i in range(self.config.num_tasks)}
        
        return super().reset()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        obs, reward, done, info = super().step(action)
        
        # 记录任务表现
        self.task_performance[self.current_task_idx].append(info['interaction_result']['success'])
        
        # 任务轮换
        self.current_task_idx = (self.current_task_idx + 1) % self.config.num_tasks
        self.current_task = self.tasks[self.current_task_idx]
        
        return obs, reward, done, info
