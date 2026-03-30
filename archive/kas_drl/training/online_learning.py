"""
KAS Training - Online Learning
在线学习机制
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from collections import deque
import threading
import time


@dataclass
class OnlineLearningConfig:
    """在线学习配置"""
    buffer_size: int = 1000
    update_interval: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    use_experience_replay: bool = True
    priority_sampling: bool = True
    drift_detection_window: int = 100
    adaptation_rate: float = 0.1


class ExperienceBuffer:
    """经验缓冲区"""
    
    def __init__(self, capacity: int, priority_sampling: bool = True):
        self.capacity = capacity
        self.priority_sampling = priority_sampling
        
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.size = 0
    
    def add(self, experience: Dict, priority: float = 1.0):
        """添加经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, alpha: float = 0.6) -> List[Dict]:
        """采样经验"""
        if self.size < batch_size:
            return self.buffer[:self.size]
        
        if self.priority_sampling:
            # 优先采样
            priorities = self.priorities[:self.size] ** alpha
            probs = priorities / priorities.sum()
            indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        else:
            indices = np.random.choice(self.size, batch_size, replace=False)
        
        return [self.buffer[i] for i in indices]
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return self.size


class OnlineLearner:
    """在线学习器"""
    
    def __init__(
        self,
        agent,
        config: Optional[OnlineLearningConfig] = None
    ):
        self.agent = agent
        self.config = config or OnlineLearningConfig()
        
        self.buffer = ExperienceBuffer(
            self.config.buffer_size,
            self.config.priority_sampling
        )
        
        self.update_counter = 0
        self.performance_history = deque(maxlen=self.config.drift_detection_window)
        self.is_adapting = False
        
        # 线程锁
        self.lock = threading.Lock()
    
    def observe(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Optional[Dict] = None
    ):
        """
        观察交互结果
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
            info: 额外信息
        """
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'timestamp': time.time(),
            'info': info or {}
        }
        
        with self.lock:
            # 计算TD误差作为优先级
            priority = abs(reward) + 0.01
            self.buffer.add(experience, priority)
            
            self.update_counter += 1
            self.performance_history.append(reward)
            
            # 检查是否需要更新
            if self.update_counter >= self.config.update_interval:
                self._adapt()
                self.update_counter = 0
    
    def _adapt(self):
        """适应性更新"""
        if len(self.buffer) < self.config.batch_size:
            return
        
        self.is_adapting = True
        
        # 检测概念漂移
        if self._detect_drift():
            print("Concept drift detected! Adapting...")
            self._aggressive_adaptation()
        else:
            self._gentle_adaptation()
        
        self.is_adapting = False
    
    def _detect_drift(self) -> bool:
        """检测概念漂移"""
        if len(self.performance_history) < self.config.drift_detection_window:
            return False
        
        # 使用CUSUM检测
        recent = list(self.performance_history)[-50:]
        older = list(self.performance_history)[:-50]
        
        if np.mean(recent) < np.mean(older) - 2 * np.std(older):
            return True
        
        return False
    
    def _gentle_adaptation(self):
        """温和适应"""
        # 小批量梯度更新
        batch = self.buffer.sample(self.config.batch_size)
        
        for experience in batch:
            if hasattr(self.agent, 'store_transition'):
                self.agent.store_transition(
                    experience['reward'],
                    experience['next_state'],
                    experience['done']
                )
        
        if hasattr(self.agent, 'update'):
            self.agent.update()
    
    def _aggressive_adaptation(self):
        """激进适应 - 概念漂移时"""
        # 增加学习率并多轮更新
        batch = self.buffer.sample(min(len(self.buffer), 100))
        
        for _ in range(5):  # 多次更新
            for experience in batch:
                if hasattr(self.agent, 'store_transition'):
                    self.agent.store_transition(
                        experience['reward'],
                        experience['next_state'],
                        experience['done']
                    )
            
            if hasattr(self.agent, 'update'):
                self.agent.update()
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计"""
        stats = {
            'buffer_size': len(self.buffer),
            'is_adapting': self.is_adapting,
            'update_counter': self.update_counter
        }
        
        if self.performance_history:
            recent_rewards = list(self.performance_history)[-100:]
            stats['mean_recent_reward'] = np.mean(recent_rewards)
            stats['std_recent_reward'] = np.std(recent_rewards)
            stats['trend'] = self._calculate_trend()
        
        return stats
    
    def _calculate_trend(self) -> float:
        """计算性能趋势"""
        if len(self.performance_history) < 10:
            return 0.0
        
        values = list(self.performance_history)
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        return slope


class ContinualLearningManager:
    """持续学习管理器 - 防止灾难性遗忘"""
    
    def __init__(
        self,
        agent,
        replay_buffer_size: int = 10000,
        ewc_lambda: float = 0.1
    ):
        self.agent = agent
        self.replay_buffer_size = replay_buffer_size
        self.ewc_lambda = ewc_lambda
        
        self.task_buffers = {}  # 各任务的经验回放
        self.current_task_id = None
        self.fisher_information = {}
        self.optimal_params = {}
    
    def set_task(self, task_id: str):
        """设置当前任务"""
        # 保存当前任务的Fisher信息
        if self.current_task_id:
            self._compute_fisher_information()
        
        self.current_task_id = task_id
        
        # 初始化任务缓冲区
        if task_id not in self.task_buffers:
            self.task_buffers[task_id] = ExperienceBuffer(
                self.replay_buffer_size,
                priority_sampling=True
            )
    
    def add_experience(self, experience: Dict):
        """添加经验"""
        if self.current_task_id:
            self.task_buffers[self.current_task_id].add(experience)
    
    def _compute_fisher_information(self):
        """计算Fisher信息矩阵"""
        # 简化的EWC实现
        if hasattr(self.agent, 'actor'):
            for name, param in self.agent.actor.named_parameters():
                if param.requires_grad:
                    self.fisher_information[name] = param.grad ** 2 if param.grad else torch.zeros_like(param)
                    self.optimal_params[name] = param.data.clone()
    
    def get_ewc_loss(self) -> torch.Tensor:
        """获取EWC正则化损失"""
        if not self.fisher_information:
            return torch.tensor(0.0)
        
        loss = 0
        if hasattr(self.agent, 'actor'):
            for name, param in self.agent.actor.named_parameters():
                if name in self.fisher_information:
                    fisher = self.fisher_information[name]
                    optimal = self.optimal_params[name]
                    loss += (fisher * (param - optimal) ** 2).sum()
        
        return self.ewc_lambda * loss
    
    def rehearsal_update(self, batch_size: int = 32):
        """使用 rehearsal 进行更新"""
        # 从所有任务中采样
        all_experiences = []
        for task_id, buffer in self.task_buffers.items():
            if task_id != self.current_task_id and len(buffer) > 0:
                samples = buffer.sample(min(batch_size // len(self.task_buffers), len(buffer)))
                all_experiences.extend(samples)
        
        return all_experiences


class FeedbackLoop:
    """反馈循环 - 从生产环境收集反馈"""
    
    def __init__(
        self,
        online_learner: OnlineLearner,
        feedback_processors: Optional[List[Callable]] = None
    ):
        self.online_learner = online_learner
        self.feedback_processors = feedback_processors or []
        
        self.feedback_queue = deque(maxlen=10000)
        self.processing_thread = None
        self.is_running = False
    
    def start(self):
        """启动反馈处理循环"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.start()
    
    def stop(self):
        """停止反馈处理"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
    
    def submit_feedback(self, feedback: Dict):
        """提交用户反馈"""
        self.feedback_queue.append(feedback)
    
    def _process_loop(self):
        """处理循环"""
        while self.is_running:
            if self.feedback_queue:
                feedback = self.feedback_queue.popleft()
                
                # 处理反馈
                for processor in self.feedback_processors:
                    feedback = processor(feedback)
                
                # 转换为学习经验
                self._convert_to_experience(feedback)
            
            time.sleep(0.1)
    
    def _convert_to_experience(self, feedback: Dict):
        """将反馈转换为学习经验"""
        if 'state' in feedback and 'action' in feedback and 'reward' in feedback:
            self.online_learner.observe(
                state=feedback['state'],
                action=feedback['action'],
                reward=feedback['reward'],
                next_state=feedback.get('next_state', feedback['state']),
                done=feedback.get('done', False),
                info=feedback.get('info', {})
            )
