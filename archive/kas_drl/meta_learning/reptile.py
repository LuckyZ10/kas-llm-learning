"""
KAS Meta-Learning - Reptile
Reptile算法实现 - MAML的一阶近似，更简单高效
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import copy


@dataclass
class ReptileConfig:
    """Reptile配置"""
    inner_lr: float = 0.01
    meta_lr: float = 0.1
    inner_steps: int = 5
    meta_batch_size: int = 10
    epsilon: float = 0.1  # 元学习率衰减


class Reptile:
    """Reptile元学习器"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[ReptileConfig] = None
    ):
        self.config = config or ReptileConfig()
        self.model = model
    
    def meta_train_step(
        self,
        task_sampler,
        loss_fn: Optional[callable] = None
    ) -> Dict[str, float]:
        """
        执行一次元训练步骤
        
        Args:
            task_sampler: 任务采样器，sample_task()返回(support_data, query_data)
            loss_fn: 损失函数，默认MSE
        
        Returns:
            metrics: 训练指标
        """
        if loss_fn is None:
            loss_fn = nn.functional.mse_loss
        
        # 保存初始权重
        initial_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        
        task_losses = []
        
        for _ in range(self.config.meta_batch_size):
            # 采样任务
            support_data, query_data = task_sampler.sample_task()
            support_x, support_y = support_data
            
            # 克隆模型进行内循环训练
            task_model = copy.deepcopy(self.model)
            optimizer = optim.SGD(task_model.parameters(), lr=self.config.inner_lr)
            
            # 内循环优化
            for _ in range(self.config.inner_steps):
                predictions = task_model(support_x)
                loss = loss_fn(predictions, support_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 计算查询损失（用于监控）
            query_x, query_y = query_data
            with torch.no_grad():
                query_pred = task_model(query_x)
                query_loss = loss_fn(query_pred, query_y)
                task_losses.append(query_loss.item())
            
            # 更新元模型（向任务模型移动）
            with torch.no_grad():
                for (name, param), (_, task_param) in zip(
                    self.model.named_parameters(),
                    task_model.named_parameters()
                ):
                    param.data += self.config.meta_lr * (task_param.data - param.data) / self.config.meta_batch_size
        
        return {
            'mean_task_loss': np.mean(task_losses),
            'std_task_loss': np.std(task_losses)
        }
    
    def adapt(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        num_steps: Optional[int] = None,
        lr: Optional[float] = None
    ) -> nn.Module:
        """
        适应到新任务
        
        Args:
            support_data: (support_x, support_y)
            num_steps: 适应步数
            lr: 学习率
        
        Returns:
            adapted_model: 适应后的模型
        """
        if num_steps is None:
            num_steps = self.config.inner_steps
        if lr is None:
            lr = self.config.inner_lr
        
        support_x, support_y = support_data
        
        # 克隆模型
        adapted_model = copy.deepcopy(self.model)
        optimizer = optim.SGD(adapted_model.parameters(), lr=lr)
        
        # 微调
        for _ in range(num_steps):
            predictions = adapted_model(support_x)
            loss = nn.functional.mse_loss(predictions, support_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return adapted_model
    
    def train(
        self,
        task_sampler,
        num_iterations: int = 1000,
        eval_interval: int = 100
    ) -> Dict[str, List]:
        """
        训练Reptile
        
        Args:
            task_sampler: 任务采样器
            num_iterations: 迭代次数
            eval_interval: 评估间隔
        
        Returns:
            history: 训练历史
        """
        history = {
            'task_losses': [],
            'eval_losses': []
        }
        
        for iteration in range(num_iterations):
            metrics = self.meta_train_step(task_sampler)
            history['task_losses'].append(metrics['mean_task_loss'])
            
            # 衰减元学习率
            self.config.meta_lr *= (1 - self.config.epsilon)
            
            if iteration % eval_interval == 0:
                # 评估
                eval_loss = self._evaluate(task_sampler)
                history['eval_losses'].append(eval_loss)
                print(f"Iteration {iteration}: Task Loss = {metrics['mean_task_loss']:.4f}, "
                      f"Eval Loss = {eval_loss:.4f}")
        
        return history
    
    def _evaluate(self, task_sampler, num_eval_tasks: int = 10) -> float:
        """评估元学习效果"""
        eval_losses = []
        
        for _ in range(num_eval_tasks):
            support_data, query_data = task_sampler.sample_task()
            
            # 适应
            adapted = self.adapt(support_data, num_steps=5)
            
            # 评估
            query_x, query_y = query_data
            with torch.no_grad():
                predictions = adapted(query_x)
                loss = nn.functional.mse_loss(predictions, query_y)
                eval_losses.append(loss.item())
        
        return np.mean(eval_losses)


class ReptileAgentPolicy:
    """使用Reptile的Agent策略"""
    
    def __init__(self, base_policy: nn.Module, config: Optional[ReptileConfig] = None):
        self.reptile = Reptile(base_policy, config)
        self.adapted_policies = {}
    
    def meta_train(self, task_sampler, num_iterations: int = 1000):
        """元训练"""
        return self.reptile.train(task_sampler, num_iterations)
    
    def adapt_to_project(self, project_id: str, support_data, num_steps: int = 10):
        """适应到项目"""
        adapted = self.reptile.adapt(support_data, num_steps)
        self.adapted_policies[project_id] = adapted
        return adapted
    
    def get_policy(self, project_id: Optional[str] = None):
        """获取策略"""
        if project_id and project_id in self.adapted_policies:
            return self.adapted_policies[project_id]
        return self.reptile.model
