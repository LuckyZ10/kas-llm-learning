"""
KAS Meta-Learning - Model-Agnostic Meta-Learning (MAML)
MAML算法实现用于快速适应新项目
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import copy


@dataclass
class MAMLConfig:
    """MAML配置"""
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    num_inner_steps: int = 5
    num_tasks_per_batch: int = 4
    first_order: bool = False  # 是否使用一阶近似(FO-MAML)


class MAML:
    """MAML元学习器"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[MAMLConfig] = None
    ):
        self.config = config or MAMLConfig()
        self.model = model
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=self.config.outer_lr)
    
    def inner_loop(
        self,
        task_data: Tuple[torch.Tensor, torch.Tensor],
        num_steps: Optional[int] = None
    ) -> Tuple[nn.Module, List[float]]:
        """
        内循环适应
        
        Args:
            task_data: (support_x, support_y) 任务的支持集
            num_steps: 内循环步数
        
        Returns:
            adapted_model: 适应后的模型
            losses: 内循环损失历史
        """
        if num_steps is None:
            num_steps = self.config.num_inner_steps
        
        support_x, support_y = task_data
        
        # 克隆模型
        adapted_model = copy.deepcopy(self.model)
        adapted_optimizer = optim.SGD(adapted_model.parameters(), lr=self.config.inner_lr)
        
        losses = []
        
        for _ in range(num_steps):
            # 前向传播
            predictions = adapted_model(support_x)
            loss = nn.functional.mse_loss(predictions, support_y)
            
            losses.append(loss.item())
            
            # 内循环梯度下降
            adapted_optimizer.zero_grad()
            loss.backward()
            adapted_optimizer.step()
        
        return adapted_model, losses
    
    def outer_step(
        self,
        tasks: List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]
    ) -> Dict[str, float]:
        """
        外循环元更新
        
        Args:
            tasks: [(support_data, query_data), ...] 任务列表
        
        Returns:
            metrics: 训练指标
        """
        meta_loss = 0.0
        task_losses = []
        
        # 保存原始参数
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        for support_data, query_data in tasks:
            # 内循环适应
            adapted_model, inner_losses = self.inner_loop(support_data)
            
            # 查询集评估
            query_x, query_y = query_data
            query_predictions = adapted_model(query_x)
            query_loss = nn.functional.mse_loss(query_predictions, query_y)
            
            task_losses.append(query_loss.item())
            meta_loss += query_loss
        
        # 平均元损失
        meta_loss = meta_loss / len(tasks)
        
        # 元更新
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return {
            'meta_loss': meta_loss.item(),
            'mean_task_loss': np.mean(task_losses),
            'std_task_loss': np.std(task_losses)
        }
    
    def adapt_to_new_task(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        num_steps: Optional[int] = None
    ) -> nn.Module:
        """
        适应到新任务
        
        Args:
            support_data: (support_x, support_y)
            num_steps: 适应步数
        
        Returns:
            adapted_model: 适应后的模型
        """
        adapted_model, _ = self.inner_loop(support_data, num_steps)
        return adapted_model


class ProjectTaskSampler:
    """项目任务采样器"""
    
    def __init__(
        self,
        project_embeddings: torch.Tensor,
        project_labels: torch.Tensor,
        k_shot: int = 5,
        q_query: int = 15
    ):
        """
        Args:
            project_embeddings: [num_projects, embedding_dim]
            project_labels: [num_projects, label_dim]
            k_shot: 支持集样本数
            q_query: 查询集样本数
        """
        self.project_embeddings = project_embeddings
        self.project_labels = project_labels
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_projects = len(project_embeddings)
    
    def sample_task(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """采样一个任务"""
        # 随机选择k+q个样本
        indices = np.random.choice(self.num_projects, self.k_shot + self.q_query, replace=False)
        
        support_indices = indices[:self.k_shot]
        query_indices = indices[self.k_shot:]
        
        support_x = self.project_embeddings[support_indices]
        support_y = self.project_labels[support_indices]
        
        query_x = self.project_embeddings[query_indices]
        query_y = self.project_labels[query_indices]
        
        return (support_x, support_y), (query_x, query_y)
    
    def sample_batch(self, batch_size: int) -> List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
        """采样一批任务"""
        return [self.sample_task() for _ in range(batch_size)]


class MAMLAgentAdapter:
    """使用MAML快速适应Agent策略"""
    
    def __init__(
        self,
        base_agent,
        maml_config: Optional[MAMLConfig] = None
    ):
        self.base_agent = base_agent
        self.maml = MAML(base_agent.actor, maml_config)
        self.adapted_agents = {}
    
    def meta_train(
        self,
        task_sampler: ProjectTaskSampler,
        num_iterations: int = 1000
    ) -> Dict[str, List]:
        """元训练"""
        history = {
            'meta_losses': [],
            'task_losses': []
        }
        
        for iteration in range(num_iterations):
            # 采样任务批次
            tasks = task_sampler.sample_batch(self.maml.config.num_tasks_per_batch)
            
            # 外循环更新
            metrics = self.maml.outer_step(tasks)
            
            history['meta_losses'].append(metrics['meta_loss'])
            history['task_losses'].append(metrics['mean_task_loss'])
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Meta Loss = {metrics['meta_loss']:.4f}")
        
        return history
    
    def adapt_to_project(
        self,
        project_id: str,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        num_steps: int = 10
    ):
        """适应到特定项目"""
        adapted_actor = self.maml.adapt_to_new_task(support_data, num_steps)
        
        # 创建新的Agent实例
        adapted_agent = copy.deepcopy(self.base_agent)
        adapted_agent.actor = adapted_actor
        
        self.adapted_agents[project_id] = adapted_agent
        
        return adapted_agent
    
    def get_agent(self, project_id: str):
        """获取适应后的Agent"""
        if project_id in self.adapted_agents:
            return self.adapted_agents[project_id]
        return self.base_agent
