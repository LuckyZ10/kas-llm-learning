"""
KAS Integration - Deployment Tools
渐进式部署工具
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
import time
from pathlib import Path


class DeploymentStrategy(Enum):
    """部署策略"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    SHADOW = "shadow"
    A_B_TEST = "a_b_test"


@dataclass
class DeploymentConfig:
    """部署配置"""
    strategy: DeploymentStrategy = DeploymentStrategy.CANARY
    canary_percentage: float = 0.1  # 金丝雀百分比
    canary_duration: float = 3600   # 金丝雀持续时间(秒)
    rollback_threshold: float = 0.8  # 回滚阈值
    evaluation_window: int = 100    # 评估窗口大小
    
    # A/B测试配置
    a_b_split: float = 0.5
    metric_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metric_weights is None:
            self.metric_weights = {
                'user_satisfaction': 0.4,
                'response_quality': 0.3,
                'latency': 0.2,
                'error_rate': 0.1
            }


class CanaryDeployment:
    """金丝雀部署"""
    
    def __init__(
        self,
        old_agent,
        new_agent,
        config: Optional[DeploymentConfig] = None
    ):
        self.old_agent = old_agent
        self.new_agent = new_agent
        self.config = config or DeploymentConfig()
        
        self.phase = "init"  # init, canary, full, rollback
        self.start_time = None
        self.canary_count = 0
        self.old_count = 0
        
        # 指标收集
        self.canary_metrics = []
        self.old_metrics = []
    
    def start(self):
        """开始金丝雀部署"""
        self.phase = "canary"
        self.start_time = time.time()
        print(f"Starting canary deployment with {self.config.canary_percentage*100}% traffic")
    
    def route_request(self, state) -> Any:
        """路由请求"""
        if self.phase == "canary":
            # 按比例路由
            if np.random.rand() < self.config.canary_percentage:
                self.canary_count += 1
                return self.new_agent.select_action(state)
            else:
                self.old_count += 1
                return self.old_agent.select_action(state)
        elif self.phase == "full":
            return self.new_agent.select_action(state)
        elif self.phase == "rollback":
            return self.old_agent.select_action(state)
        else:
            return self.old_agent.select_action(state)
    
    def record_metric(self, is_canary: bool, metric: Dict):
        """记录指标"""
        if is_canary:
            self.canary_metrics.append(metric)
        else:
            self.old_metrics.append(metric)
    
    def evaluate(self) -> Dict[str, Any]:
        """评估金丝雀效果"""
        if len(self.canary_metrics) < self.config.evaluation_window:
            return {'status': 'insufficient_data'}
        
        # 计算指标
        canary_score = self._calculate_score(self.canary_metrics[-self.config.evaluation_window:])
        old_score = self._calculate_score(self.old_metrics[-self.config.evaluation_window:])
        
        comparison = canary_score / (old_score + 1e-8)
        
        result = {
            'canary_score': canary_score,
            'old_score': old_score,
            'comparison': comparison,
            'status': 'evaluating'
        }
        
        # 检查是否超时
        if time.time() - self.start_time > self.config.canary_duration:
            if comparison >= self.config.rollback_threshold:
                result['status'] = 'promote'
                self.phase = 'full'
                print("Canary evaluation passed! Promoting to full deployment.")
            else:
                result['status'] = 'rollback'
                self.phase = 'rollback'
                print(f"Canary evaluation failed ({comparison:.2f}). Rolling back.")
        
        return result
    
    def _calculate_score(self, metrics: List[Dict]) -> float:
        """计算综合评分"""
        if not metrics:
            return 0.0
        
        scores = []
        for metric in metrics:
            score = 0
            for key, weight in self.config.metric_weights.items():
                if key in metric:
                    score += metric[key] * weight
            scores.append(score)
        
        return np.mean(scores)


class ABTestDeployment:
    """A/B测试部署"""
    
    def __init__(
        self,
        control_agent,
        treatment_agent,
        config: Optional[DeploymentConfig] = None
    ):
        self.control_agent = control_agent
        self.treatment_agent = treatment_agent
        self.config = config or DeploymentConfig()
        
        self.user_assignments = {}  # user_id -> group
        
        # 指标
        self.control_metrics = []
        self.treatment_metrics = []
    
    def assign_user(self, user_id: str) -> str:
        """分配用户到测试组"""
        if user_id not in self.user_assignments:
            # 随机分配
            group = 'treatment' if np.random.rand() < self.config.a_b_split else 'control'
            self.user_assignments[user_id] = group
        
        return self.user_assignments[user_id]
    
    def get_agent(self, user_id: str):
        """获取用户的Agent"""
        group = self.assign_user(user_id)
        return self.treatment_agent if group == 'treatment' else self.control_agent
    
    def record_outcome(self, user_id: str, outcome: Dict):
        """记录结果"""
        group = self.user_assignments.get(user_id)
        if group == 'treatment':
            self.treatment_metrics.append(outcome)
        elif group == 'control':
            self.control_metrics.append(outcome)
    
    def analyze(self) -> Dict[str, Any]:
        """分析A/B测试结果"""
        if len(self.control_metrics) < 30 or len(self.treatment_metrics) < 30:
            return {'status': 'insufficient_data'}
        
        # 计算统计显著性
        control_rewards = [m.get('reward', 0) for m in self.control_metrics]
        treatment_rewards = [m.get('reward', 0) for m in self.treatment_metrics]
        
        # 简单t检验
        from statistics import mean, stdev
        
        control_mean = mean(control_rewards)
        treatment_mean = mean(treatment_rewards)
        
        control_std = stdev(control_rewards) if len(control_rewards) > 1 else 0
        treatment_std = stdev(treatment_rewards) if len(treatment_rewards) > 1 else 0
        
        # 效应量
        pooled_std = np.sqrt((control_std**2 + treatment_std**2) / 2)
        effect_size = (treatment_mean - control_mean) / (pooled_std + 1e-8)
        
        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'effect_size': effect_size,
            'control_n': len(control_rewards),
            'treatment_n': len(treatment_rewards),
            'winner': 'treatment' if effect_size > 0.2 else 'control' if effect_size < -0.2 else 'tie'
        }


class ShadowDeployment:
    """影子部署 - 复制流量到新版本但不影响用户"""
    
    def __init__(self, production_agent, shadow_agent):
        self.production_agent = production_agent
        self.shadow_agent = shadow_agent
        
        self.shadow_results = []
        self.comparison_results = []
    
    def process(self, state) -> tuple:
        """处理请求（同时调用两个版本）"""
        # 生产环境响应
        production_action = self.production_agent.select_action(state)
        
        # 影子响应（异步）
        shadow_action = self.shadow_agent.select_action(state)
        
        # 记录差异
        diff = np.abs(production_action - shadow_action).mean()
        self.comparison_results.append(diff)
        
        return production_action, shadow_action
    
    def get_divergence_stats(self) -> Dict[str, float]:
        """获取差异统计"""
        if not self.comparison_results:
            return {}
        
        return {
            'mean_divergence': np.mean(self.comparison_results),
            'max_divergence': np.max(self.comparison_results),
            'divergence_std': np.std(self.comparison_results)
        }


class ModelRegistry:
    """模型注册表"""
    
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.models = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """加载注册表"""
        registry_file = self.registry_path / "registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """保存注册表"""
        registry_file = self.registry_path / "registry.json"
        with open(registry_file, 'w') as f:
            json.dump(self.models, f, indent=2)
    
    def register(
        self,
        model_name: str,
        version: str,
        model_path: str,
        metrics: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """注册模型"""
        if model_name not in self.models:
            self.models[model_name] = {}
        
        self.models[model_name][version] = {
            'path': model_path,
            'metrics': metrics or {},
            'metadata': metadata or {},
            'registered_at': time.time(),
            'stage': 'staging'  # staging, production, archived
        }
        
        self._save_registry()
        print(f"Registered {model_name} v{version}")
    
    def promote(self, model_name: str, version: str, stage: str):
        """提升模型阶段"""
        if model_name in self.models and version in self.models[model_name]:
            self.models[model_name][version]['stage'] = stage
            self._save_registry()
            print(f"Promoted {model_name} v{version} to {stage}")
    
    def get_model(self, model_name: str, stage: str = 'production') -> Optional[str]:
        """获取指定阶段的模型"""
        if model_name not in self.models:
            return None
        
        for version, info in self.models[model_name].items():
            if info['stage'] == stage:
                return info['path']
        
        return None
    
    def list_models(self, model_name: Optional[str] = None) -> Dict:
        """列出模型"""
        if model_name:
            return self.models.get(model_name, {})
        return self.models


class DeploymentPipeline:
    """部署流水线"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.stages = ['train', 'evaluate', 'canary', 'production']
        self.current_stage_idx = 0
    
    def run(self, model_name: str, version: str, agent, eval_env) -> bool:
        """运行部署流水线"""
        # 1. 训练（假设已完成）
        print(f"Stage 1/4: Training complete for {model_name} v{version}")
        
        # 2. 评估
        print(f"Stage 2/4: Evaluating...")
        eval_metrics = self._evaluate(agent, eval_env)
        
        if eval_metrics['mean_reward'] < 0:
            print("Evaluation failed!")
            return False
        
        # 注册模型
        model_path = f"./models/{model_name}_{version}.pt"
        agent.save(model_path)
        self.registry.register(model_name, version, model_path, eval_metrics)
        
        # 3. 金丝雀部署
        print(f"Stage 3/4: Canary deployment...")
        canary = self._run_canary(agent, eval_env)
        
        if not canary:
            print("Canary deployment failed!")
            return False
        
        # 4. 生产部署
        print(f"Stage 4/4: Production deployment...")
        self.registry.promote(model_name, version, 'production')
        
        print(f"Deployment complete: {model_name} v{version}")
        return True
    
    def _evaluate(self, agent, env, num_episodes: int = 10) -> Dict:
        """评估模型"""
        rewards = []
        
        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            
            for _ in range(500):
                action = agent.select_action(state, deterministic=True)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }
    
    def _run_canary(self, new_agent, env, duration: int = 60) -> bool:
        """运行金丝雀测试（简化版）"""
        # 模拟金丝雀测试
        print(f"Running canary for {duration}s...")
        time.sleep(1)  # 简化
        return True
