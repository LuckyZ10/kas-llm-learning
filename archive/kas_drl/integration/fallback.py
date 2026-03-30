"""
KAS Integration - Fallback Strategies
降级策略 - 确保系统稳定性
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time


class FallbackLevel(Enum):
    """降级级别"""
    NONE = 0          # 正常运行
    LIGHT = 1         # 轻度降级 - 减少功能
    MODERATE = 2      # 中度降级 - 使用备用策略
    SEVERE = 3        # 严重降级 - 回退到基础版本
    EMERGENCY = 4     # 紧急降级 - 仅保留核心功能


@dataclass
class FallbackConfig:
    """降级配置"""
    enable_auto_fallback: bool = True
    error_threshold: int = 5  # 连续错误阈值
    latency_threshold: float = 10.0  # 延迟阈值(秒)
    recovery_time: float = 60.0  # 恢复检查间隔
    
    # 各级别配置
    light_fallback_params: Dict = None
    moderate_fallback_params: Dict = None
    severe_fallback_params: Dict = None
    
    def __post_init__(self):
        if self.light_fallback_params is None:
            self.light_fallback_params = {
                'use_cache': True,
                'simplified_prompt': True,
                'reduce_max_tokens': 0.8
            }
        if self.moderate_fallback_params is None:
            self.moderate_fallback_params = {
                'use_rule_based': True,
                'disable_drl': True,
                'reduce_max_tokens': 0.5
            }
        if self.severe_fallback_params is None:
            self.severe_fallback_params = {
                'use_static_response': True,
                'disable_all_features': True
            }


class FallbackManager:
    """降级管理器"""
    
    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
        self.current_level = FallbackLevel.NONE
        
        # 监控指标
        self.error_count = 0
        self.latency_history = []
        self.last_error_time = None
        self.last_recovery_check = time.time()
        
        # 回调
        self.fallback_callbacks = []
        self.recovery_callbacks = []
    
    def on_error(self, error: Exception, context: Optional[Dict] = None):
        """处理错误"""
        self.error_count += 1
        self.last_error_time = time.time()
        
        # 检查是否需要降级
        if self.config.enable_auto_fallback:
            self._check_fallback_needed()
    
    def on_latency(self, latency: float):
        """记录延迟"""
        self.latency_history.append(latency)
        
        # 保持历史在合理范围
        if len(self.latency_history) > 100:
            self.latency_history = self.latency_history[-100:]
        
        # 检查高延迟
        if self.config.enable_auto_fallback:
            avg_latency = np.mean(self.latency_history[-10:])
            if avg_latency > self.config.latency_threshold:
                self._increase_fallback_level()
    
    def _check_fallback_needed(self):
        """检查是否需要降级"""
        if self.error_count >= self.config.error_threshold:
            self._increase_fallback_level()
            self.error_count = 0  # 重置错误计数
    
    def _increase_fallback_level(self):
        """提升降级级别"""
        if self.current_level.value < FallbackLevel.EMERGENCY.value:
            old_level = self.current_level
            self.current_level = FallbackLevel(self.current_level.value + 1)
            
            print(f"Fallback level increased: {old_level.name} -> {self.current_level.name}")
            
            # 执行回调
            for callback in self.fallback_callbacks:
                callback(self.current_level, old_level)
    
    def check_recovery(self):
        """检查是否可以恢复"""
        if self.current_level == FallbackLevel.NONE:
            return
        
        if time.time() - self.last_recovery_check < self.config.recovery_time:
            return
        
        self.last_recovery_check = time.time()
        
        # 检查指标是否恢复正常
        if self.error_count == 0 and len(self.latency_history) > 0:
            avg_latency = np.mean(self.latency_history[-10:])
            if avg_latency < self.config.latency_threshold * 0.5:
                self._decrease_fallback_level()
    
    def _decrease_fallback_level(self):
        """降低降级级别"""
        if self.current_level.value > FallbackLevel.NONE.value:
            old_level = self.current_level
            self.current_level = FallbackLevel(self.current_level.value - 1)
            
            print(f"Fallback level decreased: {old_level.name} -> {self.current_level.name}")
            
            # 执行回调
            for callback in self.recovery_callbacks:
                callback(self.current_level, old_level)
    
    def get_current_params(self) -> Dict[str, Any]:
        """获取当前降级级别的参数"""
        if self.current_level == FallbackLevel.LIGHT:
            return self.config.light_fallback_params
        elif self.current_level == FallbackLevel.MODERATE:
            return self.config.moderate_fallback_params
        elif self.current_level in [FallbackLevel.SEVERE, FallbackLevel.EMERGENCY]:
            return self.config.severe_fallback_params
        else:
            return {}
    
    def register_fallback_callback(self, callback: Callable):
        """注册降级回调"""
        self.fallback_callbacks.append(callback)
    
    def register_recovery_callback(self, callback: Callable):
        """注册恢复回调"""
        self.recovery_callbacks.append(callback)
    
    def reset(self):
        """重置降级状态"""
        self.current_level = FallbackLevel.NONE
        self.error_count = 0
        self.latency_history = []


class DRLFallbackWrapper:
    """DRL Agent降级包装器"""
    
    def __init__(
        self,
        drl_agent,
        fallback_policy: Optional[Callable] = None
    ):
        self.drl_agent = drl_agent
        self.fallback_policy = fallback_policy or self._default_fallback_policy
        self.fallback_manager = FallbackManager()
        
        # 注册回调
        self.fallback_manager.register_fallback_callback(self._on_fallback)
    
    def select_action(self, state, deterministic: bool = False):
        """选择动作（带降级）"""
        try:
            # 检查降级级别
            if self.fallback_manager.current_level.value >= FallbackLevel.MODERATE.value:
                # 使用降级策略
                return self.fallback_policy(state)
            
            # 正常调用DRL Agent
            action = self.drl_agent.select_action(state, deterministic)
            
            # 记录成功
            self.fallback_manager.check_recovery()
            
            return action
            
        except Exception as e:
            # 记录错误
            self.fallback_manager.on_error(e)
            
            # 返回降级策略
            return self.fallback_policy(state)
    
    def _default_fallback_policy(self, state) -> np.ndarray:
        """默认降级策略"""
        # 返回中等参数值
        return np.array([
            # Prompt调整 - 中性
            0, 0, 0, 0, 0, 0, 0, 0, 0.5,
            # 模板 - 选择标准模板
            0, 0, 1, 0, 0, 0, 0.5,
            # 参数 - 中等值
            0.35,  # temperature ~0.7
            0.5,   # max_tokens ~2000
            0.5,   # top_p
            0.5,   # frequency_penalty ~0
            0.5    # presence_penalty ~0
        ])
    
    def _on_fallback(self, new_level, old_level):
        """降级回调"""
        print(f"DRL Agent fallback triggered: {old_level.name} -> {new_level.name}")


class CircuitBreaker:
    """熔断器模式"""
    
    class State(Enum):
        CLOSED = "closed"       # 正常
        OPEN = "open"          # 熔断
        HALF_OPEN = "half_open"  # 半开
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = self.State.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """调用函数（带熔断保护）"""
        if self.state == self.State.OPEN:
            if self._should_attempt_reset():
                self.state = self.State.HALF_OPEN
                self.half_open_calls = 0
            else:
                raise CircuitBreakerOpen("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """检查是否应该尝试重置"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """成功处理"""
        if self.state == self.State.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = self.State.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """失败处理"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == self.State.HALF_OPEN:
            self.state = self.State.OPEN
        elif self.failure_count >= self.failure_threshold:
            self.state = self.State.OPEN


class CircuitBreakerOpen(Exception):
    """熔断器打开异常"""
    pass


class GracefulDegradation:
    """优雅降级管理"""
    
    def __init__(self):
        self.feature_flags = {
            'drl_optimization': True,
            'meta_learning': True,
            'online_learning': True,
            'complex_prompt_adjustment': True,
            'detailed_logging': True
        }
        
        self.degradation_levels = {
            FallbackLevel.NONE: [],
            FallbackLevel.LIGHT: ['detailed_logging'],
            FallbackLevel.MODERATE: ['complex_prompt_adjustment', 'online_learning', 'detailed_logging'],
            FallbackLevel.SEVERE: ['meta_learning', 'complex_prompt_adjustment', 'online_learning', 'detailed_logging'],
            FallbackLevel.EMERGENCY: ['drl_optimization', 'meta_learning', 'complex_prompt_adjustment', 'online_learning', 'detailed_logging']
        }
    
    def set_fallback_level(self, level: FallbackLevel):
        """设置降级级别"""
        # 重置所有功能
        for feature in self.feature_flags:
            self.feature_flags[feature] = True
        
        # 禁用相应级别的功能
        features_to_disable = self.degradation_levels.get(level, [])
        for feature in features_to_disable:
            self.feature_flags[feature] = False
        
        print(f"Graceful degradation: {level.name}")
        print(f"Active features: {[f for f, v in self.feature_flags.items() if v]}")
    
    def is_enabled(self, feature: str) -> bool:
        """检查功能是否启用"""
        return self.feature_flags.get(feature, False)
