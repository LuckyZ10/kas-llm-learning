"""
KAS Integration - LLM Client Adapter
与现有LLMClient系统集成
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import time


@dataclass
class LLMConfig:
    """LLM配置"""
    model_name: str = "default"
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: float = 30.0


class LLMClientAdapter:
    """LLMClient适配器"""
    
    def __init__(
        self,
        llm_client,
        drl_agent,
        state_encoder,
        config: Optional[LLMConfig] = None
    ):
        """
        Args:
            llm_client: 现有的LLMClient实例
            drl_agent: DRL Agent实例
            state_encoder: 状态编码器
            config: 配置
        """
        self.llm_client = llm_client
        self.drl_agent = drl_agent
        self.state_encoder = state_encoder
        self.config = config or LLMConfig()
        
        # 追踪状态
        self.current_state = None
        self.action_history = []
        self.feedback_history = []
    
    def generate(
        self,
        prompt: str,
        context: Optional[Dict] = None,
        use_drl: bool = True
    ) -> Dict[str, Any]:
        """
        生成响应（增强版）
        
        Args:
            prompt: 输入提示
            context: 上下文信息
            use_drl: 是否使用DRL优化
        
        Returns:
            response: 包含生成结果和元信息
        """
        start_time = time.time()
        
        if use_drl and self.current_state is not None:
            # 使用DRL选择最优参数
            action = self.drl_agent.select_action(self.current_state, deterministic=True)
            llm_params = self._action_to_llm_params(action)
        else:
            llm_params = self._get_default_params()
        
        # 调整prompt（如果需要）
        adjusted_prompt = self._adjust_prompt(prompt, action if use_drl else None)
        
        # 调用LLM
        try:
            llm_response = self.llm_client.generate(
                prompt=adjusted_prompt,
                **llm_params
            )
            
            response_time = time.time() - start_time
            
            result = {
                'text': llm_response,
                'params': llm_params,
                'adjusted_prompt': adjusted_prompt,
                'response_time': response_time,
                'success': True
            }
            
        except Exception as e:
            result = {
                'text': "",
                'params': llm_params,
                'error': str(e),
                'success': False
            }
        
        # 记录动作
        if use_drl:
            self.action_history.append({
                'action': action if use_drl else None,
                'params': llm_params,
                'timestamp': time.time()
            })
        
        return result
    
    def _action_to_llm_params(self, action: np.ndarray) -> Dict[str, Any]:
        """将动作转换为LLM参数"""
        # 解析动作
        # action: [prompt_adjustment(9), template(7), params(5)]
        
        params = action[16:21]  # 后5个是参数
        
        return {
            'temperature': float(np.clip(params[0] * 2, 0, 2)),
            'max_tokens': int(np.clip(params[1] * 4000, 1, 4096)),
            'top_p': float(np.clip(params[2], 0, 1)),
            'frequency_penalty': float(np.clip(params[3] * 4 - 2, -2, 2)),
            'presence_penalty': float(np.clip(params[4] * 4 - 2, -2, 2))
        }
    
    def _get_default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
            'top_p': self.config.top_p,
            'frequency_penalty': self.config.frequency_penalty,
            'presence_penalty': self.config.presence_penalty
        }
    
    def _adjust_prompt(self, prompt: str, action: Optional[np.ndarray]) -> str:
        """根据动作调整prompt"""
        if action is None:
            return prompt
        
        # 解析prompt调整
        prompt_adj = action[:9]
        adj_type = np.argmax(prompt_adj[:8])
        strength = prompt_adj[8]
        
        # 根据调整类型修改prompt
        if adj_type == 0 and strength > 0.5:  # EXPAND
            prompt = self._expand_prompt(prompt, strength)
        elif adj_type == 1 and strength > 0.5:  # CONDENSE
            prompt = self._condense_prompt(prompt, strength)
        elif adj_type == 2 and strength > 0.5:  # ADD_EXAMPLE
            prompt = self._add_example(prompt)
        
        return prompt
    
    def _expand_prompt(self, prompt: str, strength: float) -> str:
        """扩展prompt"""
        expansion = f"\n\nPlease provide detailed explanations and step-by-step reasoning."
        return prompt + expansion
    
    def _condense_prompt(self, prompt: str, strength: float) -> str:
        """精简prompt（这里只是示例）"""
        return prompt
    
    def _add_example(self, prompt: str) -> str:
        """添加示例"""
        example = "\n\nExample:\nInput: ...\nOutput: ..."
        return prompt + example
    
    def update_state(self, agent_state, task_features, user_feedback=None):
        """更新当前状态"""
        self.current_state = self.state_encoder.encode(
            agent_state,
            task_features,
            user_feedback
        )
    
    def feedback(self, rating: float, metadata: Optional[Dict] = None):
        """
        接收用户反馈
        
        Args:
            rating: 用户评分 1-5
            metadata: 额外元信息
        """
        self.feedback_history.append({
            'rating': rating,
            'timestamp': time.time(),
            'metadata': metadata or {}
        })
        
        # 如果有足够的历史，可以更新DRL Agent
        if len(self.feedback_history) >= 10:
            self._update_agent()
    
    def _update_agent(self):
        """根据反馈更新Agent"""
        # 计算最近反馈的平均奖励
        recent_feedback = self.feedback_history[-10:]
        avg_rating = np.mean([f['rating'] for f in recent_feedback])
        
        # 转换为奖励 (-1 到 1)
        reward = (avg_rating - 3) / 2.0
        
        # 如果Agent支持在线学习
        if hasattr(self.drl_agent, 'store_transition'):
            # 这里需要实际的next_state和done
            # 简化处理
            pass


class CompatibleLLMClient:
    """兼容层 - 让DRL增强的Agent可以替换原有LLMClient"""
    
    def __init__(self, enhanced_adapter: LLMClientAdapter):
        self.adapter = enhanced_adapter
    
    def generate(self, prompt: str, **kwargs) -> str:
        """兼容原有generate接口"""
        result = self.adapter.generate(prompt, context=kwargs, use_drl=True)
        return result.get('text', '')
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """兼容chat接口"""
        # 将messages转换为prompt
        prompt = self._messages_to_prompt(messages)
        result = self.adapter.generate(prompt, context=kwargs, use_drl=True)
        return result.get('text', '')
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """将消息列表转换为prompt"""
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            prompt_parts.append(f"{role}: {content}")
        return "\n".join(prompt_parts)


class LLMClientWrapper:
    """LLMClient包装器 - 支持多个后端"""
    
    def __init__(self):
        self.backends = {}
        self.default_backend = None
    
    def register_backend(self, name: str, client, is_default: bool = False):
        """注册后端"""
        self.backends[name] = client
        if is_default or self.default_backend is None:
            self.default_backend = name
    
    def generate(self, prompt: str, backend: Optional[str] = None, **kwargs) -> str:
        """生成"""
        backend_name = backend or self.default_backend
        
        if backend_name not in self.backends:
            raise ValueError(f"Unknown backend: {backend_name}")
        
        client = self.backends[backend_name]
        return client.generate(prompt, **kwargs)
    
    def route_by_complexity(self, prompt: str, complexity: float, **kwargs) -> str:
        """根据复杂度路由到不同后端"""
        if complexity < 0.3:
            # 简单任务用轻量级模型
            return self.generate(prompt, backend='light', **kwargs)
        elif complexity < 0.7:
            return self.generate(prompt, backend='medium', **kwargs)
        else:
            return self.generate(prompt, backend='powerful', **kwargs)
