"""
KAS LLM Client - 多提供商 LLM 客户端
支持 OpenAI、DeepSeek、Kimi、智谱 AI 等
"""
import os
import time
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM 响应"""
    content: str
    model: str
    usage: Dict[str, int]
    latency_ms: float
    provider: str


class LLMClient:
    """
    通用 LLM 客户端
    
    支持：
    - OpenAI (GPT-4, GPT-3.5)
    - DeepSeek (deepseek-chat)
    - Kimi (kimi-chat)
    - 智谱 AI (使用 Anthropic 兼容接口)
    """
    
    def __init__(
        self,
        api_key: str,
        provider: str = "openai",
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0,  # 默认5分钟
    ):
        self.api_key = api_key
        self.provider = provider.lower()
        self.base_url = base_url
        self.timeout = timeout
        
        # 默认模型映射
        model_map = {
            'openai': 'gpt-3.5-turbo',
            'deepseek': 'deepseek-chat',
            'kimi': 'moonshot-v1-8k',
            'zhipu': 'claude-3-opus-20240229',  # 智谱 Coding Plan 使用 Claude 模型名
        }
        self.model = model or model_map.get(self.provider, 'gpt-3.5-turbo')
        
        # 客户端实例
        self._client = None
    
    def _get_client(self):
        """获取或创建 LLM 客户端"""
        if self._client is not None:
            return self._client
        
        if self.provider in ['openai', 'deepseek', 'kimi']:
            # 使用 OpenAI SDK
            try:
                import openai
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
            
            client_kwargs = {'api_key': self.api_key}
            if self.base_url:
                client_kwargs['base_url'] = self.base_url
            
            self._client = openai.OpenAI(**client_kwargs)
            
        elif self.provider == 'zhipu':
            # 智谱 AI 使用 Anthropic 兼容接口
            try:
                import anthropic
            except ImportError:
                logger.warning("anthropic SDK 未安装，尝试使用 openai SDK")
                # 回退到 openai SDK
                import openai
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url or "https://open.bigmodel.cn/api/anthropic"
                )
                return self._client
            
            # 使用 anthropic SDK
            client_kwargs = {'api_key': self.api_key}
            if self.base_url:
                client_kwargs['base_url'] = self.base_url
            else:
                client_kwargs['base_url'] = "https://open.bigmodel.cn/api/anthropic"
            
            # 设置超时
            client_kwargs['timeout'] = self.timeout
            
            self._client = anthropic.Anthropic(**client_kwargs)
        
        return self._client
    
    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """
        发送对话请求
        
        Args:
            system_prompt: 系统提示词
            user_message: 用户消息
            temperature: 创造性参数
            max_tokens: 最大 token 数
            **kwargs: 其他参数
        
        Returns:
            LLMResponse 对象
        """
        start_time = time.time()
        client = self._get_client()
        
        try:
            if self.provider == 'zhipu' and hasattr(client, 'messages'):
                # 使用 Anthropic SDK (智谱 AI)
                response = client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_message}
                    ],
                    timeout=self.timeout,
                    **kwargs
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                return LLMResponse(
                    content=response.content[0].text,
                    model=self.model,
                    usage={
                        'prompt_tokens': response.usage.input_tokens,
                        'completion_tokens': response.usage.output_tokens,
                        'total_tokens': response.usage.input_tokens + response.usage.output_tokens
                    },
                    latency_ms=latency_ms,
                    provider=self.provider
                )
            else:
                # 使用 OpenAI SDK
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
                
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout,
                    **kwargs
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                usage = response.usage
                return LLMResponse(
                    content=response.choices[0].message.content,
                    model=self.model,
                    usage={
                        'prompt_tokens': usage.prompt_tokens,
                        'completion_tokens': usage.completion_tokens,
                        'total_tokens': usage.total_tokens
                    },
                    latency_ms=latency_ms,
                    provider=self.provider
                )
                
        except Exception as e:
            logger.error(f"LLM API 调用失败: {e}")
            raise
    
    def chat_stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式对话
        
        Yields:
            文本片段
        """
        client = self._get_client()
        
        try:
            if self.provider == 'zhipu' and hasattr(client, 'messages'):
                # Anthropic 流式
                with client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_message}
                    ],
                    **kwargs
                ) as stream:
                    for text in stream.text_stream:
                        yield text
            else:
                # OpenAI 流式
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    **kwargs
                )
                
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                        
        except Exception as e:
            logger.error(f"流式 LLM API 调用失败: {e}")
            raise
    
    def is_available(self) -> bool:
        """检查 LLM 服务是否可用"""
        try:
            self.chat(
                system_prompt="Test",
                user_message="Hello",
                max_tokens=10
            )
            return True
        except Exception as e:
            logger.warning(f"LLM 服务不可用: {e}")
            return False


def create_llm_client_from_config(config_manager=None) -> Optional[LLMClient]:
    """
    从配置管理器创建 LLM 客户端
    
    Args:
        config_manager: 配置管理器实例，如果为 None 则使用默认
    
    Returns:
        LLMClient 实例或 None
    """
    if config_manager is None:
        from kas.core.config import get_config
        config_manager = get_config()
    
    api_key = config_manager.get_api_key()
    if not api_key:
        logger.error("未配置 API Key")
        return None
    
    provider = config_manager.llm.provider
    base_url = config_manager.llm.base_url
    timeout = config_manager.get_timeout()
    
    # 环境变量覆盖
    if os.getenv('ANTHROPIC_BASE_URL') and provider == 'zhipu':
        base_url = os.getenv('ANTHROPIC_BASE_URL')
    
    return LLMClient(
        api_key=api_key,
        provider=provider,
        base_url=base_url,
        timeout=timeout
    )


# 便捷函数
def quick_chat(
    message: str,
    system_prompt: str = "You are a helpful assistant.",
    **kwargs
) -> str:
    """
    快速对话（使用当前配置）
    
    Args:
        message: 用户消息
        system_prompt: 系统提示词
        **kwargs: 其他参数
    
    Returns:
        响应文本
    """
    client = create_llm_client_from_config()
    if client is None:
        return "Error: 未配置 LLM API Key"
    
    try:
        response = client.chat(
            system_prompt=system_prompt,
            user_message=message,
            **kwargs
        )
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"
