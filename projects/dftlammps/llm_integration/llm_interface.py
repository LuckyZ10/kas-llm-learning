"""
Unified LLM Interface Module

Provides a standardized interface for interacting with multiple LLM providers:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- DeepSeek
- Local models (Llama, etc. via vLLM, llama.cpp)

Features:
- Async support with streaming
- Retry mechanism with exponential backoff
- Prompt engineering utilities (Few-shot, Chain-of-Thought)
- Type-safe responses
"""

from __future__ import annotations

import os
import json
import time
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
    Any,
    Protocol,
)
from contextlib import asynccontextmanager
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    LOCAL_VLLM = "local_vllm"
    LOCAL_LLAMACPP = "local_llamacpp"
    AZURE_OPENAI = "azure_openai"
    CUSTOM = "custom"


class PromptStyle(Enum):
    """Prompt engineering styles."""
    ZERO_SHOT = auto()
    FEW_SHOT = auto()
    CHAIN_OF_THOUGHT = auto()
    SELF_CONSISTENCY = auto()
    TREE_OF_THOUGHTS = auto()
    REACT = auto()  # Reasoning + Acting


@dataclass
class LLMConfig:
    """Configuration for LLM connections."""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_exponential: bool = True
    system_prompt: Optional[str] = None
    
    # Provider-specific settings
    organization: Optional[str] = None  # OpenAI
    api_version: Optional[str] = None   # Azure
    custom_headers: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls, provider: LLMProvider, model: Optional[str] = None) -> LLMConfig:
        """Create config from environment variables."""
        env_mapping = {
            LLMProvider.OPENAI: ("OPENAI_API_KEY", "OPENAI_MODEL"),
            LLMProvider.ANTHROPIC: ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"),
            LLMProvider.DEEPSEEK: ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL"),
            LLMProvider.AZURE_OPENAI: ("AZURE_OPENAI_KEY", "AZURE_OPENAI_MODEL"),
        }
        
        if provider in env_mapping:
            key_env, model_env = env_mapping[provider]
            api_key = os.getenv(key_env)
            model = model or os.getenv(model_env, "gpt-4")
            
            if not api_key:
                raise ValueError(f"Environment variable {key_env} not set")
        else:
            api_key = None
            model = model or "local-model"
        
        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=os.getenv(f"{provider.value.upper()}_BASE_URL"),
        )


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # system, user, assistant, tool
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-compatible dictionary."""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result
    
    @classmethod
    def system(cls, content: str) -> Message:
        """Create a system message."""
        return cls(role="system", content=content)
    
    @classmethod
    def user(cls, content: str, name: Optional[str] = None) -> Message:
        """Create a user message."""
        return cls(role="user", content=content, name=name)
    
    @classmethod
    def assistant(cls, content: str, tool_calls: Optional[List[Dict]] = None) -> Message:
        """Create an assistant message."""
        return cls(role="assistant", content=content, tool_calls=tool_calls)
    
    @classmethod
    def tool(cls, content: str, tool_call_id: str) -> Message:
        """Create a tool response message."""
        return cls(role="tool", content=content, tool_call_id=tool_call_id)


@dataclass
class StreamResponse:
    """A streaming response chunk."""
    content: str
    is_finished: bool = False
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionResponse:
    """A complete LLM response."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def cost_estimate(self) -> float:
        """Estimate API cost (rough approximation)."""
        # Approximate costs per 1K tokens
        costs = {
            "gpt-4": (0.03, 0.06),
            "gpt-4-turbo": (0.01, 0.03),
            "gpt-3.5-turbo": (0.0015, 0.002),
            "claude-3-opus": (0.015, 0.075),
            "claude-3-sonnet": (0.003, 0.015),
        }
        
        for model_key, (input_cost, output_cost) in costs.items():
            if model_key in self.model.lower():
                return (self.prompt_tokens / 1000 * input_cost + 
                        self.completion_tokens / 1000 * output_cost)
        return 0.0


class Conversation:
    """Manages a conversation history."""
    
    def __init__(self, system_prompt: Optional[str] = None, max_history: int = 20):
        self.messages: List[Message] = []
        self.max_history = max_history
        
        if system_prompt:
            self.messages.append(Message.system(system_prompt))
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        
        # Trim history while keeping system message
        if len(self.messages) > self.max_history + 1:
            # Keep system message and recent messages
            system_msgs = [m for m in self.messages if m.role == "system"]
            other_msgs = [m for m in self.messages if m.role != "system"]
            other_msgs = other_msgs[-self.max_history:]
            self.messages = system_msgs + other_msgs
    
    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.add_message(Message.user(content))
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message."""
        self.add_message(Message.assistant(content))
    
    def to_dicts(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries."""
        return [m.to_dict() for m in self.messages]
    
    def clear(self) -> None:
        """Clear all messages except system message."""
        system_msgs = [m for m in self.messages if m.role == "system"]
        self.messages = system_msgs
    
    def copy(self) -> Conversation:
        """Create a copy of the conversation."""
        new_conv = Conversation(max_history=self.max_history)
        new_conv.messages = self.messages.copy()
        return new_conv


T = TypeVar('T')


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate a completion."""
        pass
    
    @abstractmethod
    async def stream(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[StreamResponse]:
        """Generate a streaming completion."""
        pass
    
    async def _with_retry(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt if self.config.retry_exponential else 1)
                    logger.info(f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)
        
        raise last_exception


class OpenAIBackend(LLMBackend):
    """OpenAI API backend."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                organization=config.organization,
                timeout=config.timeout,
            )
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    async def complete(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate completion using OpenAI."""
        start_time = time.time()
        
        response = await self._with_retry(
            self.client.chat.completions.create,
            model=self.config.model,
            messages=[m.to_dict() for m in messages],
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
            **kwargs
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return CompletionResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=response.choices[0].finish_reason or "unknown",
            latency_ms=latency_ms,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
    
    async def stream(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[StreamResponse]:
        """Generate streaming completion."""
        stream = await self._with_retry(
            self.client.chat.completions.create,
            model=self.config.model,
            messages=[m.to_dict() for m in messages],
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            stream=True,
            **kwargs
        )
        
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                is_finished = chunk.choices[0].finish_reason is not None
                
                yield StreamResponse(
                    content=delta.content or "",
                    is_finished=is_finished,
                    finish_reason=chunk.choices[0].finish_reason,
                )


class AnthropicBackend(LLMBackend):
    """Anthropic Claude backend."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout,
            )
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
    
    def _convert_messages(self, messages: List[Message]) -> tuple[Optional[str], List[Dict]]:
        """Convert to Anthropic format."""
        system_msg = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            elif msg.role == "user":
                anthropic_messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                anthropic_messages.append({"role": "assistant", "content": msg.content})
        
        return system_msg, anthropic_messages
    
    async def complete(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate completion using Anthropic."""
        start_time = time.time()
        system, anthropic_messages = self._convert_messages(messages)
        
        response = await self._with_retry(
            self.client.messages.create,
            model=self.config.model,
            messages=anthropic_messages,
            system=system or "",
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens or 4096,
            **kwargs
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return CompletionResponse(
            content=response.content[0].text if response.content else "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            finish_reason=response.stop_reason or "unknown",
            latency_ms=latency_ms,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )
    
    async def stream(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[StreamResponse]:
        """Generate streaming completion."""
        system, anthropic_messages = self._convert_messages(messages)
        
        async with self.client.messages.stream(
            model=self.config.model,
            messages=anthropic_messages,
            system=system or "",
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens or 4096,
            **kwargs
        ) as stream:
            async for text in stream.text_stream:
                yield StreamResponse(
                    content=text,
                    is_finished=False,
                )
            
            yield StreamResponse(
                content="",
                is_finished=True,
                finish_reason=await stream.get_final_message().stop_reason,
            )


class DeepSeekBackend(LLMBackend):
    """DeepSeek API backend (OpenAI-compatible)."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import openai
            base_url = config.base_url or "https://api.deepseek.com/v1"
            self.client = openai.AsyncOpenAI(
                api_key=config.api_key,
                base_url=base_url,
                timeout=config.timeout,
            )
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    async def complete(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate completion using DeepSeek."""
        start_time = time.time()
        
        response = await self._with_retry(
            self.client.chat.completions.create,
            model=self.config.model,
            messages=[m.to_dict() for m in messages],
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            **kwargs
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return CompletionResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            usage=dict(response.usage) if response.usage else {},
            finish_reason=response.choices[0].finish_reason or "unknown",
            latency_ms=latency_ms,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )
    
    async def stream(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[StreamResponse]:
        """Generate streaming completion."""
        stream = await self._with_retry(
            self.client.chat.completions.create,
            model=self.config.model,
            messages=[m.to_dict() for m in messages],
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            stream=True,
            **kwargs
        )
        
        async for chunk in stream:
            if chunk.choices:
                yield StreamResponse(
                    content=chunk.choices[0].delta.content or "",
                    is_finished=chunk.choices[0].finish_reason is not None,
                    finish_reason=chunk.choices[0].finish_reason,
                )


class LocalVLLMBackend(LLMBackend):
    """Local vLLM backend for self-hosted models."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import openai
            base_url = config.base_url or "http://localhost:8000/v1"
            self.client = openai.AsyncOpenAI(
                api_key=config.api_key or "not-needed",
                base_url=base_url,
                timeout=config.timeout,
            )
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    async def complete(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate completion using local vLLM."""
        start_time = time.time()
        
        response = await self._with_retry(
            self.client.chat.completions.create,
            model=self.config.model,
            messages=[m.to_dict() for m in messages],
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            **kwargs
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        usage = response.usage
        return CompletionResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            usage=dict(usage) if usage else {},
            finish_reason=response.choices[0].finish_reason or "unknown",
            latency_ms=latency_ms,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )
    
    async def stream(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[StreamResponse]:
        """Generate streaming completion."""
        stream = await self._with_retry(
            self.client.chat.completions.create,
            model=self.config.model,
            messages=[m.to_dict() for m in messages],
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            stream=True,
            **kwargs
        )
        
        async for chunk in stream:
            if chunk.choices:
                yield StreamResponse(
                    content=chunk.choices[0].delta.content or "",
                    is_finished=chunk.choices[0].finish_reason is not None,
                    finish_reason=chunk.choices[0].finish_reason,
                )


class PromptEngineer:
    """Utilities for prompt engineering."""
    
    @staticmethod
    def create_few_shot_prompt(
        task_description: str,
        examples: List[tuple[str, str]],
        query: str,
        include_explanation: bool = False,
    ) -> str:
        """Create a few-shot prompt with examples.
        
        Args:
            task_description: Description of the task
            examples: List of (input, output) example pairs
            query: The actual query
            include_explanation: Whether to ask for explanation
        """
        prompt_parts = [task_description, ""]
        
        for i, (input_text, output_text) in enumerate(examples, 1):
            prompt_parts.append(f"Example {i}:")
            prompt_parts.append(f"Input: {input_text}")
            prompt_parts.append(f"Output: {output_text}")
            prompt_parts.append("")
        
        prompt_parts.append("Now, please process the following:")
        prompt_parts.append(f"Input: {query}")
        prompt_parts.append("Output:")
        
        if include_explanation:
            prompt_parts.append("\nPlease provide your reasoning step by step.")
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def create_chain_of_thought_prompt(question: str, reasoning_steps: Optional[List[str]] = None) -> str:
        """Create a chain-of-thought prompt.
        
        Args:
            question: The question to answer
            reasoning_steps: Optional hints for reasoning steps
        """
        prompt = f"""Please solve the following problem step by step, showing your reasoning clearly.

Question: {question}

Let's work through this systematically:
"""
        
        if reasoning_steps:
            prompt += "\nConsider these aspects:\n"
            for step in reasoning_steps:
                prompt += f"- {step}\n"
        
        prompt += "\nStep-by-step solution:\n"
        return prompt
    
    @staticmethod
    def create_react_prompt(
        task: str,
        tools_available: List[str],
    ) -> str:
        """Create a ReAct (Reasoning + Acting) prompt.
        
        Args:
            task: The task description
            tools_available: List of available tool names
        """
        tools_str = ", ".join(tools_available)
        
        return f"""You are an AI assistant that solves problems by alternating between Thought and Action.

Task: {task}

Available tools: {tools_str}

Please solve this task using the following format:
Thought: [Your reasoning about what to do next]
Action: [The tool to use or the action to take]
Observation: [The result of the action]
... (repeat Thought/Action/Observation as needed)
Thought: [Final reasoning]
Final Answer: [Your final answer]

Begin:
Thought:"""
    
    @staticmethod
    def create_self_consistency_prompt(question: str, num_paths: int = 3) -> str:
        """Create a prompt asking for multiple reasoning paths.
        
        Args:
            question: The question to answer
            num_paths: Number of different reasoning paths to generate
        """
        return f"""Please answer the following question by exploring {num_paths} different reasoning paths independently. 
Then, provide the most consistent answer based on all paths.

Question: {question}

Reasoning Path 1:
"""
    
    @staticmethod
    def create_tree_of_thoughts_prompt(
        problem: str,
        branching_factor: int = 3,
        depth: int = 2,
    ) -> str:
        """Create a Tree of Thoughts prompt.
        
        Args:
            problem: The problem to solve
            branching_factor: How many options to consider at each step
            depth: How many levels deep to explore
        """
        return f"""Solve the following problem using Tree of Thoughts reasoning.
At each step, consider {branching_factor} different approaches, evaluate them, and choose the most promising one.
Explore up to {depth} levels deep.

Problem: {problem}

Let's think through this systematically, exploring multiple possibilities at each step:

Level 1 - Initial approaches:
"""


class UnifiedLLMInterface:
    """Unified interface for multiple LLM providers."""
    
    def __init__(self, config: LLMConfig):
        """Initialize the unified interface.
        
        Args:
            config: Configuration for the LLM provider
        """
        self.config = config
        self.backend = self._create_backend(config)
        self.prompt_engineer = PromptEngineer()
    
    def _create_backend(self, config: LLMConfig) -> LLMBackend:
        """Create the appropriate backend based on config."""
        backends = {
            LLMProvider.OPENAI: OpenAIBackend,
            LLMProvider.ANTHROPIC: AnthropicBackend,
            LLMProvider.DEEPSEEK: DeepSeekBackend,
            LLMProvider.LOCAL_VLLM: LocalVLLMBackend,
            LLMProvider.AZURE_OPENAI: OpenAIBackend,  # Azure uses OpenAI client
        }
        
        if config.provider in backends:
            return backends[config.provider](config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
    
    async def complete(
        self,
        prompt: Union[str, List[Message]],
        conversation: Optional[Conversation] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate a completion.
        
        Args:
            prompt: The prompt string or list of messages
            conversation: Optional conversation context
            temperature: Override temperature
            max_tokens: Override max_tokens
            system_prompt: Optional system prompt override
            **kwargs: Additional parameters
            
        Returns:
            Completion response
        """
        messages = self._prepare_messages(prompt, conversation, system_prompt)
        
        return await self.backend.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    async def stream(
        self,
        prompt: Union[str, List[Message]],
        conversation: Optional[Conversation] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[StreamResponse]:
        """Generate a streaming completion.
        
        Args:
            prompt: The prompt string or list of messages
            conversation: Optional conversation context
            temperature: Override temperature
            max_tokens: Override max_tokens
            system_prompt: Optional system prompt override
            **kwargs: Additional parameters
            
        Yields:
            Stream response chunks
        """
        messages = self._prepare_messages(prompt, conversation, system_prompt)
        
        async for chunk in self.backend.stream(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        ):
            yield chunk
    
    def _prepare_messages(
        self,
        prompt: Union[str, List[Message]],
        conversation: Optional[Conversation],
        system_prompt: Optional[str],
    ) -> List[Message]:
        """Prepare message list from various inputs."""
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append(Message.system(system_prompt))
        elif self.config.system_prompt:
            messages.append(Message.system(self.config.system_prompt))
        
        # Add conversation history
        if conversation:
            messages.extend(conversation.messages)
        
        # Add current prompt
        if isinstance(prompt, str):
            messages.append(Message.user(prompt))
        else:
            messages.extend(prompt)
        
        return messages
    
    async def generate_with_few_shot(
        self,
        task_description: str,
        examples: List[tuple[str, str]],
        query: str,
        include_explanation: bool = False,
        **kwargs
    ) -> CompletionResponse:
        """Generate using few-shot prompting.
        
        Args:
            task_description: Description of the task
            examples: List of (input, output) example pairs
            query: The actual query
            include_explanation: Whether to ask for explanation
            **kwargs: Additional parameters
            
        Returns:
            Completion response
        """
        prompt = self.prompt_engineer.create_few_shot_prompt(
            task_description, examples, query, include_explanation
        )
        return await self.complete(prompt, **kwargs)
    
    async def generate_with_chain_of_thought(
        self,
        question: str,
        reasoning_steps: Optional[List[str]] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate using chain-of-thought prompting.
        
        Args:
            question: The question to answer
            reasoning_steps: Optional hints for reasoning steps
            **kwargs: Additional parameters
            
        Returns:
            Completion response
        """
        prompt = self.prompt_engineer.create_chain_of_thought_prompt(question, reasoning_steps)
        return await self.complete(prompt, **kwargs)
    
    async def generate_structured(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate structured JSON output.
        
        Args:
            prompt: The prompt
            output_schema: JSON schema for expected output
            **kwargs: Additional parameters
            
        Returns:
            Parsed JSON response
        """
        schema_prompt = f"""{prompt}

Please provide your response in the following JSON format:
{json.dumps(output_schema, indent=2)}

Respond with ONLY valid JSON, no other text."""
        
        response = await self.complete(schema_prompt, **kwargs)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())


class AsyncLLMInterface:
    """High-level async interface for batch operations and more."""
    
    def __init__(self, config: LLMConfig):
        self.interface = UnifiedLLMInterface(config)
    
    async def batch_complete(
        self,
        prompts: List[str],
        max_concurrent: int = 5,
        **kwargs
    ) -> List[CompletionResponse]:
        """Complete multiple prompts concurrently.
        
        Args:
            prompts: List of prompts
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional parameters for completion
            
        Returns:
            List of completion responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _complete_with_limit(prompt: str) -> CompletionResponse:
            async with semaphore:
                return await self.interface.complete(prompt, **kwargs)
        
        tasks = [_complete_with_limit(p) for p in prompts]
        return await asyncio.gather(*tasks)
    
    async def ensemble_generate(
        self,
        prompt: str,
        temperatures: List[float] = [0.3, 0.5, 0.7, 0.9],
        **kwargs
    ) -> List[CompletionResponse]:
        """Generate multiple responses with different temperatures for self-consistency.
        
        Args:
            prompt: The prompt
            temperatures: List of temperatures to try
            **kwargs: Additional parameters
            
        Returns:
            List of completion responses
        """
        tasks = [
            self.interface.complete(prompt, temperature=temp, **kwargs)
            for temp in temperatures
        ]
        return await asyncio.gather(*tasks)


# Convenience factory functions

def create_openai_interface(
    model: str = "gpt-4",
    api_key: Optional[str] = None,
    **kwargs
) -> UnifiedLLMInterface:
    """Create an OpenAI interface.
    
    Args:
        model: Model name
        api_key: API key (defaults to env var)
        **kwargs: Additional config options
    """
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model=model,
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        **kwargs
    )
    return UnifiedLLMInterface(config)


def create_anthropic_interface(
    model: str = "claude-3-sonnet-20240229",
    api_key: Optional[str] = None,
    **kwargs
) -> UnifiedLLMInterface:
    """Create an Anthropic interface.
    
    Args:
        model: Model name
        api_key: API key (defaults to env var)
        **kwargs: Additional config options
    """
    config = LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model=model,
        api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
        **kwargs
    )
    return UnifiedLLMInterface(config)


def create_deepseek_interface(
    model: str = "deepseek-chat",
    api_key: Optional[str] = None,
    **kwargs
) -> UnifiedLLMInterface:
    """Create a DeepSeek interface.
    
    Args:
        model: Model name
        api_key: API key (defaults to env var)
        **kwargs: Additional config options
    """
    config = LLMConfig(
        provider=LLMProvider.DEEPSEEK,
        model=model,
        api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
        **kwargs
    )
    return UnifiedLLMInterface(config)


def create_local_vllm_interface(
    model: str = "local-model",
    base_url: str = "http://localhost:8000/v1",
    **kwargs
) -> UnifiedLLMInterface:
    """Create a local vLLM interface.
    
    Args:
        model: Model name
        base_url: vLLM server URL
        **kwargs: Additional config options
    """
    config = LLMConfig(
        provider=LLMProvider.LOCAL_VLLM,
        model=model,
        base_url=base_url,
        **kwargs
    )
    return UnifiedLLMInterface(config)
