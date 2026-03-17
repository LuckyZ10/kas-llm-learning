"""
KAS Core - Chat Engine
简单优先的对话引擎
"""
from typing import Dict, List, Optional
from pathlib import Path
import yaml

from .models import Agent, get_params_for_task


class ChatEngine:
    """对话引擎 - 简化版"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.conversation_history = []
    
    def load_agent(self, agent_path: str) -> Agent:
        """加载Agent"""
        agent_file = Path(agent_path) / 'agent.yaml'
        
        if not agent_file.exists():
            # 尝试在默认目录查找
            agent_file = Path.home() / '.kas' / 'agents' / agent_path / 'agent.yaml'
        
        if not agent_file.exists():
            raise ValueError(f"Agent not found: {agent_path}")
        
        with open(agent_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return Agent.from_dict(data)
    
    def chat(
        self,
        agent: Agent,
        message: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        与Agent对话
        
        Args:
            agent: Agent对象
            message: 用户消息
            context: 上下文信息
        
        Returns:
            Agent响应
        """
        # 如果没有LLM客户端，返回模拟响应
        if self.llm_client is None:
            return self._mock_response(agent, message)
        
        # 构建完整Prompt
        system_prompt = agent.system_prompt
        
        # 根据任务类型调整参数
        task_type = self._detect_task_type(message)
        params = get_params_for_task(task_type)
        params.update(agent.model_config)
        
        # 调用LLM
        try:
            response = self.llm_client.chat(
                system_prompt=system_prompt,
                user_message=message,
                **params
            )
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _detect_task_type(self, message: str) -> str:
        """检测任务类型 - 简单规则"""
        message_lower = message.lower()
        
        # 简单关键词匹配
        if any(word in message_lower for word in ['simple', 'quick', 'brief']):
            return 'simple'
        
        if any(word in message_lower for word in ['complex', 'detailed', 'explain']):
            return 'complex'
        
        if any(word in message_lower for word in ['creative', 'imagine', 'suggest']):
            return 'creative'
        
        return 'standard'
    
    def _mock_response(self, agent: Agent, message: str) -> str:
        """模拟响应（无LLM时）"""
        return f"""[Mock Response from {agent.name}]

I received your message: "{message[:50]}..."

My capabilities:
{chr(10).join([f"- {cap.name}" for cap in agent.capabilities[:3]])}

[Note: This is a mock response. Connect an LLM client for real responses.]
"""
    
    def interactive_chat(self, agent_path: str):
        """交互式对话"""
        agent = self.load_agent(agent_path)
        
        print(f"\n🤖 Starting chat with {agent.name}")
        print(f"   Capabilities: {', '.join([c.name for c in agent.capabilities])}")
        print("   Type 'exit' to quit\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                response = self.chat(agent, user_input)
                print(f"\n{agent.name}: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


# 简单的LLM客户端接口示例
class SimpleLLMClient:
    """简单的LLM客户端示例"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self.model = model
    
    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        调用LLM API
        
        这里应该实现实际的API调用
        示例使用OpenAI格式
        """
        try:
            import openai
            
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return response.choices[0].message.content
            
        except ImportError:
            return "Error: openai package not installed. Run: pip install openai"
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
