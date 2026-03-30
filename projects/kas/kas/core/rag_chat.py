"""
RAG 增强对话
在对话时自动检索知识库相关内容
"""
from typing import List, Optional, Dict

from kas.core.knowledge import KnowledgeBase, UserMemory


class RAGChatEngine:
    """RAG 增强的对话引擎"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.kb: Optional[KnowledgeBase] = None
        self.memory: Optional[UserMemory] = None
        
        # 尝试初始化知识库
        try:
            from kas.core.knowledge import get_knowledge_base, get_user_memory
            self.kb = get_knowledge_base(agent_name)
            self.memory = get_user_memory(agent_name)
        except ImportError:
            pass  # ChromaDB 未安装
    
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """
        检索相关知识作为上下文
        
        Args:
            query: 用户查询
            top_k: 检索文档数量
            
        Returns:
            格式化的上下文文本
        """
        if not self.kb:
            return ""
        
        try:
            results = self.kb.search(query, top_k=top_k)
            
            if not results:
                return ""
            
            # 构建上下文
            context_parts = []
            for i, result in enumerate(results, 1):
                doc = result.document
                content = doc.content[:500]  # 限制长度
                source = doc.metadata.get('filename', doc.metadata.get('source', 'Unknown'))
                
                context_parts.append(
                    f"[参考文档 {i}]\n"
                    f"来源: {source}\n"
                    f"内容: {content}\n"
                )
            
            return "\n".join(context_parts)
            
        except Exception:
            return ""
    
    def get_memory_context(self) -> str:
        """获取用户偏好记忆作为上下文"""
        if not self.memory:
            return ""
        
        try:
            data = self.memory.show()
            if not data:
                return ""
            
            # 构建记忆上下文
            memory_parts = []
            for key, value in data.items():
                if key != 'updated_at':
                    memory_parts.append(f"- {key}: {value}")
            
            if memory_parts:
                return "[用户偏好]\n" + "\n".join(memory_parts)
            return ""
            
        except Exception:
            return ""
    
    def build_rag_prompt(
        self,
        system_prompt: str,
        user_message: str,
        include_memory: bool = True
    ) -> str:
        """
        构建 RAG 增强的 Prompt
        
        Args:
            system_prompt: 原始系统 Prompt
            user_message: 用户消息
            include_memory: 是否包含用户记忆
            
        Returns:
            增强后的 Prompt
        """
        # 检索相关知识
        knowledge_context = self.retrieve_context(user_message)
        
        # 获取用户记忆
        memory_context = ""
        if include_memory:
            memory_context = self.get_memory_context()
        
        # 构建增强 Prompt
        enhanced_prompt = system_prompt
        
        if knowledge_context:
            enhanced_prompt += f"\n\n[知识库相关内容]\n{knowledge_context}"
        
        if memory_context:
            enhanced_prompt += f"\n\n{memory_context}"
        
        return enhanced_prompt
    
    def learn_from_interaction(self, topic: str, preference: str):
        """从交互中学习用户偏好"""
        if self.memory:
            self.memory.set(topic, preference)


def enhance_chat_with_rag(chat_engine, user_message: str) -> str:
    """
    增强对话的便捷函数
    
    Args:
        chat_engine: 原对话引擎
        user_message: 用户消息
        
    Returns:
        增强后的系统 Prompt
    """
    rag_engine = RAGChatEngine(chat_engine.agent_name)
    
    return rag_engine.build_rag_prompt(
        chat_engine.system_prompt,
        user_message
    )
