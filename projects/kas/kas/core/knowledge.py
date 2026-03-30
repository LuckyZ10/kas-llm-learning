"""
KAS 知识库/RAG 系统
基于 ChromaDB 的向量存储和检索
"""
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

# 尝试导入 ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from kas.core.config import get_config


@dataclass
class Document:
    """文档对象"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """搜索结果"""
    document: Document
    score: float
    distance: float


class KnowledgeBase:
    """Agent 知识库 - 基于 ChromaDB"""
    
    def __init__(self, agent_name: str, storage_path: Optional[str] = None):
        """
        初始化知识库
        
        Args:
            agent_name: Agent 名称
            storage_path: 存储路径，默认使用配置
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB not installed. "
                "Install with: pip install chromadb"
            )
        
        self.agent_name = agent_name
        
        # 确定存储路径
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            config = get_config()
            base_path = Path(config.config_dir) / "knowledge"
            self.storage_path = base_path / agent_name
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化 ChromaDB 客户端
        self.client = chromadb.PersistentClient(
            path=str(self.storage_path),
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"agent_name": agent_name}
        )
        
        print(f"📚 KnowledgeBase initialized: {self.storage_path}")
    
    def _generate_id(self, content: str) -> str:
        """生成文档 ID"""
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _get_embedding_function(self):
        """获取嵌入函数"""
        config = get_config()
        provider = config.llm.provider
        
        # 默认使用 ChromaDB 的默认嵌入
        # 也可以配置为使用 OpenAI 或其他嵌入模型
        return None  # 使用默认
    
    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        添加文档到知识库
        
        Args:
            content: 文档内容
            metadata: 元数据
            doc_id: 文档 ID，默认自动生成
            
        Returns:
            文档 ID
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
        
        # 生成 ID
        doc_id = doc_id or self._generate_id(content)
        
        # 准备元数据
        meta = {
            'agent_name': self.agent_name,
            'created_at': datetime.now().isoformat(),
            'content_length': len(content)
        }
        if metadata:
            # ChromaDB 要求 metadata 值必须是 str/int/float/bool
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)
        
        # 添加到集合
        self.collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[meta]
        )
        
        print(f"✅ Added document: {doc_id[:8]}...")
        return doc_id
    
    def add_documents(
        self,
        contents: List[str],
        metadatas: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        批量添加文档
        
        Args:
            contents: 文档内容列表
            metadatas: 元数据列表
            
        Returns:
            文档 ID 列表
        """
        if not contents:
            return []
        
        doc_ids = [self._generate_id(c) for c in contents]
        metas = []
        
        for i, content in enumerate(contents):
            meta = {
                'agent_name': self.agent_name,
                'created_at': datetime.now().isoformat(),
                'content_length': len(content)
            }
            if metadatas and i < len(metadatas):
                for k, v in metadatas[i].items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    else:
                        meta[k] = str(v)
            metas.append(meta)
        
        self.collection.add(
            ids=doc_ids,
            documents=contents,
            metadatas=metas
        )
        
        print(f"✅ Added {len(contents)} documents")
        return doc_ids
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        搜索知识库
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_dict: 过滤条件
            
        Returns:
            搜索结果列表
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_dict
        )
        
        search_results = []
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                doc = Document(
                    id=doc_id,
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {},
                )
                distance = results['distances'][0][i] if results['distances'] else 0
                score = 1 - distance  # 转换距离为相似度分数
                
                search_results.append(SearchResult(
                    document=doc,
                    score=score,
                    distance=distance
                ))
        
        return search_results
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """获取指定文档"""
        results = self.collection.get(ids=[doc_id])
        
        if results['ids']:
            return Document(
                id=results['ids'][0],
                content=results['documents'][0],
                metadata=results['metadatas'][0] if results['metadatas'] else {}
            )
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception:
            return False
    
    def clear(self):
        """清空知识库"""
        self.client.delete_collection("documents")
        self.collection = self.client.create_collection(
            name="documents",
            metadata={"agent_name": self.agent_name}
        )
        print(f"🗑️ KnowledgeBase cleared: {self.agent_name}")
    
    def count(self) -> int:
        """获取文档数量"""
        return self.collection.count()
    
    def list_documents(self, limit: int = 100) -> List[Document]:
        """列出所有文档"""
        results = self.collection.get(limit=limit)
        
        documents = []
        for i, doc_id in enumerate(results['ids']):
            documents.append(Document(
                id=doc_id,
                content=results['documents'][i],
                metadata=results['metadatas'][i] if results['metadatas'] else {}
            ))
        
        return documents


class UserMemory:
    """用户偏好记忆 - 简单的键值存储"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        config = get_config()
        self.memory_path = Path(config.config_dir) / "memory" / f"{agent_name}.json"
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()
    
    def _load(self) -> Dict:
        """加载记忆"""
        if self.memory_path.exists():
            import json
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save(self):
        """保存记忆"""
        import json
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
    
    def get(self, key: str, default=None):
        """获取记忆"""
        return self._data.get(key, default)
    
    def set(self, key: str, value: Any):
        """设置记忆"""
        self._data[key] = value
        self._data['updated_at'] = datetime.now().isoformat()
        self._save()
    
    def update(self, data: Dict):
        """批量更新"""
        self._data.update(data)
        self._data['updated_at'] = datetime.now().isoformat()
        self._save()
    
    def delete(self, key: str):
        """删除记忆"""
        if key in self._data:
            del self._data[key]
            self._save()
    
    def clear(self):
        """清空记忆"""
        self._data = {}
        self._save()
    
    def show(self) -> Dict:
        """显示所有记忆"""
        return self._data.copy()


# 便捷函数
def get_knowledge_base(agent_name: str) -> KnowledgeBase:
    """获取 Agent 的知识库"""
    return KnowledgeBase(agent_name)


def get_user_memory(agent_name: str) -> UserMemory:
    """获取用户记忆"""
    return UserMemory(agent_name)
