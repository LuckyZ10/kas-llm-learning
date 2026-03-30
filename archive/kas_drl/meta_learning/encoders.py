"""
KAS Meta-Learning - Feature Encoders
特征编码器：Transformer/LSTM建模遥测数据序列，项目语义特征提取
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


class ProjectFeatureEncoder(nn.Module):
    """项目特征编码器 - 提取项目语义特征"""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.output_dim = output_dim
    
    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        Returns:
            features: [batch, output_dim]
        """
        # Embedding + Positional Encoding
        x = self.embedding(token_ids)
        x = self.pos_encoding(x)
        
        # Transformer编码
        if attention_mask is not None:
            # 转换mask为Transformer格式
            mask = (attention_mask == 0)
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        # 平均池化
        x = x.mean(dim=1)
        
        # 输出投影
        features = self.output_projection(x)
        
        return features


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TelemetryLSTMEncoder(nn.Module):
    """遥测数据LSTM编码器 - 建模时序数据"""
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 64,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 输出维度
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.output_dim = output_dim
    
    def forward(self, telemetry_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            telemetry_sequence: [batch, seq_len, input_dim]
        Returns:
            features: [batch, output_dim]
        """
        # LSTM编码
        lstm_out, (hidden, cell) = self.lstm(telemetry_sequence)
        
        # 使用最后一个时间步的隐藏状态
        if self.bidirectional:
            # 拼接双向的最后隐藏状态
            hidden_forward = hidden[-2]  # 前向最后一层
            hidden_backward = hidden[-1]  # 反向最后一层
            hidden_state = torch.cat([hidden_forward, hidden_backward], dim=-1)
        else:
            hidden_state = hidden[-1]
        
        # 输出投影
        features = self.output_projection(hidden_state)
        
        return features
    
    def forward_with_attention(self, telemetry_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """带注意力权重的编码"""
        lstm_out, _ = self.lstm(telemetry_sequence)
        
        # 自注意力
        attention_weights = torch.softmax(
            torch.matmul(lstm_out, lstm_out.transpose(-2, -1)) / np.sqrt(lstm_out.size(-1)),
            dim=-1
        )
        attended = torch.matmul(attention_weights, lstm_out)
        
        # 平均池化
        pooled = attended.mean(dim=1)
        features = self.output_projection(pooled)
        
        return features, attention_weights.mean(dim=1)


class MultiModalProjectEncoder(nn.Module):
    """多模态项目编码器 - 融合代码、文档、遥测数据"""
    
    def __init__(
        self,
        code_vocab_size: int = 10000,
        code_dim: int = 128,
        telemetry_dim: int = 64,
        fusion_dim: int = 256,
        output_dim: int = 128
    ):
        super().__init__()
        
        # 代码编码器
        self.code_encoder = ProjectFeatureEncoder(
            vocab_size=code_vocab_size,
            output_dim=code_dim
        )
        
        # 遥测编码器
        self.telemetry_encoder = TelemetryLSTMEncoder(
            output_dim=telemetry_dim
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(code_dim + telemetry_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, output_dim),
            nn.Tanh()
        )
        
        self.output_dim = output_dim
    
    def forward(
        self,
        code_tokens: torch.Tensor,
        telemetry_sequence: torch.Tensor,
        code_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            code_tokens: [batch, code_seq_len]
            telemetry_sequence: [batch, time_steps, telemetry_dim]
            code_mask: [batch, code_seq_len]
        Returns:
            features: [batch, output_dim]
        """
        # 编码各模态
        code_features = self.code_encoder(code_tokens, code_mask)
        telemetry_features = self.telemetry_encoder(telemetry_sequence)
        
        # 融合
        combined = torch.cat([code_features, telemetry_features], dim=-1)
        features = self.fusion(combined)
        
        return features


class ProjectSimilarityNetwork(nn.Module):
    """项目相似度网络 - 用于元学习的任务相似度计算"""
    
    def __init__(self, feature_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        self.comparison_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, project1_features: torch.Tensor, project2_features: torch.Tensor) -> torch.Tensor:
        """
        计算两个项目的相似度
        
        Args:
            project1_features: [batch, feature_dim]
            project2_features: [batch, feature_dim]
        Returns:
            similarity: [batch, 1]
        """
        combined = torch.cat([project1_features, project2_features], dim=-1)
        return self.comparison_net(combined)
    
    def compute_task_weights(
        self,
        target_project: torch.Tensor,
        source_projects: torch.Tensor
    ) -> torch.Tensor:
        """
        计算目标项目与源项目的相似度权重
        
        Args:
            target_project: [1, feature_dim]
            source_projects: [num_source, feature_dim]
        Returns:
            weights: [num_source]
        """
        # 扩展目标项目以匹配源项目数量
        target_expanded = target_project.expand(source_projects.size(0), -1)
        
        # 计算相似度
        similarities = self.forward(target_expanded, source_projects)
        
        # Softmax归一化
        weights = F.softmax(similarities.squeeze(), dim=0)
        
        return weights


class ProjectEmbeddingStore:
    """项目嵌入存储"""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.project_embeddings = {}
        self.project_metadata = {}
    
    def add_project(
        self,
        project_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None
    ):
        """添加项目嵌入"""
        self.project_embeddings[project_id] = embedding
        self.project_metadata[project_id] = metadata or {}
    
    def get_embedding(self, project_id: str) -> Optional[np.ndarray]:
        """获取项目嵌入"""
        return self.project_embeddings.get(project_id)
    
    def find_similar_projects(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """查找相似项目"""
        similarities = []
        
        for project_id, embedding in self.project_embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((project_id, float(similarity)))
        
        # 排序并返回top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_all_embeddings(self) -> Tuple[List[str], torch.Tensor]:
        """获取所有嵌入"""
        project_ids = list(self.project_embeddings.keys())
        embeddings = torch.FloatTensor(np.array([
            self.project_embeddings[pid] for pid in project_ids
        ]))
        return project_ids, embeddings
