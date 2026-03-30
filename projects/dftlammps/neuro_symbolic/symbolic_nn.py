"""
符号神经网络模块 - 神经与符号双向翻译

实现神经网络与符号表示之间的双向转换，
支持概念学习、抽象推理和层次表示学习。
"""

from typing import List, Dict, Tuple, Optional, Callable, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import re


class SymbolType(Enum):
    """符号类型枚举"""
    CONCEPT = auto()      # 概念
    RELATION = auto()     # 关系
    RULE = auto()         # 规则
    CONSTRAINT = auto()   # 约束


@dataclass
class Symbol:
    """符号表示"""
    name: str
    symbol_type: SymbolType
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def __hash__(self):
        return hash((self.name, self.symbol_type))
    
    def __eq__(self, other):
        if not isinstance(other, Symbol):
            return False
        return self.name == other.name and self.symbol_type == other.symbol_type


@dataclass
class ConceptHierarchy:
    """概念层次结构"""
    concept: Symbol
    parents: List['ConceptHierarchy'] = field(default_factory=list)
    children: List['ConceptHierarchy'] = field(default_factory=list)
    level: int = 0
    
    def add_child(self, child: 'ConceptHierarchy'):
        """添加子概念"""
        self.children.append(child)
        child.parents.append(self)
        child.level = self.level + 1
    
    def get_ancestors(self) -> List[Symbol]:
        """获取所有祖先概念"""
        ancestors = []
        for parent in self.parents:
            ancestors.append(parent.concept)
            ancestors.extend(parent.get_ancestors())
        return ancestors
    
    def get_descendants(self) -> List[Symbol]:
        """获取所有后代概念"""
        descendants = []
        for child in self.children:
            descendants.append(child.concept)
            descendants.extend(child.get_descendants())
        return descendants


class NeuralToSymbolicTranslator(nn.Module):
    """
    神经到符号翻译器
    
    将神经网络的连续表示转换为离散的符号表示。
    """
    
    def __init__(self, 
                 input_dim: int,
                 num_symbols: int,
                 embedding_dim: int = 128,
                 temperature: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_symbols = num_symbols
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # 符号嵌入
        self.symbol_embeddings = nn.Embedding(num_symbols, embedding_dim)
        
        # 神经到符号编码器
        self.neural_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        
        # 符号生成器（使用Gumbel-Softmax实现可微分离散化）
        self.symbol_generator = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_symbols)
        )
        
        # 属性预测器
        self.attribute_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    
    def forward(self, neural_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将神经表示翻译为符号
        
        Returns:
            symbol_probs: (batch_size, num_symbols) 符号概率分布
            symbol_ids: (batch_size,) 采样的符号ID
            attributes: (batch_size, 64) 预测的符号属性
        """
        # 编码神经输入
        encoded = self.neural_encoder(neural_input)
        
        # 生成符号 logits
        logits = self.symbol_generator(encoded)
        
        # Gumbel-Softmax采样（可微分离散化）
        symbol_probs = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
        
        # 硬采样（用于推理）
        symbol_ids = torch.argmax(symbol_probs, dim=-1)
        
        # 预测属性
        attributes = self.attribute_predictor(encoded)
        
        return symbol_probs, symbol_ids, attributes
    
    def decode_symbol_sequence(self, 
                               neural_sequence: torch.Tensor) -> List[List[int]]:
        """
        解码符号序列（用于序列数据）
        
        Args:
            neural_sequence: (batch_size, seq_len, input_dim)
        
        Returns:
            symbol_sequences: List of symbol ID lists
        """
        batch_size, seq_len, _ = neural_sequence.shape
        
        # 处理序列
        symbol_sequences = []
        for i in range(batch_size):
            sequence = []
            for t in range(seq_len):
                _, symbol_id, _ = self.forward(neural_sequence[i, t].unsqueeze(0))
                sequence.append(symbol_id.item())
            symbol_sequences.append(sequence)
        
        return symbol_sequences
    
    def extract_rules(self, 
                     neural_patterns: torch.Tensor,
                     threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        从神经模式中提取符号规则
        
        使用注意力机制发现输入特征之间的因果关系。
        """
        # 编码模式
        encoded = self.neural_encoder(neural_patterns)
        
        # 计算特征间的关系矩阵
        attention = torch.softmax(
            encoded @ encoded.T / np.sqrt(self.embedding_dim),
            dim=-1
        )
        
        rules = []
        # 发现高注意力权重的关系
        for i in range(attention.shape[0]):
            for j in range(attention.shape[1]):
                if i != j and attention[i, j] > threshold:
                    # 提取相关符号
                    _, premise_id, _ = self.forward(neural_patterns[i].unsqueeze(0))
                    _, conclusion_id, _ = self.forward(neural_patterns[j].unsqueeze(0))
                    
                    rules.append({
                        'premise': premise_id.item(),
                        'conclusion': conclusion_id.item(),
                        'confidence': attention[i, j].item()
                    })
        
        return rules


class SymbolicToNeuralTranslator(nn.Module):
    """
    符号到神经翻译器
    
    将离散的符号表示转换为连续的神经表示。
    """
    
    def __init__(self,
                 num_symbols: int,
                 num_relations: int,
                 output_dim: int = 128,
                 embedding_dim: int = 128):
        super().__init__()
        self.num_symbols = num_symbols
        self.num_relations = num_relations
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        
        # 符号嵌入
        self.symbol_embeddings = nn.Embedding(num_symbols, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # 符号图神经网络
        self.gnn_layers = nn.ModuleList([
            SymbolicGNNLayer(embedding_dim, embedding_dim)
            for _ in range(3)
        ])
        
        # 符号序列编码器（用于规则序列）
        self.sequence_encoder = nn.LSTM(
            embedding_dim, embedding_dim // 2,
            num_layers=2, batch_first=True, bidirectional=True
        )
        
        # 解码器
        self.neural_decoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, 
                symbols: torch.Tensor,
                relations: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        将符号转换为神经表示
        
        Args:
            symbols: (batch_size, num_symbols) 符号存在矩阵
            relations: (num_relations, 3) 关系三元组 (subject, relation, object)
        
        Returns:
            neural_repr: (batch_size, output_dim) 神经表示
        """
        batch_size = symbols.shape[0]
        
        # 获取符号嵌入
        symbol_indices = torch.arange(self.num_symbols)
        symbol_embs = self.symbol_embeddings(symbol_indices)
        
        # 应用GNN传播（如果有关系）
        if relations is not None:
            for gnn_layer in self.gnn_layers:
                symbol_embs = gnn_layer(symbol_embs, relations)
        
        # 根据输入符号选择对应的嵌入
        weighted_embs = symbols.unsqueeze(-1) * symbol_embs.unsqueeze(0)
        aggregated = weighted_embs.sum(dim=1)  # (batch_size, embedding_dim)
        
        # 如果输入是序列，使用LSTM编码
        if len(symbols.shape) == 3:
            lstm_out, _ = self.sequence_encoder(weighted_embs)
            aggregated = lstm_out[:, -1, :]
        
        # 解码为神经表示
        neural_repr = self.neural_decoder(aggregated)
        
        return neural_repr
    
    def encode_knowledge_graph(self,
                               entities: List[int],
                               relations: List[Tuple[int, int, int]]) -> torch.Tensor:
        """
        编码知识图谱为神经表示
        
        Args:
            entities: 实体ID列表
            relations: (subject, relation, object)三元组列表
        
        Returns:
            kg_embedding: 知识图谱的整体嵌入
        """
        # 实体嵌入
        entity_embs = self.symbol_embeddings(torch.tensor(entities))
        
        # 构建关系张量
        rel_tensor = torch.tensor(relations)
        
        # 消息传递
        for gnn_layer in self.gnn_layers:
            entity_embs = gnn_layer(entity_embs, rel_tensor)
        
        # 聚合所有实体表示
        kg_embedding = entity_embs.mean(dim=0)
        
        return kg_embedding


class SymbolicGNNLayer(nn.Module):
    """符号图神经网络层"""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.message_net = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        self.update_net = nn.GRUCell(out_dim, out_dim)
    
    def forward(self, 
                node_features: torch.Tensor,
                edges: torch.Tensor) -> torch.Tensor:
        """
        图神经网络前向传播
        
        Args:
            node_features: (num_nodes, in_dim)
            edges: (num_edges, 3) - (source, relation, target)
        
        Returns:
            updated_features: (num_nodes, out_dim)
        """
        num_nodes = node_features.shape[0]
        aggregated = torch.zeros(num_nodes, self.out_dim)
        
        # 聚合邻居消息
        for edge in edges:
            src, rel, tgt = edge.tolist()
            if src < num_nodes and tgt < num_nodes:
                # 构建消息
                message_input = torch.cat([
                    node_features[src],
                    node_features[tgt]
                ])
                message = self.message_net(message_input)
                aggregated[tgt] += message
        
        # 更新节点特征
        updated = self.update_net(aggregated, node_features)
        
        return updated


class ConceptLearner(nn.Module):
    """
    概念学习器
    
    从数据中自动学习概念层次结构和抽象表示。
    """
    
    def __init__(self,
                 input_dim: int,
                 num_initial_concepts: int = 10,
                 embedding_dim: int = 128,
                 max_levels: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.num_initial_concepts = num_initial_concepts
        self.embedding_dim = embedding_dim
        self.max_levels = max_levels
        
        # 概念原型（可学习）
        self.concept_prototypes = nn.Parameter(
            torch.randn(num_initial_concepts, embedding_dim)
        )
        
        # 概念编码器
        self.concept_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        
        # 层次聚类网络
        self.hierarchy_net = nn.ModuleList([
            nn.Linear(embedding_dim * 2, 1)
            for _ in range(max_levels)
        ])
        
        # 抽象网络：学习更高级的概念
        self.abstraction_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 概念分类器
        self.concept_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_initial_concepts)
        )
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        学习概念表示
        
        Returns:
            concept_probs: (batch_size, num_concepts) 概念归属概率
            abstraction: (batch_size, embedding_dim) 抽象表示
            hierarchy_scores: (batch_size, max_levels) 层次分数
        """
        # 编码输入
        encoded = self.concept_encoder(inputs)
        
        # 计算与概念原型的相似度
        similarities = F.cosine_similarity(
            encoded.unsqueeze(1),
            self.concept_prototypes.unsqueeze(0),
            dim=2
        )
        concept_probs = F.softmax(similarities * 10, dim=1)
        
        # 学习抽象表示
        # 找到最匹配的概念原型
        best_concept_idx = torch.argmax(concept_probs, dim=1)
        best_prototypes = self.concept_prototypes[best_concept_idx]
        
        # 组合输入和概念原型生成抽象
        abstraction_input = torch.cat([encoded, best_prototypes], dim=1)
        abstraction = self.abstraction_net(abstraction_input)
        
        # 计算层次分数
        hierarchy_scores = []
        for level_net in self.hierarchy_net:
            score = torch.sigmoid(level_net(abstraction_input))
            hierarchy_scores.append(score)
        hierarchy_scores = torch.cat(hierarchy_scores, dim=1)
        
        return concept_probs, abstraction, hierarchy_scores
    
    def learn_new_concept(self, 
                         examples: torch.Tensor,
                         concept_name: str,
                         parent_concepts: Optional[List[int]] = None) -> Symbol:
        """
        从示例中学习新概念
        
        Args:
            examples: 概念的示例数据
            concept_name: 概念名称
            parent_concepts: 父概念索引列表
        
        Returns:
            新学习的概念符号
        """
        # 编码示例
        encoded_examples = self.concept_encoder(examples)
        
        # 计算原型（均值）
        new_prototype = encoded_examples.mean(dim=0)
        
        # 添加到概念原型
        new_prototype_param = nn.Parameter(new_prototype.unsqueeze(0))
        self.concept_prototypes = nn.Parameter(
            torch.cat([self.concept_prototypes, new_prototype_param], dim=0)
        )
        
        # 创建概念符号
        concept = Symbol(
            name=concept_name,
            symbol_type=SymbolType.CONCEPT,
            attributes={
                'prototype': new_prototype.detach().numpy(),
                'num_examples': len(examples),
                'parent_concepts': parent_concepts or []
            }
        )
        
        return concept
    
    def discover_relations(self, 
                          concepts: List[Symbol],
                          threshold: float = 0.8) -> List[Tuple[Symbol, str, Symbol]]:
        """
        自动发现概念之间的关系
        
        基于概念原型的相似度和层次结构发现关系。
        """
        relations = []
        
        # 计算概念间的相似度
        concept_embs = torch.stack([
            torch.tensor(c.attributes.get('prototype', np.zeros(self.embedding_dim)))
            for c in concepts
        ])
        
        similarity_matrix = F.cosine_similarity(
            concept_embs.unsqueeze(1),
            concept_embs.unsqueeze(0),
            dim=2
        )
        
        for i, concept_i in enumerate(concepts):
            for j, concept_j in enumerate(concepts):
                if i != j:
                    sim = similarity_matrix[i, j].item()
                    
                    # 相似关系
                    if sim > threshold:
                        relations.append((concept_i, 'similar_to', concept_j))
                    
                    # 层次关系（基于原型距离）
                    if concept_i.attributes.get('parent_concepts'):
                        if j in concept_i.attributes['parent_concepts']:
                            relations.append((concept_i, 'is_a', concept_j))
        
        return relations


class HierarchicalRepresentationLearning(nn.Module):
    """
    层次表示学习
    
    学习从低级特征到高级抽象的层次表示。
    """
    
    def __init__(self,
                 input_dim: int,
                 level_dims: List[int] = [256, 128, 64],
                 num_levels: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.level_dims = level_dims
        self.num_levels = num_levels
        
        # 层次编码器
        self.level_encoders = nn.ModuleList()
        prev_dim = input_dim
        for dim in level_dims:
            self.level_encoders.append(nn.Sequential(
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))
            prev_dim = dim
        
        # 自上而下反馈（高级到低级）
        self.top_down_projections = nn.ModuleList()
        for i in range(len(level_dims) - 1, 0, -1):
            self.top_down_projections.append(nn.Linear(
                level_dims[i], level_dims[i-1]
            ))
        
        # 层次注意力
        self.hierarchical_attention = nn.MultiheadAttention(
            embed_dim=level_dims[-1],
            num_heads=4,
            batch_first=True
        )
        
        # 符号解码器（将最高层转换为符号）
        self.symbol_decoder = nn.Sequential(
            nn.Linear(level_dims[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    
    def forward(self, 
                inputs: torch.Tensor,
                top_down_signal: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        层次前向传播
        
        Args:
            inputs: (batch_size, input_dim) 输入特征
            top_down_signal: 自顶向下的信号（用于注意力引导）
        
        Returns:
            level_representations: 各层的表示
            symbolic_output: 符号输出
            attention_weights: 注意力权重
        """
        batch_size = inputs.shape[0]
        
        # 自底向上编码
        level_reps = []
        current = inputs
        for encoder in self.level_encoders:
            current = encoder(current)
            level_reps.append(current)
        
        # 自顶向下反馈
        if top_down_signal is not None:
            for i, projection in enumerate(self.top_down_projections):
                level_idx = len(level_reps) - 2 - i
                if level_idx >= 0:
                    feedback = projection(top_down_signal)
                    level_reps[level_idx] = level_reps[level_idx] + feedback
                    top_down_signal = level_reps[level_idx]
        
        # 层次注意力
        high_level = level_reps[-1].unsqueeze(1)  # (batch, 1, dim)
        if top_down_signal is not None:
            query = top_down_signal.unsqueeze(1)
        else:
            query = high_level
        
        attended, attention_weights = self.hierarchical_attention(
            query, high_level, high_level
        )
        
        # 生成符号输出
        symbolic_output = self.symbol_decoder(attended.squeeze(1))
        
        return {
            'level_representations': level_reps,
            'symbolic_output': symbolic_output,
            'attention_weights': attention_weights,
            'final_representation': attended.squeeze(1)
        }
    
    def get_abstraction_path(self, 
                            inputs: torch.Tensor,
                            target_level: int) -> List[torch.Tensor]:
        """
        获取从输入到目标抽象级别的路径
        
        Returns:
            path: 从输入到目标级别的所有表示
        """
        path = [inputs]
        current = inputs
        
        for i, encoder in enumerate(self.level_encoders):
            current = encoder(current)
            path.append(current)
            if i == target_level:
                break
        
        return path
    
    def interpolate_representations(self,
                                   rep1: torch.Tensor,
                                   rep2: torch.Tensor,
                                   levels: List[int]) -> List[torch.Tensor]:
        """
        在不同层次上插值两个表示
        
        用于概念间的平滑过渡和类比推理。
        """
        interpolated = []
        
        for level in levels:
            # 在指定层次获取表示
            path1 = self.get_abstraction_path(rep1, level)
            path2 = self.get_abstraction_path(rep2, level)
            
            level_rep1 = path1[-1]
            level_rep2 = path2[-1]
            
            # 线性插值
            for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                interp = (1 - alpha) * level_rep1 + alpha * level_rep2
                interpolated.append(interp)
        
        return interpolated


class BidirectionalNeuralSymbolic(nn.Module):
    """
    双向神经符号系统
    
    整合神经到符号和符号到神经的转换，
    实现端到端的双向推理。
    """
    
    def __init__(self,
                 neural_input_dim: int,
                 num_symbols: int,
                 embedding_dim: int = 128):
        super().__init__()
        self.neural_input_dim = neural_input_dim
        self.num_symbols = num_symbols
        self.embedding_dim = embedding_dim
        
        # 神经到符号翻译器
        self.neural_to_symbolic = NeuralToSymbolicTranslator(
            neural_input_dim, num_symbols, embedding_dim
        )
        
        # 符号到神经翻译器
        self.symbolic_to_neural = SymbolicToNeuralTranslator(
            num_symbols, num_symbols, neural_input_dim, embedding_dim
        )
        
        # 概念学习器
        self.concept_learner = ConceptLearner(
            neural_input_dim, num_symbols, embedding_dim
        )
        
        # 层次表示学习
        self.hierarchical_learning = HierarchicalRepresentationLearning(
            neural_input_dim
        )
        
        # 循环一致性损失权重
        self.cycle_consistency_weight = 0.5
        
        # 符号推理网络
        self.symbolic_reasoning = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=512,
                batch_first=True
            ),
            num_layers=2
        )
    
    def forward(self, neural_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        双向前向传播
        
        Returns:
            包含神经表示、符号表示及其转换的结果字典
        """
        batch_size = neural_input.shape[0]
        
        # 1. 神经到符号
        symbol_probs, symbol_ids, attributes = self.neural_to_symbolic(neural_input)
        
        # 2. 学习概念
        concept_probs, abstraction, hierarchy = self.concept_learner(neural_input)
        
        # 3. 层次表示
        hierarchical_result = self.hierarchical_learning(neural_input)
        
        # 4. 符号到神经（循环一致性）
        symbolic_input = F.one_hot(symbol_ids, self.num_symbols).float()
        reconstructed_neural = self.symbolic_to_neural(symbolic_input)
        
        # 5. 符号推理
        symbol_seq = self.symbol_embeddings(symbol_ids).unsqueeze(1)
        reasoned_symbols = self.symbolic_reasoning(symbol_seq)
        
        return {
            # 神经到符号
            'symbol_probs': symbol_probs,
            'symbol_ids': symbol_ids,
            'symbol_attributes': attributes,
            
            # 概念学习
            'concept_probs': concept_probs,
            'abstraction': abstraction,
            'hierarchy_scores': hierarchy,
            
            # 层次表示
            'level_reps': hierarchical_result['level_representations'],
            'symbolic_output': hierarchical_result['symbolic_output'],
            
            # 循环重建
            'reconstructed_neural': reconstructed_neural,
            
            # 符号推理
            'reasoned_symbols': reasoned_symbols.squeeze(1)
        }
    
    def compute_cycle_consistency_loss(self, 
                                      neural_input: torch.Tensor) -> torch.Tensor:
        """
        计算循环一致性损失
        
        确保神经->符号->神经的转换保持信息。
        """
        # 神经到符号
        symbol_probs, _, _ = self.neural_to_symbolic(neural_input)
        
        # 符号到神经
        reconstructed = self.symbolic_to_neural(symbol_probs)
        
        # 重建损失
        loss = F.mse_loss(reconstructed, neural_input)
        
        return loss
    
    def symbolic_inference(self,
                          premises: List[int],
                          rules: List[Tuple[int, int]]) -> List[int]:
        """
        符号推理
        
        在符号空间进行逻辑推理。
        """
        # 编码前提
        premise_embs = self.symbol_embeddings(torch.tensor(premises))
        
        # 应用推理规则
        conclusions = set(premises)
        
        for premise, conclusion in rules:
            if premise in conclusions:
                conclusions.add(conclusion)
        
        return list(conclusions)
    
    def explain_prediction(self, 
                          neural_input: torch.Tensor,
                          prediction: int) -> Dict[str, Any]:
        """
        生成神经预测的可解释符号描述
        
        Returns:
            包含符号解释的字典
        """
        # 获取符号表示
        result = self.forward(neural_input.unsqueeze(0))
        symbol_id = result['symbol_ids'].item()
        concept_probs = result['concept_probs'][0]
        
        # 找到激活的概念
        activated_concepts = torch.where(concept_probs > 0.3)[0].tolist()
        
        # 构建解释
        explanation = {
            'prediction': prediction,
            'primary_symbol': symbol_id,
            'activated_concepts': activated_concepts,
            'concept_probabilities': concept_probs[activated_concepts].tolist(),
            'hierarchy_level': result['hierarchy_scores'][0].argmax().item(),
            'reasoning_path': self._trace_reasoning_path(neural_input, symbol_id)
        }
        
        return explanation
    
    def _trace_reasoning_path(self, 
                             neural_input: torch.Tensor,
                             target_symbol: int) -> List[str]:
        """追踪推理路径"""
        path = []
        
        # 获取层次表示
        hier_result = self.hierarchical_learning(neural_input.unsqueeze(0))
        level_reps = hier_result['level_representations']
        
        for i, rep in enumerate(level_reps):
            # 检查该层的激活情况
            activation = torch.norm(rep, dim=-1).item()
            path.append(f"Level {i}: activation={activation:.3f}")
        
        return path


# ==================== 实用函数 ====================

def create_material_concept_hierarchy() -> ConceptHierarchy:
    """创建材料科学概念层次结构"""
    
    # 根概念：材料
    material = Symbol("Material", SymbolType.CONCEPT, {
        'description': 'Any substance used in construction or manufacturing'
    })
    root = ConceptHierarchy(material)
    
    # 一级概念：导体、半导体、绝缘体
    conductor = Symbol("Conductor", SymbolType.CONCEPT, {
        'conductivity': 'high',
        'band_gap': 'zero'
    })
    semiconductor = Symbol("Semiconductor", SymbolType.CONCEPT, {
        'conductivity': 'moderate',
        'band_gap': 'small'
    })
    insulator = Symbol("Insulator", SymbolType.CONCEPT, {
        'conductivity': 'low',
        'band_gap': 'large'
    })
    
    conductor_node = ConceptHierarchy(conductor)
    semiconductor_node = ConceptHierarchy(semiconductor)
    insulator_node = ConceptHierarchy(insulator)
    
    root.add_child(conductor_node)
    root.add_child(semiconductor_node)
    root.add_child(insulator_node)
    
    # 二级概念：具体材料
    metals = [
        Symbol("Copper", SymbolType.CONCEPT, {'group': 'IB', 'structure': 'fcc'}),
        Symbol("Silver", SymbolType.CONCEPT, {'group': 'IB', 'structure': 'fcc'}),
        Symbol("Gold", SymbolType.CONCEPT, {'group': 'IB', 'structure': 'fcc'}),
    ]
    
    for metal in metals:
        conductor_node.add_child(ConceptHierarchy(metal))
    
    semiconductors = [
        Symbol("Silicon", SymbolType.CONCEPT, {'group': 'IV', 'structure': 'diamond'}),
        Symbol("Germanium", SymbolType.CONCEPT, {'group': 'IV', 'structure': 'diamond'}),
        Symbol("GaAs", SymbolType.CONCEPT, {'group': 'III-V', 'structure': 'zincblende'}),
    ]
    
    for semi in semiconductors:
        semiconductor_node.add_child(ConceptHierarchy(semi))
    
    return root


def symbol_to_neural_pipeline(symbols: List[Symbol],
                              relations: List[Tuple[int, int, int]],
                              output_dim: int = 128) -> torch.Tensor:
    """符号到神经的完整流水线"""
    
    num_symbols = len(symbols)
    translator = SymbolicToNeuralTranslator(num_symbols, len(relations), output_dim)
    
    # 构建符号存在矩阵
    symbol_matrix = torch.eye(num_symbols)
    
    # 转换
    relations_tensor = torch.tensor(relations)
    neural_repr = translator(symbol_matrix, relations_tensor)
    
    return neural_repr


if __name__ == "__main__":
    print("=" * 60)
    print("符号神经网络测试")
    print("=" * 60)
    
    # 测试1: 神经到符号翻译
    print("\n测试1: 神经到符号翻译")
    neural_input = torch.randn(4, 128)
    translator = NeuralToSymbolicTranslator(128, 10, 64)
    symbol_probs, symbol_ids, attributes = translator(neural_input)
    print(f"输入形状: {neural_input.shape}")
    print(f"符号概率形状: {symbol_probs.shape}")
    print(f"符号ID: {symbol_ids}")
    
    # 测试2: 概念学习
    print("\n测试2: 概念学习")
    concept_learner = ConceptLearner(128, 5, 64)
    examples = torch.randn(10, 128)
    concept_probs, abstraction, hierarchy = concept_learner(examples)
    print(f"概念概率形状: {concept_probs.shape}")
    print(f"抽象表示形状: {abstraction.shape}")
    print(f"层次分数形状: {hierarchy.shape}")
    
    # 测试3: 层次表示学习
    print("\n测试3: 层次表示学习")
    hierarchical = HierarchicalRepresentationLearning(128, [256, 128, 64])
    result = hierarchical.forward(examples)
    print(f"层数: {len(result['level_representations'])}")
    print(f"符号输出形状: {result['symbolic_output'].shape}")
    
    # 测试4: 双向神经符号系统
    print("\n测试4: 双向神经符号系统")
    bidirectional = BidirectionalNeuralSymbolic(128, 10, 64)
    result = bidirectional.forward(examples[:4])
    print(f"符号概率形状: {result['symbol_probs'].shape}")
    print(f"概念概率形状: {result['concept_probs'].shape}")
    print(f"重建神经形状: {result['reconstructed_neural'].shape}")
    
    # 测试5: 概念层次结构
    print("\n测试5: 概念层次结构")
    hierarchy = create_material_concept_hierarchy()
    print(f"根概念: {hierarchy.concept.name}")
    print(f"子概念数: {len(hierarchy.children)}")
    descendants = hierarchy.get_descendants()
    print(f"后代概念: {[d.name for d in descendants]}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
