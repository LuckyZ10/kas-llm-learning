"""
神经-符号桥接 - Neural-Symbolic Bridge
实现神经网络与符号推理之间的双向转换和注意力对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

# 导入神经感知和符号推理模块
from .neural_perception import (
    NeuralPerceptionSystem, FeatureConfig, PatternConfig
)
from .symbolic_reasoning import (
    SymbolicReasoner, Literal, Term, TermType, Rule, KnowledgeGraph
)


@dataclass
class BridgeConfig:
    """桥接配置"""
    neural_dim: int = 128
    symbol_dim: int = 64
    num_symbols: int = 100
    attention_heads: int = 8
    temperature: float = 0.1
    alignment_method: str = "attention"  # attention, linear, bilinear


class SymbolEmbeddings(nn.Module):
    """
    符号嵌入层
    将离散符号映射到连续向量空间
    """
    
    def __init__(
        self,
        num_symbols: int,
        embedding_dim: int,
        padding_idx: int = 0
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_symbols,
            embedding_dim,
            padding_idx=padding_idx
        )
        self.embedding_dim = embedding_dim
    
    def forward(self, symbol_indices: torch.Tensor) -> torch.Tensor:
        """获取符号嵌入"""
        return self.embedding(symbol_indices)
    
    def get_similar_symbols(
        self,
        query_idx: int,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """查找相似符号"""
        with torch.no_grad():
            query_emb = self.embedding.weight[query_idx]
            similarities = F.cosine_similarity(
                query_emb.unsqueeze(0),
                self.embedding.weight,
                dim=1
            )
            top_indices = torch.topk(similarities, top_k + 1).indices
            # 排除自身
            results = [(idx.item(), similarities[idx].item())
                      for idx in top_indices if idx != query_idx][:top_k]
            return results


class NeuralToSymbolic(nn.Module):
    """
    神经到符号转换器
    从神经网络输出中提取符号规则
    """
    
    def __init__(
        self,
        neural_dim: int,
        symbol_dim: int,
        num_predicates: int = 50,
        num_constants: int = 100
    ):
        super().__init__()
        self.neural_dim = neural_dim
        self.symbol_dim = symbol_dim
        self.num_predicates = num_predicates
        self.num_constants = num_constants
        
        # 谓词提取器
        self.predicate_extractor = nn.Sequential(
            nn.Linear(neural_dim, neural_dim // 2),
            nn.ReLU(),
            nn.Linear(neural_dim // 2, num_predicates)
        )
        
        # 参数提取器
        self.argument_extractor = nn.Sequential(
            nn.Linear(neural_dim, neural_dim // 2),
            nn.ReLU(),
            nn.Linear(neural_dim // 2, num_constants)
        )
        
        # 关系提取器（用于发现规则）
        self.relation_extractor = nn.MultiheadAttention(
            embed_dim=neural_dim,
            num_heads=8,
            batch_first=True
        )
    
    def forward(
        self,
        neural_output: torch.Tensor,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        从神经输出提取符号表示
        
        Args:
            neural_output: 神经网络输出 [batch_size, neural_dim]
            threshold: 提取阈值
        
        Returns:
            提取的符号表示列表
        """
        batch_size = neural_output.size(0)
        
        # 提取谓词
        predicate_logits = self.predicate_extractor(neural_output)
        predicate_probs = F.softmax(predicate_logits, dim=-1)
        
        # 提取参数
        argument_logits = self.argument_extractor(neural_output)
        argument_probs = F.softmax(argument_logits, dim=-1)
        
        # 构建符号表示
        results = []
        for i in range(batch_size):
            pred_idx = torch.argmax(predicate_probs[i]).item()
            pred_conf = predicate_probs[i, pred_idx].item()
            
            arg_idx = torch.argmax(argument_probs[i]).item()
            arg_conf = argument_probs[i, arg_idx].item()
            
            if pred_conf > threshold:
                results.append({
                    "predicate_idx": pred_idx,
                    "predicate_conf": pred_conf,
                    "argument_idx": arg_idx,
                    "argument_conf": arg_conf,
                    "type": "atom"
                })
        
        return results
    
    def extract_rules(
        self,
        neural_output: torch.Tensor,
        num_rules: int = 5
    ) -> List[Dict[str, Any]]:
        """
        从神经输出提取规则
        
        Args:
            neural_output: 神经网络输出
            num_rules: 要提取的规则数量
        
        Returns:
            提取的规则列表
        """
        # 自注意力发现关系
        output_attended, attention_weights = self.relation_extractor(
            neural_output.unsqueeze(0),
            neural_output.unsqueeze(0),
            neural_output.unsqueeze(0)
        )
        
        # 基于注意力权重提取规则
        rules = []
        attn = attention_weights[0]  # [seq_len, seq_len]
        
        # 查找高注意力权重的位置对
        values, indices = torch.topk(attn.view(-1), k=num_rules * 2)
        
        for idx, val in zip(indices, values):
            if val > 0.1:  # 阈值
                i = idx // attn.size(1)
                j = idx % attn.size(1)
                
                rules.append({
                    "head_idx": i.item(),
                    "body_idx": j.item(),
                    "attention": val.item(),
                    "rule_form": f"f{j} => f{i}"
                })
        
        return rules
    
    def decode_to_literals(
        self,
        neural_output: torch.Tensor,
        id_to_predicate: Dict[int, str],
        id_to_constant: Dict[int, str]
    ) -> List[Literal]:
        """解码为逻辑文字"""
        extracted = self.forward(neural_output)
        
        literals = []
        for ext in extracted:
            pred_name = id_to_predicate.get(
                ext["predicate_idx"],
                f"P{ext['predicate_idx']}"
            )
            const_name = id_to_constant.get(
                ext["argument_idx"],
                f"C{ext['argument_idx']}"
            )
            
            term = Term(const_name, TermType.CONSTANT)
            literal = Literal(pred_name, [term])
            literals.append(literal)
        
        return literals


class SymbolicToNeural(nn.Module):
    """
    符号到神经转换器
    将符号表示编码为神经向量
    """
    
    def __init__(
        self,
        symbol_dim: int,
        neural_dim: int,
        num_predicates: int = 50,
        num_constants: int = 100
    ):
        super().__init__()
        self.symbol_dim = symbol_dim
        self.neural_dim = neural_dim
        
        # 谓词嵌入
        self.predicate_embed = nn.Embedding(num_predicates, symbol_dim)
        
        # 常量嵌入
        self.constant_embed = nn.Embedding(num_constants, symbol_dim)
        
        # 组合编码器
        self.encoder = nn.Sequential(
            nn.Linear(symbol_dim * 2, symbol_dim),
            nn.ReLU(),
            nn.Linear(symbol_dim, neural_dim)
        )
        
        # 序列编码器（用于处理多个文字）
        self.sequence_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=neural_dim,
                nhead=8,
                batch_first=True
            ),
            num_layers=2
        )
    
    def encode_literal(
        self,
        predicate_idx: torch.Tensor,
        argument_indices: torch.Tensor
    ) -> torch.Tensor:
        """编码单个文字"""
        pred_emb = self.predicate_embed(predicate_idx)
        
        # 平均参数嵌入
        arg_emb = self.constant_embed(argument_indices).mean(dim=-2)
        
        # 组合
        combined = torch.cat([pred_emb, arg_emb], dim=-1)
        neural_repr = self.encoder(combined)
        
        return neural_repr
    
    def encode_literals(
        self,
        literals: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """编码多个文字"""
        if not literals:
            return torch.zeros(1, self.neural_dim)
        
        # 编码每个文字
        encoded = []
        for lit in literals:
            pred_idx = lit["predicate"]
            arg_indices = lit["arguments"]
            emb = self.encode_literal(
                torch.tensor([pred_idx]),
                torch.tensor(arg_indices).unsqueeze(0)
            )
            encoded.append(emb)
        
        # 堆叠并应用序列编码
        sequence = torch.stack(encoded, dim=1)  # [1, seq_len, neural_dim]
        output = self.sequence_encoder(sequence)
        
        # 平均池化
        return output.mean(dim=1)
    
    def encode_rule(
        self,
        head_pred: int,
        head_args: List[int],
        body_preds: List[int],
        body_args_list: List[List[int]]
    ) -> torch.Tensor:
        """编码规则"""
        # 编码头部
        head_repr = self.encode_literal(
            torch.tensor([head_pred]),
            torch.tensor([head_args])
        )
        
        # 编码体部
        body_reprs = []
        for pred, args in zip(body_preds, body_args_list):
            repr = self.encode_literal(
                torch.tensor([pred]),
                torch.tensor([args])
            )
            body_reprs.append(repr)
        
        if body_reprs:
            body_repr = torch.stack(body_reprs).mean(dim=0)
        else:
            body_repr = torch.zeros_like(head_repr)
        
        # 组合头部和体部
        return torch.cat([head_repr, body_repr], dim=-1)


class AttentionAlignment(nn.Module):
    """
    注意力对齐模块
    对齐神经注意力和符号推理路径
    """
    
    def __init__(
        self,
        neural_dim: int,
        symbol_dim: int,
        num_heads: int = 8
    ):
        super().__init__()
        self.neural_dim = neural_dim
        self.symbol_dim = symbol_dim
        self.num_heads = num_heads
        
        # 神经到符号的注意力
        self.neural_to_symbol = nn.MultiheadAttention(
            embed_dim=symbol_dim,
            num_heads=num_heads,
            batch_first=True,
            kdim=neural_dim,
            vdim=neural_dim
        )
        
        # 符号到神经的注意力
        self.symbol_to_neural = nn.MultiheadAttention(
            embed_dim=neural_dim,
            num_heads=num_heads,
            batch_first=True,
            kdim=symbol_dim,
            vdim=symbol_dim
        )
        
        # 对齐投影
        self.alignment_proj = nn.Linear(neural_dim + symbol_dim, 1)
    
    def forward(
        self,
        neural_features: torch.Tensor,
        symbol_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        双向注意力对齐
        
        Args:
            neural_features: 神经特征 [batch, seq_n, neural_dim]
            symbol_embeddings: 符号嵌入 [batch, seq_s, symbol_dim]
        
        Returns:
            (对齐后的神经特征, 对齐后的符号特征, 对齐权重)
        """
        # 神经 -> 符号
        aligned_symbols, attn_n2s = self.neural_to_symbol(
            symbol_embeddings,
            neural_features,
            neural_features
        )
        
        # 符号 -> 神经
        aligned_neural, attn_s2n = self.symbol_to_neural(
            neural_features,
            symbol_embeddings,
            symbol_embeddings
        )
        
        # 计算对齐分数
        combined = torch.cat([
            aligned_neural.mean(dim=1),
            aligned_symbols.mean(dim=1)
        ], dim=-1)
        alignment_score = torch.sigmoid(self.alignment_proj(combined))
        
        return aligned_neural, aligned_symbols, alignment_score
    
    def compute_explanation_mask(
        self,
        neural_attention: torch.Tensor,
        symbol_importance: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        计算解释掩码
        
        Args:
            neural_attention: 神经注意力权重
            symbol_importance: 符号重要性分数
            threshold: 阈值
        
        Returns:
            解释掩码
        """
        # 归一化
        neural_norm = F.softmax(neural_attention, dim=-1)
        symbol_norm = F.softmax(symbol_importance, dim=-1)
        
        # 计算一致性
        consistency = F.cosine_similarity(
            neural_norm.flatten(1),
            symbol_norm.flatten(1),
            dim=1
        )
        
        # 生成掩码
        mask = (consistency > threshold).float()
        
        return mask


class BilingualConceptSpace(nn.Module):
    """
    双语概念空间
    神经概念和符号概念的联合嵌入
    """
    
    def __init__(
        self,
        neural_dim: int,
        symbol_dim: int,
        concept_dim: int,
        num_concepts: int = 50
    ):
        super().__init__()
        self.concept_dim = concept_dim
        self.num_concepts = num_concepts
        
        # 概念原型
        self.neural_prototypes = nn.Parameter(
            torch.randn(num_concepts, concept_dim)
        )
        self.symbol_prototypes = nn.Parameter(
            torch.randn(num_concepts, concept_dim)
        )
        
        # 投影层
        self.neural_proj = nn.Linear(neural_dim, concept_dim)
        self.symbol_proj = nn.Linear(symbol_dim, concept_dim)
        
        # 概念关联
        self.concept_relation = nn.Parameter(
            torch.randn(num_concepts, num_concepts)
        )
    
    def project_neural(self, neural_features: torch.Tensor) -> torch.Tensor:
        """投影神经特征到概念空间"""
        return self.neural_proj(neural_features)
    
    def project_symbol(self, symbol_features: torch.Tensor) -> torch.Tensor:
        """投影符号特征到概念空间"""
        return self.symbol_proj(symbol_features)
    
    def get_concept_activation(
        self,
        neural_features: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """获取神经特征的概念激活"""
        projected = self.project_neural(neural_features)
        
        # 计算与概念原型的相似度
        similarities = torch.matmul(projected, self.neural_prototypes.T)
        
        # 软分配
        activation = F.softmax(similarities / temperature, dim=-1)
        
        return activation
    
    def align_concepts(
        self,
        neural_batch: torch.Tensor,
        symbol_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        对齐神经和符号概念
        
        Returns:
            对齐损失
        """
        # 获取概念激活
        neural_concepts = self.get_concept_activation(neural_batch)
        symbol_projected = self.project_symbol(symbol_batch)
        symbol_concepts = F.softmax(
            torch.matmul(symbol_projected, self.symbol_prototypes.T) / 0.1,
            dim=-1
        )
        
        # 计算对齐损失（互信息最大化）
        joint = torch.matmul(neural_concepts.T, symbol_concepts)
        joint = joint / joint.sum()
        
        marginal_n = neural_concepts.mean(dim=0)
        marginal_s = symbol_concepts.mean(dim=0)
        
        # 互信息
        mi = (joint * (torch.log(joint + 1e-10) - 
                      torch.log(marginal_n.unsqueeze(1) + 1e-10) - 
                      torch.log(marginal_s.unsqueeze(0) + 1e-10))).sum()
        
        # 最大化互信息（最小化负互信息）
        return -mi


class NeuralSymbolicBridge(nn.Module):
    """
    神经-符号桥接主类
    整合所有桥接功能
    """
    
    def __init__(self, config: Optional[BridgeConfig] = None):
        super().__init__()
        self.config = config or BridgeConfig()
        
        # 子模块
        self.symbol_embeddings = SymbolEmbeddings(
            self.config.num_symbols,
            self.config.symbol_dim
        )
        
        self.neural_to_symbolic = NeuralToSymbolic(
            self.config.neural_dim,
            self.config.symbol_dim
        )
        
        self.symbolic_to_neural = SymbolicToNeural(
            self.config.symbol_dim,
            self.config.neural_dim
        )
        
        self.attention_alignment = AttentionAlignment(
            self.config.neural_dim,
            self.config.symbol_dim,
            self.config.attention_heads
        )
        
        self.concept_space = BilingualConceptSpace(
            self.config.neural_dim,
            self.config.symbol_dim,
            self.config.symbol_dim,
            self.config.num_symbols // 2
        )
        
        # 映射字典
        self.predicate_map: Dict[str, int] = {}
        self.constant_map: Dict[str, int] = {}
        self.reverse_predicate_map: Dict[int, str] = {}
        self.reverse_constant_map: Dict[int, str] = {}
    
    def register_predicate(self, name: str) -> int:
        """注册谓词"""
        if name not in self.predicate_map:
            idx = len(self.predicate_map)
            self.predicate_map[name] = idx
            self.reverse_predicate_map[idx] = name
        return self.predicate_map[name]
    
    def register_constant(self, name: str) -> int:
        """注册常量"""
        if name not in self.constant_map:
            idx = len(self.constant_map)
            self.constant_map[name] = idx
            self.reverse_constant_map[idx] = name
        return self.constant_map[name]
    
    def neural_to_symbol(
        self,
        neural_output: torch.Tensor,
        return_literals: bool = True
    ) -> Union[List[Dict], List[Literal]]:
        """
        神经到符号转换
        
        Args:
            neural_output: 神经网络输出
            return_literals: 是否返回逻辑文字
        
        Returns:
            符号表示或逻辑文字
        """
        extracted = self.neural_to_symbolic(neural_output)
        
        if return_literals:
            return self.neural_to_symbolic.decode_to_literals(
                neural_output,
                self.reverse_predicate_map,
                self.reverse_constant_map
            )
        
        return extracted
    
    def symbol_to_neural(
        self,
        literals: List[Literal]
    ) -> torch.Tensor:
        """
        符号到神经转换
        
        Args:
            literals: 逻辑文字列表
        
        Returns:
            神经表示
        """
        # 转换为索引
        encoded = []
        for lit in literals:
            pred_idx = self.register_predicate(lit.predicate)
            arg_indices = [
                self.register_constant(arg.name)
                for arg in lit.args
            ]
            encoded.append({
                "predicate": pred_idx,
                "arguments": arg_indices
            })
        
        return self.symbolic_to_neural.encode_literals(encoded)
    
    def align(
        self,
        neural_features: torch.Tensor,
        symbol_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        对齐神经和符号表示
        
        Args:
            neural_features: 神经特征
            symbol_embeddings: 符号嵌入
        
        Returns:
            对齐结果
        """
        aligned_neural, aligned_symbol, score = self.attention_alignment(
            neural_features,
            symbol_embeddings
        )
        
        return {
            "aligned_neural": aligned_neural,
            "aligned_symbol": aligned_symbol,
            "alignment_score": score
        }
    
    def extract_explanation(
        self,
        neural_input: torch.Tensor,
        neural_output: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        提取解释
        
        Args:
            neural_input: 神经输入
            neural_output: 神经输出
            attention_weights: 注意力权重
        
        Returns:
            解释信息
        """
        explanation = {
            "input_shape": neural_input.shape,
            "output_shape": neural_output.shape
        }
        
        # 提取符号规则
        rules = self.neural_to_symbolic.extract_rules(neural_output)
        explanation["extracted_rules"] = rules
        
        # 计算概念激活
        concept_act = self.concept_space.get_concept_activation(neural_output)
        top_concepts = torch.topk(concept_act[0], k=5)
        explanation["top_concepts"] = [
            {"idx": idx.item(), "activation": act.item()}
            for idx, act in zip(top_concepts.indices, top_concepts.values)
        ]
        
        # 如果有注意力权重，提取重要特征
        if attention_weights is not None:
            important_features = torch.topk(
                attention_weights.mean(dim=1).squeeze(),
                k=min(10, attention_weights.size(-1))
            )
            explanation["important_features"] = [
                {"idx": idx.item(), "weight": w.item()}
                for idx, w in zip(important_features.indices, important_features.values)
            ]
        
        return explanation
    
    def forward(
        self,
        neural_features: torch.Tensor,
        symbols: Optional[List[Literal]] = None
    ) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            neural_features: 神经特征
            symbols: 符号表示（可选）
        
        Returns:
            桥接结果
        """
        results = {}
        
        # 神经到符号
        extracted_symbols = self.neural_to_symbol(neural_features, return_literals=False)
        results["extracted_symbols"] = extracted_symbols
        
        # 如果有符号输入，进行对齐
        if symbols is not None:
            symbol_repr = self.symbol_to_neural(symbols)
            alignment = self.align(
                neural_features.unsqueeze(0),
                symbol_repr.unsqueeze(0)
            )
            results.update(alignment)
        
        return results


def demo():
    """演示神经-符号桥接"""
    print("=" * 60)
    print("神经-符号桥接演示")
    print("=" * 60)
    
    # 创建桥接器
    config = BridgeConfig(
        neural_dim=128,
        symbol_dim=64,
        num_symbols=100
    )
    bridge = NeuralSymbolicBridge(config)
    
    print("\n1. 神经到符号转换")
    # 模拟神经输出
    neural_output = torch.randn(5, config.neural_dim)
    
    # 注册一些谓词和常量
    for i in range(10):
        bridge.register_predicate(f"pred_{i}")
        bridge.register_constant(f"const_{i}")
    
    extracted = bridge.neural_to_symbol(neural_output, return_literals=False)
    print(f"   从神经输出提取了 {len(extracted)} 个符号表示")
    for i, ext in enumerate(extracted[:3]):
        print(f"     符号 {i}: {ext}")
    
    print("\n2. 符号到神经转换")
    # 创建一些逻辑文字
    literals = [
        Literal("pred_0", [Term("const_1", TermType.CONSTANT)]),
        Literal("pred_1", [Term("const_2", TermType.CONSTANT)]),
        Literal("pred_2", [
            Term("const_1", TermType.CONSTANT),
            Term("const_3", TermType.CONSTANT)
        ])
    ]
    
    neural_repr = bridge.symbol_to_neural(literals)
    print(f"   将 {len(literals)} 个文字编码为神经表示")
    print(f"   神经表示形状: {neural_repr.shape}")
    
    print("\n3. 注意力对齐")
    neural_features = torch.randn(2, 10, config.neural_dim)
    symbol_embeddings = torch.randn(2, 15, config.symbol_dim)
    
    alignment_result = bridge.align(neural_features, symbol_embeddings)
    print(f"   对齐后神经特征形状: {alignment_result['aligned_neural'].shape}")
    print(f"   对齐后符号特征形状: {alignment_result['aligned_symbol'].shape}")
    print(f"   对齐分数: {alignment_result['alignment_score'].mean().item():.4f}")
    
    print("\n4. 概念空间")
    concept_act = bridge.concept_space.get_concept_activation(neural_output)
    print(f"   概念激活形状: {concept_act.shape}")
    print(f"   概念激活总和: {concept_act.sum(dim=1).mean().item():.4f}")
    
    # 计算对齐损失
    symbol_batch = torch.randn(5, config.symbol_dim)
    align_loss = bridge.concept_space.align_concepts(neural_output, symbol_batch)
    print(f"   对齐损失: {align_loss.item():.4f}")
    
    print("\n5. 提取解释")
    explanation = bridge.extract_explanation(
        torch.randn(1, 50),
        neural_output[:1],
        attention_weights=torch.randn(1, 1, 50)
    )
    print(f"   提取的规则数: {len(explanation['extracted_rules'])}")
    print(f"   顶级概念: {explanation['top_concepts'][:3]}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
