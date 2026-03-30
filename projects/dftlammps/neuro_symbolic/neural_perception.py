"""
神经感知层 - Neural Perception Layer
从数据中提取特征和模式，使用深度学习进行表征学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import warnings


@dataclass
class FeatureConfig:
    """特征提取配置"""
    input_dim: int
    hidden_dims: List[int] = None
    output_dim: int = 128
    dropout: float = 0.2
    activation: str = "relu"
    batch_norm: bool = True
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 512, 256]


@dataclass
class PatternConfig:
    """模式识别配置"""
    n_clusters: int = 10
    pattern_types: List[str] = None
    temporal_window: int = 5
    
    def __post_init__(self):
        if self.pattern_types is None:
            self.pattern_types = ["correlation", "causal", "temporal", "spatial"]


class FeatureExtractor(nn.Module):
    """
    通用特征提取器
    支持表格数据、序列数据和图数据的特征提取
    """
    
    def __init__(self, config: FeatureConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        
        dims = [config.input_dim] + config.hidden_dims + [config.output_dim]
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if config.batch_norm and i < len(dims) - 2:
                self.layers.append(nn.BatchNorm1d(dims[i+1]))
            if i < len(dims) - 2:
                self.layers.append(nn.Dropout(config.dropout))
        
        self.activation = self._get_activation(config.activation)
    
    def _get_activation(self, name: str) -> Callable:
        """获取激活函数"""
        activations = {
            "relu": F.relu,
            "leaky_relu": F.leaky_relu,
            "gelu": F.gelu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "swish": lambda x: x * torch.sigmoid(x)
        }
        return activations.get(name, F.relu)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                # 只在非最后一层应用激活
                if i < len(self.layers) - 1:
                    x = self.activation(x)
            elif isinstance(layer, nn.BatchNorm1d):
                if x.dim() == 2:
                    x = layer(x)
            # Dropout在训练时自动处理
        return x


class AttentionPerception(nn.Module):
    """
    注意力感知层
    使用自注意力机制识别重要特征和关系
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 1000
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.attention_weights = None
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, embed_dim]
            mask: 注意力掩码
            return_attention: 是否返回注意力权重
        
        Returns:
            输出张量和可选的注意力权重
        """
        x = self.pos_encoding(x)
        
        # 注册钩子以捕获注意力权重
        if return_attention:
            attention_weights = []
            
            def hook_fn(module, input, output):
                # 提取注意力权重
                if hasattr(module, 'self_attn'):
                    # 捕获自注意力权重
                    pass
            
            hooks = []
            for layer in self.transformer.layers:
                hooks.append(layer.register_forward_hook(hook_fn))
        
        output = self.transformer(x, src_key_padding_mask=mask)
        
        if return_attention:
            for hook in hooks:
                hook.remove()
        
        return output
    
    def get_attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """获取注意力热力图"""
        with torch.no_grad():
            x = self.pos_encoding(x)
            # 手动计算注意力以获取权重
            batch_size, seq_len, _ = x.shape
            
            # 简化的注意力计算
            q = x[:, :1, :]  # 使用第一个token作为query
            k = x
            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim)
            attn_weights = F.softmax(scores, dim=-1)
            
            return attn_weights


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class GraphPerception(nn.Module):
    """
    图神经网络感知层
    处理图结构数据，提取节点和图级别特征
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: Optional[int] = None,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        readout: str = "mean"
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.readout = readout
        
        self.node_embeddings = nn.ModuleList()
        self.edge_embeddings = nn.ModuleList() if edge_dim else None
        
        dims = [node_dim] + [hidden_dim] * num_layers
        
        for i in range(num_layers):
            self.node_embeddings.append(
                GraphConvLayer(dims[i], dims[i+1], edge_dim)
            )
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, node_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边属性 [num_edges, edge_dim]
            batch: 批次索引 [num_nodes]
        
        Returns:
            包含节点特征和图特征的字典
        """
        node_features = []
        
        for layer in self.node_embeddings:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            node_features.append(x)
        
        # 节点级特征
        node_output = self.output_proj(x)
        
        # 图级特征
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        graph_output = self._readout(node_output, batch)
        
        return {
            "node_features": node_output,
            "graph_features": graph_output,
            "layer_features": node_features
        }
    
    def _readout(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """图级别读出操作"""
        if self.readout == "mean":
            return self._global_mean_pool(x, batch)
        elif self.readout == "max":
            return self._global_max_pool(x, batch)
        elif self.readout == "sum":
            return self._global_add_pool(x, batch)
        else:
            return x
    
    def _global_mean_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """全局平均池化"""
        batch_size = batch.max().item() + 1
        out = torch.zeros(batch_size, x.size(1), device=x.device)
        for i in range(batch_size):
            mask = batch == i
            out[i] = x[mask].mean(dim=0)
        return out
    
    def _global_max_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """全局最大池化"""
        batch_size = batch.max().item() + 1
        out = torch.zeros(batch_size, x.size(1), device=x.device)
        for i in range(batch_size):
            mask = batch == i
            out[i] = x[mask].max(dim=0)[0]
        return out
    
    def _global_add_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """全局加和池化"""
        batch_size = batch.max().item() + 1
        out = torch.zeros(batch_size, x.size(1), device=x.device)
        for i in range(batch_size):
            mask = batch == i
            out[i] = x[mask].sum(dim=0)
        return out


class GraphConvLayer(nn.Module):
    """图卷积层"""
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: Optional[int] = None
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.linear = nn.Linear(in_dim, out_dim)
        if edge_dim:
            self.edge_linear = nn.Linear(edge_dim, out_dim)
        else:
            self.edge_linear = None
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """图卷积前向传播"""
        # 聚合邻居信息
        row, col = edge_index
        
        # 源节点特征
        src = x[row]
        
        # 如果有边特征，合并
        if edge_attr is not None and self.edge_linear:
            edge_emb = self.edge_linear(edge_attr)
            src = src + edge_emb
        
        # 聚合到目标节点
        out = torch.zeros_like(x)
        out.index_add_(0, col, src)
        
        # 归一化
        deg = torch.bincount(col, minlength=x.size(0)).float()
        deg = deg.clamp(min=1).unsqueeze(1)
        out = out / deg
        
        # 线性变换
        out = self.linear(out)
        
        return out


class PatternDetector(nn.Module):
    """
    模式检测器
    自动识别数据中的各种模式
    """
    
    def __init__(self, config: PatternConfig):
        super().__init__()
        self.config = config
        
        # 为每种模式类型创建检测器
        self.detectors = nn.ModuleDict()
        
        for ptype in config.pattern_types:
            if ptype == "correlation":
                self.detectors[ptype] = CorrelationDetector()
            elif ptype == "causal":
                self.detectors[ptype] = CausalPatternDetector()
            elif ptype == "temporal":
                self.detectors[ptype] = TemporalPatternDetector(
                    config.temporal_window
                )
            elif ptype == "spatial":
                self.detectors[ptype] = SpatialPatternDetector()
    
    def forward(
        self,
        x: torch.Tensor,
        pattern_type: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        检测模式
        
        Args:
            x: 输入数据
            pattern_type: 特定模式类型，None则检测所有
        
        Returns:
            检测到的模式字典
        """
        results = {}
        
        if pattern_type:
            if pattern_type in self.detectors:
                results[pattern_type] = self.detectors[pattern_type](x)
        else:
            for name, detector in self.detectors.items():
                results[name] = detector(x)
        
        return results
    
    def get_pattern_scores(self, x: torch.Tensor) -> Dict[str, float]:
        """获取模式得分"""
        scores = {}
        with torch.no_grad():
            patterns = self.forward(x)
            for name, pattern in patterns.items():
                if isinstance(pattern, torch.Tensor):
                    scores[name] = pattern.mean().item()
        return scores


class CorrelationDetector(nn.Module):
    """相关性模式检测器"""
    
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """检测相关性模式"""
        if x.dim() == 2:
            # 计算特征间相关性
            corr = torch.corrcoef(x.T)
            # 提取上三角矩阵（不包括对角线）
            mask = torch.triu(torch.ones_like(corr), diagonal=1).bool()
            correlations = corr[mask]
            # 返回强相关性指标
            return (correlations.abs() > self.threshold).float().mean()
        return torch.tensor(0.0)


class CausalPatternDetector(nn.Module):
    """因果模式检测器"""
    
    def __init__(self, lag: int = 1):
        super().__init__()
        self.lag = lag
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """检测时间序列中的因果关系模式"""
        if x.dim() == 2 and x.size(0) > self.lag:
            # 简单的Granger因果检验近似
            x_current = x[self.lag:]
            x_lagged = x[:-self.lag]
            
            # 计算自回归关系强度
            correlations = []
            for i in range(x.size(1)):
                for j in range(x.size(1)):
                    if i != j:
                        corr = torch.corrcoef(
                            torch.stack([x_current[:, i], x_lagged[:, j]])
                        )[0, 1]
                        correlations.append(corr.abs())
            
            if correlations:
                return torch.stack(correlations).mean()
        return torch.tensor(0.0)


class TemporalPatternDetector(nn.Module):
    """时序模式检测器"""
    
    def __init__(self, window: int = 5):
        super().__init__()
        self.window = window
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=2,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """检测时序模式"""
        if x.dim() == 2:
            # 对每个特征进行时序分析
            patterns = []
            for i in range(x.size(1)):
                seq = x[:, i:i+1].unsqueeze(0)  # [1, seq_len, 1]
                if seq.size(1) >= self.window:
                    _, (h, _) = self.lstm(seq)
                    patterns.append(h[-1].squeeze())
            
            if patterns:
                return torch.stack(patterns).mean()
        return torch.tensor(0.0)


class SpatialPatternDetector(nn.Module):
    """空间模式检测器"""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """检测空间模式"""
        if x.dim() >= 2:
            # 将数据重塑为图像格式进行检测
            n = int(np.sqrt(x.size(-1)))
            if n * n == x.size(-1):
                x_img = x.view(-1, 1, n, n)
                features = self.conv(x_img)
                return features.mean()
        return torch.tensor(0.0)


class NeuralPerceptionSystem:
    """
    神经感知系统主类
    整合多种感知模块，提供统一接口
    """
    
    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        pattern_config: Optional[PatternConfig] = None,
        device: str = "auto"
    ):
        self.device = self._get_device(device)
        self.feature_config = feature_config or FeatureConfig(input_dim=100)
        self.pattern_config = pattern_config or PatternConfig()
        
        self.feature_extractor = None
        self.pattern_detector = None
        self.attention_module = None
        self.graph_module = None
        
        self._initialized = False
    
    def _get_device(self, device: str) -> torch.device:
        """获取计算设备"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def initialize(self, input_dim: int, data_type: str = "tabular"):
        """初始化感知模块"""
        self.feature_config.input_dim = input_dim
        
        if data_type == "tabular":
            self.feature_extractor = FeatureExtractor(
                self.feature_config
            ).to(self.device)
        
        elif data_type == "sequence":
            self.attention_module = AttentionPerception(
                embed_dim=input_dim
            ).to(self.device)
        
        elif data_type == "graph":
            self.graph_module = GraphPerception(
                node_dim=input_dim
            ).to(self.device)
        
        self.pattern_detector = PatternDetector(
            self.pattern_config
        ).to(self.device)
        
        self._initialized = True
    
    def extract_features(
        self,
        data: Union[np.ndarray, torch.Tensor],
        data_type: str = "tabular"
    ) -> torch.Tensor:
        """提取特征"""
        if not self._initialized:
            if isinstance(data, np.ndarray):
                input_dim = data.shape[-1]
            else:
                input_dim = data.size(-1)
            self.initialize(input_dim, data_type)
        
        x = self._to_tensor(data)
        
        if data_type == "tabular" and self.feature_extractor:
            return self.feature_extractor(x)
        
        elif data_type == "sequence" and self.attention_module:
            return self.attention_module(x)
        
        elif data_type == "graph" and self.graph_module:
            # 需要额外的图结构信息
            return x  # 简化处理
        
        return x
    
    def detect_patterns(
        self,
        data: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, Any]:
        """检测数据模式"""
        x = self._to_tensor(data)
        
        if not self._initialized:
            self.initialize(x.size(-1))
        
        with torch.no_grad():
            patterns = self.pattern_detector(x)
        
        # 转换为可序列化的格式
        results = {}
        for name, pattern in patterns.items():
            if isinstance(pattern, torch.Tensor):
                results[name] = pattern.cpu().numpy()
            else:
                results[name] = pattern
        
        return results
    
    def get_feature_importance(
        self,
        data: Union[np.ndarray, torch.Tensor],
        target: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> np.ndarray:
        """计算特征重要性"""
        x = self._to_tensor(data)
        x.requires_grad = True
        
        features = self.extract_features(x)
        
        if target is not None:
            target_t = self._to_tensor(target)
            loss = F.mse_loss(features, target_t)
        else:
            loss = features.sum()
        
        loss.backward()
        
        importance = x.grad.abs().mean(dim=0).cpu().numpy()
        return importance
    
    def _to_tensor(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """转换为张量"""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        return data.to(self.device)
    
    def save(self, path: str):
        """保存模型"""
        state = {
            "feature_extractor": self.feature_extractor.state_dict() if self.feature_extractor else None,
            "pattern_detector": self.pattern_detector.state_dict() if self.pattern_detector else None,
            "attention_module": self.attention_module.state_dict() if self.attention_module else None,
            "graph_module": self.graph_module.state_dict() if self.graph_module else None,
            "config": {
                "feature": self.feature_config,
                "pattern": self.pattern_config
            }
        }
        torch.save(state, path)
    
    def load(self, path: str):
        """加载模型"""
        state = torch.load(path, map_location=self.device)
        
        if state["feature_extractor"] and self.feature_extractor:
            self.feature_extractor.load_state_dict(state["feature_extractor"])
        if state["pattern_detector"] and self.pattern_detector:
            self.pattern_detector.load_state_dict(state["pattern_detector"])
        if state["attention_module"] and self.attention_module:
            self.attention_module.load_state_dict(state["attention_module"])
        if state["graph_module"] and self.graph_module:
            self.graph_module.load_state_dict(state["graph_module"])


def demo():
    """演示神经感知系统"""
    print("=" * 60)
    print("神经感知系统演示")
    print("=" * 60)
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # 生成具有特定模式的数据
    t = np.linspace(0, 4*np.pi, n_samples)
    data = np.zeros((n_samples, n_features))
    
    # 添加正弦波模式
    data[:, 0] = np.sin(t) + 0.1 * np.random.randn(n_samples)
    data[:, 1] = np.cos(t) + 0.1 * np.random.randn(n_samples)
    # 添加因果关系
    data[:, 2] = 0.7 * data[:, 0] + 0.3 * np.random.randn(n_samples)
    # 添加噪声特征
    data[:, 3:] = np.random.randn(n_samples, n_features - 3)
    
    print(f"\n数据形状: {data.shape}")
    print(f"样本数: {n_samples}, 特征数: {n_features}")
    
    # 初始化系统
    config = FeatureConfig(
        input_dim=n_features,
        hidden_dims=[128, 256, 128],
        output_dim=64,
        dropout=0.2
    )
    
    system = NeuralPerceptionSystem(feature_config=config)
    
    # 提取特征
    print("\n1. 特征提取")
    features = system.extract_features(data, data_type="tabular")
    print(f"   原始特征维度: {n_features}")
    print(f"   提取后维度: {features.shape[-1]}")
    print(f"   特征统计: mean={features.mean():.4f}, std={features.std():.4f}")
    
    # 检测模式
    print("\n2. 模式检测")
    patterns = system.detect_patterns(data)
    for name, score in patterns.items():
        if isinstance(score, np.ndarray):
            print(f"   {name}: {score.mean():.4f}")
        else:
            print(f"   {name}: {score}")
    
    # 计算特征重要性
    print("\n3. 特征重要性分析")
    importance = system.get_feature_importance(data)
    top_features = np.argsort(importance)[-5:][::-1]
    print(f"   最重要的5个特征: {top_features}")
    print(f"   对应重要性: {importance[top_features]}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    
    return system, features, patterns


if __name__ == "__main__":
    demo()
