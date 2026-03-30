"""
知识迁移引擎 (Knowledge Transfer Engine)
实现模型蒸馏、参数复用等知识迁移技术

作者: DFT-LAMMPS Team
版本: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
from copy import deepcopy
import warnings
from collections import OrderedDict


@dataclass
class KnowledgeTransferConfig:
    """知识迁移配置"""
    method: str = "distillation"  # distillation, fine_tuning, feature_extraction
    temperature: float = 4.0  # 蒸馏温度
    alpha: float = 0.7  # 硬标签损失权重
    beta: float = 0.3   # 软标签损失权重
    learning_rate: float = 1e-3
    num_epochs: int = 100
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 参数复用配置
    freeze_layers: List[str] = None
    unfreeze_epoch: int = 50  # 解冻epoch
    
    # 渐进式迁移
    progressive: bool = False
    progressive_steps: int = 5
    
    # 关系迁移
    relation_lambda: float = 0.1


class KnowledgeDistillationLoss(nn.Module):
    """知识蒸馏损失"""
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算蒸馏损失
        
        Args:
            student_logits: 学生模型输出
            teacher_logits: 教师模型输出
            labels: 真实标签
        
        Returns:
            总损失
        """
        # 软标签损失 (KL散度)
        soft_student = F.log_softmax(
            student_logits / self.temperature, dim=1
        )
        soft_teacher = F.softmax(
            teacher_logits / self.temperature, dim=1
        )
        
        soft_loss = self.kl_div(soft_student, soft_teacher) * \
                   (self.temperature ** 2)
        
        # 硬标签损失 (交叉熵)
        hard_loss = self.ce_loss(student_logits, labels)
        
        # 总损失
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return total_loss, hard_loss, soft_loss


class AttentionTransferLoss(nn.Module):
    """注意力迁移损失 - 迁移注意力图"""
    
    def __init__(self, mode: str = "attention"):
        super().__init__()
        self.mode = mode  # attention, fitnet
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算注意力迁移损失
        
        使用特征图的注意力映射进行迁移
        """
        # 计算注意力图
        if self.mode == "attention":
            # 基于激活的注意力
            teacher_attention = self._compute_attention(teacher_features)
            student_attention = self._compute_attention(student_features)
        else:  # fitnet
            # 直接匹配特征
            teacher_attention = teacher_features
            student_attention = student_features
        
        # 如果维度不匹配，进行投影
        if student_attention.shape != teacher_attention.shape:
            student_attention = F.adaptive_avg_pool2d(
                student_attention.unsqueeze(1),
                teacher_attention.shape
            ).squeeze(1)
        
        # L2损失
        loss = F.mse_loss(student_attention, teacher_attention)
        
        return loss
    
    def _compute_attention(self, features: torch.Tensor) -> torch.Tensor:
        """计算注意力图"""
        # 基于激活值的注意力
        return F.normalize(features.pow(2).mean(dim=1, keepdim=True), dim=0)


class RelationKnowledgeDistillation(nn.Module):
    """关系知识蒸馏 (RKD) - 迁移样本间关系"""
    
    def __init__(
        self,
        distance_weight: float = 1.0,
        angle_weight: float = 2.0
    ):
        super().__init__()
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight
    
    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算关系蒸馏损失
        
        包括距离关系和角度关系
        """
        # 距离损失
        distance_loss = self._distance_wise_loss(
            student_embeddings, teacher_embeddings
        )
        
        # 角度损失
        angle_loss = self._angle_wise_loss(
            student_embeddings, teacher_embeddings
        )
        
        total_loss = (
            self.distance_weight * distance_loss +
            self.angle_weight * angle_loss
        )
        
        return total_loss, distance_loss, angle_loss
    
    def _distance_wise_loss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor
    ) -> torch.Tensor:
        """距离关系损失"""
        # 计算成对距离
        student_dist = self._pairwise_distance(student)
        teacher_dist = self._pairwise_distance(teacher)
        
        # Huber损失
        loss = F.smooth_l1_loss(student_dist, teacher_dist)
        
        return loss
    
    def _angle_wise_loss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor
    ) -> torch.Tensor:
        """角度关系损失"""
        n = student.size(0)
        
        if n < 3:
            return torch.tensor(0.0, device=student.device)
        
        # 采样三元组
        loss = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    student_angle = self._angle(
                        student[i], student[j], student[k]
                    )
                    teacher_angle = self._angle(
                        teacher[i], teacher[j], teacher[k]
                    )
                    
                    loss += F.smooth_l1_loss(student_angle, teacher_angle)
                    count += 1
        
        return loss / max(count, 1)
    
    def _pairwise_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        """计算成对距离"""
        n = embeddings.size(0)
        dist = torch.zeros(n, n, device=embeddings.device)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist[i, j] = F.pairwise_distance(
                    embeddings[i:i+1], embeddings[j:j+1]
                )
                dist[j, i] = dist[i, j]
        
        return dist
    
    def _angle(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        e3: torch.Tensor
    ) -> torch.Tensor:
        """计算角度"""
        v1 = e1 - e2
        v2 = e3 - e2
        
        cos_angle = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
        
        return torch.acos(torch.clamp(cos_angle, -1 + 1e-7, 1 - 1e-7))


class HintLearning(nn.Module):
    """Hint Learning - 直接指导中间层"""
    
    def __init__(self, regressor_hidden_dim: int = 256):
        super().__init__()
        self.regressor = None
        self.regressor_hidden_dim = regressor_hidden_dim
    
    def forward(
        self,
        student_hint: torch.Tensor,
        teacher_hint: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Hint损失
        
        指导学生模型的guided layer学习教师模型的hint layer
        """
        # 初始化回归器（如果需要）
        if self.regressor is None:
            self._init_regressor(
                student_hint.shape[1:],
                teacher_hint.shape[1:]
            )
        
        # 通过回归器
        student_projected = self.regressor(student_hint)
        
        # L2损失
        loss = F.mse_loss(student_projected, teacher_hint)
        
        return loss
    
    def _init_regressor(
        self,
        student_shape: Tuple,
        teacher_shape: Tuple
    ):
        """初始化回归网络"""
        # 简化的全连接回归器
        student_dim = np.prod(student_shape)
        teacher_dim = np.prod(teacher_shape)
        
        self.regressor = nn.Sequential(
            nn.Linear(student_dim, self.regressor_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.regressor_hidden_dim, teacher_dim)
        )


class KnowledgeTransferEngine:
    """知识迁移引擎主类"""
    
    def __init__(self, config: KnowledgeTransferConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.teacher_model = None
        self.student_model = None
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "soft_loss": [],
            "hard_loss": []
        }
    
    def register_teacher(self, teacher_model: nn.Module):
        """注册教师模型"""
        self.teacher_model = teacher_model.to(self.device)
        self.teacher_model.eval()
        
        # 冻结教师参数
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def register_student(self, student_model: nn.Module):
        """注册学生模型"""
        self.student_model = student_model.to(self.device)
    
    def distillation_training(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        use_attention: bool = False,
        use_relation: bool = False
    ) -> Dict[str, List[float]]:
        """
        知识蒸馏训练
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            use_attention: 是否使用注意力迁移
            use_relation: 是否使用关系知识蒸馏
        
        Returns:
            训练历史
        """
        if self.teacher_model is None or self.student_model is None:
            raise ValueError("Teacher and student models must be registered")
        
        self.student_model.train()
        
        optimizer = torch.optim.Adam(
            self.student_model.parameters(),
            lr=self.config.learning_rate
        )
        
        distill_loss = KnowledgeDistillationLoss(
            temperature=self.config.temperature,
            alpha=self.config.alpha
        )
        
        attention_loss = AttentionTransferLoss() if use_attention else None
        relation_loss = RelationKnowledgeDistillation() if use_relation else None
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            epoch_hard_loss = 0.0
            epoch_soft_loss = 0.0
            num_batches = 0
            
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # 教师预测
                with torch.no_grad():
                    teacher_output = self.teacher_model(batch_data)
                
                # 学生预测
                student_output = self.student_model(batch_data)
                
                # 蒸馏损失
                loss, hard_loss, soft_loss = distill_loss(
                    student_output, teacher_output, batch_labels
                )
                
                # 注意力迁移损失
                if use_attention:
                    # 需要获取中间特征
                    student_features = self._get_intermediate_features(
                        self.student_model, batch_data
                    )
                    teacher_features = self._get_intermediate_features(
                        self.teacher_model, batch_data
                    )
                    
                    if student_features is not None and teacher_features is not None:
                        att_loss = attention_loss(student_features, teacher_features)
                        loss += 0.1 * att_loss
                
                # 关系知识蒸馏
                if use_relation:
                    # 获取嵌入
                    student_emb = self._get_embeddings(student_output)
                    teacher_emb = self._get_embeddings(teacher_output)
                    
                    rel_loss, _, _ = relation_loss(student_emb, teacher_emb)
                    loss += self.config.relation_lambda * rel_loss
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_hard_loss += hard_loss.item()
                epoch_soft_loss += soft_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            avg_hard = epoch_hard_loss / num_batches
            avg_soft = epoch_soft_loss / num_batches
            
            self.history["train_loss"].append(avg_loss)
            self.history["hard_loss"].append(avg_hard)
            self.history["soft_loss"].append(avg_soft)
            
            # 验证
            if val_loader is not None:
                val_loss = self._evaluate(val_loader)
                self.history["val_loss"].append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                      f"Loss={avg_loss:.4f}, "
                      f"Hard={avg_hard:.4f}, "
                      f"Soft={avg_soft:.4f}")
        
        return self.history
    
    def fine_tuning(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        freeze_strategy: str = "progressive"
    ) -> Dict[str, List[float]]:
        """
        微调训练
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            freeze_strategy: 冻结策略 (progressive, fixed, none)
        """
        if self.student_model is None:
            raise ValueError("Student model must be registered")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.student_model.parameters(),
            lr=self.config.learning_rate
        )
        
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        
        for epoch in range(self.config.num_epochs):
            # 应用冻结策略
            if freeze_strategy == "progressive":
                self._apply_progressive_freezing(epoch)
            elif freeze_strategy == "fixed" and epoch == 0:
                self._freeze_layers(self.config.freeze_layers or [])
            
            self.student_model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                output = self.student_model(batch_data)
                loss = criterion(output, batch_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            history["train_loss"].append(avg_loss)
            
            # 验证
            if val_loader is not None:
                val_metrics = self._evaluate_with_acc(val_loader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_acc"].append(val_metrics["accuracy"])
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                      f"Loss={avg_loss:.4f}")
        
        return history
    
    def parameter_reuse(
        self,
        source_model: nn.Module,
        target_model: nn.Module,
        mapping: Optional[Dict[str, str]] = None
    ) -> nn.Module:
        """
        参数复用
        
        将源模型的参数复用到目标模型
        
        Args:
            source_model: 源模型
            target_model: 目标模型
            mapping: 层名映射 (source_name -> target_name)
        
        Returns:
            初始化后的目标模型
        """
        source_state = source_model.state_dict()
        target_state = target_model.state_dict()
        
        if mapping is None:
            # 自动匹配同名层
            mapping = {
                k: k for k in source_state.keys()
                if k in target_state
            }
        
        # 复制参数
        transferred = 0
        for src_name, tgt_name in mapping.items():
            if src_name in source_state and tgt_name in target_state:
                src_param = source_state[src_name]
                tgt_param = target_state[tgt_name]
                
                if src_param.shape == tgt_param.shape:
                    target_state[tgt_name] = src_param.clone()
                    transferred += 1
                else:
                    # 形状不匹配，尝试部分复制
                    if len(src_param.shape) == len(tgt_param.shape):
                        # 尝试广播
                        try:
                            target_state[tgt_name] = src_param[:tgt_param.shape[0]]
                            transferred += 1
                        except:
                            pass
        
        target_model.load_state_dict(target_state, strict=False)
        print(f"Transferred {transferred} parameters")
        
        return target_model
    
    def progressive_transfer(
        self,
        train_loader: torch.utils.data.DataLoader,
        layer_groups: List[List[str]]
    ) -> Dict[str, List[float]]:
        """
        渐进式迁移
        
        逐步解冻和训练层组
        
        Args:
            train_loader: 训练数据加载器
            layer_groups: 层组列表，按顺序训练
        """
        history = {"train_loss": [], "train_acc": []}
        
        # 初始冻结所有层
        for param in self.student_model.parameters():
            param.requires_grad = False
        
        criterion = nn.CrossEntropyLoss()
        
        for group_idx, layer_names in enumerate(layer_groups):
            print(f"\nTraining layer group {group_idx + 1}/{len(layer_groups)}")
            
            # 解冻当前组
            self._unfreeze_layers(layer_names)
            
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad,
                      self.student_model.parameters()),
                lr=self.config.learning_rate / (group_idx + 1)
            )
            
            steps_per_group = self.config.num_epochs // len(layer_groups)
            
            for epoch in range(steps_per_group):
                self.student_model.train()
                epoch_loss = 0.0
                correct = 0
                total = 0
                
                for batch_data, batch_labels in train_loader:
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    output = self.student_model(batch_data)
                    loss = criterion(output, batch_labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    _, predicted = output.max(1)
                    total += batch_labels.size(0)
                    correct += predicted.eq(batch_labels).sum().item()
                
                avg_loss = epoch_loss / len(train_loader)
                acc = 100. * correct / total
                
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{steps_per_group}: "
                          f"Loss={avg_loss:.4f}, Acc={acc:.2f}%")
        
        return history
    
    def _get_intermediate_features(
        self,
        model: nn.Module,
        x: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """获取中间层特征"""
        features = []
        
        def hook(module, input, output):
            features.append(output)
        
        # 注册hook到倒数第二层
        handles = []
        for name, module in list(model.named_children())[-2:-1]:
            handles.append(module.register_forward_hook(hook))
        
        with torch.no_grad():
            model(x)
        
        for handle in handles:
            handle.remove()
        
        return features[0] if features else None
    
    def _get_embeddings(self, output: torch.Tensor) -> torch.Tensor:
        """从输出获取嵌入"""
        # 如果是logits，使用softmax后的概率
        if output.dim() > 1 and output.size(1) > 1:
            return F.softmax(output, dim=1)
        return output
    
    def _apply_progressive_freezing(self, epoch: int):
        """应用渐进式冻结"""
        if epoch == self.config.unfreeze_epoch:
            print("Unfreezing all layers")
            for param in self.student_model.parameters():
                param.requires_grad = True
    
    def _freeze_layers(self, layer_names: List[str]):
        """冻结指定层"""
        for name, param in self.student_model.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = False
                    break
    
    def _unfreeze_layers(self, layer_names: List[str]):
        """解冻指定层"""
        for name, param in self.student_model.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = True
                    break
    
    def _evaluate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> float:
        """评估损失"""
        self.student_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                output = self.student_model(batch_data)
                loss = criterion(output, batch_labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _evaluate_with_acc(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """评估损失和准确率"""
        self.student_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                output = self.student_model(batch_data)
                loss = criterion(output, batch_labels)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
        
        return {
            "loss": total_loss / len(val_loader),
            "accuracy": 100. * correct / total
        }
    
    def evaluate_transfer_effectiveness(
        self,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """评估迁移效果"""
        self.student_model.eval()
        
        # 学生模型性能
        student_metrics = self._evaluate_with_acc(test_loader)
        
        # 教师模型性能
        teacher_metrics = None
        if self.teacher_model is not None:
            self.teacher_model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_data, batch_labels in test_loader:
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    output = self.teacher_model(batch_data)
                    _, predicted = output.max(1)
                    total += batch_labels.size(0)
                    correct += predicted.eq(batch_labels).sum().item()
            
            teacher_metrics = 100. * correct / total
        
        results = {
            "student_accuracy": student_metrics["accuracy"],
            "student_loss": student_metrics["loss"]
        }
        
        if teacher_metrics is not None:
            results["teacher_accuracy"] = teacher_metrics
            results["accuracy_ratio"] = student_metrics["accuracy"] / teacher_metrics
        
        return results


class TransferLearningScheduler:
    """迁移学习调度器"""
    
    def __init__(
        self,
        engine: KnowledgeTransferEngine,
        strategy: str = "gradual_unfreeze"
    ):
        self.engine = engine
        self.strategy = strategy
        self.stage = 0
    
    def step(self):
        """执行下一个迁移阶段"""
        if self.strategy == "gradual_unfreeze":
            self._gradual_unfreeze()
        elif self.strategy == "discriminative_fine_tuning":
            self._discriminative_fine_tuning()
        
        self.stage += 1
    
    def _gradual_unfreeze(self):
        """渐进式解冻"""
        # 从最后一层开始逐步解冻
        layers = list(self.engine.student_model.named_parameters())
        unfreeze_count = min((self.stage + 1) * 2, len(layers))
        
        for i, (name, param) in enumerate(reversed(layers)):
            if i < unfreeze_count:
                param.requires_grad = True
                print(f"Unfrozen: {name}")
    
    def _discriminative_fine_tuning(self):
        """差异化微调 - 不同层使用不同学习率"""
        # 较低层使用较小学习率
        base_lr = self.engine.config.learning_rate
        
        param_groups = []
        for i, (name, param) in enumerate(self.engine.student_model.named_parameters()):
            lr = base_lr * (0.1 ** (3 - i // 2))  # 逐层递减
            param_groups.append({"params": param, "lr": lr})
        
        return param_groups


def compare_transfer_methods(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    methods: List[str] = ["distillation", "fine_tuning", "from_scratch"]
) -> Dict[str, Dict]:
    """比较不同迁移方法的效果"""
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Training with method: {method}")
        print('='*60)
        
        # 重新初始化学生模型
        student_copy = deepcopy(student_model)
        
        config = KnowledgeTransferConfig(method=method)
        engine = KnowledgeTransferEngine(config)
        
        if method == "distillation":
            engine.register_teacher(teacher_model)
            engine.register_student(student_copy)
            engine.distillation_training(train_loader)
        elif method == "fine_tuning":
            engine.register_student(student_copy)
            # 复制教师权重作为初始
            engine.parameter_reuse(teacher_model, student_copy)
            engine.fine_tuning(train_loader)
        else:  # from_scratch
            engine.register_student(student_copy)
            engine.fine_tuning(train_loader)
        
        # 评估
        metrics = engine.evaluate_transfer_effectiveness(test_loader)
        results[method] = metrics
        
        print(f"\nResults for {method}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    
    return results


# 演示代码
if __name__ == "__main__":
    print("=" * 60)
    print("知识迁移引擎演示 (Knowledge Transfer Engine Demo)")
    print("=" * 60)
    
    # 创建简单的教师和学生模型
    class SimpleModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # 生成模拟数据
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 1000
    n_features = 50
    n_classes = 5
    
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    
    # 划分训练测试
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )
    
    print(f"\nData: {n_samples} samples, {n_features} features, {n_classes} classes")
    
    # 创建模型
    teacher = SimpleModel(n_features, 128, n_classes)
    student = SimpleModel(n_features, 64, n_classes)
    
    # 预训练教师模型
    print("\nPre-training teacher model...")
    teacher.train()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(20):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = teacher(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    
    # 评估教师
    teacher.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            output = teacher(batch_x)
            _, predicted = output.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
    
    teacher_acc = 100. * correct / total
    print(f"Teacher accuracy: {teacher_acc:.2f}%")
    
    # 知识蒸馏
    print("\n" + "=" * 60)
    print("Knowledge Distillation")
    print("=" * 60)
    
    config = KnowledgeTransferConfig(
        temperature=4.0,
        alpha=0.7,
        num_epochs=50
    )
    
    engine = KnowledgeTransferEngine(config)
    engine.register_teacher(teacher)
    engine.register_student(student)
    
    history = engine.distillation_training(train_loader)
    
    # 评估
    results = engine.evaluate_transfer_effectiveness(test_loader)
    print(f"\nDistillation Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n" + "=" * 60)
    print("Knowledge Transfer Engine Demo Complete!")
    print("=" * 60)
