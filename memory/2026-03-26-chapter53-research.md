# 第五十三章 图神经网络与几何深度学习 - 深度研究笔记

## 研究时间: 2026-03-26 19:55-20:05

---

## 1. 核心文献梳理

### 1.1 GCN (Kipf & Welling, 2017) - ICLR
**核心贡献**: 首次提出可扩展的图卷积网络用于半监督学习

**关键创新**:
- 谱图卷积的一阶近似
- 层间传播的简洁规则: $H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$
- 线性复杂度于边数，结合局部图结构和节点特征

**费曼法比喻**: GCN就像社交网络中的"口碑传播"——每个人的观点都是他自己和他朋友观点的加权平均，权重取决于他们在社交网络中的连接紧密度。

### 1.2 GAT (Veličković et al., 2018) - ICLR
**核心贡献**: 将自注意力机制引入图神经网络

**关键创新**:
- 邻居加权不再基于度信息，而是基于注意力系数
- 多头注意力并行建模多种关系类型
- 稀疏注意力矩阵可解释异常数据

**注意力系数计算**:
$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T[Wh_i \| Wh_j]))}{\sum_{k \in \mathcal{N}_i} \exp(\text{LeakyReLU}(\mathbf{a}^T[Wh_i \| Wh_k]))}$$

**费曼法比喻**: GAT就像"选择性倾听"——不是听所有朋友的意见，而是根据话题的相关性，只关注那些在这个领域真正有见地的人。

### 1.3 Geometric Deep Learning (Bronstein et al., 2021)
**核心贡献**: 几何深度学习的统一框架

**"5G"框架**:
- **Grids**: 网格数据 (CNNs)
- **Groups**: 群对称性 (Equivariant NNs)
- **Graphs**: 图结构 (GNNs)
- **Geodesics**: 流形/测地线
- **Gauges**: 规范场

**费曼法比喻**: 几何深度学习提供了一个"通用翻译器"——无论数据是图像、分子、社交网络还是3D点云，都能用统一的语言描述和处理。

---

## 2. 关键技术要点

### 2.1 消息传递机制 (Message Passing)
**MPNN统一框架** (Gilmer et al., 2017):
1. **消息函数**: $m_{ij}^{(t)} = M_t(h_i^{(t-1)}, h_j^{(t-1)}, e_{ij})$
2. **聚合函数**: $m_i^{(t)} = \sum_{j \in N(i)} m_{ij}^{(t)}$
3. **更新函数**: $h_i^{(t)} = U_t(h_i^{(t-1)}, m_i^{(t)})$

GCN、GAT、GraphSAGE都是此框架的特例。

### 2.2 深层GNN的挑战
**过平滑 (Over-smoothing)**: 节点特征在多层传播后趋于一致
**过挤压 (Over-squashing)**: 远程信息在压缩到固定维度时丢失

**解决方案**:
- 残差连接: GCNII (Chen et al., 2020)
- 跳跃连接: Jumping Knowledge Networks
- DropEdge: 随机移除边

### 2.3 等变神经网络
**E(3)等变性**: 对旋转、平移、反射保持不变

**SchNet (Schütt et al., 2018)**:
- 连续滤波器卷积
- 径向基函数距离建模
- 分子能量预测SOTA

**费曼法比喻**: 等变神经网络就像"转动地图"——地图随你转动，但北方始终指向地理北方，相对关系不变。

---

## 3. 应用领域

| 领域 | 典型应用 | 代表模型 |
|------|----------|----------|
| 药物发现 | 分子性质预测 | SchNet, DimeNet, E(3)-NN |
| 社交网络 | 用户行为预测 | GCN, GraphSAGE |
| 推荐系统 | 协同过滤 | LightGCN, NGCF |
| 知识图谱 | 链接预测, 推理 | R-GCN, CompGCN |
| 3D视觉 | 点云分割, 目标检测 | PointNet++, DGCNN |
| 交通预测 | 流量预测 | STGCN, DCRNN |

---

## 4. 数学推导重点

### 4.1 图卷积的谱推导
1. 图拉普拉斯: $L = D - A$
2. 归一化拉普拉斯: $L_{sym} = D^{-1/2}LD^{-1/2}$
3. 谱卷积: $x * G g = U g_\theta U^T x$
4. Chebyshev近似 → GCN简化

### 4.2 置换不变性
- 图数据对节点编号不敏感
- 聚合函数需满足: $f(\{x_1, ..., x_n\}) = f(\{x_{\pi(1)}, ..., x_{\pi(n)}\})$
- sum/mean/max都是置换不变的

### 4.3 等变性约束
对于群G的作用 $\rho(g)$ 和 $\rho'(g)$:
$$f(\rho(g)x) = \rho'(g)f(x)$$

这限制了网络权重结构，但提高了泛化能力。

---

## 5. 代码实现要点

### 5.1 GCN Layer
```python
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj_normalized):
        # adj_normalized = D^{-1/2} A D^{-1/2}
        h = self.linear(x)
        h = torch.matmul(adj_normalized, h)
        return F.relu(h)
```

### 5.2 GAT Layer
```python
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads=8):
        super().__init__()
        self.W = nn.Linear(in_features, out_features * n_heads)
        self.a = nn.Parameter(torch.randn(n_heads, 2 * out_features))
        self.n_heads = n_heads
    
    def forward(self, x, adj):
        # Multi-head attention
        # Compute attention coefficients
        # Aggregate with attention weights
        pass
```

---

## 6. 参考文献 (APA格式 - 已确认)

1. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *International Conference on Learning Representations*.

2. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. *International Conference on Learning Representations*.

3. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. *Advances in Neural Information Processing Systems*, 30.

4. Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message passing for quantum chemistry. *International Conference on Machine Learning*, 1263-1272.

5. Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021). Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. *arXiv preprint arXiv:2104.13478*.

6. Schütt, K. T., Sauceda, H. E., Kindermans, P. J., Tkatchenko, A., & Müller, K. R. (2018). SchNet – A deep learning architecture for molecules and materials. *The Journal of Chemical Physics*, 148(24), 241722.

7. Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). PointNet: Deep learning on point sets for 3D classification and segmentation. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 652-660.

8. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? *International Conference on Learning Representations*.

---

## 7. 下一章研究计划

**第54章建议主题**: 强化学习进阶 (Advanced Reinforcement Learning)
- Model-based RL
- Offline RL
- Hierarchical RL
- Multi-agent RL进阶

或

**神经符号AI** (Neuro-symbolic AI) — 连接神经网络与符号推理

---

*研究完成时间: 2026-03-26 20:05*
*状态: 准备开始正文撰写*
