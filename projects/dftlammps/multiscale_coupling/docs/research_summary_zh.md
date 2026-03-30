# 多尺度耦合与跨尺度建模研究总结

## 1. QM/MM、QM/MD、MD/连续介质耦合最新方法 (2024-2025)

### 1.1 MiMiC框架
- **来源**: Levy et al., CHIMIA 2025, 79, 220-223
- **特点**: 
  - 采用client-server架构，MPMD (Multiple-Program Multiple-Data) 模型
  - 高度灵活的外部程序耦合
  - 保持高计算效率，避免重复启动开销
  - 支持CPMD、CP2K、GROMACS、OpenMM
  - 短程(SR)和长程(LR)相互作用分离

### 1.2 加性QM/MM耦合方案
```
E_QM/MM = E_QM(QM) + E_MM(MM) + E_QM/MM^int(QM+MM)
```

其中耦合项包括:
- **键合项**: 当QM-MM边界切断共价键时
- **范德华相互作用**: Lennard-Jones类型
- **静电相互作用**: 
  - 机械嵌入: 固定电荷
  - 静电嵌入: QM电子密度被MM电荷极化
  - 极化嵌入: 允许QM和MM相互极化

### 1.3 边界处理方法
1. **链接原子法(Link Atom)**: 在切断的键上添加H原子
2. **边界原子法(Boundary Atom)**: 使用杂化轨道
3. **外推法**: 通过势函数外推

### 1.4 最新进展 (2024-2025)
- **自适应QM/MM**: 动态调整QM区域大小
- **机器学习QM/MM**: 用ML势替代部分QM计算
- **系统收敛性研究**: QM区域大小的系统收敛性测试

## 2. 机器学习跨尺度桥接方法

### 2.1 粗粒化映射学习

#### 自编码器方法 (Husic et al., 2020)
- 使用Gumbel-softmax重参数化学习离散CG映射
- 变分优化编码器和解码器网络
- 最小化重构损失和力波动

**损失函数**:
```
L_ae = E[(D(E(x)) - x)² + ρ F_inst(E(x))²]
```

#### 力匹配方法
- 从原子力计算CG平均力
- 约束NVT模拟获取平均力
- 迭代训练工作流程

### 2.2 CG-GNN力场 (2024)
- **来源**: Strachan Group, npj Comput. Mater. 2024
- **特点**:
  - 分子内和分子间相互作用分离
  - E(3)等变架构
  - 图Transformer消息传递
  - 迭代训练与NN解码器重构

**架构特点**:
- 节点特征: CG粒子类型
- 边特征: 距离、角度信息
- 输出: CG粒子受力

## 3. 图神经网络在跨尺度建模中的应用

### 3.1 分子表示学习

#### GemNet (2021-2023)
- 方向性消息传递
- 包含双角(dihedral angles)的几何信息
- 在QM9等数据集上SOTA性能

#### MEGNet (2019)
- 全局状态变量融入
- 统一分子和晶体材料模型
- 多保真度数据训练

#### DimeNet++ / SphereNet
- 球面消息传递
- 3D几何信息编码
- 旋转等变性保证

### 3.2 多尺度GNN架构

#### 跨尺度注意力机制
```python
# 从CG查询原子特征
query = W_q @ h_cg
keys = W_k @ h_atom
values = W_v @ h_atom

# 注意力权重
attention = softmax(keys @ query / sqrt(d))

# 更新CG特征
h_cg_updated = h_cg + attention @ values
```

#### E(3)等变性保证
- 标量特征: 距离、角度
- 向量特征: 方向向量
- 张量积: 组合不同阶特征

### 3.3 训练策略
1. **迭代训练**: 主动学习扩充数据集
2. **多保真度训练**: 结合DFT和半经验方法
3. **迁移学习**: 元素嵌入预训练
4. **不确定性量化**: 贝叶斯神经网络

## 4. 验证方法

### 4.1 能量守恒验证
- 检查总能量漂移
- 线性拟合能量时间序列
- 容忍度: 相对漂移 < 10^-4

### 4.2 力一致性验证
```
F_CG^i = Σ(j∈bead_i) F_atom^j
```
- CG力应等于原子力之和
- 相对误差容忍度: < 1%

### 4.3 热力学一致性
- 能量分布比较
- 涨落分析 (热容代理)
- 自由能面比较

### 4.4 结构一致性
- RMSD比较
- 径向分布函数
- 角分布函数

## 5. 实现要点

### 5.1 模块架构
```
dftlammps/multiscale_coupling/
├── qmmm/          # QM/MM接口
├── ml_cg/         # ML粗粒化
├── gnn_models/    # GNN模型
├── validation/    # 验证工具
├── utils/         # 工具函数
└── examples/      # 示例代码
```

### 5.2 关键类设计
- `QMMMInterface`: 主QM/MM接口
- `VASPEngine`/`LAMMPSEngine`: QM/MM引擎
- `CoarseGrainer`: 粗粒化基类
- `CGGNN`: CG图神经网络
- `CrossScaleValidator`: 跨尺度验证器

### 5.3 未来发展方向
1. **端到端可微分模拟**
2. **不确定性量化集成**
3. **实时自适应多尺度**
4. **大规模并行训练**

## 参考文献

1. Levy et al. (2025). Multiscale Molecular Dynamics with MiMiC. CHIMIA 79, 220-223.
2. Husic et al. (2020). Coarse graining molecular dynamics with graph neural networks. J. Chem. Phys. 153, 194101.
3. Batzner et al. (2021). E(3)-Equivariant graph neural networks. Nat. Commun. 13, 1-11.
4. Gasteiger et al. (2020). Directional message passing. ICLR.
5. Chen & Ong (2022). Universal graph deep learning interatomic potential. Nat. Comput. Sci. 2, 718-728.
6. Ruza et al. (2020). Temperature-transferable coarse-graining with dual GNNs. J. Chem. Phys. 153, 164501.
