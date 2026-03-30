# 专题2：表面态与费米弧分析研究报告

## 1. 理论基础

### 1.1 体-边对应原理（Bulk-Boundary Correspondence）

体-边对应原理是拓扑物态的核心概念：
- **三维拓扑绝缘体**：体态绝缘，表面存在导电的狄拉克锥表面态
- **Weyl半金属**：手性相反的Weyl点在表面投影之间通过费米弧连接
- **拓扑保护**：表面态/费米弧受拓扑不变量保护，对微扰具有鲁棒性

### 1.2 拓扑绝缘体表面态特征

#### 1.2.1 Bi2Se3系列拓扑绝缘体
- **能隙**：约0.34 eV（Bi2Se3）
- **表面态**：单个狄拉克锥位于Γ点
- **自旋纹理**：螺旋自旋结构，自旋-动量锁定
- **ARPES验证**：角分辨光电子能谱直接观测到狄拉克锥

#### 1.2.2 自旋纹理（Spin Texture）
- **面内自旋**：沿费米面切线方向旋转（右手螺旋）
- **面外自旋**：由于六角变形产生z方向自旋分量，周期为sin(3θ)
- **自旋轨道耦合**：导致自旋-轨道纹理，可用光偏振调控

### 1.3 Weyl半金属与费米弧

#### 1.3.1 Weyl点特征
- **定义**：三维动量空间中非简并的线性能带交叉点
- **手性**：每个Weyl点具有+1或-1的手性（Chern数）
- **成对出现**：Weyl点总是成对出现，总手性为零

#### 1.3.2 费米弧（Fermi Arc）
- **形成机制**：手性相反的Weyl点在表面布里渊区的投影通过费米弧连接
- **几何形状**：开放的弧状费米面（不同于闭合的费米口袋）
- **拓扑保护**：费米弧受Weyl点拓扑保护，稳定存在

#### 1.3.3 典型材料
- **TaAs家族**：TaAs、TaP、NbAs、NbP
- **磁性Weyl半金属**：Co3Sn2S2、HgCr2Se4
- **Type-I/II Weyl点**：
  - Type-I：点状费米面
  - Type-II：倾斜锥，破坏洛伦兹不变性

## 2. 计算方法

### 2.1 超胞方法（Slab Method）

**基本原理**：
- 构建有限厚度的板状结构（通常10-50层）
- 平面内保持周期性边界条件
- 垂直方向为开边界，产生两个表面

**实现步骤**：
```python
# PythTB示例：构建Slab结构
def make_slab(model, num_layers, direction=0):
    """从体材料模型构建slab结构"""
    slab_model = model.cut_piece(num_layers, direction)
    return slab_model

# 计算表面态能带
k_path = [[-0.5, 0], [0.5, 0]]  # k空间路径
(k_vec, k_dist, k_node) = slab_model.k_path(k_path, 100)
evals = slab_model.solve_all(k_vec)
```

**优缺点**：
- ✅ 概念简单，易于实现
- ✅ 可同时获得两个表面的态
- ❌ 受有限厚度效应影响
- ❌ 计算量随厚度增加而增大

### 2.2 表面格林函数方法

#### 2.2.1 理论基础

对于半无限体系，哈密顿量呈块三对角形式：

$$H = \begin{pmatrix}
H_{00} & H_{01} & & \\
H_{01}^\dagger & H_{00} & H_{01} & \\
& H_{01}^\dagger & H_{00} & \cdots \\
& & \vdots & \ddots
\end{pmatrix}$$

表面格林函数满足：
$$G_{00}^{-1} = (\omega + i\eta - H_{00}) - \Sigma$$

其中自能Σ描述半无限体部分的影响。

#### 2.2.2 迭代算法（Sancho方法）

**核心思想**：通过迭代计算有效哈密顿量，快速收敛到表面格林函数。

**迭代公式**：
```
初始化：
α₀ = H₀₁, β₀ = H₀₁†, ε₀ = H₀₀, ε₀ˢ = H₀₀

迭代步骤：
αᵢ = αᵢ₋₁(ω - εᵢ₋₁)⁻¹αᵢ₋₁
βᵢ = βᵢ₋₁(ω - εᵢ₋₁)⁻¹βᵢ₋₁
εᵢ = εᵢ₋₁ + αᵢ₋₁(ω - εᵢ₋₁)⁻¹βᵢ₋₁ + βᵢ₋₁(ω - εᵢ₋₁)⁻¹αᵢ₋₁
εᵢˢ = εᵢ₋₁ˢ + αᵢ₋₁(ω - εᵢ₋₁)⁻¹βᵢ₋₁

收敛后：
Gₛᵤᵣf = (ω + iη - εᶠⁱⁿᵃˡˢ)⁻¹
```

**Python实现**：
```python
import numpy as np

def surface_green_function(omega, h00, h01, eta=1e-10, max_iter=100):
    """
    计算表面格林函数（Sancho迭代法）
    
    参数:
        omega: 能量
        h00: 层内哈密顿量
        h01: 层间跃迁
        eta: 无穷小量
        max_iter: 最大迭代次数
    """
    # 初始化
    alpha = h01 @ np.linalg.inv((omega + 1j*eta) * np.eye(h00.shape[0]) - h00) @ h01
    beta = h01.T.conj() @ np.linalg.inv((omega + 1j*eta) * np.eye(h00.shape[0]) - h00) @ h01.T.conj()
    eps = h00 + h01 @ np.linalg.inv((omega + 1j*eta) * np.eye(h00.shape[0]) - h00) @ h01.T.conj() + \
          h01.T.conj() @ np.linalg.inv((omega + 1j*eta) * np.eye(h00.shape[0]) - h00) @ h01
    eps_s = h00 + h01 @ np.linalg.inv((omega + 1j*eta) * np.eye(h00.shape[0]) - h00) @ h01.T.conj()
    
    # 迭代
    for _ in range(max_iter):
        g_inv = np.linalg.inv((omega + 1j*eta) * np.eye(h00.shape[0]) - eps)
        
        alpha_new = alpha @ g_inv @ alpha
        beta_new = beta @ g_inv @ beta
        eps_new = eps + alpha @ g_inv @ beta + beta @ g_inv @ alpha
        eps_s_new = eps_s + alpha @ g_inv @ beta
        
        # 检查收敛
        if np.allclose(alpha_new, alpha, atol=1e-12) and np.allclose(beta_new, beta, atol=1e-12):
            break
            
        alpha, beta, eps, eps_s = alpha_new, beta_new, eps_new, eps_s_new
    
    # 计算表面格林函数
    g_surf = np.linalg.inv((omega + 1j*eta) * np.eye(h00.shape[0]) - eps_s)
    return g_surf

# 计算局域态密度（LDOS）
def calculate_ldos(omega_vals, h00, h01, surface_atoms=None):
    """计算表面局域态密度"""
    ldos = []
    for omega in omega_vals:
        g_surf = surface_green_function(omega, h00, h01)
        # LDOS = -Im(Tr(G)) / π
        if surface_atoms is None:
            dos = -np.imag(np.trace(g_surf)) / np.pi
        else:
            dos = -np.imag(np.trace(g_surf[np.ix_(surface_atoms, surface_atoms)])) / np.pi
        ldos.append(dos)
    return np.array(ldos)
```

### 2.3 Wannier函数基紧束缚模型

#### 2.3.1 Wannier90 + WannierTools流程

**计算流程**：
1. 第一性原理计算（VASP/Quantum ESPRESSO）
2. Wannier90投影得到紧束缚模型
3. WannierTools计算表面态和费米弧

**WannierTools输入参数**：
```fortran
! WT.in 示例
&PARAMETERS
  Eta_Arc = 0.001      ! 费米弧展宽
  E_arc = 0.0          ! 费米弧能量
  OmegaMin = -1.0      ! 能量下限
  OmegaMax = 1.0       ! 能量上限
  Nk1 = 100            ! k点网格
  Nk2 = 100
  NP = 2               ! 主层数
/
```

#### 2.3.2 表面格林函数递归计算

WannierTools使用递推矩阵方法计算表面格林函数：
```
G_surf = (ω - εᵢˢ)⁻¹

其中 εᵢˢ 通过迭代获得：
εᵢˢ = εᵢ₋₁ˢ + αᵢ₋₁(ω - εᵢ₋₁)⁻¹βᵢ₋₁
```

## 3. 费米弧的识别与表征

### 3.1 费米弧判据

#### 3.1.1 计数判据
对于Weyl半金属，费米弧可通过以下判据识别：

1. **开放费米面判据**：费米弧为开放曲线，与闭合的体态费米口袋不同
2. **连接判据**：费米弧连接手性相反的Weyl点投影
3. **色散判据**：沿垂直于费米弧方向存在线性色散（Weyl锥）
4. **计数论证**：穿过闭合回路的能带交叉次数

#### 3.1.2 自旋纹理分析
- 费米弧上存在自旋极化
- 自旋方向与费米弧切线方向相关
- 可用于区分拓扑表面态和平庸表面态

### 3.2 典型材料费米弧特征

#### 3.2.1 TaAs（001）表面
- **费米弧形状**：蝴蝶结形（bowtie）、蝌蚪形（tadpole）
- **连接方式**：连接相邻Weyl点
- **实验验证**：ARPES观测与理论计算一致

#### 3.2.2 Co3Sn2S2（001）表面
- **磁性Weyl半金属**
- **费米弧**：连接磁性Weyl点
- **反常霍尔效应**：与Weyl点手性相关

#### 3.2.3 HgTe（压缩应变）
- **应变诱导Weyl点**：压缩应变破坏反演对称性
- **费米弧密度**：约 0.8×10¹¹ cm⁻²（1%应变）
- **（001）表面**：四个费米弧呈变形环状分布

## 4. 计算实践

### 4.1 PythTB计算表面态示例

```python
from pythtb import *
import numpy as np
import matplotlib.pyplot as plt

# 定义BHZ模型（2D拓扑绝缘体）
def make_bhz_model(M=2.0, v=0.5):
    lat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    orb = [[0.0, 0.0, 0.0] for _ in range(4)]
    
    model = tb_model(3, 3, lat, orb)
    
    # 设置参数
    model.set_onsite([M, M, -M, -M])
    
    # 跃迁项设置...
    # (完整代码见研究资料)
    
    return model

# 构建slab
model_3d = make_bhz_model()
slab = model_3d.cut_piece(20, 0)  # 沿x方向切割20层

# 计算能带
k_path = [[-0.5, 0], [0, 0], [0.5, 0]]
k_vec, k_dist, k_node = slab.k_path(k_path, 100)
evals = slab.solve_all(k_vec)

# 绘图
fig, ax = plt.subplots()
for band in range(evals.shape[0]):
    ax.plot(k_dist, evals[band], 'b-', linewidth=0.5)
ax.set_xlabel('k')
ax.set_ylabel('Energy')
plt.show()
```

### 4.2 Z2Pack与表面态

```python
import z2pack
import matplotlib.pyplot as plt

# 定义系统
system = z2pack.hm.System(hamiltonian, bands=2)

# 计算表面态（通过WCC）
result = z2pack.surface.run(
    system=system,
    surface=lambda s, t: [t, s, 0],  # kz=0表面
    num_lines=50,
    pos_tol=1e-2
)

# 绘制WCC演化
fig, ax = plt.subplots()
z2pack.plot.wcc(result, axis=ax)
plt.show()
```

## 5. 前沿进展

### 5.1 新型费米弧
- **II型Weyl半金属**：倾斜Weyl锥导致的特殊费米弧
- **多极Weyl点**：高阶拓扑导致的多分支费米弧
- **非厄米Weyl半金属**： exceptional rings连接的费米弧

### 5.2 实验探测技术
- **ARPES**：直接观测能带色散和费米面
- **自旋分辨ARPES**：测量自旋纹理
- **CD-ARPES**：圆二色性测量轨道纹理
- **STM/STS**：实空间局域态密度成像

### 5.3 应用前景
- **低功耗电子学**：拓扑保护减少背散射
- **自旋电子学**：利用自旋-动量锁定
- **量子计算**：Majorana零能模

## 6. 常见问题与解决方案

### 6.1 有限厚度效应
**问题**：slab太薄导致表面态杂化  
**解决**：
- 增加slab厚度（通常>10层）
- 使用格林函数方法

### 6.2 表面终止面选择
**问题**：不同终止面导致不同表面态  
**解决**：
- 比较多种终止面的能量
- 与实验ARPES结果对照

### 6.3 数值收敛性
**问题**：格林函数迭代不收敛  
**解决**：
- 调整能量虚部η
- 增加迭代次数
- 检查哈密顿量构建

## 参考文献

1. Hasan, M.Z., Kane, C.L. (2010). Colloquium: Topological insulators.
2. Xu, S.Y., et al. (2015). Discovery of Weyl semimetal TaAs.
3. Lopez Sancho, M.P., et al. (1985). Highly convergent schemes for surface Green functions.
4. Wu, Q., et al. (2018). WannierTools: Software for topological materials.
5. Zhang, H., et al. (2009). Topological insulators in Bi2Se3, Bi2Te3.

---
**研究时间**：2026-03-08  
**专题状态**：✅ 完成
