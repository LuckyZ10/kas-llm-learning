# 专题3：Wannier函数与紧束缚模型研究报告

## 1. 理论基础

### 1.1 Wannier函数的定义

Wannier函数是通过傅里叶变换将布洛赫波函数转换到实空间得到的局域化轨道：

$$|Rm\rangle = \frac{V}{(2\pi)^3} \int_{BZ} d k \, e^{-i k \cdot R} |\psi_{mk}\rangle$$

其中：
- $|Rm\rangle$：位于晶格点$R$的第$m$个Wannier函数
- $|\psi_{mk}\rangle$：布洛赫波函数
- $V$：原胞体积

### 1.2 最大局域化Wannier函数（MLWF）

**核心思想**：通过幺正变换$U^{(k)}_{mn}$最小化Wannier函数的空间展宽：

$$|Rm\rangle = \frac{V}{(2\pi)^3} \int_{BZ} d k \, e^{-i k \cdot R} \sum_n U^{(k)}_{mn} |\psi_{nk}\rangle$$

**展宽泛函**（Marzari-Vanderbilt）：
$$\Omega = \sum_n \left[ \langle w_{0n}|r^2|w_{0n}\rangle - (\langle w_{0n}|r|w_{0n}\rangle)^2 \right]$$

**分解**：
- 规范依赖部分：$\Omega_I$（Wannier函数间交叠）
- 规范独立部分：$\Omega_{OD}$（由能带结构决定）

### 1.3 部分占据Wannier函数

对于包含导带的情况：
- **固定空间**：占据态完全保留
- **活性空间**：导带中部分态参与优化
- **解纠缠**：最小化"character change"泛函

## 2. Wannier90计算流程

### 2.1 从第一性原理到紧束缚模型

**完整流程**：
```
DFT计算 → Wannier90输入 → 投影+优化 → 紧束缚哈密顿量
  (VASP/    (Wannier90)   (MLWF)      (hr.dat)
   QE/)
```

### 2.2 输入文件设置

#### 2.2.1 Wannier90主输入文件（.win）

```fortran
! 基本设置
num_wann = 18          ! Wannier函数数目
num_iter = 1000        ! 优化迭代次数

! 能量窗口
dis_win_min = -10.0    ! 外窗口下限
dis_win_max = 10.0     ! 外窗口上限
dis_froz_min = -5.0    ! 内窗口（冻结态）下限
dis_froz_max = 5.0     ! 内窗口上限

! 初始投影
begin projections
Bi: px; py; pz         ! Bi原子的p轨道
Se: px; py; pz         ! Se原子的p轨道
end projections

! k点网格
mp_grid = 6 6 6

! 紧束缚矩阵输出
hr_plot = .true.       ! 输出_hr.dat文件
```

#### 2.2.2 与DFT代码的接口

**VASP + Wannier90**：
```bash
# INCAR设置
LWANNIER90 = .TRUE.
ISYM = -1              ! 关闭对称性（推荐）

# 运行流程
vasp_std               # 自洽计算
wannier90.x -pp bi2se3 # 预处理
vasp_std               # 计算M矩阵等
wannier90.x bi2se3     # Wannier化
```

### 2.3 输出文件与处理

#### 2.3.1 核心输出文件

| 文件 | 内容 | 用途 |
|------|------|------|
| `_hr.dat` | 实空间紧束缚哈密顿量 | TB模型计算 |
| `_centres.xyz` | Wannier中心坐标 | 可视化 |
| `wannier90.wout` | 日志和收敛信息 | 诊断 |

#### 2.3.2 紧束缚哈密顿量格式

```
bi2se3_hr.dat 格式：
----------------
18                    ! num_wann
289                   ! nrpts
1 1 1 1 1 ...         ! 简并因子（每行15个）
  0   0   0   1   1   0.12345   0.00000   ! R, m, n, Re(H), Im(H)
  ...
```

**哈密顿量表达式**：
$$H_{mn}(k) = \sum_R H_{mn}(R) e^{i k \cdot (R + t_m - t_n)}$$

## 3. 典型紧束缚模型

### 3.1 SSH模型（一维）

**哈密顿量**：
$$H = \sum_n (t_1 c_{A,n}^\dagger c_{B,n} + t_2 c_{A,n+1}^\dagger c_{B,n} + \text{h.c.})$$

**Python实现**：
```python
import numpy as np

def ssh_hamiltonian(k, t1=1.0, t2=0.5):
    """
    SSH模型哈密顿量
    t1: 强跃迁 (intra-cell)
    t2: 弱跃迁 (inter-cell)
    """
    h00 = np.array([[0, t1], [t1, 0]])
    h01 = np.array([[0, 0], [t2, 0]])
    
    # 傅里叶变换
    H_k = h00 + h01 * np.exp(1j * k) + h01.T.conj() * np.exp(-1j * k)
    return H_k
```

**拓扑相**：
- $|t_1| > |t_2|$：平凡相（卷绕数0）
- $|t_1| < |t_2|$：拓扑相（卷绕数1）

### 3.2 Haldane模型（二维）

**哈密顿量**：
$$H = \sum_{\langle ij \rangle} t_1 c_i^\dagger c_j + \sum_{\langle\langle ij \rangle\rangle} t_2 e^{i\phi} c_i^\dagger c_j + \text{h.c.} + \sum_i \epsilon_i c_i^\dagger c_i$$

**参数**：
- $t_1$：最近邻跃迁
- $t_2 = 0.15 e^{i\pi/2}$：次近邻复跃迁
- $\epsilon_A = -\epsilon_B = \delta$：在位能

**PythTB实现**：
```python
from pythtb import *
import numpy as np

def make_haldane(delta=0.2, t=-1.0, t2=0.15):
    """构建Haldane模型"""
    lat = [[1.0, 0.0], [0.5, np.sqrt(3)/2]]
    orb = [[1/3., 1/3.], [2/3., 2/3.]]
    
    model = tb_model(2, 2, lat, orb)
    model.set_onsite([-delta, delta])
    
    # 最近邻
    t2_phase = t2 * np.exp(1j * np.pi / 2)
    model.set_hop(t, 0, 1, [0, 0])
    model.set_hop(t, 1, 0, [1, 0])
    model.set_hop(t, 1, 0, [0, 1])
    
    # 次近邻（复跃迁）
    model.set_hop(t2_phase, 0, 0, [1, 0])
    model.set_hop(t2_phase.conj(), 0, 0, [0, 1])
    model.set_hop(t2_phase.conj(), 0, 0, [-1, 1])
    model.set_hop(t2_phase, 1, 1, [1, 0])
    model.set_hop(t2_phase.conj(), 1, 1, [0, 1])
    model.set_hop(t2_phase, 1, 1, [-1, 1])
    
    return model
```

**Chern数**：
- 上能带：+1
- 下能带：-1

### 3.3 Kane-Mele模型（二维拓扑绝缘体）

**哈密顿量**：
$$H = H_0 + H_{SOC} + H_{Rashba}$$

其中：
- $H_0$：石墨烯最近邻跃迁
- $H_{SOC} = i t_2 \nu_{ij} s_z c_i^\dagger c_j$：自旋轨道耦合
- $H_{Rashba}$：Rashba耦合（破缺z方向镜面对称）

**完整Python实现**：
```python
import numpy as np
import matplotlib.pyplot as plt

def kane_mele_hamiltonian(kx, ky, t=1.0, t2=0.06, lambda_r=0.05, delta=0.0):
    """
    Kane-Mele模型哈密顿量（4×4矩阵）
    基矢：(A↑, A↓, B↑, B↓)
    """
    # Pauli矩阵
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    sigma_0 = np.eye(2)
    
    # 最近邻项
    d_x = t * (1 + np.cos(kx) + np.cos(ky))
    d_y = t * (np.sin(kx) + np.sin(ky))
    
    # 次近邻SOC项
    h_soc = 2 * t2 * (np.sin(kx) - np.sin(ky) - np.sin(kx - ky))
    
    # Rashba项
    r_x = lambda_r * (1 - np.cos(kx) - np.cos(ky))
    r_y = lambda_r * (np.sin(kx) - np.sin(ky))
    
    # 构建4×4哈密顿量
    H = np.zeros((4, 4), dtype=complex)
    
    # 对角块（在位能+SOC）
    H[0:2, 0:2] = (delta + h_soc) * sigma_z
    H[2:4, 2:4] = (-delta - h_soc) * sigma_z
    
    # 非对角块（最近邻+Rashba）
    H[0:2, 2:4] = d_x * sigma_0 + 1j * d_y * sigma_0 + r_x * sigma_x + r_y * sigma_y
    H[2:4, 0:2] = H[0:2, 2:4].T.conj()
    
    return H

# 计算能带
k_path = np.linspace(-np.pi, np.pi, 100)
energies = []

for k in k_path:
    H = kane_mele_hamiltonian(k, 0)
    eigs = np.linalg.eigvalsh(H)
    energies.append(eigs)

energies = np.array(energies)

# 绘图
plt.figure(figsize=(8, 6))
for i in range(4):
    plt.plot(k_path / np.pi, energies[:, i], 'b-', linewidth=1)
plt.xlabel('k/π')
plt.ylabel('Energy')
plt.title('Kane-Mele Model Band Structure')
plt.show()
```

## 4. WannierTools应用

### 4.1 拓扑材料分析流程

**Bi2Se3计算示例**：
```fortran
! WT.in 输入文件
&CONTROL
BulkBand_calc = T      ! 体能带计算
SlabBand_calc = T      ! 板能带计算
WireBand_calc = F      ! 纳米线能带
Dos_calc = F
FermiArc_calc = T      ! 费米弧计算
/

&SYSTEM
SOC = 1                 ! 自旋轨道耦合
E_FERMI = 4.4195        ! 费米能（从DFT获取）
/

&PARAMETERS
Eta_Arc = 0.001         ! 展宽
E_arc = 0.0             ! 费米弧能量
Nk1 = 41                ! k点网格
Nk2 = 41
/

KPATH_BULK
4
G 0.000 0.000 0.000   Z 0.000 0.000 0.500
Z 0.000 0.000 0.500   F 0.500 0.500 0.000
F 0.500 0.500 0.000   G 0.000 0.000 0.000
G 0.000 0.000 0.000   L 0.500 0.000 0.000
```

### 4.2 表面态计算

**关键参数**：
```fortran
&PARAMETERS
Nk1 = 101               ! 表面布里渊区k点
Nk2 = 101
OmegaMin = -1.0         ! 能量范围
OmegaMax = 1.0
OmegaNum = 500          ! 能量点数
/

KPATH_SLAB
2
Y 0.500 0.000 0.000   G 0.000 0.000 0.000
G 0.000 0.000 0.000   X 0.000 0.500 0.000
```

### 4.3 自旋纹理计算

```fortran
&CONTROL
Spintexture_calc = T
/

&PARAMETERS
Nk = 101
/

KPLANE_SLAB
-0.1 0.1              ! kx范围
-0.1 0.1              ! ky范围
0.0                   ! 能量
```

## 5. 高级应用

### 5.1 对称化Wannier函数

**问题**：标准MLWF可能破坏晶体对称性  
**解决方案**：
- 对称约束Wannier函数
- 保持点群对称性
- 应用：拓扑不变量计算

### 5.2 应变材料模型

**流程**：
1. DFT计算应变结构
2. 构建各应变下的Wannier模型
3. 线性插值得到连续应变依赖

```python
# 应变插值示例
def interpolate_tb(strain, tb_0, tb_1, tb_2):
    """
    二次插值紧束缚参数
    """
    # 拟合参数随应变的变化
    params = np.polyfit([0, 0.01, 0.02], [tb_0, tb_1, tb_2], 2)
    return np.polyval(params, strain)
```

### 5.3 异质结构建模

**方法**：
- 分别构建各层Wannier函数
- 层间耦合通过重叠积分估计
- 构建大的超胞哈密顿量

### 5.4 机器学习加速

**应用**：
- 神经网络预测Wannier中心
- 加速大体系优化
- 自动化参数选择

## 6. 软件工具对比

| 工具 | 特点 | 适用场景 |
|------|------|----------|
| Wannier90 | 标准MLWF构建 | 通用第一性原理→TB |
| PythTB | Python紧束缚 | 模型研究、教学 |
| TBPLaS | 大规模计算 | 纳米结构、输运 |
| WannierTools | 拓扑分析 | 表面态、费米弧 |
| ASE Wannier | ASE集成 | GPAW用户 |

## 7. 常见问题与技巧

### 7.1 收敛性问题

**症状**：展宽不收敛  
**解决**：
- 检查初始投影选择
- 调整能量窗口
- 增加k点密度
- 尝试不同spin/轨道组合

### 7.2 能带插值质量差

**诊断**：
- 比较DFT和Wannier能带
- 检查Wannier中心位置

**改进**：
- 增加Wannier函数数目
- 扩大外能量窗口
- 使用disentanglement

### 7.3 拓扑不变量计算

**关键点**：
- 确保能带插值精确
- 检查规范一致性
- 使用Wilson loop方法

## 参考文献

1. Marzari, N., Vanderbilt, D. (1997). Maximally localized generalized Wannier functions.
2. Souza, I., Marzari, N., Vanderbilt, D. (2001). Maximally localized Wannier functions.
3. Mostofi, A.A., et al. (2008). Wannier90: A tool for obtaining maximally-localised Wannier functions.
4. Wu, Q., et al. (2018). WannierTools: A software package for novel topological materials.
5. Carr, S., et al. (2019). Derivation of Wannier orbitals for twisted bilayer graphene.

---
**研究时间**：2026-03-08  
**专题状态**：✅ 完成
