# 专题1：拓扑不变量计算（Z2/Chern数）研究报告

## 1. 理论基础

### 1.1 Chern数的定义

Chern数是表征二维拓扑材料拓扑性质的重要不变量，定义为：

$$C = \sum_n^{occ.} \frac{1}{2\pi} \int_{BZ} d k_x d k_y F_n = 0, \pm 1, \pm 2, \pm 3, \cdots$$

其中：
- $F_n = (\nabla \times A_n)_z$ 是Berry曲率
- $A_n = i\langle u_{nk}|\frac{\partial}{\partial k}|u_{nk}\rangle$ 是Berry联络

**物理意义**：
- $C = 0$：平凡拓扑态
- $C = \pm 1, \pm 2, \cdots$：Chern绝缘体态（量子反常霍尔效应）
- 霍尔电导率：$\sigma_{xy} = C \frac{e^2}{h}$

### 1.2 Z2不变量的定义

Z2不变量用于表征具有时间反演对称性的拓扑绝缘体：

$$Z_2 = \sum_n^{occ.} \frac{1}{2\pi} \left( \oint_{Half BZ} A_n \cdot d k - \int_{Half BZ} d k_x d k_y F_n \right) = 0 \text{ or } 1 \pmod{2}$$

**物理意义**：
- $Z_2 = 0$：平凡拓扑态
- $Z_2 = 1$：拓扑非平庸态（量子自旋霍尔效应）

### 1.3 三维拓扑绝缘体的分类

三维拓扑绝缘体由4个Z2拓扑数表征：
- 1个强拓扑数：$\nu_0$
- 3个弱拓扑数：$\nu_x, \nu_y, \nu_z$

计算公式：
$$(-1)^{\nu_0} = \prod_{n_j=0,1} \delta_{n_1n_2n_3}$$
$$(-1)^{\nu_{i=1,2,3}} = \prod_{n_j \neq i=0,1; n_i=1} \delta_{n_1n_2n_3}$$

## 2. 计算方法

### 2.1 Fukui-Hatsugai方法（格点化方法）

**核心思想**：在离散化的布里渊区上计算Berry曲率和Berry联络。

**计算步骤**：
1. 定义重叠矩阵：$U_{\mu}(k) = \det \langle u_n(k)|u_n(k+\mu)\rangle$
2. 计算Berry联络：$A_{\mu}(k) = \text{Im} \log U_{\mu}(k)$
3. 计算Berry曲率：
   $$F(k) = \text{Im} \log U_{k_x}(k) U_{k_y}(k+\Delta k_x) U_{k_x}^{-1}(k+\Delta k_y) U_{k_y}^{-1}(k)$$
4. 格点Chern数：
   $$n(k) = \frac{1}{2\pi} \left[ (A_{12}+A_{23}+A_{34}+A_{41}) - F_k \right]$$
5. 总Chern数：$C = \sum_k n(k)$

### 2.2 Wilson Loop方法

**核心思想**：通过计算Wilson loop的本征值来获取Berry相位。

**Wilson Loop定义**：
$$W(C) = M^{k_0,k_1} \cdots M^{k_{n-1},k_n}$$

其中重叠矩阵：
$$M_{m,n}^{k_i,k_j} = \langle u_{m,k_i}|u_{n,k_j}\rangle$$

**Berry相位与Wilson loop本征值的关系**：
$$\gamma_C = \sum_i \arg \lambda_i$$

其中$\lambda_i$是Wilson loop的本征值。

### 2.3 Wannier Charge Centers (WCC)方法

**核心思想**：通过追踪混合Wannier电荷中心的演化来计算拓扑不变量。

**混合Wannier轨道**：
$$|R_x, k_y, k_z; n\rangle = \frac{a_x}{2\pi} \int_{-\pi/a_x}^{\pi/a_x} e^{-ik_x R_x} |\psi_{nk}\rangle dk_x$$

**混合WCC位置**：
$$\bar{x}_n(k_y, k_z) = \langle 0, k_y, k_z; n | \hat{x} | 0, k_y, k_z; n \rangle$$

**Chern数与WCC的关系**：
$$C = \frac{1}{a} \sum_n \bar{x}_n$$

### 2.4 基于宇称的计算方法（适用于有反演对称性的系统）

当系统具有空间反演对称性时，Z2不变量可通过占据态波函数的宇称乘积计算：

$$(-1)^{\nu} = \prod_{n_j=0}^{1} \delta_{n_1n_2}$$

其中：
$$\delta_{n_j} = \prod_{m=1}^{N} \varepsilon_{2m}(\Gamma_{n_j})$$

$\varepsilon_{2m}(\Gamma_n)$代表第n个TRIM点上第2m条占据态的宇称。

### 2.5 Kubo公式方法

Berry曲率的Kubo公式表达：
$$\Omega_{\mu\nu}^n(R) = i \sum_{n' \neq n} \frac{\langle n|\frac{\partial H}{\partial R_\mu}|n'\rangle \langle n'|\frac{\partial H}{\partial R_\nu}|n\rangle - (\nu \leftrightarrow \mu)}{(\varepsilon_n - \varepsilon_{n'})^2}$$

**优点**：不需要计算波函数的导数，仅需哈密顿量对参数的导数，适合数值计算。

## 3. 实用软件工具

### 3.1 Z2Pack

**简介**：用于计算拓扑不变量的Python库，支持模型哈密顿量和第一性原理计算。

**安装**：
```bash
pip install z2pack
```

**基本用法**：
```python
import z2pack

# 定义系统（以紧束缚模型为例）
def hamiltonian(k):
    # 返回H(k)矩阵
    pass

system = z2pack.hm.System(hamiltonian, bands=2)

# 定义计算表面
result = z2pack.surface.run(
    system=system,
    surface=lambda kx, ky: [kx, ky, 0],  # kz=0平面
    num_lines=11,
    pos_tol=1e-2
)

# 计算拓扑不变量
chern_number = z2pack.invariant.chern(result)
z2_invariant = z2pack.invariant.z2(result)

print(f"Chern数: {chern_number}")
print(f"Z2不变量: {z2_invariant}")
```

### 3.2 PythTB

**简介**：紧束缚模型计算的Python包，内置Berry相位和Chern数计算功能。

**Haldane模型示例**：
```python
from pythtb import TBModel, Lattice, WFArray, Mesh
import numpy as np

# 定义晶格
lat_vecs = [[1, 0], [1/2, np.sqrt(3)/2]]
orb_vecs = [[1/3, 1/3], [2/3, 2/3]]
lat = Lattice(lat_vecs, orb_vecs)

# 创建Haldane模型
model = TBModel(lat)
delta, t = 0, -1
t2 = 0.15 * np.exp(1j * np.pi / 2)
t2c = t2.conjugate()

model.set_onsite([-delta, delta])
model.set_hop(t, 0, 1, [0, 0])
model.set_hop(t, 1, 0, [1, 0])
model.set_hop(t, 1, 0, [0, 1])
# ... 添加次近邻跃迁

# 计算Chern数
mesh = Mesh([100, 100])
wfa = WFArray(model.lattice, mesh)
wfa.solve_model(model)

chern0 = wfa.chern_number(state_idx=[0], plane=(0, 1))
chern1 = wfa.chern_number(state_idx=[1], plane=(0, 1))
print(f"Lower band Chern数: {chern0}")
print(f"Upper band Chern数: {chern1}")
```

### 3.3 TBPLaS

**简介**：紧束缚模型传播方法库，支持大规模系统计算。

**示例**：
```python
import tbplas as tb
import numpy as np

# 创建Haldane模型
def make_haldane():
    lat = 1.0
    delta = 0.2
    t = -1.0
    t2 = 0.15 * np.exp(1j * np.pi / 2.)
    t2c = t2.conjugate()
    
    vectors = tb.gen_lattice_vectors(a=lat, b=lat, c=1.0, gamma=60)
    cell = tb.PrimitiveCell(vectors, unit=tb.NM)
    cell.add_orbital((1/3., 1/3.), energy=-delta)
    cell.add_orbital((2/3., 2/3.), energy=delta)
    # 添加跃迁...
    return cell

# 计算Chern数
cell = make_haldane()
solver = tb.Berry(cell)
solver.config.k_grid_size = (100, 100, 1)
solver.config.num_occ = 1

# 使用Kubo公式
solver.calc_berry_curvature_kubo()

# 使用Wilson loop
solver.calc_berry_curvature_wilson()
```

## 4. 典型材料计算实例

### 4.1 Bi2Se3（强拓扑绝缘体）

使用Z2Pack计算结果：
- $k_z=0$平面：Z2 = 1（非平庸）
- $k_z=0.5$平面：Z2 = 0（平庸）
- 结论：强拓扑绝缘体

### 4.2 BiTeI（压力诱导拓扑相变）

- 0 GPa：Z2 = 0（平庸绝缘体）
- 5 GPa：Z2 = 1（非平庸拓扑绝缘体）

### 4.3 Haldane模型

- 下能带Chern数：-1
- 上能带Chern数：+1
- 体现能带反转导致的非平庸拓扑

## 5. 计算技巧与注意事项

### 5.1 规范固定问题
- Z2不变量计算需要固定规范
- 时间反演不变点(TRIM)需要特别注意
- 使用连续规范追踪方法

### 5.2 数值收敛性
- k点密度需要足够高
- WCC计算需要检查收敛性
- Wilson loop本征值收敛性检查

### 5.3 能带交叉处理
- 避免在能带交叉处计算
- 使用自适应k点加密
- 检查简并点的Kramers对

## 6. 前沿进展

### 6.1 实空间拓扑不变量
- Local Chern marker方法
- 适用于非晶系统和无序系统
- 单Γ点计算方法

### 6.2 高阶拓扑不变量
- Nested Wilson loop
- 高阶拓扑绝缘体分类
- Wannier极化多极矩

### 6.3 机器学习方法
- 神经网络预测拓扑不变量
- 加速大规模材料筛选
- 拓扑材料数据库建设

## 参考文献

1. Fukui, T., Hatsugai, Y. (2007). Chern number and Z2 invariant.
2. Yu, R., et al. (2011). Z2拓扑不变量与拓扑绝缘体.
3. Gresch, D., et al. (2017). Z2Pack: Numerical implementation.
4. Soluyanov, A.A., Vanderbilt, D. (2011). Wannier charge centers.
5. Kane, C.L., Mele, E.J. (2005). Quantum Spin Hall Effect.

---
**研究时间**：2026-03-08  
**专题状态**：✅ 完成
