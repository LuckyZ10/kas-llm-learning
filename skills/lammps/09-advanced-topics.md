# 09. 高级专题与专家技巧

> LAMMPS专家级技巧、高级模拟技术和研究前沿

---

## 目录
- [反应力场模拟](#反应力场模拟)
- [相变与成核](#相变与成核)
- [缺陷演化模拟](#缺陷演化模拟)
- [非平衡分子动力学](#非平衡分子动力学)
- [机器学习增强采样](#机器学习增强采样)
- [不确定性量化](#不确定性量化)

---

## 反应力场模拟

### ReaxFF高级设置

```lammps
# 含能材料反应模拟
units real
atom_style charge

read_data tatb.data

# ReaxFF势
pair_style reax/c NULL
pair_coeff * * ffield.reax.CHONSSi C H O N

# 关键: 必须启用QEq电荷平衡
fix 1 all qeq/reax 1 0.0 10.0 1.0e-6 reax/c

# 反应跟踪
fix 2 all reax/c/species 10 10 100 species.tatb
fix 3 all reax/c/bonds 100 bonds.reax

# 温度控制 - 加热引发反应
variable t ramp(300,3000)
fix 4 all nvt temp v_t v_t 100.0

thermo 100
thermo_style custom step temp pe press density

dump 1 all custom 1000 reaction.dump id type x y z q
timestep 0.25
run 100000
```

### 反应产物分析

```python
# analyze_reactions.py
import re
from collections import Counter

def parse_species_file(filename):
    """解析ReaxFF物种文件"""
    species_evolution = {}
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.split()
            timestep = int(parts[0])
            nspecies = int(parts[1])
            
            species_counts = {}
            for i in range(nspecies):
                formula = parts[2 + 2*i]
                count = int(parts[3 + 2*i])
                species_counts[formula] = count
            
            species_evolution[timestep] = species_counts
    
    return species_evolution

def compute_reaction_rate(species_evolution, target_species):
    """计算反应速率"""
    timesteps = sorted(species_evolution.keys())
    concentrations = [species_evolution[t].get(target_species, 0) 
                     for t in timesteps]
    
    # 计算生成速率
    rates = np.gradient(concentrations, timesteps)
    return timesteps, concentrations, rates
```

---

## 相变与成核

### 成核模拟设置

```lammps
# 均相成核模拟
units metal
atom_style atomic

# 创建过饱和体系
region box block 0 50 0 50 0 50
create_box 1 box
create_atoms 1 random 50000 12345 box

pair_style lj/cut 10.0
pair_coeff 1 1 0.2381 3.405

# 计算局部序参数
coordinate cna all cna/atom 3.5

# 追踪团簇
coordinate cluster all cluster/atom 3.5

# 输出序参数
dump 1 all custom 1000 nucleation.dump id type x y z c_cna c_cluster
dump_modify 1 sort id

# 在临界过饱和度下运行
fix 1 all npt temp 0.5 0.5 0.1 iso 0.1 0.1 1.0

thermo 1000
thermo_style custom step temp pe press density

timestep 0.001
run 500000
```

### 成核率计算

```python
# nucleation_rate.py
import numpy as np
from scipy.optimize import curve_fit

def critical_cluster_size(clusters, time):
    """
    计算临界团簇尺寸
    
    使用基于经典成核理论的拟合
    """
    sizes = [len(c) for c in clusters]
    size_distribution = Counter(sizes)
    
    # 拟合到CNT分布
    def cnt_distribution(n, n_c, beta_delta_G):
        return A * np.exp(-beta_delta_G * (n/n_c)**(2/3))
    
    return size_distribution

def compute_nucleation_rate(cluster_evolution, volume, time):
    """
    计算成核率 J = N/(V*t)
    
    Parameters:
    -----------
    cluster_evolution: list of cluster counts over time
    volume: simulation volume
    time: observation time
    """
    # 找到稳态成核阶段
    stable_clusters = [c for c in cluster_evolution if c > 0]
    
    if len(stable_clusters) < 2:
        return 0
    
    # 线性拟合
    slope, intercept = np.polyfit(time, stable_clusters, 1)
    
    # J = (dN/dt) / V
    J = slope / volume
    
    return J
```

---

## 缺陷演化模拟

### 级联碰撞模拟

```lammps
# 辐射损伤级联模拟
units metal
atom_style atomic
boundary p p p

read_data cu_crystal.data

# EAM势
pair_style eam/alloy
pair_coeff * * Cu_u3.eam.alloy Cu

# 选择初级撞出原子 (PKA)
group pka id 1000
velocity pka set 1000.0 0.0 0.0  # 10 keV

# 追踪缺陷
coordinate wigner all cna/atom 3.5

# 电子阻止 (可选)
fix 1 all electron/stopping 1000.0

# 输出
dump 1 all custom 10 cascade.dump id type x y z c_wigner
dump_modify 1 sort id

# NVE模拟
fix 2 all nve
thermo 10
thermo_style custom step temp pe ke

timestep 0.0001  # 0.1 fs for cascade
run 10000
```

### 缺陷分析

```python
# defect_analysis.py
import numpy as np
from collections import defaultdict

class DefectAnalyzer:
    """晶体缺陷分析器"""
    
    def __init__(self, reference_positions):
        self.reference = reference_positions
        
    def identify_defects(self, current_positions, threshold=0.3):
        """
        识别点缺陷
        
        Returns:
            vacancies: 空位位置
            interstitials: 间隙位置
            antisites: 反位缺陷
        """
        # Wigner-Seitz缺陷分析
        vacancies = []
        interstitials = []
        
        # 计算位移
        displacements = current_positions - self.reference
        
        # 识别空位 (参考位置没有原子)
        for i, ref_pos in enumerate(self.reference):
            distances = np.linalg.norm(current_positions - ref_pos, axis=1)
            if np.min(distances) > threshold:
                vacancies.append(ref_pos)
        
        # 识别间隙 (当前位置没有参考)
        for i, curr_pos in enumerate(current_positions):
            distances = np.linalg.norm(self.reference - curr_pos, axis=1)
            if np.min(distances) > threshold:
                interstitials.append(curr_pos)
        
        return vacancies, interstitials
    
    def compute_displacement_cascade(self, trajectory):
        """计算位移级联大小"""
        displacements = []
        
        for frame in trajectory:
            disp = np.linalg.norm(frame - self.reference, axis=1)
            displacements.append(disp)
        
        return displacements
```

---

## 非平衡分子动力学

### 热导率计算 (NEMD)

```lammps
# 非平衡MD计算热导率
units metal
atom_style atomic

read_data si.data

# Tersoff势
pair_style tersoff
pair_coeff * * Si.tersoff Si

# 创建热浴区域
region hot block INF INF INF INF 0 5
region cold block INF INF INF INF 45 50

group hot region hot
group cold region cold

# 能量交换
fix 1 hot langevin 350 350 0.1 12345 tally yes
fix 2 cold langevin 250 250 0.1 23456 tally yes
fix 3 all nve

# 计算热流
compute ke_hot hot ke/atom
compute ke_cold cold ke/atom

# 输出温度分布
compute temp_profile all chunk/atom bin/1d z lower 1.0
fix 4 all ave/chunk 100 100 10000 temp_profile temp file temp_profile.dat

dump 1 all custom 1000 nemd.dump id type x y z
timestep 0.001
run 1000000
```

### Green-Kubo方法

```python
# thermal_conductivity.py
import numpy as np
from scipy import integrate

def green_kubo_kappa(time, heat_flux, volume, temperature):
    """
    Green-Kubo计算热导率
    
    κ = (V/kT²) ∫_0^∞ <J(t)·J(0)> dt
    """
    kB = 1.380649e-23  # J/K
    
    # 计算热流自相关
    acf = compute_autocorrelation(heat_flux)
    
    # 积分
    kappa = (volume / (kB * temperature**2)) * integrate.trapz(acf, time)
    
    return kappa

def compute_autocorrelation(signal):
    """计算自相关函数"""
    n = len(signal)
    signal = np.array(signal) - np.mean(signal)
    
    # FFT方法
    f_signal = np.fft.fft(signal, n=2*n)
    acf = np.fft.ifft(f_signal * np.conjugate(f_signal)).real[:n]
    acf = acf / (n - np.arange(n))
    
    return acf
```

---

## 机器学习增强采样

### 深度学习CV

```python
# deep_cv.py
import torch
import torch.nn as nn

class DeepCollectiveVariable(nn.Module):
    """基于深度学习的集体变量"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32, 16]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class DeepTICA(nn.Module):
    """
    深度TICA (Time-lagged Independent Component Analysis)
    
    自动发现慢速集体变量
    """
    
    def __init__(self, input_dim, n_components=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_components)
        )
    
    def forward(self, x_t, x_tau):
        """
        x_t: 时间t的构型
        x_tau: 时间t+τ的构型
        """
        z_t = self.encoder(x_t)
        z_tau = self.encoder(x_tau)
        
        # TICA目标函数
        return self.tica_loss(z_t, z_tau)
    
    def tica_loss(self, z_t, z_tau):
        """TICA损失函数"""
        # 协方差矩阵
        C_0 = torch.matmul(z_t.T, z_t) / len(z_t)
        C_tau = torch.matmul(z_t.T, z_tau) / len(z_t)
        
        # 特征值问题
        eigvals = torch.linalg.eigvals(torch.matmul(torch.inverse(C_0), C_tau))
        
        # 最大化慢速特征值
        return -torch.sum(eigvals.real)
```

### 主动学习采样

```python
# active_learning_sampling.py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class ActiveLearningSampler:
    """
    主动学习增强采样
    
    使用不确定性指导采样
    """
    
    def __init__(self, cv_bounds, n_initial=10):
        self.cv_bounds = cv_bounds
        self.n_initial = n_initial
        self.gp = GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1),
            n_restarts_optimizer=10
        )
        self.sampled_points = []
        self.energies = []
    
    def initialize(self, sampler):
        """初始均匀采样"""
        for _ in range(self.n_initial):
            point = np.random.uniform(
                [b[0] for b in self.cv_bounds],
                [b[1] for b in self.cv_bounds]
            )
            energy = sampler(point)
            self.sampled_points.append(point)
            self.energies.append(energy)
    
    def suggest_next_point(self):
        """建议下一个采样点"""
        # 训练GP
        X = np.array(self.sampled_points)
        y = np.array(self.energies)
        self.gp.fit(X, y)
        
        # 在网格上评估不确定性
        grid_points = self._create_grid()
        _, std = self.gp.predict(grid_points, return_std=True)
        
        # 选择不确定性最大的点
        next_point = grid_points[np.argmax(std)]
        
        return next_point
    
    def _create_grid(self, n_points=100):
        """创建搜索网格"""
        grids = [np.linspace(b[0], b[1], n_points) for b in self.cv_bounds]
        mesh = np.meshgrid(*grids)
        return np.array([m.flatten() for m in mesh]).T
```

---

## 不确定性量化

### 贝叶斯误差估计

```python
# bayesian_error_estimation.py
import numpy as np
from scipy.stats import norm

class BayesianErrorEstimator:
    """
    自由能计算的贝叶斯误差估计
    """
    
    def __init__(self, n_bootstraps=1000):
        self.n_bootstraps = n_bootstraps
    
    def bootstrap_pmf(self, histograms, bin_centers, temperature):
        """
        Bootstrap误差估计
        """
        kB = 0.001987  # kcal/mol/K
        beta = 1.0 / (kB * temperature)
        
        pmf_samples = []
        
        for _ in range(self.n_bootstraps):
            # 重采样
            resampled = self._resample_histograms(histograms)
            
            # 计算PMF
            pmf = self._compute_pmf(resampled, bin_centers, beta)
            pmf_samples.append(pmf)
        
        # 计算均值和置信区间
        pmf_mean = np.mean(pmf_samples, axis=0)
        pmf_std = np.std(pmf_samples, axis=0)
        pmf_ci_low = np.percentile(pmf_samples, 2.5, axis=0)
        pmf_ci_high = np.percentile(pmf_samples, 97.5, axis=0)
        
        return pmf_mean, pmf_std, pmf_ci_low, pmf_ci_high
    
    def _resample_histograms(self, histograms):
        """重采样直方图"""
        resampled = []
        for hist in histograms:
            # 从多项分布重采样
            total = np.sum(hist)
            probs = hist / total
            resampled_hist = np.random.multinomial(total, probs)
            resampled.append(resampled_hist)
        return resampled
    
    def _compute_pmf(self, histograms, bin_centers, beta):
        """计算PMF"""
        combined = np.sum(histograms, axis=0)
        pmf = -np.log(combined + 1e-10) / beta
        pmf -= np.min(pmf)
        return pmf

# 使用示例
estimator = BayesianErrorEstimator(n_bootstraps=1000)
pmf_mean, pmf_std, ci_low, ci_high = estimator.bootstrap_pmf(
    histograms, bin_centers, temperature=300.0
)

# 绘图
plt.fill_between(bin_centers, ci_low, ci_high, alpha=0.3, label='95% CI')
plt.plot(bin_centers, pmf_mean, 'b-', label='Mean PMF')
plt.xlabel('Reaction Coordinate')
plt.ylabel('Free Energy (kcal/mol)')
plt.legend()
```

### ML势不确定性

```python
# ml_uncertainty.py
import torch
import numpy as np

class MLPotentialUncertainty:
    """
    机器学习势的不确定性估计
    """
    
    def __init__(self, model, method='ensemble'):
        self.model = model
        self.method = method
    
    def estimate_uncertainty(self, positions, n_samples=100):
        """
        估计力预测的不确定性
        """
        if self.method == 'ensemble':
            return self._ensemble_uncertainty(positions)
        elif self.method == 'dropout':
            return self._dropout_uncertainty(positions, n_samples)
        elif self.method == 'evidential':
            return self._evidential_uncertainty(positions)
    
    def _ensemble_uncertainty(self, positions):
        """集成模型不确定性"""
        forces_list = []
        
        for model in self.model.models:
            with torch.no_grad():
                forces = model(positions)
                forces_list.append(forces.cpu().numpy())
        
        forces_array = np.array(forces_list)
        mean_forces = np.mean(forces_array, axis=0)
        force_std = np.std(forces_array, axis=0)
        
        return mean_forces, force_std
    
    def _dropout_uncertainty(self, positions, n_samples):
        """MC Dropout不确定性"""
        self.model.train()  # 启用dropout
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.model(positions)
                predictions.append(pred.cpu().numpy())
        
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)
```

---

## 专家技巧汇总

### 数值稳定性

```lammps
# 防止数值不稳定

# 1. 速度限制
fix 1 all limit/v 100.0  # 限制最大速度

# 2. 力限制
fix 2 all limit/f 1000.0  # 限制最大力

# 3. 温度漂移校正
fix 3 all momentum 100 linear 1 1 1

# 4. 能量漂移检查
variable e0 equal pe
variable edrift equal "abs((pe - v_e0)/v_e0)"
if "${edrift} > 0.5" then "quit"
```

### 长时程稳定性

```lammps
# 确保能量守恒

# 1. 正确的时间步长选择
timestep $(0.01/sqrt(dmax))  # 基于最大频率

# 2. 漂移校正
fix 1 all recenter INIT INIT INIT shift all

# 3. 约束消除漂移
fix 2 all momentum 100 angular
```

### 大数据集处理

```python
# 处理超大轨迹文件
import h5py
import numpy as np

def convert_to_hdf5(dump_file, hdf5_file):
    """将dump文件转换为HDF5以高效读取"""
    
    with h5py.File(hdf5_file, 'w') as hf:
        # 创建数据集
        positions = hf.create_dataset(
            'positions', 
            shape=(0, natoms, 3),
            maxshape=(None, natoms, 3),
            dtype='float32',
            chunks=True,
            compression='gzip'
        )
        
        # 分块读取dump文件
        for chunk in read_dump_chunks(dump_file):
            positions.resize(positions.shape[0] + 1, axis=0)
            positions[-1] = chunk['positions']
```

---

## 研究前沿

### 当前热点
1. **图神经网络势**: NequIP, MACE, Allegro
2. **等变神经网络**: 保持物理对称性
3. **主动学习**: 高效训练数据选择
4. **深度增强采样**: 神经网络CV
5. **不确定性量化**: 可靠ML预测

### 推荐资源
- [DeepMind AlphaFold](https://alphafold.ebi.ac.uk/)
- [Open Catalyst Project](https://opencatalystproject.org/)
- [Matlantis](https://matlantis.com/) - 计算材料平台
- [AI4Materials](https://github.com/AI4Materials) - 开源工具
