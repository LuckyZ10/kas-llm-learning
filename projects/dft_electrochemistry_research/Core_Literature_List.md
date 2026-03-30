# DFT电化学核心文献清单
## 2024-2025前沿研究精选

---

## 一、恒电位方法与理论框架

### CIP-DFT (2024突破性工作)
1. **Melander et al.**, "Constant inner potential DFT for modelling electrochemical systems under constant potential and bias"
   - *npj Computational Materials* 10, 5 (2024)
   - 核心贡献：提出恒内势方法，解决外球反应和双电极系统的恒电位模拟难题
   - 实现代码：GPAW+SJM

### GC-DFT综述
2. **Sundararaman et al.**, "Improving the Accuracy of Atomistic Simulations of the Electrochemical Interface"
   - *Chemical Reviews* 122, 10651 (2022)
   - BEAST项目核心综述，涵盖溶剂化、GC-DFT、RPA方法

3. **Goodpaster et al.**, "Identification of possible pathways for C-C bond formation during electrochemical reduction of CO2"
   - *J. Phys. Chem. Lett.* (2016)
   - GC-DFT在CO2RR中的应用

### CHE模型原始文献
4. **Nørskov et al.**, "Origin of the overpotential for oxygen reduction at a fuel-cell cathode"
   - *J. Phys. Chem. B* 108, 17886 (2004)
   - CHE模型奠基工作

---

## 二、溶剂化与界面模型

### 隐式溶剂化模型
5. **VASPsol**: Mathew et al., "Implicit solvation model for density-functional study of nanocrystal surfaces and reaction pathways"
   - *J. Chem. Phys.* 140, 084106 (2014)
   - 开源实现：https://github.com/henniggroup/VASPsol

6. **JDFTx**: Sundararaman et al., "JDFTx: software for joint density-functional theory"
   - *SoftwareX* 6, 278 (2017)
   - 开源代码：http://jdftx.org/

7. **CANDLE**: Sundararaman et al., "CANDLE: A concave, adaptable, non-local dielectric model for water"
   - 推荐用于带电溶质

---

## 三、CO2还原反应(CO2RR)

### 单原子催化剂
8. **Chala et al.**, "Cooperative dual single atom Ni/Cu catalyst for highly selective CO2-to-ethanol reduction"
   - *Appl. Catal. B: Environ. Energy* (2024)
   - 92.2%乙醇法拉第效率

9. **Wang et al.**, "Non-nitrogen Mn-O4 coordination environment boosting CO2 electroreduction"
   - *Molecular Catalysis* (2024)
   - 非氮配位环境SAC设计

10. **Shen et al.**, "Enhanced electrochemical CO2-to-ethylene conversion through second-shell coordination"
    - *J. Mater. Chem. A* (2024)
    - 第二壳层配位调控

11. **Liu et al.**, "Theoretical insights into lanthanide rare earth single-atom catalysts"
    - *J. Mater. Chem. A* (2024)
    - 稀土单原子催化剂

### DFT泛函影响
12. **Diliberto et al.**, "CO2 electroreduction on single atom catalysts: the role of the DFT functional"
    - *Phys. Chem. Chem. Phys.* (2024)
    - PBE vs PBE+U vs PBE0对比

---

## 四、氮还原反应(NRR)

### GC-DFT应用
13. **Aubry et al.**, "Activating Nitrogen for Electrochemical Ammonia Synthesis via an Electrified Transition-Metal Dichalcogenide Catalyst"
    - *J. Phys. Chem. C* 128, 7063 (2024)
    - GC-DFT揭示电位激活N2机制

### 磁性催化剂
14. **Li et al.**, "Novel magneto-electrocatalyst Cr2CO2-MXene for boosting nitrogen reduction"
    - *Materials Horizons* (2024)
    - 自旋催化新方向

15. **Wen et al.**, "Tailoring the d-Band Center of WS2 by Metal and Nonmetal Dual-Doping"
    - *Small* (2024)
    - d带中心调控策略

### 双原子催化剂
16. **Lei et al.**, "Revolutionizing nitrogen electrocatalysis: Atomically dispersed ruthenium catalyzed by manganese"
    - *Appl. Catal. A: General* (2024)
    - RuMn双原子催化剂

---

## 五、HER/OER/ORR机理

### 水分解催化剂
17. **Meng et al.**, "Oxygen Vacancy-Enhanced Ni3N-CeO2/NF Nanoparticle Catalysts"
    - *Nanomaterials* 14, 935 (2024)
    - 氧空位工程

18. **Various**, "Recent Advances in Transition Metal Chalcogenides Electrocatalysts"
    - *Catalysts* 15, 124 (2025)
    - TMC催化剂综述

### 火山图理论
19. **Man et al.**, "Universality in oxygen evolution electrocatalysis on oxide surfaces"
    - *ChemCatChem* 3, 1159 (2011)
    - 经典OER火山图

---

## 六、机器学习势与AI驱动筛选

### MLIP综述
20. **Ou et al.**, "From descriptors to machine learning interatomic potentials"
    - *AI Agent* (2025)
    - MLIP在电催化中的应用综述

21. **Olajide et al.**, "Application of machine learning interatomic potentials in heterogeneous catalysis"
    - *J. Catal.* (2025)
    - MLIP催化应用综述

### 前沿MLIP模型
22. **Shiota et al.**, MACE-Osaka24
    - TEA对齐有机/无机数据，能垒误差0.7 kcal/mol
    - 数据集：https://www.repository.cam.ac.uk/

23. **Bilbrey et al.**, "Uncertainty quantification for neural network potential foundation models"
    - *npj Comput. Mater.* 11, 109 (2025)
    - 证据深度学习

### AI驱动筛选
24. **Rahman et al.**, "High-throughput screening of single atom co-catalysts in ZnIn2S4"
    - *Mater. Adv.* 5, 8673 (2024)
    - 172种SAC高通量筛选

25. **Zhang et al.**, "Active learning accelerated exploration of single-atom local environments"
    - *npj Comput. Mater.* 10, 32 (2024)
    - 主动学习加速SAC筛选

26. **Sun et al.**, "Interpretable machine learning-assisted high-throughput screening"
    - *Energy Environ. Mater.* 7, e12693 (2024)
    - 可解释ML用于NRR筛选

---

## 七、方法论论文

### 电化学模拟方法
27. **Jinnouchi et al.**, 表面溶胶模型与修正Poisson-Boltzmann理论
    - 早期恒电位实现

28. **Gauthier et al.**, "Myriad implicit electrolyte model"
    - 有效表面电荷密度描述符

29. **Kastlunger et al.**, "Advances and challenges for multi-electron multi-proton transfer"
    - *Phys. Chem. Chem. Phys.* (2020)
    - GCE-DFT与动力学

### 恒电荷方法
30. **Huang et al.**, "Grand canonical DFT based constant charge method for HER/DER"
    - *Comput. Mater. Sci.* (2023/2025)
    - 同位素效应微动力学

---

## 八、会议与教育资源

### 重要会议
31. **BEAST Workshop** (2022-2025)
    - Beyond-DFT Electrochemistry with Accelerated and Solvated Techniques
    - 年度研讨会，含JDFTx/BerkeleyGW教程

32. **Psi-k Young Researcher's School** (2023)
    - "Theory and Simulation in Electrochemical Conversion Processes"
    - MetalWalls, CP2K, GPAW, QE, tranSIESTA教程

### 在线教程
33. **IPAM Tutorial** (2025)
    - "Continuum Descriptions of Liquid Environments"
    - Richard Hennig主讲

34. **JDFTx官方教程**
    - http://jdftx.org/Tutorials.html
    - 从溶剂化分子到电化学界面

---

## 九、软件与数据库

### 开源软件
| 软件 | 功能 | 网址 |
|------|------|------|
| JDFTx | GC-DFT, JDFT | http://jdftx.org/ |
| GPAW | CIP-DFT (SJM) | https://gpaw.readthedocs.io/ |
| VASPsol | 隐式溶剂化 | GitHub |
| Quantum ESPRESSO+Environ | 溶剂化 | https://www.quantum-espresso.org/ |
| CP2K | 有限场MD | https://www.cp2k.org/ |
| MetalWalls | 经典MD | 电化学专用 |

### 预训练MLIP
| 模型 | 覆盖范围 | 获取方式 |
|------|----------|----------|
| MACE-MP-0 | 大多数元素 | pip install mace-torch |
| MACE-Osaka24 | 有机+无机 | Cambridge Repository |
| CHGNet | 晶体材料 | GitHub |
| SevenNet | 多元素 | GitHub |
| ORB | 通用 | GitHub |

### 数据库
| 数据库 | 内容 | 网址 |
|--------|------|------|
| BEAST-DB | GC-DFT电化学 | 开发中 |
| Materials Project | 材料计算 | materialsproject.org |
| Open Catalyst | 催化反应 | opencatalystproject.org |
| Catalysis-Hub | 反应机理 | catalysis-hub.org |

---

## 十、必读综述推荐

### 入门必读
1. Nørskov et al., J. Phys. Chem. B 2004 (CHE模型)
2. Sundararaman et al., Chem Rev 2022 (溶剂化)
3. Ou et al., AI Agent 2025 (MLIP)

### 进阶深入
4. Melander et al., npj Comput Mater 2024 (CIP-DFT)
5. Kastlunger et al., PCCP 2020 (GCE-DFT动力学)

### 应用导向
6. Chala et al., Appl. Catal. B 2024 (CO2RR)
7. Aubry et al., JPCC 2024 (NRR GC-DFT)

---

*文献清单版本: 2025.03  
整理时间: 2026-03-08*
