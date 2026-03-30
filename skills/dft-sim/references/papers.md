# 重要文献

## VASP相关文献

### 核心方法论文献

1. **Kresse, G. & Hafner, J.** (1993). *Ab initio molecular dynamics for liquid metals*. Physical Review B, 47(1), 558.
   - VASP程序的基础

2. **Kresse, G. & Furthmüller, J.** (1996). *Efficiency of ab-initio total energy calculations for metals and semiconductors using a plane-wave basis set*. Computational Materials Science, 6(1), 15-50.
   - VASP算法细节

3. **Kresse, G. & Joubert, D.** (1999). *From ultrasoft pseudopotentials to the projector augmented-wave method*. Physical Review B, 59(3), 1758.
   - PAW方法

4. **Blöchl, P. E.** (1994). *Projector augmented-wave method*. Physical Review B, 50(24), 17953.
   - PAW方法原始论文

### 机器学习力场 (MLFF)

5. **Jinnouchi, R., Karsai, F. & Kresse, G.** (2019). *On-the-fly machine learning force field generation: Application to melting points*. Physical Review B, 100(1), 014105.

6. **Jinnouchi, R., Karsai, F. & Kresse, G.** (2020). *Machine learning force fields for molecular dynamics*. arXiv:2002.00698.

7. **Jinnouchi, R., Asahi, R. & Kresse, G.** (2020). *Machine-learning force field for ZrO2: transferable to polymorphs with different coordination*. Physical Review Materials, 4(8), 083801.

### 最新方法进展 (2024-2025)

8. **VASP 6.5 Release Notes** (2025). *New features: Electron-phonon coupling, Python plugins, BSE GPU acceleration*. VASP Software GmbH.

9. **VASP 6.4 Release Notes** (2024). *Machine learning force fields: Fast-prediction mode, improved neighbor list algorithm*. VASP Software GmbH.

## Quantum ESPRESSO相关文献

### 核心文献

10. **Giannozzi, P., et al.** (2009). *QUANTUM ESPRESSO: a modular and open-source software project for quantum simulations of materials*. Journal of Physics: Condensed Matter, 21(39), 395502.
    - QE原始论文

11. **Giannozzi, P., et al.** (2017). *Advanced capabilities for materials modelling with Quantum ESPRESSO*. Journal of Physics: Condensed Matter, 29(46), 465901.
    - QE更新版本

12. **Giannozzi, P., et al.** (2020). *Quantum ESPRESSO toward the exascale*. The Journal of Chemical Physics, 152(15), 154105.
    - QE高性能计算

### GPU加速

13. **Spiga, F. & Girotto, I.** (2024). *QE-GPU: between performance, correctness and sustainability*. University of Cambridge / Quantum ESPRESSO Foundation.
    - GPU加速实现

14. **Oracle Cloud Infrastructure** (2024). *Accelerate Quantum Espresso simulation with GPU Shapes on OCI*.
    - GPU性能基准测试

## DFT方法论文献

### 基础理论

15. **Hohenberg, P. & Kohn, W.** (1964). *Inhomogeneous Electron Gas*. Physical Review, 136(3B), B864.
    - DFT基础定理1

16. **Kohn, W. & Sham, L. J.** (1965). *Self-Consistent Equations Including Exchange and Correlation Effects*. Physical Review, 140(4A), A1133.
    - Kohn-Sham方程

17. **Perdew, J. P., Burke, K. & Ernzerhof, M.** (1996). *Generalized Gradient Approximation Made Simple*. Physical Review Letters, 77(18), 3865.
    - PBE泛函

### 高级方法

18. **Dudarev, S. L., et al.** (1998). *Electron-energy-loss spectra and the structural stability of nickel oxide: An LSDA+U study*. Physical Review B, 57(3), 1505.
    - DFT+U (Dudarev方法)

19. **Heyd, J., Scuseria, G. E. & Ernzerhof, M.** (2003). *Hybrid functionals based on a screened Coulomb potential*. The Journal of Chemical Physics, 118(18), 8207.
    - HSE06杂化泛函

20. **Hedin, L.** (1965). *New Method for Calculating the One-Particle Green's Function with Application to the Electron-Gas Problem*. Physical Review, 139(3A), A796.
    - GW近似

### 最新进展 (2024-2025)

21. **Jacob, C. R., et al.** (2024). *Subsystem density‐functional theory (update)*. WIREs Computational Molecular Science, 14(1), e1700.
    - 子系统DFT最新进展

22. **Lee En Hew, N., et al.** (2025). *Density Functional Theory ToolKit (DFTTK) to automate first-principles thermodynamics*. Computational Materials Science, 244, 113587.
    - 自动化热力学计算

23. **Density functional theory and molecular dynamics simulations for resistive switching research** (2024). Surface Science Reports, 84, 100652.
    - DFT+MD综述

## 计算实践文献

### 最佳实践

24. **Natarajan, S. K. & Mathew, K.** (2022). *Best Practices in Plane-Wave Density Functional Theory Calculations*. In: Computational Materials Science.

25. **Lejaeghere, K., et al.** (2016). *Reproducibility in density functional theory calculations of solids*. Science, 351(6280), aad3000.
    - 计算可重复性

### 赝势与基组

26. **Prandini, G., et al.** (2018). *Precision and efficiency in solid-state pseudopotential calculations*. npj Computational Materials, 4(1), 72.
    - SSSP赝势库

27. **Hamann, D. R.** (2013). *Optimized norm-conserving Vanderbilt pseudopotentials*. Physical Review B, 88(8), 085117.
    - ONCV赝势

## 应用文献

### 材料科学

28. **Jain, A., et al.** (2013). *Commentary: The Materials Project: A materials genome approach to accelerating materials innovation*. APL Materials, 1(1), 011002.

29. **Saal, J. E., Kirklin, S., Aykol, M., Meredig, B. & Wolverton, C.** (2013). *Materials design and discovery with high-throughput density functional theory: the Open Quantum Materials Database (OQMD)*. JOM, 65(11), 1501-1509.

### 催化与表面

30. **Nørskov, J. K., et al.** (2009). *Towards the computational design of solid catalysts*. Nature Chemistry, 1(1), 37-46.

31. **Studt, F., et al.** (2010). *Identification of non-precious metal alloy catalysts for selective hydrogenation of acetylene*. Science, 320(5881), 1320-1322.
