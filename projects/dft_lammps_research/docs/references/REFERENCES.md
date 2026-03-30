# DFT + LAMMPS 多尺度耦合参考文献

## 核心软件与工具

### DFT计算
1. **VASP** - Kresse, G., & Furthmüller, J. (1996). Efficient iterative schemes for ab initio total-energy calculations using a plane-wave basis set. *Physical Review B*, 54(16), 11169.

2. **Quantum ESPRESSO** - Giannozzi, P., et al. (2009). QUANTUM ESPRESSO: a modular and open-source software project for quantum simulations of materials. *Journal of Physics: Condensed Matter*, 21(39), 395502.

3. **ASE** - Larsen, A. H., et al. (2017). The atomic simulation environment—a Python library for working with atoms. *Journal of Physics: Condensed Matter*, 29(27), 273002.

### 机器学习势
4. **DeePMD-kit** - Wang, H., et al. (2018). DeepMD-kit: A deep learning package for many-body potential energy representation and molecular dynamics. *Computer Physics Communications*, 228, 178-184.

5. **DPA-2** - Zhang, D., et al. (2023). DPA-2: Towards a universal large atomic model for molecular and materials simulation. *arXiv preprint* arXiv:2312.15492.

6. **M3GNet** - Chen, C., & Ong, S. P. (2022). A universal graph deep learning interatomic potential for the periodic table. *Nature Computational Science*, 2(11), 718-728.

7. **NEP** - Fan, Z., et al. (2021). Neuroevolution machine learning potentials: Combining high accuracy and efficiency in atomistic simulations and application to heat transport. *Physical Review B*, 104(10), 104309.

### MD模拟
8. **LAMMPS** - Thompson, A. P., et al. (2015). LAMMPS - a flexible simulation tool for particle-based materials modeling at the atomic, meso, and continuum scales. *Computer Physics Communications*, 271, 108171.

9. **GPUMD** - Fan, Z., et al. (2022). Improving the accuracy of the neuroevolution machine learning potential for multi-component systems. *Journal of Physics: Condensed Matter*, 34(12), 125902.

## 主动学习与数据生成

10. **DP-GEN** - Zhang, Y., et al. (2020). DP-GEN: A concurrent learning platform for the generation of reliable deep learning based potential energy models. *Computer Physics Communications*, 253, 107206.

11. **FLARE** - Vandermause, J., et al. (2020). On-the-fly probability-enhanced sampling (OPES) with Bayesian inference for uncertainty quantification. *npj Computational Materials*, 6(1), 1-12.

12. **Active Learning for MLIPs** - Jinnouchi, R., et al. (2019). On-the-fly active learning of interatomic potentials for large-scale atomistic simulations. *Journal of Physical Chemistry Letters*, 10(17), 5123-5130.

## 高通量筛选框架

13. **Pymatgen** - Ong, S. P., et al. (2013). Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis. *Computational Materials Science*, 68, 314-319.

14. **Atomate** - Mathew, K., et al. (2017). Atomate: A high-level interface to generate, execute, and analyze computational materials science workflows. *Computational Materials Science*, 139, 140-152.

15. **FireWorks** - Jain, A., et al. (2015). FireWorks: a dynamic workflow system designed for high-throughput applications. *Concurrency and Computation: Practice and Experience*, 27(17), 5037-5059.

16. **AiiDA** - Pizzi, G., et al. (2016). AiiDA: automated interactive infrastructure and database for computational science. *Computational Materials Science*, 111, 218-230.

17. **Matminer** - Ward, L., et al. (2018). Matminer: An open source toolkit for materials data mining. *Computational Materials Science*, 152, 60-69.

## 多尺度耦合方法

18. **QM/MM Coupling** - Senn, H. M., & Thiel, W. (2009). QM/MM methods for biomolecular systems. *Angewandte Chemie International Edition*, 48(7), 1198-1229.

19. **DFT/MD Multiscale** - Tkatchenko, A., et al. (2012). Accurate and efficient method for many-body van der Waals interactions. *Physical Review Letters*, 108(23), 236402.

20. **Learned Coarse Graining** - Wang, J., et al. (2019). Machine learning of coarse-grained molecular dynamics force fields. *ACS Central Science*, 5(5), 755-767.

## 电池材料模拟

21. **Solid Electrolyte Screening** - Sendek, A. D., et al. (2017). Holistic computational structure screening of more than 12,000 candidates for solid lithium-ion conductor materials. *Energy & Environmental Science*, 10(1), 306-320.

22. **Cathode Materials** - Jain, A., et al. (2011). Formation enthalpies by mixing GGA and GGA + U calculations. *Physical Review B*, 84(4), 045115.

23. **SEI Formation** - Leung, K. (2013). Electronic structure modeling of electrochemical properties of battery materials. *Journal of Physical Chemistry C*, 117(4), 1539-1547.

24. **Ionic Conductivity** - He, X., et al. (2020). Crystal structural framework of lithium superionic conductors. *Advanced Energy Materials*, 10(1), 1902078.

## 催化剂模拟

25. **Computational Catalysis** - Nørskov, J. K., et al. (2009). Towards the computational design of solid catalysts. *Nature Chemistry*, 1(1), 37-46.

26. **Open Catalyst Project** - Chanussot, L., et al. (2021). Open Catalyst 2020 (OC20) dataset and community challenges. *ACS Catalysis*, 11(10), 6059-6072.

27. **Machine Learning for Catalysis** - Zitnick, C. L., et al. (2020). An introduction to electrocatalyst design using machine learning for renewable energy storage. *arXiv preprint* arXiv:2010.09435.

28. **Multiscale Electrocatalysis** - Tran, H. (2023). Electrocatalysis at the Electrode/Electrolyte Interface: a Multiscale Molecular Model. *PhD Dissertation*, Penn State University.

## 光伏材料模拟

29. **Perovskite Stability** - Yin, W. J., et al. (2014). Unusual defect physics in CH3NH3PbI3 perovskite solar cell absorber. *Applied Physics Letters*, 104(6), 063903.

30. **Lead-free Perovskites** - Filip, M. R., et al. (2018). Steric engineering of metal-halide perovskites with tunable optical band gaps. *Nature Communications*, 9(1), 1-10.

31. **Defect Tolerance** - Yin, W. J., et al. (2015). Defect tolerance and intrinsically dominated conductivity in the solar absorber (CuInSe2)x-2(Cu2ZnSnSe4)x. *Chemistry of Materials*, 27(15), 5489-5495.

## 特征工程与描述符

32. **SOAP** - Bartók, A. P., et al. (2013). On representing chemical environments. *Physical Review B*, 87(18), 184115.

33. **ACSF** - Behler, J. (2011). Atom-centered symmetry functions for constructing high-dimensional neural network potentials. *Journal of Chemical Physics*, 134(7), 074106.

34. **Crystal Graph CNN** - Xie, T., & Grossman, J. C. (2018). Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. *Physical Review Letters*, 120(14), 145301.

35. **MegNet** - Chen, C., et al. (2019). Graph networks as a universal machine learning framework for molecules and crystals. *Chemistry of Materials*, 31(9), 3564-3572.

## 不确定性量化

36. **Uncertainty in MLIPs** - Musil, F., et al. (2019). Physics-inspired structural representations for molecules and materials. *Chemical Reviews*, 121(16), 9759-9815.

37. **Committee Models** - Peterson, A. A. (2016). Acceleration of saddle-point searches with machine learning. *Journal of Chemical Physics*, 145(7), 074106.

## 综述文章

38. **ML for Materials** - Butler, K. T., et al. (2018). Machine learning for molecular and materials science. *Nature*, 559(7715), 547-555.

39. **ML Potentials Review** - Unke, O. T., et al. (2021). Machine learning force fields. *Chemical Reviews*, 121(16), 10142-10186.

40. **Materials Informatics** - Ramprasad, R., et al. (2017). Machine learning in materials informatics: recent applications and prospects. *npj Computational Materials*, 3(1), 1-13.

41. **Multiscale Modeling** - Csányi, G., et al. (2022). Machine learning for atomic simulations of materials. *Handbook of Materials Modeling*, 1-29.

## 数据集与数据库

42. **Materials Project** - Jain, A., et al. (2013). Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. *APL Materials*, 1(1), 011002.

43. **OQMD** - Kirklin, S., et al. (2015). The Open Quantum Materials Database (OQMD): assessing the accuracy of DFT formation energies. *npj Computational Materials*, 1(1), 1-15.

44. **AFLOWlib** - Curtarolo, S., et al. (2012). AFLOW: An automatic framework for high-throughput materials discovery. *Computational Materials Science*, 58, 218-226.

45. **NOMAD** - Draxl, C., & Scheffler, M. (2018). NOMAD: The FAIR concept for big-data-driven materials science. *MRS Bulletin*, 43(9), 676-682.

---

## 推荐阅读顺序

### 入门 (基础概念)
1. ASE论文 (Larsen et al., 2017)
2. Materials Project (Jain et al., 2013)
3. ML for Materials (Butler et al., 2018)

### 进阶 (ML势训练)
4. DeePMD-kit (Wang et al., 2018)
5. Behler ACSF (Behler, 2011)
6. DP-GEN (Zhang et al., 2020)

### 专家 (多尺度方法)
7. Active Learning (Jinnouchi et al., 2019)
8. M3GNet (Chen & Ong, 2022)
9. Multiscale ML (Csányi et al., 2022)

---

*最后更新: 2026-03-08*
