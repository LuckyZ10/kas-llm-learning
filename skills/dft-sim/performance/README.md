# DFT计算性能优化模块

本模块提供DFT计算性能优化的全面指南，涵盖并行计算、内存管理、GPU加速等方面。

---

## 文档列表

### [optimization_guide.md](optimization_guide.md) - 性能优化指南

**内容涵盖**:
- 并行策略 (NCORE, KPAR, NPAR配置)
- QE并行参数优化
- 内存控制与估算
- IO优化策略
- 算法加速技巧
- 快速检查清单

**适用场景**:
- 提升现有计算效率
- 优化资源分配
- 大规模计算规划

---

### [gpu_acceleration.md](gpu_acceleration.md) - GPU加速指南

**内容涵盖**:
- GPU加速原理与硬件选择
- VASP GPU版本编译与配置
- QE GPU支持详解
- 混合精度计算
- 多GPU并行优化
- 成本效益分析

**适用场景**:
- GPU集群配置
- 大规模加速计算
- 成本优化决策

---

## 快速参考

### 并行配置速查

| 体系大小 | CPU/GPU | 推荐配置 | 预期效率 |
|----------|---------|----------|----------|
| <50原子 | CPU | NCORE=4, KPAR=k点 | 90%+ |
| 50-200原子 | CPU | NCORE=8, KPAR=4 | 85%+ |
| 200-500原子 | GPU | 4×A100, -npool 4 | 8x加速 |
| >500原子 | GPU | 8×H100, -npool 8 | 12x加速 |

### 内存估算公式

```python
# VASP (GB)
mem_vasp = n_electrons * n_bands * n_kpoints * 32 / 1e9

# QE (GB)
mem_qe = n_pwaves * n_bands * n_kpoints * 16 / 1e9
```

---

## 性能诊断工具

### VASP计时分析

```bash
# 提取计算时间
grep "Elapsed time" OUTCAR
grep "LOOP+" OUTCAR
```

### QE性能统计

```bash
# 详细计时输出
mpirun -np 64 pw.x -nk 8 -v -in pw.in
```

### GPU监控

```bash
# 实时监控
watch -n 1 nvidia-smi

# 详细统计
nvidia-smi dmon -s pucvmet
```

---

## 最佳实践总结

### 1. 并行配置

- **k点多**: 优先使用KPAR/-nk
- **能带多**: 考虑NPAR/-ndiag
- **共享内存节点**: 增大NCORE

### 2. 内存管理

- **大体系**: 使用LREAL=Auto
- **临时文件**: 放NVMe SSD
- **定期清理**: WAVECAR/CHGCAR

### 3. GPU加速

- **小体系 (<100原子)**: CPU可能更快
- **中体系 (100-500原子)**: GPU优势明显
- **BSE/MD**: GPU加速最佳

---

*最后更新: 2026-03-08*
