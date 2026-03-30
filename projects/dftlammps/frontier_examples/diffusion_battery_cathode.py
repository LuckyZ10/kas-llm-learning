"""
diffusion_battery_cathode.py
扩散模型设计电池正极材料

演示如何使用CDVAE/DiffCSP等扩散模型生成新型电池正极材料结构。

应用场景:
- 高通量生成候选正极结构
- 优化离子扩散通道
- 探索新型高电压正极材料
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
sys.path.insert(0, '/root/.openclaw/workspace')

from dftlammps.frontier.diffusion_materials import (
    CDVAE, DiffCSP, CrystalGenerator, CrystalFeatures,
    evaluate_structure_quality
)


def design_battery_cathode(
    target_composition: Dict[str, float] = None,
    target_voltage: float = 4.5,
    num_candidates: int = 20
) -> List[Dict]:
    """
    设计电池正极材料
    
    Args:
        target_composition: 目标组分, 如 {'Li': 1, 'Co': 1, 'O': 2}
        target_voltage: 目标工作电压 (V)
        num_candidates: 生成候选数量
    Returns:
        候选结构列表
    """
    print("=" * 60)
    print("Battery Cathode Design using Diffusion Models")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 初始化CDVAE生成器
    print("\n[1] Initializing CDVAE generator...")
    generator = CrystalGenerator(model_type='cdvae', device=device)
    
    # 设置目标组分 (默认LiCoO2-like)
    if target_composition is None:
        target_composition = {'Li': 1, 'Co': 1, 'O': 2}
    
    print(f"\n[2] Target composition: {target_composition}")
    print(f"    Target voltage: {target_voltage} V")
    
    # 生成候选结构
    print(f"\n[3] Generating {num_candidates} candidate structures...")
    
    candidates = []
    
    with torch.no_grad():
        for i in range(num_candidates):
            print(f"  Generating candidate {i+1}/{num_candidates}...")
            
            # 从潜在空间采样
            z = torch.randn(generator.model.latent_dim, device=device)
            
            # 生成结构
            num_atoms = sum(int(v) for v in target_composition.values())
            struct = generator.model.decode(z, num_atoms=num_atoms)
            
            # 评估结构质量
            metrics = evaluate_structure_quality(struct)
            
            # 计算"正极适用性分数"
            cathode_score = evaluate_cathode_suitability(struct, target_composition)
            
            candidate = {
                'id': i + 1,
                'structure': struct,
                'quality_metrics': metrics,
                'cathode_score': cathode_score,
                'predicted_voltage': predict_voltage(struct)
            }
            
            candidates.append(candidate)
    
    # 排序并筛选
    candidates.sort(key=lambda x: x['cathode_score'], reverse=True)
    
    # 输出结果
    print("\n[4] Top 5 Candidates:")
    print("-" * 60)
    
    for i, cand in enumerate(candidates[:5], 1):
        print(f"\nRank {i} (ID: {cand['id']}):")
        print(f"  Cathode Score: {cand['cathode_score']:.3f}")
        print(f"  Predicted Voltage: {cand['predicted_voltage']:.2f} V")
        print(f"  Structure Valid: {cand['quality_metrics']['valid_coords']}")
        print(f"  Min Atomic Distance: {cand['quality_metrics']['min_atomic_distance']:.3f} Å")
        print(f"  Num Atoms: {cand['structure'].num_atoms}")
    
    return candidates


def evaluate_cathode_suitability(
    structure: CrystalFeatures,
    target_composition: Dict[str, float]
) -> float:
    """
    评估结构作为正极材料的适用性
    
    基于结构特征计算启发式分数
    """
    score = 0.0
    
    # 1. 结构合理性
    metrics = evaluate_structure_quality(structure)
    if metrics['valid_coords'] and metrics['no_overlap']:
        score += 0.3
    
    # 2. 晶胞大小 (正极材料通常有适中的晶胞)
    volume = torch.prod(structure.lengths).item()
    if 50 < volume < 500:
        score += 0.2
    
    # 3. 对称性 (高对称性通常更好)
    # 简化: 检查角度接近90度
    angle_deviation = torch.sum(torch.abs(structure.angles - 90)).item()
    if angle_deviation < 10:
        score += 0.2
    
    # 4. 离子通道评估 (简化: 检查空隙)
    # 实际应用中可以使用Voronoi分析
    void_fraction = estimate_void_fraction(structure)
    if 0.3 < void_fraction < 0.7:
        score += 0.3
    
    return score


def predict_voltage(structure: CrystalFeatures) -> float:
    """
    预测工作电压 (简化模型)
    
    实际应用应该使用DFT或训练好的ML模型
    """
    # 基于晶胞参数和组成的启发式预测
    base_voltage = 3.5
    
    # 晶胞体积影响
    volume = torch.prod(structure.lengths).item()
    volume_factor = (volume - 100) / 200  # 归一化
    
    # 原子类型影响 (过渡金属增加电压)
    tm_count = sum(1 for z in structure.atom_types if z in [25, 26, 27, 28])  # Mn, Fe, Co, Ni
    tm_factor = tm_count * 0.2
    
    voltage = base_voltage + volume_factor * 0.5 + tm_factor
    
    return np.clip(voltage, 2.5, 5.0)


def estimate_void_fraction(structure: CrystalFeatures) -> float:
    """估算空隙分数"""
    # 简化估算
    volume = torch.prod(structure.lengths).item()
    n_atoms = structure.num_atoms
    
    # 假设每个原子占据约10 Å³
    occupied_volume = n_atoms * 10
    void_fraction = 1 - (occupied_volume / volume)
    
    return np.clip(void_fraction, 0, 1)


def optimize_cathode_structure(
    initial_structure: CrystalFeatures,
    target_voltage: float = 4.5,
    num_steps: int = 100
) -> CrystalFeatures:
    """
    优化正极结构以达到目标电压
    
    使用潜在空间优化
    """
    print("\n" + "=" * 60)
    print("Optimizing Cathode Structure")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = CrystalGenerator(model_type='cdvae', device=device)
    
    # 编码初始结构
    z, mu, logvar = generator.model.encode(
        initial_structure.atom_types,
        initial_structure.frac_coords,
        initial_structure.lengths,
        initial_structure.angles
    )
    
    # 优化潜在向量
    z_opt = z.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z_opt], lr=0.01)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # 解码生成结构
        struct = generator.model.decode(z_opt, num_atoms=initial_structure.num_atoms)
        
        # 预测电压
        voltage = predict_voltage(struct)
        
        # 目标: 接近目标电压
        loss = (voltage - target_voltage) ** 2
        
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step}: Voltage = {voltage:.3f} V, Loss = {loss.item():.4f}")
    
    # 生成最终结构
    with torch.no_grad():
        final_struct = generator.model.decode(z_opt, num_atoms=initial_structure.num_atoms)
    
    final_voltage = predict_voltage(final_struct)
    print(f"\nFinal predicted voltage: {final_voltage:.3f} V")
    
    return final_struct


def batch_screen_cathodes(
    compositions: List[Dict[str, float]],
    num_per_composition: int = 10
) -> Dict[str, List[Dict]]:
    """
    批量筛选多种组分的正极材料
    """
    print("\n" + "=" * 60)
    print("Batch Cathode Screening")
    print("=" * 60)
    
    results = {}
    
    for comp in compositions:
        formula = ''.join(f"{k}{int(v) if v==int(v) else v}" for k, v in comp.items())
        print(f"\nScreening {formula}...")
        
        candidates = design_battery_cathode(
            target_composition=comp,
            num_candidates=num_per_composition
        )
        
        results[formula] = candidates
    
    # 输出汇总
    print("\n" + "=" * 60)
    print("Screening Summary")
    print("=" * 60)
    
    for formula, candidates in results.items():
        best = candidates[0] if candidates else None
        if best:
            print(f"\n{formula}:")
            print(f"  Best cathode score: {best['cathode_score']:.3f}")
            print(f"  Predicted voltage: {best['predicted_voltage']:.2f} V")
    
    return results


if __name__ == "__main__":
    # 演示1: 单组分设计
    print("\n" + "#" * 60)
    print("# Demo 1: Single Composition Design")
    print("#" * 60)
    
    candidates = design_battery_cathode(
        target_composition={'Li': 1, 'Mn': 2, 'O': 4},  # LiMn2O4-like
        target_voltage=4.2,
        num_candidates=10
    )
    
    # 演示2: 结构优化
    if candidates:
        print("\n" + "#" * 60)
        print("# Demo 2: Structure Optimization")
        print("#" * 60)
        
        best_initial = candidates[0]['structure']
        optimized = optimize_cathode_structure(
            best_initial,
            target_voltage=4.5,
            num_steps=50
        )
    
    # 演示3: 批量筛选
    print("\n" + "#" * 60)
    print("# Demo 3: Batch Screening")
    print("#" * 60)
    
    compositions = [
        {'Li': 1, 'Co': 1, 'O': 2},
        {'Li': 1, 'Mn': 2, 'O': 4},
        {'Na': 1, 'Fe': 1, 'O': 2},
    ]
    
    batch_results = batch_screen_cathodes(compositions, num_per_composition=5)
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
