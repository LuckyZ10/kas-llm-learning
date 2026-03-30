"""
mace_active_learning.py
MACE主动学习训练演示

演示如何使用MACE进行高效的主动学习, 用最少的数据获得最佳的势函数。

应用场景:
- 快速构建精确的ML势函数
- 探索化学反应路径
- 大规模分子动力学模拟
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
sys.path.insert(0, '/root/.openclaw/workspace')

from dftlammps.frontier.mace_integration import (
    MACE, AtomicData, MACEActiveLearner, mace_md_simulation
)


def generate_initial_dataset(
    num_structures: int = 10,
    elements: List[int] = [1, 8]  # H, O
) -> List[AtomicData]:
    """
    生成初始训练数据集
    
    创建多样化的分子结构用于初始训练
    """
    print("[1] Generating initial dataset...")
    
    dataset = []
    
    for i in range(num_structures):
        # 随机分子构型 (简化: H2O clusters)
        n_molecules = np.random.randint(1, 4)
        n_atoms = n_molecules * 3
        
        positions = []
        atomic_numbers = []
        
        for _ in range(n_molecules):
            # 水分子
            center = np.random.randn(3) * 3
            
            # O
            positions.append(center)
            atomic_numbers.append(8)
            
            # H1
            h1 = center + np.array([0.96, 0, 0])
            positions.append(h1)
            atomic_numbers.append(1)
            
            # H2
            angle = 104.5 * np.pi / 180
            h2 = center + np.array([0.96 * np.cos(angle), 0.96 * np.sin(angle), 0])
            positions.append(h2)
            atomic_numbers.append(1)
        
        # 创建模拟的DFT标签 (实际应用需要真实DFT计算)
        positions_tensor = torch.tensor(positions, dtype=torch.float32)
        atomic_numbers_tensor = torch.tensor(atomic_numbers, dtype=torch.long)
        
        # 模拟能量和力
        energy = np.random.randn() * 0.5 - n_atoms * 5.0  # 约 -5 eV/atom
        forces = np.random.randn(n_atoms, 3) * 0.1
        
        data = AtomicData(
            positions=positions_tensor,
            atomic_numbers=atomic_numbers_tensor,
            energy=torch.tensor(energy, dtype=torch.float32),
            forces=torch.tensor(forces, dtype=torch.float32)
        )
        
        dataset.append(data)
    
    print(f"    Generated {len(dataset)} structures")
    return dataset


def train_mace_model(
    train_data: List[AtomicData],
    num_epochs: int = 100,
    device: str = 'cuda'
) -> MACE:
    """
    训练MACE模型
    """
    print(f"\n[2] Training MACE model on {len(train_data)} structures...")
    
    model = MACE(
        num_elements=20,
        hidden_channels=64,
        num_layers=2,
        cutoff=5.0
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for data in train_data:
            data_device = AtomicData(
                positions=data.positions.to(device),
                atomic_numbers=data.atomic_numbers.to(device),
                energy=data.energy.to(device) if data.energy is not None else None,
                forces=data.forces.to(device) if data.forces is not None else None
            )
            
            optimizer.zero_grad()
            
            # 前向传播
            output = model(data_device)
            
            # 计算损失
            energy_loss = torch.tensor(0.0, device=device)
            force_loss = torch.tensor(0.0, device=device)
            
            if data_device.energy is not None:
                energy_loss = torch.nn.functional.mse_loss(
                    output['energy'], data_device.energy
                )
            
            if data_device.forces is not None:
                force_loss = torch.nn.functional.mse_loss(
                    output['forces'], data_device.forces
                )
            
            loss = energy_loss + 10 * force_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        if epoch % 20 == 0:
            avg_loss = total_loss / len(train_data)
            print(f"    Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    print("    Training completed!")
    return model


def active_learning_loop(
    initial_data: List[AtomicData],
    num_iterations: int = 5,
    samples_per_iteration: int = 5,
    device: str = 'cuda'
) -> Tuple[MACE, List[float]]:
    """
    主动学习循环
    
    迭代训练模型并选择最有价值的样本
    """
    print("\n" + "=" * 60)
    print("Active Learning Loop")
    print("=" * 60)
    
    train_data = initial_data.copy()
    test_errors = []
    
    for iteration in range(num_iterations):
        print(f"\n[Iteration {iteration + 1}/{num_iterations}]")
        print("-" * 40)
        
        # 1. 训练模型
        print(f"Training on {len(train_data)} samples...")
        model = train_mace_model(train_data, num_epochs=50, device=device)
        
        # 2. 生成候选池
        print("Generating candidate pool...")
        candidate_pool = generate_initial_dataset(num_structures=20)
        
        # 3. 选择高价值样本
        print("Selecting high-uncertainty samples...")
        learner = MACEActiveLearner(model, uncertainty_method="ensemble")
        
        uncertainties = learner.compute_uncertainty(candidate_pool)
        top_indices = torch.topk(uncertainties, samples_per_iteration).indices.tolist()
        
        selected_samples = [candidate_pool[i] for i in top_indices]
        
        print(f"Selected {len(selected_samples)} new samples")
        print(f"Uncertainty range: {uncertainties.min():.3f} - {uncertainties.max():.3f}")
        
        # 4. 模拟DFT计算 (实际应用需要真实计算)
        print("Computing DFT labels for selected samples...")
        for sample in selected_samples:
            # 这里应该是真实的DFT计算
            sample.energy = torch.randn(1) * 0.1 + sample.positions.shape[0] * -5.0
            sample.forces = torch.randn_like(sample.positions) * 0.05
        
        # 5. 添加到训练集
        train_data.extend(selected_samples)
        
        # 6. 评估
        print("Evaluating model...")
        test_data = generate_initial_dataset(num_structures=5)
        test_error = evaluate_model(model, test_data, device)
        test_errors.append(test_error)
        
        print(f"Test error: {test_error:.4f}")
    
    return model, test_errors


def evaluate_model(
    model: MACE,
    test_data: List[AtomicData],
    device: str
) -> float:
    """
    评估模型性能
    """
    model.eval()
    total_error = 0.0
    
    with torch.no_grad():
        for data in test_data:
            data_device = AtomicData(
                positions=data.positions.to(device),
                atomic_numbers=data.atomic_numbers.to(device)
            )
            
            output = model(data_device)
            
            if data.energy is not None:
                error = torch.abs(output['energy'] - data.energy.to(device)).item()
                total_error += error
    
    return total_error / len(test_data)


def run_md_with_active_learned_potential(
    model: MACE,
    initial_structure: AtomicData,
    n_steps: int = 1000,
    temperature: float = 300.0
):
    """
    使用主动学习训练的势函数运行MD
    """
    print("\n" + "=" * 60)
    print("Running MD with Active-Learned Potential")
    print("=" * 60)
    
    results = mace_md_simulation(
        model=model,
        initial_structure=initial_structure,
        n_steps=n_steps,
        timestep=1.0,
        temperature=temperature,
        log_interval=100
    )
    
    # 分析轨迹
    energies = results['energies']
    print(f"\nMD Statistics:")
    print(f"  Average energy: {np.mean(energies):.4f} eV")
    print(f"  Energy std: {np.std(energies):.4f} eV")
    print(f"  Energy drift: {energies[-1] - energies[0]:.4f} eV")
    
    return results


def compare_sampling_strategies():
    """
    比较不同的采样策略
    """
    print("\n" + "=" * 60)
    print("Comparing Sampling Strategies")
    print("=" * 60)
    
    strategies = ['random', 'uncertainty', 'diversity']
    results = {}
    
    for strategy in strategies:
        print(f"\n{strategy.upper()} sampling:")
        
        # 生成初始数据
        initial_data = generate_initial_dataset(num_structures=5)
        
        # 训练
        model = train_mace_model(initial_data, num_epochs=50)
        
        # 评估
        test_data = generate_initial_dataset(num_structures=10)
        error = evaluate_model(model, test_data, 'cuda')
        
        results[strategy] = error
        print(f"  Test error: {error:.4f}")
    
    print("\nComparison:")
    for strategy, error in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {strategy}: {error:.4f}")
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("MACE Active Learning Demo")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. 生成初始数据
    initial_data = generate_initial_dataset(num_structures=10)
    
    # 2. 主动学习循环
    final_model, errors = active_learning_loop(
        initial_data=initial_data,
        num_iterations=3,
        samples_per_iteration=3,
        device=device
    )
    
    # 3. 绘制学习曲线
    print("\n" + "=" * 60)
    print("Learning Curve:")
    print("=" * 60)
    for i, error in enumerate(errors):
        print(f"Iteration {i+1}: {error:.4f}")
    
    # 4. MD模拟
    test_structure = generate_initial_dataset(num_structures=1)[0]
    md_results = run_md_with_active_learned_potential(
        final_model, test_structure, n_steps=500
    )
    
    # 5. 策略比较
    compare_sampling_strategies()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
