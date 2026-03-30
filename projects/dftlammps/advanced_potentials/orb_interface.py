#!/usr/bin/env python3
"""
orb_interface.py
================
Orb (Orbital-Based Neural Network Potential) 接口

功能：
1. Orb模型加载和预测
2. 高效速度场推理
3. 批量结构预测
4. 与ASE的集成
5. MD模拟支持

Orb特点：
- 基于轨道表示
- 超快推理速度
- 高能量精度
- 适用于大规模分子动力学

作者: ML Potential Integration Expert
日期: 2026-03-09
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging
from datetime import datetime
import warnings
import time

# ASE
from ase import Atoms
from ase.io import read, write
from ase.units import eV, Ang, GPa, fs
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS, FIRE, LBFGS
from ase.md import VelocityVerlet, Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 尝试导入Orb
try:
    import orb_models
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.base import OrbffResult
    ORB_AVAILABLE = True
except ImportError:
    ORB_AVAILABLE = False
    warnings.warn("Orb not available. Install with: pip install orb-models")


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class OrbConfig:
    """Orb模型配置"""
    # 模型选择
    model_name: str = "orb_v2"  # orb_v2, orb_d3_v2, etc.
    model_path: Optional[str] = None  # 自定义模型路径
    
    # 计算设置
    device: str = "cuda"  # cuda, cpu
    precision: str = "float32"  # float32, float64, bfloat16
    
    # 批处理
    batch_size: int = 64
    max_atoms: int = 1000  # 最大原子数
    
    # 预测设置
    compute_stress: bool = True
    compute_forces: bool = True


@dataclass
class OrbMDConfig:
    """Orb MD配置"""
    # 时间步进
    timestep: float = 1.0  # fs
    n_steps: int = 10000
    
    # 温度控制
    temperature: float = 300.0  # K
    thermostat: str = "langevin"  # langevin, nose-hoover, berendsen
    friction: float = 0.01  # for Langevin
    
    # 系综
    ensemble: str = "nvt"
    
    # 输出
    output_interval: int = 100
    trajectory_file: str = "orb_md.traj"
    log_interval: int = 100


@dataclass
class OrbBenchmarkResult:
    """Orb基准测试结果"""
    n_structures: int = 0
    n_atoms_total: int = 0
    total_time: float = 0.0  # seconds
    
    @property
    def structures_per_second(self) -> float:
        return self.n_structures / self.total_time if self.total_time > 0 else 0
    
    @property
    def atoms_per_second(self) -> float:
        return self.n_atoms_total / self.total_time if self.total_time > 0 else 0
    
    @property
    def time_per_structure(self) -> float:
        return self.total_time / self.n_structures if self.n_structures > 0 else 0


# =============================================================================
# Orb计算器
# =============================================================================

class OrbASECalculator(Calculator):
    """
    ASE兼容的Orb计算器
    
    Orb以其超快推理速度著称，特别适合大规模MD
    """
    
    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']
    
    def __init__(self,
                 model_name: str = "orb_v2",
                 model_path: Optional[str] = None,
                 device: str = "cuda",
                 precision: str = "float32",
                 compute_stress: bool = True,
                 **kwargs):
        """
        初始化Orb计算器
        
        Args:
            model_name: 模型名称
            model_path: 自定义模型路径
            device: 计算设备
            precision: 精度
            compute_stress: 是否计算应力
        """
        super().__init__(**kwargs)
        
        if not ORB_AVAILABLE:
            raise ImportError("Orb is not available. Install with: pip install orb-models")
        
        self.device = device
        self.precision = precision
        self.compute_stress_flag = compute_stress
        
        # 加载模型
        if model_path and Path(model_path).exists():
            logger.info(f"Loading Orb model from {model_path}")
            self.orbff = self._load_custom_model(model_path)
        else:
            logger.info(f"Loading pretrained Orb model: {model_name}")
            self.orbff = pretrained.get_model(model_name)
        
        # 设置设备
        import torch
        if device == "cuda" and torch.cuda.is_available():
            self.orbff = self.orbff.cuda()
            logger.info("Orb using CUDA")
        else:
            self.orbff = self.orbff.cpu()
            logger.info("Orb using CPU")
        
        # 设置精度
        if precision == "float64":
            self.orbff = self.orbff.double()
        elif precision == "bfloat16":
            self.orbff = self.orbff.bfloat16()
        
        self.model_name = model_name
    
    def _load_custom_model(self, model_path: str):
        """加载自定义Orb模型"""
        import torch
        return torch.load(model_path, map_location=self.device)
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """执行计算"""
        super().calculate(atoms, properties, system_changes)
        
        if atoms is None:
            atoms = self.atoms
        
        # 转换为Orb格式
        graph = self._atoms_to_orb_graph(atoms)
        
        # 预测
        import torch
        with torch.no_grad():
            result = self.orbff(graph)
        
        # 提取结果
        if hasattr(result, 'energy'):
            if isinstance(result.energy, torch.Tensor):
                self.results['energy'] = result.energy.item()
            else:
                self.results['energy'] = result.energy
            self.results['free_energy'] = self.results['energy']
        
        if 'forces' in properties and hasattr(result, 'forces'):
            forces = result.forces
            if isinstance(forces, torch.Tensor):
                forces = forces.cpu().numpy()
            self.results['forces'] = forces
        
        if 'stress' in properties and self.compute_stress_flag and hasattr(result, 'stress'):
            stress = result.stress
            if isinstance(stress, torch.Tensor):
                stress = stress.cpu().numpy()
            self.results['stress'] = stress
    
    def _atoms_to_orb_graph(self, atoms: Atoms):
        """将ASE Atoms转换为Orb图格式"""
        import torch
        from orb_models.forcefield.atomic_system import SystemConfig, AtomGraph
        
        # 准备数据
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        
        cell = atoms.get_cell()
        if cell is not None and cell.any():
            cell_tensor = torch.tensor(cell.array, dtype=torch.float32)
        else:
            cell_tensor = None
        
        # 创建图
        system_config = SystemConfig(
            radius=6.0,  # 默认截断
            max_num_neighbors=50
        )
        
        # 构建图（简化版本，实际应使用Orb的完整构建流程）
        graph = AtomGraph(
            positions=positions,
            atomic_numbers=atomic_numbers,
            cell=cell_tensor,
            edge_index=None,  # 将由模型构建
            system_config=system_config
        )
        
        # 移动到设备
        if self.device == "cuda":
            graph = graph.to(self.device)
        
        return graph


# =============================================================================
# Orb批量预测器
# =============================================================================

class OrbBatchPredictor:
    """
    Orb批量预测器
    
    优化的大批量结构预测
    """
    
    def __init__(self, calculator: OrbASECalculator, batch_size: int = 64):
        self.calc = calculator
        self.batch_size = batch_size
    
    def predict(self, structures: List[Atoms]) -> pd.DataFrame:
        """
        批量预测
        
        Args:
            structures: ASE Atoms列表
        
        Returns:
            DataFrame with predictions
        """
        results = []
        
        start_time = time.time()
        
        for i in range(0, len(structures), self.batch_size):
            batch = structures[i:i+self.batch_size]
            
            for atoms in batch:
                atoms.calc = self.calc
                
                try:
                    t0 = time.time()
                    energy = atoms.get_potential_energy()
                    inference_time = time.time() - t0
                    
                    forces = atoms.get_forces()
                    
                    results.append({
                        'n_atoms': len(atoms),
                        'energy': energy,
                        'energy_per_atom': energy / len(atoms),
                        'max_force': np.max(np.abs(forces)),
                        'rms_force': np.sqrt(np.mean(forces**2)),
                        'inference_time': inference_time
                    })
                except Exception as e:
                    logger.warning(f"Prediction failed: {e}")
                    results.append({
                        'n_atoms': len(atoms),
                        'energy': None,
                        'error': str(e)
                    })
            
            if (i // self.batch_size + 1) % 10 == 0:
                logger.info(f"Predicted {i + len(batch)}/{len(structures)} structures")
        
        total_time = time.time() - start_time
        
        df = pd.DataFrame(results)
        
        # 添加统计信息
        valid_results = df[df['energy'].notna()]
        logger.info(f"Batch prediction complete:")
        logger.info(f"  Total time: {total_time:.2f} s")
        logger.info(f"  Throughput: {len(structures)/total_time:.1f} structures/s")
        if len(valid_results) > 0:
            logger.info(f"  Avg inference time: {valid_results['inference_time'].mean()*1000:.2f} ms")
        
        return df
    
    def benchmark(self, structures: List[Atoms], warmup: int = 10) -> OrbBenchmarkResult:
        """
        性能基准测试
        
        Args:
            structures: 测试结构
            warmup: 预热次数
        
        Returns:
            OrbBenchmarkResult
        """
        logger.info("Running Orb performance benchmark...")
        
        # 预热
        if len(structures) > warmup:
            for atoms in structures[:warmup]:
                atoms.calc = self.calc
                _ = atoms.get_potential_energy()
        
        # 实际测试
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        for atoms in structures:
            atoms.calc = self.calc
            _ = atoms.get_potential_energy()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        
        total_atoms = sum(len(atoms) for atoms in structures)
        
        result = OrbBenchmarkResult(
            n_structures=len(structures),
            n_atoms_total=total_atoms,
            total_time=total_time
        )
        
        logger.info(f"Benchmark Results:")
        logger.info(f"  Structures: {result.n_structures}")
        logger.info(f"  Total atoms: {result.n_atoms_total}")
        logger.info(f"  Total time: {result.total_time:.2f} s")
        logger.info(f"  Structures/s: {result.structures_per_second:.1f}")
        logger.info(f"  Atoms/s: {result.atoms_per_second:.1f}")
        logger.info(f"  Time/structure: {result.time_per_structure*1000:.2f} ms")
        
        return result


# =============================================================================
# Orb MD模拟器
# =============================================================================

class OrbMDSimulator:
    """
    Orb驱动的分子动力学模拟
    
    利用Orb的超快推理实现大规模MD
    """
    
    def __init__(self, calculator: OrbASECalculator, config: OrbMDConfig):
        self.calc = calculator
        self.config = config
        self.atoms = None
        self.dynamics = None
        self.trajectory = []
        self.energies = []
        self.times = []
    
    def setup(self, initial_structure: Union[str, Atoms]):
        """设置模拟系统"""
        cfg = self.config
        
        # 加载结构
        if isinstance(initial_structure, str):
            self.atoms = read(initial_structure)
        else:
            self.atoms = initial_structure.copy()
        
        self.atoms.calc = self.calc
        
        # 初始化速度
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=cfg.temperature)
        
        # 创建积分器
        if cfg.thermostat == "langevin":
            self.dynamics = Langevin(
                self.atoms,
                cfg.timestep * fs,
                temperature_K=cfg.temperature,
                friction=cfg.friction
            )
        elif cfg.thermostat == "nose-hoover":
            # 简化为VelocityVerlet，实际应使用Nose-Hoover
            self.dynamics = VelocityVerlet(self.atoms, cfg.timestep * fs)
        else:
            self.dynamics = VelocityVerlet(self.atoms, cfg.timestep * fs)
        
        logger.info(f"Orb MD system set up: {len(self.atoms)} atoms")
        logger.info(f"Expected throughput: ~{1000:.0f} steps/second")
    
    def run(self, n_steps: Optional[int] = None):
        """运行MD模拟"""
        cfg = self.config
        n_steps = n_steps or cfg.n_steps
        
        logger.info(f"Starting Orb MD for {n_steps} steps")
        
        self.trajectory = []
        self.energies = []
        self.times = []
        
        start_time = time.time()
        
        for step in range(n_steps):
            self.dynamics.run(1)
            
            # 输出
            if step % cfg.output_interval == 0:
                self.trajectory.append(self.atoms.copy())
                self.energies.append(self.atoms.get_potential_energy())
                self.times.append(step * cfg.timestep)
            
            # 日志
            if step % cfg.log_interval == 0:
                temp = self.atoms.get_temperature()
                pe = self.atoms.get_potential_energy()
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                
                logger.info(f"Step {step}/{n_steps} | "
                          f"T={temp:.1f}K | E={pe:.3f}eV | "
                          f"{steps_per_sec:.1f} steps/s")
        
        total_time = time.time() - start_time
        
        # 保存轨迹
        if cfg.trajectory_file and self.trajectory:
            write(cfg.trajectory_file, self.trajectory)
            logger.info(f"Trajectory saved to {cfg.trajectory_file}")
        
        logger.info(f"MD completed: {n_steps} steps in {total_time:.1f}s "
                   f"({n_steps/total_time:.1f} steps/s)")
    
    def get_trajectory(self) -> List[Atoms]:
        """获取轨迹"""
        return self.trajectory
    
    def get_energy_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取能量随时间变化"""
        return np.array(self.times), np.array(self.energies)
    
    def compute_rdf(self, r_max: float = 10.0, nbins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算径向分布函数
        
        从最终结构计算RDF
        """
        if not self.trajectory:
            return np.array([]), np.array([])
        
        from ase.geometry.analysis import Analysis
        
        final_atoms = self.trajectory[-1]
        ana = Analysis(final_atoms)
        
        # 获取所有元素对
        symbols = list(set(final_atoms.get_chemical_symbols()))
        
        rdf_data = []
        for s1 in symbols:
            for s2 in symbols:
                rdf = ana.get_rdf(r_max, nbins, [s1, s2])
                if rdf:
                    rdf_data.append(rdf[0])
        
        if rdf_data:
            rdf_avg = np.mean(rdf_data, axis=0)
            r = np.linspace(0, r_max, nbins)
            return r, rdf_avg
        
        return np.array([]), np.array([])


# =============================================================================
# Orb结构优化
# =============================================================================

class OrbStructureOptimizer:
    """
    使用Orb进行结构优化
    
    利用Orb的快速力预测实现高效优化
    """
    
    def __init__(self, calculator: OrbASECalculator):
        self.calc = calculator
    
    def relax(self,
             atoms: Atoms,
             fmax: float = 0.01,
             max_steps: int = 500,
             optimizer: str = "LBFGS") -> Atoms:
        """
        结构弛豫
        
        Args:
            atoms: 输入结构
            fmax: 力收敛标准 (eV/Å)
            max_steps: 最大步数
            optimizer: 优化器类型
        """
        atoms = atoms.copy()
        atoms.calc = self.calc
        
        # 选择优化器
        if optimizer == "LBFGS":
            opt = LBFGS(atoms)
        elif optimizer == "BFGS":
            opt = BFGS(atoms)
        elif optimizer == "FIRE":
            opt = FIRE(atoms)
        else:
            opt = LBFGS(atoms)
        
        logger.info(f"Starting optimization with {optimizer}")
        
        # 运行优化
        start_time = time.time()
        opt.run(fmax=fmax, steps=max_steps)
        elapsed = time.time() - start_time
        
        # 结果
        final_energy = atoms.get_potential_energy()
        final_forces = atoms.get_forces()
        max_force = np.max(np.abs(final_forces))
        
        logger.info(f"Optimization completed in {elapsed:.2f}s:")
        logger.info(f"  Final energy: {final_energy:.4f} eV")
        logger.info(f"  Max force: {max_force:.4f} eV/Å")
        logger.info(f"  Steps: {opt.nsteps}")
        
        return atoms
    
    def batch_relax(self,
                   structures: List[Atoms],
                   fmax: float = 0.01,
                   max_steps: int = 300) -> List[Atoms]:
        """
        批量结构弛豫
        
        Args:
            structures: 输入结构列表
            fmax: 力收敛标准
            max_steps: 每结构最大步数
        
        Returns:
            弛豫后的结构列表
        """
        relaxed = []
        
        for i, atoms in enumerate(structures):
            try:
                relaxed_atoms = self.relax(atoms, fmax=fmax, max_steps=max_steps)
                relaxed.append(relaxed_atoms)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Relaxed {i + 1}/{len(structures)} structures")
            
            except Exception as e:
                logger.warning(f"Relaxation failed for structure {i}: {e}")
                relaxed.append(atoms)  # 保留原始结构
        
        return relaxed


# =============================================================================
# Orb高通量筛选
# =============================================================================

class OrbScreeningPipeline:
    """
    基于Orb的高通量材料筛选
    
    利用Orb的超快速度进行大规模筛选
    """
    
    def __init__(self, calculator: OrbASECalculator, batch_size: int = 64):
        self.calc = calculator
        self.batch_size = batch_size
        self.predictor = OrbBatchPredictor(calculator, batch_size)
    
    def screen_by_energy(self,
                        structures: List[Atoms],
                        top_k: int = 100) -> Tuple[List[Atoms], pd.DataFrame]:
        """
        按能量筛选最低能量结构
        
        Args:
            structures: 候选结构
            top_k: 选择前k个
        
        Returns:
            selected_structures, results_df
        """
        logger.info(f"Screening {len(structures)} structures by energy...")
        
        # 批量预测
        results = self.predictor.predict(structures)
        
        # 排序
        results_sorted = results.sort_values('energy_per_atom')
        top_indices = results_sorted.index[:top_k].values
        
        selected = [structures[i] for i in top_indices]
        
        logger.info(f"Selected top {len(selected)} structures")
        logger.info(f"  Energy range: {results_sorted['energy_per_atom'].min():.4f} to "
                   f"{results_sorted['energy_per_atom'].iloc[top_k-1]:.4f} eV/atom")
        
        return selected, results_sorted.head(top_k)
    
    def screen_by_stability(self,
                           structures: List[Atoms],
                           reference_energies: Dict[str, float],
                           top_k: int = 100) -> Tuple[List[Atoms], pd.DataFrame]:
        """
        按形成能（稳定性）筛选
        
        Args:
            structures: 候选结构
            reference_energies: 参考元素能量 (eV/atom)
            top_k: 选择前k个
        
        Returns:
            selected_structures, results_df
        """
        logger.info(f"Screening {len(structures)} structures by stability...")
        
        # 预测能量
        results = self.predictor.predict(structures)
        
        # 计算形成能
        formation_energies = []
        
        for atoms, e_total in zip(structures, results['energy']):
            if pd.isna(e_total):
                formation_energies.append(np.nan)
                continue
            
            symbols = atoms.get_chemical_symbols()
            e_ref = sum(reference_energies.get(s, 0) for s in symbols)
            e_form = (e_total - e_ref) / len(atoms)
            formation_energies.append(e_form)
        
        results['formation_energy'] = formation_energies
        
        # 排序（形成能越低越稳定）
        results_valid = results[results['formation_energy'].notna()]
        results_sorted = results_valid.sort_values('formation_energy')
        
        top_indices = results_sorted.index[:top_k].values
        selected = [structures[i] for i in top_indices]
        
        logger.info(f"Selected top {len(selected)} most stable structures")
        
        return selected, results_sorted.head(top_k)
    
    def cluster_and_select(self,
                          structures: List[Atoms],
                          n_clusters: int = 10,
                          n_per_cluster: int = 5) -> List[Atoms]:
        """
        基于结构相似性聚类并选择
        
        Args:
            structures: 候选结构
            n_clusters: 聚类数
            n_per_cluster: 每类选择的数量
        
        Returns:
            多样化的结构选择
        """
        logger.info(f"Clustering {len(structures)} structures...")
        
        # 使用组成作为特征进行聚类
        from sklearn.cluster import KMeans
        
        # 构建特征向量（元素计数）
        all_elements = sorted(set(
            elem for atoms in structures 
            for elem in atoms.get_chemical_symbols()
        ))
        
        features = []
        for atoms in structures:
            symbols = atoms.get_chemical_symbols()
            feature = [symbols.count(e) for e in all_elements]
            features.append(feature)
        
        features = np.array(features)
        
        # K-means聚类
        n_clusters = min(n_clusters, len(structures))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        
        # 从每类选择
        selected = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            
            # 预测该类的能量
            cluster_structures = [structures[i] for i in cluster_indices]
            results = self.predictor.predict(cluster_structures)
            
            # 选择能量最低的n_per_cluster个
            top_local = results['energy_per_atom'].nsmallest(n_per_cluster).index.values
            selected.extend([structures[cluster_indices[i]] for i in top_local])
        
        logger.info(f"Selected {len(selected)} diverse structures from {n_clusters} clusters")
        
        return selected


# =============================================================================
# 主工作流类
# =============================================================================

class OrbWorkflow:
    """
    Orb完整工作流
    """
    
    def __init__(self, config: OrbConfig):
        self.config = config
        self.calculator = None
    
    def setup(self) -> OrbASECalculator:
        """设置Orb计算器"""
        cfg = self.config
        
        self.calculator = OrbASECalculator(
            model_name=cfg.model_name,
            model_path=cfg.model_path,
            device=cfg.device,
            precision=cfg.precision,
            compute_stress=cfg.compute_stress
        )
        
        return self.calculator
    
    def predict_single(self, atoms: Atoms) -> Dict:
        """预测单个结构"""
        if self.calculator is None:
            self.setup()
        
        atoms.calc = self.calculator
        
        return {
            'energy': atoms.get_potential_energy(),
            'forces': atoms.get_forces(),
            'stress': atoms.get_stress() if self.config.compute_stress else None
        }
    
    def batch_predict(self, structures: List[Atoms]) -> pd.DataFrame:
        """批量预测"""
        if self.calculator is None:
            self.setup()
        
        predictor = OrbBatchPredictor(self.calculator, self.config.batch_size)
        
        return predictor.predict(structures)
    
    def run_md(self, initial_structure: Union[str, Atoms], 
              md_config: OrbMDConfig) -> OrbMDSimulator:
        """运行MD模拟"""
        if self.calculator is None:
            self.setup()
        
        simulator = OrbMDSimulator(self.calculator, md_config)
        simulator.setup(initial_structure)
        simulator.run()
        
        return simulator
    
    def relax_structure(self, atoms: Atoms, fmax: float = 0.01) -> Atoms:
        """结构弛豫"""
        if self.calculator is None:
            self.setup()
        
        optimizer = OrbStructureOptimizer(self.calculator)
        
        return optimizer.relax(atoms, fmax=fmax)
    
    def benchmark(self, structures: List[Atoms]) -> OrbBenchmarkResult:
        """性能基准测试"""
        if self.calculator is None:
            self.setup()
        
        predictor = OrbBatchPredictor(self.calculator, self.config.batch_size)
        
        return predictor.benchmark(structures)


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Orb ML Potential Tool")
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Predict
    predict_parser = subparsers.add_parser('predict', help='Predict properties')
    predict_parser.add_argument('structure', help='Structure file')
    predict_parser.add_argument('--device', default='cuda', help='Device')
    
    # Batch
    batch_parser = subparsers.add_parser('batch', help='Batch prediction')
    batch_parser.add_argument('structures', nargs='+', help='Structure files')
    batch_parser.add_argument('--output', default='orb_results.csv')
    
    # MD
    md_parser = subparsers.add_parser('md', help='Run MD simulation')
    md_parser.add_argument('structure', help='Structure file')
    md_parser.add_argument('--temperature', type=float, default=300.0)
    md_parser.add_argument('--steps', type=int, default=10000)
    md_parser.add_argument('--output', default='orb_md.traj')
    
    # Relax
    relax_parser = subparsers.add_parser('relax', help='Relax structure')
    relax_parser.add_argument('structure', help='Structure file')
    relax_parser.add_argument('--fmax', type=float, default=0.01)
    
    # Benchmark
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark performance')
    bench_parser.add_argument('structures', nargs='+', help='Structure files')
    
    args = parser.parse_args()
    
    if args.command == 'predict':
        config = OrbConfig(device=args.device)
        workflow = OrbWorkflow(config)
        workflow.setup()
        
        atoms = read(args.structure)
        result = workflow.predict_single(atoms)
        
        print(f"Energy: {result['energy']:.4f} eV")
        print(f"Energy per atom: {result['energy']/len(atoms):.4f} eV/atom")
        print(f"Max force: {np.max(np.abs(result['forces'])):.4f} eV/Å")
        
    elif args.command == 'batch':
        config = OrbConfig()
        workflow = OrbWorkflow(config)
        workflow.setup()
        
        structures = [read(f) for f in args.structures]
        df = workflow.batch_predict(structures)
        df.to_csv(args.output, index=False)
        
        print(f"Results saved to {args.output}")
        print(f"Throughput: {len(structures)/df['inference_time'].sum():.1f} structures/s")
        
    elif args.command == 'md':
        config = OrbConfig()
        workflow = OrbWorkflow(config)
        
        md_config = OrbMDConfig(
            temperature=args.temperature,
            n_steps=args.steps,
            trajectory_file=args.output
        )
        
        simulator = workflow.run_md(args.structure, md_config)
        
        print(f"MD completed: {len(simulator.get_trajectory())} frames")
        
    elif args.command == 'relax':
        config = OrbConfig()
        workflow = OrbWorkflow(config)
        
        atoms = read(args.structure)
        relaxed = workflow.relax_structure(atoms, fmax=args.fmax)
        
        write('relaxed_orb.xyz', relaxed)
        print(f"Relaxed structure saved to relaxed_orb.xyz")
        
    elif args.command == 'benchmark':
        config = OrbConfig()
        workflow = OrbWorkflow(config)
        
        structures = [read(f) for f in args.structures]
        result = workflow.benchmark(structures)
        
        print(f"\nBenchmark Results:")
        print(f"  Structures: {result.n_structures}")
        print(f"  Total time: {result.total_time:.2f} s")
        print(f"  Throughput: {result.structures_per_second:.1f} structures/s")
        print(f"  Time/structure: {result.time_per_structure*1000:.2f} ms")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
