"""
科学计算回归测试套件
Scientific Regression Test Suite
================================

本模块包含对DFT、MD、ML计算结果的一致性测试，
确保科学计算的准确性和可重复性。

测试类别:
    1. DFT计算结果一致性测试
    2. MD轨迹可重复性测试
    3. ML势预测稳定性测试
    4. 数值稳定性测试
    5. 跨平台可重复性测试

使用方法:
    pytest tests/regression/test_dft_regression.py -v
    pytest tests/regression/test_md_regression.py -v
    pytest tests/regression/test_ml_regression.py -v
"""

import os
import json
import hashlib
import pickle
import numpy as np
import pytest
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


# =============================================================================
# 回归测试数据管理
# =============================================================================

@dataclass
class DFTReferenceData:
    """DFT参考数据结构"""
    material_id: str
    calculator: str
    functional: str
    encut: float
    kpoints: List[int]
    total_energy: float
    forces: np.ndarray
    stress: np.ndarray
    lattice: np.ndarray
    positions: np.ndarray
    symbols: List[str]
    fermi_energy: Optional[float] = None
    band_gap: Optional[float] = None
    
    def __post_init__(self):
        """确保numpy数组正确存储"""
        if isinstance(self.forces, list):
            self.forces = np.array(self.forces)
        if isinstance(self.stress, list):
            self.stress = np.array(self.stress)
        if isinstance(self.lattice, list):
            self.lattice = np.array(self.lattice)
        if isinstance(self.positions, list):
            self.positions = np.array(self.positions)
    
    def to_dict(self) -> Dict:
        """转换为字典（用于JSON序列化）"""
        return {
            'material_id': self.material_id,
            'calculator': self.calculator,
            'functional': self.functional,
            'encut': self.encut,
            'kpoints': self.kpoints,
            'total_energy': float(self.total_energy),
            'forces': self.forces.tolist(),
            'stress': self.stress.tolist(),
            'lattice': self.lattice.tolist(),
            'positions': self.positions.tolist(),
            'symbols': self.symbols,
            'fermi_energy': float(self.fermi_energy) if self.fermi_energy else None,
            'band_gap': float(self.band_gap) if self.band_gap else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DFTReferenceData':
        """从字典创建实例"""
        return cls(**data)


@dataclass
class MDReferenceData:
    """MD参考数据结构"""
    system_name: str
    potential_type: str
    temperature: float
    pressure: float
    timestep: float
    n_steps: int
    ensemble: str
    
    # 轨迹统计
    final_energy: float
    mean_energy: float
    energy_std: float
    mean_temperature: float
    temp_std: float
    
    # 结构属性
    final_density: float
    mean_density: float
    
    # 可重复性检查
    trajectory_hash: str
    seed: int
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MDReferenceData':
        return cls(**data)


@dataclass
class MLReferenceData:
    """ML势参考数据结构"""
    model_type: str
    model_version: str
    training_dataset: str
    
    # 模型性能
    rmse_energy: float
    rmse_forces: float
    r2_energy: float
    r2_forces: float
    
    # 预测一致性
    test_predictions: Dict[str, float]
    
    # 数值稳定性
    gradient_norm: float
    hessian_eigenvalues: List[float]
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'hessian_eigenvalues': list(self.hessian_eigenvalues) if isinstance(self.hessian_eigenvalues, np.ndarray) else self.hessian_eigenvalues
        }


class ReferenceDataManager:
    """参考数据管理器"""
    
    def __init__(self, reference_dir: Path):
        self.reference_dir = Path(reference_dir)
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        
        self.dft_dir = self.reference_dir / "dft"
        self.md_dir = self.reference_dir / "md"
        self.ml_dir = self.reference_dir / "ml"
        
        for d in [self.dft_dir, self.md_dir, self.ml_dir]:
            d.mkdir(exist_ok=True)
    
    def save_dft_reference(self, data: DFTReferenceData, name: str):
        """保存DFT参考数据"""
        filepath = self.dft_dir / f"{name}.json"
        with open(filepath, 'w') as f:
            json.dump(data.to_dict(), f, indent=2)
    
    def load_dft_reference(self, name: str) -> DFTReferenceData:
        """加载DFT参考数据"""
        filepath = self.dft_dir / f"{name}.json"
        with open(filepath, 'r') as f:
            data = json.load(f)
        return DFTReferenceData.from_dict(data)
    
    def save_md_reference(self, data: MDReferenceData, name: str):
        """保存MD参考数据"""
        filepath = self.md_dir / f"{name}.json"
        with open(filepath, 'w') as f:
            json.dump(data.to_dict(), f, indent=2)
    
    def load_md_reference(self, name: str) -> MDReferenceData:
        """加载MD参考数据"""
        filepath = self.md_dir / f"{name}.json"
        with open(filepath, 'r') as f:
            data = json.load(f)
        return MDReferenceData.from_dict(data)
    
    def save_ml_reference(self, data: MLReferenceData, name: str):
        """保存ML参考数据"""
        filepath = self.ml_dir / f"{name}.json"
        with open(filepath, 'w') as f:
            json.dump(data.to_dict(), f, indent=2)
    
    def load_ml_reference(self, name: str) -> MLReferenceData:
        """加载ML参考数据"""
        filepath = self.ml_dir / f"{name}.json"
        with open(filepath, 'r') as f:
            data = json.load(f)
        return MLReferenceData(**data)
    
    def list_references(self, category: str) -> List[str]:
        """列出所有参考数据"""
        if category == 'dft':
            dir_path = self.dft_dir
        elif category == 'md':
            dir_path = self.md_dir
        elif category == 'ml':
            dir_path = self.ml_dir
        else:
            return []
        
        return [f.stem for f in dir_path.glob("*.json")]


# =============================================================================
# 数值比较工具
# =============================================================================

class NumericalComparator:
    """数值比较工具类"""
    
    @staticmethod
    def compare_scalars(val1: float, val2: float, 
                        rtol: float = 1e-5, 
                        atol: float = 1e-8) -> bool:
        """
        比较两个标量值是否在容差范围内
        
        Args:
            val1: 第一个值
            val2: 第二个值  
            rtol: 相对容差
            atol: 绝对容差
            
        Returns:
            bool: 是否在容差范围内
        """
        return np.isclose(val1, val2, rtol=rtol, atol=atol)
    
    @staticmethod
    def compare_arrays(arr1: np.ndarray, arr2: np.ndarray,
                       rtol: float = 1e-5,
                       atol: float = 1e-8,
                       norm_order: Optional[int] = None) -> bool:
        """
        比较两个数组
        
        Args:
            arr1: 第一个数组
            arr2: 第二个数组
            rtol: 相对容差
            atol: 绝对容差
            norm_order: 使用范数比较（None表示逐元素比较）
            
        Returns:
            bool: 是否在容差范围内
        """
        if arr1.shape != arr2.shape:
            return False
        
        if norm_order is not None:
            diff_norm = np.linalg.norm(arr1 - arr2, ord=norm_order)
            ref_norm = np.linalg.norm(arr2, ord=norm_order)
            return diff_norm <= atol + rtol * ref_norm
        else:
            return np.allclose(arr1, arr2, rtol=rtol, atol=atol)
    
    @staticmethod
    def compare_energies(energy1: float, energy2: float,
                        unit: str = 'eV',
                        per_atom: bool = False,
                        n_atoms: int = 1) -> bool:
        """
        比较能量值（使用DFT适当的容差）
        
        Args:
            energy1: 第一个能量值
            energy2: 第二个能量值
            unit: 能量单位
            per_atom: 是否按原子归一化
            n_atoms: 原子数量
        """
        # DFT能量比较容差（取决于单位）
        tolerances = {
            'eV': (1e-4, 1e-6),      # (rtol, atol)
            'meV': (1e-3, 1e-3),
            'Ha': (1e-6, 1e-8),
            'Ry': (1e-6, 1e-8),
        }
        
        rtol, atol = tolerances.get(unit, (1e-5, 1e-8))
        
        if per_atom:
            energy1 /= n_atoms
            energy2 /= n_atoms
            # 按原子能量需要更严格的容差
            rtol *= 0.1
            atol *= 0.1
        
        return np.isclose(energy1, energy2, rtol=rtol, atol=atol)
    
    @staticmethod
    def compare_forces(forces1: np.ndarray, forces2: np.ndarray,
                      threshold: float = 0.01) -> Dict[str, Any]:
        """
        比较力矢量
        
        Args:
            forces1: 第一个力数组 [n_atoms, 3]
            forces2: 第二个力数组 [n_atoms, 3]
            threshold: 最大允许偏差 (eV/Å)
            
        Returns:
            Dict包含比较结果
        """
        diff = forces1 - forces2
        max_diff = np.max(np.abs(diff))
        rms_diff = np.sqrt(np.mean(diff**2))
        
        # 计算方向一致性
        dot_products = np.sum(forces1 * forces2, axis=1)
        norms1 = np.linalg.norm(forces1, axis=1)
        norms2 = np.linalg.norm(forces2, axis=1)
        
        # 避免除零
        valid = (norms1 > 1e-10) & (norms2 > 1e-10)
        cos_angles = np.zeros_like(norms1)
        cos_angles[valid] = dot_products[valid] / (norms1[valid] * norms2[valid])
        
        return {
            'max_diff': max_diff,
            'rms_diff': rms_diff,
            'mean_cos_angle': np.mean(cos_angles[valid]) if np.any(valid) else 1.0,
            'passed': max_diff <= threshold,
            'threshold': threshold
        }
    
    @staticmethod
    def compute_trajectory_hash(trajectory: List[np.ndarray]) -> str:
        """
        计算轨迹的确定性哈希
        
        Args:
            trajectory: 轨迹帧列表，每帧是位置数组
            
        Returns:
            轨迹哈希字符串
        """
        hasher = hashlib.sha256()
        for frame in trajectory:
            # 四舍五入到合理精度避免浮点误差
            rounded = np.round(frame, decimals=6)
            hasher.update(rounded.tobytes())
        return hasher.hexdigest()[:16]


# =============================================================================
# 测试夹具
# =============================================================================

@pytest.fixture
def reference_manager():
    """提供参考数据管理器"""
    ref_dir = Path(__file__).parent / "fixtures" / "reference_data"
    return ReferenceDataManager(ref_dir)


@pytest.fixture
def numerical_comparator():
    """提供数值比较器"""
    return NumericalComparator()


@pytest.fixture
def tolerance_specs():
    """提供不同测试类型的容差规范"""
    return {
        'strict': {'rtol': 1e-8, 'atol': 1e-10},
        'normal': {'rtol': 1e-5, 'atol': 1e-8},
        'relaxed': {'rtol': 1e-3, 'atol': 1e-6},
        'dft_energy': {'rtol': 1e-6, 'atol': 1e-5},
        'md_trajectory': {'rtol': 1e-4, 'atol': 1e-6},
        'ml_prediction': {'rtol': 1e-3, 'atol': 1e-4},
    }


# =============================================================================
# 报告生成
# =============================================================================

def generate_regression_report(results: List[Dict], output_file: Path):
    """生成回归测试报告"""
    report = {
        'timestamp': str(np.datetime64('now')),
        'total_tests': len(results),
        'passed': sum(1 for r in results if r.get('passed', False)),
        'failed': sum(1 for r in results if not r.get('passed', False)),
        'results': results
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report
