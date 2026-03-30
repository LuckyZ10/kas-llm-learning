#!/usr/bin/env python3
"""
LAMMPS + 机器学习势集成
ml_potential_interface.py

支持: DeepMD, NequIP, MACE, TorchANI
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict
import json
from pathlib import Path

class MLPotentialInterface:
    """机器学习势接口基类"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        self.model_path = Path(model_path)
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.type_map = {}
    
    def load_model(self):
        """加载ML模型 - 子类实现"""
        raise NotImplementedError
    
    def calculate(self, positions: np.ndarray, 
                  cell: np.ndarray,
                  atom_types: list) -> Tuple[float, np.ndarray]:
        """
        计算能量和力
        
        Returns:
            energy: 总能量 (eV)
            forces: 力 (eV/Å), shape (natoms, 3)
        """
        raise NotImplementedError


class DeepMDInterface(MLPotentialInterface):
    """DeepMD-kit接口"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        super().__init__(model_path, device)
        try:
            from deepmd.infer import DeepPot
            self.DeepPot = DeepPot
        except ImportError:
            raise ImportError("DeepMD-kit not installed. Install: pip install deepmd-kit")
    
    def load_model(self):
        """加载DeepMD模型"""
        self.model = self.DeepPot(str(self.model_path))
        self.type_map = {i: t for i, t in enumerate(self.model.get_type_map())}
        print(f"Loaded DeepMD model: {self.model_path}")
        print(f"Type map: {self.type_map}")
    
    def calculate(self, positions: np.ndarray,
                  cell: np.ndarray,
                  atom_types: list) -> Tuple[float, np.ndarray]:
        """
        使用DeepMD计算
        
        atom_types: 元素符号列表, e.g., ['H', 'O', 'O', 'H']
        """
        # 转换类型为索引
        type_indices = [list(self.type_map.values()).index(t) for t in atom_types]
        
        # 计算
        energy, force, virial = self.model.eval(
            coords=positions.reshape(1, -1),
            cells=cell.reshape(1, -1),
            atom_types=np.array(type_indices).reshape(1, -1)
        )
        
        return float(energy), force.reshape(-1, 3)


class NequIPInterface(MLPotentialInterface):
    """NequIP接口"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        super().__init__(model_path, device)
        try:
            from nequip.model import model_from_config
            from nequip.data import AtomicData
            self.model_from_config = model_from_config
            self.AtomicData = AtomicData
        except ImportError:
            raise ImportError("NequIP not installed. See https://github.com/mir-group/nequip")
    
    def load_model(self):
        """加载NequIP模型"""
        # NequIP通常从训练配置加载
        import yaml
        config_file = self.model_path / 'config.yaml'
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        self.model = self.model_from_config(config)
        checkpoint = torch.load(self.model_path / 'best_model.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded NequIP model from {self.model_path}")
    
    def calculate(self, positions: np.ndarray,
                  cell: np.ndarray,
                  atom_types: list) -> Tuple[float, np.ndarray]:
        """使用NequIP计算"""
        # 构建AtomicData对象
        atomic_numbers = self._get_atomic_numbers(atom_types)
        
        data = self.AtomicData(
            pos=torch.tensor(positions, dtype=torch.float32, device=self.device),
            cell=torch.tensor(cell, dtype=torch.float32, device=self.device).unsqueeze(0),
            atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long, device=self.device),
            pbc=torch.tensor([True, True, True], device=self.device)
        )
        
        with torch.no_grad():
            out = self.model(data)
            energy = out['total_energy'].item()
            forces = out['forces'].cpu().numpy()
        
        return energy, forces
    
    def _get_atomic_numbers(self, atom_types):
        """元素符号转原子序数"""
        periodic_table = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16,
            'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24,
            'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32,
            # ... 可扩展
        }
        return [periodic_table.get(t, 0) for t in atom_types]


class MACEInterface(MLPotentialInterface):
    """MACE接口"""
    
    def __init__(self, model_path: str, device: str = 'auto', model_size: str = 'medium'):
        super().__init__(model_path, device)
        self.model_size = model_size
        try:
            from mace.calculators import mace_mp
            self.mace_mp = mace_mp
        except ImportError:
            raise ImportError("MACE not installed. See https://github.com/ACEsuit/mace")
    
    def load_model(self):
        """加载MACE模型"""
        self.model = self.mace_mp(model=self.model_size, device=self.device)
        print(f"Loaded MACE-{self.model_size} model")
    
    def calculate(self, positions: np.ndarray,
                  cell: np.ndarray,
                  atom_types: list) -> Tuple[float, np.ndarray]:
        """使用MACE计算"""
        from ase import Atoms
        
        atoms = Atoms(
            symbols=atom_types,
            positions=positions,
            cell=cell,
            pbc=True
        )
        
        atoms.calc = self.model
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        return float(energy), forces


class TorchANIInterface(MLPotentialInterface):
    """TorchANI接口 (ANI-2x)"""
    
    def __init__(self, model_path: str = None, device: str = 'auto'):
        super().__init__(model_path, device)
        try:
            import torchani
            self.torchani = torchani
        except ImportError:
            raise ImportError("TorchANI not installed. pip install torchani")
    
    def load_model(self):
        """加载ANI模型"""
        self.model = self.torchani.models.ANI2x(periodic_table_index=True).to(self.device)
        self.model.eval()
        print("Loaded ANI-2x model")
    
    def calculate(self, positions: np.ndarray,
                  cell: np.ndarray,
                  atom_types: list) -> Tuple[float, np.ndarray]:
        """使用ANI计算"""
        # 转换元素符号为原子序数
        atomic_numbers = self._get_atomic_numbers(atom_types)
        
        species = torch.tensor([atomic_numbers], device=self.device)
        coords = torch.tensor(positions.reshape(1, -1, 3), 
                             dtype=torch.float32, device=self.device, requires_grad=True)
        
        with torch.no_grad():
            energy = self.model((species, coords)).energies
            
        # 计算力 (需要梯度)
        coords.requires_grad_(True)
        energy = self.model((species, coords)).energies
        forces = -torch.autograd.grad(energy, coords)[0]
        
        return energy.item(), forces.cpu().numpy().reshape(-1, 3)
    
    def _get_atomic_numbers(self, atom_types):
        periodic_table = {
            'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16, 'Cl': 17
            # ANI只支持这些元素
        }
        return [periodic_table.get(t, 0) for t in atom_types]


class LAMMPSMLDriver:
    """
    LAMMPS与ML势的驱动接口
    
    可用于fix external或socket通讯
    """
    
    def __init__(self, ml_interface: MLPotentialInterface):
        self.ml_interface = ml_interface
        self.step_count = 0
        self.energy_history = []
        self.force_history = []
    
    def step_callback(self, timestep: int, positions: np.ndarray,
                      cell: np.ndarray, atom_types: list) -> np.ndarray:
        """
        LAMMPS每步调用的回调函数
        
        Returns:
            forces: 返回给LAMMPS的力
        """
        energy, forces = self.ml_interface.calculate(positions, cell, atom_types)
        
        self.step_count = timestep
        self.energy_history.append(energy)
        self.force_history.append(forces.copy())
        
        return forces
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'steps': self.step_count,
            'mean_energy': np.mean(self.energy_history) if self.energy_history else 0,
            'force_std': np.std(self.force_history) if self.force_history else 0
        }


def create_ml_potential(ml_type: str, model_path: str, **kwargs) -> MLPotentialInterface:
    """
    工厂函数创建ML势接口
    
    Args:
        ml_type: 'deepmd', 'nequip', 'mace', 'torchani'
        model_path: 模型路径
        **kwargs: 额外参数
    
    Returns:
        MLPotentialInterface实例
    """
    interfaces = {
        'deepmd': DeepMDInterface,
        'nequip': NequIPInterface,
        'mace': MACEInterface,
        'torchani': TorchANIInterface
    }
    
    if ml_type.lower() not in interfaces:
        raise ValueError(f"Unknown ML type: {ml_type}. Supported: {list(interfaces.keys())}")
    
    interface = interfaces[ml_type.lower()](model_path, **kwargs)
    interface.load_model()
    
    return interface


# 示例使用
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ML potential interface')
    parser.add_argument('--type', required=True, choices=['deepmd', 'nequip', 'mace', 'torchani'])
    parser.add_argument('--model', required=True, help='Model path')
    parser.add_argument('--test-pos', help='Test positions (JSON file)')
    
    args = parser.parse_args()
    
    # 创建接口
    ml_pot = create_ml_potential(args.type, args.model)
    
    # 测试计算
    if args.test_pos:
        import json
        with open(args.test_pos, 'r') as f:
            test_data = json.load(f)
        
        positions = np.array(test_data['positions'])
        cell = np.array(test_data['cell'])
        atom_types = test_data['atom_types']
        
        energy, forces = ml_pot.calculate(positions, cell, atom_types)
        
        print(f"Energy: {energy:.6f} eV")
        print(f"Forces shape: {forces.shape}")
        print(f"Force magnitude: {np.linalg.norm(forces, axis=1)}")
    else:
        # 默认测试 (水分子)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0]
        ])
        cell = np.array([10.0, 10.0, 10.0])
        atom_types = ['O', 'H', 'H']
        
        energy, forces = ml_pot.calculate(positions, cell, atom_types)
        print(f"Test calculation - Energy: {energy:.6f} eV")
