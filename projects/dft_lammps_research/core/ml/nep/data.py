"""
nep_training/data.py
====================
数据加载和预处理模块

包含:
- NEP数据集类
- 高效数据加载器
- 数据增强
"""

import os
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any, Iterator, Callable
from pathlib import Path
from dataclasses import dataclass
import pickle
import hashlib

from ase import Atoms
from ase.io import read, write
from ase.io.vasp import read_vasp_out
from ase.neighborlist import neighbor_list

logger = logging.getLogger(__name__)


@dataclass
class NEPFrame:
    """NEP单帧数据结构"""
    atoms: Atoms
    energy: float
    forces: np.ndarray
    stress: Optional[np.ndarray] = None
    virial: Optional[np.ndarray] = None
    weight: float = 1.0  # 样本权重
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class NEPDataset:
    """
    NEP数据集类
    
    高效的内存数据集管理，支持懒加载和缓存
    """
    
    def __init__(self, frames: Optional[List[NEPFrame]] = None,
                 xyz_file: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        self.frames = frames or []
        self.xyz_file = xyz_file
        self.cache_dir = cache_dir
        
        self._cache_loaded = False
        self._type_map: Optional[List[str]] = None
        
        if xyz_file and not frames:
            self._load_from_xyz(xyz_file)
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> NEPFrame:
        return self.frames[idx]
    
    def _load_from_xyz(self, xyz_file: str):
        """从XYZ文件加载"""
        logger.info(f"Loading dataset from {xyz_file}")
        
        # 检查缓存
        if self.cache_dir and self._try_load_cache(xyz_file):
            logger.info("Loaded from cache")
            return
        
        # 使用ASE读取
        atoms_list = read(xyz_file, index=':', format='extxyz')
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        
        for atoms in atoms_list:
            frame = self._atoms_to_frame(atoms)
            if frame:
                self.frames.append(frame)
        
        logger.info(f"Loaded {len(self.frames)} frames")
        
        # 保存缓存
        if self.cache_dir:
            self._save_cache(xyz_file)
    
    def _atoms_to_frame(self, atoms: Atoms) -> Optional[NEPFrame]:
        """将ASE Atoms转换为NEPFrame"""
        # 能量
        energy = None
        if atoms.calc is not None:
            try:
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
            except:
                pass
        
        # 从info/arrays读取
        if energy is None:
            energy = atoms.info.get('energy', None)
        if 'forces' not in locals():
            forces = atoms.arrays.get('forces', None)
            if forces is None:
                forces = np.zeros((len(atoms), 3))
        
        if energy is None:
            return None
        
        # 应力
        stress = atoms.info.get('stress', None)
        virial = atoms.info.get('virial', None)
        
        # 权重
        weight = atoms.info.get('weight', 1.0)
        
        return NEPFrame(
            atoms=atoms,
            energy=energy,
            forces=np.array(forces),
            stress=np.array(stress) if stress is not None else None,
            virial=np.array(virial) if virial is not None else None,
            weight=weight,
            metadata=dict(atoms.info)
        )
    
    def _try_load_cache(self, xyz_file: str) -> bool:
        """尝试从缓存加载"""
        cache_path = self._get_cache_path(xyz_file)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                self.frames = cached['frames']
                self._cache_loaded = True
                return True
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return False
    
    def _save_cache(self, xyz_file: str):
        """保存缓存"""
        cache_path = self._get_cache_path(xyz_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'frames': self.frames,
                    'source': xyz_file
                }, f)
            logger.info(f"Saved cache to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _get_cache_path(self, xyz_file: str) -> Path:
        """获取缓存路径"""
        # 基于文件内容哈希生成缓存文件名
        file_hash = hashlib.md5(open(xyz_file, 'rb').read()).hexdigest()[:16]
        cache_name = f"{Path(xyz_file).stem}_{file_hash}.pkl"
        return Path(self.cache_dir) / cache_name
    
    def get_type_map(self) -> List[str]:
        """获取元素类型映射"""
        if self._type_map is None:
            if self.frames:
                symbols = self.frames[0].atoms.get_chemical_symbols()
                self._type_map = sorted(set(symbols))
            else:
                self._type_map = []
        return self._type_map
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        if not self.frames:
            return {}
        
        energies = [f.energy for f in self.frames]
        n_atoms = [len(f.atoms) for f in self.frames]
        
        # 力统计
        all_forces = np.concatenate([f.forces.flatten() for f in self.frames])
        
        stats = {
            'n_frames': len(self.frames),
            'n_atoms_mean': np.mean(n_atoms),
            'n_atoms_std': np.std(n_atoms),
            'elements': self.get_type_map(),
            'energy': {
                'total_mean': float(np.mean(energies)),
                'total_std': float(np.std(energies)),
                'per_atom_mean': float(np.mean(energies) / np.mean(n_atoms)),
                'per_atom_std': float(np.std(energies) / np.mean(n_atoms)),
                'min': float(np.min(energies)),
                'max': float(np.max(energies)),
            },
            'forces': {
                'mean': float(np.mean(all_forces)),
                'std': float(np.std(all_forces)),
                'max_abs': float(np.max(np.abs(all_forces))),
            }
        }
        
        return stats
    
    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1,
              stratify_by: Optional[str] = None, seed: int = 42) -> Tuple['NEPDataset', 'NEPDataset', 'NEPDataset']:
        """
        分割数据集
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            stratify_by: 分层依据 (元素组成、结构类型等)
            seed: 随机种子
        """
        np.random.seed(seed)
        
        n = len(self.frames)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        indices = np.random.permutation(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        train_set = NEPDataset([self.frames[i] for i in train_idx])
        val_set = NEPDataset([self.frames[i] for i in val_idx])
        test_set = NEPDataset([self.frames[i] for i in test_idx])
        
        return train_set, val_set, test_set
    
    def filter_by_energy(self, min_e: Optional[float] = None,
                        max_e: Optional[float] = None,
                        per_atom: bool = True):
        """按能量过滤"""
        filtered = []
        for frame in self.frames:
            e = frame.energy
            if per_atom:
                e /= len(frame.atoms)
            
            if (min_e is None or e >= min_e) and (max_e is None or e <= max_e):
                filtered.append(frame)
        
        self.frames = filtered
        logger.info(f"Filtered to {len(self.frames)} frames by energy")
    
    def filter_by_force(self, max_force: float):
        """按最大力过滤"""
        filtered = []
        for frame in self.frames:
            max_f = np.max(np.abs(frame.forces))
            if max_f <= max_force:
                filtered.append(frame)
        
        self.frames = filtered
        logger.info(f"Filtered to {len(self.frames)} frames by force")
    
    def save_xyz(self, output_path: str):
        """保存为XYZ格式"""
        with open(output_path, 'w') as f:
            for frame in self.frames:
                atoms = frame.atoms
                n_atoms = len(atoms)
                
                # 晶格
                cell = atoms.get_cell()
                lattice_str = " ".join([f"{x:.10f}" for x in cell.flatten()])
                
                # 能量
                energy = frame.energy
                
                # 应力/位力
                if frame.virial is not None:
                    virial_flat = [
                        frame.virial[0, 0], frame.virial[1, 1], frame.virial[2, 2],
                        frame.virial[1, 2], frame.virial[0, 2], frame.virial[0, 1]
                    ]
                    virial_str = " ".join([f"{v:.10f}" for v in virial_flat])
                    property_str = f"energy={energy:.10f} virial=\"{virial_str}\""
                else:
                    property_str = f"energy={energy:.10f}"
                
                # 写入
                f.write(f"{n_atoms}\n")
                f.write(f"Lattice=\"{lattice_str}\" Properties=species:S:1:pos:R:3:forces:R:3 {property_str}\n")
                
                symbols = atoms.get_chemical_symbols()
                positions = atoms.get_positions()
                forces = frame.forces
                
                for symbol, pos, force in zip(symbols, positions, forces):
                    f.write(f"{symbol:>3} {pos[0]:15.8f} {pos[1]:15.8f} {pos[2]:15.8f} "
                           f"{force[0]:15.8f} {force[1]:15.8f} {force[2]:15.8f}\n")
        
        logger.info(f"Saved {len(self.frames)} frames to {output_path}")


class NEPDataLoader:
    """
    NEP数据加载器
    
    支持批量加载和多种数据格式
    """
    
    def __init__(self, dataset: NEPDataset, batch_size: int = 32,
                 shuffle: bool = True, num_workers: int = 0,
                 prefetch_factor: int = 2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
    
    def __iter__(self) -> Iterator[List[NEPFrame]]:
        """迭代器"""
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            yield [self.dataset[idx] for idx in batch_idx]
    
    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class DataAugmenter:
    """
    数据增强器
    
    通过旋转、平移、添加噪声等方式增加数据多样性
    """
    
    def __init__(self, rotation: bool = True, translation: bool = False,
                 noise: bool = True, noise_std: float = 0.01,
                 reflection: bool = False):
        self.rotation = rotation
        self.translation = translation
        self.noise = noise
        self.noise_std = noise_std
        self.reflection = reflection
    
    def augment(self, frame: NEPFrame, n_augment: int = 1) -> List[NEPFrame]:
        """
        对单个frame进行数据增强
        
        Args:
            frame: 原始frame
            n_augment: 增强数量
            
        Returns:
            增强后的frames列表
        """
        augmented = [frame]  # 包含原始frame
        
        for _ in range(n_augment):
            new_atoms = frame.atoms.copy()
            new_forces = frame.forces.copy()
            
            # 旋转
            if self.rotation:
                rot = self._random_rotation_matrix()
                new_atoms.set_positions(new_atoms.get_positions() @ rot.T)
                new_forces = new_forces @ rot.T
                new_atoms.set_cell(new_atoms.get_cell() @ rot.T, scale_atoms=False)
            
            # 平移
            if self.translation:
                shift = np.random.randn(3) * 0.1
                new_atoms.set_positions(new_atoms.get_positions() + shift)
            
            # 添加噪声
            if self.noise:
                noise = np.random.randn(*new_atoms.get_positions().shape) * self.noise_std
                new_atoms.set_positions(new_atoms.get_positions() + noise)
            
            # 反射
            if self.reflection and np.random.random() < 0.5:
                axis = np.random.randint(3)
                pos = new_atoms.get_positions()
                pos[:, axis] *= -1
                new_atoms.set_positions(pos)
                new_forces[:, axis] *= -1
            
            augmented.append(NEPFrame(
                atoms=new_atoms,
                energy=frame.energy,
                forces=new_forces,
                stress=frame.stress.copy() if frame.stress is not None else None,
                virial=frame.virial.copy() if frame.virial is not None else None,
                weight=frame.weight * 0.5,  # 增强数据的权重较低
                metadata=frame.metadata.copy() if frame.metadata else {}
            ))
        
        return augmented
    
    def _random_rotation_matrix(self) -> np.ndarray:
        """生成随机旋转矩阵"""
        # 使用Rodrigues旋转公式
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle = np.random.uniform(0, 2 * np.pi)
        
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R
    
    def augment_dataset(self, dataset: NEPDataset, n_augment: int = 1,
                       augment_ratio: float = 0.3) -> NEPDataset:
        """
        对整个数据集进行增强
        
        Args:
            dataset: 原始数据集
            n_augment: 每个样本增强数量
            augment_ratio: 增强数据占总数据的比例
            
        Returns:
            增强后的数据集
        """
        n_to_augment = int(len(dataset) * augment_ratio)
        indices = np.random.choice(len(dataset), n_to_augment, replace=False)
        
        all_frames = list(dataset.frames)
        
        for idx in indices:
            augmented = self.augment(dataset[idx], n_augment=n_augment)
            all_frames.extend(augmented[1:])  # 跳过原始frame
        
        return NEPDataset(frames=all_frames)


class NEPDataPreparer:
    """
    NEP数据准备器 (增强版)
    
    从多种数据源准备NEP训练数据
    """
    
    def __init__(self, type_map: List[str]):
        self.type_map = type_map
    
    def from_vasp_outcars(self, outcar_files: List[str]) -> NEPDataset:
        """从VASP OUTCAR文件加载"""
        frames = []
        
        for outcar in outcar_files:
            logger.info(f"Loading from {outcar}")
            try:
                atoms_list = read_vasp_out(outcar, index=':')
                if not isinstance(atoms_list, list):
                    atoms_list = [atoms_list]
                
                for atoms in atoms_list:
                    if atoms.calc is not None:
                        frame = NEPFrame(
                            atoms=atoms.copy(),
                            energy=atoms.get_potential_energy(),
                            forces=atoms.get_forces(),
                            stress=atoms.get_stress(voigt=True) if hasattr(atoms, 'get_stress') else None
                        )
                        frames.append(frame)
                
                logger.info(f"Loaded {len(atoms_list)} frames from {outcar}")
            except Exception as e:
                logger.error(f"Failed to load {outcar}: {e}")
        
        return NEPDataset(frames=frames)
    
    def from_deepmd(self, deepmd_dir: str) -> NEPDataset:
        """从DeepMD格式加载"""
        try:
            import dpdata
            
            logger.info(f"Loading DeepMD data from {deepmd_dir}")
            system = dpdata.LabeledSystem(deepmd_dir, fmt='deepmd/npy')
            
            frames = []
            for i in range(len(system)):
                atoms = Atoms(
                    numbers=system['atom_types'],
                    positions=system['coords'][i],
                    cell=system['cells'][i],
                    pbc=True
                )
                frame = NEPFrame(
                    atoms=atoms,
                    energy=system['energies'][i],
                    forces=system['forces'][i]
                )
                frames.append(frame)
            
            logger.info(f"Loaded {len(frames)} frames from DeepMD")
            return NEPDataset(frames=frames)
            
        except ImportError:
            logger.error("dpdata not installed")
            raise
    
    def from_ase_db(self, db_path: str, selection: Optional[str] = None) -> NEPDataset:
        """从ASE数据库加载"""
        from ase.db import connect
        
        frames = []
        db = connect(db_path)
        
        for row in db.select(selection):
            atoms = row.toatoms()
            
            # 获取能量和力
            energy = row.get('energy', None)
            forces = row.get('forces', None)
            
            if energy is not None:
                frame = NEPFrame(
                    atoms=atoms,
                    energy=energy,
                    forces=np.array(forces) if forces is not None else np.zeros((len(atoms), 3))
                )
                frames.append(frame)
        
        logger.info(f"Loaded {len(frames)} frames from ASE database")
        return NEPDataset(frames=frames)
