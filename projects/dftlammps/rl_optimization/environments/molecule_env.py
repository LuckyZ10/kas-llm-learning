"""
Molecular Environments for RL
==============================

实现分子生成环境:
- 分子图环境
- SMILES环境
- 基于RDKit的化学约束
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class MoleculeEnvConfig:
    """分子环境配置"""
    max_atoms: int = 50
    atom_types: List[str] = None
    bond_types: List[str] = None
    allow_aromatic: bool = True
    allow_rings: bool = True
    sanitize: bool = True
    
    def __post_init__(self):
        if self.atom_types is None:
            self.atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P', 'H']
        if self.bond_types is None:
            self.bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE']


class MoleculeEnvironment(ABC):
    """
    分子生成环境基类
    
    抽象接口，定义分子生成任务的基本方法
    """
    
    def __init__(self, config: Optional[MoleculeEnvConfig] = None):
        self.config = config or MoleculeEnvConfig()
        self.state = None
        self.step_count = 0
        
    @abstractmethod
    def reset(self) -> np.ndarray:
        """重置环境"""
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        pass
    
    @abstractmethod
    def get_valid_actions(self) -> List[int]:
        """获取有效动作"""
        pass
    
    @abstractmethod
    def get_sample(self) -> Dict[str, Any]:
        """获取当前样本"""
        pass
    
    @abstractmethod
    def compute_reward(self) -> float:
        """计算奖励"""
        pass


class MolecularGraphEnv(MoleculeEnvironment):
    """
    分子图生成环境
    
    通过逐步添加原子和键来构建分子图
    """
    
    def __init__(self, config: Optional[MoleculeEnvConfig] = None):
        super().__init__(config)
        
        # 动作空间定义
        self.atom_add_actions = list(range(len(self.config.atom_types)))
        self.bond_add_actions = list(range(
            len(self.config.atom_types),
            len(self.config.atom_types) + len(self.config.bond_types)
        ))
        self.terminate_action = len(self.config.atom_types) + len(self.config.bond_types)
        
        self.action_dim = self.terminate_action + 1
        
        # 分子状态
        self.atoms = []  # 原子类型列表
        self.bonds = []  # 键列表 [(atom_idx1, atom_idx2, bond_type)]
        self.atom_positions = []  # 3D位置
        
        # 当前焦点原子 (用于添加键)
        self.focus_atom = None
    
    def reset(self) -> np.ndarray:
        """重置环境 - 开始新的分子"""
        self.atoms = []
        self.bonds = []
        self.atom_positions = []
        self.focus_atom = None
        self.step_count = 0
        
        return self._get_state_vector()
    
    def _get_state_vector(self) -> np.ndarray:
        """将分子状态转换为向量表示"""
        # 简单编码: 原子类型one-hot + 统计特征
        max_atoms = self.config.max_atoms
        num_atom_types = len(self.config.atom_types)
        
        # 原子类型矩阵 [max_atoms, num_atom_types]
        atom_features = np.zeros((max_atoms, num_atom_types))
        for i, atom in enumerate(self.atoms):
            if i < max_atoms and atom in self.config.atom_types:
                idx = self.config.atom_types.index(atom)
                atom_features[i, idx] = 1
        
        # 统计特征
        stats = np.array([
            len(self.atoms) / max_atoms,
            len(self.bonds) / max(1, len(self.atoms) * 3),
            1.0 if self.focus_atom is not None else 0.0,
            self.step_count / self.config.max_atoms,
        ])
        
        # 展平
        state = np.concatenate([
            atom_features.flatten(),
            stats
        ])
        
        # 填充到固定大小
        target_dim = max_atoms * num_atom_types + 100
        if len(state) < target_dim:
            state = np.pad(state, (0, target_dim - len(state)))
        
        return state[:target_dim]
    
    def get_valid_actions(self) -> List[int]:
        """获取有效动作"""
        valid_actions = []
        
        # 如果没有原子，只能添加原子
        if len(self.atoms) == 0:
            return self.atom_add_actions
        
        # 如果达到最大原子数，不能添加原子
        if len(self.atoms) < self.config.max_atoms:
            valid_actions.extend(self.atom_add_actions)
        
        # 如果有焦点原子，可以添加键
        if self.focus_atom is not None and len(self.atoms) > 1:
            # 检查可以连接到哪些原子
            for i in range(len(self.atoms)):
                if i != self.focus_atom and not self._bond_exists(self.focus_atom, i):
                    valid_actions.extend(self.bond_add_actions)
                    break
        
        # 如果有至少2个原子，可以终止
        if len(self.atoms) >= 2:
            valid_actions.append(self.terminate_action)
        
        return list(set(valid_actions))
    
    def _bond_exists(self, atom1: int, atom2: int) -> bool:
        """检查键是否已存在"""
        for a1, a2, _ in self.bonds:
            if (a1 == atom1 and a2 == atom2) or (a1 == atom2 and a2 == atom1):
                return True
        return False
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        self.step_count += 1
        done = False
        info = {}
        
        # 添加原子
        if action in self.atom_add_actions:
            atom_idx = action
            atom_type = self.config.atom_types[atom_idx]
            self.atoms.append(atom_type)
            self.atom_positions.append(self._random_position())
            
            # 新原子成为焦点
            self.focus_atom = len(self.atoms) - 1
            
        # 添加键
        elif action in self.bond_add_actions:
            bond_idx = action - len(self.config.atom_types)
            bond_type = self.config.bond_types[bond_idx]
            
            # 连接到最近的非连接原子
            if self.focus_atom is not None:
                target = self._find_nearest_atom(self.focus_atom)
                if target is not None and not self._bond_exists(self.focus_atom, target):
                    self.bonds.append((self.focus_atom, target, bond_type))
                    # 切换焦点
                    self.focus_atom = target
        
        # 终止
        elif action == self.terminate_action:
            done = True
            info['molecule'] = self._get_molecule_dict()
        
        # 检查是否达到最大步数
        if self.step_count >= self.config.max_atoms * 2:
            done = True
        
        reward = self.compute_reward() if done else 0.0
        
        return self._get_state_vector(), reward, done, info
    
    def _random_position(self) -> np.ndarray:
        """生成随机3D位置"""
        return np.random.randn(3) * 0.5
    
    def _find_nearest_atom(self, atom_idx: int) -> Optional[int]:
        """找到最近的非连接原子"""
        if len(self.atoms) <= 1:
            return None
        
        pos = self.atom_positions[atom_idx]
        min_dist = float('inf')
        nearest = None
        
        for i, other_pos in enumerate(self.atom_positions):
            if i != atom_idx and not self._bond_exists(atom_idx, i):
                dist = np.linalg.norm(pos - other_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest = i
        
        return nearest
    
    def _get_molecule_dict(self) -> Dict[str, Any]:
        """获取分子字典表示"""
        return {
            'atoms': self.atoms.copy(),
            'bonds': self.bonds.copy(),
            'positions': [p.tolist() for p in self.atom_positions],
            'num_atoms': len(self.atoms),
            'num_bonds': len(self.bonds),
        }
    
    def compute_reward(self) -> float:
        """计算分子奖励"""
        if len(self.atoms) < 2:
            return 0.0
        
        reward = 0.0
        
        # 1. 大小奖励
        reward += min(len(self.atoms) / 10, 1.0)
        
        # 2. 连接性奖励
        if len(self.atoms) > 1:
            connectivity = len(self.bonds) / max(1, len(self.atoms) - 1)
            reward += min(connectivity, 1.0)
        
        # 3. 多样性奖励
        unique_atoms = len(set(self.atoms))
        reward += unique_atoms / len(self.config.atom_types)
        
        # 4. 化学有效性 (简化检查)
        if self._is_valid():
            reward += 1.0
        
        return reward
    
    def _is_valid(self) -> bool:
        """简单化学有效性检查"""
        # 检查每个原子的键数是否合理
        atom_bond_counts = [0] * len(self.atoms)
        for a1, a2, bond_type in self.bonds:
            bond_order = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, 'AROMATIC': 1.5}.get(bond_type, 1)
            atom_bond_counts[a1] += bond_order
            atom_bond_counts[a2] += bond_order
        
        # 简单化合价检查
        for i, (atom, count) in enumerate(zip(self.atoms, atom_bond_counts)):
            max_valence = {
                'H': 1, 'C': 4, 'N': 3, 'O': 2, 'S': 6,
                'F': 1, 'Cl': 1, 'Br': 1, 'P': 5
            }.get(atom, 4)
            
            if count > max_valence:
                return False
        
        return True
    
    def get_sample(self) -> Dict[str, Any]:
        """获取当前样本"""
        return self._get_molecule_dict()
    
    def to_smiles(self) -> Optional[str]:
        """转换为SMILES (如果RDKit可用)"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            # 创建可编辑分子
            mol = Chem.RWMol()
            atom_idx_map = {}
            
            # 添加原子
            for i, atom_type in enumerate(self.atoms):
                atom = Chem.Atom(atom_type)
                idx = mol.AddAtom(atom)
                atom_idx_map[i] = idx
            
            # 添加键
            bond_type_map = {
                'SINGLE': Chem.BondType.SINGLE,
                'DOUBLE': Chem.BondType.DOUBLE,
                'TRIPLE': Chem.BondType.TRIPLE,
                'AROMATIC': Chem.BondType.AROMATIC,
            }
            
            for a1, a2, bond_type in self.bonds:
                if a1 in atom_idx_map and a2 in atom_idx_map:
                    mol.AddBond(atom_idx_map[a1], atom_idx_map[a2], 
                               bond_type_map.get(bond_type, Chem.BondType.SINGLE))
            
            # 转换为分子并生成SMILES
            mol = mol.GetMol()
            if self.config.sanitize:
                Chem.SanitizeMol(mol)
            
            return Chem.MolToSmiles(mol)
        except:
            return None


class SMILESEnv(MoleculeEnvironment):
    """
    SMILES字符串生成环境
    
    通过逐步添加字符生成SMILES字符串
    """
    
    def __init__(self, config: Optional[MoleculeEnvConfig] = None):
        super().__init__(config)
        
        # SMILES字符集
        self.vocab = self._build_vocab()
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}
        
        self.max_length = 100
        self.current_smiles = ""
        
        # 动作空间
        self.action_dim = len(self.vocab)
        self.terminate_token = self.char_to_idx.get('[END]', len(self.vocab) - 1)
    
    def _build_vocab(self) -> List[str]:
        """构建SMILES字符集"""
        base_vocab = [
            'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
            'c', 'n', 'o', 's', 'p',
            '(', ')', '[', ']', '=', '#', '@', '@@',
            '+', '-', 'H', 
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '.', '/', '\\', '%', 
            '[START]', '[END]', '[PAD]'
        ]
        
        # 添加原子类型
        for atom in self.config.atom_types:
            if atom not in base_vocab:
                base_vocab.append(atom)
        
        return base_vocab
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_smiles = "[START]"
        self.step_count = 0
        return self._get_state_vector()
    
    def _get_state_vector(self) -> np.ndarray:
        """获取状态向量"""
        # 使用字符索引序列
        indices = [self.char_to_idx.get(c, self.char_to_idx['[PAD]']) 
                   for c in self.current_smiles]
        
        # 填充或截断
        if len(indices) < self.max_length:
            indices.extend([self.char_to_idx['[PAD]']] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]
        
        return np.array(indices, dtype=np.float32)
    
    def get_valid_actions(self) -> List[int]:
        """获取有效动作"""
        valid = list(range(len(self.vocab)))
        
        # 如果达到最大长度，只能终止
        if len(self.current_smiles) >= self.max_length:
            return [self.terminate_token]
        
        return valid
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        self.step_count += 1
        done = False
        info = {}
        
        char = self.idx_to_char[action]
        
        if char == '[END]':
            done = True
            info['smiles'] = self.current_smiles.replace('[START]', '')
        elif char != '[PAD]':
            self.current_smiles += char
        
        # 检查长度限制
        if len(self.current_smiles) >= self.max_length:
            done = True
            info['smiles'] = self.current_smiles.replace('[START]', '')
        
        reward = self.compute_reward() if done else 0.0
        
        return self._get_state_vector(), reward, done, info
    
    def compute_reward(self) -> float:
        """计算SMILES奖励"""
        smiles = self.current_smiles.replace('[START]', '')
        
        if not smiles:
            return 0.0
        
        reward = 0.0
        
        # 1. 长度奖励
        reward += min(len(smiles) / 20, 1.0)
        
        # 2. 化学有效性 (使用RDKit)
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                reward += 2.0  # 有效分子
                
                # 额外奖励
                reward += min(mol.GetNumAtoms() / 20, 1.0)
                
                # 多样性奖励
                fingerprints = set()
                for atom in mol.GetAtoms():
                    fingerprints.add(atom.GetSymbol())
                reward += len(fingerprints) / 10
        except:
            pass
        
        return reward
    
    def get_sample(self) -> Dict[str, Any]:
        """获取当前样本"""
        return {
            'smiles': self.current_smiles.replace('[START]', ''),
            'length': len(self.current_smiles),
        }


def demo():
    """演示分子环境"""
    print("=" * 60)
    print("Molecular Environments Demo")
    print("=" * 60)
    
    config = MoleculeEnvConfig(
        max_atoms=10,
        atom_types=['C', 'N', 'O', 'H']
    )
    
    # 1. 分子图环境
    print("\n1. Molecular Graph Environment")
    env = MolecularGraphEnv(config)
    
    state = env.reset()
    print(f"   Initial state shape: {state.shape}")
    print(f"   Valid actions: {env.get_valid_actions()[:5]}...")
    
    # 运行几个步骤
    total_reward = 0
    for i in range(10):
        valid_actions = env.get_valid_actions()
        action = np.random.choice(valid_actions)
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"   Episode finished at step {i+1}")
            print(f"   Molecule: {env.get_sample()}")
            break
    
    print(f"   Total reward: {total_reward:.2f}")
    
    # 2. SMILES环境
    print("\n2. SMILES Environment")
    smiles_env = SMILESEnv(config)
    
    state = smiles_env.reset()
    print(f"   Initial state shape: {state.shape}")
    print(f"   Vocab size: {len(smiles_env.vocab)}")
    
    # 运行几个步骤
    for i in range(20):
        valid_actions = smiles_env.get_valid_actions()
        action = np.random.choice(valid_actions)
        state, reward, done, info = smiles_env.step(action)
        
        if done:
            print(f"   Episode finished at step {i+1}")
            print(f"   Generated: {smiles_env.get_sample()}")
            break
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
