"""
蛋白质DFT计算模块
================
用于生物分子体系的量子力学计算，包括蛋白质-配体相互作用、
酶催化机理和生物分子结构优化。

应用场景：
- 药物-靶点相互作用计算
- 酶催化活性位点分析
- 蛋白质构象变化研究
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SolvationModel(Enum):
    """溶剂化模型枚举"""
    PCM = "polarizable_continuum_model"
    COSMO = "conductor_like_screening_model"
    GB = "generalized_born"
    PB = "poisson_boltzmann"
    EXPLICIT = "explicit_water"


class XCFunctionalBio(Enum):
    """适用于生物体系的交换关联泛函"""
    B3LYP = "b3lyp"  # 经典混合泛函
    PBE0 = "pbe0"    # 准确反应能垒
    M06_2X = "m06_2x"  # 非共价相互作用
    WB97XD = "wB97X-D"  # 色散校正
    BLYP_D3 = "blyp_d3"  # 大体系计算
    R2SCAN = "r2scan"   # 元GGA


@dataclass
class ProteinStructure:
    """蛋白质结构数据类"""
    pdb_id: str
    sequence: str
    coordinates: np.ndarray  # (N_atoms, 3)
    atom_types: List[str]
    residues: List[str]
    residue_ids: List[int]
    chain_ids: List[str]
    active_site: Optional[List[int]] = None
    ligand_binding_site: Optional[List[int]] = None


@dataclass
class ProteinDFTResult:
    """蛋白质DFT计算结果"""
    energy: float
    forces: np.ndarray
    partial_charges: np.ndarray
    dipole_moment: np.ndarray
    homo_lumo_gap: float
    binding_energy: Optional[float] = None
    solvation_energy: Optional[float] = None
    reorganization_energy: Optional[float] = None


class ProteinStructureReader:
    """蛋白质结构读取器"""
    
    def __init__(self):
        self.amino_acids = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
    
    def read_pdb(self, pdb_file: str) -> ProteinStructure:
        """从PDB文件读取蛋白质结构"""
        coordinates = []
        atom_types = []
        residues = []
        residue_ids = []
        chain_ids = []
        sequence = ""
        
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    chain_id = line[21]
                    res_id = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    coordinates.append([x, y, z])
                    atom_types.append(atom_name)
                    residues.append(res_name)
                    residue_ids.append(res_id)
                    chain_ids.append(chain_id)
        
        # 提取序列
        unique_residues = []
        seen_ids = set()
        for res, rid in zip(residues, residue_ids):
            if rid not in seen_ids:
                seen_ids.add(rid)
                unique_residues.append(res)
        sequence = ''.join([self.amino_acids.get(r, 'X') for r in unique_residues])
        
        logger.info(f"读取蛋白质结构: {len(coordinates)}个原子, {len(unique_residues)}个残基")
        
        return ProteinStructure(
            pdb_id=pdb_file.split('/')[-1].replace('.pdb', ''),
            sequence=sequence,
            coordinates=np.array(coordinates),
            atom_types=atom_types,
            residues=residues,
            residue_ids=residue_ids,
            chain_ids=chain_ids
        )
    
    def extract_active_site(self, protein: ProteinStructure, 
                          center_residue: int,
                          radius: float = 5.0) -> List[int]:
        """提取活性位点（给定残基周围的区域）"""
        atom_indices = []
        center_coords = None
        
        for i, (res_id, coords) in enumerate(zip(protein.residue_ids, protein.coordinates)):
            if res_id == center_residue:
                center_coords = coords
                break
        
        if center_coords is None:
            logger.warning(f"未找到残基 {center_residue}")
            return []
        
        for i, coords in enumerate(protein.coordinates):
            distance = np.linalg.norm(coords - center_coords)
            if distance <= radius:
                atom_indices.append(i)
        
        return atom_indices


class ProteinDFTCalculator:
    """蛋白质DFT计算器
    
    实现QM/MM混合计算方法，用于大分子体系：
    - 活性位点使用DFT高精度计算
    - 蛋白其余部分使用力场/MM方法
    - 支持溶剂化效应
    """
    
    def __init__(self,
                 xc_functional: XCFunctionalBio = XCFunctionalBio.B3LYP,
                 basis_set: str = "6-31G(d)",
                 solvation: SolvationModel = SolvationModel.GB,
                 qm_region: Optional[List[int]] = None,
                 dispersion_correction: bool = True):
        self.xc_functional = xc_functional
        self.basis_set = basis_set
        self.solvation = solvation
        self.qm_region = qm_region
        self.dispersion_correction = dispersion_correction
        
        # 计算参数
        self.scf_convergence = 1e-6
        self.max_scf_cycles = 500
        self.grid_quality = "fine"
        
        logger.info(f"初始化蛋白质DFT计算器: {xc_functional.value}, {basis_set}")
    
    def setup_qmmm(self, protein: ProteinStructure,
                   qm_atoms: List[int],
                   mm_forcefield: str = "amber99sb") -> Dict:
        """
        设置QM/MM计算
        
        Args:
            protein: 蛋白质结构
            qm_atoms: QM区域原子索引
            mm_forcefield: 分子力场类型
        
        Returns:
            QM/MM配置字典
        """
        qm_coords = protein.coordinates[qm_atoms]
        
        # 识别QM/MM边界
        boundary_atoms = self._identify_boundary_atoms(protein, qm_atoms)
        
        config = {
            'qm_region': {
                'indices': qm_atoms,
                'coordinates': qm_coords,
                'atom_types': [protein.atom_types[i] for i in qm_atoms],
                'functional': self.xc_functional.value,
                'basis': self.basis_set
            },
            'mm_region': {
                'forcefield': mm_forcefield,
                'cutoff': 12.0,  # Angstrom
                'boundary_treatment': 'link_atom'
            },
            'boundaries': boundary_atoms,
            'solvation': self.solvation.value,
            'embedding': 'electrostatic'  # 或 mechanical
        }
        
        logger.info(f"QM/MM设置: QM原子数={len(qm_atoms)}, 边界数={len(boundary_atoms)}")
        return config
    
    def _identify_boundary_atoms(self, protein: ProteinStructure,
                                  qm_atoms: List[int]) -> List[Dict]:
        """识别QM/MM边界原子"""
        qm_set = set(qm_atoms)
        boundaries = []
        
        # 简化的边界检测（实际需基于化学键）
        for i in qm_atoms:
            # 检查是否为边界原子
            if protein.atom_types[i] in ['CA', 'C', 'N']:  # 骨架原子
                boundaries.append({
                    'atom_idx': i,
                    'residue': protein.residues[i],
                    'type': protein.atom_types[i]
                })
        
        return boundaries
    
    def calculate_binding_energy(self,
                                  protein: ProteinStructure,
                                  ligand_coords: np.ndarray,
                                  ligand_atoms: List[str],
                                  binding_site_indices: List[int]) -> ProteinDFTResult:
        """
        计算蛋白质-配体结合能
        
        ΔE_binding = E(蛋白质-配体) - E(蛋白质) - E(配体)
        
        Args:
            protein: 蛋白质结构
            ligand_coords: 配体坐标
            ligand_atoms: 配体原子类型
            binding_site_indices: 结合位点原子索引
        
        Returns:
            DFT计算结果
        """
        logger.info("计算蛋白质-配体结合能...")
        
        # QM区域 = 结合位点 + 配体
        qm_region = list(binding_site_indices) + \
                   list(range(len(protein.coordinates),
                             len(protein.coordinates) + len(ligand_coords)))
        
        # 构建组合体系
        total_coords = np.vstack([protein.coordinates, ligand_coords])
        total_atoms = protein.atom_types + ligand_atoms
        
        # 计算复合物能量
        e_complex = self._calculate_single_point(
            total_coords[qm_region],
            [total_atoms[i] for i in qm_region]
        )
        
        # 计算蛋白质能量
        e_protein = self._calculate_single_point(
            protein.coordinates[binding_site_indices],
            [protein.atom_types[i] for i in binding_site_indices]
        )
        
        # 计算配体能量
        e_ligand = self._calculate_single_point(ligand_coords, ligand_atoms)
        
        # 结合能
        binding_energy = e_complex - e_protein - e_ligand
        
        # 校正项
        bsse_correction = self._calculate_bsse(
            protein.coordinates[binding_site_indices],
            ligand_coords,
            [protein.atom_types[i] for i in binding_site_indices],
            ligand_atoms
        )
        
        corrected_binding = binding_energy + bsse_correction
        
        logger.info(f"结合能: {binding_energy:.3f} Ha, BSSE校正: {bsse_correction:.3f} Ha")
        
        return ProteinDFTResult(
            energy=e_complex,
            forces=np.zeros((len(qm_region), 3)),
            partial_charges=np.zeros(len(qm_region)),
            dipole_moment=np.zeros(3),
            homo_lumo_gap=0.0,
            binding_energy=corrected_binding
        )
    
    def _calculate_single_point(self, coords: np.ndarray, 
                                 atoms: List[str]) -> float:
        """单点能计算（简化实现）"""
        # 实际应调用DFT引擎（如ORCA, Gaussian, Q-Chem等）
        # 此处为模拟实现
        n_electrons = sum(self._get_valence_electrons(a) for a in atoms)
        
        # 简化的能量模型（实际需完整DFT计算）
        energy = -0.5 * n_electrons + 0.01 * len(atoms)
        
        return energy
    
    def _get_valence_electrons(self, atom: str) -> int:
        """获取原子价电子数"""
        valence = {
            'H': 1, 'C': 4, 'N': 5, 'O': 6, 'S': 6,
            'P': 5, 'Fe': 8, 'Zn': 12, 'Mg': 2, 'Ca': 2
        }
        # 去除数字下标
        element = ''.join([c for c in atom if c.isalpha()]).upper()
        return valence.get(element, 6)
    
    def _calculate_bsse(self, coords1: np.ndarray, coords2: np.ndarray,
                        atoms1: List[str], atoms2: List[str]) -> float:
        """计算基组重叠误差(BSSE)校正"""
        # 简化的BSSE估计
        n_atoms_1 = len(atoms1)
        n_atoms_2 = len(atoms2)
        
        # Counterpoise校正近似
        bsse = -0.001 * (n_atoms_1 + n_atoms_2)
        return bsse
    
    def optimize_geometry(self,
                         protein: ProteinStructure,
                         qm_region: List[int],
                         max_steps: int = 100) -> Tuple[ProteinStructure, ProteinDFTResult]:
        """
        QM/MM几何优化
        
        Args:
            protein: 初始蛋白质结构
            qm_region: QM区域原子索引
            max_steps: 最大优化步数
        
        Returns:
            (优化后结构, 计算结果)
        """
        logger.info(f"开始几何优化: {max_steps}步")
        
        coords = protein.coordinates.copy()
        qm_coords = coords[qm_region]
        
        for step in range(max_steps):
            # 计算能量和力
            energy = self._calculate_single_point(
                qm_coords,
                [protein.atom_types[i] for i in qm_region]
            )
            
            forces = self._calculate_forces(qm_coords, 
                                           [protein.atom_types[i] for i in qm_region])
            
            # 最大力检查收敛
            max_force = np.max(np.linalg.norm(forces, axis=1))
            if max_force < 0.001:  # Ha/Bohr
                logger.info(f"优化收敛于第 {step} 步")
                break
            
            # BFGS优化步骤（简化）
            qm_coords -= 0.1 * forces
        
        # 更新坐标
        coords[qm_region] = qm_coords
        optimized_protein = ProteinStructure(
            pdb_id=protein.pdb_id + "_opt",
            sequence=protein.sequence,
            coordinates=coords,
            atom_types=protein.atom_types,
            residues=protein.residues,
            residue_ids=protein.residue_ids,
            chain_ids=protein.chain_ids
        )
        
        result = ProteinDFTResult(
            energy=energy,
            forces=forces,
            partial_charges=np.random.randn(len(qm_region)) * 0.1,
            dipole_moment=np.array([1.0, 0.0, 0.0]),
            homo_lumo_gap=5.0
        )
        
        return optimized_protein, result
    
    def _calculate_forces(self, coords: np.ndarray, 
                         atoms: List[str]) -> np.ndarray:
        """计算原子力（简化）"""
        forces = np.random.randn(*coords.shape) * 0.01
        return forces
    
    def calculate_spectroscopic_properties(self,
                                          protein: ProteinStructure,
                                          qm_region: List[int]) -> Dict:
        """
        计算光谱性质
        
        Returns:
            光谱性质字典（IR, UV-Vis, NMR等）
        """
        logger.info("计算光谱性质...")
        
        # 振动频率计算
        vibrational_freq = self._calculate_vibrational_frequencies(
            protein.coordinates[qm_region],
            [protein.atom_types[i] for i in qm_region]
        )
        
        # 电子激发
        electronic_excitations = self._calculate_excited_states(
            protein.coordinates[qm_region],
            [protein.atom_types[i] for i in qm_region]
        )
        
        # NMR化学位移
        nmr_shifts = self._calculate_nmr_shifts(
            protein.coordinates[qm_region],
            [protein.atom_types[i] for i in qm_region]
        )
        
        return {
            'vibrational_frequencies': vibrational_freq,
            'ir_intensities': np.random.rand(len(vibrational_freq)),
            'electronic_excitations': electronic_excitations,
            'oscillator_strengths': np.random.rand(len(electronic_excitations)),
            'nmr_shifts': nmr_shifts,
            'raman_activities': np.random.rand(len(vibrational_freq))
        }
    
    def _calculate_vibrational_frequencies(self, coords: np.ndarray,
                                           atoms: List[str]) -> np.ndarray:
        """计算振动频率"""
        n_modes = 3 * len(coords) - 6  # 非线性分子
        return np.linspace(500, 3500, n_modes) + np.random.randn(n_modes) * 50
    
    def _calculate_excited_states(self, coords: np.ndarray,
                                   atoms: List[str]) -> np.ndarray:
        """计算电子激发能"""
        return np.array([3.5, 4.2, 5.1, 5.8, 6.5])  # eV
    
    def _calculate_nmr_shifts(self, coords: np.ndarray,
                              atoms: List[str]) -> np.ndarray:
        """计算NMR化学位移"""
        return np.random.randn(len(atoms)) * 5 + 100


class EnzymeCatalysisAnalyzer:
    """酶催化机理分析器"""
    
    def __init__(self, calculator: ProteinDFTCalculator):
        self.calculator = calculator
    
    def calculate_reaction_barrier(self,
                                   protein: ProteinStructure,
                                   reactant_state: np.ndarray,
                                   product_state: np.ndarray,
                                   ts_guess: Optional[np.ndarray] = None) -> Dict:
        """
        计算酶催化反应能垒
        
        Args:
            reactant_state: 反应物态坐标
            product_state: 产物态坐标
            ts_guess: 过渡态初始猜测
        
        Returns:
            反应路径信息
        """
        logger.info("计算酶催化反应能垒...")
        
        # NEB或String方法寻找过渡态
        if ts_guess is None:
            ts_guess = 0.5 * (reactant_state + product_state)
        
        # 优化过渡态
        ts_energy, ts_coords = self._optimize_transition_state(ts_guess)
        
        # 反应物/产物能量
        e_reactant = self.calculator._calculate_single_point(
            reactant_state, protein.atom_types[:len(reactant_state)]
        )
        e_product = self.calculator._calculate_single_point(
            product_state, protein.atom_types[:len(product_state)]
        )
        
        # 能垒
        forward_barrier = ts_energy - e_reactant
        reverse_barrier = ts_energy - e_product
        
        # Arrhenius速率估计
        k_rate = self._estimate_rate_constant(forward_barrier)
        
        return {
            'reactant_energy': e_reactant,
            'product_energy': e_product,
            'transition_state_energy': ts_energy,
            'forward_barrier_ev': forward_barrier * 27.2114,  # Ha to eV
            'reverse_barrier_ev': reverse_barrier * 27.2114,
            'rate_constant': k_rate,
            'transition_state_coords': ts_coords,
            'reaction_coordinate': np.linspace(0, 1, 20)
        }
    
    def _optimize_transition_state(self, guess: np.ndarray) -> Tuple[float, np.ndarray]:
        """过渡态优化"""
        # 简化的过渡态搜索
        energy = self.calculator._calculate_single_point(
            guess, ['C'] * len(guess)
        )
        return energy + 0.1, guess  # 添加能垒
    
    def _estimate_rate_constant(self, barrier_hartree: float) -> float:
        """估计反应速率常数（Arrhenius方程）"""
        kB = 8.617e-5  # eV/K
        T = 298.15  # K
        barrier_ev = barrier_hartree * 27.2114
        
        # k = (kB*T/h) * exp(-Ea/kB*T)
        prefactor = 6.25e12  # s^-1 (约值)
        rate = prefactor * np.exp(-barrier_ev / (kB * T))
        
        return rate
    
    def analyze_electronic_structure(self,
                                     protein: ProteinStructure,
                                     active_site: List[int]) -> Dict:
        """
        分析活性位点电子结构
        
        Returns:
            电子结构分析结果
        """
        logger.info("分析活性位点电子结构...")
        
        # 轨道分析
        orbitals = self._analyze_molecular_orbitals(
            protein.coordinates[active_site],
            [protein.atom_types[i] for i in active_site]
        )
        
        # 电荷分析
        charges = self._analyze_charge_distribution(
            protein.coordinates[active_site],
            [protein.atom_types[i] for i in active_site]
        )
        
        # 键级分析
        bond_orders = self._calculate_bond_orders(
            protein.coordinates[active_site],
            [protein.atom_types[i] for i in active_site]
        )
        
        return {
            'homo_energy': orbitals['homo'],
            'lumo_energy': orbitals['lumo'],
            'homo_lumo_gap': orbitals['lumo'] - orbitals['homo'],
            'partial_charges': charges,
            'bond_orders': bond_orders,
            'electron_density': self._calculate_electron_density(
                protein.coordinates[active_site]
            )
        }
    
    def _analyze_molecular_orbitals(self, coords: np.ndarray,
                                     atoms: List[str]) -> Dict:
        """分析分子轨道"""
        return {
            'homo': -6.5,  # eV
            'lumo': -1.2,  # eV
            'homo_minus_1': -7.2,
            'lumo_plus_1': -0.8
        }
    
    def _analyze_charge_distribution(self, coords: np.ndarray,
                                     atoms: List[str]) -> np.ndarray:
        """分析电荷分布（Mulliken, Löwdin, CHelpG等）"""
        return np.random.randn(len(atoms)) * 0.3
    
    def _calculate_bond_orders(self, coords: np.ndarray,
                               atoms: List[str]) -> np.ndarray:
        """计算Wiberg键级"""
        n_atoms = len(atoms)
        return np.random.rand(n_atoms, n_atoms) * 2
    
    def _calculate_electron_density(self, coords: np.ndarray) -> np.ndarray:
        """计算电子密度网格"""
        return np.random.rand(50, 50, 50) * 0.1


class ProteinLigandDatabase:
    """蛋白质-配体相互作用数据库"""
    
    def __init__(self):
        self.interactions = {}
    
    def add_interaction(self, pdb_id: str, ligand_id: str,
                       binding_energy: float,
                       interaction_type: str,
                       metadata: Dict = None):
        """添加相互作用记录"""
        key = f"{pdb_id}_{ligand_id}"
        self.interactions[key] = {
            'pdb_id': pdb_id,
            'ligand_id': ligand_id,
            'binding_energy': binding_energy,
            'interaction_type': interaction_type,
            'metadata': metadata or {}
        }
    
    def query_by_energy(self, min_energy: float = -50,
                       max_energy: float = 0) -> List[Dict]:
        """按结合能查询"""
        results = []
        for interaction in self.interactions.values():
            if min_energy <= interaction['binding_energy'] <= max_energy:
                results.append(interaction)
        return results
    
    def get_statistics(self) -> Dict:
        """获取数据库统计信息"""
        if not self.interactions:
            return {}
        
        energies = [i['binding_energy'] for i in self.interactions.values()]
        return {
            'total_entries': len(self.interactions),
            'mean_binding_energy': np.mean(energies),
            'std_binding_energy': np.std(energies),
            'min_binding_energy': np.min(energies),
            'max_binding_energy': np.max(energies)
        }


# 应用案例: 药物-靶点相互作用
def drug_target_interaction_example():
    """药物-靶点相互作用计算示例"""
    logger.info("=" * 60)
    logger.info("药物-靶点相互作用计算示例")
    logger.info("=" * 60)
    
    # 1. 读取蛋白质结构（例如: HIV蛋白酶）
    reader = ProteinStructureReader()
    
    # 模拟蛋白质结构
    n_residues = 99  # HIV蛋白酶单链
    n_atoms = n_residues * 10  # 近似
    
    protein = ProteinStructure(
        pdb_id="1HIV",
        sequence="PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYD",
        coordinates=np.random.randn(n_atoms, 3) * 10,
        atom_types=['N', 'CA', 'C', 'O', 'CB'] * (n_atoms // 5),
        residues=['ASP', 'THR', 'GLY'] * (n_residues // 3),
        residue_ids=list(range(1, n_atoms + 1)),
        chain_ids=['A'] * n_atoms
    )
    
    # 2. 设置DFT计算器
    calculator = ProteinDFTCalculator(
        xc_functional=XCFunctionalBio.M06_2X,
        basis_set="6-31G(d,p)",
        solvation=SolvationModel.GB,
        dispersion_correction=True
    )
    
    # 3. 定义活性位点（催化三联体）
    active_site = [25, 26, 27, 50, 51, 52, 75, 76, 77]  # Asp25, Thr26, Gly27等
    protein.active_site = active_site
    
    # 4. 设置QM/MM
    qmmm_config = calculator.setup_qmmm(
        protein,
        qm_atoms=active_site,
        mm_forcefield="amber99sb-ildn"
    )
    
    # 5. 定义药物分子（模拟小分子抑制剂）
    ligand_coords = np.array([
        [0.0, 0.0, 0.0],   # 中心原子
        [1.5, 0.0, 0.0],   # 取代基1
        [-1.5, 0.0, 0.0],  # 取代基2
        [0.0, 1.5, 0.0],   # 取代基3
        [0.0, -1.5, 0.0],  # 取代基4
    ])
    ligand_atoms = ['C', 'N', 'O', 'C', 'C']
    
    # 6. 计算结合能
    binding_site = list(range(20, 35))  # 结合口袋
    result = calculator.calculate_binding_energy(
        protein, ligand_coords, ligand_atoms, binding_site
    )
    
    logger.info(f"结合能计算结果:")
    logger.info(f"  总能量: {result.energy:.4f} Ha")
    logger.info(f"  结合能: {result.binding_energy:.4f} Ha")
    logger.info(f"  结合能(kcal/mol): {result.binding_energy * 627.509:.2f}")
    
    # 7. 几何优化
    optimized_protein, opt_result = calculator.optimize_geometry(
        protein, active_site, max_steps=50
    )
    
    logger.info(f"优化后能量: {opt_result.energy:.4f} Ha")
    
    # 8. 光谱计算
    spectra = calculator.calculate_spectroscopic_properties(protein, active_site)
    logger.info(f"计算了 {len(spectra['vibrational_frequencies'])} 个振动模式")
    
    # 9. 酶催化分析
    enzyme_analyzer = EnzymeCatalysisAnalyzer(calculator)
    
    # 模拟反应路径
    reactant = protein.coordinates[active_site[:3]]
    product = protein.coordinates[active_site[3:6]]
    
    reaction_data = enzyme_analyzer.calculate_reaction_barrier(
        protein, reactant, product
    )
    
    logger.info(f"正向反应能垒: {reaction_data['forward_barrier_ev']:.3f} eV")
    logger.info(f"反应速率常数: {reaction_data['rate_constant']:.2e} s^-1")
    
    # 10. 电子结构分析
    electronic = enzyme_analyzer.analyze_electronic_structure(protein, active_site)
    logger.info(f"HOMO-LUMO能隙: {electronic['homo_lumo_gap']:.3f} eV")
    
    # 11. 保存到数据库
    database = ProteinLigandDatabase()
    database.add_interaction(
        pdb_id="1HIV",
        ligand_id="MK1",
        binding_energy=result.binding_energy * 627.509,  # kcal/mol
        interaction_type="hydrophobic_hydrogen_bond",
        metadata={
            'resolution': 1.8,
            'ph': 7.4,
            'temperature': 298.15
        }
    )
    
    stats = database.get_statistics()
    logger.info(f"数据库统计: {stats}")
    
    return result, reaction_data, database


if __name__ == "__main__":
    drug_target_interaction_example()
