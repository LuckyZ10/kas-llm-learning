"""
DNA-材料相互作用模块
===================
模拟DNA分子与纳米材料表面的相互作用，用于：
- 基因递送载体设计
- 生物传感器开发
- 纳米毒理学研究
- DNA纳米技术

功能特性：
- DNA-纳米粒子相互作用
- DNA-石墨烯/碳纳米管相互作用
- 金纳米粒子-DNA复合物
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MaterialType(Enum):
    """材料类型枚举"""
    GOLD_NP = "gold_nanoparticle"
    GRAPHENE = "graphene"
    CNT = "carbon_nanotube"
    QUANTUM_DOT = "quantum_dot"
    POLYMER = "polymeric_nanoparticle"
    LIPOSOME = "liposome"
    SILICA = "silica_nanoparticle"
    TIO2 = "titanium_dioxide"


class DNABase(Enum):
    """DNA碱基枚举"""
    ADENINE = 'A'
    THYMINE = 'T'
    CYTOSINE = 'C'
    GUANINE = 'G'


@dataclass
class DNAStructure:
    """DNA结构数据类"""
    sequence: str
    coordinates: np.ndarray  # (N_bases, 3) - 碱基中心坐标
    backbone_coords: np.ndarray  # 骨架坐标
    is_double_strand: bool
    base_pairs: Optional[List[Tuple[int, int]]] = None
    grooves: Optional[Dict] = None  # 大沟小沟信息
    
    def __post_init__(self):
        if self.base_pairs is None and self.is_double_strand:
            self._infer_base_pairs()
    
    def _infer_base_pairs(self):
        """推断碱基配对"""
        pairs = []
        seq_len = len(self.sequence)
        for i, base in enumerate(self.sequence[:seq_len//2]):
            complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
            if self.sequence[seq_len - 1 - i] == complement.get(base):
                pairs.append((i, seq_len - 1 - i))
        self.base_pairs = pairs


@dataclass
class Nanoparticle:
    """纳米粒子数据类"""
    material: MaterialType
    radius: float  # nm
    coordinates: np.ndarray  # 中心坐标
    surface_atoms: np.ndarray  # 表面原子坐标
    charge: float  # 总电荷
    ligands: Optional[List[Dict]] = None  # 表面配体
    
    def get_surface_area(self) -> float:
        """计算表面积"""
        return 4 * np.pi * (self.radius ** 2)
    
    def get_volume(self) -> float:
        """计算体积"""
        return (4/3) * np.pi * (self.radius ** 3)


@dataclass
class DNAAdsorptionResult:
    """DNA吸附计算结果"""
    binding_energy: float  # eV
    adsorption_configuration: np.ndarray
    contact_area: float
    hydrogen_bonds: int
    stacking_interactions: int
    electrostatic_energy: float
    vdw_energy: float
    solvation_energy: float
    binding_free_energy: float
    residence_time: float  # ps


class DNAForceField:
    """DNA分子力场参数"""
    
    def __init__(self, forcefield_type: str = "amber"):
        self.forcefield_type = forcefield_type
        
        # Amber力场DNA参数
        self.bond_params = {
            ('P', 'O5'): (1000, 1.6),   # kcal/mol/A^2, A
            ('O5', "C5'"): (300, 1.44),
            ("C5'", "C4'"): (300, 1.52),
            ("C4'", "C3'"): (300, 1.52),
            ("C3'", "O3'"): (300, 1.43),
            ("O3'", 'P'): (1000, 1.6),
        }
        
        # 碱基配对能量 (kcal/mol)
        self.base_pairing = {
            ('A', 'T'): -13.0,
            ('T', 'A'): -13.0,
            ('C', 'G'): -24.0,
            ('G', 'C'): -24.0,
        }
        
        # 范德华参数 (epsilon kcal/mol, sigma A)
        self.vdw_params = {
            'P': (0.2, 3.74),
            'O': (0.12, 2.96),
            'N': (0.17, 3.25),
            'C': (0.11, 3.40),
            'S': (0.25, 3.60),
        }
    
    def get_base_pairing_energy(self, base1: str, base2: str) -> float:
        """获取碱基配对能量"""
        return self.base_pairing.get((base1, base2), 0.0)
    
    def calculate_vdw_energy(self, r: float, atom1: str, atom2: str) -> float:
        """计算范德华相互作用"""
        eps1, sig1 = self.vdw_params.get(atom1, (0.1, 3.0))
        eps2, sig2 = self.vdw_params.get(atom2, (0.1, 3.0))
        
        eps = np.sqrt(eps1 * eps2)
        sig = (sig1 + sig2) / 2
        
        # Lennard-Jones势
        sr6 = (sig / r) ** 6
        energy = 4 * eps * (sr6**2 - sr6)
        return energy


class DNANanoparticleCalculator:
    """DNA-纳米粒子相互作用计算器"""
    
    def __init__(self,
                 forcefield: DNAForceField = None,
                 solvent: str = "water",
                 temperature: float = 300.0,
                 ionic_strength: float = 0.1):
        self.forcefield = forcefield or DNAForceField()
        self.solvent = solvent
        self.temperature = temperature
        self.ionic_strength = ionic_strength
        self.kB = 8.617e-5  # eV/K
        
        logger.info(f"初始化DNA-纳米粒子计算器: T={temperature}K, I={ionic_strength}M")
    
    def calculate_adsorption_energy(self,
                                     dna: DNAStructure,
                                     nanoparticle: Nanoparticle,
                                     binding_mode: str = "end_on") -> DNAAdsorptionResult:
        """
        计算DNA在纳米粒子表面的吸附能
        
        Args:
            dna: DNA结构
            nanoparticle: 纳米粒子
            binding_mode: 结合模式 (end_on, side_on, groove_binding)
        
        Returns:
            吸附能计算结果
        """
        logger.info(f"计算DNA-{nanoparticle.material.value}吸附能, 模式: {binding_mode}")
        
        # 根据结合模式调整构型
        dna_config = self._position_dna_on_surface(
            dna, nanoparticle, binding_mode
        )
        
        # 静电相互作用
        elec_energy = self._calculate_electrostatic(
            dna_config, nanoparticle
        )
        
        # 范德华相互作用
        vdw_energy = self._calculate_vdw(
            dna_config, nanoparticle
        )
        
        # 氢键
        h_bonds = self._count_hydrogen_bonds(dna_config, nanoparticle)
        h_bond_energy = h_bonds * -0.5  # 每个氢键约0.5 eV
        
        # π-π堆积（对石墨烯、CNT）
        stacking = self._calculate_stacking(dna_config, nanoparticle)
        
        # 溶剂化效应
        solvation = self._calculate_solvation_energy(
            dna_config, nanoparticle
        )
        
        # 总结合能
        total_binding = elec_energy + vdw_energy + h_bond_energy + stacking
        
        # 结合自由能（包含熵贡献）
        binding_free = total_binding + solvation
        
        # 接触面积
        contact_area = self._calculate_contact_area(dna_config, nanoparticle)
        
        # 驻留时间估计
        residence_time = self._estimate_residence_time(binding_free)
        
        return DNAAdsorptionResult(
            binding_energy=total_binding,
            adsorption_configuration=dna_config,
            contact_area=contact_area,
            hydrogen_bonds=h_bonds,
            stacking_interactions=int(abs(stacking) / 0.3),
            electrostatic_energy=elec_energy,
            vdw_energy=vdw_energy,
            solvation_energy=solvation,
            binding_free_energy=binding_free,
            residence_time=residence_time
        )
    
    def _position_dna_on_surface(self, dna: DNAStructure,
                                   nanoparticle: Nanoparticle,
                                   mode: str) -> np.ndarray:
        """将DNA定位到纳米粒子表面"""
        coords = dna.coordinates.copy()
        
        if mode == "end_on":
            # 一端吸附
            end_point = nanoparticle.coordinates + np.array([0, 0, nanoparticle.radius])
            coords += end_point - coords[0]
        
        elif mode == "side_on":
            # 侧面吸附（适用于平面）
            surface_point = nanoparticle.coordinates + np.array([0, 0, nanoparticle.radius])
            coords += surface_point - np.mean(coords, axis=0)
        
        elif mode == "groove_binding":
            # 沟槽结合
            coords = self._wrap_around_particle(coords, nanoparticle)
        
        return coords
    
    def _wrap_around_particle(self, coords: np.ndarray,
                               nanoparticle: Nanoparticle) -> np.ndarray:
        """将DNA缠绕在纳米粒子周围"""
        center = nanoparticle.coordinates
        radius = nanoparticle.radius
        
        wrapped = []
        n_points = len(coords)
        for i, coord in enumerate(coords):
            angle = 2 * np.pi * i / n_points
            offset = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0.5 * i  # 螺旋上升
            ])
            wrapped.append(center + offset)
        
        return np.array(wrapped)
    
    def _calculate_electrostatic(self, dna_coords: np.ndarray,
                                  nanoparticle: Nanoparticle) -> float:
        """计算静电相互作用"""
        # Debye-Hückel近似
        kappa = np.sqrt(self.ionic_strength) * 3.3  # 1/nm
        
        # DNA电荷（磷酸骨架带负电）
        dna_charge = -2.0 * len(dna_coords)  # 每个碱基对约-2e
        
        # 纳米粒子表面电荷
        np_charge = nanoparticle.charge
        
        # 平均距离
        distances = np.linalg.norm(
            dna_coords - nanoparticle.coordinates, axis=1
        )
        avg_distance = np.mean(distances)
        
        # 屏蔽库仑相互作用
        energy = (dna_charge * np_charge * 1.44) / avg_distance  # eV·Å/e^2
        energy *= np.exp(-kappa * avg_distance / 10)  # 屏蔽
        
        return energy
    
    def _calculate_vdw(self, dna_coords: np.ndarray,
                       nanoparticle: Nanoparticle) -> float:
        """计算范德华相互作用"""
        total_vdw = 0.0
        
        # Hamaker常数估计
        A_H = 40e-21  # J (水性介质)
        
        for coord in dna_coords:
            # 距离纳米粒子表面
            r = np.linalg.norm(coord - nanoparticle.coordinates)
            surface_dist = r - nanoparticle.radius
            
            if surface_dist > 0.5:  # 避免接触
                # 简化范德华吸引
                energy = -A_H / (6 * surface_dist * 1e-9)  # 转换为eV
                total_vdw += energy / 1.6e-19  # J to eV
        
        return total_vdw * 0.01  # 缩放因子
    
    def _count_hydrogen_bonds(self, dna_coords: np.ndarray,
                               nanoparticle: Nanoparticle) -> int:
        """统计氢键数量"""
        # 简化的氢键计数
        surface_ligands = nanoparticle.ligands or []
        
        h_bonds = 0
        for ligand in surface_ligands:
            if ligand.get('type') in ['amine', 'carboxyl', 'thiol']:
                # 检查距离
                for coord in dna_coords:
                    dist = np.linalg.norm(coord - ligand['position'])
                    if 1.5 < dist < 3.0:  # 氢键距离范围
                        h_bonds += 1
        
        return min(h_bonds, len(dna_coords))  # 上限
    
    def _calculate_stacking(self, dna_coords: np.ndarray,
                           nanoparticle: Nanoparticle) -> float:
        """计算π-π堆积相互作用"""
        stacking_energy = 0.0
        
        # 只有sp2碳材料才有强堆积
        if nanoparticle.material in [MaterialType.GRAPHENE, MaterialType.CNT]:
            # 碱基与表面的π-π相互作用
            for coord in dna_coords:
                dist = np.linalg.norm(coord - nanoparticle.coordinates)
                surface_dist = abs(dist - nanoparticle.radius)
                
                if surface_dist < 0.4:  # 堆积距离
                    stacking_energy -= 0.3  # 每个碱基约0.3 eV
        
        return stacking_energy
    
    def _calculate_solvation_energy(self, dna_coords: np.ndarray,
                                     nanoparticle: Nanoparticle) -> float:
        """计算溶剂化自由能变化"""
        # 疏水效应（疏水纳米粒子）
        if nanoparticle.material in [MaterialType.GRAPHENE, MaterialType.CNT]:
            # 疏水表面暴露
            return 0.05 * len(dna_coords)  # kcal/mol per residue
        
        # 亲水纳米粒子
        return -0.02 * len(dna_coords)
    
    def _calculate_contact_area(self, dna_coords: np.ndarray,
                                 nanoparticle: Nanoparticle) -> float:
        """计算接触面积"""
        # 简化的接触面积估计
        contact_atoms = 0
        for coord in dna_coords:
            dist = np.linalg.norm(coord - nanoparticle.coordinates)
            if abs(dist - nanoparticle.radius) < 0.5:
                contact_atoms += 1
        
        # 每个原子约20 Å^2
        return contact_atoms * 20
    
    def _estimate_residence_time(self, binding_energy: float) -> float:
        """估计DNA在表面的驻留时间"""
        # 从结合能估计速率常数
        k_off = 1e12 * np.exp(binding_energy / (self.kB * self.temperature))
        tau = 1e12 / k_off  # ps
        return tau
    
    def simulate_dynamics(self,
                         dna: DNAStructure,
                         nanoparticle: Nanoparticle,
                         n_steps: int = 10000,
                         dt: float = 1.0) -> Dict:
        """
        分子动力学模拟DNA-纳米粒子复合物
        
        Args:
            n_steps: MD步数
            dt: 时间步长 (fs)
        
        Returns:
            轨迹和统计信息
        """
        logger.info(f"开始MD模拟: {n_steps}步")
        
        # 初始化
        coords = dna.coordinates.copy()
        velocities = np.random.randn(*coords.shape) * 0.1
        
        trajectory = []
        energies = []
        
        for step in range(n_steps):
            # 计算力
            forces = self._calculate_forces(coords, nanoparticle)
            
            # 速度Verlet积分
            velocities += 0.5 * forces * dt
            coords += velocities * dt
            
            # 约束：保持在纳米粒子附近
            coords = self._apply_constraints(coords, nanoparticle)
            
            forces = self._calculate_forces(coords, nanoparticle)
            velocities += 0.5 * forces * dt
            
            # 能量计算
            if step % 100 == 0:
                energy = self._calculate_total_energy(coords, nanoparticle)
                energies.append(energy)
                trajectory.append(coords.copy())
        
        return {
            'trajectory': np.array(trajectory),
            'energies': np.array(energies),
            'final_coords': coords,
            'rmsd': self._calculate_rmsd(trajectory)
        }
    
    def _calculate_forces(self, coords: np.ndarray,
                         nanoparticle: Nanoparticle) -> np.ndarray:
        """计算原子力（简化）"""
        forces = np.zeros_like(coords)
        
        # 吸附到表面的力
        center = nanoparticle.coordinates
        for i, coord in enumerate(coords):
            r = coord - center
            dist = np.linalg.norm(r)
            
            # 简化的吸引势
            if dist > nanoparticle.radius:
                force_mag = -0.01 / (dist - nanoparticle.radius + 0.1)**2
                forces[i] = force_mag * r / dist
        
        return forces
    
    def _apply_constraints(self, coords: np.ndarray,
                          nanoparticle: Nanoparticle) -> np.ndarray:
        """应用几何约束"""
        constrained = coords.copy()
        center = nanoparticle.coordinates
        
        for i, coord in enumerate(constrained):
            r = coord - center
            dist = np.linalg.norm(r)
            
            # 限制最小距离（避免穿透）
            min_dist = nanoparticle.radius + 0.2
            if dist < min_dist:
                constrained[i] = center + r * (min_dist / dist)
        
        return constrained
    
    def _calculate_total_energy(self, coords: np.ndarray,
                                nanoparticle: Nanoparticle) -> float:
        """计算总能量"""
        elec = self._calculate_electrostatic(coords, nanoparticle)
        vdw = self._calculate_vdw(coords, nanoparticle)
        return elec + vdw
    
    def _calculate_rmsd(self, trajectory: List[np.ndarray]) -> np.ndarray:
        """计算RMSD"""
        if len(trajectory) < 2:
            return np.array([0.0])
        
        ref = trajectory[0]
        rmsds = []
        for frame in trajectory:
            rmsd = np.sqrt(np.mean((frame - ref)**2))
            rmsds.append(rmsd)
        
        return np.array(rmsds)


class GeneDeliveryOptimizer:
    """基因递送载体优化器"""
    
    def __init__(self, calculator: DNANanoparticleCalculator):
        self.calculator = calculator
    
    def optimize_nanoparticle_size(self,
                                   dna: DNAStructure,
                                   material: MaterialType,
                                   size_range: Tuple[float, float] = (1, 50)) -> Dict:
        """
        优化纳米粒子尺寸以实现最佳DNA包装
        
        Args:
            size_range: 尺寸范围 (nm)
        
        Returns:
            优化结果
        """
        logger.info(f"优化{material.value}纳米粒子尺寸")
        
        sizes = np.linspace(size_range[0], size_range[1], 20)
        results = []
        
        for radius in sizes:
            nanoparticle = Nanoparticle(
                material=material,
                radius=radius,
                coordinates=np.array([0, 0, 0]),
                surface_atoms=np.array([]),
                charge=0.0 if material in [MaterialType.GRAPHENE] else 10.0
            )
            
            result = self.calculator.calculate_adsorption_energy(
                dna, nanoparticle, binding_mode="groove_binding"
            )
            
            # 计算包装效率
            dna_volume = len(dna.sequence) * 0.34 * 2.0  # nm^3 (简化)
            np_volume = nanoparticle.get_volume()
            packing_efficiency = dna_volume / np_volume
            
            results.append({
                'radius': radius,
                'binding_energy': result.binding_energy,
                'packing_efficiency': packing_efficiency,
                'score': result.binding_energy * packing_efficiency
            })
        
        # 找到最佳尺寸
        best = min(results, key=lambda x: x['score'])
        
        return {
            'optimal_radius_nm': best['radius'],
            'binding_energy_ev': best['binding_energy'],
            'all_results': results
        }
    
    def design_surface_ligands(self,
                               dna: DNAStructure,
                               nanoparticle: Nanoparticle,
                               target_property: str = "transfection") -> List[Dict]:
        """
        设计纳米粒子表面配体
        
        Args:
            target_property: 目标性质 (transfection, targeting, stability)
        
        Returns:
            推荐配体列表
        """
        logger.info(f"设计表面配体用于{target_property}")
        
        ligand_library = [
            {'name': 'PEI', 'charge': +20, 'hydrophobic': False, 'targeting': False},
            {'name': 'PEG', 'charge': 0, 'hydrophobic': False, 'targeting': False},
            {'name': 'folate', 'charge': -1, 'hydrophobic': False, 'targeting': True},
            {'name': 'RGD', 'charge': +2, 'hydrophobic': False, 'targeting': True},
            {'name': 'lipid', 'charge': 0, 'hydrophobic': True, 'targeting': False},
            {'name': 'chitosan', 'charge': +50, 'hydrophobic': False, 'targeting': False},
        ]
        
        scores = []
        for ligand in ligand_library:
            score = self._score_ligand(ligand, target_property)
            scores.append({**ligand, 'score': score})
        
        # 排序
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        return scores[:5]
    
    def _score_ligand(self, ligand: Dict, target: str) -> float:
        """评分配体"""
        if target == "transfection":
            # 需要正电荷用于DNA结合和细胞摄取
            return ligand['charge'] * 0.1 + (10 if not ligand['hydrophobic'] else 5)
        
        elif target == "targeting":
            return 20 if ligand['targeting'] else 0
        
        elif target == "stability":
            return 10 if ligand['name'] == 'PEG' else 5
        
        return 0.0
    
    def predict_transfection_efficiency(self,
                                       dna: DNAStructure,
                                       nanoparticle: Nanoparticle) -> float:
        """预测转染效率"""
        # 计算复合物性质
        result = self.calculator.calculate_adsorption_energy(
            dna, nanoparticle
        )
        
        # 基于结合能和表面电荷预测转染效率
        # 经验模型
        charge_factor = min(abs(nanoparticle.charge) / 30, 1.0)
        binding_factor = min(abs(result.binding_energy) / 10, 1.0)
        size_factor = 1.0 if 10 < nanoparticle.radius < 100 else 0.5
        
        efficiency = charge_factor * binding_factor * size_factor * 100
        
        return efficiency


class BiosensorDesigner:
    """生物传感器设计器"""
    
    def __init__(self, calculator: DNANanoparticleCalculator):
        self.calculator = calculator
    
    def design_dna_sensor(self,
                         target_sequence: str,
                         material: MaterialType = MaterialType.GOLD_NP) -> Dict:
        """
        设计基于DNA的纳米生物传感器
        
        Args:
            target_sequence: 目标检测序列
            material: 传感材料
        
        Returns:
            传感器设计
        """
        logger.info(f"设计DNA传感器用于检测: {target_sequence}")
        
        # 设计探针序列
        probe = self._design_probe(target_sequence)
        
        # 计算探针-靶标结合
        probe_dna = DNAStructure(
            sequence=probe,
            coordinates=np.random.randn(len(probe), 3) * 2,
            backbone_coords=np.random.randn(len(probe), 3),
            is_double_strand=False
        )
        
        target_dna = DNAStructure(
            sequence=target_sequence,
            coordinates=np.random.randn(len(target_sequence), 3) * 2,
            backbone_coords=np.random.randn(len(target_sequence), 3),
            is_double_strand=False
        )
        
        # 传感器纳米粒子
        nanoparticle = Nanoparticle(
            material=material,
            radius=5.0,
            coordinates=np.array([0, 0, 0]),
            surface_atoms=np.random.randn(100, 3) * 5,
            charge=-10.0,
            ligands=[{'type': 'thiol', 'position': np.random.randn(3)}]
        )
        
        # 计算相互作用
        probe_result = self.calculator.calculate_adsorption_energy(
            probe_dna, nanoparticle, binding_mode="end_on"
        )
        
        # 预测检测限
        detection_limit = self._estimate_detection_limit(
            probe_result.binding_energy
        )
        
        # 选择性评估
        selectivity = self._evaluate_selectivity(probe, target_sequence)
        
        return {
            'probe_sequence': probe,
            'nanoparticle_material': material.value,
            'probe_binding_energy': probe_result.binding_energy,
            'predicted_detection_limit_nM': detection_limit,
            'selectivity_score': selectivity,
            'signal_mechanism': self._get_signal_mechanism(material)
        }
    
    def _design_probe(self, target: str) -> str:
        """设计互补探针序列"""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        return ''.join([complement.get(b, b) for b in target[::-1]])
    
    def _estimate_detection_limit(self, binding_energy: float) -> float:
        """估计检测限"""
        # 结合能越强，检测限越低
        return 10 ** (binding_energy / 2)  # nM
    
    def _evaluate_selectivity(self, probe: str, target: str) -> float:
        """评估选择性（与错配序列的区分度）"""
        # 计算序列相似度
        matches = sum(p == t for p, t in zip(probe[::-1], target))
        return matches / len(target)
    
    def _get_signal_mechanism(self, material: MaterialType) -> str:
        """获取信号转导机制"""
        mechanisms = {
            MaterialType.GOLD_NP: "SPR_shift",
            MaterialType.QUANTUM_DOT: "fluorescence_quenching",
            MaterialType.GRAPHENE: "FRET_quenching",
            MaterialType.CNT: "conductance_change"
        }
        return mechanisms.get(material, "unknown")


# 应用案例演示
def dna_material_interaction_examples():
    """DNA-材料相互作用应用示例"""
    logger.info("=" * 60)
    logger.info("DNA-材料相互作用应用示例")
    logger.info("=" * 60)
    
    # 1. 创建DNA结构
    sequence = "AGCTAGCTAGCTAGCTAGCT"
    dna = DNAStructure(
        sequence=sequence,
        coordinates=np.random.randn(len(sequence), 3) * 3,
        backbone_coords=np.random.randn(len(sequence), 3),
        is_double_strand=True
    )
    
    logger.info(f"DNA序列: {sequence}")
    logger.info(f"双链DNA: {dna.is_double_strand}")
    
    # 2. 初始化计算器
    calculator = DNANanoparticleCalculator(
        solvent="water",
        temperature=300.0,
        ionic_strength=0.15
    )
    
    # 3. 金纳米粒子-DNA相互作用
    gold_np = Nanoparticle(
        material=MaterialType.GOLD_NP,
        radius=5.0,  # nm
        coordinates=np.array([0, 0, 0]),
        surface_atoms=np.random.randn(200, 3) * 5,
        charge=+30,  # 正电荷修饰
        ligands=[
            {'type': 'amine', 'position': np.array([5, 0, 0])},
            {'type': 'thiol', 'position': np.array([-5, 0, 0])}
        ]
    )
    
    result_gold = calculator.calculate_adsorption_energy(
        dna, gold_np, binding_mode="end_on"
    )
    
    logger.info(f"\n金纳米粒子-DNA相互作用:")
    logger.info(f"  结合能: {result_gold.binding_energy:.3f} eV")
    logger.info(f"  氢键数: {result_gold.hydrogen_bonds}")
    logger.info(f"  驻留时间: {result_gold.residence_time:.2e} ps")
    
    # 4. 石墨烯-DNA相互作用
    graphene = Nanoparticle(
        material=MaterialType.GRAPHENE,
        radius=50.0,  # 平面尺寸
        coordinates=np.array([0, 0, 0]),
        surface_atoms=np.random.randn(500, 3) * 50,
        charge=0.0
    )
    
    result_graphene = calculator.calculate_adsorption_energy(
        dna, graphene, binding_mode="side_on"
    )
    
    logger.info(f"\n石墨烯-DNA相互作用:")
    logger.info(f"  结合能: {result_graphene.binding_energy:.3f} eV")
    logger.info(f"  π-π堆积: {result_graphene.stacking_interactions}")
    logger.info(f"  接触面积: {result_graphene.contact_area:.1f} Å²")
    
    # 5. 基因递送优化
    optimizer = GeneDeliveryOptimizer(calculator)
    
    size_opt = optimizer.optimize_nanoparticle_size(
        dna, MaterialType.GOLD_NP, size_range=(1, 20)
    )
    logger.info(f"\n最佳纳米粒子尺寸: {size_opt['optimal_radius_nm']:.2f} nm")
    
    # 配体设计
    ligands = optimizer.design_surface_ligands(
        dna, gold_np, target_property="transfection"
    )
    logger.info(f"推荐配体: {[l['name'] for l in ligands[:3]]}")
    
    # 转染效率预测
    efficiency = optimizer.predict_transfection_efficiency(dna, gold_np)
    logger.info(f"预测转染效率: {efficiency:.1f}%")
    
    # 6. 生物传感器设计
    designer = BiosensorDesigner(calculator)
    
    sensor = designer.design_dna_sensor(
        target_sequence="AGCTAGCT",
        material=MaterialType.GOLD_NP
    )
    
    logger.info(f"\n生物传感器设计:")
    logger.info(f"  探针序列: {sensor['probe_sequence']}")
    logger.info(f"  检测限: {sensor['predicted_detection_limit_nM']:.2f} nM")
    logger.info(f"  信号机制: {sensor['signal_mechanism']}")
    
    # 7. MD模拟
    md_results = calculator.simulate_dynamics(
        dna, gold_np, n_steps=1000, dt=2.0
    )
    
    logger.info(f"\nMD模拟结果:")
    logger.info(f"  轨迹长度: {len(md_results['trajectory'])}帧")
    logger.info(f"  最终RMSD: {md_results['rmsd'][-1]:.3f} Å")
    
    return {
        'gold_result': result_gold,
        'graphene_result': result_graphene,
        'size_optimization': size_opt,
        'ligand_design': ligands,
        'sensor_design': sensor,
        'md_results': md_results
    }


if __name__ == "__main__":
    dna_material_interaction_examples()
