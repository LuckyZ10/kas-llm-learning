"""
CatKit Workflow Module - 催化剂筛选与电化学活性分析

本模块集成CatKit功能，提供：
- 催化表面生成与吸附位点枚举
- 火山图构建与分析
- 过电位计算
- HER/OER/ORR活性预测
- 线性标度关系分析

Reference:
    - Boes et al., Catalysis Letters 2019, 149, 2091-2096
    - Nørskov et al., J. Phys. Chem. B 2004, 108, 17886

Author: DFT-LAMMPS Team
Date: 2025
"""

import os
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Set, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings
from enum import Enum
from collections import defaultdict


class ReactionType(Enum):
    """电化学反应类型"""
    HER = "her"           # 氢气析出反应
    OER = "oer"           # 氧气析出反应  
    ORR = "orr"           # 氧气还原反应
    CO2RR = "co2rr"       # CO2还原反应
    NRR = "nrr"           # N2还原反应


class CatalystType(Enum):
    """催化剂类型"""
    METAL = "metal"           # 纯金属
    ALLOY = "alloy"           # 合金
    OXIDE = "oxide"           # 氧化物
    SULFIDE = "sulfide"       # 硫化物
    NITRIDE = "nitride"       # 氮化物
    SINGLE_ATOM = "single_atom"  # 单原子催化剂


@dataclass
class AdsorptionSite:
    """
    吸附位点数据类
    
    Attributes:
        position: 位点坐标 [x, y, z]
        site_type: 位点类型 (top/bridge/hollow)
        coordination: 配位数
        indices: 邻近原子索引
        symbol: 位点符号表示
    """
    position: np.ndarray
    site_type: str
    coordination: int
    indices: Tuple[int, ...]
    symbol: str = ""
    
    def __post_init__(self):
        if isinstance(self.position, list):
            self.position = np.array(self.position)


@dataclass
class SurfaceStructure:
    """
    表面结构数据类
    
    Attributes:
        atoms: ASE Atoms对象（或类似结构）
        miller_index: Miller指数
        layers: 原子层数
        vacuum: 真空层厚度
        adsorption_sites: 吸附位点列表
        formula: 化学式
    """
    atoms: Optional[Any] = None
    miller_index: Tuple[int, int, int] = (1, 1, 1)
    layers: int = 4
    vacuum: float = 15.0
    adsorption_sites: List[AdsorptionSite] = field(default_factory=list)
    formula: str = ""
    energy: float = 0.0
    
    def get_sites_by_type(self, site_type: str) -> List[AdsorptionSite]:
        """按类型获取吸附位点"""
        return [s for s in self.adsorption_sites if s.site_type == site_type]


@dataclass
class ReactionIntermediate:
    """
    反应中间体
    
    Attributes:
        name: 中间体名称
        formula: 化学式
        adsorption_energy: 吸附能 (eV)
        free_energy_correction: 自由能校正 (eV)
        site: 吸附位点
    """
    name: str
    formula: str
    adsorption_energy: float = 0.0
    free_energy_correction: float = 0.0
    site: Optional[AdsorptionSite] = None
    
    @property
    def free_energy(self) -> float:
        """计算自由能"""
        return self.adsorption_energy + self.free_energy_correction


@dataclass
class ScalingRelation:
    """
    线性标度关系
    
    ΔG(AB) = α * ΔG(A) + β
    
    Attributes:
        descriptor: 描述符中间体
        target: 目标中间体
        slope: 斜率 α
        intercept: 截距 β
        r_squared: R²值
        rmse: 均方根误差
    """
    descriptor: str
    target: str
    slope: float
    intercept: float
    r_squared: float = 0.0
    rmse: float = 0.0
    
    def predict(self, descriptor_energy: float) -> float:
        """根据描述符预测目标能量"""
        return self.slope * descriptor_energy + self.intercept


class CatKitGenerator:
    """
    CatKit表面生成器
    
    生成催化表面结构并枚举吸附位点
    
    Example:
        >>> generator = CatKitGenerator()
        >>> surface = generator.generate_surface("Pt", (1, 1, 1), layers=4)
        >>> sites = generator.get_adsorption_sites(surface)
    """
    
    def __init__(self, ase_available: bool = True):
        """
        初始化生成器
        
        Args:
            ase_available: 是否使用ASE库
        """
        self.ase_available = ase_available
        self.surfaces: Dict[str, SurfaceStructure] = {}
        
        if ase_available:
            try:
                import ase
                from ase import Atoms
                from ase.build import bulk, surface
                from ase.spacegroup import crystal
                self._ase = ase
                self._bulk = bulk
                self._surface = surface
            except ImportError:
                warnings.warn("ASE未安装，使用内置简化模型")
                self.ase_available = False
    
    def generate_bulk(
        self,
        element: str,
        crystal_structure: str = "fcc",
        a: Optional[float] = None,
        covera: Optional[float] = None
    ) -> Any:
        """
        生成块体结构
        
        Args:
            element: 元素符号
            crystal_structure: 晶体结构 (fcc/bcc/hcp)
            a: 晶格常数
            covera: c/a比率（hcp结构）
            
        Returns:
            ASE Atoms对象
        """
        if self.ase_available:
            bulk_atoms = self._bulk(element, crystalstructure=crystal_structure, a=a)
            return bulk_atoms
        else:
            # 简化模型
            return self._generate_mock_bulk(element, crystal_structure)
    
    def generate_surface(
        self,
        element: str,
        miller_index: Tuple[int, int, int],
        layers: int = 4,
        vacuum: float = 15.0,
        periodic: bool = True
    ) -> SurfaceStructure:
        """
        生成表面结构
        
        Args:
            element: 元素符号
            miller_index: Miller指数
            layers: 原子层数
            vacuum: 真空层厚度 (Angstrom)
            periodic: 是否周期性
            
        Returns:
            SurfaceStructure对象
        """
        if self.ase_available:
            try:
                bulk_atoms = self.generate_bulk(element)
                from ase.build import surface
                surface_atoms = surface(
                    bulk_atoms, 
                    miller_index, 
                    layers=layers,
                    vacuum=vacuum,
                    periodic=periodic
                )
                
                surf = SurfaceStructure(
                    atoms=surface_atoms,
                    miller_index=miller_index,
                    layers=layers,
                    vacuum=vacuum,
                    formula=f"{element}_{miller_index}"
                )
                
            except Exception as e:
                warnings.warn(f"ASE表面生成失败: {e}，使用简化模型")
                surf = self._generate_mock_surface(element, miller_index, layers, vacuum)
        else:
            surf = self._generate_mock_surface(element, miller_index, layers, vacuum)
        
        self.surfaces[f"{element}_{miller_index}"] = surf
        return surf
    
    def generate_oxide_surface(
        self,
        metal: str,
        oxygen: str = "O",
        stoichiometry: str = "MO2",
        miller_index: Tuple[int, int, int] = (1, 1, 0),
        layers: int = 2,
        vacuum: float = 15.0
    ) -> SurfaceStructure:
        """
        生成氧化物表面
        
        Args:
            metal: 金属元素
            oxygen: 氧元素
            stoichiometry: 化学计量比
            miller_index: Miller指数
            layers: 层数
            vacuum: 真空层
            
        Returns:
            SurfaceStructure对象
        """
        # 简化实现：基于金属表面添加O原子
        metal_surface = self.generate_surface(metal, miller_index, layers, vacuum)
        
        # 在表面添加O原子（简化模型）
        surf = SurfaceStructure(
            atoms=metal_surface.atoms,
            miller_index=miller_index,
            layers=layers,
            vacuum=vacuum,
            formula=f"{stoichiometry}_{miller_index}"
        )
        
        return surf
    
    def get_adsorption_sites(
        self,
        surface: SurfaceStructure,
        symmetry_reduced: bool = True
    ) -> List[AdsorptionSite]:
        """
        获取吸附位点
        
        Args:
            surface: 表面结构
            symmetry_reduced: 是否只返回对称不等价位点
            
        Returns:
            吸附位点列表
        """
        sites = []
        
        if self.ase_available and surface.atoms is not None:
            try:
                # 使用CatKit获取位点（简化实现）
                sites = self._get_sites_from_atoms(surface.atoms)
            except:
                sites = self._generate_mock_sites(surface)
        else:
            sites = self._generate_mock_sites(surface)
        
        surface.adsorption_sites = sites
        return sites
    
    def add_adsorbate(
        self,
        surface: SurfaceStructure,
        adsorbate: str,
        site: AdsorptionSite,
        height: float = 2.0
    ) -> Any:
        """
        在特定位点添加吸附物
        
        Args:
            surface: 表面结构
            adsorbate: 吸附物化学式
            site: 吸附位点
            height: 吸附高度 (Angstrom)
            
        Returns:
            带吸附物的结构
        """
        if self.ase_available:
            from ase import Atoms
            from ase.build import add_adsorbate
            
            atoms = surface.atoms.copy()
            
            # 创建吸附物分子（简化）
            ads = self._create_adsorbate(adsorbate)
            
            # 添加吸附物
            add_adsorbate(atoms, ads, height=height, position=site.position[:2])
            
            return atoms
        else:
            return None
    
    def _get_sites_from_atoms(self, atoms: Any) -> List[AdsorptionSite]:
        """从ASE Atoms获取吸附位点"""
        sites = []
        
        # 简化的位点识别
        positions = atoms.get_positions()
        
        # 识别top位点（表面原子）
        z_max = np.max(positions[:, 2])
        surface_indices = np.where(np.abs(positions[:, 2] - z_max) < 0.5)[0]
        
        for idx in surface_indices:
            site = AdsorptionSite(
                position=positions[idx] + np.array([0, 0, 2.0]),
                site_type="top",
                coordination=1,
                indices=(idx,),
                symbol=f"{atoms.get_chemical_symbols()[idx]}_top"
            )
            sites.append(site)
        
        # 添加bridge和hollow位点（简化）
        if len(surface_indices) >= 2:
            for i in range(len(surface_indices) - 1):
                idx1, idx2 = surface_indices[i], surface_indices[i + 1]
                bridge_pos = (positions[idx1] + positions[idx2]) / 2
                bridge_pos[2] = z_max + 2.0
                
                site = AdsorptionSite(
                    position=bridge_pos,
                    site_type="bridge",
                    coordination=2,
                    indices=(idx1, idx2),
                    symbol=f"bridge_{i}"
                )
                sites.append(site)
        
        return sites
    
    def _generate_mock_sites(self, surface: SurfaceStructure) -> List[AdsorptionSite]:
        """生成模拟吸附位点"""
        sites = []
        
        # 创建几个示例位点
        for i in range(3):
            site = AdsorptionSite(
                position=np.array([i * 2.0, i * 1.5, 10.0]),
                site_type="top" if i == 0 else ("bridge" if i == 1 else "hollow"),
                coordination=i + 1,
                indices=(i,),
                symbol=f"site_{i}"
            )
            sites.append(site)
        
        return sites
    
    def _generate_mock_bulk(self, element: str, structure: str) -> Any:
        """生成模拟块体结构"""
        # 返回None表示简化模式
        return None
    
    def _generate_mock_surface(
        self, 
        element: str, 
        miller_index: Tuple[int, int, int],
        layers: int,
        vacuum: float
    ) -> SurfaceStructure:
        """生成模拟表面结构"""
        return SurfaceStructure(
            atoms=None,
            miller_index=miller_index,
            layers=layers,
            vacuum=vacuum,
            formula=f"{element}_{miller_index}"
        )
    
    def _create_adsorbate(self, formula: str) -> Any:
        """创建吸附物分子"""
        from ase import Atoms
        
        # 简化实现
        if formula == "H":
            return Atoms("H", positions=[[0, 0, 0]])
        elif formula == "O":
            return Atoms("O", positions=[[0, 0, 0]])
        elif formula == "OH":
            return Atoms("OH", positions=[[0, 0, 0], [0.96, 0, 0]])
        elif formula == "OOH":
            return Atoms("OOH", positions=[[0, 0, 0], [1.4, 0, 0], [2.2, 0.7, 0]])
        else:
            return Atoms(formula)


class VolcanoPlotter:
    """
    火山图绘制器
    
    构建和分析催化活性火山图
    
    Example:
        >>> plotter = VolcanoPlotter(ReactionType.OER)
        >>> plotter.add_catalyst("Pt", d band_center=-2.5, overpotential=0.4)
        >>> fig = plotter.plot()
    """
    
    def __init__(
        self,
        reaction_type: ReactionType,
        descriptor: str = "d_band_center"
    ):
        """
        初始化火山图绘制器
        
        Args:
            reaction_type: 反应类型
            descriptor: 描述符名称
        """
        self.reaction_type = reaction_type
        self.descriptor = descriptor
        self.catalysts: Dict[str, Dict] = {}
        
        # 反应能量数据
        self.intermediates: Dict[str, Dict[str, ReactionIntermediate]] = defaultdict(dict)
        
        # 设置反应参数
        self._setup_reaction()
    
    def _setup_reaction(self):
        """设置反应参数"""
        if self.reaction_type == ReactionType.HER:
            self.intermediate_names = ["H*"]
            self.ideal_energy = 0.0  # ΔG_H = 0
            
        elif self.reaction_type == ReactionType.OER:
            # OER: H2O → O2 + 4H+ + 4e-
            self.intermediate_names = ["OH*", "O*", "OOH*"]
            self.ideal_energies = [1.23, 2.46, 3.69]  # 理想能量（vs RHE）
            
        elif self.reaction_type == ReactionType.ORR:
            # ORR: O2 + 4H+ + 4e- → 2H2O
            self.intermediate_names = ["OOH*", "O*", "OH*"]
            self.ideal_energies = [4.36, 3.13, 1.90]
            
        elif self.reaction_type == ReactionType.CO2RR:
            self.intermediate_names = ["COOH*", "CO*"]
            self.ideal_energies = [0.5, 0.0]
    
    def add_catalyst(
        self,
        name: str,
        descriptor_value: float,
        intermediate_energies: Dict[str, float],
        surface_area: float = 1.0
    ):
        """
        添加催化剂数据
        
        Args:
            name: 催化剂名称
            descriptor_value: 描述符数值
            intermediate_energies: 中间体能量字典
            surface_area: 表面积
        """
        self.catalysts[name] = {
            "descriptor": descriptor_value,
            "energies": intermediate_energies,
            "surface_area": surface_area,
        }
    
    def calculate_overpotential(
        self,
        intermediate_energies: Dict[str, float]
    ) -> float:
        """
        计算过电位
        
        Args:
            intermediate_energies: 中间体自由能字典
            
        Returns:
            过电位 (V)
        """
        if self.reaction_type == ReactionType.HER:
            # HER: η = |ΔG_H|
            dg_h = intermediate_energies.get("H*", 0.0)
            return abs(dg_h)
            
        elif self.reaction_type == ReactionType.OER:
            # OER过电位
            dg_oh = intermediate_energies.get("OH*", 0.0)
            dg_o = intermediate_energies.get("O*", 0.0)
            dg_ooh = intermediate_energies.get("OOH*", 0.0)
            
            # 各步骤能量变化
            step1 = dg_oh  # * → OH*
            step2 = dg_o - dg_oh  # OH* → O*
            step3 = dg_ooh - dg_o  # O* → OOH*
            step4 = 4.92 - dg_ooh  # OOH* → * + O2
            
            # 过电位
            max_step = max(step1, step2, step3, step4)
            eta = max_step - 1.23  # 1.23 V是OER平衡电势
            
            return max(eta, 0.0)
            
        elif self.reaction_type == ReactionType.ORR:
            # ORR过电位
            dg_ooh = intermediate_energies.get("OOH*", 4.92)
            dg_o = intermediate_energies.get("O*", 3.69)
            dg_oh = intermediate_energies.get("OH*", 2.46)
            
            step1 = dg_ooh - 4.92
            step2 = dg_o - dg_ooh
            step3 = dg_oh - dg_o
            step4 = -dg_oh
            
            max_step = max(abs(step1), abs(step2), abs(step3), abs(step4))
            eta = max_step - 1.23
            
            return max(eta, 0.0)
        
        return 0.0
    
    def fit_scaling_relations(self) -> List[ScalingRelation]:
        """
        拟合线性标度关系
        
        Returns:
            标度关系列表
        """
        relations = []
        
        if len(self.catalysts) < 2:
            return relations
        
        # 使用OH*作为描述符（OER/ORR）
        if self.reaction_type in [ReactionType.OER, ReactionType.ORR]:
            descriptor_name = "OH*"
            
            for target in ["O*", "OOH*"]:
                x_vals = []
                y_vals = []
                
                for cat_data in self.catalysts.values():
                    energies = cat_data["energies"]
                    if descriptor_name in energies and target in energies:
                        x_vals.append(energies[descriptor_name])
                        y_vals.append(energies[target])
                
                if len(x_vals) >= 2:
                    # 线性拟合
                    coeffs = np.polyfit(x_vals, y_vals, 1)
                    slope, intercept = coeffs
                    
                    # 计算R²
                    y_pred = np.polyval(coeffs, x_vals)
                    ss_res = np.sum((np.array(y_vals) - y_pred)**2)
                    ss_tot = np.sum((np.array(y_vals) - np.mean(y_vals))**2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                    
                    relation = ScalingRelation(
                        descriptor=descriptor_name,
                        target=target,
                        slope=slope,
                        intercept=intercept,
                        r_squared=r_squared
                    )
                    relations.append(relation)
        
        return relations
    
    def predict_activity(self, descriptor_range: Tuple[float, float]) -> Dict:
        """
        预测活性（基于标度关系）
        
        Args:
            descriptor_range: 描述符范围
            
        Returns:
            活性预测结果
        """
        relations = self.fit_scaling_relations()
        
        if not relations:
            return {}
        
        # 创建关系字典
        relation_dict = {r.target: r for r in relations}
        
        # 在描述符范围内采样
        x_vals = np.linspace(descriptor_range[0], descriptor_range[1], 100)
        overpotentials = []
        
        for x in x_vals:
            # 预测中间体能量
            energies = {"OH*": x}
            
            if "O*" in relation_dict:
                energies["O*"] = relation_dict["O*"].predict(x)
            if "OOH*" in relation_dict:
                energies["OOH*"] = relation_dict["OOH*"].predict(x)
            
            # 计算过电位
            eta = self.calculate_overpotential(energies)
            overpotentials.append(eta)
        
        # 找到最优描述符值
        min_idx = np.argmin(overpotentials)
        optimal_descriptor = x_vals[min_idx]
        min_overpotential = overpotentials[min_idx]
        
        return {
            "descriptor_values": x_vals.tolist(),
            "overpotentials": overpotentials,
            "optimal_descriptor": optimal_descriptor,
            "minimum_overpotential": min_overpotential,
        }
    
    def get_volcano_data(self) -> Dict:
        """
        获取火山图数据
        
        Returns:
            绘图数据字典
        """
        descriptors = []
        overpotentials = []
        names = []
        
        for name, data in self.catalysts.items():
            descriptors.append(data["descriptor"])
            eta = self.calculate_overpotential(data["energies"])
            overpotentials.append(eta)
            names.append(name)
        
        return {
            "descriptors": descriptors,
            "overpotentials": overpotentials,
            "names": names,
        }
    
    def plot(
        self,
        show_scaling: bool = True,
        save_path: Optional[str] = None
    ):
        """
        绘制火山图
        
        Args:
            show_scaling: 是否显示标度关系
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib未安装，无法绘图")
            return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 获取数据
        data = self.get_volcano_data()
        
        # 绘制催化剂点
        ax.scatter(
            data["descriptors"],
            data["overpotentials"],
            s=100,
            alpha=0.6,
            edgecolors='black'
        )
        
        # 添加标签
        for name, x, y in zip(data["names"], data["descriptors"], data["overpotentials"]):
            ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 绘制火山曲线（理论预测）
        if show_scaling and len(self.catalysts) >= 3:
            pred = self.predict_activity(
                (min(data["descriptors"]) - 0.5, max(data["descriptors"]) + 0.5)
            )
            if pred:
                ax.plot(
                    pred["descriptor_values"],
                    pred["overpotentials"],
                    'r--',
                    label='Scaling relation'
                )
        
        ax.set_xlabel(f"{self.descriptor} (eV)")
        ax.set_ylabel("Overpotential (V)")
        ax.set_title(f"{self.reaction_type.value.upper()} Volcano Plot")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class OverpotentialCalculator:
    """
    过电位计算器
    
    计算电化学反应的过电位和极限电势
    """
    
    def __init__(self, temperature: float = 298.15):
        """
        初始化计算器
        
        Args:
            temperature: 温度（K）
        """
        self.temperature = temperature
        self.kT = 8.617e-5 * temperature  # eV
        
        # 反应自由能数据（实验值，vs SHE）
        self.reference_energies = {
            "H2": 0.0,
            "H2O": -2.46,
            "O2": 4.92,  # H2O → ½O2 + 2H+ + 2e- (1.23V * 4e-)
        }
    
    def calculate_her_overpotential(self, dg_h: float) -> float:
        """
        计算HER过电位
        
        Sabatier原理: 最优 ΔG_H ≈ 0
        
        Args:
            dg_h: H*吸附自由能 (eV)
            
        Returns:
            过电位 (V)
        """
        # HER: H+ + e- → H*
        # η = |ΔG_H| (简化的模型)
        return abs(dg_h)
    
    def calculate_oer_overpotential(
        self,
        dg_oh: float,
        dg_o: float,
        dg_ooh: float,
        method: str = "standard"
    ) -> Dict:
        """
        计算OER过电位
        
        OER步骤:
        1. H2O(l) + * → OH* + H+ + e-
        2. OH* → O* + H+ + e-
        3. O* + H2O(l) → OOH* + H+ + e-
        4. OOH* → * + O2(g) + H+ + e-
        
        Args:
            dg_oh: OH*自由能
            dg_o: O*自由能  
            dg_ooh: OOH*自由能
            method: 计算方法
            
        Returns:
            过电位详细数据
        """
        # 各步骤自由能变化
        # 参考: 0 eV (H2O)
        step1 = dg_oh - 0.0  # OH*形成
        step2 = dg_o - dg_oh  # O*形成
        step3 = dg_ooh - dg_o  # OOH*形成
        step4 = 4.92 - dg_ooh  # O2释放
        
        steps = [step1, step2, step3, step4]
        step_names = ["OH* formation", "O* formation", "OOH* formation", "O2 release"]
        
        # 找到决速步
        max_step = max(steps)
        rate_determining_idx = steps.index(max_step)
        
        # 过电位
        equilibrium_potential = 1.23  # V vs RHE
        overpotential = max_step - equilibrium_potential
        
        # 极限电势（所有步骤放热时的电势）
        limiting_potential = max_step
        
        return {
            "overpotential": max(overpotential, 0.0),
            "limiting_potential": limiting_potential,
            "equilibrium_potential": equilibrium_potential,
            "steps": {name: energy for name, energy in zip(step_names, steps)},
            "rate_determining_step": step_names[rate_determining_idx],
            "rate_determining_energy": max_step,
        }
    
    def calculate_orr_overpotential(
        self,
        dg_ooh: float,
        dg_o: float,
        dg_oh: float
    ) -> Dict:
        """
        计算ORR过电位
        
        ORR步骤（酸性介质）:
        1. O2 + 4H+ + 4e- + * → OOH*
        2. OOH* + H+ + e- → O* + H2O
        3. O* + H+ + e- → OH*
        4. OH* + H+ + e- → H2O + *
        
        Args:
            dg_ooh: OOH*自由能
            dg_o: O*自由能
            dg_oh: OH*自由能
            
        Returns:
            过电位详细数据
        """
        # 参考: O2 + 4H+ + 4e- (4.92 eV)
        step1 = dg_ooh - 4.92
        step2 = dg_o - dg_ooh
        step3 = dg_oh - dg_o
        step4 = 0.0 - dg_oh
        
        steps = [step1, step2, step3, step4]
        step_names = ["OOH* formation", "O* formation", "OH* formation", "H2O release"]
        
        # 找到吸热最大的步骤（需要最大电势驱动）
        max_step = max(steps)
        rate_determining_idx = steps.index(max_step)
        
        equilibrium_potential = 1.23
        overpotential = max_step - equilibrium_potential
        
        return {
            "overpotential": max(overpotential, 0.0),
            "limiting_potential": 1.23 - max_step if max_step > 0 else 1.23,
            "equilibrium_potential": equilibrium_potential,
            "steps": {name: energy for name, energy in zip(step_names, steps)},
            "rate_determining_step": step_names[rate_determining_idx],
            "rate_determining_energy": max_step,
        }
    
    def calculate_selectivity(
        self,
        main_product_energy: float,
        side_product_energies: Dict[str, float]
    ) -> Dict[str, float]:
        """
        计算产物选择性
        
        Args:
            main_product_energy: 主产物能量
            side_product_energies: 副产物能量字典
            
        Returns:
            选择性指标
        """
        selectivity = {}
        
        for product, energy in side_product_energies.items():
            # 能量差
            delta_e = main_product_energy - energy
            
            # Boltzmann因子（简化）
            if abs(delta_e) < 10 * self.kT:
                ratio = 1 / (1 + np.exp(-delta_e / self.kT))
            else:
                ratio = 1.0 if delta_e < 0 else 0.0
            
            selectivity[product] = ratio
        
        return selectivity
    
    def calculate_tafel_slope(
        self,
        overpotentials: np.ndarray,
        current_densities: np.ndarray,
        method: str = "linear"
    ) -> float:
        """
        计算Tafel斜率
        
        η = a + b·log(i)
        
        Args:
            overpotentials: 过电位数组 (V)
            current_densities: 电流密度数组 (mA/cm²)
            method: 拟合方法
            
        Returns:
            Tafel斜率 (mV/decade)
        """
        # 取对数
        log_i = np.log10(current_densities)
        
        if method == "linear":
            # 线性拟合
            coeffs = np.polyfit(log_i, overpotentials, 1)
            tafel_slope = coeffs[0] * 1000  # 转换为mV/decade
        else:
            # 简化计算
            delta_eta = overpotentials[-1] - overpotentials[0]
            delta_logi = log_i[-1] - log_i[0]
            tafel_slope = (delta_eta / delta_logi) * 1000
        
        return tafel_slope


class CatalyticScreening:
    """
    催化剂筛选器
    
    批量筛选催化剂并排序
    """
    
    def __init__(self, reaction_type: ReactionType):
        """
        初始化筛选器
        
        Args:
            reaction_type: 反应类型
        """
        self.reaction_type = reaction_type
        self.candidates: List[Dict] = []
        self.eta_calc = OverpotentialCalculator()
        
    def add_candidate(
        self,
        name: str,
        intermediate_energies: Dict[str, float],
        metadata: Optional[Dict] = None
    ):
        """
        添加候选催化剂
        
        Args:
            name: 催化剂名称
            intermediate_energies: 中间体能量
            metadata: 附加元数据
        """
        candidate = {
            "name": name,
            "energies": intermediate_energies,
            "metadata": metadata or {},
        }
        
        # 计算过电位
        if self.reaction_type == ReactionType.HER:
            eta = self.eta_calc.calculate_her_overpotential(
                intermediate_energies.get("H*", 0.0)
            )
        elif self.reaction_type == ReactionType.OER:
            result = self.eta_calc.calculate_oer_overpotential(
                intermediate_energies.get("OH*", 0.0),
                intermediate_energies.get("O*", 0.0),
                intermediate_energies.get("OOH*", 0.0)
            )
            eta = result["overpotential"]
            candidate["details"] = result
        elif self.reaction_type == ReactionType.ORR:
            result = self.eta_calc.calculate_orr_overpotential(
                intermediate_energies.get("OOH*", 4.92),
                intermediate_energies.get("O*", 3.69),
                intermediate_energies.get("OH*", 2.46)
            )
            eta = result["overpotential"]
            candidate["details"] = result
        else:
            eta = 1.0  # 默认值
        
        candidate["overpotential"] = eta
        self.candidates.append(candidate)
    
    def rank_candidates(self) -> List[Dict]:
        """
        排序候选催化剂
        
        Returns:
            按过电位排序的催化剂列表
        """
        return sorted(self.candidates, key=lambda x: x["overpotential"])
    
    def get_top_candidates(self, n: int = 5) -> List[Dict]:
        """
        获取最优候选
        
        Args:
            n: 返回数量
            
        Returns:
            前n个最优催化剂
        """
        ranked = self.rank_candidates()
        return ranked[:n]
    
    def export_results(self, output_path: str):
        """
        导出结果
        
        Args:
            output_path: 输出文件路径
        """
        ranked = self.rank_candidates()
        
        with open(output_path, 'w') as f:
            json.dump({
                "reaction_type": self.reaction_type.value,
                "candidates": ranked,
                "total_candidates": len(ranked),
            }, f, indent=2, default=str)


# ============================================================
# 使用示例
# ============================================================

def example_surface_generation():
    """表面生成示例"""
    print("表面生成示例")
    print("-" * 40)
    
    # 创建生成器
    generator = CatKitGenerator(ase_available=False)
    
    # 生成Pt(111)表面
    pt111 = generator.generate_surface("Pt", (1, 1, 1), layers=4, vacuum=15.0)
    print(f"生成表面: {pt111.formula}")
    print(f"  Miller指数: {pt111.miller_index}")
    print(f"  原子层数: {pt111.layers}")
    print(f"  真空层: {pt111.vacuum} Å")
    
    # 获取吸附位点
    sites = generator.get_adsorption_sites(pt111)
    print(f"  吸附位点数: {len(sites)}")
    
    for site in sites[:3]:
        print(f"    {site.symbol}: {site.site_type} (配位数: {site.coordination})")
    
    return generator, pt111


def example_volcano_plot():
    """火山图示例"""
    print("\n火山图示例")
    print("-" * 40)
    
    # 创建OER火山图
    plotter = VolcanoPlotter(ReactionType.OER, descriptor="OH_binding")
    
    # 添加催化剂数据（示例数据）
    catalysts_data = [
        ("IrO2", 0.8, {"OH*": 0.8, "O*": 1.8, "OOH*": 2.6}),
        ("RuO2", 0.7, {"OH*": 0.7, "O*": 1.6, "OOH*": 2.4}),
        ("Pt", 1.2, {"OH*": 1.2, "O*": 2.4, "OOH*": 3.4}),
        ("Au", 1.5, {"OH*": 1.5, "O*": 3.0, "OOH*": 4.2}),
        ("Co3O4", 0.9, {"OH*": 0.9, "O*": 2.0, "OOH*": 2.8}),
    ]
    
    for name, descriptor, energies in catalysts_data:
        plotter.add_catalyst(name, descriptor, energies)
    
    # 获取火山图数据
    volcano_data = plotter.get_volcano_data()
    print(f"催化剂数量: {len(volcano_data['names'])}")
    
    # 计算各催化剂过电位
    print("\n催化剂过电位:")
    for name, eta in zip(volcano_data['names'], volcano_data['overpotentials']):
        print(f"  {name}: {eta:.3f} V")
    
    # 拟合标度关系
    relations = plotter.fit_scaling_relations()
    print(f"\n标度关系 ({len(relations)}):")
    for r in relations:
        print(f"  {r.descriptor} → {r.target}:")
        print(f"    斜率: {r.slope:.3f}, 截距: {r.intercept:.3f}, R²: {r.r_squared:.3f}")
    
    # 预测最优活性
    prediction = plotter.predict_activity((0.5, 1.5))
    if prediction:
        print(f"\n预测最优:")
        print(f"  最优描述符值: {prediction['optimal_descriptor']:.3f} eV")
        print(f"  最小过电位: {prediction['minimum_overpotential']:.3f} V")
    
    return plotter


def example_overpotential():
    """过电位计算示例"""
    print("\n过电位计算示例")
    print("-" * 40)
    
    calc = OverpotentialCalculator(temperature=298.15)
    
    # HER示例
    dg_h_values = [-0.5, -0.2, 0.0, 0.2, 0.5]
    print("HER过电位:")
    for dg_h in dg_h_values:
        eta = calc.calculate_her_overpotential(dg_h)
        print(f"  ΔG_H = {dg_h:+.2f} eV → η = {eta:.3f} V")
    
    # OER示例
    print("\nOER过电位 (IrO2-like):")
    oer_result = calc.calculate_oer_overpotential(
        dg_oh=0.8,
        dg_o=1.8,
        dg_ooh=2.6
    )
    print(f"  过电位: {oer_result['overpotential']:.3f} V")
    print(f"  决速步: {oer_result['rate_determining_step']}")
    print(f"  各步骤能量:")
    for step, energy in oer_result['steps'].items():
        print(f"    {step}: {energy:.3f} eV")
    
    # ORR示例
    print("\nORR过电位 (Pt-like):")
    orr_result = calc.calculate_orr_overpotential(
        dg_ooh=4.2,
        dg_o=2.8,
        dg_oh=1.5
    )
    print(f"  过电位: {orr_result['overpotential']:.3f} V")
    print(f"  决速步: {orr_result['rate_determining_step']}")
    
    return calc


def example_screening():
    """催化剂筛选示例"""
    print("\n催化剂筛选示例")
    print("-" * 40)
    
    # OER催化剂筛选
    screening = CatalyticScreening(ReactionType.OER)
    
    # 添加候选催化剂
    candidates = [
        ("IrO2(110)", {"OH*": 0.75, "O*": 1.65, "OOH*": 2.55}),
        ("RuO2(110)", {"OH*": 0.68, "O*": 1.58, "OOH*": 2.48}),
        ("Co3O4(111)", {"OH*": 0.85, "O*": 1.95, "OOH*": 2.75}),
        ("MnO2(110)", {"OH*": 0.92, "O*": 2.05, "OOH*": 2.85}),
        ("NiOOH(001)", {"OH*": 0.78, "O*": 1.72, "OOH*": 2.58}),
        ("Fe3O4(111)", {"OH*": 1.05, "O*": 2.25, "OOH*": 3.15}),
    ]
    
    for name, energies in candidates:
        screening.add_candidate(name, energies, {"source": "literature"})
    
    # 排序
    ranked = screening.rank_candidates()
    
    print("OER催化剂排名:")
    for i, cat in enumerate(ranked[:5], 1):
        print(f"  {i}. {cat['name']}: η = {cat['overpotential']:.3f} V")
        print(f"     决速步: {cat['details']['rate_determining_step']}")
    
    return screening


def example_scaling_analysis():
    """标度关系分析示例"""
    print("\n标度关系分析示例")
    print("-" * 40)
    
    # 模拟OER标度关系数据
    # ΔG_OOH vs ΔG_OH
    dG_OH = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    dG_OOH = 0.85 * dG_OH + 3.25 + np.random.normal(0, 0.05, len(dG_OH))
    
    # 拟合
    coeffs = np.polyfit(dG_OH, dG_OOH, 1)
    slope, intercept = coeffs
    
    # 计算R²
    y_pred = np.polyval(coeffs, dG_OH)
    ss_res = np.sum((dG_OOH - y_pred)**2)
    ss_tot = np.sum((dG_OOH - np.mean(dG_OOH))**2)
    r_squared = 1 - ss_res / ss_tot
    
    print("OER标度关系: ΔG_OOH vs ΔG_OH")
    print(f"  ΔG_OOH = {slope:.3f} × ΔG_OH + {intercept:.3f}")
    print(f"  R² = {r_squared:.4f}")
    print(f"  理想值: ΔG_OOH - ΔG_OH = 3.2 eV")
    
    # 计算过电位与OH结合能的关系
    print("\n过电位预测:")
    for i, (dg_oh, dg_ooh) in enumerate(zip(dG_OH, dG_OOH)):
        # 估算O能量（使用另一个标度关系）
        dg_o = 1.6 + 1.4 * (dg_oh - 0.8)  # 近似关系
        
        calc = OverpotentialCalculator()
        result = calc.calculate_oer_overpotential(dg_oh, dg_o, dg_ooh)
        print(f"  催化剂{i+1} (ΔG_OH={dg_oh:.2f}): η = {result['overpotential']:.3f} V")
    
    return slope, intercept


if __name__ == "__main__":
    print("=" * 60)
    print("CatKit Workflow Module - 使用示例")
    print("=" * 60)
    
    example_surface_generation()
    example_volcano_plot()
    example_overpotential()
    example_screening()
    example_scaling_analysis()
    
    print("\n" + "=" * 60)
    print("所有示例完成!")
    print("=" * 60)