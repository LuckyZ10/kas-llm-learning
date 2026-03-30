#!/usr/bin/env python3
"""
Battery Screening Pipeline - 固态电解质高通量筛选管道
========================================================
针对Li/Na离子导体的高通量计算筛选工作流

功能特性:
1. 自动从Materials Project获取候选结构
2. 特征工程: Matminer结构描述符 + Dscribe SOAP描述符
3. 多层级筛选: DFT初筛 → ML势加速MD深度采样
4. 性能预测: 形成能、离子电导率、稳定性排名
5. 针对应用: 固态电解质(Li/Na离子导体)

作者: AI Assistant
版本: 1.0.0
"""

import os
import json
import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('battery_screening.log')
    ]
)
logger = logging.getLogger(__name__)

# Materials Project
from mp_api.client import MPRester

# Pymatgen
from pymatgen.core import Structure, Composition, Element, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.diffusion.neb.pathfinder import DistinctPathFinder

# ASE
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS, FIRE
from ase.units import fs, kB, GPa

# Matminer
from matminer.featurizers.structure import (
    SiteStatsFingerprint,
    StructuralHeterogeneity,
    ChemicalOrdering,
    DensityFeatures,
    RadialDistributionFunction,
    ElectronicRadialDistributionFunction,
    GlobalSymmetryFeatures,
)
from matminer.featurizers.composition import (
    ElementProperty,
    OxidationStates,
)

# Dscribe (SOAP描述符)
try:
    from dscribe.descriptors import SOAP
    from dscribe.kernels import REMatchKernel, AverageKernel
    DSCRIBE_AVAILABLE = True
except ImportError:
    DSCRIBE_AVAILABLE = False
    logger.warning("Dscribe not available. SOAP descriptors will be skipped.")

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# 导入现有模块
import sys
sys.path.insert(0, '/root/.openclaw/workspace/dft_lammps_research/code_templates')
from dft_workflow import DFTConfig, StructureOptimizer
from md_simulation_lammps import LAMMPSConfig, MDSimulationWorkflow, MDAnalyzer
from ml_potential_training import DeepMDConfig, DeepMDTrainer, DataPreprocessor


# ==============================================================================
# 配置类
# ==============================================================================

@dataclass
class BatteryScreeningConfig:
    """电池筛选配置"""
    
    # API设置
    mp_api_key: Optional[str] = None
    
    # 筛选目标
    target_ion: str = "Li"  # "Li", "Na", "K", "Mg", "Ca"
    max_elements: int = 4
    
    # 结构筛选条件
    max_natoms: int = 100
    min_gap: float = 2.0  # eV, 最小带隙
    max_ehull: float = 0.1  # eV/atom, 最大能量高于凸包
    
    # 计算设置
    dft_code: str = "vasp"
    ml_potential: Optional[str] = None
    
    # 模拟参数
    md_temperatures: List[float] = field(default_factory=lambda: [300, 500, 700, 900])
    md_timestep: float = 1.0  # fs
    md_nsteps_equil: int = 50000
    md_nsteps_prod: int = 500000
    
    # 工作流设置
    max_parallel: int = 10
    use_ml_acceleration: bool = True
    
    # 路径设置
    work_dir: str = "./battery_screening"
    db_path: str = "./battery_screening/db"
    dft_results_path: str = "./battery_screening/dft_results"
    md_results_path: str = "./battery_screening/md_results"
    models_path: str = "./battery_screening/models"
    
    # 筛选阈值
    min_ionic_conductivity: float = 1e-5  # S/cm
    max_migration_barrier: float = 0.5  # eV
    min_bulk_modulus: float = 10.0  # GPa
    
    def __post_init__(self):
        # 创建工作目录
        for path in [self.work_dir, self.db_path, self.dft_results_path, 
                     self.md_results_path, self.models_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # 获取API密钥
        if self.mp_api_key is None:
            self.mp_api_key = os.environ.get("MP_API_KEY")


@dataclass
class ScreeningCriteria:
    """筛选标准"""
    
    # 元素组成
    target_ion: str = "Li"
    allowed_anions: List[str] = field(default_factory=lambda: ["O", "S", "Se", "F", "Cl", "N"])
    allowed_cations: List[str] = field(default_factory=lambda: ["P", "Si", "Ge", "Al", "B", "Zr", "Ti"])
    
    # 结构约束
    max_natoms: int = 100
    min_gap: float = 2.0
    max_ehull: float = 0.1
    
    # 性能要求
    min_ionic_conductivity: Optional[float] = None
    max_migration_barrier: Optional[float] = None
    min_bulk_modulus: Optional[float] = None


@dataclass
class CandidateMaterial:
    """候选材料数据结构"""
    
    material_id: str
    formula: str
    structure: Structure
    source: str = "materials_project"  # materials_project, generated, modified
    
    # 基本性质
    band_gap: Optional[float] = None
    energy_above_hull: Optional[float] = None
    formation_energy: Optional[float] = None
    
    # 结构特征
    features: Dict[str, float] = field(default_factory=dict)
    soap_features: Optional[np.ndarray] = None
    
    # 计算结果
    dft_formation_energy: Optional[float] = None
    dft_bulk_modulus: Optional[float] = None
    migration_barrier: Optional[float] = None
    
    # MD结果
    diffusion_coefficient: Optional[float] = None
    ionic_conductivity: Optional[float] = None
    activation_energy: Optional[float] = None
    
    # 排名
    score: Optional[float] = None
    rank: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        # 处理结构对象
        data['structure'] = self.structure.as_dict()
        # 处理numpy数组
        if self.soap_features is not None:
            data['soap_features'] = self.soap_features.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CandidateMaterial':
        """从字典创建"""
        data['structure'] = Structure.from_dict(data['structure'])
        if 'soap_features' in data and data['soap_features'] is not None:
            data['soap_features'] = np.array(data['soap_features'])
        return cls(**data)


# ==============================================================================
# 特征工程模块
# ==============================================================================

class FeatureEngineer:
    """材料特征工程 - Matminer + Dscribe SOAP"""
    
    def __init__(self, config: BatteryScreeningConfig):
        self.config = config
        self._init_featurizers()
        
    def _init_featurizers(self):
        """初始化特征提取器"""
        # 结构特征
        self.structure_featurizers = [
            ('density', DensityFeatures()),
            ('struct_hetero', StructuralHeterogeneity()),
            ('chem_order', ChemicalOrdering()),
            ('global_sym', GlobalSymmetryFeatures()),
        ]
        
        # 组分特征
        self.composition_featurizers = [
            ('element_prop', ElementProperty.from_preset('magpie')),
            ('oxidation', OxidationStates()),
        ]
        
        # SOAP描述符
        self.soap_descriptor = None
        if DSCRIBE_AVAILABLE:
            self.soap_descriptor = None  # 延迟初始化，需要知道元素种类
    
    def featurize_structure(self, structure: Structure) -> Dict[str, float]:
        """提取结构特征 (Matminer)"""
        features = {}
        
        # 基本结构信息
        features['num_atoms'] = len(structure)
        features['volume'] = structure.volume
        features['volume_per_atom'] = structure.volume / len(structure)
        features['density'] = structure.density
        
        # 空间群信息
        try:
            spa = SpacegroupAnalyzer(structure)
            features['spacegroup_number'] = spa.get_space_group_number()
            features['crystal_system'] = spa.get_crystal_system()
            features['is_centrosymmetric'] = 1 if spa.is_laue() else 0
        except:
            features['spacegroup_number'] = 1
            features['crystal_system'] = 'triclinic'
            features['is_centrosymmetric'] = 0
        
        # Matminer特征
        for name, featurizer in self.structure_featurizers:
            try:
                feat = featurizer.featurize(structure)
                feat_names = featurizer.feature_labels()
                for fn, fv in zip(feat_names, feat):
                    features[f"{name}_{fn}"] = fv
            except Exception as e:
                logger.debug(f"Featurization failed for {name}: {e}")
        
        # 组分特征
        comp_features = self.featurize_composition(structure.composition)
        features.update(comp_features)
        
        return features
    
    def featurize_composition(self, composition: Composition) -> Dict[str, float]:
        """提取组分特征"""
        features = {}
        
        for name, featurizer in self.composition_featurizers:
            try:
                feat = featurizer.featurize(composition)
                feat_names = featurizer.feature_labels()
                for fn, fv in zip(feat_names, feat):
                    features[f"{name}_{fn}"] = fv
            except Exception as e:
                logger.debug(f"Featurization failed for {name}: {e}")
        
        # 目标离子分数
        target_element = Element(self.config.target_ion)
        if target_element in composition.elements:
            features[f'{self.config.target_ion}_fraction'] = composition.get_atomic_fraction(target_element)
        else:
            features[f'{self.config.target_ion}_fraction'] = 0.0
        
        return features
    
    def compute_soap_descriptor(self, 
                                 structure: Structure,
                                 species: Optional[List[str]] = None,
                                 rcut: float = 6.0,
                                 nmax: int = 8,
                                 lmax: int = 6) -> np.ndarray:
        """计算SOAP描述符 (Dscribe)"""
        if not DSCRIBE_AVAILABLE:
            logger.warning("Dscribe not available, skipping SOAP descriptor")
            return np.array([])
        
        if species is None:
            species = list(set([str(e) for e in structure.composition.elements]))
        
        # 初始化SOAP
        soap = SOAP(
            species=species,
            rcut=rcut,
            nmax=nmax,
            lmax=lmax,
            average="outer",
            periodic=True,
            rbf="gto",
        )
        
        # 转换为ASE atoms
        ase_adaptor = AseAtomsAdaptor()
        atoms = ase_adaptor.get_atoms(structure)
        
        # 计算SOAP
        soap_features = soap.create(atoms)
        
        return soap_features
    
    def featurize_batch(self, 
                       structures: List[Structure],
                       use_soap: bool = True) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """批量特征提取"""
        features_list = []
        soap_features_list = []
        
        logger.info(f"Featurizing {len(structures)} structures...")
        
        for i, structure in enumerate(structures):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(structures)} structures")
            
            # Matminer特征
            features = self.featurize_structure(structure)
            features_list.append(features)
            
            # SOAP特征
            if use_soap and DSCRIBE_AVAILABLE:
                try:
                    soap_feat = self.compute_soap_descriptor(structure)
                    soap_features_list.append(soap_feat)
                except Exception as e:
                    logger.debug(f"SOAP computation failed: {e}")
                    soap_features_list.append(np.array([]))
        
        # 转换为DataFrame
        features_df = pd.DataFrame(features_list)
        
        # SOAP特征数组
        soap_array = None
        if use_soap and soap_features_list:
            # 找到最大长度
            max_len = max(len(f) for f in soap_features_list) if soap_features_list else 0
            if max_len > 0:
                soap_array = np.array([np.pad(f, (0, max_len - len(f)), 'constant') 
                                       for f in soap_features_list])
        
        return features_df, soap_array


# ==============================================================================
# Materials Project接口
# ==============================================================================

class MaterialsProjectInterface:
    """Materials Project接口 - 固态电解质专用查询"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("MP_API_KEY")
        if not self.api_key:
            raise ValueError("MP API key not provided. Set MP_API_KEY environment variable.")
        self.mpr = MPRester(self.api_key)
        
    def query_solid_electrolytes(self,
                                 criteria: ScreeningCriteria,
                                 max_entries: int = 1000) -> List[CandidateMaterial]:
        """查询固态电解质候选材料"""
        
        logger.info(f"Querying Materials Project for {criteria.target_ion} conductors...")
        
        # 构建查询条件
        mp_filters = {}
        
        # 必须包含目标离子
        target_elements = [criteria.target_ion]
        if criteria.allowed_anions:
            target_elements.extend(criteria.allowed_anions[:3])  # 限制以提高查询效率
        
        mp_filters['elements'] = target_elements
        mp_filters['nelements'] = {'$lte': criteria.max_elements}
        
        # 带隙筛选
        if criteria.min_gap is not None:
            mp_filters['band_gap'] = {'$gte': criteria.min_gap}
        
        # 稳定性筛选
        if criteria.max_ehull is not None:
            mp_filters['energy_above_hull'] = {'$lte': criteria.max_ehull}
        
        try:
            docs = self.mpr.summary.search(
                **mp_filters,
                fields=["material_id", "formula_pretty", "structure", 
                       "band_gap", "energy_above_hull", "formation_energy_per_atom",
                       "volume", "density", "symmetry"],
                num_chunks=5,
                chunk_size=min(max_entries, 1000)
            )
        except Exception as e:
            logger.error(f"MP query failed: {e}")
            return []
        
        # 转换为候选材料对象
        candidates = []
        for doc in docs:
            try:
                # 筛选原子数
                if len(doc.structure) > criteria.max_natoms:
                    continue
                
                # 筛选是否含有足够的导电离子
                comp = doc.structure.composition
                target_el = Element(criteria.target_ion)
                if target_el not in comp.elements:
                    continue
                
                # 创建候选对象
                candidate = CandidateMaterial(
                    material_id=doc.material_id,
                    formula=doc.formula_pretty,
                    structure=doc.structure,
                    source="materials_project",
                    band_gap=doc.band_gap,
                    energy_above_hull=doc.energy_above_hull,
                    formation_energy=doc.formation_energy_per_atom if hasattr(doc, 'formation_energy_per_atom') else None
                )
                candidates.append(candidate)
                
            except Exception as e:
                logger.debug(f"Failed to process {doc.material_id}: {e}")
        
        logger.info(f"Found {len(candidates)} candidate materials from Materials Project")
        return candidates
    
    def query_by_chemsys(self,
                        elements: List[str],
                        max_entries: int = 100) -> List[CandidateMaterial]:
        """按化学系统查询"""
        
        logger.info(f"Querying chemical system: {'-'.join(elements)}")
        
        try:
            docs = self.mpr.summary.search(
                chemsys='-'.join(elements),
                fields=["material_id", "formula_pretty", "structure",
                       "band_gap", "energy_above_hull", "formation_energy_per_atom"],
                num_chunks=1,
                chunk_size=max_entries
            )
        except Exception as e:
            logger.error(f"MP query failed: {e}")
            return []
        
        candidates = []
        for doc in docs:
            candidate = CandidateMaterial(
                material_id=doc.material_id,
                formula=doc.formula_pretty,
                structure=doc.structure,
                source="materials_project",
                band_gap=doc.band_gap,
                energy_above_hull=doc.energy_above_hull,
                formation_energy=doc.formation_energy_per_atom if hasattr(doc, 'formation_energy_per_atom') else None
            )
            candidates.append(candidate)
        
        return candidates


# ==============================================================================
# 结构生成器 - 生成候选变体
# ==============================================================================

class StructureGenerator:
    """结构生成器 - 生成替代和缺陷结构"""
    
    def __init__(self, config: BatteryScreeningConfig):
        self.config = config
    
    def generate_substitution_series(self,
                                    base_structure: Structure,
                                    target_ion: str,
                                    substitutes: List[str],
                                    max_substitutions: int = 1) -> List[CandidateMaterial]:
        """生成离子替换系列"""
        
        candidates = []
        base_formula = base_structure.formula
        
        for sub in substitutes:
            try:
                new_struct = base_structure.copy()
                new_struct.replace_species({target_ion: sub})
                
                candidate = CandidateMaterial(
                    material_id=f"subst_{base_formula.replace(' ', '')}_{target_ion}2{sub}",
                    formula=new_struct.formula,
                    structure=new_struct,
                    source="substitution",
                )
                candidates.append(candidate)
            except Exception as e:
                logger.debug(f"Substitution failed: {e}")
        
        return candidates
    
    def generate_doped_structures(self,
                                 base_structure: Structure,
                                 dopant: str,
                                 concentration_range: List[float] = [0.05, 0.1, 0.15]) -> List[CandidateMaterial]:
        """生成掺杂结构"""
        
        candidates = []
        
        for conc in concentration_range:
            try:
                # 简化实现：随机替换
                n_dopant = max(1, int(len(base_structure) * conc))
                new_struct = base_structure.copy()
                
                # 这里可以实现更复杂的掺杂逻辑
                # 例如：替换特定位置，保持电荷平衡等
                
                candidate = CandidateMaterial(
                    material_id=f"doped_{base_structure.formula.replace(' ', '')}_{dopant}{conc}",
                    formula=new_struct.formula,
                    structure=new_struct,
                    source="doped",
                )
                candidates.append(candidate)
            except Exception as e:
                logger.debug(f"Doping failed: {e}")
        
        return candidates


# ==============================================================================
# DFT计算模块 - 初筛
# ==============================================================================

class DFTScreeningCalculator:
    """DFT筛选计算器"""
    
    def __init__(self, config: BatteryScreeningConfig):
        self.config = config
        self.dft_config = DFTConfig(
            code=config.dft_code,
            functional="PBE",
            encut=520,
            ncores=32
        )
    
    def screen_candidate(self, candidate: CandidateMaterial) -> CandidateMaterial:
        """对单个候选进行DFT初筛"""
        
        logger.info(f"DFT screening: {candidate.material_id}")
        
        # 转换为ASE atoms
        ase_adaptor = AseAtomsAdaptor()
        atoms = ase_adaptor.get_atoms(candidate.structure)
        
        # 创建优化器
        optimizer = StructureOptimizer(self.dft_config)
        optimizer.atoms = atoms
        
        try:
            # 结构优化
            optimized = optimizer.relax_structure(fmax=0.05)
            
            # 记录结果
            candidate.dft_formation_energy = optimized.get_potential_energy() / len(optimized)
            
            # 保存优化后的结构
            output_dir = Path(self.config.dft_results_path) / candidate.material_id
            output_dir.mkdir(parents=True, exist_ok=True)
            write(output_dir / "CONTCAR", optimized)
            
            logger.info(f"  Formation energy: {candidate.dft_formation_energy:.4f} eV/atom")
            
        except Exception as e:
            logger.error(f"DFT screening failed for {candidate.material_id}: {e}")
        
        return candidate
    
    def screen_batch(self, 
                    candidates: List[CandidateMaterial],
                    n_parallel: int = 5) -> List[CandidateMaterial]:
        """批量DFT筛选"""
        
        logger.info(f"Starting batch DFT screening for {len(candidates)} candidates")
        
        results = []
        for candidate in candidates:
            result = self.screen_candidate(candidate)
            results.append(result)
        
        return results


# ==============================================================================
# ML势加速MD模块 - 深度采样
# ==============================================================================

class MLAcceleratedSampler:
    """ML势加速采样器"""
    
    def __init__(self, config: BatteryScreeningConfig, ml_potential_path: Optional[str] = None):
        self.config = config
        self.ml_potential_path = ml_potential_path
        
    def setup_ml_potential(self, training_structures: List[Structure]):
        """从DFT结果训练ML势"""
        
        logger.info("Training ML potential from DFT results...")
        
        # 收集DFT计算数据
        vasp_dirs = []
        for struct_path in Path(self.config.dft_results_path).glob("*/"):
            if (struct_path / "OUTCAR").exists():
                vasp_dirs.append(str(struct_path))
        
        if not vasp_dirs:
            logger.warning("No DFT data found for training")
            return None
        
        # 确定元素类型
        all_elements = set()
        for s in training_structures:
            all_elements.update([str(e) for e in s.composition.elements])
        type_map = sorted(list(all_elements))
        
        # 配置DeepMD
        deepmd_config = DeepMDConfig(
            type_map=type_map,
            descriptor_type="se_e2_a",
            rcut=6.0,
            numb_steps=1000000,
            output_dir=self.config.models_path
        )
        
        # 数据预处理
        preprocessor = DataPreprocessor(type_map)
        train_dir, valid_dir = preprocessor.convert_vasp_to_deepmd(
            vasp_dirs=vasp_dirs,
            output_dir=f"{self.config.models_path}/training_data",
            train_ratio=0.9
        )
        
        deepmd_config.training_data = train_dir
        deepmd_config.validation_data = valid_dir
        
        # 训练
        trainer = DeepMDTrainer(deepmd_config)
        model_path = trainer.train()
        
        # 冻结模型
        frozen_model = trainer.freeze_model(self.config.models_path)
        self.ml_potential_path = frozen_model
        
        logger.info(f"ML potential trained: {frozen_model}")
        
        return frozen_model
    
    def run_deep_sampling(self, 
                         candidate: CandidateMaterial,
                         temperatures: Optional[List[float]] = None) -> CandidateMaterial:
        """对候选进行ML-MD深度采样"""
        
        if self.ml_potential_path is None:
            logger.warning("No ML potential available for sampling")
            return candidate
        
        if temperatures is None:
            temperatures = self.config.md_temperatures
        
        logger.info(f"ML-MD sampling: {candidate.material_id}")
        
        # 准备LAMMPS配置
        lammps_config = LAMMPSConfig(
            pair_style="deepmd",
            potential_file=self.ml_potential_path,
            ensemble="nvt",
            temperature=temperatures[0],
            timestep=self.config.md_timestep,
            nsteps=self.config.md_nsteps_equil + self.config.md_nsteps_prod,
            dump_interval=100,
            nprocs=4,
            working_dir=f"{self.config.md_results_path}/{candidate.material_id}"
        )
        
        # 转换为ASE atoms
        ase_adaptor = AseAtomsAdaptor()
        atoms = ase_adaptor.get_atoms(candidate.structure)
        
        # 运行MD
        workflow = MDSimulationWorkflow(lammps_config)
        
        try:
            # 平衡化
            equilibrated = workflow.equilibration_run(
                atoms,
                nvt_steps=self.config.md_nsteps_equil,
                npt_steps=0
            )
            
            # 生产模拟
            trajectory = workflow.production_run(
                equilibrated,
                nsteps=self.config.md_nsteps_prod
            )
            
            # 分析
            analyzer = MDAnalyzer(trajectory)
            report = analyzer.generate_report()
            
            # 提取扩散系数
            D = analyzer.compute_diffusion_coefficient(
                atom_type=self.config.target_ion,
                timestep=self.config.md_timestep
            )
            
            candidate.diffusion_coefficient = D
            
            # 计算离子电导率 (Nernst-Einstein)
            temperature = temperatures[0]
            conductivity = self._compute_conductivity(
                candidate.structure, D, temperature
            )
            candidate.ionic_conductivity = conductivity
            
            logger.info(f"  Diffusion D: {D:.2e} cm²/s")
            logger.info(f"  Conductivity σ: {conductivity:.2e} S/cm")
            
        except Exception as e:
            logger.error(f"ML-MD sampling failed: {e}")
        
        return candidate
    
    def _compute_conductivity(self, 
                             structure: Structure,
                             D: float,
                             temperature: float) -> float:
        """使用Nernst-Einstein方程计算离子电导率"""
        
        # 目标离子数密度
        target_el = Element(self.config.target_ion)
        n_ion = structure.composition[structure.composition.reduced_formula].get_atomic_fraction(target_el)
        volume_cm3 = structure.volume * 1e-24  # Å³ -> cm³
        n_density = n_ion / volume_cm3
        
        # 电荷
        q = 1.0 * 1.602e-19  # C (假设+1价)
        
        # kB
        kB = 1.380649e-23  # J/K
        
        # Nernst-Einstein: σ = n * q² * D / (kB * T)
        sigma = n_density * q**2 * D * 1e-4 / (kB * temperature)  # S/cm
        
        return sigma
    
    def compute_activation_energy(self,
                                 candidate: CandidateMaterial,
                                 temperatures: List[float]) -> float:
        """从多温度MD计算活化能 (Arrhenius拟合)"""
        
        if len(temperatures) < 2:
            logger.warning("Need at least 2 temperatures for Arrhenius fit")
            return 0.0
        
        diffusion_data = []
        
        for T in temperatures:
            # 在该温度运行MD
            lammps_config = LAMMPSConfig(
                pair_style="deepmd",
                potential_file=self.ml_potential_path,
                ensemble="nvt",
                temperature=T,
                timestep=self.config.md_timestep,
                nsteps=100000,
                working_dir=f"{self.config.md_results_path}/{candidate.material_id}_T{T}"
            )
            
            workflow = MDSimulationWorkflow(lammps_config)
            ase_adaptor = AseAtomsAdaptor()
            atoms = ase_adaptor.get_atoms(candidate.structure)
            
            try:
                result = workflow.full_workflow(atoms)
                analyzer = MDAnalyzer(result['trajectory'])
                D = analyzer.compute_diffusion_coefficient(
                    atom_type=self.config.target_ion
                )
                diffusion_data.append((T, D))
            except Exception as e:
                logger.error(f"MD at {T}K failed: {e}")
        
        if len(diffusion_data) < 2:
            return 0.0
        
        # Arrhenius拟合: ln(D) = ln(D0) - Ea/(kB*T)
        temps = np.array([d[0] for d in diffusion_data])
        diffs = np.array([d[1] for d in diffusion_data])
        
        kB = 8.617e-5  # eV/K
        ln_D = np.log(diffs)
        inv_T = 1 / temps
        
        # 线性拟合
        slope, intercept = np.polyfit(inv_T, ln_D, 1)
        Ea = -slope * kB  # eV
        
        candidate.activation_energy = Ea
        
        return Ea


# ==============================================================================
# 机器学习预测模块
# ==============================================================================

class PropertyPredictor:
    """性质预测器 - 基于特征和ML模型"""
    
    def __init__(self, config: BatteryScreeningConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.models = {}
    
    def train_conductivity_model(self,
                                 features: pd.DataFrame,
                                 conductivities: np.ndarray,
                                 model_type: str = "gbr") -> Any:
        """训练电导率预测模型"""
        
        logger.info(f"Training {model_type} conductivity model...")
        
        # 处理缺失值
        features_clean = features.fillna(features.mean())
        
        # 标准化
        X_scaled = self.scaler.fit_transform(features_clean)
        
        # 选择模型
        if model_type == "rf":
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                n_jobs=-1,
                random_state=42
            )
        elif model_type == "gbr":
            model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 训练
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, np.log10(conductivities), test_size=0.2, random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # 评估
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Model performance: RMSE={rmse:.4f}, R²={r2:.4f}")
        
        self.models['conductivity'] = model
        
        # 特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = pd.Series(
                model.feature_importances_,
                index=features_clean.columns
            ).sort_values(ascending=False)
            logger.info("Top 10 important features:")
            logger.info(importances.head(10))
        
        return model
    
    def predict_conductivity(self, features: pd.DataFrame) -> np.ndarray:
        """预测电导率"""
        
        if 'conductivity' not in self.models:
            logger.warning("No conductivity model available")
            return np.full(len(features), np.nan)
        
        features_clean = features.fillna(features.mean())
        X_scaled = self.scaler.transform(features_clean)
        
        log_conductivity = self.models['conductivity'].predict(X_scaled)
        return 10 ** log_conductivity


# ==============================================================================
# 主筛选管道
# ==============================================================================

class BatteryScreeningPipeline:
    """
    固态电解质高通量筛选管道
    
    工作流:
    1. 从Materials Project获取候选结构
    2. 特征工程 (Matminer + SOAP)
    3. DFT初筛 (形成能计算)
    4. ML势训练 (基于DFT数据)
    5. ML-MD深度采样 (扩散系数、电导率)
    6. 综合排名输出
    """
    
    def __init__(self, config: BatteryScreeningConfig):
        self.config = config
        self.mp_interface = MaterialsProjectInterface(config.mp_api_key)
        self.feature_engineer = FeatureEngineer(config)
        self.structure_generator = StructureGenerator(config)
        self.dft_calculator = DFTScreeningCalculator(config)
        self.ml_sampler = None
        self.predictor = PropertyPredictor(config)
        
        self.candidates: List[CandidateMaterial] = []
        self.screened_candidates: List[CandidateMaterial] = []
        self.final_candidates: List[CandidateMaterial] = []
    
    def fetch_candidates(self, 
                        criteria: ScreeningCriteria,
                        max_entries: int = 500,
                        generate_variants: bool = True) -> List[CandidateMaterial]:
        """获取候选材料"""
        
        logger.info("="*60)
        logger.info("STEP 1: Fetching candidate structures from Materials Project")
        logger.info("="*60)
        
        # 从MP查询
        candidates = self.mp_interface.query_solid_electrolytes(criteria, max_entries)
        
        # 生成变体结构
        if generate_variants and len(candidates) > 0:
            logger.info("Generating structural variants...")
            variants = []
            
            # 对前10个结构生成替换变体
            for base in candidates[:10]:
                if criteria.target_ion == "Li":
                    substitutes = ["Na", "K"]
                elif criteria.target_ion == "Na":
                    substitutes = ["Li", "K"]
                else:
                    substitutes = []
                
                subs = self.structure_generator.generate_substitution_series(
                    base.structure, criteria.target_ion, substitutes
                )
                variants.extend(subs)
            
            candidates.extend(variants)
        
        self.candidates = candidates
        logger.info(f"Total candidates: {len(candidates)}")
        
        return candidates
    
    def compute_features(self, use_soap: bool = True) -> pd.DataFrame:
        """计算结构特征"""
        
        logger.info("="*60)
        logger.info("STEP 2: Feature Engineering (Matminer + SOAP)")
        logger.info("="*60)
        
        structures = [c.structure for c in self.candidates]
        
        # 计算特征
        features_df, soap_array = self.feature_engineer.featurize_batch(
            structures, use_soap=use_soap
        )
        
        # 保存特征
        features_df.to_csv(f"{self.config.work_dir}/features.csv", index=False)
        
        # 更新候选对象
        for i, candidate in enumerate(self.candidates):
            candidate.features = features_df.iloc[i].to_dict()
            if soap_array is not None:
                candidate.soap_features = soap_array[i]
        
        logger.info(f"Computed {len(features_df.columns)} features per structure")
        
        return features_df
    
    def run_dft_screening(self, 
                         n_candidates: int = 20,
                         filter_by_stability: bool = True) -> List[CandidateMaterial]:
        """DFT初筛"""
        
        logger.info("="*60)
        logger.info("STEP 3: DFT Initial Screening")
        logger.info("="*60)
        
        # 选择候选进行DFT计算
        # 优先选择：高对称性、低能量凸包、合适带隙
        candidates_to_screen = []
        
        # 基于MP数据的预筛选
        for c in self.candidates:
            score = 0
            if c.band_gap and c.band_gap > self.config.min_gap:
                score += 1
            if c.energy_above_hull is not None and c.energy_above_hull < self.config.max_ehull:
                score += 1
            if c.features.get(f'{self.config.target_ion}_fraction', 0) > 0.1:
                score += 1
            
            if score >= 2:
                candidates_to_screen.append((score, c))
        
        # 按分数排序，选择前N个
        candidates_to_screen.sort(key=lambda x: x[0], reverse=True)
        selected = [c for _, c in candidates_to_screen[:n_candidates]]
        
        logger.info(f"Selected {len(selected)} candidates for DFT screening")
        
        # DFT计算
        screened = self.dft_calculator.screen_batch(selected)
        
        # 稳定性筛选
        if filter_by_stability:
            screened = [c for c in screened 
                       if c.dft_formation_energy is not None 
                       and c.dft_formation_energy < 0.5]
        
        self.screened_candidates = screened
        logger.info(f"DFT screening complete. {len(screened)} candidates passed.")
        
        return screened
    
    def train_ml_potential(self) -> Optional[str]:
        """训练ML势"""
        
        logger.info("="*60)
        logger.info("STEP 4: Training ML Potential")
        logger.info("="*60)
        
        if not self.config.use_ml_acceleration:
            logger.info("ML acceleration disabled")
            return None
        
        structures = [c.structure for c in self.screened_candidates]
        
        self.ml_sampler = MLAcceleratedSampler(self.config)
        model_path = self.ml_sampler.setup_ml_potential(structures)
        
        return model_path
    
    def run_ml_sampling(self) -> List[CandidateMaterial]:
        """ML-MD深度采样"""
        
        logger.info("="*60)
        logger.info("STEP 5: ML-MD Deep Sampling")
        logger.info("="*60)
        
        if self.ml_sampler is None or self.ml_sampler.ml_potential_path is None:
            logger.warning("No ML potential available. Skipping ML sampling.")
            return self.screened_candidates
        
        final_candidates = []
        
        for candidate in self.screened_candidates:
            sampled = self.ml_sampler.run_deep_sampling(candidate)
            final_candidates.append(sampled)
        
        self.final_candidates = final_candidates
        logger.info(f"ML sampling complete for {len(final_candidates)} candidates")
        
        return final_candidates
    
    def rank_candidates(self) -> pd.DataFrame:
        """对候选材料进行排名"""
        
        logger.info("="*60)
        logger.info("STEP 6: Ranking and Final Evaluation")
        logger.info("="*60)
        
        # 计算综合得分
        for candidate in self.final_candidates:
            score = 0
            
            # 离子电导率权重: 0.5
            if candidate.ionic_conductivity:
                # log scale scoring
                log_cond = np.log10(candidate.ionic_conductivity + 1e-10)
                score += 0.5 * (log_cond + 10) / 10  # 归一化到0-1
            
            # 稳定性权重: 0.3
            if candidate.dft_formation_energy:
                stability_score = max(0, 1 - abs(candidate.dft_formation_energy))
                score += 0.3 * stability_score
            
            # 带隙权重: 0.2 (电子绝缘性)
            if candidate.band_gap:
                gap_score = min(1, candidate.band_gap / 5.0)
                score += 0.2 * gap_score
            
            candidate.score = score
        
        # 按分数排序
        ranked = sorted(self.final_candidates, key=lambda x: x.score or 0, reverse=True)
        
        for i, c in enumerate(ranked):
            c.rank = i + 1
        
        # 转换为DataFrame
        data = []
        for c in ranked:
            row = {
                'rank': c.rank,
                'material_id': c.material_id,
                'formula': c.formula,
                'source': c.source,
                'band_gap_eV': c.band_gap,
                'energy_above_hull_eV': c.energy_above_hull,
                'dft_formation_energy_eV': c.dft_formation_energy,
                'diffusion_coefficient_cm2s': c.diffusion_coefficient,
                'ionic_conductivity_S_cm': c.ionic_conductivity,
                'activation_energy_eV': c.activation_energy,
                'score': c.score,
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{self.config.work_dir}/ranking_results_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to: {output_file}")
        
        # 保存完整候选数据
        candidates_dict = [c.to_dict() for c in ranked]
        with open(f"{self.config.work_dir}/candidates_{timestamp}.json", 'w') as f:
            json.dump(candidates_dict, f, indent=2, default=str)
        
        return df
    
    def run_full_pipeline(self,
                         criteria: ScreeningCriteria,
                         max_candidates: int = 100,
                         n_dft_screen: int = 20) -> pd.DataFrame:
        """运行完整筛选管道"""
        
        logger.info("\n" + "="*60)
        logger.info("BATTERY SCREENING PIPELINE STARTED")
        logger.info(f"Target ion: {criteria.target_ion}")
        logger.info("="*60 + "\n")
        
        # Step 1: 获取候选
        self.fetch_candidates(criteria, max_candidates)
        
        # Step 2: 特征工程
        self.compute_features(use_soap=True)
        
        # Step 3: DFT初筛
        self.run_dft_screening(n_dft_screen)
        
        # Step 4: 训练ML势
        self.train_ml_potential()
        
        # Step 5: ML-MD采样
        self.run_ml_sampling()
        
        # Step 6: 排名输出
        results = self.rank_candidates()
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED")
        logger.info("="*60)
        
        return results


# ==============================================================================
# 实用函数
# ==============================================================================

def load_config_from_yaml(yaml_path: str) -> BatteryScreeningConfig:
    """从YAML加载配置"""
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return BatteryScreeningConfig(**config_dict)


def create_default_config(output_path: str = "screening_config.yaml"):
    """创建默认配置文件"""
    config = BatteryScreeningConfig()
    config_dict = asdict(config)
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    logger.info(f"Default config saved to: {output_path}")


def print_results_summary(results_df: pd.DataFrame, top_n: int = 10):
    """打印结果摘要"""
    
    print("\n" + "="*80)
    print("SOLID ELECTROLYTE SCREENING RESULTS")
    print("="*80)
    
    if len(results_df) == 0:
        print("No candidates found.")
        return
    
    # 打印前N名
    print(f"\nTop {min(top_n, len(results_df))} Candidates:\n")
    
    display_cols = ['rank', 'material_id', 'formula', 'ionic_conductivity_S_cm', 
                   'dft_formation_energy_eV', 'band_gap_eV', 'score']
    
    available_cols = [c for c in display_cols if c in results_df.columns]
    print(results_df[available_cols].head(top_n).to_string(index=False))
    
    print("\n" + "="*80)
    print(f"Total candidates evaluated: {len(results_df)}")
    print("="*80)


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    """示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Battery Screening Pipeline')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--target', type=str, default='Li', choices=['Li', 'Na', 'K'],
                       help='Target conducting ion')
    parser.add_argument('--max-candidates', type=int, default=100,
                       help='Maximum number of candidates to fetch')
    parser.add_argument('--n-dft', type=int, default=10,
                       help='Number of candidates for DFT screening')
    parser.add_argument('--create-config', action='store_true',
                       help='Create default config file and exit')
    
    args = parser.parse_args()
    
    # 创建默认配置
    if args.create_config:
        create_default_config()
        return
    
    # 加载配置
    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        config = BatteryScreeningConfig(target_ion=args.target)
    
    # 创建筛选标准
    criteria = ScreeningCriteria(
        target_ion=args.target,
        allowed_anions=["O", "S", "Se", "F", "Cl", "N"],
        allowed_cations=["P", "Si", "Ge", "Al", "B", "Zr", "Ti"],
        max_natoms=100,
        min_gap=2.0,
        max_ehull=0.1,
    )
    
    # 运行管道
    pipeline = BatteryScreeningPipeline(config)
    results = pipeline.run_full_pipeline(
        criteria=criteria,
        max_candidates=args.max_candidates,
        n_dft_screen=args.n_dft
    )
    
    # 打印结果
    print_results_summary(results)


if __name__ == "__main__":
    main()
