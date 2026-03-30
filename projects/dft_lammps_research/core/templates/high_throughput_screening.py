#!/usr/bin/env python3
"""
高通量材料筛选自动化框架
集成DFT、ML势、MD进行材料性质预测
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

# Materials Project
from mp_api.client import MPRester

# Pymatgen
from pymatgen.core import Structure, Composition, Element
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.analysis.elasticity import ElasticTensor
from pymatgen.ext.matproj import MPRester as PMGMPRester

# ASE
from ase import Atoms
from ase.io import read, write

# FireWorks
from fireworks import Firework, Workflow, LaunchPad
from fireworks.core.rocket_launcher import rapidfire
from fireworks.queue.queue_adapter import CommonAdapter
from fireworks.user_objects.queue_adapters.common_adapter import CommonAdapter

# Atomate
from atomate.vasp.workflows.base.core import get_wf
from atomate.vasp.fireworks.core import OptimizeFW, StaticFW

# AiiDA (可选)
try:
    from aiida import load_profile, orm
    from aiida.engine import submit, run
    AIIDA_AVAILABLE = True
except ImportError:
    AIIDA_AVAILABLE = False

# Matminer
from matminer.featurizers.structure import (
    SiteStatsFingerprint,
    StructuralHeterogeneity,
    ChemicalOrdering,
    DensityFeatures,
    RadialDistributionFunction,
    ElectronicRadialDistributionFunction,
)
from matminer.featurizers.composition import ElementProperty

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScreeningCriteria:
    """筛选标准配置"""
    # 组成约束
    elements_include: Optional[List[str]] = None
    elements_exclude: Optional[List[str]] = None
    max_elements: int = 4
    
    # 结构约束
    max_natoms: int = 100
    min_gap: Optional[float] = None
    max_gap: Optional[float] = None
    
    # 稳定性
    max_ehull: float = 0.1  # eV/atom
    
    # 力学性质
    min_bulk_modulus: Optional[float] = None
    max_bulk_modulus: Optional[float] = None
    
    # 电化学
    min_ionic_conductivity: Optional[float] = None
    min_voltage: Optional[float] = None
    
    # 自定义过滤器
    custom_filters: Optional[List[Callable]] = None


@dataclass
class HighThroughputConfig:
    """高通量配置"""
    # 计算设置
    dft_code: str = "vasp"
    ml_potential: Optional[str] = None
    
    # 工作流管理
    workflow_manager: str = "fireworks"  # fireworks, aiida, local
    
    # 并行设置
    max_parallel: int = 10
    
    # 数据库
    db_type: str = "json"  # json, mongodb, postgres
    db_path: str = "./screening_db"
    
    # API密钥
    mp_api_key: Optional[str] = None


class StructureDatabase:
    """结构数据库管理"""
    
    def __init__(self, config: HighThroughputConfig):
        self.config = config
        self.db_path = Path(config.db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
    def save_structure(self, 
                      structure_id: str,
                      structure: Structure,
                      metadata: Optional[Dict] = None):
        """保存结构到数据库"""
        entry = {
            'structure_id': structure_id,
            'structure': structure.as_dict(),
            'metadata': metadata or {}
        }
        
        file_path = self.db_path / f"{structure_id}.json"
        with open(file_path, 'w') as f:
            json.dump(entry, f, indent=2)
    
    def load_structure(self, structure_id: str) -> Tuple[Structure, Dict]:
        """从数据库加载结构"""
        file_path = self.db_path / f"{structure_id}.json"
        
        with open(file_path, 'r') as f:
            entry = json.load(f)
        
        structure = Structure.from_dict(entry['structure'])
        return structure, entry.get('metadata', {})
    
    def query(self, filters: Dict) -> List[str]:
        """查询数据库"""
        matching_ids = []
        
        for json_file in self.db_path.glob("*.json"):
            with open(json_file, 'r') as f:
                entry = json.load(f)
            
            # 简单过滤器检查
            match = True
            for key, value in filters.items():
                if key in entry['metadata']:
                    if entry['metadata'][key] != value:
                        match = False
                        break
            
            if match:
                matching_ids.append(entry['structure_id'])
        
        return matching_ids
    
    def export_to_csv(self, output_file: str):
        """导出所有数据到CSV"""
        data = []
        
        for json_file in self.db_path.glob("*.json"):
            with open(json_file, 'r') as f:
                entry = json.load(f)
            
            row = {'structure_id': entry['structure_id']}
            row.update(entry['metadata'])
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        return df


class MaterialsProjectInterface:
    """Materials Project接口"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("MP_API_KEY")
        self.mpr = MPRester(self.api_key) if self.api_key else None
        
    def query_materials(self, 
                       criteria: ScreeningCriteria,
                       max_entries: int = 1000) -> List[Dict]:
        """从Materials Project查询材料"""
        if self.mpr is None:
            raise ValueError("MP API key not provided")
        
        logger.info(f"Querying Materials Project...")
        
        # 构建查询条件
        mp_filters = {}
        
        if criteria.elements_include:
            mp_filters['elements'] = criteria.elements_include
        
        if criteria.max_elements:
            mp_filters['nelements'] = {'$lte': criteria.max_elements}
        
        if criteria.min_gap is not None or criteria.max_gap is not None:
            gap_filter = {}
            if criteria.min_gap is not None:
                gap_filter['$gte'] = criteria.min_gap
            if criteria.max_gap is not None:
                gap_filter['$lte'] = criteria.max_gap
            mp_filters['band_gap'] = gap_filter
        
        # 查询
        docs = self.mpr.summary.search(
            **mp_filters,
            fields=["material_id", "formula_pretty", "structure", 
                   "band_gap", "energy_above_hull", "formation_energy_per_atom"],
            num_chunks=1,
            chunk_size=max_entries
        )
        
        # 转换为字典
        results = []
        for doc in docs:
            result = {
                'material_id': doc.material_id,
                'formula': doc.formula_pretty,
                'structure': doc.structure,
                'band_gap': doc.band_gap,
                'energy_above_hull': doc.energy_above_hull,
                'formation_energy': doc.formation_energy_per_atom,
            }
            results.append(result)
        
        logger.info(f"Found {len(results)} materials from Materials Project")
        
        return results
    
    def get_elastic_tensor(self, material_id: str) -> Optional[ElasticTensor]:
        """获取弹性张量"""
        try:
            elasticity_doc = self.mpr.elasticity.search(material_ids=[material_id])
            if elasticity_doc:
                return ElasticTensor.from_voigt(elasticity_doc[0].elasticity.elastic_tensor)
        except:
            pass
        return None


class StructureGenerator:
    """结构生成器"""
    
    def __init__(self):
        self.structures = []
    
    def generate_substitutions(self,
                              base_structure: Structure,
                              substitution_dict: Dict[str, List[str]],
                              max_substitutions: int = 1) -> List[Structure]:
        """
        生成替换结构
        
        Args:
            base_structure: 基础结构
            substitution_dict: 替换规则，如 {"Li": ["Na", "K"], "O": ["S", "Se"]}
            max_substitutions: 最大替换数
        """
        structures = [base_structure]
        
        for original, replacements in substitution_dict.items():
            new_structures = []
            
            for struct in structures:
                for replacement in replacements:
                    try:
                        new_struct = struct.copy()
                        new_struct.replace_species({original: replacement})
                        new_structures.append(new_struct)
                    except:
                        pass
            
            structures.extend(new_structures)
        
        return structures
    
    def generate_alloys(self,
                       base_composition: str,
                       substituent: str,
                       concentrations: List[float]) -> List[Composition]:
        """生成合金组分"""
        base = Composition(base_composition)
        compositions = []
        
        for x in concentrations:
            # 简化：均匀替换
            new_comp = base.copy()
            # 这里需要根据具体体系实现合金生成逻辑
            compositions.append(new_comp)
        
        return compositions
    
    def generate_defect_structures(self,
                                  structure: Structure,
                                  defect_type: str = "vacancy",
                                  max_concentration: float = 0.1) -> List[Structure]:
        """生成缺陷结构"""
        defect_structures = []
        
        if defect_type == "vacancy":
            # 生成空位
            for i in range(len(structure)):
                new_struct = structure.copy()
                del new_struct[i]
                defect_structures.append(new_struct)
        
        elif defect_type == "interstitial":
            # 生成间隙 (简化版本)
            pass
        
        return defect_structures
    
    def enumerate_surfaces(self,
                          structure: Structure,
                          miller_indices: List[Tuple[int, int, int]],
                          min_slab_size: float = 10.0,
                          min_vacuum_size: float = 15.0) -> List[Structure]:
        """枚举表面结构"""
        from pymatgen.core.surface import SlabGenerator, get_symmetrically_distinct_miller_indices
        
        surfaces = []
        
        for hkl in miller_indices:
            try:
                slab_gen = SlabGenerator(
                    structure, hkl,
                    min_slab_size=min_slab_size,
                    min_vacuum_size=min_vacuum_size
                )
                slabs = slab_gen.get_slabs()
                surfaces.extend(slabs)
            except:
                pass
        
        return surfaces
    
    def enumerate_adsorption_sites(self,
                                  surface: Structure,
                                  adsorbate: Atoms,
                                  adsorption_height: float = 2.0) -> List[Atoms]:
        """枚举吸附位点"""
        from pymatgen.analysis.adsorption import AdsorbateSiteFinder
        
        # 转换为pymatgen结构
        ase_adaptor = AseAtomsAdaptor()
        pmg_surface = ase_adaptor.get_structure(surface)
        
        # 查找吸附位点
        asf = AdsorbateSiteFinder(pmg_surface)
        ads_sites = asf.find_adsorption_sites()
        
        # 生成吸附结构
        adsorption_structures = []
        for site in ads_sites['all']:
            new_surface = surface.copy()
            # 添加吸附物 (简化版本)
            adsorption_structures.append(new_surface)
        
        return adsorption_structures


class FeatureEngineer:
    """特征工程"""
    
    def __init__(self):
        self.structure_featurizers = [
            ('density', DensityFeatures()),
            ('struct_hetero', StructuralHeterogeneity()),
            ('chem_order', ChemicalOrdering()),
        ]
        
        self.composition_featurizers = [
            ('element_prop', ElementProperty.from_preset('magpie')),
        ]
    
    def featurize_structure(self, structure: Structure) -> Dict[str, float]:
        """提取结构特征"""
        features = {}
        
        for name, featurizer in self.structure_featurizers:
            try:
                feat = featurizer.featurize(structure)
                feat_names = featurizer.feature_labels()
                for fn, fv in zip(feat_names, feat):
                    features[f"{name}_{fn}"] = fv
            except Exception as e:
                logger.warning(f"Featurization failed for {name}: {e}")
        
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
                logger.warning(f"Featurization failed for {name}: {e}")
        
        return features


class PropertyCalculator:
    """性质计算器"""
    
    def __init__(self, 
                 dft_workflow=None,
                 ml_potential=None,
                 md_workflow=None):
        self.dft_workflow = dft_workflow
        self.ml_potential = ml_potential
        self.md_workflow = md_workflow
    
    def calculate_formation_energy(self, structure: Structure) -> float:
        """计算形成能 (DFT)"""
        if self.dft_workflow is None:
            raise ValueError("DFT workflow not provided")
        
        from dft_workflow import StructureOptimizer
        
        # 转换为ASE atoms
        ase_adaptor = AseAtomsAdaptor()
        atoms = ase_adaptor.get_atoms(structure)
        
        # DFT计算
        optimizer = StructureOptimizer(self.dft_workflow.config)
        optimized = optimizer.relax_structure()
        
        return optimized.get_potential_energy()
    
    def calculate_ionic_conductivity(self, 
                                    structure: Structure,
                                    temperature: float = 300) -> float:
        """
        计算离子电导率
        
        流程: DFT NEB -> ML势MD -> 扩散系数 -> 电导率
        """
        # 步骤1: DFT NEB计算迁移能垒
        # (简化，实际需要实现NEB)
        barrier = self._estimate_migration_barrier(structure)
        
        # 步骤2: ML势MD计算扩散
        if self.ml_potential and self.md_workflow:
            from md_simulation_lammps import MDSimulationWorkflow, LAMMPSConfig
            
            config = LAMMPSConfig(
                pair_style="deepmd",
                potential_file=self.ml_potential,
                ensemble="nvt",
                temperature=temperature,
                nsteps=1000000
            )
            
            workflow = MDSimulationWorkflow(config)
            
            ase_adaptor = AseAtomsAdaptor()
            atoms = ase_adaptor.get_atoms(structure)
            
            result = workflow.full_workflow(atoms)
            
            # 从报告获取扩散系数
            D = result['report']['properties']['diffusion_coefficient']['value']
            
            # Nernst-Einstein方程计算电导率
            # σ = n * q^2 * D / (kB * T)
            # 简化计算
            conductivity = D * 1e4  # 占位符
            
            return conductivity
        
        # 如果没有ML势，使用Arrhenius近似
        else:
            D = 1e-10 * np.exp(-barrier / (8.617e-5 * temperature))
            return D * 1e4
    
    def _estimate_migration_barrier(self, structure: Structure) -> float:
        """估计迁移能垒 (简化)"""
        # 基于晶体结构类型和经验规则估计
        # 实际应使用NEB计算
        return 0.3  # eV, 占位符
    
    def calculate_bulk_modulus(self, structure: Structure) -> float:
        """计算体模量"""
        if self.dft_workflow is None:
            raise ValueError("DFT workflow not provided")
        
        # 执行EOS拟合
        from ase.eos import EquationOfState
        
        volumes = []
        energies = []
        
        ase_adaptor = AseAtomsAdaptor()
        atoms = ase_adaptor.get_atoms(structure)
        
        # 缩放体积
        for scale in np.linspace(0.95, 1.05, 7):
            scaled = atoms.copy()
            scaled.set_cell(atoms.get_cell() * scale, scale_atoms=True)
            
            # DFT单点计算
            # energy = self.dft_workflow.single_point(scaled)
            # volumes.append(scaled.get_volume())
            # energies.append(energy)
        
        # 拟合EOS
        # eos = EquationOfState(volumes, energies)
        # v0, e0, B = eos.fit()
        
        return 100.0  # GPa, 占位符


class HighThroughputScreening:
    """高通量筛选主类"""
    
    def __init__(self, config: HighThroughputConfig):
        self.config = config
        self.db = StructureDatabase(config)
        self.mp_interface = MaterialsProjectInterface(config.mp_api_key)
        self.structure_generator = StructureGenerator()
        self.feature_engineer = FeatureEngineer()
        self.property_calculator = PropertyCalculator()
        self.results = []
    
    def run_screening(self, 
                     criteria: ScreeningCriteria,
                     properties_to_calculate: List[str],
                     max_structures: int = 100) -> pd.DataFrame:
        """
        执行高通量筛选
        
        Args:
            criteria: 筛选标准
            properties_to_calculate: 要计算的性质列表
            max_structures: 最大结构数
            
        Returns:
            results_df: 结果DataFrame
        """
        logger.info("Starting high-throughput screening...")
        
        # 步骤1: 获取候选结构
        candidates = self._get_candidates(criteria, max_structures)
        
        # 步骤2: 计算性质
        results = []
        for i, candidate in enumerate(candidates):
            logger.info(f"Processing structure {i+1}/{len(candidates)}: {candidate.get('formula', 'N/A')}")
            
            try:
                result = self._calculate_properties(
                    candidate['structure'],
                    properties_to_calculate
                )
                result.update({
                    'structure_id': candidate.get('material_id', f'struct_{i}'),
                    'formula': candidate.get('formula', ''),
                })
                results.append(result)
                
                # 保存到数据库
                self.db.save_structure(
                    result['structure_id'],
                    candidate['structure'],
                    metadata=result
                )
                
            except Exception as e:
                logger.error(f"Failed to process structure: {e}")
        
        # 步骤3: 应用后筛选
        filtered_results = self._apply_filters(results, criteria)
        
        # 转换为DataFrame
        df = pd.DataFrame(filtered_results)
        
        # 保存结果
        df.to_csv(f"{self.config.db_path}/screening_results.csv", index=False)
        
        logger.info(f"Screening completed. {len(df)} structures passed.")
        
        return df
    
    def _get_candidates(self, 
                       criteria: ScreeningCriteria,
                       max_structures: int) -> List[Dict]:
        """获取候选结构"""
        candidates = []
        
        # 从Materials Project获取
        mp_results = self.mp_interface.query_materials(criteria, max_structures)
        candidates.extend(mp_results)
        
        # 生成衍生结构 (如果需要)
        if len(candidates) < max_structures:
            generated = self._generate_derivatives(candidates[:10])
            candidates.extend(generated)
        
        return candidates[:max_structures]
    
    def _generate_derivatives(self, 
                             base_structures: List[Dict],
                             max_per_base: int = 10) -> List[Dict]:
        """生成衍生结构"""
        derivatives = []
        
        for base in base_structures:
            # 合金替代
            subs = self.structure_generator.generate_substitutions(
                base['structure'],
                substitution_dict={"Li": ["Na", "K"]},
                max_substitutions=1
            )
            
            for sub in subs[:max_per_base]:
                derivatives.append({
                    'structure': sub,
                    'formula': sub.formula,
                    'derived_from': base.get('material_id')
                })
        
        return derivatives
    
    def _calculate_properties(self,
                             structure: Structure,
                             properties: List[str]) -> Dict[str, Any]:
        """计算指定性质"""
        results = {}
        
        for prop in properties:
            if prop == 'formation_energy':
                results[prop] = self.property_calculator.calculate_formation_energy(structure)
            
            elif prop == 'bulk_modulus':
                results[prop] = self.property_calculator.calculate_bulk_modulus(structure)
            
            elif prop == 'ionic_conductivity':
                results[prop] = self.property_calculator.calculate_ionic_conductivity(structure)
            
            elif prop == 'features':
                results.update(self.feature_engineer.featurize_structure(structure))
        
        return results
    
    def _apply_filters(self, 
                      results: List[Dict],
                      criteria: ScreeningCriteria) -> List[Dict]:
        """应用筛选条件"""
        filtered = []
        
        for result in results:
            passed = True
            
            # 稳定性筛选
            if 'energy_above_hull' in result:
                if result['energy_above_hull'] > criteria.max_ehull:
                    passed = False
            
            # 体模量筛选
            if criteria.min_bulk_modulus and 'bulk_modulus' in result:
                if result['bulk_modulus'] < criteria.min_bulk_modulus:
                    passed = False
            
            # 电导率筛选
            if criteria.min_ionic_conductivity and 'ionic_conductivity' in result:
                if result['ionic_conductivity'] < criteria.min_ionic_conductivity:
                    passed = False
            
            # 自定义过滤器
            if criteria.custom_filters:
                for filter_func in criteria.custom_filters:
                    if not filter_func(result):
                        passed = False
                        break
            
            if passed:
                filtered.append(result)
        
        return filtered
    
    def rank_candidates(self, 
                       df: pd.DataFrame,
                       ranking_weights: Dict[str, float]) -> pd.DataFrame:
        """对候选材料排序"""
        # 计算综合得分
        scores = np.zeros(len(df))
        
        for prop, weight in ranking_weights.items():
            if prop in df.columns:
                # 归一化
                normalized = (df[prop] - df[prop].min()) / (df[prop].max() - df[prop].min())
                scores += weight * normalized
        
        df['score'] = scores
        df = df.sort_values('score', ascending=False)
        
        return df


class SolidElectrolyteScreening(HighThroughputScreening):
    """固态电解质专用筛选"""
    
    def __init__(self, config: HighThroughputConfig):
        super().__init__(config)
    
    def run(self, max_structures: int = 100) -> pd.DataFrame:
        """运行固态电解质筛选"""
        
        # 定义筛选标准
        criteria = ScreeningCriteria(
            elements_include=["Li", "Na", "K", "Mg", "Ca", "Zn"],
            max_elements=4,
            max_natoms=50,
            min_gap=3.0,  # 宽带隙
            max_ehull=0.05,  # 高稳定性
        )
        
        # 要计算的性质
        properties = [
            'formation_energy',
            'bulk_modulus',
            'ionic_conductivity',
            'features'
        ]
        
        # 运行筛选
        df = self.run_screening(criteria, properties, max_structures)
        
        # 排序
        ranking_weights = {
            'ionic_conductivity': 0.5,
            'bulk_modulus': 0.2,
            'formation_energy': 0.3
        }
        
        ranked_df = self.rank_candidates(df, ranking_weights)
        
        return ranked_df


class CatalystScreening(HighThroughputScreening):
    """催化剂专用筛选"""
    
    def __init__(self, config: HighThroughputConfig):
        super().__init__(config)
    
    def run(self, 
           reaction: str = "ORR",
           max_structures: int = 100) -> pd.DataFrame:
        """运行催化剂筛选"""
        
        criteria = ScreeningCriteria(
            elements_include=["Pt", "Pd", "Au", "Ag", "Cu", "Ni", "Co", "Fe"],
            max_elements=3,
        )
        
        # 针对特定反应的筛选
        properties = [
            'formation_energy',
            'features'
        ]
        
        df = self.run_screening(criteria, properties, max_structures)
        
        # 计算吸附能 (简化)
        df['adsorption_energy'] = df.apply(
            lambda row: self._estimate_adsorption(row, reaction),
            axis=1
        )
        
        # 计算过电位 (简化)
        df['overpotential'] = df['adsorption_energy'].apply(
            lambda x: max(0, x - 0.4)
        )
        
        # 排序
        ranking_weights = {
            'overpotential': -0.6,  # 越小越好
            'formation_energy': -0.4
        }
        
        ranked_df = self.rank_candidates(df, ranking_weights)
        
        return ranked_df
    
    def _estimate_adsorption(self, row: pd.Series, reaction: str) -> float:
        """估计吸附能 (简化)"""
        # 基于特征估计
        return np.random.uniform(-2.0, 0.0)  # 占位符


def main():
    """示例用法"""
    
    # 配置
    config = HighThroughputConfig(
        dft_code="vasp",
        workflow_manager="fireworks",
        max_parallel=10,
        mp_api_key=os.environ.get("MP_API_KEY")
    )
    
    # 创建筛选器
    screener = HighThroughputScreening(config)
    
    # 定义标准
    criteria = ScreeningCriteria(
        elements_include=["Li", "O"],
        max_elements=3,
        min_gap=2.0
    )
    
    # 运行筛选
    # results = screener.run_screening(
    #     criteria=criteria,
    #     properties_to_calculate=['formation_energy', 'features'],
    #     max_structures=50
    # )
    
    print("High-throughput screening template ready!")


if __name__ == "__main__":
    main()
