"""
DFT-LAMMPS 催化剂套件
=====================
吸附能+选择性+稳定性+Volcano图

提供催化剂设计与筛选的完整工作流。

Author: DFT-LAMMPS Team
Phase: 56 - Orchestration System
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ..orchestration.workflow_composer import Workflow, WorkflowStep, WorkflowType
from ..orchestration.topic_template import ResearchTopic
from ..integration_layer.unified_data_model import PropertyType


logger = logging.getLogger("catalyst_kit")


@dataclass
class CatalystSpec:
    """催化剂规格"""
    name: str
    bulk_structure_file: str
    surface_miller: Tuple[int, int, int] = (1, 1, 1)
    surface_size: Tuple[int, int] = (3, 3)
    vacuum: float = 15.0


@dataclass
class AdsorptionSite:
    """吸附位点"""
    site_type: str  # "top", "bridge", "hollow", "fcc", "hcp"
    position: Tuple[float, float, float]
    coordination: int
    surface_atom: str


@dataclass
class AdsorptionResult:
    """吸附能结果"""
    adsorbate: str
    site: AdsorptionSite
    adsorption_energy: float  # eV
    binding_distance: float   # Å
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "adsorbate": self.adsorbate,
            "site_type": self.site.site_type,
            "adsorption_energy": self.adsorption_energy,
            "binding_distance": self.binding_distance
        }


@dataclass
class ScalingRelation:
    """标度关系"""
    descriptor: str
    slope: float
    intercept: float
    r_squared: float
    
    def predict(self, descriptor_value: float) -> float:
        """基于标度关系预测"""
        return self.slope * descriptor_value + self.intercept


class CatalystKit:
    """
    催化剂设计套件
    
    提供催化剂研究的一站式解决方案
    
    Example:
        kit = CatalystKit()
        
        catalyst = CatalystSpec(
            name="Pt(111)",
            bulk_structure_file="Pt.cif",
            surface_miller=(1, 1, 1)
        )
        
        # 计算吸附能
        results = kit.calculate_adsorption_energies(
            catalyst, 
            adsorbates=["H", "O", "OH", "OOH"]
        )
        
        # 生成Volcano图
        volcano = kit.generate_volcano_plot(results, reaction="ORR")
    """
    
    def __init__(self):
        self._adsorption_sites_cache: Dict[str, List[AdsorptionSite]] = {}
        self._scaling_relations: Dict[str, ScalingRelation] = {}
    
    def run_full_analysis(
        self,
        catalyst: CatalystSpec,
        target_reaction: str,
        analysis_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """运行完整的催化剂分析"""
        options = analysis_options or {}
        results = {"catalyst": catalyst.name}
        
        logger.info(f"Starting catalyst analysis for {catalyst.name}")
        
        # 1. 构建表面
        logger.info("Step 1: Building surface model")
        surface = self.build_surface(catalyst)
        results['surface'] = surface
        
        # 2. 识别吸附位点
        logger.info("Step 2: Identifying adsorption sites")
        sites = self.identify_adsorption_sites(catalyst)
        results['adsorption_sites'] = sites
        
        # 3. 计算吸附能
        logger.info("Step 3: Calculating adsorption energies")
        adsorbates = options.get('adsorbates', ["H", "O", "OH", "OOH", "N", "CO"])
        adsorption_results = self.calculate_adsorption_energies(catalyst, adsorbates, sites)
        results['adsorption_energies'] = adsorption_results
        
        # 4. 构建标度关系
        logger.info("Step 4: Building scaling relations")
        scaling = self.build_scaling_relations(adsorption_results)
        results['scaling_relations'] = scaling
        
        # 5. 计算选择性
        logger.info("Step 5: Analyzing selectivity")
        selectivity = self.analyze_selectivity(catalyst, adsorption_results, target_reaction)
        results['selectivity'] = selectivity
        
        # 6. 分析稳定性
        logger.info("Step 6: Analyzing stability")
        stability = self.analyze_stability(catalyst)
        results['stability'] = stability
        
        # 7. 生成Volcano图
        logger.info("Step 7: Generating volcano plot")
        volcano = self.generate_volcano_plot(adsorption_results, target_reaction, scaling)
        results['volcano_plot'] = volcano
        
        # 8. 生成综合报告
        results['report'] = self._generate_report(catalyst, results)
        
        logger.info(f"Catalyst analysis completed for {catalyst.name}")
        return results
    
    def build_surface(self, catalyst: CatalystSpec) -> Dict[str, Any]:
        """构建表面模型"""
        workflow = Workflow(
            id=f"surface_{catalyst.name}",
            name=f"Build {catalyst.name} surface",
            workflow_type=WorkflowType.SEQUENTIAL
        )
        
        workflow.steps.append(WorkflowStep(
            id="build_surface",
            name="Build Surface",
            module_name="structure",
            capability_name="build_surface",
            inputs={
                "bulk_structure": catalyst.bulk_structure_file,
                "miller_index": catalyst.surface_miller,
                "size": catalyst.surface_size,
                "vacuum": catalyst.vacuum
            },
            outputs={"surface_structure": "surface_structure"}
        ))
        
        workflow.steps.append(WorkflowStep(
            id="relax_surface",
            name="Relax Surface",
            module_name="vasp",
            capability_name="relax_structure",
            inputs={"structure": "$surface_structure"},
            outputs={"relaxed_surface": "relaxed_surface"},
            depends_on=["build_surface"]
        ))
        
        return {
            "catalyst": catalyst.name,
            "miller_index": catalyst.surface_miller,
            "size": catalyst.surface_size,
            "workflow": workflow
        }
    
    def identify_adsorption_sites(
        self,
        catalyst: CatalystSpec
    ) -> List[AdsorptionSite]:
        """识别吸附位点"""
        cache_key = f"{catalyst.name}_{catalyst.surface_miller}"
        
        if cache_key in self._adsorption_sites_cache:
            return self._adsorption_sites_cache[cache_key]
        
        # 基于表面结构识别位点
        # 简化的实现，实际应基于几何分析
        sites = [
            AdsorptionSite("top", (0.0, 0.0, 0.0), 1, "Pt"),
            AdsorptionSite("bridge", (0.5, 0.0, 0.0), 2, "Pt"),
            AdsorptionSite("fcc", (0.33, 0.33, 0.0), 3, "Pt"),
            AdsorptionSite("hcp", (0.67, 0.67, 0.0), 3, "Pt")
        ]
        
        self._adsorption_sites_cache[cache_key] = sites
        return sites
    
    def calculate_adsorption_energies(
        self,
        catalyst: CatalystSpec,
        adsorbates: List[str],
        sites: Optional[List[AdsorptionSite]] = None
    ) -> Dict[str, List[AdsorptionResult]]:
        """计算吸附能"""
        if sites is None:
            sites = self.identify_adsorption_sites(catalyst)
        
        results = {}
        
        for adsorbate in adsorbates:
            logger.info(f"Calculating adsorption energy for {adsorbate}")
            
            adsorption_results = []
            
            for site in sites:
                # 创建吸附构型并计算能量
                result = self._calculate_single_adsorption(
                    catalyst, adsorbate, site
                )
                
                if result:
                    adsorption_results.append(result)
            
            # 按吸附能排序
            adsorption_results.sort(key=lambda x: x.adsorption_energy)
            results[adsorbate] = adsorption_results
        
        return results
    
    def _calculate_single_adsorption(
        self,
        catalyst: CatalystSpec,
        adsorbate: str,
        site: AdsorptionSite
    ) -> Optional[AdsorptionResult]:
        """计算单个吸附构型的能量"""
        workflow = Workflow(
            id=f"ads_{catalyst.name}_{adsorbate}_{site.site_type}",
            name=f"Adsorption: {adsorbate} on {catalyst.name} {site.site_type}",
            workflow_type=WorkflowType.SEQUENTIAL
        )
        
        # 这里简化处理，实际应调用DFT计算
        # 使用简化的能量估算
        base_energy = self._estimate_adsorption_energy(adsorbate, site)
        
        return AdsorptionResult(
            adsorbate=adsorbate,
            site=site,
            adsorption_energy=base_energy,
            binding_distance=2.0
        )
    
    def _estimate_adsorption_energy(self, adsorbate: str, site: AdsorptionSite) -> float:
        """估算吸附能（简化模型）"""
        # 简化的估算，实际应基于DFT
        base_energies = {
            "H": -0.5,
            "O": -1.0,
            "OH": -0.8,
            "OOH": -0.6,
            "N": -1.2,
            "CO": -0.7
        }
        
        # 位点修正
        site_correction = {
            "top": 0.0,
            "bridge": -0.1,
            "fcc": -0.2,
            "hcp": -0.15
        }
        
        base = base_energies.get(adsorbate, -0.5)
        correction = site_correction.get(site.site_type, 0.0)
        
        return base + correction + np.random.normal(0, 0.05)  # 添加小的随机扰动
    
    def build_scaling_relations(
        self,
        adsorption_results: Dict[str, List[AdsorptionResult]]
    ) -> Dict[str, ScalingRelation]:
        """构建标度关系"""
        scaling = {}
        
        # O* vs OH* 标度关系
        if "O" in adsorption_results and "OH" in adsorption_results:
            o_energies = [r.adsorption_energy for r in adsorption_results["O"]]
            oh_energies = [r.adsorption_energy for r in adsorption_results["OH"]]
            
            if len(o_energies) == len(oh_energies) and len(o_energies) > 1:
                slope, intercept, r_squared = self._linear_regression(
                    o_energies, oh_energies
                )
                scaling["O_vs_OH"] = ScalingRelation("O", slope, intercept, r_squared)
        
        # OOH* vs OH* 标度关系
        if "OOH" in adsorption_results and "OH" in adsorption_results:
            ooh_energies = [r.adsorption_energy for r in adsorption_results["OOH"]]
            oh_energies = [r.adsorption_energy for r in adsorption_results["OH"]]
            
            if len(ooh_energies) == len(oh_energies) and len(ooh_energies) > 1:
                slope, intercept, r_squared = self._linear_regression(
                    oh_energies, ooh_energies
                )
                scaling["OOH_vs_OH"] = ScalingRelation("OH", slope, intercept, r_squared)
        
        self._scaling_relations = scaling
        return scaling
    
    def _linear_regression(
        self,
        x: List[float],
        y: List[float]
    ) -> Tuple[float, float, float]:
        """线性回归"""
        n = len(x)
        if n < 2:
            return 0.0, 0.0, 0.0
        
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        ss_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        ss_xx = sum((xi - x_mean) ** 2 for xi in x)
        
        if ss_xx == 0:
            return 0.0, y_mean, 0.0
        
        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        
        # R²
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return slope, intercept, r_squared
    
    def analyze_selectivity(
        self,
        catalyst: CatalystSpec,
        adsorption_results: Dict[str, List[AdsorptionResult]],
        target_reaction: str
    ) -> Dict[str, Any]:
        """分析选择性"""
        selectivity = {
            "catalyst": catalyst.name,
            "target_reaction": target_reaction,
            "selectivity_score": 0.0,
            "competing_reactions": []
        }
        
        if target_reaction == "ORR":  # 氧还原反应
            # 理想ORR催化剂：O和OH吸附能适中
            if "O" in adsorption_results and "OH" in adsorption_results:
                o_energy = adsorption_results["O"][0].adsorption_energy
                oh_energy = adsorption_results["OH"][0].adsorption_energy
                
                # 理想范围：OH* ≈ 0.0 to -0.5 eV
                if -0.5 <= oh_energy <= 0.0:
                    selectivity["selectivity_score"] = 0.9
                else:
                    selectivity["selectivity_score"] = 0.5
        
        elif target_reaction == "HER":  # 氢析出反应
            # 理想HER催化剂：H吸附能 ≈ 0 eV
            if "H" in adsorption_results:
                h_energy = adsorption_results["H"][0].adsorption_energy
                
                # 理想范围：H* ≈ -0.1 to 0.1 eV (相对于SHE)
                if abs(h_energy) < 0.2:
                    selectivity["selectivity_score"] = 0.95
                else:
                    selectivity["selectivity_score"] = max(0, 1 - abs(h_energy))
        
        return selectivity
    
    def analyze_stability(self, catalyst: CatalystSpec) -> Dict[str, Any]:
        """分析催化剂稳定性"""
        stability = {
            "catalyst": catalyst.name,
            "surface_energy": 0.0,
            "cohesive_energy": 0.0,
            "dissolution_potential": 0.0,
            "stability_score": 0.0
        }
        
        # 这里应进行实际的稳定性计算
        # 简化实现
        stability["stability_score"] = 0.85  # 示例值
        
        return stability
    
    def generate_volcano_plot(
        self,
        adsorption_results: Dict[str, List[AdsorptionResult]],
        reaction: str,
        scaling: Optional[Dict[str, ScalingRelation]] = None
    ) -> Dict[str, Any]:
        """生成Volcano图"""
        volcano_data = {
            "reaction": reaction,
            "descriptor": None,
            "activities": [],
            "optimal_descriptor": None,
            "optimal_activity": None
        }
        
        if reaction == "ORR":
            # ORR Volcano: 以OH*吸附能为描述符
            if "OH" not in adsorption_results:
                return volcano_data
            
            oh_energies = [r.adsorption_energy for r in adsorption_results["OH"]]
            descriptor_range = np.linspace(min(oh_energies) - 0.5, max(oh_energies) + 0.5, 100)
            
            activities = []
            for d_oh in descriptor_range:
                # 使用标度关系估计O和OOH的吸附能
                if scaling and "O_vs_OH" in scaling:
                    d_o = scaling["O_vs_OH"].predict(d_oh)
                else:
                    d_o = d_oh + 0.5  # 默认近似
                
                if scaling and "OOH_vs_OH" in scaling:
                    d_ooh = scaling["OOH_vs_OH"].predict(d_oh)
                else:
                    d_ooh = d_oh + 2.0  # 默认近似
                
                # 计算过电位
                overpotential = self._calculate_orr_overpotential(d_oh, d_o, d_ooh)
                activity = -overpotential  # 负的过电位作为活性度量
                
                activities.append(activity)
            
            volcano_data["descriptor"] = "OH_adsorption_energy"
            volcano_data["descriptor_values"] = descriptor_range.tolist()
            volcano_data["activities"] = activities
            
            # 找到最优值
            max_idx = np.argmax(activities)
            volcano_data["optimal_descriptor"] = descriptor_range[max_idx]
            volcano_data["optimal_activity"] = activities[max_idx]
        
        elif reaction == "HER":
            # HER Volcano: 以H*吸附能为描述符
            if "H" not in adsorption_results:
                return volcano_data
            
            h_energies = [r.adsorption_energy for r in adsorption_results["H"]]
            descriptor_range = np.linspace(min(h_energies) - 0.5, max(h_energies) + 0.5, 100)
            
            activities = []
            for d_h in descriptor_range:
                # 理想H*吸附能接近0
                overpotential = abs(d_h)
                activity = -overpotential
                activities.append(activity)
            
            volcano_data["descriptor"] = "H_adsorption_energy"
            volcano_data["descriptor_values"] = descriptor_range.tolist()
            volcano_data["activities"] = activities
            
            max_idx = np.argmax(activities)
            volcano_data["optimal_descriptor"] = descriptor_range[max_idx]
            volcano_data["optimal_activity"] = activities[max_idx]
        
        return volcano_data
    
    def _calculate_orr_overpotential(
        self,
        e_oh: float,
        e_o: float,
        e_ooh: float
    ) -> float:
        """计算ORR过电位"""
        # 简化的过电位计算
        # 实际应基于完整的热力学分析
        
        # ORR步骤的自由能变化（近似）
        delta_g1 = -e_ooh + 4.92  # OOH* -> O* + OH*
        delta_g2 = e_o - e_oh     # O* + OH* -> 2OH*
        delta_g3 = -e_oh          # 最后还原步骤
        
        # 过电位由最大自由能上坡决定
        max_delta_g = max(delta_g1, delta_g2, delta_g3)
        overpotential = max(0, max_delta_g - 1.23)  # 1.23 V是平衡电位
        
        return overpotential
    
    def screen_catalysts(
        self,
        candidates: List[CatalystSpec],
        target_reaction: str,
        screening_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """筛选催化剂"""
        criteria = screening_criteria or {
            "max_adsorption_energy": -0.2,
            "min_stability_score": 0.7
        }
        
        screened = []
        
        for catalyst in candidates:
            logger.info(f"Screening {catalyst.name}")
            
            # 快速分析
            results = self.run_full_analysis(
                catalyst, target_reaction,
                analysis_options={'adsorbates': ['OH', 'O']}
            )
            
            # 评估
            volcano = results.get('volcano_plot', {})
            stability = results.get('stability', {})
            
            optimal_activity = volcano.get('optimal_activity', -999)
            stability_score = stability.get('stability_score', 0)
            
            passes = (
                optimal_activity >= criteria.get('max_adsorption_energy', -0.2) and
                stability_score >= criteria.get('min_stability_score', 0.7)
            )
            
            screened.append({
                "catalyst": catalyst,
                "passes": passes,
                "optimal_activity": optimal_activity,
                "stability_score": stability_score,
                "volcano_data": volcano
            })
        
        # 按活性排序
        screened.sort(key=lambda x: x['optimal_activity'], reverse=True)
        
        return screened
    
    def one_click_catalyst_design(
        self,
        bulk_structure_file: str,
        target_reaction: str = "ORR",
        **kwargs
    ) -> Dict[str, Any]:
        """一键催化剂设计"""
        import os
        
        name = kwargs.get('name', os.path.basename(bulk_structure_file).split('.')[0])
        
        catalyst = CatalystSpec(
            name=name,
            bulk_structure_file=bulk_structure_file,
            surface_miller=kwargs.get('miller_index', (1, 1, 1))
        )
        
        return self.run_full_analysis(catalyst, target_reaction)
    
    def _generate_report(
        self,
        catalyst: CatalystSpec,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成综合报告"""
        volcano = results.get('volcano_plot', {})
        selectivity = results.get('selectivity', {})
        stability = results.get('stability', {})
        
        return {
            "catalyst_name": catalyst.name,
            "surface_miller": catalyst.surface_miller,
            "summary": {
                "optimal_descriptor_value": volcano.get('optimal_descriptor'),
                "predicted_activity": volcano.get('optimal_activity'),
                "selectivity_score": selectivity.get('selectivity_score'),
                "stability_score": stability.get('stability_score')
            },
            "recommendations": self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成设计建议"""
        recommendations = []
        
        volcano = results.get('volcano_plot', {})
        selectivity = results.get('selectivity', {})
        
        optimal = volcano.get('optimal_descriptor')
        if optimal is not None:
            if optimal < -0.5:
                recommendations.append("Adsorption too strong. Consider alloying with weaker binding metals.")
            elif optimal > -0.1:
                recommendations.append("Adsorption too weak. Consider alloying with stronger binding metals.")
        
        stability = results.get('stability', {}).get('stability_score', 1.0)
        if stability < 0.7:
            recommendations.append("Low stability predicted. Consider protective coatings or core-shell structures.")
        
        return recommendations


# 便捷函数
def quick_catalyst_analysis(
    bulk_structure_file: str,
    target_reaction: str = "ORR",
    **kwargs
) -> Dict[str, Any]:
    """快速催化剂分析"""
    kit = CatalystKit()
    return kit.one_click_catalyst_design(bulk_structure_file, target_reaction, **kwargs)