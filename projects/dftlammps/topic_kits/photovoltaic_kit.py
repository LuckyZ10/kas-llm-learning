"""
DFT-LAMMPS 光伏套件
===================
带隙+光吸收+载流子寿命

提供光伏材料筛选与设计的完整工作流。

Author: DFT-LAMMPS Team
Phase: 56 - Orchestration System
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..orchestration.workflow_composer import Workflow, WorkflowStep, WorkflowType
from ..orchestration.topic_template import ResearchTopic


logger = logging.getLogger("photovoltaic_kit")


@dataclass
class PVSpec:
    """光伏材料规格"""
    name: str
    formula: str
    structure_file: str
    material_type: str = "thin_film"  # 或 "perovskite", "organic", "quantum_dot"
    
    # 实验数据（可选，用于验证）
    exp_band_gap: Optional[float] = None
    exp_efficiency: Optional[float] = None


@dataclass
class ElectronicProperties:
    """电子性质"""
    band_gap: float               # eV
    band_gap_type: str            # "direct" 或 "indirect"
    vbm_energy: float             # eV
    cbm_energy: float             # eV
    fermi_level: float            # eV
    effective_mass_electron: float  # m0
    effective_mass_hole: float    # m0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "band_gap": self.band_gap,
            "band_gap_type": self.band_gap_type,
            "vbm_energy": self.vbm_energy,
            "cbm_energy": self.cbm_energy,
            "fermi_level": self.fermi_level,
            "effective_mass_electron": self.effective_mass_electron,
            "effective_mass_hole": self.effective_mass_hole
        }


@dataclass
class OpticalProperties:
    """光学性质"""
    absorption_spectrum: List[Tuple[float, float]]  # (能量eV, 吸收系数)
    dielectric_constant_real: float
    dielectric_constant_imag: float
    refractive_index: float
    extinction_coefficient: float
    
    def get_band_edge_absorption(self) -> float:
        """获取带边吸收系数"""
        if not self.absorption_spectrum:
            return 0.0
        return self.absorption_spectrum[0][1] if self.absorption_spectrum else 0.0


@dataclass
class TransportProperties:
    """输运性质"""
    electron_mobility: float      # cm²/Vs
    hole_mobility: float          # cm²/Vs
    carrier_lifetime: float       # ns
    diffusion_length_electron: float  # μm
    diffusion_length_hole: float  # μm
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "electron_mobility": self.electron_mobility,
            "hole_mobility": self.hole_mobility,
            "carrier_lifetime": self.carrier_lifetime,
            "diffusion_length_electron": self.diffusion_length_electron,
            "diffusion_length_hole": self.diffusion_length_hole
        }


class PhotovoltaicKit:
    """
    光伏材料套件
    
    提供太阳能电池材料研究的一站式解决方案
    
    Example:
        kit = PhotovoltaicKit()
        
        material = PVSpec(
            name="MAPbI3",
            formula="CH3NH3PbI3",
            structure_file="MAPbI3.cif",
            material_type="perovskite"
        )
        
        # 运行完整分析
        results = kit.run_full_analysis(material)
        
        # 或单独分析
        electronic = kit.analyze_electronic_properties(material)
        optical = kit.analyze_optical_properties(material)
        efficiency = kit.predict_solar_cell_efficiency(material)
    """
    
    def __init__(self):
        self._absorption_cache: Dict[str, OpticalProperties] = {}
    
    def run_full_analysis(
        self,
        material: PVSpec,
        analysis_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """运行完整的光伏材料分析"""
        options = analysis_options or {}
        results = {"material": material.name}
        
        logger.info(f"Starting photovoltaic analysis for {material.name}")
        
        # 1. 结构优化
        if options.get('include_relaxation', True):
            logger.info("Step 1: Structure relaxation")
            results['structure'] = self._run_structure_relaxation(material)
        
        # 2. 电子性质分析
        logger.info("Step 2: Electronic properties")
        results['electronic'] = self.analyze_electronic_properties(material)
        
        # 3. 光学性质分析
        logger.info("Step 3: Optical properties")
        results['optical'] = self.analyze_optical_properties(material)
        
        # 4. 载流子输运分析
        if options.get('include_transport', True):
            logger.info("Step 4: Carrier transport properties")
            results['transport'] = self.analyze_transport_properties(material)
        
        # 5. 缺陷分析
        if options.get('include_defects', False):
            logger.info("Step 5: Defect analysis")
            results['defects'] = self.analyze_defects(material)
        
        # 6. 激子性质
        if options.get('include_excitons', True):
            logger.info("Step 6: Exciton properties")
            results['excitons'] = self.analyze_exciton_properties(material)
        
        # 7. 电池效率预测
        logger.info("Step 7: Solar cell efficiency prediction")
        results['efficiency'] = self.predict_solar_cell_efficiency(material, results)
        
        # 8. 生成报告
        results['report'] = self._generate_report(material, results)
        
        logger.info(f"Photovoltaic analysis completed for {material.name}")
        return results
    
    def analyze_electronic_properties(self, material: PVSpec) -> ElectronicProperties:
        """
        分析电子性质
        
        使用HSE06计算能带结构
        """
        logger.info(f"Calculating electronic properties for {material.name}")
        
        workflow = Workflow(
            id=f"electronic_{material.name}",
            name=f"Electronic properties of {material.name}",
            workflow_type=WorkflowType.SEQUENTIAL
        )
        
        # 步骤1: 能带计算
        workflow.steps.append(WorkflowStep(
            id="band_structure",
            name="Band Structure (HSE06)",
            module_name="vasp",
            capability_name="calculate_bands",
            inputs={"structure": material.structure_file},
            outputs={
                "band_structure": "band_structure",
                "band_gap": "band_gap",
                "vbm": "vbm",
                "cbm": "cbm"
            },
            parameters={"functional": "HSE06", "hf_ratio": 0.25}
        ))
        
        # 步骤2: 有效质量计算
        workflow.steps.append(WorkflowStep(
            id="effective_mass",
            name="Effective Mass Calculation",
            module_name="analysis",
            capability_name="effective_mass",
            inputs={"band_structure": "$band_structure"},
            outputs={
                "electron_mass": "m_e",
                "hole_mass": "m_h"
            },
            depends_on=["band_structure"]
        ))
        
        # 这里简化处理，实际应执行工作流
        # 使用模拟值
        return ElectronicProperties(
            band_gap=np.random.uniform(1.0, 2.0),
            band_gap_type="direct",
            vbm_energy=0.0,
            cbm_energy=1.5,
            fermi_level=0.75,
            effective_mass_electron=0.2,
            effective_mass_hole=0.3
        )
    
    def analyze_optical_properties(self, material: PVSpec) -> OpticalProperties:
        """
        分析光学性质
        
        计算光吸收谱和介电函数
        """
        logger.info(f"Calculating optical properties for {material.name}")
        
        # 生成模拟的吸收谱
        # 实际应基于DFT计算
        energies = np.linspace(0.5, 5.0, 100)
        
        # 模拟的吸收边
        band_gap = 1.5  # eV (从电子性质获得)
        absorption = []
        
        for E in energies:
            if E < band_gap:
                alpha = 1e3 * np.exp(-(band_gap - E) / 0.1)  # Urbach尾
            else:
                alpha = 1e5 * np.sqrt(E - band_gap)  # Tauc定律
            absorption.append((float(E), float(alpha)))
        
        return OpticalProperties(
            absorption_spectrum=absorption,
            dielectric_constant_real=10.0,
            dielectric_constant_imag=0.1,
            refractive_index=3.0,
            extinction_coefficient=0.1
        )
    
    def analyze_transport_properties(self, material: PVSpec) -> TransportProperties:
        """
        分析载流子输运性质
        
        计算迁移率和扩散长度
        """
        logger.info(f"Calculating transport properties for {material.name}")
        
        workflow = Workflow(
            id=f"transport_{material.name}",
            name=f"Transport properties of {material.name}",
            workflow_type=WorkflowType.SEQUENTIAL
        )
        
        # 步骤: 变形势计算
        workflow.steps.append(WorkflowStep(
            id="deformation",
            name="Deformation Potential",
            module_name="vasp",
            capability_name="deformation_potential",
            inputs={"structure": material.structure_file},
            outputs={"mobility": "mobility"}
        ))
        
        # 模拟值
        return TransportProperties(
            electron_mobility=100.0,  # cm²/Vs
            hole_mobility=50.0,
            carrier_lifetime=100.0,   # ns
            diffusion_length_electron=1.0,  # μm
            diffusion_length_hole=0.5
        )
    
    def analyze_defects(self, material: PVSpec) -> Dict[str, Any]:
        """分析缺陷性质"""
        logger.info(f"Analyzing defects for {material.name}")
        
        # 浅能级缺陷
        shallow_defects = [
            {"name": "V_MA", "energy": 0.05, "type": "acceptor"},
            {"name": "I_Pb", "energy": 0.08, "type": "donor"}
        ]
        
        # 深能级缺陷
        deep_defects = [
            {"name": "V_I", "energy": 0.6, "type": "recombination_center"}
        ]
        
        return {
            "shallow_defects": shallow_defects,
            "deep_defects": deep_defects,
            "recombination_rate": 1e6  # s^-1
        }
    
    def analyze_exciton_properties(self, material: PVSpec) -> Dict[str, Any]:
        """分析激子性质"""
        logger.info(f"Analyzing exciton properties for {material.name}")
        
        # 激子结合能计算
        workflow = Workflow(
            id=f"exciton_{material.name}",
            name=f"Exciton properties of {material.name}",
            workflow_type=WorkflowType.SEQUENTIAL
        )
        
        workflow.steps.append(WorkflowStep(
            id="bse",
            name="BSE Calculation",
            module_name="vasp",
            capability_name="bse_calculation",
            inputs={"structure": material.structure_file},
            outputs={"exciton_energy": "exciton_energy"}
        ))
        
        # 模拟值
        return {
            "binding_energy": 0.02,  # eV
            "bohr_radius": 3.0,      # nm
            "exciton_type": "Wannier-Mott"
        }
    
    def predict_solar_cell_efficiency(
        self,
        material: PVSpec,
        previous_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        预测太阳能电池效率
        
        基于SQ极限和详细平衡原理
        """
        logger.info(f"Predicting solar cell efficiency for {material.name}")
        
        # 获取所需参数
        if previous_results:
            electronic = previous_results.get('electronic', self.analyze_electronic_properties(material))
            optical = previous_results.get('optical', self.analyze_optical_properties(material))
            transport = previous_results.get('transport', self.analyze_transport_properties(material))
        else:
            electronic = self.analyze_electronic_properties(material)
            optical = self.analyze_optical_properties(material)
            transport = self.analyze_transport_properties(material)
        
        band_gap = electronic.band_gap
        
        # Shockley-Queisser效率极限
        sq_efficiency = self._calculate_sq_limit(band_gap)
        
        # 考虑实际损失
        # 1. 非辐射复合
        lifetime_factor = min(1.0, transport.carrier_lifetime / 100)  # 归一化
        
        # 2. 载流子收集损失
        collection_efficiency = self._estimate_collection_efficiency(transport)
        
        # 3. 反射损失
        reflection_loss = 0.1  # 10%
        
        # 综合效率预测
        predicted_efficiency = (
            sq_efficiency * 
            lifetime_factor * 
            collection_efficiency * 
            (1 - reflection_loss)
        )
        
        return {
            "band_gap": band_gap,
            "sq_limit": sq_efficiency,
            "predicted_efficiency": predicted_efficiency,
            "voc": self._estimate_voc(electronic),
            "jsc": self._estimate_jsc(optical),
            "fill_factor": 0.8,
            "loss_analysis": {
                "recombination_loss": 1 - lifetime_factor,
                "collection_loss": 1 - collection_efficiency,
                "reflection_loss": reflection_loss
            }
        }
    
    def _calculate_sq_limit(self, band_gap: float) -> float:
        """计算Shockley-Queisser效率极限"""
        # 简化的SQ极限计算
        # 实际应基于详细的积分计算
        
        # 经验公式近似
        if band_gap < 0.9:
            return 0.3
        elif band_gap < 1.4:
            return 0.33
        elif band_gap < 2.0:
            return 0.33 - (band_gap - 1.4) * 0.1
        else:
            return max(0.1, 0.27 - (band_gap - 2.0) * 0.15)
    
    def _estimate_voc(self, electronic: ElectronicProperties) -> float:
        """估计开路电压"""
        # Voc ≈ Eg/q - 0.3 V (典型损失)
        return electronic.band_gap - 0.3
    
    def _estimate_jsc(self, optical: OpticalProperties) -> float:
        """估计短路电流密度"""
        # 简化的Jsc估计
        # 实际应基于AM1.5G光谱积分
        return 20.0  # mA/cm²
    
    def _estimate_collection_efficiency(self, transport: TransportProperties) -> float:
        """估计载流子收集效率"""
        # 基于扩散长度
        min_diffusion_length = min(
            transport.diffusion_length_electron,
            transport.diffusion_length_hole
        )
        
        # 假设吸收层厚度约1μm
        thickness = 1.0  # μm
        
        return min(1.0, min_diffusion_length / thickness)
    
    def screen_pv_materials(
        self,
        candidates: List[PVSpec],
        screening_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """筛选光伏材料"""
        criteria = screening_criteria or {
            "min_band_gap": 1.0,
            "max_band_gap": 2.0,
            "min_efficiency": 15.0,
            "band_gap_type": "direct"
        }
        
        screened = []
        
        for material in candidates:
            logger.info(f"Screening {material.name}")
            
            # 快速筛选：仅计算电子性质
            electronic = self.analyze_electronic_properties(material)
            
            passes = True
            
            # 带隙筛选
            if not (criteria["min_band_gap"] <= electronic.band_gap <= criteria["max_band_gap"]):
                passes = False
            
            # 带隙类型筛选
            if criteria.get("band_gap_type") and electronic.band_gap_type != criteria["band_gap_type"]:
                passes = False
            
            # 效率预测
            if passes:
                efficiency = self.predict_solar_cell_efficiency(material, {"electronic": electronic})
                pred_eff = efficiency.get("predicted_efficiency", 0) * 100
                
                if pred_eff < criteria["min_efficiency"]:
                    passes = False
            else:
                pred_eff = 0
            
            screened.append({
                "material": material,
                "passes": passes,
                "band_gap": electronic.band_gap,
                "band_gap_type": electronic.band_gap_type,
                "predicted_efficiency": pred_eff
            })
        
        # 按预测效率排序
        screened.sort(key=lambda x: x["predicted_efficiency"], reverse=True)
        
        return screened
    
    def one_click_pv_analysis(
        self,
        structure_file: str,
        material_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """一键光伏材料分析"""
        import os
        
        name = material_name or os.path.basename(structure_file).split('.')[0]
        
        # 尝试推断化学式
        try:
            from pymatgen.core import Structure
            structure = Structure.from_file(structure_file)
            formula = structure.formula
        except:
            formula = name
        
        material = PVSpec(
            name=name,
            formula=formula,
            structure_file=structure_file,
            material_type=kwargs.get('material_type', 'thin_film')
        )
        
        return self.run_full_analysis(material, kwargs)
    
    def _run_structure_relaxation(self, material: PVSpec) -> Dict[str, Any]:
        """运行结构优化"""
        return {"relaxed": True, "material": material.name}
    
    def _generate_report(
        self,
        material: PVSpec,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成综合报告"""
        electronic = results.get('electronic', ElectronicProperties(0, "direct", 0, 0, 0, 1, 1))
        efficiency = results.get('efficiency', {})
        
        return {
            "material_name": material.name,
            "formula": material.formula,
            "material_type": material.material_type,
            "summary": {
                "band_gap": electronic.band_gap,
                "band_gap_type": electronic.band_gap_type,
                "predicted_efficiency": efficiency.get('predicted_efficiency', 0),
                "sq_limit": efficiency.get('sq_limit', 0)
            },
            "recommendations": self._generate_recommendations(electronic, efficiency)
        }
    
    def _generate_recommendations(
        self,
        electronic: ElectronicProperties,
        efficiency: Dict[str, Any]
    ) -> List[str]:
        """生成研究建议"""
        recommendations = []
        
        band_gap = electronic.band_gap
        
        if band_gap < 1.0:
            recommendations.append("Band gap too small. Low Voc expected. Consider alloying with wider gap materials.")
        elif band_gap > 2.0:
            recommendations.append("Band gap too large. Low Jsc expected. Consider alloying with narrower gap materials.")
        elif 1.1 <= band_gap <= 1.6:
            recommendations.append("Optimal band gap range for single-junction solar cells.")
        
        if electronic.band_gap_type == "indirect":
            recommendations.append("Indirect band gap detected. Thicker absorption layer may be needed.")
        
        pred_eff = efficiency.get('predicted_efficiency', 0)
        if pred_eff < 0.1:
            recommendations.append("Low predicted efficiency. Consider different material system.")
        
        return recommendations


# 便捷函数
def quick_pv_analysis(structure_file: str, **kwargs) -> Dict[str, Any]:
    """快速光伏材料分析"""
    kit = PhotovoltaicKit()
    return kit.one_click_pv_analysis(structure_file, **kwargs)


def screen_solar_cell_materials(
    structure_files: List[str],
    **kwargs
) -> List[Dict[str, Any]]:
    """批量筛选太阳能电池材料"""
    kit = PhotovoltaicKit()
    
    candidates = []
    for file in structure_files:
        try:
            from pymatgen.core import Structure
            structure = Structure.from_file(file)
            formula = structure.formula
        except:
            import os
            formula = os.path.basename(file).split('.')[0]
        
        candidates.append(PVSpec(
            name=formula,
            formula=formula,
            structure_file=file
        ))
    
    return kit.screen_pv_materials(candidates, kwargs)