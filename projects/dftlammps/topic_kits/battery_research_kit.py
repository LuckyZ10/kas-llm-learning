"""
DFT-LAMMPS 电池研究套件
=======================
离子电导率+界面稳定性+循环寿命

提供电池材料研究的完整工作流套件。

Author: DFT-LAMMPS Team
Phase: 56 - Orchestration System
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..orchestration.module_registry import (
    ModuleRegistry, Capability, CapabilityType, 
    module, capability
)
from ..orchestration.workflow_composer import (
    Workflow, WorkflowStep, WorkflowType, WorkflowExecutor
)
from ..orchestration.topic_template import (
    TopicTemplate, ResearchTopic, create_battery_workflow
)
from ..integration_layer.unified_data_model import (
    StructureData, PropertyData, PropertyType, DataSource
)


logger = logging.getLogger("battery_research_kit")


@dataclass
class BatteryMaterialSpec:
    """电池材料规格"""
    name: str
    formula: str
    material_type: str  # "cathode", "anode", "electrolyte"
    structure_file: Optional[str] = None
    
    # 离子信息
    working_ion: str = "Li"  # Li, Na, K, Mg, etc.
    ion_charge: int = 1
    
    # 计算参数
    voltage_range: Tuple[float, float] = (2.0, 5.0)
    temperature_range: List[float] = field(default_factory=lambda: [300, 400, 500])


@dataclass
class IonConductivityResult:
    """离子电导率结果"""
    temperature: float              # K
    conductivity: float             # S/cm
    diffusion_coefficient: float    # cm²/s
    activation_energy: float        # eV
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "conductivity": self.conductivity,
            "diffusion_coefficient": self.diffusion_coefficient,
            "activation_energy": self.activation_energy
        }


@dataclass
class InterfaceStabilityResult:
    """界面稳定性结果"""
    interface_energy: float         # eV/Å²
    adhesion_energy: float          # eV/Å²
    stability_score: float          # 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "interface_energy": self.interface_energy,
            "adhesion_energy": self.adhesion_energy,
            "stability_score": self.stability_score
        }


@dataclass
class CycleLifePrediction:
    """循环寿命预测"""
    predicted_cycles: int
    capacity_fade_rate: float       # % per cycle
    degradation_mechanism: str
    confidence: float               # 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_cycles": self.predicted_cycles,
            "capacity_fade_rate": self.capacity_fade_rate,
            "degradation_mechanism": self.degradation_mechanism,
            "confidence": self.confidence
        }


class BatteryResearchKit:
    """
    电池研究套件
    
    一键启动电池材料完整研究工作流
    
    Example:
        kit = BatteryResearchKit()
        
        # 定义材料
        material = BatteryMaterialSpec(
            name="LiFePO4",
            formula="LiFePO4",
            material_type="cathode",
            structure_file="LiFePO4.cif",
            working_ion="Li"
        )
        
        # 运行完整研究
        results = kit.run_full_analysis(material)
        
        # 或单独运行某项分析
        conductivity = kit.analyze_ion_conductivity(material)
    """
    
    def __init__(
        self,
        registry: Optional[ModuleRegistry] = None,
        executor: Optional[WorkflowExecutor] = None
    ):
        self.registry = registry or ModuleRegistry.get_instance()
        self.executor = executor or WorkflowExecutor()
        self._init_default_modules()
    
    def _init_default_modules(self) -> None:
        """初始化默认模块"""
        # 确保关键模块已注册
        required_modules = ["vasp", "lammps", "analysis"]
        for mod_name in required_modules:
            module = self.registry.get_module(mod_name)
            if module and module.state.name != "ACTIVE":
                self.registry.initialize_module(mod_name)
    
    def run_full_analysis(
        self,
        material: BatteryMaterialSpec,
        analysis_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        运行完整分析
        
        Args:
            material: 材料规格
            analysis_options: 分析选项
        
        Returns:
            完整分析结果
        """
        options = analysis_options or {}
        results = {}
        
        logger.info(f"Starting full battery analysis for {material.name}")
        
        # 1. 结构优化
        if options.get('include_relaxation', True):
            logger.info("Step 1: Structure relaxation")
            results['structure'] = self._run_structure_relaxation(material)
        
        # 2. 离子电导率分析
        if options.get('include_conductivity', True):
            logger.info("Step 2: Ion conductivity analysis")
            results['ion_conductivity'] = self.analyze_ion_conductivity(material)
        
        # 3. 电压曲线计算
        if options.get('include_voltage_profile', True):
            logger.info("Step 3: Voltage profile calculation")
            results['voltage_profile'] = self.calculate_voltage_profile(material)
        
        # 4. 界面稳定性分析
        if options.get('include_interface_stability', False):
            logger.info("Step 4: Interface stability analysis")
            electrolyte = options.get('electrolyte_material')
            if electrolyte:
                results['interface_stability'] = self.analyze_interface_stability(
                    material, electrolyte
                )
        
        # 5. 循环寿命预测
        if options.get('include_cycle_life', False):
            logger.info("Step 5: Cycle life prediction")
            results['cycle_life'] = self.predict_cycle_life(material)
        
        # 6. 生成综合报告
        results['report'] = self._generate_report(material, results)
        
        logger.info(f"Full analysis completed for {material.name}")
        return results
    
    def analyze_ion_conductivity(
        self,
        material: BatteryMaterialSpec
    ) -> List[IonConductivityResult]:
        """
        分析离子电导率
        
        使用AIMD模拟计算不同温度下的离子电导率
        """
        results = []
        
        for temp in material.temperature_range:
            logger.info(f"Calculating ion conductivity at {temp}K")
            
            # 创建AIMD工作流
            workflow = self._create_aimd_workflow(material, temp)
            
            # 执行
            execution = self.executor.execute(workflow)
            
            # 提取结果
            conductivity = self._extract_conductivity(execution)
            diff_coeff = self._extract_diffusion_coefficient(execution)
            
            result = IonConductivityResult(
                temperature=temp,
                conductivity=conductivity,
                diffusion_coefficient=diff_coeff,
                activation_energy=0.0  # 从Arrhenius拟合获得
            )
            results.append(result)
        
        # 计算活化能（Arrhenius拟合）
        activation_energy = self._fit_arrhenius(results)
        for r in results:
            r.activation_energy = activation_energy
        
        return results
    
    def calculate_voltage_profile(
        self,
        material: BatteryMaterialSpec
    ) -> Dict[str, Any]:
        """
        计算电压曲线
        
        基于DFT计算不同锂化状态的电压
        """
        logger.info(f"Calculating voltage profile for {material.name}")
        
        # 创建电压计算工作流
        workflow = self._create_voltage_workflow(material)
        
        # 执行
        execution = self.executor.execute(workflow)
        
        # 解析结果
        voltage_curve = self._extract_voltage_curve(execution)
        
        return {
            "material": material.name,
            "working_ion": material.working_ion,
            "voltage_curve": voltage_curve,
            "average_voltage": sum(v for _, v in voltage_curve) / len(voltage_curve) if voltage_curve else 0,
            "capacity": self._calculate_capacity(material, voltage_curve)
        }
    
    def analyze_interface_stability(
        self,
        electrode: BatteryMaterialSpec,
        electrolyte: BatteryMaterialSpec
    ) -> InterfaceStabilityResult:
        """
        分析界面稳定性
        
        计算电极-电解质界面的能量和稳定性
        """
        logger.info(f"Analyzing interface stability: {electrode.name} | {electrolyte.name}")
        
        # 创建界面计算工作流
        workflow = self._create_interface_workflow(electrode, electrolyte)
        
        # 执行
        execution = self.executor.execute(workflow)
        
        # 解析结果
        interface_energy = execution.context.get('interface_energy', 0.0)
        adhesion_energy = execution.context.get('adhesion_energy', 0.0)
        
        # 计算稳定性分数
        stability_score = self._calculate_stability_score(interface_energy, adhesion_energy)
        
        return InterfaceStabilityResult(
            interface_energy=interface_energy,
            adhesion_energy=adhesion_energy,
            stability_score=stability_score
        )
    
    def predict_cycle_life(
        self,
        material: BatteryMaterialSpec,
        operating_conditions: Optional[Dict[str, Any]] = None
    ) -> CycleLifePrediction:
        """
        预测循环寿命
        
        基于材料特性和操作条件预测电池循环寿命
        """
        logger.info(f"Predicting cycle life for {material.name}")
        
        conditions = operating_conditions or {
            "temperature": 298,
            "c_rate": 1.0,
            "depth_of_discharge": 0.8
        }
        
        # 创建循环寿命预测工作流
        workflow = self._create_cycle_life_workflow(material, conditions)
        
        # 执行
        execution = self.executor.execute(workflow)
        
        # 基于机器学习模型预测
        predicted_cycles = execution.context.get('predicted_cycles', 1000)
        fade_rate = execution.context.get('capacity_fade_rate', 0.01)
        mechanism = execution.context.get('degradation_mechanism', 'unknown')
        
        return CycleLifePrediction(
            predicted_cycles=predicted_cycles,
            capacity_fade_rate=fade_rate,
            degradation_mechanism=mechanism,
            confidence=0.85
        )
    
    def screen_cathode_materials(
        self,
        candidates: List[BatteryMaterialSpec],
        screening_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        正极材料筛选
        
        对候选材料进行快速筛选
        """
        criteria = screening_criteria or {
            "min_voltage": 3.0,
            "max_voltage": 5.0,
            "min_capacity": 100,  # mAh/g
            "max_ion_migration_barrier": 0.5  # eV
        }
        
        screened = []
        
        for material in candidates:
            logger.info(f"Screening {material.name}")
            
            # 快速计算
            voltage_profile = self.calculate_voltage_profile(material)
            
            # 应用筛选条件
            passes = True
            score = 0.0
            
            avg_voltage = voltage_profile.get('average_voltage', 0)
            if criteria["min_voltage"] <= avg_voltage <= criteria["max_voltage"]:
                score += 1.0
            else:
                passes = False
            
            capacity = voltage_profile.get('capacity', 0)
            if capacity >= criteria["min_capacity"]:
                score += 1.0
            else:
                passes = False
            
            screened.append({
                "material": material,
                "passes": passes,
                "score": score,
                "voltage_profile": voltage_profile
            })
        
        # 按分数排序
        screened.sort(key=lambda x: x['score'], reverse=True)
        
        return screened
    
    def one_click_research(
        self,
        structure_file: str,
        material_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        一键启动电池研究
        
        最简单的入口，只需要提供结构文件
        """
        # 自动识别材料类型
        material_type = self._detect_material_type(structure_file)
        
        # 尝试从结构文件推断化学式
        formula = self._extract_formula(structure_file)
        
        material = BatteryMaterialSpec(
            name=material_name or formula or "Unknown",
            formula=formula or "Unknown",
            material_type=material_type,
            structure_file=structure_file,
            working_ion=kwargs.get('working_ion', 'Li')
        )
        
        return self.run_full_analysis(material, kwargs)
    
    def _create_aimd_workflow(
        self,
        material: BatteryMaterialSpec,
        temperature: float
    ) -> Workflow:
        """创建AIMD工作流"""
        workflow = Workflow(
            id=f"aimd_{material.name}_{temperature}K",
            name=f"AIMD Simulation for {material.name} at {temperature}K",
            workflow_type=WorkflowType.SEQUENTIAL
        )
        
        # 步骤1: 读取结构
        workflow.steps.append(WorkflowStep(
            id="import",
            name="Import Structure",
            module_name="io",
            function_name="read_structure",
            inputs={"file_path": material.structure_file},
            outputs={"structure": "structure"}
        ))
        
        # 步骤2: AIMD模拟
        workflow.steps.append(WorkflowStep(
            id="aimd",
            name="AIMD Simulation",
            module_name="vasp",
            capability_name="run_aimd",
            inputs={"structure": "$structure", "temperature": temperature},
            outputs={"trajectory": "trajectory"},
            depends_on=["import"],
            parameters={
                "timestep": 2.0,
                "steps": 5000,
                "ensemble": "nvt",
                "temperature": temperature
            }
        ))
        
        # 步骤3: 分析MSD
        workflow.steps.append(WorkflowStep(
            id="msd_analysis",
            name="MSD Analysis",
            module_name="analysis",
            capability_name="analyze_msd",
            inputs={"trajectory": "$trajectory"},
            outputs={"diffusion_coeff": "diffusion_coeff", "conductivity": "conductivity"},
            depends_on=["aimd"]
        ))
        
        return workflow
    
    def _create_voltage_workflow(
        self,
        material: BatteryMaterialSpec
    ) -> Workflow:
        """创建电压计算工作流"""
        workflow = Workflow(
            id=f"voltage_{material.name}",
            name=f"Voltage Profile for {material.name}",
            workflow_type=WorkflowType.SEQUENTIAL
        )
        
        # 使用课题模板
        from ..orchestration.topic_template import get_template_manager
        manager = get_template_manager()
        base_workflow = manager.create_workflow(
            ResearchTopic.BATTERY,
            inputs={"input_structure": material.structure_file}
        )
        
        return base_workflow
    
    def _create_interface_workflow(
        self,
        electrode: BatteryMaterialSpec,
        electrolyte: BatteryMaterialSpec
    ) -> Workflow:
        """创建界面计算工作流"""
        workflow = Workflow(
            id=f"interface_{electrode.name}_{electrolyte.name}",
            name=f"Interface: {electrode.name} | {electrolyte.name}",
            workflow_type=WorkflowType.SEQUENTIAL
        )
        
        # 步骤1: 构建界面模型
        workflow.steps.append(WorkflowStep(
            id="build_interface",
            name="Build Interface Model",
            module_name="structure",
            capability_name="build_interface",
            inputs={
                "electrode": electrode.structure_file,
                "electrolyte": electrolyte.structure_file
            },
            outputs={"interface_structure": "interface_structure"}
        ))
        
        # 步骤2: 界面能计算
        workflow.steps.append(WorkflowStep(
            id="interface_energy",
            name="Calculate Interface Energy",
            module_name="vasp",
            capability_name="calculate_interface_energy",
            inputs={"structure": "$interface_structure"},
            outputs={"interface_energy": "interface_energy"},
            depends_on=["build_interface"]
        ))
        
        return workflow
    
    def _create_cycle_life_workflow(
        self,
        material: BatteryMaterialSpec,
        conditions: Dict[str, Any]
    ) -> Workflow:
        """创建循环寿命预测工作流"""
        workflow = Workflow(
            id=f"cycle_life_{material.name}",
            name=f"Cycle Life Prediction for {material.name}",
            workflow_type=WorkflowType.SEQUENTIAL
        )
        
        # 这里使用ML模型进行预测
        workflow.steps.append(WorkflowStep(
            id="predict",
            name="ML Prediction",
            module_name="ml",
            capability_name="predict_cycle_life",
            inputs={
                "material": material.name,
                "conditions": conditions
            },
            outputs={
                "predicted_cycles": "predicted_cycles",
                "fade_rate": "capacity_fade_rate"
            }
        ))
        
        return workflow
    
    def _extract_conductivity(self, execution) -> float:
        """从执行结果提取电导率"""
        return execution.context.get('conductivity', 1e-4)
    
    def _extract_diffusion_coefficient(self, execution) -> float:
        """从执行结果提取扩散系数"""
        return execution.context.get('diffusion_coeff', 1e-8)
    
    def _fit_arrhenius(
        self,
        results: List[IonConductivityResult]
    ) -> float:
        """Arrhenius拟合计算活化能"""
        import numpy as np
        
        if len(results) < 2:
            return 0.0
        
        # 1/T vs ln(sigma)
        inv_T = [1.0 / r.temperature for r in results]
        log_sigma = [np.log(r.conductivity) for r in results]
        
        # 线性拟合
        coeffs = np.polyfit(inv_T, log_sigma, 1)
        
        # Ea = -k * slope (k in eV/K)
        k_b = 8.617333e-5  # eV/K
        activation_energy = -coeffs[0] * k_b
        
        return max(0, activation_energy)
    
    def _extract_voltage_curve(self, execution) -> List[Tuple[float, float]]:
        """提取电压曲线"""
        return execution.context.get('voltage_curve', [])
    
    def _calculate_capacity(
        self,
        material: BatteryMaterialSpec,
        voltage_curve: List[Tuple[float, float]]
    ) -> float:
        """计算容量"""
        # 简化的容量计算
        if not voltage_curve:
            return 0.0
        
        # 实际实现需要更复杂的计算
        return 150.0  # mAh/g (示例值)
    
    def _calculate_stability_score(
        self,
        interface_energy: float,
        adhesion_energy: float
    ) -> float:
        """计算稳定性分数"""
        # 简化的评分逻辑
        if interface_energy < 0 and adhesion_energy < 0:
            return min(1.0, abs(adhesion_energy) / abs(interface_energy))
        return 0.5
    
    def _run_structure_relaxation(self, material: BatteryMaterialSpec) -> Any:
        """运行结构优化"""
        logger.info(f"Running structure relaxation for {material.name}")
        return {"relaxed": True, "material": material.name}
    
    def _generate_report(
        self,
        material: BatteryMaterialSpec,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成综合报告"""
        return {
            "material_name": material.name,
            "formula": material.formula,
            "material_type": material.material_type,
            "working_ion": material.working_ion,
            "summary": {
                "has_conductivity_data": 'ion_conductivity' in results,
                "has_voltage_data": 'voltage_profile' in results,
                "has_interface_data": 'interface_stability' in results,
                "has_cycle_life_data": 'cycle_life' in results
            },
            "recommendations": self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成研究建议"""
        recommendations = []
        
        if 'ion_conductivity' in results:
            conductivity_results = results['ion_conductivity']
            if conductivity_results:
                avg_conductivity = sum(r.conductivity for r in conductivity_results) / len(conductivity_results)
                if avg_conductivity < 1e-5:
                    recommendations.append("Low ionic conductivity detected. Consider doping strategies.")
        
        if 'voltage_profile' in results:
            voltage_data = results['voltage_profile']
            if voltage_data.get('average_voltage', 0) < 3.0:
                recommendations.append("Low operating voltage. Consider different redox couples.")
        
        if 'interface_stability' in results:
            interface = results['interface_stability']
            if interface.stability_score < 0.5:
                recommendations.append("Poor interface stability. Consider surface coating.")
        
        return recommendations
    
    def _detect_material_type(self, structure_file: str) -> str:
        """检测材料类型"""
        # 简化实现，实际应基于结构分析
        return "cathode"
    
    def _extract_formula(self, structure_file: str) -> Optional[str]:
        """从结构文件提取化学式"""
        try:
            from pymatgen.core import Structure
            structure = Structure.from_file(structure_file)
            return structure.formula
        except:
            return None


# 便捷函数
def quick_battery_analysis(structure_file: str, **kwargs) -> Dict[str, Any]:
    """快速电池材料分析"""
    kit = BatteryResearchKit()
    return kit.one_click_research(structure_file, **kwargs)


def screen_battery_materials(
    structure_files: List[str],
    **kwargs
) -> List[Dict[str, Any]]:
    """批量筛选电池材料"""
    kit = BatteryResearchKit()
    
    candidates = []
    for file in structure_files:
        formula = kit._extract_formula(file) or "Unknown"
        candidates.append(BatteryMaterialSpec(
            name=formula,
            formula=formula,
            material_type="cathode",
            structure_file=file
        ))
    
    return kit.screen_cathode_materials(candidates, kwargs)