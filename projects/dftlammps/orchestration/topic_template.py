"""
DFT-LAMMPS 课题模板系统
=======================
电池/催化剂/光伏/合金等预设组合

提供针对特定研究领域的预配置工作流模板。

Author: DFT-LAMMPS Team
Phase: 56 - Orchestration System
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum

from .module_registry import ModuleRegistry, CapabilityType
from .workflow_composer import (
    Workflow, WorkflowStep, WorkflowType,
    CodeBasedComposer, DeclarativeComposer
)


logger = logging.getLogger("topic_templates")


class ResearchTopic(Enum):
    """研究课题类型"""
    BATTERY = "battery"
    CATALYST = "catalyst"
    PHOTOVOLTAIC = "photovoltaic"
    ALLOY = "alloy"
    ELECTRONIC = "electronic"
    MECHANICAL = "mechanical"
    THERMAL = "thermal"
    CUSTOM = "custom"


@dataclass
class TopicTemplate:
    """课题模板"""
    topic_type: ResearchTopic
    name: str
    description: str
    version: str = "1.0.0"
    
    # 模板配置
    required_capabilities: List[str] = field(default_factory=list)
    recommended_modules: List[str] = field(default_factory=list)
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # 输入输出定义
    input_spec: Dict[str, Any] = field(default_factory=dict)
    output_spec: Dict[str, Any] = field(default_factory=dict)
    
    # 工作流定义
    workflow_template: Optional[Workflow] = None
    
    # 元数据
    tags: List[str] = field(default_factory=list)
    author: str = ""
    references: List[str] = field(default_factory=list)
    
    def instantiate(
        self,
        inputs: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None
    ) -> Workflow:
        """
        实例化模板为具体工作流
        
        Args:
            inputs: 输入数据
            overrides: 参数覆盖
        """
        overrides = overrides or {}
        
        if not self.workflow_template:
            raise ValueError("No workflow template defined")
        
        # 复制模板
        from copy import deepcopy
        workflow = deepcopy(self.workflow_template)
        
        # 应用输入
        for key, value in inputs.items():
            # 更新步骤中的输入引用
            for step in workflow.steps:
                for input_key, input_val in step.inputs.items():
                    if input_val == f"${key}":
                        step.inputs[input_key] = value
        
        # 应用参数覆盖
        params = {**self.default_parameters, **overrides}
        workflow.global_parameters.update(params)
        
        return workflow


class TopicTemplateManager:
    """
    课题模板管理器
    
    管理和提供各类研究课题的预配置模板
    """
    
    def __init__(self):
        self._templates: Dict[ResearchTopic, TopicTemplate] = {}
        self._registry: Optional[ModuleRegistry] = None
        self._initialize_default_templates()
    
    def register_template(self, template: TopicTemplate) -> None:
        """注册模板"""
        self._templates[template.topic_type] = template
        logger.info(f"Registered template: {template.name}")
    
    def get_template(self, topic: ResearchTopic) -> Optional[TopicTemplate]:
        """获取模板"""
        return self._templates.get(topic)
    
    def list_templates(self) -> List[TopicTemplate]:
        """列出所有模板"""
        return list(self._templates.values())
    
    def create_workflow(
        self,
        topic: ResearchTopic,
        inputs: Dict[str, Any],
        **kwargs
    ) -> Workflow:
        """基于模板创建工作流"""
        template = self.get_template(topic)
        if not template:
            raise ValueError(f"Template not found for topic: {topic}")
        
        return template.instantiate(inputs, kwargs)
    
    def _initialize_default_templates(self) -> None:
        """初始化默认模板"""
        # 电池研究模板
        self.register_template(self._create_battery_template())
        
        # 催化剂模板
        self.register_template(self._create_catalyst_template())
        
        # 光伏模板
        self.register_template(self._create_photovoltaic_template())
        
        # 合金模板
        self.register_template(self._create_alloy_template())
    
    def _create_battery_template(self) -> TopicTemplate:
        """创建电池研究模板"""
        workflow = Workflow(
            id="battery_workflow_template",
            name="Battery Research Workflow",
            description="Complete workflow for battery material research",
            workflow_type=WorkflowType.DAG
        )
        
        # 步骤1: 结构导入
        workflow.steps.append(WorkflowStep(
            id="step_1",
            name="Import Structure",
            module_name="io",
            capability_name="read_structure",
            inputs={"file_path": "$input_structure"},
            outputs={"structure": "structure"}
        ))
        
        # 步骤2: 结构优化
        workflow.steps.append(WorkflowStep(
            id="step_2",
            name="Structure Relaxation",
            module_name="vasp",
            capability_name="relax_structure",
            inputs={"structure": "$structure"},
            outputs={"relaxed_structure": "relaxed_structure"},
            depends_on=["step_1"],
            parameters={"encut": 520, "kpoints": [5, 5, 5]}
        ))
        
        # 步骤3: 体积计算（用于电压曲线）
        workflow.steps.append(WorkflowStep(
            id="step_3",
            name="Calculate Volume",
            module_name="vasp",
            capability_name="calculate_volume",
            inputs={"structure": "$relaxed_structure"},
            outputs={"volume": "volume"},
            depends_on=["step_2"]
        ))
        
        # 步骤4: 电子态密度（用于理解导电性）
        workflow.steps.append(WorkflowStep(
            id="step_4",
            name="Electronic DOS",
            module_name="vasp",
            capability_name="calculate_dos",
            inputs={"structure": "$relaxed_structure"},
            outputs={"dos": "dos", "band_gap": "band_gap"},
            depends_on=["step_2"]
        ))
        
        # 步骤5: NEB计算（离子迁移）
        workflow.steps.append(WorkflowStep(
            id="step_5",
            name="NEB Calculation",
            module_name="vasp",
            capability_name="neb_calculation",
            inputs={"structure": "$relaxed_structure"},
            outputs={"migration_barrier": "migration_barrier", "path": "neb_path"},
            depends_on=["step_2"],
            parameters={"images": 7, "spring_constant": -5}
        ))
        
        # 步骤6: 分子动力学（离子扩散系数）
        workflow.steps.append(WorkflowStep(
            id="step_6",
            name="AIMD Simulation",
            module_name="lammps",
            capability_name="run_aimd",
            inputs={"structure": "$relaxed_structure", "temperature": "$temperature"},
            outputs={"trajectory": "trajectory", "msd": "msd"},
            depends_on=["step_2"],
            parameters={"timestep": 2.0, "steps": 10000, "ensemble": "nvt"}
        ))
        
        # 步骤7: 分析离子电导率
        workflow.steps.append(WorkflowStep(
            id="step_7",
            name="Ion Conductivity Analysis",
            module_name="analysis",
            capability_name="analyze_conductivity",
            inputs={"trajectory": "$trajectory", "msd": "$msd"},
            outputs={"conductivity": "conductivity", "diffusion_coeff": "diffusion_coeff"},
            depends_on=["step_6"]
        ))
        
        # 步骤8: 生成电压曲线
        workflow.steps.append(WorkflowStep(
            id="step_8",
            name="Generate Voltage Profile",
            module_name="analysis",
            capability_name="voltage_profile",
            inputs={"volumes": "$volume", "energies": "$energy"},
            outputs={"voltage_curve": "voltage_curve"},
            depends_on=["step_3"]
        ))
        
        # 步骤9: 综合报告
        workflow.steps.append(WorkflowStep(
            id="step_9",
            name="Generate Report",
            module_name="reporting",
            capability_name="generate_report",
            inputs={
                "conductivity": "$conductivity",
                "voltage_curve": "$voltage_curve",
                "migration_barrier": "$migration_barrier"
            },
            outputs={"report": "final_report"},
            depends_on=["step_7", "step_8", "step_5"]
        ))
        
        return TopicTemplate(
            topic_type=ResearchTopic.BATTERY,
            name="Battery Material Research",
            description="Complete workflow for battery electrode/electrolyte research including ion conductivity, voltage profile, and stability analysis",
            version="1.0.0",
            required_capabilities=[
                "read_structure",
                "relax_structure", 
                "calculate_volume",
                "calculate_dos",
                "neb_calculation",
                "run_aimd",
                "analyze_conductivity",
                "voltage_profile"
            ],
            recommended_modules=["vasp", "lammps", "analysis"],
            default_parameters={
                "encut": 520,
                "kpoints_density": 0.03,
                "temperature": 300,
                "neb_images": 7
            },
            input_spec={
                "input_structure": {"type": "file", "format": ["cif", "poscar", "xyz"]},
                "temperature": {"type": "float", "default": 300, "unit": "K"}
            },
            output_spec={
                "conductivity": {"type": "float", "unit": "S/cm"},
                "voltage_curve": {"type": "plot"},
                "diffusion_coeff": {"type": "float", "unit": "cm²/s"}
            },
            workflow_template=workflow,
            tags=["battery", "electrode", "electrolyte", "ion-conductivity"],
            references=[
                "Aydinol et al., PRB 1997",
                "Ceder et al., Nature 2001"
            ]
        )
    
    def _create_catalyst_template(self) -> TopicTemplate:
        """创建催化剂模板"""
        workflow = Workflow(
            id="catalyst_workflow_template",
            name="Catalyst Design Workflow",
            description="Workflow for catalyst screening and optimization",
            workflow_type=WorkflowType.DAG
        )
        
        # 步骤1: 构建表面
        workflow.steps.append(WorkflowStep(
            id="step_1",
            name="Build Surface",
            module_name="structure",
            capability_name="build_surface",
            inputs={"bulk_structure": "$input_structure", "miller_index": "$miller_index"},
            outputs={"surface": "surface"}
        ))
        
        # 步骤2: 添加吸附质
        workflow.steps.append(WorkflowStep(
            id="step_2",
            name="Add Adsorbate",
            module_name="structure",
            capability_name="add_adsorbate",
            inputs={"surface": "$surface", "adsorbate": "$adsorbate"},
            outputs={"adsorption_configs": "adsorption_configs"},
            depends_on=["step_1"]
        ))
        
        # 步骤3: 优化吸附构型
        workflow.steps.append(WorkflowStep(
            id="step_3",
            name="Relax Adsorption Configs",
            module_name="vasp",
            capability_name="relax_structure",
            inputs={"structures": "$adsorption_configs"},
            outputs={"relaxed_configs": "relaxed_configs"},
            depends_on=["step_2"],
            parameters={"encut": 450, "ispin": 2}
        ))
        
        # 步骤4: 计算吸附能
        workflow.steps.append(WorkflowStep(
            id="step_4",
            name="Calculate Adsorption Energy",
            module_name="analysis",
            capability_name="adsorption_energy",
            inputs={"adsorbed": "$relaxed_configs", "clean": "$surface", "gas": "$adsorbate"},
            outputs={"adsorption_energies": "adsorption_energies"},
            depends_on=["step_3"]
        ))
        
        # 步骤5: NEB计算（反应路径）
        workflow.steps.append(WorkflowStep(
            id="step_5",
            name="Reaction Pathway",
            module_name="vasp",
            capability_name="neb_calculation",
            inputs={"initial": "$initial_state", "final": "$final_state"},
            outputs={"barrier": "reaction_barrier"},
            depends_on=["step_3"]
        ))
        
        # 步骤6: d带中心计算
        workflow.steps.append(WorkflowStep(
            id="step_6",
            name="d-band Center",
            module_name="vasp",
            capability_name="calculate_dos",
            inputs={"structure": "$surface"},
            outputs={"d_band_center": "d_band_center"},
            depends_on=["step_1"]
        ))
        
        # 步骤7: Volcano图生成
        workflow.steps.append(WorkflowStep(
            id="step_7",
            name="Generate Volcano Plot",
            module_name="analysis",
            capability_name="volcano_plot",
            inputs={"adsorption_energies": "$adsorption_energies", "d_band_center": "$d_band_center"},
            outputs={"volcano_plot": "volcano_plot"},
            depends_on=["step_4", "step_6"]
        ))
        
        # 步骤8: 选择性分析
        workflow.steps.append(WorkflowStep(
            id="step_8",
            name="Selectivity Analysis",
            module_name="analysis",
            capability_name="selectivity_analysis",
            inputs={
                "barriers": "$reaction_barrier",
                "adsorption_energies": "$adsorption_energies"
            },
            outputs={"selectivity": "selectivity"},
            depends_on=["step_5", "step_4"]
        ))
        
        return TopicTemplate(
            topic_type=ResearchTopic.CATALYST,
            name="Catalyst Design",
            description="Catalyst screening workflow including adsorption energy, volcano plot, and selectivity analysis",
            version="1.0.0",
            required_capabilities=[
                "build_surface",
                "add_adsorbate",
                "relax_structure",
                "adsorption_energy",
                "neb_calculation",
                "calculate_dos",
                "volcano_plot"
            ],
            recommended_modules=["vasp", "structure", "analysis"],
            default_parameters={
                "surface_layers": 4,
                "vacuum": 15.0,
                "adsorption_height": 2.0
            },
            input_spec={
                "input_structure": {"type": "file", "format": ["cif", "poscar"]},
                "miller_index": {"type": "list", "default": [1, 1, 1]},
                "adsorbate": {"type": "string", "options": ["H", "O", "N", "CO", "OH", "OOH"]}
            },
            output_spec={
                "adsorption_energies": {"type": "dict"},
                "volcano_plot": {"type": "plot"},
                "selectivity": {"type": "float"}
            },
            workflow_template=workflow,
            tags=["catalyst", "surface", "adsorption", "volcano", "selectivity"],
            references=[
                "Nørskov et al., JPCB 2004",
                "Hammer & Nørskov, Advances in Catalysis 2000"
            ]
        )
    
    def _create_photovoltaic_template(self) -> TopicTemplate:
        """创建光伏模板"""
        workflow = Workflow(
            id="pv_workflow_template",
            name="Photovoltaic Material Workflow",
            description="Workflow for solar cell material screening",
            workflow_type=WorkflowType.DAG
        )
        
        # 步骤1: 结构优化
        workflow.steps.append(WorkflowStep(
            id="step_1",
            name="Structure Relaxation",
            module_name="vasp",
            capability_name="relax_structure",
            inputs={"structure": "$input_structure"},
            outputs={"relaxed_structure": "relaxed_structure"}
        ))
        
        # 步骤2: 能带计算（HSE06）
        workflow.steps.append(WorkflowStep(
            id="step_2",
            name="Band Structure (HSE06)",
            module_name="vasp",
            capability_name="calculate_bands",
            inputs={"structure": "$relaxed_structure"},
            outputs={"band_structure": "band_structure", "band_gap": "band_gap"},
            depends_on=["step_1"],
            parameters={"functional": "HSE06", "hf_ratio": 0.25}
        ))
        
        # 步骤3: 光学性质计算
        workflow.steps.append(WorkflowStep(
            id="step_3",
            name="Optical Properties",
            module_name="vasp",
            capability_name="optical_properties",
            inputs={"structure": "$relaxed_structure"},
            outputs={"dielectric": "dielectric", "absorption": "absorption"},
            depends_on=["step_1"]
        ))
        
        # 步骤4: 有效质量计算
        workflow.steps.append(WorkflowStep(
            id="step_4",
            name="Effective Mass",
            module_name="analysis",
            capability_name="effective_mass",
            inputs={"band_structure": "$band_structure"},
            outputs={"hole_mass": "hole_mass", "electron_mass": "electron_mass"},
            depends_on=["step_2"]
        ))
        
        # 步骤5: 激子结合能
        workflow.steps.append(WorkflowStep(
            id="step_5",
            name="Exciton Binding Energy",
            module_name="vasp",
            capability_name="bse_calculation",
            inputs={"structure": "$relaxed_structure", "dielectric": "$dielectric"},
            outputs={"exciton_energy": "exciton_energy"},
            depends_on=["step_1", "step_3"]
        ))
        
        # 步骤6: 载流子寿命估计
        workflow.steps.append(WorkflowStep(
            id="step_6",
            name="Carrier Lifetime",
            module_name="analysis",
            capability_name="carrier_lifetime",
            inputs={
                "effective_masses": ["$hole_mass", "$electron_mass"],
                "band_gap": "$band_gap"
            },
            outputs={"lifetime": "carrier_lifetime"},
            depends_on=["step_4", "step_2"]
        ))
        
        # 步骤7: 光伏效率预测
        workflow.steps.append(WorkflowStep(
            id="step_7",
            name="Solar Cell Efficiency",
            module_name="analysis",
            capability_name="predict_efficiency",
            inputs={
                "band_gap": "$band_gap",
                "absorption": "$absorption",
                "carrier_lifetime": "$carrier_lifetime"
            },
            outputs={"efficiency": "predicted_efficiency"},
            depends_on=["step_2", "step_3", "step_6"]
        ))
        
        return TopicTemplate(
            topic_type=ResearchTopic.PHOTOVOLTAIC,
            name="Photovoltaic Material Screening",
            description="Solar cell material screening including band gap, absorption, and efficiency prediction",
            version="1.0.0",
            required_capabilities=[
                "relax_structure",
                "calculate_bands",
                "optical_properties",
                "effective_mass",
                "bse_calculation",
                "carrier_lifetime"
            ],
            recommended_modules=["vasp", "analysis"],
            default_parameters={
                "functional": "HSE06",
                "nbands": 64,
                "nedos": 5000
            },
            input_spec={
                "input_structure": {"type": "file", "format": ["cif", "poscar"]}
            },
            output_spec={
                "band_gap": {"type": "float", "unit": "eV"},
                "absorption": {"type": "array"},
                "predicted_efficiency": {"type": "float", "unit": "%"}
            },
            workflow_template=workflow,
            tags=["photovoltaic", "solar-cell", "band-gap", "absorption"],
            references=[
                "Shockley & Queisser, JAP 1961",
                "Yu & Zunger, PRL 2012"
            ]
        )
    
    def _create_alloy_template(self) -> TopicTemplate:
        """创建合金模板"""
        workflow = Workflow(
            id="alloy_workflow_template",
            name="Alloy Design Workflow",
            description="Workflow for high-entropy alloy design",
            workflow_type=WorkflowType.DAG
        )
        
        # 步骤1: 生成SQS结构
        workflow.steps.append(WorkflowStep(
            id="step_1",
            name="Generate SQS",
            module_name="structure",
            capability_name="generate_sqs",
            inputs={"composition": "$composition", "structure_type": "$structure_type"},
            outputs={"sqs_structures": "sqs_structures"}
        ))
        
        # 步骤2: 训练CE模型
        workflow.steps.append(WorkflowStep(
            id="step_2",
            name="Train Cluster Expansion",
            module_name="ce",
            capability_name="fit_cluster_expansion",
            inputs={"structures": "$sqs_structures", "energies": "$training_energies"},
            outputs={"ce_model": "ce_model"},
            depends_on=["step_1"]
        ))
        
        # 步骤3: 蒙特卡洛模拟
        workflow.steps.append(WorkflowStep(
            id="step_3",
            name="Monte Carlo Simulation",
            module_name="ce",
            capability_name="monte_carlo",
            inputs={"ce_model": "$ce_model", "temperatures": "$temperatures"},
            outputs={"phase_diagram": "phase_diagram"},
            depends_on=["step_2"]
        ))
        
        # 步骤4: 力学性质计算
        workflow.steps.append(WorkflowStep(
            id="step_4",
            name="Elastic Constants",
            module_name="vasp",
            capability_name="elastic_constants",
            inputs={"structure": "$sqs_structures"},
            outputs={"elastic_tensor": "elastic_tensor"},
            depends_on=["step_1"]
        ))
        
        # 步骤5: 电子结构计算
        workflow.steps.append(WorkflowStep(
            id="step_5",
            name="Electronic Structure",
            module_name="vasp",
            capability_name="calculate_dos",
            inputs={"structure": "$sqs_structures"},
            outputs={"dos": "dos", "fermi_level": "fermi_level"},
            depends_on=["step_1"]
        ))
        
        # 步骤6: 硬度预测
        workflow.steps.append(WorkflowStep(
            id="step_6",
            name="Predict Hardness",
            module_name="analysis",
            capability_name="predict_hardness",
            inputs={"elastic_tensor": "$elastic_tensor", "composition": "$composition"},
            outputs={"hardness": "predicted_hardness"},
            depends_on=["step_4"]
        ))
        
        # 步骤7: 腐蚀抗性分析
        workflow.steps.append(WorkflowStep(
            id="step_7",
            name="Corrosion Analysis",
            module_name="analysis",
            capability_name="corrosion_resistance",
            inputs={"composition": "$composition", "dos": "$dos"},
            outputs={"corrosion_index": "corrosion_index"},
            depends_on=["step_5"]
        ))
        
        # 步骤8: 生成相图
        workflow.steps.append(WorkflowStep(
            id="step_8",
            name="Generate Phase Diagram",
            module_name="analysis",
            capability_name="plot_phase_diagram",
            inputs={"phase_diagram": "$phase_diagram"},
            outputs={"phase_plot": "phase_plot"},
            depends_on=["step_3"]
        ))
        
        return TopicTemplate(
            topic_type=ResearchTopic.ALLOY,
            name="High-Entropy Alloy Design",
            description="HEA design workflow including phase diagram, mechanical properties, and corrosion resistance",
            version="1.0.0",
            required_capabilities=[
                "generate_sqs",
                "fit_cluster_expansion",
                "monte_carlo",
                "elastic_constants",
                "calculate_dos",
                "predict_hardness",
                "corrosion_resistance"
            ],
            recommended_modules=["vasp", "ce", "structure", "analysis"],
            default_parameters={
                "sqs_size": 100,
                "mc_steps": 10000,
                "temperature_range": [300, 2000]
            },
            input_spec={
                "composition": {"type": "dict", "example": {"Co": 0.2, "Cr": 0.2, "Fe": 0.2, "Mn": 0.2, "Ni": 0.2}},
                "structure_type": {"type": "string", "options": ["fcc", "bcc", "hcp"]}
            },
            output_spec={
                "phase_diagram": {"type": "plot"},
                "predicted_hardness": {"type": "float", "unit": "GPa"},
                "corrosion_index": {"type": "float"}
            },
            workflow_template=workflow,
            tags=["alloy", "hea", "phase-diagram", "mechanical-properties"],
            references=[
                "Cantor et al., Materials Science and Engineering A 2004",
                "Yeh et al., Advanced Engineering Materials 2004"
            ]
        )


# 便捷函数
def get_template_manager() -> TopicTemplateManager:
    """获取模板管理器实例"""
    return TopicTemplateManager()


def create_battery_workflow(
    structure_file: str,
    temperature: float = 300.0,
    **kwargs
) -> Workflow:
    """便捷函数：创建电池研究工作流"""
    manager = get_template_manager()
    return manager.create_workflow(
        ResearchTopic.BATTERY,
        inputs={"input_structure": structure_file, "temperature": temperature},
        **kwargs
    )


def create_catalyst_workflow(
    structure_file: str,
    miller_index: List[int],
    adsorbate: str,
    **kwargs
) -> Workflow:
    """便捷函数：创建催化剂工作流"""
    manager = get_template_manager()
    return manager.create_workflow(
        ResearchTopic.CATALYST,
        inputs={
            "input_structure": structure_file,
            "miller_index": miller_index,
            "adsorbate": adsorbate
        },
        **kwargs
    )


def create_photovoltaic_workflow(
    structure_file: str,
    **kwargs
) -> Workflow:
    """便捷函数：创建光伏材料工作流"""
    manager = get_template_manager()
    return manager.create_workflow(
        ResearchTopic.PHOTOVOLTAIC,
        inputs={"input_structure": structure_file},
        **kwargs
    )


def create_alloy_workflow(
    composition: Dict[str, float],
    structure_type: str = "fcc",
    **kwargs
) -> Workflow:
    """便捷函数：创建合金设计工作流"""
    manager = get_template_manager()
    return manager.create_workflow(
        ResearchTopic.ALLOY,
        inputs={"composition": composition, "structure_type": structure_type},
        **kwargs
    )