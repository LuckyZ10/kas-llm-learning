"""
组装电池研究工作流示例
=======================

演示如何使用编排系统组装和执行电池研究工作流。

Author: DFT-LAMMPS Team
Phase: 56 - Orchestration System
"""

import logging
from typing import Any, Dict, List, Optional

from ..orchestration.module_registry import (
    ModuleRegistry, CapabilityType, module, capability
)
from ..orchestration.capability_graph import CapabilityGraph, build_graph_from_registry
from ..orchestration.workflow_composer import (
    Workflow, WorkflowStep, WorkflowType,
    DeclarativeComposer, CodeBasedComposer, SmartComposer, WorkflowExecutor
)
from ..orchestration.topic_template import (
    TopicTemplateManager, ResearchTopic, create_battery_workflow
)
from ..integration_layer.event_bus import EventBus, EventType, event_handler
from ..integration_layer.state_manager import StateManager, WorkflowState
from ..integration_layer.resource_scheduler import ResourceScheduler


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("compose_battery_workflow")


def example_1_declarative_battery_workflow():
    """
    示例1: 使用声明式方式创建电池研究工作流
    
    通过YAML定义工作流
    """
    print("=" * 60)
    print("示例1: 声明式电池研究工作流")
    print("=" * 60)
    
    yaml_definition = """
id: battery_analysis_workflow
name: LiFePO4 Battery Analysis
version: "1.0.0"
type: dag
description: Complete analysis workflow for LiFePO4 cathode material

parameters:
  structure_file: LiFePO4.cif
  working_ion: Li
  temperature: 300
  pressure: 1.0

steps:
  - id: import_structure
    name: Import Crystal Structure
    module: io
    capability: read_structure
    inputs:
      file_path: $structure_file
    outputs:
      structure: crystal_structure

  - id: relax_structure
    name: Structure Relaxation
    module: vasp
    capability: relax_structure
    inputs:
      structure: $crystal_structure
    outputs:
      relaxed_structure: optimized_structure
      energy: total_energy
    parameters:
      encut: 520
      kpoints: [5, 5, 5]
      force_threshold: 0.01
    depends_on: [import_structure]

  - id: calculate_dos
    name: Electronic DOS
    module: vasp
    capability: calculate_dos
    inputs:
      structure: $optimized_structure
    outputs:
      dos: electronic_dos
      band_gap: band_gap
    depends_on: [relax_structure]

  - id: neb_calculation
    name: Ion Migration Barrier
    module: vasp
    capability: neb_calculation
    inputs:
      structure: $optimized_structure
      ion: $working_ion
    outputs:
      barrier: migration_barrier
      path: diffusion_path
    parameters:
      images: 7
      spring_constant: -5
    depends_on: [relax_structure]

  - id: aimd_simulation
    name: AIMD for Ion Diffusion
    module: lammps
    capability: run_aimd
    inputs:
      structure: $optimized_structure
      temperature: $temperature
    outputs:
      trajectory: md_trajectory
      msd: mean_squared_displacement
    parameters:
      timestep: 2.0
      steps: 10000
      ensemble: nvt
    depends_on: [relax_structure]

  - id: analyze_conductivity
    name: Ionic Conductivity Analysis
    module: analysis
    capability: analyze_conductivity
    inputs:
      trajectory: $md_trajectory
      msd: $mean_squared_displacement
      temperature: $temperature
    outputs:
      conductivity: ionic_conductivity
      diffusion_coeff: diffusion_coefficient
    depends_on: [aimd_simulation]

  - id: voltage_profile
    name: Calculate Voltage Profile
    module: analysis
    capability: voltage_profile
    inputs:
      structures: $optimized_structure
      working_ion: $working_ion
    outputs:
      voltage_curve: voltage_profile
      capacity: theoretical_capacity
    depends_on: [relax_structure]

  - id: generate_report
    name: Generate Final Report
    module: reporting
    capability: generate_report
    inputs:
      band_gap: $band_gap
      migration_barrier: $migration_barrier
      conductivity: $ionic_conductivity
      voltage_curve: $voltage_profile
    outputs:
      report: final_report
    depends_on: [calculate_dos, neb_calculation, analyze_conductivity, voltage_profile]

input_schema:
  structure_file:
    type: string
    description: Path to input structure file
  working_ion:
    type: string
    default: Li
  temperature:
    type: number
    default: 300

output_schema:
  final_report:
    type: file
    format: pdf
  ionic_conductivity:
    type: number
    unit: S/cm
  theoretical_capacity:
    type: number
    unit: mAh/g
"""
    
    # 使用声明式组合器
    composer = DeclarativeComposer()
    workflow = composer.compose(yaml_content=yaml_definition)
    
    # 验证工作流
    valid, errors = workflow.validate()
    print(f"\n工作流验证: {'通过' if valid else '失败'}")
    if not valid:
        print(f"错误: {errors}")
        return
    
    print(f"工作流名称: {workflow.name}")
    print(f"步骤数量: {len(workflow.steps)}")
    print(f"工作流类型: {workflow.workflow_type.value}")
    
    # 显示步骤依赖图
    print("\n步骤依赖关系:")
    for step in workflow.steps:
        deps = step.depends_on if step.depends_on else "无"
        print(f"  - {step.name} (依赖: {deps})")
    
    return workflow


def example_2_code_based_battery_workflow():
    """
    示例2: 使用代码方式创建电池研究工作流
    
    通过Python代码直接构建工作流
    """
    print("\n" + "=" * 60)
    print("示例2: 代码式电池研究工作流")
    print("=" * 60)
    
    composer = CodeBasedComposer()
    
    # 使用上下文管理器定义工作流
    with composer.workflow(
        name="NMC Cathode Analysis",
        description="Analysis of NMC (LiNiMnCoO2) cathode material",
        version="1.0.0",
        workflow_type=WorkflowType.DAG
    ) as wf:
        
        # 步骤1: 导入结构
        wf.step(
            name="Import Structure",
            module="io",
            capability="read_structure",
            inputs={"file_path": "NMC.cif"},
            outputs={"structure": "structure"}
        )
        
        # 步骤2: 结构优化
        wf.step(
            name="Relax Structure",
            module="vasp",
            capability="relax_structure",
            inputs={"structure": "$structure"},
            outputs={"relaxed_structure": "relaxed"},
            parameters={"encut": 520, "kpoints": [4, 4, 4]},
            depends_on=["Import Structure"]
        )
        
        # 步骤3: 计算不同锂化状态
        wf.step(
            name="Lithiated States",
            module="vasp",
            capability="calculate_energy",
            inputs={"structure": "$relaxed"},
            outputs={"energies": "lithiation_energies"},
            depends_on=["Relax Structure"]
        )
        
        # 步骤4: 电子态密度
        wf.step(
            name="Electronic Structure",
            module="vasp",
            capability="calculate_dos",
            inputs={"structure": "$relaxed"},
            outputs={"dos": "dos", "band_gap": "gap"},
            depends_on=["Relax Structure"]
        )
        
        # 步骤5: 过渡金属氧化态分析
        wf.step(
            name="Oxidation States",
            module="analysis",
            capability="analyze_oxidation_states",
            inputs={"dos": "$dos", "structure": "$relaxed"},
            outputs={"oxidation_states": "oxidation"},
            depends_on=["Electronic Structure"]
        )
        
        # 步骤6: 电压曲线
        wf.step(
            name="Voltage Profile",
            module="analysis",
            capability="voltage_profile",
            inputs={"energies": "$lithiation_energies"},
            outputs={"voltage": "voltage_curve"},
            depends_on=["Lithiated States"]
        )
        
        # 步骤7: 热稳定性分析
        wf.step(
            name="Thermal Stability",
            module="analysis",
            capability="decomposition_analysis",
            inputs={"structure": "$relaxed", "energies": "$lithiation_energies"},
            outputs={"stability": "thermal_stability"},
            depends_on=["Lithiated States", "Relax Structure"]
        )
        
        # 步骤8: 生成报告
        wf.step(
            name="Final Report",
            module="reporting",
            capability="generate_report",
            inputs={
                "voltage": "$voltage_curve",
                "stability": "$thermal_stability",
                "oxidation": "$oxidation"
            },
            outputs={"report": "final_report"},
            depends_on=["Voltage Profile", "Thermal Stability", "Oxidation States"]
        )
    
    workflow = composer.compose()
    
    print(f"工作流名称: {workflow.name}")
    print(f"步骤数量: {len(workflow.steps)}")
    
    # 导出为YAML
    yaml_content = composer.export(workflow, format="yaml")
    print("\n生成的YAML定义（前50行）:")
    print("-" * 40)
    print('\n'.join(yaml_content.split('\n')[:50]))
    
    return workflow


def example_3_smart_composition():
    """
    示例3: 使用智能组合器自动创建工作流
    
    基于目标自动推荐和组合模块
    """
    print("\n" + "=" * 60)
    print("示例3: 智能电池工作流组合")
    print("=" * 60)
    
    # 首先构建能力图谱
    registry = ModuleRegistry.get_instance()
    graph = build_graph_from_registry(registry)
    
    print(f"能力图谱统计:")
    stats = graph.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 使用智能组合器
    smart_composer = SmartComposer(registry, graph)
    
    # 基于目标自动组合工作流
    workflow = smart_composer.compose(
        goal="analyze_battery_ion_conductivity",
        inputs={
            "structure_file": "LiFePO4.cif",
            "temperature": 300
        },
        preferences={
            "accuracy": "high",
            "parallel": True
        }
    )
    
    print(f"\n智能生成的工作流:")
    print(f"  名称: {workflow.name}")
    print(f"  步骤: {len(workflow.steps)}")
    
    print("\n自动选择的步骤:")
    for i, step in enumerate(workflow.steps, 1):
        print(f"  {i}. {step.name}")
        if step.module_name:
            print(f"     模块: {step.module_name}")
        if step.capability_name:
            print(f"     能力: {step.capability_name}")
    
    return workflow


def example_4_execute_battery_workflow():
    """
    示例4: 执行电池研究工作流
    
    演示完整的工作流执行流程
    """
    print("\n" + "=" * 60)
    print("示例4: 执行电池研究工作流")
    print("=" * 60)
    
    # 使用课题模板快速创建工作流
    workflow = create_battery_workflow(
        structure_file="LiFePO4.cif",
        temperature=300,
        include_interface=False,
        include_cycle_life=False
    )
    
    print(f"从模板创建的工作流: {workflow.name}")
    
    # 设置执行环境
    executor = WorkflowExecutor()
    state_manager = StateManager()
    event_bus = EventBus.get_instance()
    
    # 注册事件处理器
    @event_handler(EventType.WORKFLOW_STEP_STARTED)
    def on_step_started(event):
        print(f"  [EVENT] 步骤开始: {event.data.get('step_name')}")
    
    @event_handler(EventType.WORKFLOW_STEP_COMPLETED)
    def on_step_completed(event):
        print(f"  [EVENT] 步骤完成: {event.data.get('step_name')}")
    
    event_bus.start()
    
    # 执行工作流
    print("\n开始执行工作流...")
    execution = executor.execute(
        workflow,
        inputs={
            "structure_file": "LiFePO4.cif",
            "temperature": 300
        },
        dry_run=True  # 演示模式，不实际执行计算
    )
    
    print(f"\n执行状态: {execution.status}")
    print(f"执行ID: {execution.execution_id}")
    
    if execution.start_time and execution.end_time:
        duration = execution.end_time - execution.start_time
        print(f"执行时间: {duration:.2f}秒")
    
    # 清理
    event_bus.stop()
    
    return execution


def example_5_cross_module_battery_workflow():
    """
    示例5: 跨模块电池工作流
    
    展示如何组合多个计算引擎（VASP + LAMMPS + ML）
    """
    print("\n" + "=" * 60)
    print("示例5: 跨模块电池工作流")
    print("=" * 60)
    
    composer = CodeBasedComposer()
    
    with composer.workflow(
        name="Multi-Scale Battery Analysis",
        description="Combining DFT, MD, and ML for battery research",
        workflow_type=WorkflowType.DAG
    ) as wf:
        
        # DFT部分
        wf.step(
            name="DFT Structure Relaxation",
            module="vasp",
            capability="relax_structure",
            inputs={"structure": "$input_structure"},
            outputs={"relaxed": "dft_structure"}
        )
        
        wf.step(
            name="DFT Electronic Properties",
            module="vasp",
            capability="calculate_dos",
            inputs={"structure": "$dft_structure"},
            outputs={"electronic": "electronic_data"},
            depends_on=["DFT Structure Relaxation"]
        )
        
        # MD部分（使用DFT优化的结构）
        wf.step(
            name="MD Force Field Training",
            module="ml",
            capability="train_force_field",
            inputs={"dft_data": "$electronic_data"},
            outputs={"force_field": "ff_params"},
            depends_on=["DFT Electronic Properties"]
        )
        
        wf.step(
            name="Large Scale MD",
            module="lammps",
            capability="run_md",
            inputs={
                "structure": "$dft_structure",
                "force_field": "$ff_params"
            },
            outputs={"trajectory": "large_trajectory"},
            depends_on=["MD Force Field Training", "DFT Structure Relaxation"]
        )
        
        # ML部分
        wf.step(
            name="ML Property Prediction",
            module="ml",
            capability="predict_properties",
            inputs={"electronic": "$electronic_data"},
            outputs={"predictions": "ml_predictions"},
            depends_on=["DFT Electronic Properties"]
        )
        
        # 综合分析
        wf.step(
            name="Integrate Results",
            module="analysis",
            capability="multi_scale_analysis",
            inputs={
                "dft_data": "$electronic_data",
                "md_trajectory": "$large_trajectory",
                "ml_predictions": "$ml_predictions"
            },
            outputs={"integrated": "final_results"},
            depends_on=["DFT Electronic Properties", "Large Scale MD", "ML Property Prediction"]
        )
    
    workflow = composer.compose()
    
    print(f"跨模块工作流: {workflow.name}")
    print(f"步骤数: {len(workflow.steps)}")
    print("\n跨模块流程:")
    print("  DFT (VASP) → ML训练 → MD (LAMMPS)")
    print("       ↓           ↓")
    print("  电子性质 ← ML预测 → 综合报告")
    
    return workflow


def run_all_examples():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("DFT-LAMMPS 电池工作流编排示例")
    print("=" * 60)
    
    try:
        example_1_declarative_battery_workflow()
    except Exception as e:
        print(f"示例1错误: {e}")
    
    try:
        example_2_code_based_battery_workflow()
    except Exception as e:
        print(f"示例2错误: {e}")
    
    try:
        example_3_smart_composition()
    except Exception as e:
        print(f"示例3错误: {e}")
    
    try:
        example_4_execute_battery_workflow()
    except Exception as e:
        print(f"示例4错误: {e}")
    
    try:
        example_5_cross_module_battery_workflow()
    except Exception as e:
        print(f"示例5错误: {e}")
    
    print("\n" + "=" * 60)
    print("所有示例运行完成")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()