"""
跨域组合演示 (DFT → ML → 数字孪生)
====================================

演示如何构建跨领域的工作流管道，整合DFT计算、机器学习和数字孪生仿真。

Author: DFT-LAMMPS Team
Phase: 56 - Orchestration System
"""

import logging
from typing import Any, Dict, List, Optional

from ..orchestration.module_registry import (
    ModuleRegistry, CapabilityType, Capability
)
from ..orchestration.capability_graph import (
    CapabilityGraph, CapabilityNode, CapabilityEdge, NodeType, EdgeType
)
from ..orchestration.workflow_composer import (
    Workflow, WorkflowStep, WorkflowType,
    DeclarativeComposer, WorkflowExecutor
)
from ..orchestration.cross_module_bridge import CrossModuleBridge, StructureConverter
from ..integration_layer.unified_data_model import (
    StructureData, PropertyData, PropertyType, CalculationResultData
)
from ..integration_layer.event_bus import EventBus, EventType, publish_event
from ..integration_layer.state_manager import StateManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cross_domain_pipeline")


def create_dft_ml_digitaltwin_pipeline():
    """
    创建DFT→ML→数字孪生跨域管道
    
    展示如何整合三个领域的计算能力
    """
    print("=" * 70)
    print("跨域管道: DFT → ML → 数字孪生")
    print("=" * 70)
    
    # 初始化各层
    bridge = CrossModuleBridge()
    registry = ModuleRegistry.get_instance()
    
    # 定义YAML工作流
    yaml_definition = """
id: cross_domain_battery_pipeline
name: Cross-Domain Battery Material Pipeline
version: "2.0.0"
type: dag
description: |
  Multi-scale pipeline integrating:
  - DFT (Quantum-level accuracy)
  - ML (Surrogate modeling & screening)
  - Digital Twin (System-level simulation)

# ==================== Phase 1: DFT Foundation ====================
phase_dft:
  description: High-accuracy quantum mechanical calculations
  
steps:
  - id: dft_relaxation
    name: DFT Structure Relaxation
    domain: dft
    module: vasp
    capability: relax_structure
    inputs:
      structure: $input_structure
    outputs:
      relaxed_structure: dft_relaxed
      total_energy: dft_energy
    parameters:
      functional: PBE
      encut: 520
      kpoints: [6, 6, 6]
      precision: high

  - id: dft_electronic
    name: DFT Electronic Properties
    domain: dft
    module: vasp
    capability: calculate_bands
    inputs:
      structure: $dft_relaxed
    outputs:
      band_structure: dft_bands
      dos: dft_dos
      band_gap: dft_gap
    parameters:
      functional: HSE06
      hf_ratio: 0.25
    depends_on: [dft_relaxation]

  - id: dft_phonon
    name: DFT Phonon Calculation
    domain: dft
    module: vasp
    capability: phonon_calculation
    inputs:
      structure: $dft_relaxed
    outputs:
      phonon_dos: dft_phonon
      vibrational_entropy: vib_entropy
    depends_on: [dft_relaxation]

  - id: dft_aimd
    name: Ab Initio MD
    domain: dft
    module: vasp
    capability: run_aimd
    inputs:
      structure: $dft_relaxed
      temperature: $temperature
    outputs:
      trajectory: dft_trajectory
      msd: dft_msd
    parameters:
      timestep: 2.0
      steps: 5000
      ensemble: NVT
    depends_on: [dft_relaxation]

# ==================== Phase 2: ML Model Training ====================
phase_ml:
  description: Machine learning surrogate model development
  
steps:
  - id: ml_feature_extraction
    name: ML Feature Extraction
    domain: ml
    module: ml
    capability: extract_features
    inputs:
      structure: $dft_relaxed
      electronic: $dft_bands
      phonon: $dft_phonon
    outputs:
      features: ml_features
    depends_on: [dft_electronic, dft_phonon]

  - id: ml_training
    name: Train ML Surrogate Model
    domain: ml
    module: ml
    capability: train_model
    inputs:
      features: $ml_features
      targets:
        energy: $dft_energy
        band_gap: $dft_gap
    outputs:
      trained_model: ml_model
      validation_metrics: ml_metrics
    parameters:
      model_type: GNN
      epochs: 1000
      learning_rate: 0.001
    depends_on: [ml_feature_extraction]

  - id: ml_screening
    name: High-Throughput ML Screening
    domain: ml
    module: ml
    capability: screen_materials
    inputs:
      model: $ml_model
      candidate_database: $candidate_structures
    outputs:
      top_candidates: ml_candidates
      predicted_properties: ml_predictions
    parameters:
      top_k: 100
      confidence_threshold: 0.9
    depends_on: [ml_training]

  - id: ml_force_field
    name: Train ML Force Field
    domain: ml
    module: ml
    capability: train_force_field
    inputs:
      dft_trajectory: $dft_trajectory
      dft_forces: $dft_forces
    outputs:
      force_field: ml_ff
      ff_accuracy: ff_metrics
    depends_on: [dft_aimd]

# ==================== Phase 3: Digital Twin ====================
phase_digital_twin:
  description: System-level simulation and optimization
  
steps:
  - id: dt_model_calibration
    name: Calibrate Digital Twin
    domain: digital_twin
    module: digital_twin
    capability: calibrate_model
    inputs:
      ml_model: $ml_model
      ml_force_field: $ml_ff
      experimental_data: $exp_validation
    outputs:
      calibrated_model: dt_model
      calibration_error: dt_error
    depends_on: [ml_training, ml_force_field]

  - id: dt_meso_simulation
    name: Mesoscale Simulation
    domain: digital_twin
    module: digital_twin
    capability: meso_simulation
    inputs:
      model: $dt_model
      microstructure: $microstructure
      force_field: $ml_ff
    outputs:
      effective_properties: meso_props
      stress_strain: meso_mechanical
    parameters:
      rve_size: [100, 100, 100]  # nm
      boundary_conditions: periodic
    depends_on: [dt_model_calibration]

  - id: dt_battery_cell
    name: Battery Cell Simulation
    domain: digital_twin
    module: digital_twin
    capability: battery_cell_simulation
    inputs:
      material_properties: $meso_props
      electrode_geometry: $electrode_design
      electrolyte_model: $electrolyte_params
    outputs:
      cell_performance: cell_data
      discharge_curves: discharge_profiles
      heat_generation: thermal_data
    parameters:
      c_rate: [0.1, 0.5, 1.0, 2.0]
      temperature: [298, 313, 333]
    depends_on: [dt_meso_simulation]

  - id: dt_lifetime_prediction
    name: Battery Lifetime Prediction
    domain: digital_twin
    module: digital_twin
    capability: predict_lifetime
    inputs:
      cell_performance: $cell_data
      degradation_model: $degradation_params
      usage_profile: $drive_cycle
    outputs:
      capacity_fade: fade_prediction
      eol_prediction: eol_data
      warranty_analysis: warranty_report
    depends_on: [dt_battery_cell]

# ==================== Phase 4: Feedback Loop ====================
phase_feedback:
  description: Active learning and model refinement
  
steps:
  - id: active_learning
    name: Active Learning Selection
    domain: ml
    module: ml
    capability: active_learning
    inputs:
      current_model: $ml_model
      prediction_uncertainty: $ml_predictions
      digital_twin_results: $cell_data
    outputs:
      new_training_candidates: al_candidates
      uncertainty_map: uncertainty_viz
    depends_on: [ml_screening, dt_battery_cell]

  - id: dft_validation
    name: Validate with Selective DFT
    domain: dft
    module: vasp
    capability: batch_calculations
    inputs:
      candidates: $al_candidates
    outputs:
      validation_results: dft_validation_data
    parameters:
      priority: high
    depends_on: [active_learning]

  - id: model_refinement
    name: Refine ML Model
    domain: ml
    module: ml
    capability: incremental_training
    inputs:
      existing_model: $ml_model
      new_data: $dft_validation_data
    outputs:
      refined_model: ml_model_v2
      improvement_metrics: refinement_report
    depends_on: [dft_validation]

# ==================== Final Output ====================
final_steps:
  - id: integrated_report
    name: Generate Integrated Report
    domain: analysis
    module: reporting
    capability: generate_report
    inputs:
      dft_results: $dft_bands
      ml_predictions: $ml_predictions
      digital_twin: $cell_data
      lifetime: $fade_prediction
      refined_model: $ml_model_v2
    outputs:
      final_report: comprehensive_report
      design_recommendations: recommendations
    depends_on: [dt_lifetime_prediction, model_refinement]
"""
    
    composer = DeclarativeComposer()
    workflow = composer.compose(yaml_content=yaml_definition)
    
    print(f"\n跨域管道创建成功!")
    print(f"工作流名称: {workflow.name}")
    print(f"总步骤数: {len(workflow.steps)}")
    
    # 按域分组统计
    domains = {}
    for step in workflow.steps:
        domain = step.parameters.get('domain', 'unknown')
        domains[domain] = domains.get(domain, 0) + 1
    
    print("\n各域步骤统计:")
    for domain, count in domains.items():
        print(f"  {domain.upper()}: {count} 步骤")
    
    return workflow


def demonstrate_data_flow():
    """
    演示跨域数据流
    
    展示数据如何在DFT、ML和数字孪生之间流动
    """
    print("\n" + "=" * 70)
    print("跨域数据流演示")
    print("=" * 70)
    
    bridge = CrossModuleBridge()
    
    # 模拟DFT输出
    print("\n1. DFT 输出数据:")
    dft_output = {
        "energy": -100.5,  # eV
        "forces": [[0.1, -0.1, 0.0], [0.0, 0.1, -0.1]],  # eV/Ang
        "stress": [1.2, 1.2, 1.2, 0.0, 0.0, 0.0],  # kbar
        "structure": "POSCAR_format_data"
    }
    print(f"   能量: {dft_output['energy']} eV")
    print(f"   力: {len(dft_output['forces'])} 原子")
    
    # 数据转换
    print("\n2. 数据转换 (DFT → ML):")
    
    # 能量单位转换
    ml_energy = bridge.convert(
        dft_output["energy"],
        "energy_eV",
        "energy_kJ/mol"
    )
    print(f"   能量: {dft_output['energy']} eV → {ml_energy:.2f} kJ/mol")
    
    # 力单位转换
    ml_forces = bridge.convert(
        dft_output["forces"],
        "force_eV/Ang",
        "force_Ha/Bohr"
    )
    print(f"   力: eV/Å → Ha/Bohr (已转换)")
    
    # ML预测结果
    print("\n3. ML 预测输出:")
    ml_output = {
        "predicted_energy": -100.3,
        "uncertainty": 0.2,
        "confidence": 0.95,
        "band_gap_prediction": 2.1
    }
    print(f"   预测能量: {ml_output['predicted_energy']} ± {ml_output['uncertainty']} eV")
    print(f"   置信度: {ml_output['confidence']}")
    
    # ML → 数字孪生
    print("\n4. 数据适配 (ML → 数字孪生):")
    dt_input = bridge.adapt(
        {"energy": ml_output["predicted_energy"], "band_gap": ml_output["band_gap_prediction"]},
        "ml_output",
        "digital_twin_input"
    )
    print(f"   已适配为数字孪生输入格式")
    
    # 数字孪生输出
    print("\n5. 数字孪生 系统级预测:")
    dt_output = {
        "cell_capacity": 150.0,  # mAh/g
        "cycle_life": 2000,      # cycles
        "energy_density": 250.0  # Wh/kg
    }
    print(f"   电池容量: {dt_output['cell_capacity']} mAh/g")
    print(f"   循环寿命: {dt_output['cycle_life']} 次")
    print(f"   能量密度: {dt_output['energy_density']} Wh/kg")
    
    print("\n数据流示意图:")
    print("  ┌─────┐    ┌─────┐    ┌──────────┐")
    print("  │ DFT │───→│  ML │───→│ Digital  │")
    print("  │     │    │     │    │ Twin     │")
    print("  └─────┘    └─────┘    └──────────┘")
    print("     ↓          ↓            ↓")
    print("  [量子精度] [代理模型]   [系统仿真]")


def demonstrate_active_learning_loop():
    """
    演示主动学习反馈循环
    
    展示DFT和ML之间的迭代优化
    """
    print("\n" + "=" * 70)
    print("主动学习反馈循环演示")
    print("=" * 70)
    
    print("""
主动学习循环流程:

第1轮:
  ┌─────────────────────────────────────┐
  │ 1. 初始DFT计算 (10个结构)           │
  │    → 得到初始训练数据               │
  └──────────────┬──────────────────────┘
                 ↓
  ┌─────────────────────────────────────┐
  │ 2. 训练ML代理模型                   │
  │    → 验证RMSE: 0.15 eV/atom         │
  └──────────────┬──────────────────────┘
                 ↓
  ┌─────────────────────────────────────┐
  │ 3. ML筛选候选材料 (1000个结构)      │
  │    → 识别高不确定性区域             │
  └──────────────┬──────────────────────┘
                 ↓
  ┌─────────────────────────────────────┐
  │ 4. 选择性DFT验证 (10个结构)         │
  │    → 优先计算不确定结构             │
  └──────────────┬──────────────────────┘
                 ↓
第2轮:
  ┌─────────────────────────────────────┐
  │ 5. 增量学习更新ML模型               │
  │    → 验证RMSE: 0.08 eV/atom (改善!) │
  └──────────────┬──────────────────────┘
                 ↓
  ┌─────────────────────────────────────┐
  │ 6. 数字孪生验证                     │
  │    → 预测循环寿命与实验对比         │
  └──────────────┬──────────────────────┘
                 ↓
  ┌─────────────────────────────────────┐
  │ 7. 模型部署与监控                   │
  │    → 持续收集反馈数据               │
  └─────────────────────────────────────┘

优势:
- 减少90%的DFT计算需求
- 保持量子力学精度
- 加速材料筛选1000倍
""")
    
    # 模拟主动学习过程
    iterations = [
        {"round": 1, "dft_calcs": 10, "ml_rmse": 0.15, "candidates_screened": 1000},
        {"round": 2, "dft_calcs": 10, "ml_rmse": 0.08, "candidates_screened": 5000},
        {"round": 3, "dft_calcs": 10, "ml_rmse": 0.05, "candidates_screened": 10000},
    ]
    
    print("\n主动学习迭代结果:")
    print(f"{'轮次':<8} {'DFT计算':<12} {'ML RMSE (eV)':<15} {'筛选材料数':<12}")
    print("-" * 50)
    for it in iterations:
        print(f"{it['round']:<8} {it['dft_calcs']:<12} {it['ml_rmse']:<15.3f} {it['candidates_screened']:<12}")
    
    print("\n效率提升:")
    print(f"  传统方法: {sum(it['dft_calcs'] for it in iterations)} × 1000 = 30000 次DFT计算")
    print(f"  主动学习: {sum(it['dft_calcs'] for it in iterations)} 次DFT计算")
    print(f"  加速比: 1000×")


def demonstrate_multi_scale_simulation():
    """
    演示多尺度仿真集成
    """
    print("\n" + "=" * 70)
    print("多尺度仿真集成")
    print("=" * 70)
    
    scales = [
        {
            "level": "量子尺度",
            "method": "DFT (VASP)",
            "scale": "~100 原子",
            "time": "~ps",
            "properties": ["电子结构", "能带间隙", "形成能", "迁移势垒"],
            "accuracy": "高 (量子精度)"
        },
        {
            "level": "原子尺度",
            "method": "ML力场 (NEP/CHGNet)",
            "scale": "~10⁶ 原子",
            "time": "~ns",
            "properties": ["离子扩散", "相变", "缺陷形成"],
            "accuracy": "中 (DFT精度±5%)"
        },
        {
            "level": "介观尺度",
            "method": "相场/有限元",
            "scale": "~μm³",
            "time": "~ms",
            "properties": ["微观结构", "晶粒生长", "应力分布"],
            "accuracy": "中 (基于ML参数)"
        },
        {
            "level": "宏观尺度",
            "method": "数字孪生",
            "scale": "电池单体",
            "time": "~年",
            "properties": ["容量衰减", "循环寿命", "热管理"],
            "accuracy": "校准后高"
        }
    ]
    
    print("\n多尺度架构:")
    print("-" * 70)
    
    for scale in scales:
        print(f"\n【{scale['level']}】 - {scale['method']}")
        print(f"  空间尺度: {scale['scale']}")
        print(f"  时间尺度: {scale['time']}")
        print(f"  计算性质: {', '.join(scale['properties'])}")
        print(f"  精度水平: {scale['accuracy']}")
    
    print("\n" + "-" * 70)
    print("\n尺度间信息传递:")
    print("  量子 → 原子: DFT训练ML力场参数")
    print("  原子 → 介观: ML预测相图和本构关系")
    print("  介观 → 宏观: 有效性能参数输入数字孪生")
    print("  宏观 → 原子: 实测数据反馈校准ML模型")


def run_cross_domain_examples():
    """运行所有跨域示例"""
    print("\n" + "=" * 70)
    print("跨域组合演示 (DFT → ML → 数字孪生)")
    print("=" * 70)
    
    try:
        create_dft_ml_digitaltwin_pipeline()
    except Exception as e:
        print(f"管道创建错误: {e}")
    
    demonstrate_data_flow()
    demonstrate_active_learning_loop()
    demonstrate_multi_scale_simulation()
    
    print("\n" + "=" * 70)
    print("跨域演示完成")
    print("=" * 70)


if __name__ == "__main__":
    run_cross_domain_examples()