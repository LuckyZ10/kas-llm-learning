"""
DFT-LAMMPS Orchestration Examples
=================================

编排示例

演示如何使用编排系统构建工作流。

Examples:
    compose_battery_workflow: 组装电池研究工作流
    cross_domain_pipeline: 跨域组合演示
    auto_select_modules: 自动模块选择

Author: DFT-LAMMPS Team
Phase: 56 - Orchestration System
"""

from .compose_battery_workflow import (
    example_1_declarative_battery_workflow,
    example_2_code_based_battery_workflow,
    example_3_smart_composition,
    example_4_execute_battery_workflow,
    example_5_cross_module_battery_workflow,
    run_all_examples
)

from .cross_domain_pipeline import (
    create_dft_ml_digitaltwin_pipeline,
    demonstrate_data_flow,
    demonstrate_active_learning_loop,
    demonstrate_multi_scale_simulation,
    run_cross_domain_examples
)

from .auto_select_modules import (
    AutoModuleSelector,
    ResearchGoal,
    ModuleRecommendation,
    WorkflowProposal,
    demo_auto_selection,
    demo_constraint_optimization,
    demo_natural_language_goals,
    run_auto_select_examples
)

__all__ = [
    # Battery Workflow Examples
    'example_1_declarative_battery_workflow',
    'example_2_code_based_battery_workflow',
    'example_3_smart_composition',
    'example_4_execute_battery_workflow',
    'example_5_cross_module_battery_workflow',
    'run_all_examples',
    
    # Cross Domain Pipeline
    'create_dft_ml_digitaltwin_pipeline',
    'demonstrate_data_flow',
    'demonstrate_active_learning_loop',
    'demonstrate_multi_scale_simulation',
    'run_cross_domain_examples',
    
    # Auto Select Modules
    'AutoModuleSelector',
    'ResearchGoal',
    'ModuleRecommendation',
    'WorkflowProposal',
    'demo_auto_selection',
    'demo_constraint_optimization',
    'demo_natural_language_goals',
    'run_auto_select_examples'
]