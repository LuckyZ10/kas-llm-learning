#!/usr/bin/env python3
"""
主动学习工作流示例
Active Learning Workflow Example
"""

import sys
sys.path.insert(0, '/root/.openclaw/workspace/dft_lammps_research')

from code_templates.active_learning_workflow import (
    ActiveLearningWorkflow,
    ActiveLearningConfig,
    UncertaintyConfig,
    ExplorationConfig,
    DeepMDConfig
)
from ase.io import read
import yaml


def load_config(config_file: str) -> ActiveLearningConfig:
    """从YAML加载配置"""
    with open(config_file) as f:
        data = yaml.safe_load(f)
    
    return ActiveLearningConfig(
        max_iterations=data['max_iterations'],
        uncertainty=UncertaintyConfig(
            f_trust_lo=data['uncertainty']['f_trust_lo'],
            f_trust_hi=data['uncertainty']['f_trust_hi'],
            adaptive_threshold=data['uncertainty']['adaptive_threshold']
        ),
        exploration=ExplorationConfig(
            temperature_range=tuple(data['exploration']['temperature_range']),
            pressure_range=tuple(data['exploration']['pressure_range'])
        ),
        deepmd=DeepMDConfig(
            type_map=data['deepmd']['type_map'],
            rcut=data['deepmd']['rcut'],
            numb_steps=data['deepmd']['numb_steps']
        ),
        work_dir=data['work_dir']
    )


def main():
    """主动学习示例主函数"""
    
    print("="*60)
    print("主动学习工作流示例")
    print("Active Learning Workflow Example")
    print("="*60)
    
    # 加载配置
    print("\n[1/4] 加载配置...")
    # config = load_config("config.yaml")
    config = ActiveLearningConfig(
        max_iterations=5,  # 演示用较少迭代
        work_dir="./al_example"
    )
    
    # 创建主动学习工作流
    print("[2/4] 初始化主动学习工作流...")
    workflow = ActiveLearningWorkflow(config)
    
    # 加载初始结构
    print("[3/4] 加载初始结构...")
    # initial_structures = [read("Li3PS4.vasp")]
    
    # 初始化 (需要DFT计算初始数据)
    print("[4/4] 运行主动学习循环...")
    print("(演示模式 - 跳过实际计算)")
    
    # workflow.initialize(initial_structures)
    # final_model = workflow.run()
    
    print("\n" + "="*60)
    print("主动学习流程示例完成!")
    print("="*60)
    print("\n完整流程:")
    print("1. 准备初始结构")
    print("2. DFT计算初始数据")
    print("3. 训练初始ML模型")
    print("4. 运行主动学习循环:")
    print("   - ML-MD探索构型空间")
    print("   - 选择不确定性高的结构")
    print("   - DFT标注候选结构")
    print("   - 重新训练模型")
    print("5. 直到收敛或达到最大迭代次数")


if __name__ == "__main__":
    main()
