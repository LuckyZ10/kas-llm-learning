#!/usr/bin/env python3
"""
NEP Training Integration Test
=============================
测试与平台的集成
"""

import os
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, '/root/.openclaw/workspace/dft_lammps_research')

from nep_training import (
    NEPWorkflowModule,
    NEPNodeExecutor,
    NEPWorkflowBuilder,
    NEPWorkflowConfig,
)


def test_module_info():
    """测试模块信息获取"""
    print("=" * 60)
    print("Test: Module Information")
    print("=" * 60)
    
    module = NEPWorkflowModule()
    info = module.get_info()
    
    print(f"Module Name: {info['name']}")
    print(f"Version: {info['version']}")
    print(f"Description: {info['description']}")
    print(f"\nFeatures ({len(info['features'])}):")
    for feature in info['features']:
        print(f"  - {feature}")
    print(f"\nPresets: {', '.join(info['presets'])}")
    
    return info


def test_workflow_builder():
    """测试工作流构建器"""
    print("\n" + "=" * 60)
    print("Test: Workflow Builder")
    print("=" * 60)
    
    builder = NEPWorkflowBuilder()
    
    # 构建简单工作流
    workflow1 = builder.build_simple_workflow(
        name="basic_training",
        input_data={'xyz_files': ['data/train.xyz']},
        output_dir="./test_output"
    )
    
    print(f"\nSimple Workflow: {workflow1['name']}")
    print(f"  Nodes: {len(workflow1['nodes'])}")
    for node in workflow1['nodes']:
        deps = node.get('depends_on', [])
        print(f"    - {node['id']} ({node['type']})"
              f"{' [depends: ' + ', '.join(deps) + ']' if deps else ''}")
    
    # 构建主动学习工作流
    workflow2 = builder.build_active_learning_workflow(
        name="al_workflow",
        initial_structures=[],
        output_dir="./al_output",
        max_iterations=5
    )
    
    print(f"\nActive Learning Workflow: {workflow2['name']}")
    print(f"  Type: {workflow2['type']}")
    print(f"  Max iterations: {workflow2['nodes'][0]['config']['max_iterations']}")
    
    # 构建迁移学习工作流
    workflow3 = builder.build_transfer_learning_workflow(
        name="transfer_workflow",
        pretrained_model="generic_solid",
        new_data={'xyz_files': ['new_data.xyz']},
        output_dir="./transfer_output"
    )
    
    print(f"\nTransfer Learning Workflow: {workflow3['name']}")
    print(f"  Type: {workflow3['type']}")
    
    return [workflow1, workflow2, workflow3]


def test_node_executor():
    """测试节点执行器"""
    print("\n" + "=" * 60)
    print("Test: Node Executor")
    print("=" * 60)
    
    config = NEPWorkflowConfig(
        model_preset="fast",
        use_ensemble=False,
        enable_monitoring=False
    )
    
    executor = NEPNodeExecutor(config)
    
    print(f"Executor created with preset: {config.model_preset}")
    print(f"  Ensemble: {config.use_ensemble}")
    print(f"  Monitoring: {config.enable_monitoring}")
    print(f"  Active Learning: {config.use_active_learning}")
    
    # 测试进度更新
    async def test_progress():
        progress_log = []
        
        async def progress_callback(progress, message):
            progress_log.append((progress, message))
            print(f"  [{progress*100:.0f}%] {message}")
        
        # 模拟执行
        await executor._update_progress(progress_callback, 0.0, "Starting...")
        await executor._update_progress(progress_callback, 0.5, "Halfway...")
        await executor._update_progress(progress_callback, 1.0, "Done!")
        
        return progress_log
    
    progress_log = asyncio.run(test_progress())
    print(f"\nProgress updates: {len(progress_log)}")
    
    return executor


def test_pretrained_models():
    """测试预训练模型"""
    print("\n" + "=" * 60)
    print("Test: Pretrained Models")
    print("=" * 60)
    
    module = NEPWorkflowModule()
    
    # 获取所有模型
    all_models = module.get_pretrained_models()
    print(f"Total models in library: {len(all_models)}")
    
    # 按元素搜索
    si_models = module.get_pretrained_models(elements=["Si"])
    print(f"Si-related models: {len(si_models)}")
    
    # 按多元素搜索
    multi_models = module.get_pretrained_models(elements=["Li", "O"])
    print(f"Li-O models: {len(multi_models)}")
    
    return all_models


def test_integration_with_existing_workflow():
    """测试与现有工作流系统的集成"""
    print("\n" + "=" * 60)
    print("Test: Integration with Existing Workflow System")
    print("=" * 60)
    
    # 模拟工作流引擎的节点格式
    workflow_node = {
        'id': 'nep_training_node',
        'type': 'nep_training',
        'data': {
            'node_type': 'nep_training',
            'label': 'NEP Training',
            'description': 'Train NEP potential',
            'config': {
                'preset': 'balanced',
                'use_ensemble': True,
                'ensemble_size': 4
            }
        }
    }
    
    # 创建执行器
    module = NEPWorkflowModule()
    executor = module.create_executor(workflow_node['data']['config'])
    
    print(f"Created executor from workflow node:")
    print(f"  Node ID: {workflow_node['id']}")
    print(f"  Node Type: {workflow_node['type']}")
    print(f"  Preset: {workflow_node['data']['config']['preset']}")
    print(f"  Ensemble: {workflow_node['data']['config']['use_ensemble']}")
    
    return workflow_node


def test_data_preparation():
    """测试数据准备集成"""
    print("\n" + "=" * 60)
    print("Test: Data Preparation")
    print("=" * 60)
    
    from nep_training import NEPDataConfig
    
    # 测试不同数据源配置
    configs = [
        {
            'name': 'XYZ files',
            'config': NEPDataConfig(
                existing_xyz='data.xyz',
                type_map=['Si', 'O']
            )
        },
        {
            'name': 'VASP OUTCARs',
            'config': NEPDataConfig(
                vasp_outcars=['OUTCAR_1', 'OUTCAR_2'],
                type_map=['Si', 'O']
            )
        },
        {
            'name': 'DeepMD format',
            'config': NEPDataConfig(
                deepmd_data='./deepmd_data',
                type_map=['Si', 'O']
            )
        }
    ]
    
    for item in configs:
        print(f"\n{item['name']}:")
        print(f"  Type map: {item['config'].type_map}")
        print(f"  Augment: {item['config'].augment_data}")
        print(f"  Train ratio: {item['config'].train_ratio}")
    
    return configs


def test_training_strategies():
    """测试训练策略"""
    print("\n" + "=" * 60)
    print("Test: Training Strategies")
    print("=" * 60)
    
    from nep_training.strategies import (
        ExponentialDecayScheduler,
        CosineAnnealingScheduler,
        EarlyStopping,
        AdaptiveStrategy
    )
    
    # 测试学习率调度器
    schedulers = [
        ('Exponential Decay', ExponentialDecayScheduler(initial_lr=0.1)),
        ('Cosine Annealing', CosineAnnealingScheduler(initial_lr=0.1, T_max=1000)),
    ]
    
    print("\nLearning Rate Schedulers:")
    for name, scheduler in schedulers:
        lrs = [scheduler.step(gen, 0.1) for gen in [0, 100, 500, 1000]]
        print(f"  {name}:")
        print(f"    LR at [0, 100, 500, 1000]: {[f'{lr:.6f}' for lr in lrs]}")
    
    # 测试早停
    early_stop = EarlyStopping(patience=5, min_delta=0.001)
    
    print("\nEarly Stopping:")
    losses = [1.0, 0.9, 0.85, 0.83, 0.82, 0.81, 0.80, 0.80, 0.80, 0.80, 0.80]
    for gen, loss in enumerate(losses):
        should_stop, improved = early_stop.step(gen, loss)
        status = "STOP" if should_stop else ("improved" if improved else "no change")
        print(f"  Gen {gen}: loss={loss:.3f} -> {status}")
        if should_stop:
            break
    
    # 测试自适应策略
    adaptive = AdaptiveStrategy()
    
    print("\nAdaptive Strategy:")
    for gen in range(0, 50, 10):
        train_loss = 1.0 / (gen + 1)
        val_loss = 1.1 / (gen + 1)
        suggestions = adaptive.update(gen, train_loss, val_loss)
        if suggestions:
            print(f"  Gen {gen}: {suggestions}")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("NEP Training Enhanced - Integration Tests")
    print("=" * 60)
    
    results = {}
    
    try:
        results['module_info'] = test_module_info()
        results['workflow_builder'] = test_workflow_builder()
        results['node_executor'] = test_node_executor()
        results['pretrained_models'] = test_pretrained_models()
        results['integration'] = test_integration_with_existing_workflow()
        results['data_preparation'] = test_data_preparation()
        results['training_strategies'] = test_training_strategies()
        
        print("\n" + "=" * 60)
        print("All Integration Tests Passed!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
