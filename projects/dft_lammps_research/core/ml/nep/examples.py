#!/usr/bin/env python3
"""
NEP Training Examples
=====================
强化版NEP训练模块使用示例
"""

import os
import sys
sys.path.insert(0, '/root/.openclaw/workspace/dft_lammps_research')

from nep_training import (
    NEPDataConfig, NEPModelConfig, NEPTrainingConfig,
    NEPTrainerV2, NEPDataset, NEPDataPreparer,
    EnsembleTrainer, EnsembleConfig,
    NEPActiveLearning, ALConfig,
    NEPModelLibrary, TransferLearning,
    TrainingMonitor, TrainingDashboard,
    train_nep, NEPWorkflowModule
)

# =============================================================================
# 示例1: 基础训练
# =============================================================================
def example_basic_training():
    """基础NEP训练示例"""
    
    print("=" * 60)
    print("Example 1: Basic NEP Training")
    print("=" * 60)
    
    # 配置数据
    data_config = NEPDataConfig(
        type_map=["Si", "O"],
        train_ratio=0.85,
        val_ratio=0.10,
        test_ratio=0.05
    )
    
    # 配置模型
    model_config = NEPModelConfig(
        type_list=["Si", "O"],
        version=4,
        cutoff_radial=6.0,
        cutoff_angular=4.0,
        n_max_radial=6,
        n_max_angular=6,
        neuron=30,
        population_size=50,
        maximum_generation=10000
    )
    
    # 配置训练
    training_config = NEPTrainingConfig(
        working_dir="./example_basic",
        use_early_stopping=True,
        use_lr_scheduler=True,
        early_stopping_patience=20
    )
    
    # 创建训练器
    trainer = NEPTrainerV2(model_config, training_config)
    
    # 准备数据 (假设有数据文件)
    # dataset = NEPDataset(xyz_file="data.xyz")
    # train_set, val_set, _ = dataset.split()
    # trainer.setup_training(train_set, val_set)
    
    # 训练
    # model_path = trainer.train()
    # print(f"Model saved to: {model_path}")
    
    print("Configuration created successfully!")
    print(f"  Working dir: {training_config.working_dir}")
    print(f"  Max generations: {model_config.maximum_generation}")


# =============================================================================
# 示例2: 使用预设配置快速训练
# =============================================================================
def example_preset_training():
    """使用预设配置快速训练"""
    
    print("\n" + "=" * 60)
    print("Example 2: Preset Training")
    print("=" * 60)
    
    # 可用的预设: fast, balanced, accurate, light, transfer
    from nep_training.core import get_preset_config
    
    for preset_name in ["fast", "balanced", "accurate"]:
        config = get_preset_config(preset_name)
        print(f"\nPreset: {preset_name}")
        print(f"  Description: {config['description']}")
        print(f"  Version: {config['version']}")
        print(f"  Neurons: {config['neuron']}")
        print(f"  Max generations: {config['maximum_generation']}")
    
    # 使用便捷函数训练 (实际需要有数据文件)
    # model_path = train_nep(
    #     input_file="data.xyz",
    #     elements=["Si", "O"],
    #     output_dir="./example_preset",
    #     preset="accurate"
    # )


# =============================================================================
# 示例3: 模型集成训练
# =============================================================================
def example_ensemble_training():
    """模型集成训练示例"""
    
    print("\n" + "=" * 60)
    print("Example 3: Ensemble Training")
    print("=" * 60)
    
    model_config = NEPModelConfig(
        type_list=["Li", "Co", "O"],
        version=4,
        neuron=50
    )
    
    training_config = NEPTrainingConfig(
        working_dir="./example_ensemble"
    )
    
    # 集成配置
    ensemble_config = EnsembleConfig(
        n_models=4,
        bootstrap=True,
        bootstrap_ratio=0.8,
        seeds=[42, 43, 44, 45]
    )
    
    # 创建集成训练器
    ensemble_trainer = EnsembleTrainer(
        model_config, training_config, ensemble_config
    )
    
    print(f"Ensemble size: {ensemble_config.n_models}")
    print(f"Bootstrap: {ensemble_config.bootstrap}")
    print(f"Seeds: {ensemble_config.seeds}")
    
    # 训练
    # dataset = NEPDataset(xyz_file="data.xyz")
    # train_set, val_set, _ = dataset.split()
    # ensemble_trainer.setup(train_set, val_set)
    # model_paths = ensemble_trainer.train()
    # print(f"Trained {len(model_paths)} models")


# =============================================================================
# 示例4: 主动学习
# =============================================================================
def example_active_learning():
    """主动学习示例"""
    
    print("\n" + "=" * 60)
    print("Example 4: Active Learning")
    print("=" * 60)
    
    # 主动学习配置
    al_config = ALConfig(
        strategy="hybrid",  # uncertainty, diversity, hybrid
        n_samples_per_iteration=50,
        max_iterations=10,
        uncertainty_threshold=0.3,
        exploration_temperatures=[300, 500, 800]
    )
    
    print(f"Strategy: {al_config.strategy}")
    print(f"Max iterations: {al_config.max_iterations}")
    print(f"Samples per iteration: {al_config.n_samples_per_iteration}")
    
    # 创建训练器和主动学习工作流
    model_config = NEPModelConfig(type_list=["Si", "O"])
    training_config = NEPTrainingConfig(working_dir="./example_al")
    trainer = NEPTrainerV2(model_config, training_config)
    
    # 定义DFT计算器 (示例)
    def dft_calculator(structures):
        results = []
        for struct in structures:
            results.append({
                'energy': 0.0,  # 实际应调用DFT
                'forces': [[0.0, 0.0, 0.0]] * len(struct)
            })
        return results
    
    al_workflow = NEPActiveLearning(
        trainer=trainer,
        dft_calculator=dft_calculator,
        config=al_config
    )
    
    print("Active learning workflow created!")
    
    # 运行主动学习 (实际需要有结构数据)
    # from ase.build import bulk
    # initial_structures = [bulk("Si", "diamond", a=5.43)]
    # base_structure = bulk("Si", "diamond", a=5.43)
    # summary = al_workflow.run(initial_structures, base_structure)
    # print(f"Total labeled: {summary['total_labeled']}")


# =============================================================================
# 示例5: 预训练模型和迁移学习
# =============================================================================
def example_transfer_learning():
    """迁移学习示例"""
    
    print("\n" + "=" * 60)
    print("Example 5: Transfer Learning")
    print("=" * 60)
    
    # 创建模型库
    library = NEPModelLibrary(library_dir="~/.nep_models")
    
    # 列出可用模型
    models = library.list_models()
    print(f"Available models in library: {len(models)}")
    
    # 搜索模型
    si_models = library.search_models("Si")
    print(f"Si-related models: {len(si_models)}")
    
    # 迁移学习
    transfer = TransferLearning(library)
    
    # 加载预训练模型 (如果存在)
    # transfer.load_pretrained("generic_solid")
    
    # 准备迁移学习配置
    # new_config = transfer.prepare_for_transfer(
    #     new_type_map=["Si", "O", "Li"],
    #     freeze_descriptor=False
    # )
    
    # 微调模型
    # model_path = transfer.fine_tune(
    #     train_dataset=dataset,
    #     output_dir="./example_transfer",
    #     epochs=5000
    # )
    
    print("Transfer learning setup complete!")


# =============================================================================
# 示例6: 实时监控
# =============================================================================
def example_monitoring():
    """实时监控示例"""
    
    print("\n" + "=" * 60)
    print("Example 6: Real-time Monitoring")
    print("=" * 60)
    
    # 创建监控器
    monitor = TrainingMonitor(
        log_dir="./example_monitoring/logs",
        enable_tensorboard=True,
        enable_wandb=False,  # 设置为True需要配置wandb
        enable_websocket=True
    )
    
    # 开始训练运行
    monitor.start_run({
        'model_preset': 'balanced',
        'batch_size': 1000,
        'learning_rate': 0.001
    })
    
    # 模拟训练过程
    for step in range(100):
        # 模拟损失
        train_loss = 1.0 / (step + 1)
        val_loss = 1.2 / (step + 1)
        
        # 记录指标
        monitor.log_metrics(step, {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': 0.001 * (0.95 ** (step / 1000))
        })
    
    # 结束运行
    monitor.finish_run()
    
    print("Monitoring setup complete!")
    print(f"  Backends: {monitor.backends}")
    print(f"  Log directory: {monitor.log_dir}")


# =============================================================================
# 示例7: 平台集成
# =============================================================================
def example_platform_integration():
    """平台集成示例"""
    
    print("\n" + "=" * 60)
    print("Example 7: Platform Integration")
    print("=" * 60)
    
    # 创建工作流模块
    module = NEPWorkflowModule()
    
    # 获取模块信息
    info = module.get_info()
    print(f"Module: {info['name']} v{info['version']}")
    print(f"Features: {', '.join(info['features'])}")
    print(f"Presets: {', '.join(info['presets'])}")
    
    # 构建工作流
    builder = module.workflow_builder
    
    workflow = builder.build_simple_workflow(
        name="my_nep_training",
        input_data={
            'xyz_files': ['data.xyz'],
            'vasp_outcars': ['OUTCAR']
        },
        output_dir="./example_workflow"
    )
    
    print(f"\nWorkflow '{workflow['name']}' created with {len(workflow['nodes'])} nodes")
    for node in workflow['nodes']:
        print(f"  - {node['id']}: {node['type']}")
    
    # 执行工作流
    # executor = module.create_executor({
    #     'model_preset': 'accurate',
    #     'use_ensemble': True
    # })
    # results = await executor.execute(context, progress_callback)


# =============================================================================
# 示例8: 高级配置
# =============================================================================
def example_advanced_config():
    """高级配置示例"""
    
    print("\n" + "=" * 60)
    print("Example 8: Advanced Configuration")
    print("=" * 60)
    
    # 完整的高级配置
    from nep_training import PrecisionMode, DistributedConfig
    
    data_config = NEPDataConfig(
        type_map=["Si", "O", "H"],
        # 数据增强
        augment_data=True,
        rotation_augment=True,
        noise_augment=True,
        noise_std=0.01,
        # 过滤
        energy_threshold=50.0,
        force_threshold=50.0,
        # 分割
        stratified_split=True
    )
    
    model_config = NEPModelConfig(
        type_list=["Si", "O", "H"],
        version=4,
        # 描述符
        cutoff_radial=6.0,
        cutoff_angular=4.0,
        n_max_radial=6,
        n_max_angular=6,
        basis_size_radial=12,
        basis_size_angular=12,
        l_max_3body=6,
        # 网络
        neuron=50,
        # 学习率
        initial_lr=0.1,
        min_lr=1e-6,
        lr_decay_steps=5000,
        lr_decay_rate=0.95
    )
    
    training_config = NEPTrainingConfig(
        working_dir="./example_advanced",
        # 精度
        precision=PrecisionMode.FP16,
        # 分布式
        distributed=DistributedConfig(
            enabled=False,  # 设为True启用多GPU
            world_size=2,
            backend="nccl"
        ),
        # 高级功能
        use_lr_scheduler=True,
        use_early_stopping=True,
        early_stopping_patience=20,
        save_checkpoints=True,
        checkpoint_frequency=1000,
        # 性能优化
        pin_memory=True,
        num_workers=4
    )
    
    print("Advanced configuration created:")
    print(f"  Precision: {training_config.precision.value}")
    print(f"  Distributed: {training_config.distributed.enabled}")
    print(f"  Data augmentation: {data_config.augment_data}")
    print(f"  Early stopping: {training_config.use_early_stopping}")


# =============================================================================
# 主函数
# =============================================================================
def main():
    """运行所有示例"""
    
    print("\n" + "=" * 60)
    print("NEP Training Enhanced - Usage Examples")
    print("=" * 60)
    
    example_basic_training()
    example_preset_training()
    example_ensemble_training()
    example_active_learning()
    example_transfer_learning()
    example_monitoring()
    example_platform_integration()
    example_advanced_config()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
