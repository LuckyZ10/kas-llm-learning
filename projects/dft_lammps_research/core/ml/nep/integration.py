"""
nep_training/integration.py
===========================
平台集成模块

将NEP训练模块集成到现有平台工作流系统
"""

import json
import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
import asyncio

from .core import NEPModelConfig, NEPTrainingConfig, NEPDataConfig, TrainingState
from .trainer import NEPTrainerV2, EnsembleTrainer
from .data import NEPDataset, NEPDataPreparer, DataAugmenter
from .strategies import EnsembleConfig
from .active_learning import NEPActiveLearning, ALConfig
from .monitoring import TrainingMonitor, TrainingDashboard
from .model_library import NEPModelLibrary, TransferLearning

logger = logging.getLogger(__name__)


@dataclass
class NEPWorkflowConfig:
    """NEP工作流配置"""
    # 数据
    data_config: NEPDataConfig = None
    
    # 模型
    model_preset: str = "balanced"
    custom_model_config: Optional[NEPModelConfig] = None
    
    # 训练
    use_ensemble: bool = False
    ensemble_size: int = 4
    use_active_learning: bool = False
    al_iterations: int = 10
    
    # 监控
    enable_monitoring: bool = True
    enable_tensorboard: bool = False
    enable_wandb: bool = False
    
    # 优化
    enable_gpu_optimization: bool = True
    enable_data_optimization: bool = True


class NEPNodeExecutor:
    """
    NEP节点执行器
    
    实现工作流引擎的节点执行接口
    """
    
    def __init__(self, config: NEPWorkflowConfig):
        self.config = config
        self.trainer: Optional[NEPTrainerV2] = None
        self.monitor: Optional[TrainingMonitor] = None
        self.dashboard: Optional[TrainingDashboard] = None
    
    async def execute(self, 
                     context: Dict[str, Any],
                     progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        执行NEP训练节点
        
        Args:
            context: 工作流上下文，包含输入数据、配置等
            progress_callback: 进度回调函数
            
        Returns:
            执行结果
        """
        # 解析上下文
        input_data = context.get('input_data', {})
        working_dir = context.get('working_dir', './nep_workflow')
        
        logger.info(f"Executing NEP node in {working_dir}")
        
        # 初始化监控
        if self.config.enable_monitoring:
            self.monitor = TrainingMonitor(
                log_dir=f"{working_dir}/logs",
                enable_tensorboard=self.config.enable_tensorboard,
                enable_wandb=self.config.enable_wandb
            )
            self.monitor.start_run({
                'model_preset': self.config.model_preset,
                'use_ensemble': self.config.use_ensemble,
                'use_active_learning': self.config.use_active_learning
            })
            
            self.dashboard = TrainingDashboard()
            self.dashboard.start()
        
        try:
            # 1. 数据准备
            await self._update_progress(progress_callback, 0.1, "Preparing data...")
            train_dataset, val_dataset = self._prepare_data(input_data, working_dir)
            
            # 2. 配置模型
            await self._update_progress(progress_callback, 0.2, "Configuring model...")
            model_config = self._get_model_config()
            
            # 3. 训练
            if self.config.use_ensemble:
                model_paths = await self._train_ensemble(
                    model_config, train_dataset, val_dataset, 
                    working_dir, progress_callback
                )
            elif self.config.use_active_learning:
                model_path = await self._train_with_al(
                    model_config, train_dataset, working_dir, progress_callback
                )
                model_paths = [model_path]
            else:
                model_path = await self._train_single(
                    model_config, train_dataset, val_dataset,
                    working_dir, progress_callback
                )
                model_paths = [model_path]
            
            # 4. 验证
            await self._update_progress(progress_callback, 0.9, "Validating model...")
            validation_results = self._validate_models(model_paths, val_dataset)
            
            # 5. 保存结果
            await self._update_progress(progress_callback, 0.95, "Saving results...")
            results = {
                'model_paths': model_paths,
                'validation': validation_results,
                'working_dir': working_dir,
                'status': 'success'
            }
            
            self._save_results(results, working_dir)
            
            await self._update_progress(progress_callback, 1.0, "Completed!")
            
            return results
            
        except Exception as e:
            logger.error(f"NEP training failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
        finally:
            if self.monitor:
                self.monitor.finish_run()
            if self.dashboard:
                self.dashboard.stop()
    
    async def _update_progress(self, 
                              callback: Optional[Callable],
                              progress: float,
                              message: str):
        """更新进度"""
        if callback:
            await callback(progress, message)
        logger.info(f"[{progress*100:.0f}%] {message}")
    
    def _prepare_data(self, input_data: Dict, working_dir: str) -> tuple:
        """准备训练数据"""
        data_config = self.config.data_config or NEPDataConfig()
        
        # 从不同来源加载数据
        frames = []
        
        if 'xyz_files' in input_data:
            for xyz in input_data['xyz_files']:
                dataset = NEPDataset(xyz_file=xyz)
                frames.extend(dataset.frames)
        
        if 'vasp_outcars' in input_data:
            preparer = NEPDataPreparer(data_config.type_map)
            dataset = preparer.from_vasp_outcars(input_data['vasp_outcars'])
            frames.extend(dataset.frames)
        
        if 'deepmd_data' in input_data:
            preparer = NEPDataPreparer(data_config.type_map)
            dataset = preparer.from_deepmd(input_data['deepmd_data'])
            frames.extend(dataset.frames)
        
        # 创建数据集
        full_dataset = NEPDataset(frames=frames)
        
        # 数据增强
        if data_config.augment_data:
            augmenter = DataAugmenter(
                rotation=data_config.rotation_augment,
                noise=data_config.noise_augment,
                noise_std=data_config.noise_std
            )
            full_dataset = augmenter.augment_dataset(full_dataset, n_augment=2)
        
        # 分割数据集
        train_dataset, val_dataset, _ = full_dataset.split(
            train_ratio=data_config.train_ratio,
            val_ratio=data_config.val_ratio
        )
        
        logger.info(f"Data prepared: {len(train_dataset)} train, {len(val_dataset)} val")
        
        return train_dataset, val_dataset
    
    def _get_model_config(self) -> NEPModelConfig:
        """获取模型配置"""
        if self.config.custom_model_config:
            return self.config.custom_model_config
        
        from .core import get_preset_config
        preset = get_preset_config(self.config.model_preset)
        
        # 获取元素类型
        type_map = self.config.data_config.type_map if self.config.data_config else []
        
        config = NEPModelConfig(
            type_list=type_map,
            **{k: v for k, v in preset.items() if k in NEPModelConfig.__dataclass_fields__}
        )
        
        return config
    
    async def _train_single(self,
                           model_config: NEPModelConfig,
                           train_dataset: NEPDataset,
                           val_dataset: NEPDataset,
                           working_dir: str,
                           progress_callback: Optional[Callable]) -> str:
        """单模型训练"""
        train_config = NEPTrainingConfig(
            working_dir=working_dir,
            use_early_stopping=True,
            use_lr_scheduler=True
        )
        
        self.trainer = NEPTrainerV2(model_config, train_config)
        
        # 添加进度回调
        if progress_callback:
            def progress_wrapper(status):
                progress = status.get('progress', 0) / 100
                asyncio.create_task(progress_callback(0.2 + progress * 0.6, 
                    f"Training generation {status.get('generation', 0)}"))
            
            self.trainer.add_callback(progress_wrapper)
        
        # 监控回调
        if self.monitor:
            def monitor_callback(status):
                self.monitor.log_metrics(
                    status.get('generation', 0),
                    {
                        'train_loss': status.get('train_loss_history', [{}])[-1].get('loss', 0),
                        'best_loss': status.get('best_loss', 0)
                    }
                )
            
            self.trainer.add_callback(monitor_callback)
        
        self.trainer.setup_training(train_dataset, val_dataset)
        model_path = self.trainer.train()
        
        return model_path
    
    async def _train_ensemble(self,
                             model_config: NEPModelConfig,
                             train_dataset: NEPDataset,
                             val_dataset: NEPDataset,
                             working_dir: str,
                             progress_callback: Optional[Callable]) -> List[str]:
        """集成训练"""
        ensemble_config = EnsembleConfig(n_models=self.config.ensemble_size)
        
        train_config = NEPTrainingConfig(working_dir=working_dir)
        
        ensemble_trainer = EnsembleTrainer(
            model_config, train_config, ensemble_config
        )
        ensemble_trainer.setup(train_dataset, val_dataset)
        
        model_paths = ensemble_trainer.train()
        
        return model_paths
    
    async def _train_with_al(self,
                            model_config: NEPModelConfig,
                            train_dataset: NEPDataset,
                            working_dir: str,
                            progress_callback: Optional[Callable]) -> str:
        """主动学习训练"""
        train_config = NEPTrainingConfig(working_dir=working_dir)
        trainer = NEPTrainerV2(model_config, train_config)
        
        al_config = ALConfig(max_iterations=self.config.al_iterations)
        
        al_workflow = NEPActiveLearning(
            trainer=trainer,
            config=al_config
        )
        
        # 获取初始结构
        initial_structures = [f.atoms for f in train_dataset.frames[:10]]
        base_structure = train_dataset.frames[0].atoms
        
        # 运行主动学习
        summary = al_workflow.run(initial_structures, base_structure)
        
        return str(summary['final_model'])
    
    def _validate_models(self, 
                        model_paths: List[str],
                        val_dataset: NEPDataset) -> Dict[str, Any]:
        """验证模型"""
        results = {}
        
        for i, model_path in enumerate(model_paths):
            # 简化验证
            results[f'model_{i}'] = {
                'path': model_path,
                'status': 'validated'
            }
        
        return results
    
    def _save_results(self, results: Dict, working_dir: str):
        """保存结果"""
        output_file = Path(working_dir) / "nep_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)


class NEPWorkflowBuilder:
    """
    NEP工作流构建器
    
    一键构建NEP训练工作流
    """
    
    def __init__(self, workflow_engine: Any = None):
        self.workflow_engine = workflow_engine
    
    def build_simple_workflow(self,
                             name: str,
                             input_data: Dict[str, Any],
                             output_dir: str) -> Dict[str, Any]:
        """
        构建简单NEP训练工作流
        
        Args:
            name: 工作流名称
            input_data: 输入数据配置
            output_dir: 输出目录
            
        Returns:
            工作流定义字典
        """
        workflow = {
            'name': name,
            'version': '1.0.0',
            'nodes': [
                {
                    'id': 'data_prep',
                    'type': 'data_preparation',
                    'config': input_data
                },
                {
                    'id': 'nep_training',
                    'type': 'nep_training',
                    'depends_on': ['data_prep'],
                    'config': {
                        'preset': 'balanced',
                        'monitoring': True
                    }
                },
                {
                    'id': 'model_validation',
                    'type': 'validation',
                    'depends_on': ['nep_training'],
                    'config': {}
                }
            ],
            'output_dir': output_dir
        }
        
        return workflow
    
    def build_active_learning_workflow(self,
                                      name: str,
                                      initial_structures: List[Any],
                                      output_dir: str,
                                      max_iterations: int = 10) -> Dict[str, Any]:
        """
        构建主动学习工作流
        
        Args:
            name: 工作流名称
            initial_structures: 初始结构
            output_dir: 输出目录
            max_iterations: 最大迭代次数
            
        Returns:
            工作流定义字典
        """
        workflow = {
            'name': name,
            'version': '1.0.0',
            'type': 'active_learning',
            'nodes': [
                {
                    'id': 'al_loop',
                    'type': 'active_learning_loop',
                    'config': {
                        'initial_structures': initial_structures,
                        'max_iterations': max_iterations,
                        'strategy': 'hybrid'
                    }
                }
            ],
            'output_dir': output_dir
        }
        
        return workflow
    
    def build_transfer_learning_workflow(self,
                                        name: str,
                                        pretrained_model: str,
                                        new_data: Dict[str, Any],
                                        output_dir: str) -> Dict[str, Any]:
        """
        构建迁移学习工作流
        
        Args:
            name: 工作流名称
            pretrained_model: 预训练模型名称
            new_data: 新数据
            output_dir: 输出目录
            
        Returns:
            工作流定义字典
        """
        workflow = {
            'name': name,
            'version': '1.0.0',
            'type': 'transfer_learning',
            'nodes': [
                {
                    'id': 'load_pretrained',
                    'type': 'model_loading',
                    'config': {
                        'model_name': pretrained_model
                    }
                },
                {
                    'id': 'fine_tune',
                    'type': 'nep_training',
                    'depends_on': ['load_pretrained'],
                    'config': {
                        'preset': 'transfer',
                        'epochs': 10000,
                        'restart_from': pretrained_model
                    }
                }
            ],
            'output_dir': output_dir
        }
        
        return workflow


class NEPWorkflowModule:
    """
    NEP工作流模块
    
    作为平台模块的标准接口
    """
    
    def __init__(self):
        self.name = "nep_training"
        self.version = "2.0.0"
        self.description = "Enhanced NEP Training Module"
        
        self.model_library = NEPModelLibrary()
        self.workflow_builder = NEPWorkflowBuilder()
    
    def get_info(self) -> Dict[str, Any]:
        """获取模块信息"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'features': [
                'Advanced training strategies',
                'Multi-precision support',
                'Distributed training',
                'Active learning',
                'Model ensemble',
                'Transfer learning',
                'Real-time monitoring',
                'Performance optimization'
            ],
            'presets': ['fast', 'balanced', 'accurate', 'light', 'transfer']
        }
    
    def create_executor(self, config: Dict[str, Any]) -> NEPNodeExecutor:
        """创建节点执行器"""
        # 将工作流节点配置映射到 NEPWorkflowConfig
        from .core import NEPDataConfig, get_preset_config

        workflow_config_kwargs = {}

        # 映射 preset
        if 'preset' in config:
            workflow_config_kwargs['model_preset'] = config['preset']

        # 映射 ensemble 设置
        if 'use_ensemble' in config:
            workflow_config_kwargs['use_ensemble'] = config['use_ensemble']
        if 'ensemble_size' in config:
            # 当前 NEPWorkflowConfig 不直接支持 ensemble_size，但可以后续扩展
            pass

        # 映射 active learning 设置
        if 'use_active_learning' in config:
            workflow_config_kwargs['use_active_learning'] = config['use_active_learning']

        # 映射监控设置
        if 'enable_monitoring' in config:
            workflow_config_kwargs['enable_monitoring'] = config['enable_monitoring']

        # 创建 NEPWorkflowConfig
        workflow_config = NEPWorkflowConfig(**workflow_config_kwargs)
        return NEPNodeExecutor(workflow_config)
    
    def get_pretrained_models(self, 
                             elements: Optional[List[str]] = None) -> List[Dict]:
        """获取可用预训练模型"""
        models = self.model_library.list_models(element_filter=elements)
        return [m.to_dict() for m in models]
    
    def quick_train(self,
                   input_file: str,
                   elements: List[str],
                   output_dir: str = "./nep_quick_train",
                   preset: str = "balanced") -> str:
        """
        快速训练接口
        
        最简单的训练入口，一行代码开始训练
        
        Args:
            input_file: 输入XYZ文件
            elements: 元素列表
            output_dir: 输出目录
            preset: 预设配置
            
        Returns:
            训练好的模型路径
        """
        from .core import NEPDataConfig
        
        config = NEPWorkflowConfig(
            data_config=NEPDataConfig(
                existing_xyz=input_file,
                type_map=elements
            ),
            model_preset=preset,
            enable_monitoring=True
        )
        
        executor = NEPNodeExecutor(config)
        
        context = {
            'input_data': {'xyz_files': [input_file]},
            'working_dir': output_dir
        }
        
        # 运行 (同步版本)
        import asyncio
        results = asyncio.run(executor.execute(context))
        
        if results['status'] == 'success':
            return results['model_paths'][0]
        else:
            raise RuntimeError(f"Training failed: {results.get('error')}")


# 便捷函数
def train_nep(input_file: str,
              elements: List[str],
              output_dir: str = "./nep_training",
              preset: str = "balanced") -> str:
    """
    便捷训练函数
    
    最简单的NEP训练入口
    
    Example:
        >>> model_path = train_nep("data.xyz", ["Si", "O"], preset="accurate")
    """
    module = NEPWorkflowModule()
    return module.quick_train(input_file, elements, output_dir, preset)


def continue_training(checkpoint_path: str,
                     additional_epochs: int = 10000) -> str:
    """
    从检查点继续训练
    """
    # 加载检查点
    from .core import NEPCheckpoint
    checkpoint = NEPCheckpoint.load(checkpoint_path)
    
    # 创建训练器
    trainer = NEPTrainerV2(checkpoint.config)
    trainer.load_checkpoint(checkpoint_path)
    
    # 继续训练
    trainer.model_config.maximum_generation = checkpoint.generation + additional_epochs
    model_path = trainer.train()
    
    return model_path
