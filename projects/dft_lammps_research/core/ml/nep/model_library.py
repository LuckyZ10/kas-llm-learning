"""
nep_training/model_library.py
=============================
NEP模型库模块

包含:
- 预训练模型管理
- 模型版本控制
- 迁移学习支持
- 模型评估基准
"""

import os
import json
import shutil
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import hashlib
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from ase import Atoms
from ase.io import read

from .core import NEPModelConfig
from .data import NEPDataset

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """模型版本信息"""
    version: str
    description: str
    created_at: str
    author: str
    parent_version: Optional[str] = None
    commit_hash: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ModelVersion":
        return cls(**data)


@dataclass
class PretrainedModel:
    """预训练模型信息"""
    name: str
    version: ModelVersion
    model_path: str
    config_path: str
    type_map: List[str]
    description: str = ""
    
    # 元数据
    n_train_structures: int = 0
    train_energy_range: Tuple[float, float] = (0.0, 0.0)
    elements: List[str] = field(default_factory=list)
    
    # 性能指标
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # 文件哈希 (用于完整性验证)
    model_hash: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'version': self.version.to_dict(),
            'model_path': self.model_path,
            'config_path': self.config_path,
            'type_map': self.type_map,
            'description': self.description,
            'n_train_structures': self.n_train_structures,
            'train_energy_range': self.train_energy_range,
            'elements': self.elements,
            'metrics': self.metrics,
            'model_hash': self.model_hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PretrainedModel":
        data = data.copy()
        data['version'] = ModelVersion.from_dict(data['version'])
        data['train_energy_range'] = tuple(data.get('train_energy_range', (0.0, 0.0)))
        return cls(**data)
    
    def compute_hash(self) -> str:
        """计算模型文件哈希"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        return ""


class NEPModelLibrary:
    """
    NEP预训练模型库
    
    管理预训练模型的存储、检索和版本控制
    """
    
    def __init__(self, library_dir: str = "~/.nep_models"):
        self.library_dir = Path(library_dir).expanduser()
        self.library_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.library_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.library_dir / "library_metadata.json"
        self.models: Dict[str, PretrainedModel] = {}
        
        self._load_metadata()
    
    def _load_metadata(self):
        """加载模型库元数据"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            for name, model_data in data.get('models', {}).items():
                try:
                    self.models[name] = PretrainedModel.from_dict(model_data)
                except Exception as e:
                    logger.warning(f"Failed to load model metadata for {name}: {e}")
        
        logger.info(f"Loaded {len(self.models)} models from library")
    
    def _save_metadata(self):
        """保存模型库元数据"""
        data = {
            'models': {name: model.to_dict() for name, model in self.models.items()},
            'updated_at': datetime.now().isoformat()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_model(self, 
                  name: str,
                  model_path: str,
                  config: NEPModelConfig,
                  version: Optional[ModelVersion] = None,
                  description: str = "",
                  metrics: Optional[Dict[str, float]] = None) -> PretrainedModel:
        """
        添加模型到库
        
        Args:
            name: 模型名称
            model_path: 模型文件路径
            config: 模型配置
            version: 版本信息
            description: 模型描述
            metrics: 性能指标
            
        Returns:
            PretrainedModel对象
        """
        # 创建版本
        if version is None:
            version = ModelVersion(
                version="1.0.0",
                description="Initial version",
                created_at=datetime.now().isoformat(),
                author=os.environ.get('USER', 'unknown')
            )
        
        # 复制模型到库目录
        model_dir = self.models_dir / name / version.version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        dst_model_path = model_dir / "nep.txt"
        shutil.copy(model_path, dst_model_path)
        
        # 保存配置
        config_path = model_dir / "config.json"
        config.save(str(config_path))
        
        # 创建模型信息
        pretrained = PretrainedModel(
            name=name,
            version=version,
            model_path=str(dst_model_path),
            config_path=str(config_path),
            type_map=config.type_list,
            description=description,
            metrics=metrics or {}
        )
        
        pretrained.model_hash = pretrained.compute_hash()
        
        # 添加到库
        model_key = f"{name}@{version.version}"
        self.models[model_key] = pretrained
        self._save_metadata()
        
        logger.info(f"Added model {model_key} to library")
        
        return pretrained
    
    def get_model(self, name: str, version: Optional[str] = None) -> Optional[PretrainedModel]:
        """
        获取模型
        
        Args:
            name: 模型名称 (可以包含版本，如 "model@1.0.0")
            version: 版本号 (如果name中未指定)
            
        Returns:
            PretrainedModel或None
        """
        # 解析名称和版本
        if '@' in name:
            name, version = name.split('@')
        
        if version:
            model_key = f"{name}@{version}"
            return self.models.get(model_key)
        else:
            # 返回最新版本
            matching = [k for k in self.models.keys() if k.startswith(f"{name}@")]
            if matching:
                # 按版本号排序，返回最新的
                return self.models[sorted(matching)[-1]]
        
        return None
    
    def list_models(self, 
                   element_filter: Optional[List[str]] = None,
                   tag_filter: Optional[List[str]] = None) -> List[PretrainedModel]:
        """
        列出模型库中的模型
        
        Args:
            element_filter: 元素过滤
            tag_filter: 标签过滤
            
        Returns:
            符合条件的模型列表
        """
        results = []
        
        for model in self.models.values():
            # 元素过滤
            if element_filter:
                if not all(elem in model.elements for elem in element_filter):
                    continue
            
            # 标签过滤
            if tag_filter:
                if not any(tag in model.version.tags for tag in tag_filter):
                    continue
            
            results.append(model)
        
        return results
    
    def search_models(self, query: str) -> List[PretrainedModel]:
        """
        搜索模型
        
        Args:
            query: 搜索关键词
            
        Returns:
            匹配的模型列表
        """
        results = []
        query = query.lower()
        
        for model in self.models.values():
            if (query in model.name.lower() or
                query in model.description.lower() or
                any(query in elem.lower() for elem in model.elements) or
                any(query in tag.lower() for tag in model.version.tags)):
                results.append(model)
        
        return results
    
    def delete_model(self, name: str, version: Optional[str] = None) -> bool:
        """删除模型"""
        model = self.get_model(name, version)
        
        if model:
            model_key = f"{model.name}@{model.version.version}"
            del self.models[model_key]
            
            # 删除文件
            model_dir = Path(model.model_path).parent
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            self._save_metadata()
            logger.info(f"Deleted model {model_key}")
            return True
        
        return False
    
    def get_model_path(self, name: str, version: Optional[str] = None) -> Optional[str]:
        """获取模型文件路径"""
        model = self.get_model(name, version)
        return model.model_path if model else None


class TransferLearning:
    """
    迁移学习支持
    
    基于预训练模型进行微调
    """
    
    def __init__(self, 
                 model_library: NEPModelLibrary,
                 pretrained_model: Optional[PretrainedModel] = None):
        self.library = model_library
        self.pretrained_model = pretrained_model
    
    def load_pretrained(self, name: str, version: Optional[str] = None):
        """加载预训练模型"""
        self.pretrained_model = self.library.get_model(name, version)
        
        if self.pretrained_model is None:
            raise ValueError(f"Model {name}@{version or 'latest'} not found in library")
        
        logger.info(f"Loaded pretrained model: {self.pretrained_model.name}")
        logger.info(f"  Version: {self.pretrained_model.version.version}")
        logger.info(f"  Elements: {self.pretrained_model.elements}")
        
        return self.pretrained_model
    
    def prepare_for_transfer(self,
                            new_type_map: List[str],
                            freeze_descriptor: bool = False,
                            freeze_layers: Optional[List[int]] = None) -> NEPModelConfig:
        """
        准备迁移学习配置
        
        Args:
            new_type_map: 新系统的元素类型
            freeze_descriptor: 是否冻结描述符参数
            freeze_layers: 冻结的神经网络层
            
        Returns:
            迁移学习配置
        """
        if self.pretrained_model is None:
            raise ValueError("No pretrained model loaded")
        
        # 加载原始配置
        config = NEPModelConfig.load(self.pretrained_model.config_path)
        
        # 更新元素类型
        config.type_list = new_type_map
        
        # 调整训练参数以适应迁移学习
        config.population_size = max(20, config.population_size // 2)
        config.maximum_generation = max(10000, config.maximum_generation // 10)
        
        logger.info(f"Prepared transfer learning config:")
        logger.info(f"  Original elements: {self.pretrained_model.type_map}")
        logger.info(f"  New elements: {new_type_map}")
        logger.info(f"  Reduced training steps for fine-tuning")
        
        return config
    
    def fine_tune(self,
                 train_dataset: NEPDataset,
                 output_dir: str,
                 epochs: int = 10000) -> str:
        """
        微调预训练模型
        
        Args:
            train_dataset: 新数据集
            output_dir: 输出目录
            epochs: 训练代数
            
        Returns:
            微调后的模型路径
        """
        from .trainer import NEPTrainerV2
        from .core import NEPTrainingConfig
        
        # 准备配置
        config = self.prepare_for_transfer(train_dataset.get_type_map())
        config.maximum_generation = epochs
        
        # 创建训练器
        train_config = NEPTrainingConfig(
            working_dir=output_dir,
            restart=True,  # 从预训练模型重启
            checkpoint_path=self.pretrained_model.model_path
        )
        
        trainer = NEPTrainerV2(config, train_config)
        trainer.setup_training(train_dataset)
        
        # 复制预训练模型作为初始模型
        shutil.copy(self.pretrained_model.model_path, 
                   Path(output_dir) / "nep_restart.txt")
        
        # 训练
        model_path = trainer.train()
        
        logger.info(f"Fine-tuning completed: {model_path}")
        
        return model_path


@dataclass
class BenchmarkTask:
    """基准测试任务"""
    name: str
    description: str
    test_structures: List[Atoms]
    reference_energies: np.ndarray
    reference_forces: Optional[List[np.ndarray]] = None
    metrics: List[str] = field(default_factory=lambda: ['rmse_energy', 'rmse_force', 'mae_energy'])


class BenchmarkSuite:
    """
    模型评估基准套件
    
    提供标准化的模型性能评估
    """
    
    def __init__(self, benchmark_dir: str = "./benchmarks"):
        self.benchmark_dir = Path(benchmark_dir)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        self.tasks: Dict[str, BenchmarkTask] = {}
    
    def add_task(self, task: BenchmarkTask):
        """添加基准测试任务"""
        self.tasks[task.name] = task
        logger.info(f"Added benchmark task: {task.name}")
    
    def load_standard_benchmarks(self, elements: List[str]):
        """加载标准基准测试"""
        # 这里可以加载预定义的基准测试
        # 例如: QM9、MD17、Materials Project等
        
        # 简化版本：创建一些基础测试
        for elem in elements:
            task = self._create_element_benchmark(elem)
            self.add_task(task)
    
    def _create_element_benchmark(self, element: str) -> BenchmarkTask:
        """创建元素基准测试"""
        # 创建简单的测试结构
        from ase.build import bulk
        
        structures = []
        try:
            # 晶体结构
            bcc = bulk(element, 'bcc', a=3.0)
            fcc = bulk(element, 'fcc', a=3.5)
            structures.extend([bcc, fcc])
        except:
            pass
        
        # 参考值 (占位符)
        ref_energies = np.zeros(len(structures))
        
        return BenchmarkTask(
            name=f"{element}_bulk",
            description=f"Bulk properties of {element}",
            test_structures=structures,
            reference_energies=ref_energies
        )
    
    def run_benchmark(self, 
                     model_path: str,
                     task_name: Optional[str] = None) -> Dict[str, Any]:
        """
        运行基准测试
        
        Args:
            model_path: 模型文件路径
            task_name: 任务名称 (None表示运行所有任务)
            
        Returns:
            测试结果
        """
        results = {}
        
        tasks_to_run = ([self.tasks[task_name]] if task_name 
                       else list(self.tasks.values()))
        
        for task in tasks_to_run:
            logger.info(f"Running benchmark: {task.name}")
            
            # 加载模型并进行预测
            predictions = self._predict(model_path, task.test_structures)
            
            # 计算指标
            task_results = self._compute_metrics(
                predictions, 
                task.reference_energies,
                task.reference_forces,
                task.metrics
            )
            
            results[task.name] = task_results
        
        # 汇总
        summary = {
            'model_path': model_path,
            'timestamp': datetime.now().isoformat(),
            'tasks': results,
            'overall_score': np.mean([
                r.get('rmse_energy', 0) for r in results.values()
            ])
        }
        
        return summary
    
    def _predict(self, model_path: str, structures: List[Atoms]) -> Dict[str, np.ndarray]:
        """使用模型进行预测"""
        # 简化版本: 返回随机预测
        # 实际应调用GPUMD或NEP计算器
        
        n = len(structures)
        return {
            'energies': np.random.randn(n),
            'forces': [np.random.randn(len(s), 3) for s in structures]
        }
    
    def _compute_metrics(self,
                        predictions: Dict,
                        ref_energies: np.ndarray,
                        ref_forces: Optional[List[np.ndarray]],
                        metrics: List[str]) -> Dict[str, float]:
        """计算评估指标"""
        results = {}
        
        pred_e = predictions['energies']
        
        if 'rmse_energy' in metrics:
            results['rmse_energy'] = float(np.sqrt(np.mean((pred_e - ref_energies)**2)))
        
        if 'mae_energy' in metrics:
            results['mae_energy'] = float(np.mean(np.abs(pred_e - ref_energies)))
        
        if ref_forces and 'rmse_force' in metrics:
            pred_f = predictions['forces']
            force_errors = [
                np.sqrt(np.mean((pf - rf)**2))
                for pf, rf in zip(pred_f, ref_forces)
            ]
            results['rmse_force'] = float(np.mean(force_errors))
        
        return results
    
    def compare_models(self, 
                      model_paths: List[str],
                      output_file: Optional[str] = None) -> pd.DataFrame:
        """
        比较多个模型
        
        Args:
            model_paths: 模型路径列表
            output_file: 输出文件路径
            
        Returns:
            比较结果DataFrame
        """
        import pandas as pd
        
        all_results = []
        
        for model_path in model_paths:
            results = self.run_benchmark(model_path)
            results['model'] = Path(model_path).name
            all_results.append(results)
        
        # 创建对比表
        df = pd.DataFrame([
            {
                'model': r['model'],
                'overall_score': r['overall_score'],
                **{f"{task}_{metric}": value 
                   for task, metrics in r['tasks'].items()
                   for metric, value in metrics.items()}
            }
            for r in all_results
        ])
        
        if output_file:
            df.to_csv(output_file, index=False)
        
        return df


# 预定义的标准模型
STANDARD_MODELS = {
    "generic_solid": {
        "description": "通用固体势，适用于多种晶体结构",
        "elements": ["H", "Li", "C", "N", "O", "Na", "Si", "P", "S"],
        "recommended_for": ["bulk", "surface", "defect"]
    },
    "battery_cathode": {
        "description": "锂电池正极材料势",
        "elements": ["Li", "Co", "Ni", "Mn", "O"],
        "recommended_for": ["LiCoO2", "NMC", "Li-ion battery"]
    },
    "semiconductor": {
        "description": "半导体材料势",
        "elements": ["Si", "Ge", "C", "N", "Ga", "As"],
        "recommended_for": ["Si", "GaAs", "diamond"]
    },
    "catalyst": {
        "description": "催化材料势",
        "elements": ["Pt", "Pd", "Ru", "Rh", "Au", "Cu"],
        "recommended_for": ["surface catalysis", "nanoparticles"]
    }
}
