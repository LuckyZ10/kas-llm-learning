"""
Batch Validator
===============
批量验证工具

用于批量处理多个验证任务
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from .validation_workflow import ValidationWorkflow, ValidationConfig, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class BatchValidationConfig:
    """批量验证配置"""
    # 数据设置
    exp_data_dir: str = ""
    sim_data_dir: str = ""
    data_type: str = "auto"
    
    # 文件匹配
    exp_file_pattern: str = "*"
    sim_file_pattern: str = "*"
    
    # 输出设置
    output_dir: str = "./batch_validation_results"
    individual_reports: bool = True
    summary_report: bool = True
    
    # 并行设置
    n_workers: int = 1
    
    # 验证配置
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)


@dataclass
class BatchValidationResult:
    """批量验证结果"""
    results: Dict[str, ValidationResult] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    failed_files: List[str] = field(default_factory=list)
    
    def add_result(self, name: str, result: ValidationResult):
        """添加单个结果"""
        self.results[name] = result
    
    def calculate_summary(self) -> Dict[str, Any]:
        """计算汇总统计"""
        if not self.results:
            return {}
        
        # 统计成功率
        success_count = sum(1 for r in self.results.values() if r.success == True)
        partial_count = sum(1 for r in self.results.values() if r.success == 'partial')
        failure_count = sum(1 for r in self.results.values() if r.success == False)
        
        summary = {
            'total_validations': len(self.results),
            'success': success_count,
            'partial': partial_count,
            'failure': failure_count,
            'success_rate': success_count / len(self.results) * 100,
        }
        
        # 聚合指标
        all_metrics = {}
        for result in self.results.values():
            for key, value in result.comparison_metrics.items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
        
        # 计算统计量
        metric_stats = {}
        for key, values in all_metrics.items():
            metric_stats[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
            }
        
        summary['metric_statistics'] = metric_stats
        self.summary = summary
        
        return summary
    
    def save(self, output_dir: str):
        """保存所有结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存汇总
        summary_path = output_path / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'summary': self.summary,
                'failed_files': self.failed_files,
                'individual_results': {
                    name: result.to_dict() 
                    for name, result in self.results.items()
                }
            }, f, indent=2)
        
        logger.info(f"Saved batch results to {summary_path}")


class BatchValidator:
    """
    批量验证器
    
    用于批量验证多个实验-计算数据对
    
    使用示例:
        >>> config = BatchValidationConfig(
        ...     exp_data_dir="./experimental_data",
        ...     sim_data_dir="./simulated_data",
        ...     data_type="xrd"
        ... )
        >>> validator = BatchValidator(config)
        >>> results = validator.run_batch()
        >>> validator.save_results("./batch_results")
    """
    
    def __init__(self, config: Optional[BatchValidationConfig] = None):
        self.config = config or BatchValidationConfig()
        self.results = BatchValidationResult()
        
        # 创建输出目录
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run_batch(self, 
                  file_pairs: Optional[List[Tuple[str, str]]] = None) -> BatchValidationResult:
        """
        运行批量验证
        
        Args:
            file_pairs: 显式指定文件对列表 [(exp_file, sim_file), ...]
                       如果为None，则自动匹配目录中的文件
            
        Returns:
            BatchValidationResult对象
        """
        if file_pairs is None:
            file_pairs = self._auto_match_files()
        
        if not file_pairs:
            self.logger.warning("No file pairs to validate")
            return self.results
        
        self.logger.info(f"Starting batch validation of {len(file_pairs)} file pairs")
        
        if self.config.n_workers > 1:
            self._run_parallel(file_pairs)
        else:
            self._run_sequential(file_pairs)
        
        # 计算汇总
        self.results.calculate_summary()
        
        # 保存结果
        if self.config.summary_report:
            self.results.save(self.config.output_dir)
        
        self.logger.info("Batch validation completed")
        
        return self.results
    
    def _auto_match_files(self) -> List[Tuple[str, str]]:
        """自动匹配实验和模拟文件"""
        exp_dir = Path(self.config.exp_data_dir)
        sim_dir = Path(self.config.sim_data_dir)
        
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experimental data directory not found: {exp_dir}")
        if not sim_dir.exists():
            raise FileNotFoundError(f"Simulated data directory not found: {sim_dir}")
        
        # 获取文件列表
        exp_files = list(exp_dir.glob(self.config.exp_file_pattern))
        sim_files = list(sim_dir.glob(self.config.sim_file_pattern))
        
        self.logger.info(f"Found {len(exp_files)} experimental files and {len(sim_files)} simulated files")
        
        # 匹配文件
        pairs = []
        for exp_file in exp_files:
            # 尝试找到对应的模拟文件
            sim_file = self._find_matching_file(exp_file, sim_files)
            if sim_file:
                pairs.append((str(exp_file), str(sim_file)))
            else:
                self.logger.warning(f"No matching simulated file for {exp_file}")
                self.results.failed_files.append(str(exp_file))
        
        return pairs
    
    def _find_matching_file(self, exp_file: Path, sim_files: List[Path]) -> Optional[Path]:
        """找到匹配的模拟文件"""
        exp_stem = exp_file.stem
        
        # 尝试完全匹配
        for sim_file in sim_files:
            if sim_file.stem == exp_stem:
                return sim_file
        
        # 尝试移除常见后缀后的匹配
        suffixes_to_remove = ['_exp', '_experimental', '_sim', '_simulated', '_calc']
        
        exp_base = exp_stem
        for suffix in suffixes_to_remove:
            exp_base = exp_base.replace(suffix, '')
        
        for sim_file in sim_files:
            sim_base = sim_file.stem
            for suffix in suffixes_to_remove:
                sim_base = sim_base.replace(suffix, '')
            
            if sim_base == exp_base:
                return sim_file
        
        return None
    
    def _run_sequential(self, file_pairs: List[Tuple[str, str]]):
        """顺序运行验证"""
        for exp_file, sim_file in file_pairs:
            self._validate_single_pair(exp_file, sim_file)
    
    def _run_parallel(self, file_pairs: List[Tuple[str, str]]):
        """并行运行验证"""
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            futures = {
                executor.submit(self._validate_single_pair_wrapper, exp, sim): (exp, sim)
                for exp, sim in file_pairs
            }
            
            for future in as_completed(futures):
                exp_file, sim_file = futures[future]
                try:
                    name, result = future.result()
                    self.results.add_result(name, result)
                except Exception as e:
                    self.logger.error(f"Validation failed for {exp_file}: {e}")
                    self.results.failed_files.append(exp_file)
    
    def _validate_single_pair_wrapper(self, exp_file: str, sim_file: str) -> Tuple[str, ValidationResult]:
        """包装函数用于多进程"""
        return self._validate_single_pair(exp_file, sim_file)
    
    def _validate_single_pair(self, exp_file: str, sim_file: str) -> Tuple[str, ValidationResult]:
        """
        验证单个文件对
        
        Returns:
            (名称, 验证结果)
        """
        name = Path(exp_file).stem
        self.logger.info(f"Validating: {name}")
        
        try:
            # 创建验证工作流
            config = self.config.validation_config
            config.output_dir = str(Path(self.config.output_dir) / name)
            
            workflow = ValidationWorkflow(config)
            
            # 加载数据
            workflow.load_experimental_data(exp_file, data_type=self.config.data_type)
            workflow.load_computational_data(sim_file)
            
            # 运行验证
            result = workflow.run_validation()
            
            # 生成报告
            if self.config.individual_reports:
                workflow.generate_report()
            
            self.results.add_result(name, result)
            
            return name, result
            
        except Exception as e:
            self.logger.error(f"Failed to validate {name}: {e}")
            self.results.failed_files.append(exp_file)
            
            # 创建失败的结果
            failed_result = ValidationResult(
                success=False,
                data_type=self.config.data_type,
                errors=[str(e)]
            )
            self.results.add_result(name, failed_result)
            
            return name, failed_result
    
    def get_failed_validations(self) -> List[str]:
        """获取失败的验证列表"""
        return [
            name for name, result in self.results.results.items()
            if result.success == False
        ]
    
    def get_best_matches(self, metric: str = 'r2', n: int = 5) -> List[Tuple[str, float]]:
        """
        获取最佳匹配
        
        Args:
            metric: 用于排序的指标
            n: 返回的数量
            
        Returns:
            [(名称, 指标值), ...]
        """
        scores = []
        
        for name, result in self.results.results.items():
            if metric in result.comparison_metrics:
                scores.append((name, result.comparison_metrics[metric]))
        
        # 排序（R²等指标越大越好，RMSE等指标越小越好）
        reverse = metric in ['r2', 'pearson_r', 'cosine_similarity']
        scores.sort(key=lambda x: x[1], reverse=reverse)
        
        return scores[:n]
    
    def compare_systems(self) -> Dict[str, Any]:
        """
        比较不同系统的验证结果
        
        Returns:
            系统间比较结果
        """
        comparison = {
            'best_overall': None,
            'worst_overall': None,
            'by_metric': {}
        }
        
        # 找出整体最佳和最差
        avg_scores = {}
        for name, result in self.results.results.items():
            metrics = result.comparison_metrics
            if 'r2' in metrics:
                avg_scores[name] = metrics['r2']
        
        if avg_scores:
            comparison['best_overall'] = max(avg_scores, key=avg_scores.get)
            comparison['worst_overall'] = min(avg_scores, key=avg_scores.get)
        
        # 各指标最佳
        for metric in ['r2', 'rmse', 'mae']:
            best = self.get_best_matches(metric, n=1)
            if best:
                comparison['by_metric'][metric] = best[0]
        
        return comparison
    
    def generate_summary_plots(self, output_dir: Optional[str] = None):
        """
        生成汇总图表
        
        Args:
            output_dir: 输出目录
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not available for plotting")
            return
        
        output_dir = output_dir or self.config.output_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 提取数据
        names = list(self.results.results.keys())
        
        # R²分布图
        r2_values = [
            r.comparison_metrics.get('r2', 0) 
            for r in self.results.results.values()
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # R²柱状图
        axes[0, 0].bar(range(len(names)), r2_values)
        axes[0, 0].set_xticks(range(len(names)))
        axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].set_title('R² Score by System')
        axes[0, 0].axhline(y=0.9, color='r', linestyle='--', label='Threshold')
        axes[0, 0].legend()
        
        # RMSE分布
        rmse_values = [
            r.comparison_metrics.get('rmse', 0)
            for r in self.results.results.values()
        ]
        axes[0, 1].bar(range(len(names)), rmse_values, color='orange')
        axes[0, 1].set_xticks(range(len(names)))
        axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('RMSE by System')
        
        # 成功率饼图
        summary = self.results.summary
        success_data = [
            summary.get('success', 0),
            summary.get('partial', 0),
            summary.get('failure', 0)
        ]
        labels = ['Success', 'Partial', 'Failure']
        colors = ['green', 'orange', 'red']
        axes[1, 0].pie([x for x in success_data if x > 0], 
                      labels=[l for l, x in zip(labels, success_data) if x > 0],
                      colors=[c for c, x in zip(colors, success_data) if x > 0],
                      autopct='%1.1f%%')
        axes[1, 0].set_title('Validation Success Rate')
        
        # 散点图：R² vs RMSE
        axes[1, 1].scatter(rmse_values, r2_values, s=100, alpha=0.6)
        for i, name in enumerate(names):
            axes[1, 1].annotate(name, (rmse_values[i], r2_values[i]), 
                               fontsize=8, ha='right')
        axes[1, 1].set_xlabel('RMSE')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].set_title('R² vs RMSE')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = output_path / "batch_summary_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved summary plots to {plot_path}")
