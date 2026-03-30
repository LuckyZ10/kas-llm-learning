"""
Validation Workflow
===================
实验验证工作流

自动化对比计算结果与实验数据的流程
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np
import logging
from datetime import datetime

from ..connectors.base_connector import ExperimentalData
from ..connectors.xrd_connector import XRDConnector
from ..connectors.electrochemical_connector import ElectrochemicalConnector
from ..analyzers.structure_analyzer import XRDComparator, StructureComparator
from ..analyzers.performance_analyzer import ElectrochemicalComparator, PropertyComparator
from ..analyzers.statistical_analyzer import StatisticalAnalyzer
from ..analyzers.visualizer import ValidationVisualizer

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """验证配置"""
    # 数据设置
    experimental_data_path: str = ""
    computational_data_path: str = ""
    data_type: str = "auto"  # 'xrd', 'electrochemical', 'spectroscopy', 'auto'
    
    # 分析设置
    analysis_methods: List[str] = field(default_factory=lambda: ['all'])
    confidence_level: float = 0.95
    outlier_threshold: float = 3.0
    
    # 输出设置
    output_dir: str = "./validation_results"
    generate_report: bool = True
    generate_plots: bool = True
    report_format: str = "json"  # 'json', 'html', 'pdf'
    
    # 阈值设置
    tolerance_percent: float = 10.0
    rmse_threshold: float = 0.1
    r2_threshold: float = 0.9
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ValidationResult:
    """验证结果"""
    success: bool
    data_type: str
    comparison_metrics: Dict[str, Any] = field(default_factory=dict)
    statistical_analysis: Dict[str, Any] = field(default_factory=dict)
    outliers: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'data_type': self.data_type,
            'comparison_metrics': self.comparison_metrics,
            'statistical_analysis': self.statistical_analysis,
            'outliers': self.outliers,
            'warnings': self.warnings,
            'errors': self.errors,
            'timestamp': self.timestamp,
        }
    
    def save(self, filepath: str):
        """保存结果"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved validation result to {filepath}")


class ValidationWorkflow:
    """
    验证工作流
    
    主工作流类，协调数据加载、分析和报告生成
    
    使用示例:
        >>> workflow = ValidationWorkflow(config)
        >>> workflow.load_experimental_data("exp_data.csv", data_type="xrd")
        >>> workflow.load_computational_data("sim_data.csv")
        >>> result = workflow.run_validation()
        >>> workflow.generate_report("report.html")
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.exp_data: Optional[ExperimentalData] = None
        self.sim_data: Optional[ExperimentalData] = None
        self.result: Optional[ValidationResult] = None
        
        # 创建输出目录
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化分析器
        self.xrd_comparator = XRDComparator()
        self.echem_comparator = ElectrochemicalComparator()
        self.prop_comparator = PropertyComparator()
        self.stat_analyzer = StatisticalAnalyzer({
            'confidence_level': self.config.confidence_level,
            'outlier_threshold': self.config.outlier_threshold,
        })
        self.visualizer = ValidationVisualizer()
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_experimental_data(self, filepath: str, 
                               data_type: str = "auto",
                               **kwargs) -> ExperimentalData:
        """
        加载实验数据
        
        Args:
            filepath: 数据文件路径
            data_type: 数据类型 ('xrd', 'gcd', 'cv', 'eis', 'auto')
            **kwargs: 传递给连接器的参数
            
        Returns:
            ExperimentalData对象
        """
        data_type = data_type if data_type != "auto" else self.config.data_type
        
        if data_type == "auto":
            data_type = self._detect_data_type(filepath)
        
        if data_type in ['xrd']:
            connector = XRDConnector()
            self.exp_data = connector.read(filepath, **kwargs)
        elif data_type in ['gcd', 'cv', 'eis', 'electrochemical']:
            connector = ElectrochemicalConnector()
            test_type = 'gcd' if data_type == 'electrochemical' else data_type
            self.exp_data = connector.read(filepath, test_type=test_type, **kwargs)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        self.config.data_type = data_type
        self.logger.info(f"Loaded experimental {data_type} data from {filepath}")
        
        return self.exp_data
    
    def load_computational_data(self, filepath: str,
                                data_type: str = None,
                                **kwargs) -> ExperimentalData:
        """
        加载计算数据
        
        Args:
            filepath: 数据文件路径
            data_type: 数据类型（默认使用实验数据类型）
            **kwargs: 传递给连接器的参数
            
        Returns:
            ExperimentalData对象
        """
        data_type = data_type or self.config.data_type
        
        if data_type in ['xrd']:
            connector = XRDConnector()
            self.sim_data = connector.read(filepath, **kwargs)
        elif data_type in ['gcd', 'cv', 'eis', 'electrochemical']:
            connector = ElectrochemicalConnector()
            test_type = 'gcd' if data_type == 'electrochemical' else data_type
            self.sim_data = connector.read(filepath, test_type=test_type, **kwargs)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        self.logger.info(f"Loaded computational {data_type} data from {filepath}")
        
        return self.sim_data
    
    def _detect_data_type(self, filepath: str) -> str:
        """自动检测数据类型"""
        path = Path(filepath)
        filename = path.name.lower()
        
        if any(x in filename for x in ['xrd', 'diffraction', 'cif']):
            return 'xrd'
        elif any(x in filename for x in ['gcd', 'charge', 'discharge', 'cycle']):
            return 'gcd'
        elif any(x in filename for x in ['cv', 'voltammetry']):
            return 'cv'
        elif any(x in filename for x in ['eis', 'impedance']):
            return 'eis'
        else:
            # 尝试从文件内容检测
            return 'xrd'  # 默认
    
    def run_validation(self) -> ValidationResult:
        """
        运行验证流程
        
        Returns:
            ValidationResult对象
        """
        if self.exp_data is None or self.sim_data is None:
            raise ValueError("Both experimental and computational data must be loaded")
        
        self.logger.info("Starting validation workflow...")
        
        result = ValidationResult(
            success=True,
            data_type=self.config.data_type,
        )
        
        try:
            # 根据数据类型执行相应分析
            if self.config.data_type == 'xrd':
                self._validate_xrd(result)
            elif self.config.data_type in ['gcd', 'cv', 'eis']:
                self._validate_electrochemical(result)
            else:
                self._validate_generic(result)
            
            # 异常值检测
            result.outliers = self._detect_outliers()
            
            # 检查是否通过验证
            self._check_validation_pass(result)
            
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            self.logger.error(f"Validation failed: {e}")
        
        self.result = result
        self.logger.info("Validation workflow completed")
        
        return result
    
    def _validate_xrd(self, result: ValidationResult):
        """XRD验证"""
        self.logger.info("Running XRD validation...")
        
        # 结构比较
        comparison = self.xrd_comparator.compare(
            self.exp_data, self.sim_data, methods=['all']
        )
        result.comparison_metrics = comparison
        
        # 差分图谱
        diff_profile = self.xrd_comparator.calculate_difference_profile(
            self.exp_data, self.sim_data
        )
        
        # 保存差分数据
        diff_path = Path(self.config.output_dir) / "xrd_difference_profile.csv"
        np.savetxt(diff_path, diff_profile, delimiter=',',
                  header='2theta,exp_intensity,sim_intensity,difference')
        
        # 生成图表
        if self.config.generate_plots:
            plot_path = Path(self.config.output_dir) / "xrd_comparison.png"
            self.visualizer.plot_xrd_comparison(
                self.exp_data, self.sim_data, diff_profile, str(plot_path)
            )
    
    def _validate_electrochemical(self, result: ValidationResult):
        """电化学验证"""
        self.logger.info("Running electrochemical validation...")
        
        if self.config.data_type == 'gcd':
            # 充放电曲线对比
            comparison = self.echem_comparator.compare_gcd_curves(
                self.exp_data, self.sim_data
            )
            result.comparison_metrics = comparison
            
            # 生成图表
            if self.config.generate_plots:
                plot_path = Path(self.config.output_dir) / "gcd_comparison.png"
                self.visualizer.plot_gcd_comparison(
                    self.exp_data, self.sim_data, str(plot_path)
                )
                
        elif self.config.data_type == 'cv':
            # CV对比
            comparison = self.echem_comparator.compare_cv_curves(
                self.exp_data, self.sim_data
            )
            result.comparison_metrics = comparison
            
            if self.config.generate_plots:
                plot_path = Path(self.config.output_dir) / "cv_comparison.png"
                self.visualizer.plot_cv_comparison(
                    self.exp_data, self.sim_data, str(plot_path)
                )
    
    def _validate_generic(self, result: ValidationResult):
        """通用验证"""
        self.logger.info("Running generic validation...")
        
        # 提取数值数组
        exp_values = self.exp_data.processed_data[:, -1]  # 最后一列
        sim_values = self.sim_data.processed_data[:, -1]
        
        # 统计分析
        stat_results = self.stat_analyzer.comprehensive_analysis(exp_values, sim_values)
        result.statistical_analysis = stat_results
        result.comparison_metrics = stat_results['errors']
        
        # 生成图表
        if self.config.generate_plots:
            # 散点图
            scatter_path = Path(self.config.output_dir) / "scatter_plot.png"
            self.visualizer.plot_scatter(exp_values, sim_values, str(scatter_path))
            
            # 残差图
            residual_path = Path(self.config.output_dir) / "residual_plot.png"
            self.visualizer.plot_residuals(exp_values, sim_values, str(residual_path))
            
            # Bland-Altman图
            ba_path = Path(self.config.output_dir) / "bland_altman_plot.png"
            self.visualizer.plot_bland_altman(exp_values, sim_values, str(ba_path))
    
    def _detect_outliers(self) -> List[Dict]:
        """检测异常值"""
        exp_values = self.exp_data.processed_data[:, -1]
        sim_values = self.sim_data.processed_data[:, -1]
        
        outliers = self.stat_analyzer.detect_outliers(exp_values, sim_values)
        
        if outliers:
            self.logger.warning(f"Detected {len(outliers)} outliers")
        
        return outliers
    
    def _check_validation_pass(self, result: ValidationResult):
        """检查验证是否通过"""
        metrics = result.comparison_metrics
        
        # 检查RMSE
        if 'rmse' in metrics:
            if metrics['rmse'] > self.config.rmse_threshold:
                result.warnings.append(
                    f"RMSE ({metrics['rmse']:.4f}) exceeds threshold ({self.config.rmse_threshold})"
                )
        
        # 检查R²
        if 'r2' in metrics:
            if metrics['r2'] < self.config.r2_threshold:
                result.warnings.append(
                    f"R² ({metrics['r2']:.4f}) below threshold ({self.config.r2_threshold})"
                )
        
        # 如果有警告，标记为部分成功
        if result.warnings:
            result.success = 'partial'
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        生成验证报告
        
        Args:
            output_path: 输出路径（默认使用配置中的output_dir）
            
        Returns:
            报告文件路径
        """
        if self.result is None:
            raise ValueError("Must run validation before generating report")
        
        output_path = output_path or Path(self.config.output_dir) / f"validation_report.{self.config.report_format}"
        output_path = Path(output_path)
        
        if self.config.report_format == 'json':
            self._generate_json_report(output_path)
        elif self.config.report_format == 'html':
            self._generate_html_report(output_path)
        else:
            self._generate_text_report(output_path)
        
        self.logger.info(f"Generated validation report: {output_path}")
        
        return str(output_path)
    
    def _generate_json_report(self, output_path: Path):
        """生成JSON报告"""
        report = {
            'validation_summary': {
                'success': self.result.success,
                'data_type': self.result.data_type,
                'timestamp': self.result.timestamp,
                'config': self.config.to_dict(),
            },
            'comparison_metrics': self.result.comparison_metrics,
            'statistical_analysis': self.result.statistical_analysis,
            'outliers': self.result.outliers,
            'warnings': self.result.warnings,
            'errors': self.result.errors,
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def _generate_html_report(self, output_path: Path):
        """生成HTML报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
                .success {{ color: green; }}
                .partial {{ color: orange; }}
                .failure {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-family: monospace; background-color: #f5f5f5; padding: 2px 5px; }}
            </style>
        </head>
        <body>
            <h1>Validation Report</h1>
            
            <h2>Summary</h2>
            <p>Data Type: {self.result.data_type}</p>
            <p>Status: 
                <span class="{'success' if self.result.success == True else 'partial' if self.result.success == 'partial' else 'failure'}">
                    {self.result.success}
                </span>
            </p>
            <p>Timestamp: {self.result.timestamp}</p>
            
            <h2>Comparison Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
        """
        
        for key, value in self.result.comparison_metrics.items():
            if isinstance(value, float):
                html_content += f"                <tr><td>{key}</td><td>{value:.4f}</td></tr>\n"
            else:
                html_content += f"                <tr><td>{key}</td><td>{value}</td></tr>\n"
        
        html_content += """
            </table>
            
            <h2>Warnings</h2>
            <ul>
        """
        
        for warning in self.result.warnings:
            html_content += f"                <li>{warning}</li>\n"
        
        if not self.result.warnings:
            html_content += "                <li>No warnings</li>\n"
        
        html_content += """
            </ul>
            
            <h2>Errors</h2>
            <ul>
        """
        
        for error in self.result.errors:
            html_content += f"                <li>{error}</li>\n"
        
        if not self.result.errors:
            html_content += "                <li>No errors</li>\n"
        
        html_content += """
            </ul>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_text_report(self, output_path: Path):
        """生成文本报告"""
        lines = [
            "=" * 60,
            "VALIDATION REPORT",
            "=" * 60,
            "",
            f"Data Type: {self.result.data_type}",
            f"Status: {self.result.success}",
            f"Timestamp: {self.result.timestamp}",
            "",
            "COMPARISON METRICS",
            "-" * 40,
        ]
        
        for key, value in self.result.comparison_metrics.items():
            if isinstance(value, float):
                lines.append(f"{key:30s}: {value:.4f}")
            else:
                lines.append(f"{key:30s}: {value}")
        
        lines.extend([
            "",
            "WARNINGS",
            "-" * 40,
        ])
        
        if self.result.warnings:
            for warning in self.result.warnings:
                lines.append(f"- {warning}")
        else:
            lines.append("No warnings")
        
        lines.extend([
            "",
            "ERRORS",
            "-" * 40,
        ])
        
        if self.result.errors:
            for error in self.result.errors:
                lines.append(f"- {error}")
        else:
            lines.append("No errors")
        
        lines.extend([
            "",
            "=" * 60,
        ])
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
