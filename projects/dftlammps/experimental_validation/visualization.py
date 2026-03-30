"""
实验数据验证可视化模块
提供误差统计、对比图表、验证报告生成
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings

from .comparison import (
    ComparisonResult, ValidationReport, StatisticalAnalysis,
    StructureComparator
)
from .data_formats import CrystalStructure, ExperimentalDataset

# 尝试导入matplotlib，如果不可用则提供模拟实现
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Visualization functions will be limited.")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class PlotConfig:
    """绘图配置"""
    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 150
    style: str = 'seaborn-v0_8-darkgrid' if MATPLOTLIB_AVAILABLE else 'default'
    color_primary: str = '#3498db'
    color_secondary: str = '#e74c3c'
    color_accent: str = '#2ecc71'
    save_path: Optional[str] = None
    show_plot: bool = True


class ValidationVisualizer:
    """验证结果可视化器"""
    
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        if MATPLOTLIB_AVAILABLE:
            plt.style.use(self.config.style)
    
    def plot_parity(self, 
                   computed: List[float],
                   experimental: List[float],
                   uncertainties: Optional[List[float]] = None,
                   property_name: str = "Property",
                   unit: str = "",
                   **kwargs) -> Any:
        """绘制parity plot（计算值vs实验值）"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for plotting")
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        computed = np.array(computed)
        experimental = np.array(experimental)
        
        # 计算范围
        min_val = min(computed.min(), experimental.min())
        max_val = max(computed.max(), experimental.max())
        padding = (max_val - min_val) * 0.05
        
        # 绘制对角线（完美匹配）
        ax.plot([min_val-padding, max_val+padding], 
                [min_val-padding, max_val+padding], 
                'k--', alpha=0.5, linewidth=1, label='Perfect agreement')
        
        # 绘制±10%误差线
        ax.plot([min_val-padding, max_val+padding], 
                [min_val-padding*0.9, max_val+padding*0.9], 
                'k:', alpha=0.3, linewidth=1, label='±10%')
        ax.plot([min_val-padding, max_val+padding], 
                [min_val-padding*1.1, max_val+padding*1.1], 
                'k:', alpha=0.3, linewidth=1)
        
        # 绘制数据点
        if uncertainties:
            ax.errorbar(experimental, computed, xerr=uncertainties, 
                       fmt='o', color=self.config.color_primary,
                       ecolor='gray', alpha=0.7, capsize=3, label='Data')
        else:
            ax.scatter(experimental, computed, 
                      color=self.config.color_primary, alpha=0.7, s=50)
        
        # 计算并显示R²
        if len(computed) > 1 and SCIPY_AVAILABLE:
            r, _ = stats.pearsonr(computed, experimental)
            r2 = r ** 2
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', 
                   transform=ax.transAxes, fontsize=12,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(f'Experimental {property_name} ({unit})', fontsize=12)
        ax.set_ylabel(f'Computed {property_name} ({unit})', fontsize=12)
        ax.set_title(f'Parity Plot: {property_name}', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.save_path:
            plt.savefig(f"{self.config.save_path}_{property_name}_parity.png", 
                       dpi=self.config.dpi, bbox_inches='tight')
        
        if self.config.show_plot:
            plt.show()
        
        return fig
    
    def plot_error_distribution(self,
                               errors: List[float],
                               property_name: str = "Property",
                               **kwargs) -> Any:
        """绘制误差分布直方图"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for plotting")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        errors = np.array(errors)
        
        # 直方图
        ax1 = axes[0]
        n, bins, patches = ax1.hist(errors, bins=20, color=self.config.color_primary, 
                                     alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
        ax1.axvline(x=np.mean(errors), color='green', linestyle='-', linewidth=2, 
                   label=f'Mean: {np.mean(errors):.3f}')
        ax1.set_xlabel('Error', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Error Distribution: {property_name}', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q图
        ax2 = axes[1]
        if SCIPY_AVAILABLE:
            stats.probplot(errors, dist="norm", plot=ax2)
            ax2.set_title(f'Q-Q Plot: {property_name}', fontsize=14)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Scipy not available', ha='center', va='center')
        
        plt.tight_layout()
        
        if self.config.save_path:
            plt.savefig(f"{self.config.save_path}_{property_name}_error_dist.png", 
                       dpi=self.config.dpi, bbox_inches='tight')
        
        if self.config.show_plot:
            plt.show()
        
        return fig
    
    def plot_validation_report(self, 
                              report: ValidationReport,
                              **kwargs) -> Any:
        """绘制完整验证报告"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for plotting")
            return None
        
        if not report.results:
            print("No results to plot")
            return None
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig)
        
        computed = [r.computed_value for r in report.results]
        experimental = [r.experimental_value for r in report.results]
        errors = [r.relative_error * 100 for r in report.results]
        uncertainties = [r.experimental_uncertainty for r in report.results 
                        if r.experimental_uncertainty is not None]
        
        # 1. Parity plot
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_parity_on_ax(ax1, computed, experimental, uncertainties, report.property_name)
        
        # 2. 误差分布
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(errors, bins=15, color=self.config.color_primary, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.axvline(x=np.mean(errors), color='green', linestyle='-', linewidth=2)
        ax2.set_xlabel('Relative Error (%)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Error Distribution', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 3. 误差vs实验值
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(experimental, errors, color=self.config.color_secondary, alpha=0.7)
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax3.axhline(y=10, color='orange', linestyle=':', linewidth=1, label='±10%')
        ax3.axhline(y=-10, color='orange', linestyle=':', linewidth=1)
        ax3.set_xlabel('Experimental Value', fontsize=11)
        ax3.set_ylabel('Relative Error (%)', fontsize=11)
        ax3.set_title('Error vs Experimental Value', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 一致性等级分布
        ax4 = fig.add_subplot(gs[1, 0])
        levels = [r.agreement_level.value for r in report.results]
        level_counts = {}
        for level in levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6']
        ax4.bar(level_counts.keys(), level_counts.values(), color=colors[:len(level_counts)])
        ax4.set_xlabel('Agreement Level', fontsize=11)
        ax4.set_ylabel('Count', fontsize=11)
        ax4.set_title('Agreement Level Distribution', fontsize=12)
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. 统计指标表格
        ax5 = fig.add_subplot(gs[1, 1:])
        ax5.axis('off')
        
        if report.statistics:
            stat = report.statistics
            table_data = [
                ['Metric', 'Value'],
                ['Number of Samples', f'{stat.n_samples}'],
                ['MAE', f'{stat.mae:.4f}'],
                ['RMSE', f'{stat.rmse:.4f}'],
                ['MAPE (%)', f'{stat.mape:.2f}'],
                ['R²', f'{stat.r2:.4f}'],
                ['Pearson r', f'{stat.pearson_r:.4f}'],
                ['Mean Bias', f'{stat.mean_bias:.4f}'],
                ['Within 2σ', f'{stat.within_2sigma*100:.1f}%'],
                ['Validation Score', f'{report.validation_score:.1f}/100'],
                ['Validated', '✓ Yes' if report.is_validated else '✗ No']
            ]
            
            table = ax5.table(cellText=table_data, cellLoc='left', loc='center',
                            colWidths=[0.4, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # 高亮表头
            for i in range(2):
                table[(0, i)].set_facecolor('#3498db')
                table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.suptitle(f'Validation Report: {report.property_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if self.config.save_path:
            plt.savefig(f"{self.config.save_path}_{report.property_name}_report.png", 
                       dpi=self.config.dpi, bbox_inches='tight')
        
        if self.config.show_plot:
            plt.show()
        
        return fig
    
    def _plot_parity_on_ax(self, ax, computed, experimental, uncertainties, property_name):
        """在指定轴上绘制parity图"""
        computed = np.array(computed)
        experimental = np.array(experimental)
        
        min_val = min(computed.min(), experimental.min())
        max_val = max(computed.max(), experimental.max())
        padding = (max_val - min_val) * 0.05
        
        ax.plot([min_val-padding, max_val+padding], 
                [min_val-padding, max_val+padding], 
                'k--', alpha=0.5, linewidth=1)
        ax.scatter(experimental, computed, color=self.config.color_primary, alpha=0.7, s=40)
        
        if uncertainties:
            # 简化的误差条
            pass
        
        ax.set_xlabel('Experimental', fontsize=10)
        ax.set_ylabel('Computed', fontsize=10)
        ax.set_title('Parity Plot', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    def plot_structure_comparison(self,
                                 struct1: CrystalStructure,
                                 struct2: CrystalStructure,
                                 labels: Tuple[str, str] = ("Computed", "Experimental"),
                                 **kwargs) -> Any:
        """可视化结构比较"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for plotting")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 绘制两个结构
        for idx, (struct, label) in enumerate([(struct1, labels[0]), (struct2, labels[1])]):
            ax = axes[idx]
            self._plot_structure_2d(ax, struct, label)
        
        # 添加比较信息
        comparison = StructureComparator.compare_lattice(struct1, struct2)
        info_text = f"Lattice Error: {comparison['mean_lattice_error']:.2f}%"
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=12)
        
        plt.tight_layout()
        
        if self.config.save_path:
            plt.savefig(f"{self.config.save_path}_structure_comparison.png", 
                       dpi=self.config.dpi, bbox_inches='tight')
        
        if self.config.show_plot:
            plt.show()
        
        return fig
    
    def _plot_structure_2d(self, ax, structure: CrystalStructure, title: str):
        """2D可视化晶体结构（简化）"""
        # 提取原子位置（投影到xy平面）
        coords = np.array([[site.x, site.y, site.z] for site in structure.sites])
        elements = [site.element for site in structure.sites]
        
        # 元素到颜色的映射
        element_colors = {
            'H': 'white', 'C': 'gray', 'N': 'blue', 'O': 'red',
            'F': 'green', 'Na': 'purple', 'Mg': 'orange', 'Al': 'silver',
            'Si': 'yellow', 'P': 'orange', 'S': 'yellow', 'Cl': 'green',
            'K': 'purple', 'Ca': 'green', 'Ti': 'gray', 'Fe': 'orange',
            'Co': 'blue', 'Ni': 'green', 'Cu': 'orange', 'Zn': 'gray',
            'Ga': 'silver', 'Ge': 'gray', 'As': 'purple', 'Se': 'orange',
            'Li': 'purple'
        }
        
        colors = [element_colors.get(e, 'pink') for e in elements]
        sizes = [100 if e in ['Na', 'Cl', 'K'] else 50 for e in elements]
        
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=sizes, edgecolors='black', alpha=0.7)
        
        # 添加元素标签
        for i, (coord, elem) in enumerate(zip(coords, elements)):
            ax.annotate(elem, (coord[0], coord[1]), fontsize=8, ha='center')
        
        # 绘制晶胞边界
        a, b = structure.lattice.a, structure.lattice.b
        rect = mpatches.Rectangle((0, 0), 1, 1, linewidth=2, 
                                   edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('x (fractional)', fontsize=10)
        ax.set_ylabel('y (fractional)', fontsize=10)
        ax.grid(True, alpha=0.3)


class ReportGenerator:
    """验证报告生成器"""
    
    def __init__(self):
        self.reports: List[ValidationReport] = []
    
    def add_report(self, report: ValidationReport):
        """添加验证报告"""
        self.reports.append(report)
    
    def generate_html_report(self, output_path: str):
        """生成HTML报告"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>DFT-LAMMPS 实验验证报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; 
                            border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                h2 { color: #34495e; margin-top: 30px; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background: #3498db; color: white; }
                tr:hover { background: #f5f5f5; }
                .metric-box { display: inline-block; margin: 10px; padding: 20px; 
                             background: #ecf0f1; border-radius: 8px; min-width: 150px; }
                .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
                .metric-label { font-size: 14px; color: #7f8c8d; }
                .status-pass { color: #27ae60; font-weight: bold; }
                .status-fail { color: #e74c3c; font-weight: bold; }
                .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                               gap: 20px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔬 DFT-LAMMPS 实验数据验证报告</h1>
                <div class="summary-grid">
        """
        
        # 总体统计
        total_reports = len(self.reports)
        validated = sum(1 for r in self.reports if r.is_validated)
        avg_score = np.mean([r.validation_score for r in self.reports]) if self.reports else 0
        
        html_content += f"""
                    <div class="metric-box">
                        <div class="metric-value">{total_reports}</div>
                        <div class="metric-label">验证属性数</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{validated}/{total_reports}</div>
                        <div class="metric-label">通过验证</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{avg_score:.1f}</div>
                        <div class="metric-label">平均得分</div>
                    </div>
                </div>
        """
        
        # 详细报告
        html_content += "<h2>详细验证结果</h2><table>"
        html_content += "<tr><th>属性</th><th>样本数</th><th>MAE</th><th>RMSE</th><th>MAPE(%)</th><th>R²</th><th>得分</th><th>状态</th></tr>"
        
        for report in self.reports:
            if report.statistics:
                stat = report.statistics
                status_class = "status-pass" if report.is_validated else "status-fail"
                status_text = "✓ 通过" if report.is_validated else "✗ 未通过"
                
                html_content += f"""
                    <tr>
                        <td>{report.property_name}</td>
                        <td>{stat.n_samples}</td>
                        <td>{stat.mae:.4f}</td>
                        <td>{stat.rmse:.4f}</td>
                        <td>{stat.mape:.2f}</td>
                        <td>{stat.r2:.4f}</td>
                        <td>{report.validation_score:.1f}</td>
                        <td class="{status_class}">{status_text}</td>
                    </tr>
                """
        
        html_content += "</table>"
        
        # 页脚
        html_content += f"""
                <p style="margin-top: 40px; color: #7f8c8d; text-align: center;">
                    生成时间: {np.datetime64('now')}
                </p>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML报告已保存: {output_path}")
    
    def generate_markdown_report(self, output_path: str):
        """生成Markdown报告"""
        lines = [
            "# 🔬 DFT-LAMMPS 实验数据验证报告",
            "",
            "## 总体概况",
            ""
        ]
        
        total_reports = len(self.reports)
        validated = sum(1 for r in self.reports if r.is_validated)
        avg_score = np.mean([r.validation_score for r in self.reports]) if self.reports else 0
        
        lines.extend([
            f"- **验证属性数**: {total_reports}",
            f"- **通过验证**: {validated}/{total_reports}",
            f"- **平均得分**: {avg_score:.1f}/100",
            "",
            "## 详细验证结果",
            "",
            "| 属性 | 样本数 | MAE | RMSE | MAPE(%) | R² | 得分 | 状态 |",
            "|------|--------|-----|------|---------|----|-------|------|"
        ])
        
        for report in self.reports:
            if report.statistics:
                stat = report.statistics
                status = "✓ 通过" if report.is_validated else "✗ 未通过"
                lines.append(
                    f"| {report.property_name} | {stat.n_samples} | {stat.mae:.4f} | "
                    f"{stat.rmse:.4f} | {stat.mape:.2f} | {stat.r2:.4f} | "
                    f"{report.validation_score:.1f} | {status} |"
                )
        
        lines.extend([
            "",
            "## 结论",
            "",
            f"本报告共验证 {total_reports} 个属性，其中 {validated} 个通过验证。",
            f"总体平均得分为 {avg_score:.1f}/100。",
            "",
            f"---",
            f"*生成时间: {np.datetime64('now')}*"
        ])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"Markdown报告已保存: {output_path}")


# =============================================================================
# 便捷函数
# =============================================================================

def plot_parity(computed: List[float], 
               experimental: List[float],
               property_name: str = "Property",
               save_path: Optional[str] = None) -> Any:
    """便捷函数：绘制parity图"""
    config = PlotConfig(save_path=save_path, show_plot=True)
    viz = ValidationVisualizer(config)
    return viz.plot_parity(computed, experimental, property_name=property_name)


def plot_validation_summary(report: ValidationReport, save_path: Optional[str] = None) -> Any:
    """便捷函数：绘制验证摘要"""
    config = PlotConfig(save_path=save_path, show_plot=True)
    viz = ValidationVisualizer(config)
    return viz.plot_validation_report(report)


def generate_full_report(reports: List[ValidationReport], 
                        output_dir: str,
                        formats: List[str] = ['html', 'md']) -> Dict[str, str]:
    """生成完整报告（多种格式）"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    generator = ReportGenerator()
    for report in reports:
        generator.add_report(report)
    
    output_paths = {}
    
    if 'html' in formats:
        html_path = os.path.join(output_dir, 'validation_report.html')
        generator.generate_html_report(html_path)
        output_paths['html'] = html_path
    
    if 'md' in formats:
        md_path = os.path.join(output_dir, 'validation_report.md')
        generator.generate_markdown_report(md_path)
        output_paths['md'] = md_path
    
    return output_paths


# =============================================================================
# 演示
# =============================================================================

def demo():
    """演示可视化功能"""
    print("=" * 80)
    print("📊 实验数据验证可视化演示")
    print("=" * 80)
    
    if not MATPLOTLIB_AVAILABLE:
        print("\n⚠️ Matplotlib 不可用，跳过绘图演示")
        print("请安装 matplotlib: pip install matplotlib")
        return
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 50
    
    # 模拟带噪声的数据
    true_values = np.random.uniform(1, 10, n_samples)
    experimental = true_values + np.random.normal(0, 0.15, n_samples)
    computed = true_values + np.random.normal(0.1, 0.25, n_samples)
    uncertainties = np.random.uniform(0.05, 0.3, n_samples)
    
    print("\n🔹 创建可视化配置...")
    config = PlotConfig(
        figsize=(10, 6),
        dpi=100,
        show_plot=False,  # 非交互式环境不显示
        save_path='/tmp/demo_validation'
    )
    
    viz = ValidationVisualizer(config)
    
    print("   ✓ 可视化器已创建")
    
    # 生成验证报告
    from .comparison import validate_properties
    
    print("\n🔹 生成验证报告...")
    report = validate_properties(
        computed.tolist(),
        experimental.tolist(),
        "band_gap",
        uncertainties.tolist()
    )
    
    print(f"   ✓ 报告: {report.property_name}")
    print(f"   ✓ 样本数: {len(report.results)}")
    
    # 演示报告生成器
    print("\n🔹 生成综合报告...")
    generator = ReportGenerator()
    generator.add_report(report)
    
    # 添加更多报告
    for prop in ['lattice_constant', 'bulk_modulus']:
        n = np.random.randint(20, 40)
        true = np.random.uniform(1, 100, n)
        exp = true + np.random.normal(0, 0.05 * true.std(), n)
        comp = true + np.random.normal(0, 0.08 * true.std(), n)
        r = validate_properties(comp.tolist(), exp.tolist(), prop)
        generator.add_report(r)
    
    # 生成HTML报告
    try:
        html_path = '/tmp/demo_validation_report.html'
        generator.generate_html_report(html_path)
        print(f"   ✓ HTML报告: {html_path}")
        
        md_path = '/tmp/demo_validation_report.md'
        generator.generate_markdown_report(md_path)
        print(f"   ✓ Markdown报告: {md_path}")
    except Exception as e:
        print(f"   ✗ 报告生成失败: {e}")
    
    print("\n🔹 验证统计摘要:")
    if report.statistics:
        stat = report.statistics
        print(f"   MAE: {stat.mae:.4f}")
        print(f"   RMSE: {stat.rmse:.4f}")
        print(f"   MAPE: {stat.mape:.2f}%")
        print(f"   R²: {stat.r2:.4f}")
        print(f"   验证得分: {report.validation_score:.1f}/100")
        print(f"   通过验证: {'是' if report.is_validated else '否'}")
    
    print("\n" + "=" * 80)
    print("✅ 可视化演示完成!")
    print("=" * 80)


if __name__ == '__main__':
    demo()
