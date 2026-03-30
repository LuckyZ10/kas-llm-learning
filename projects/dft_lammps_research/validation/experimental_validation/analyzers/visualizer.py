"""
Validation Visualizer
=====================
验证结果可视化工具

生成:
- 对比图表
- 差异热图
- 散点图
- 残差图
- Bland-Altman图
- 统计分布图
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)


class ValidationVisualizer:
    """
    验证结果可视化器
    
    用于生成各种对比可视化图表
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.config.setdefault('figure_size', (10, 6))
        self.config.setdefault('dpi', 150)
        self.config.setdefault('style', 'seaborn-v0_8-whitegrid')
        
        # 设置样式
        try:
            import matplotlib.pyplot as plt
            plt.style.use(self.config['style'])
        except:
            pass
    
    def plot_xrd_comparison(self,
                           experimental_data,
                           simulated_data,
                           difference_profile: Optional[np.ndarray] = None,
                           save_path: Optional[str] = None,
                           title: str = "XRD Pattern Comparison") -> Any:
        """
        绘制XRD对比图
        
        Args:
            experimental_data: 实验XRD数据
            simulated_data: 模拟XRD数据
            difference_profile: 差分图谱（可选）
            save_path: 保存路径
            title: 图表标题
            
        Returns:
            matplotlib figure对象
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization")
        
        # 确定子图数量
        n_plots = 3 if difference_profile is not None else 2
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3*n_plots), 
                                 sharex=True, dpi=self.config['dpi'])
        if n_plots == 1:
            axes = [axes]
        
        # 提取数据
        exp_2theta = experimental_data.processed_data[:, 0]
        exp_intensity = experimental_data.processed_data[:, 1]
        sim_2theta = simulated_data.processed_data[:, 0]
        sim_intensity = simulated_data.processed_data[:, 1]
        
        # 实验数据
        axes[0].plot(exp_2theta, exp_intensity, 'b-', label='Experimental', linewidth=1)
        axes[0].set_ylabel('Intensity (a.u.)')
        axes[0].legend()
        axes[0].set_title(title)
        
        # 模拟数据
        axes[1].plot(sim_2theta, sim_intensity, 'r-', label='Simulated', linewidth=1)
        axes[1].set_ylabel('Intensity (a.u.)')
        axes[1].legend()
        
        # 差分图谱
        if difference_profile is not None and n_plots > 2:
            axes[2].plot(difference_profile[:, 0], difference_profile[:, 3], 
                        'g-', label='Difference', linewidth=1)
            axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            axes[2].set_xlabel('2θ (degrees)')
            axes[2].set_ylabel('Difference')
            axes[2].legend()
        else:
            axes[-1].set_xlabel('2θ (degrees)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info(f"Saved XRD comparison plot to {save_path}")
        
        return fig
    
    def plot_gcd_comparison(self,
                           experimental_data,
                           simulated_data,
                           save_path: Optional[str] = None,
                           title: str = "Galvanostatic Charge-Discharge") -> Any:
        """
        绘制充放电曲线对比图
        
        Args:
            experimental_data: 实验GCD数据
            simulated_data: 模拟GCD数据
            save_path: 保存路径
            title: 图表标题
            
        Returns:
            matplotlib figure对象
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization")
        
        fig, ax = plt.subplots(figsize=self.config['figure_size'], dpi=self.config['dpi'])
        
        # 提取数据
        exp_df = experimental_data.to_dataframe()
        sim_df = simulated_data.to_dataframe()
        
        # 找到容量和电压列
        cap_col_exp = self._find_column(exp_df, ['capacity', 'specific_capacity'])
        v_col_exp = self._find_column(exp_df, ['voltage', 'potential'])
        
        cap_col_sim = self._find_column(sim_df, ['capacity', 'specific_capacity'])
        v_col_sim = self._find_column(sim_df, ['voltage', 'potential'])
        
        if cap_col_exp and v_col_exp:
            ax.plot(exp_df[cap_col_exp], exp_df[v_col_exp], 
                   'b-', label='Experimental', linewidth=2)
        
        if cap_col_sim and v_col_sim:
            ax.plot(sim_df[cap_col_sim], sim_df[v_col_sim],
                   'r--', label='Simulated', linewidth=2)
        
        ax.set_xlabel('Specific Capacity (mAh/g)')
        ax.set_ylabel('Voltage (V)')
        ax.legend()
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info(f"Saved GCD comparison plot to {save_path}")
        
        return fig
    
    def _find_column(self, df, possible_names: List[str]) -> Optional[str]:
        """查找列名"""
        for col in df.columns:
            col_lower = col.lower()
            for name in possible_names:
                if name in col_lower:
                    return col
        return None
    
    def plot_scatter(self,
                    experimental: np.ndarray,
                    predicted: np.ndarray,
                    save_path: Optional[str] = None,
                    title: str = "Experimental vs Predicted",
                    xlabel: str = "Experimental",
                    ylabel: str = "Predicted") -> Any:
        """
        绘制散点图
        
        Args:
            experimental: 实验值
            predicted: 预测值
            save_path: 保存路径
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
            
        Returns:
            matplotlib figure对象
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization")
        
        fig, ax = plt.subplots(figsize=self.config['figure_size'], dpi=self.config['dpi'])
        
        # 散点图
        ax.scatter(experimental, predicted, alpha=0.5, s=20)
        
        # 对角线（理想情况）
        min_val = min(np.min(experimental), np.min(predicted))
        max_val = max(np.max(experimental), np.max(predicted))
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', label='Perfect Match', linewidth=2)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info(f"Saved scatter plot to {save_path}")
        
        return fig
    
    def plot_residuals(self,
                      experimental: np.ndarray,
                      predicted: np.ndarray,
                      save_path: Optional[str] = None,
                      title: str = "Residual Analysis") -> Any:
        """
        绘制残差图
        
        包括残差分布直方图和残差vs预测值图
        
        Args:
            experimental: 实验值
            predicted: 预测值
            save_path: 保存路径
            title: 图表标题
            
        Returns:
            matplotlib figure对象
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization")
        
        residuals = experimental - predicted
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=self.config['dpi'])
        
        # 残差分布
        axes[0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Residual')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Residual Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # 残差vs预测值
        axes[1].scatter(predicted, residuals, alpha=0.5, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Residual')
        axes[1].set_title('Residuals vs Predicted')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info(f"Saved residual plot to {save_path}")
        
        return fig
    
    def plot_bland_altman(self,
                         experimental: np.ndarray,
                         predicted: np.ndarray,
                         save_path: Optional[str] = None,
                         title: str = "Bland-Altman Plot") -> Any:
        """
        绘制Bland-Altman图
        
        Args:
            experimental: 实验值
            predicted: 预测值
            save_path: 保存路径
            title: 图表标题
            
        Returns:
            matplotlib figure对象
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization")
        
        differences = experimental - predicted
        means = (experimental + predicted) / 2
        
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        loa_lower = mean_diff - 1.96 * std_diff
        loa_upper = mean_diff + 1.96 * std_diff
        
        fig, ax = plt.subplots(figsize=self.config['figure_size'], dpi=self.config['dpi'])
        
        # 散点图
        ax.scatter(means, differences, alpha=0.5, s=20)
        
        # 参考线
        ax.axhline(y=mean_diff, color='r', linestyle='-', linewidth=2, label=f'Mean: {mean_diff:.3f}')
        ax.axhline(y=loa_lower, color='g', linestyle='--', linewidth=1.5, label=f'-1.96 SD: {loa_lower:.3f}')
        ax.axhline(y=loa_upper, color='g', linestyle='--', linewidth=1.5, label=f'+1.96 SD: {loa_upper:.3f}')
        ax.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Mean of Experimental and Predicted')
        ax.set_ylabel('Difference (Experimental - Predicted)')
        ax.legend()
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info(f"Saved Bland-Altman plot to {save_path}")
        
        return fig
    
    def plot_heatmap(self,
                    data_matrix: np.ndarray,
                    row_labels: List[str],
                    col_labels: List[str],
                    save_path: Optional[str] = None,
                    title: str = "Difference Heatmap",
                    cmap: str = 'RdBu_r') -> Any:
        """
        绘制差异热图
        
        Args:
            data_matrix: 数据矩阵
            row_labels: 行标签
            col_labels: 列标签
            save_path: 保存路径
            title: 图表标题
            cmap: 颜色映射
            
        Returns:
            matplotlib figure对象
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization")
        
        fig, ax = plt.subplots(figsize=self.config['figure_size'], dpi=self.config['dpi'])
        
        # 归一化颜色范围
        vmax = np.max(np.abs(data_matrix))
        vmin = -vmax
        
        im = ax.imshow(data_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        # 设置刻度
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加数值
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info(f"Saved heatmap to {save_path}")
        
        return fig
    
    def plot_cycling_stability(self,
                              cycles_exp: List[int],
                              capacities_exp: List[float],
                              cycles_sim: List[int],
                              capacities_sim: List[float],
                              save_path: Optional[str] = None,
                              title: str = "Cycling Stability") -> Any:
        """
        绘制循环稳定性对比图
        
        Args:
            cycles_exp: 实验循环数
            capacities_exp: 实验容量
            cycles_sim: 模拟循环数
            capacities_sim: 模拟容量
            save_path: 保存路径
            title: 图表标题
            
        Returns:
            matplotlib figure对象
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization")
        
        fig, ax = plt.subplots(figsize=self.config['figure_size'], dpi=self.config['dpi'])
        
        ax.plot(cycles_exp, capacities_exp, 'b-o', label='Experimental', 
               linewidth=2, markersize=6)
        ax.plot(cycles_sim, capacities_sim, 'r--s', label='Simulated',
               linewidth=2, markersize=6)
        
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Specific Capacity (mAh/g)')
        ax.legend()
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info(f"Saved cycling stability plot to {save_path}")
        
        return fig
    
    def plot_cv_comparison(self,
                          experimental_data,
                          simulated_data,
                          save_path: Optional[str] = None,
                          title: str = "Cyclic Voltammetry") -> Any:
        """
        绘制CV曲线对比图
        
        Args:
            experimental_data: 实验CV数据
            simulated_data: 模拟CV数据
            save_path: 保存路径
            title: 图表标题
            
        Returns:
            matplotlib figure对象
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization")
        
        fig, ax = plt.subplots(figsize=self.config['figure_size'], dpi=self.config['dpi'])
        
        # 提取数据
        exp_df = experimental_data.to_dataframe()
        sim_df = simulated_data.to_dataframe()
        
        v_col_exp = self._find_column(exp_df, ['voltage', 'potential', 'ewe'])
        i_col_exp = self._find_column(exp_df, ['current', 'i', '<i>'])
        
        v_col_sim = self._find_column(sim_df, ['voltage', 'potential', 'ewe'])
        i_col_sim = self._find_column(sim_df, ['current', 'i', '<i>'])
        
        if v_col_exp and i_col_exp:
            ax.plot(exp_df[v_col_exp], exp_df[i_col_exp],
                   'b-', label='Experimental', linewidth=1.5)
        
        if v_col_sim and i_col_sim:
            ax.plot(sim_df[v_col_sim], sim_df[i_col_sim],
                   'r--', label='Simulated', linewidth=1.5)
        
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Current (mA)')
        ax.legend()
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info(f"Saved CV comparison plot to {save_path}")
        
        return fig
    
    def generate_report_figure(self,
                              validation_results: Dict[str, Any],
                              save_path: str) -> Any:
        """
        生成综合报告图
        
        Args:
            validation_results: 验证结果字典
            save_path: 保存路径
            
        Returns:
            matplotlib figure对象
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization")
        
        fig = plt.figure(figsize=(16, 12), dpi=self.config['dpi'])
        
        # 创建子图布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 这里可以根据validation_results的内容添加各种子图
        # 这是一个示例框架
        
        ax_text = fig.add_subplot(gs[0, :])
        ax_text.axis('off')
        
        # 添加验证结果摘要文本
        summary_text = "Validation Report\n" + "="*50 + "\n\n"
        if 'errors' in validation_results:
            errors = validation_results['errors']
            summary_text += f"MAE: {errors.get('mae', 'N/A'):.4f}\n"
            summary_text += f"RMSE: {errors.get('rmse', 'N/A'):.4f}\n"
            summary_text += f"R²: {errors.get('r2', 'N/A'):.4f}\n"
        
        ax_text.text(0.1, 0.5, summary_text, transform=ax_text.transAxes,
                    fontsize=12, verticalalignment='center', fontfamily='monospace')
        
        plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        logger.info(f"Saved report figure to {save_path}")
        
        return fig
