#!/usr/bin/env python3
"""
可视化工具模块 - RL材料优化的可视化

包含:
- 优化过程可视化
- 结构可视化
- 奖励曲线绘制
- 帕累托前沿可视化
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class OptimizationPlotter:
    """优化过程绘图器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.history = []
    
    def add_point(self, step: int, reward: float, info: Optional[Dict] = None):
        """添加数据点"""
        self.history.append({
            'step': step,
            'reward': reward,
            'info': info or {}
        })
    
    def plot_reward_curve(self, save_path: Optional[str] = None):
        """绘制奖励曲线"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return
        
        if not self.history:
            return
        
        steps = [h['step'] for h in self.history]
        rewards = [h['reward'] for h in self.history]
        
        fig, axes = plt.subplots(2, 1, figsize=self.figsize)
        
        # 原始奖励
        axes[0].plot(steps, rewards, 'b-', alpha=0.7)
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Reward Curve')
        axes[0].grid(True)
        
        # 移动平均
        window = min(10, len(rewards) // 2)
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0].plot(steps[window-1:], moving_avg, 'r-', linewidth=2, label='Moving Average')
            axes[0].legend()
        
        # 奖励分布
        axes[1].hist(rewards, bins=30, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Reward')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Reward Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    def plot_multi_objective_front(
        self,
        pareto_front,
        objective_names: List[str] = None,
        save_path: Optional[str] = None
    ):
        """绘制多目标帕累托前沿"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return
        
        if pareto_front is None or len(pareto_front) == 0:
            return
        
        objectives = pareto_front.get_objectives()
        n_objectives = objectives.shape[1]
        
        if objective_names is None:
            objective_names = [f'Objective {i+1}' for i in range(n_objectives)]
        
        if n_objectives == 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(objectives[:, 0], objectives[:, 1], c='blue', s=50, alpha=0.6)
            ax.set_xlabel(objective_names[0])
            ax.set_ylabel(objective_names[1])
            ax.set_title('Pareto Front')
            ax.grid(True)
        
        elif n_objectives == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                      c='blue', s=50, alpha=0.6)
            ax.set_xlabel(objective_names[0])
            ax.set_ylabel(objective_names[1])
            ax.set_zlabel(objective_names[2])
            ax.set_title('Pareto Front (3D)')
        
        else:
            # 配对图
            fig, axes = plt.subplots(n_objectives, n_objectives, figsize=(12, 12))
            
            for i in range(n_objectives):
                for j in range(n_objectives):
                    if i == j:
                        axes[i, j].hist(objectives[:, i], bins=20, alpha=0.7)
                        axes[i, j].set_title(objective_names[i])
                    else:
                        axes[i, j].scatter(objectives[:, j], objectives[:, i],
                                         c='blue', s=20, alpha=0.5)
                        axes[i, j].set_xlabel(objective_names[j])
                        axes[i, j].set_ylabel(objective_names[i])
            
            plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    def plot_training_stats(self, stats: Dict[str, List[float]], save_path: Optional[str] = None):
        """绘制训练统计"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return
        
        n_plots = len(stats)
        if n_plots == 0:
            return
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        for ax, (name, values) in zip(axes, stats.items()):
            ax.plot(values, linewidth=2)
            ax.set_xlabel('Episode')
            ax.set_ylabel(name)
            ax.set_title(name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()


class StructureVisualizer:
    """结构可视化器"""
    
    def __init__(self):
        pass
    
    def visualize_crystal(
        self,
        structure: Any,
        save_path: Optional[str] = None,
        show_bonds: bool = True,
        highlight_atoms: Optional[List[int]] = None
    ):
        """可视化晶体结构"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            logger.warning("matplotlib not available")
            return
        
        if not hasattr(structure, 'positions') or not hasattr(structure, 'elements'):
            return
        
        positions = structure.positions @ structure.lattice  # 转换为笛卡尔坐标
        elements = structure.elements
        
        # 元素颜色映射
        element_colors = {
            'Li': '#FF6B6B', 'Na': '#FF8787', 'K': '#FFA8A8',
            'Mg': '#4ECDC4', 'Ca': '#45B7B8',
            'Al': '#96CEB4', 'Si': '#FFEAA7',
            'O': '#FF7675', 'S': '#FDCB6E', 'F': '#74B9FF', 'Cl': '#A29BFE',
            'Fe': '#636E72', 'Co': '#B2BEC3', 'Ni': '#DFE6E9',
            'Cu': '#D63031', 'Zn': '#B2BEC3',
            'Ti': '#E17055', 'V': '#D63031', 'Cr': '#6C5CE7',
            'Bi': '#00B894', 'Sb': '#00CEC9'
        }
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制原子
        for i, (pos, elem) in enumerate(zip(positions, elements)):
            color = element_colors.get(elem, '#808080')
            size = 100 if highlight_atoms and i in highlight_atoms else 50
            alpha = 1.0 if highlight_atoms and i in highlight_atoms else 0.7
            
            ax.scatter(*pos, c=color, s=size, alpha=alpha, edgecolors='black', linewidth=0.5)
            
            # 标签
            if highlight_atoms and i in highlight_atoms:
                ax.text(pos[0], pos[1], pos[2], elem, fontsize=8)
        
        # 绘制晶胞边界
        if hasattr(structure, 'lattice'):
            self._draw_unit_cell(ax, structure.lattice)
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title('Crystal Structure')
        
        # 添加图例
        unique_elements = list(set(elements))
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=element_colors.get(e, '#808080'),
                      markersize=10, label=e)
            for e in unique_elements
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Structure saved to {save_path}")
        
        plt.close()
    
    def _draw_unit_cell(self, ax, lattice: np.ndarray):
        """绘制晶胞边界"""
        # 晶胞顶点
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ]) @ lattice
        
        # 边
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 垂直边
        ]
        
        for edge in edges:
            points = vertices[edge]
            ax.plot3D(*points.T, 'k-', alpha=0.3, linewidth=1)
    
    def plot_radial_distribution(
        self,
        structure: Any,
        bins: int = 50,
        r_max: float = 10.0,
        save_path: Optional[str] = None
    ):
        """绘制径向分布函数"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return
        
        if not hasattr(structure, 'positions'):
            return
        
        positions = structure.positions @ structure.lattice
        
        # 计算所有距离
        from scipy.spatial.distance import pdist
        distances = pdist(positions)
        
        # 创建直方图
        hist, bin_edges = np.histogram(distances, bins=bins, range=(0, r_max))
        
        # 归一化
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        shell_volumes = 4/3 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
        rdf = hist / shell_volumes
        rdf = rdf / rdf.max()  # 归一化到1
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(bin_centers, rdf, linewidth=2)
        ax.set_xlabel('r (Å)')
        ax.set_ylabel('g(r)')
        ax.set_title('Radial Distribution Function')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"RDF plot saved to {save_path}")
        
        plt.close()


class RewardVisualizer:
    """奖励函数可视化器"""
    
    def __init__(self):
        pass
    
    def plot_reward_components(
        self,
        rewards: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        """绘制奖励组成"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return
        
        fig, axes = plt.subplots(len(rewards), 1, figsize=(12, 3 * len(rewards)))
        
        if len(rewards) == 1:
            axes = [axes]
        
        for ax, (name, values) in zip(axes, rewards.items()):
            ax.plot(values, linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel('Reward')
            ax.set_title(f'{name} Reward')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Reward components saved to {save_path}")
        
        plt.close()
    
    def plot_reward_landscape_2d(
        self,
        reward_fn: callable,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        n_points: int = 50,
        save_path: Optional[str] = None
    ):
        """绘制2D奖励景观"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return
        
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(n_points):
            for j in range(n_points):
                Z[i, j] = reward_fn(np.array([X[i, j], Y[i, j]]))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(contour, ax=ax, label='Reward')
        
        ax.set_xlabel('Parameter 1')
        ax.set_ylabel('Parameter 2')
        ax.set_title('Reward Landscape')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Reward landscape saved to {save_path}")
        
        plt.close()


class Dashboard:
    """优化仪表板"""
    
    def __init__(self):
        self.plotter = OptimizationPlotter()
        self.structure_viz = StructureVisualizer()
        self.reward_viz = RewardVisualizer()
    
    def create_summary_report(
        self,
        optimizer: Any,
        save_dir: str = './rl_optimizer_reports'
    ):
        """创建总结报告"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 绘制奖励曲线
        self.plotter.plot_reward_curve(f"{save_dir}/reward_curve.png")
        
        # 绘制结构
        if hasattr(optimizer, 'env') and hasattr(optimizer.env, 'get_structure'):
            structure = optimizer.env.get_structure()
            self.structure_viz.visualize_crystal(
                structure,
                save_path=f"{save_dir}/best_structure.png"
            )
        
        logger.info(f"Summary report created in {save_dir}")
