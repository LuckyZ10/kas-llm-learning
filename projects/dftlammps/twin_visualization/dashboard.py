"""
实时监控仪表板 (Real-Time Monitoring Dashboard)

提供数字孪生系统的实时监控和可视化界面。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Protocol, Set,
    Tuple, TypeVar, Union, Iterator
)
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


# 尝试导入可选依赖
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from flask import Flask, jsonify, render_template_string
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


try:
    from ..digital_twin.twin_core import StateVector, Observation, DigitalTwinCore, TwinState
    from ..digital_twin.predictive_model import HealthIndicator, RULPrediction
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from digital_twin.twin_core import StateVector, Observation, DigitalTwinCore, TwinState
    from digital_twin.predictive_model import HealthIndicator, RULPrediction


@dataclass
class DashboardConfig:
    """仪表板配置"""
    refresh_rate_hz: float = 10.0
    max_history_points: int = 1000
    enable_web_interface: bool = True
    web_port: int = 5000
    plot_style: str = "dark"  # 'dark' or 'light'
    show_predictions: bool = True
    show_uncertainty: bool = True


@dataclass
class MetricValue:
    """指标值"""
    name: str
    value: float
    unit: str
    timestamp: float
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    trend: str = "stable"  # 'up', 'down', 'stable'
    
    @property
    def status(self) -> str:
        """根据阈值确定状态"""
        if self.threshold_critical is not None and self.value >= self.threshold_critical:
            return 'critical'
        if self.threshold_warning is not None and self.value >= self.threshold_warning:
            return 'warning'
        return 'normal'


class TimeSeriesBuffer:
    """
    时间序列数据缓冲区
    
    高效存储和管理时序数据
    """
    
    def __init__(self, max_size: int = 1000, n_channels: int = 1):
        self.max_size = max_size
        self.n_channels = n_channels
        self.timestamps: deque[float] = deque(maxlen=max_size)
        self.data: deque[NDArray[np.float64]] = deque(maxlen=max_size)
        
    def add(self, timestamp: float, values: NDArray[np.float64]) -> None:
        """添加数据点"""
        self.timestamps.append(timestamp)
        self.data.append(values)
    
    def get_data(self, n_points: Optional[int] = None) -> Tuple[List[float], NDArray[np.float64]]:
        """获取数据"""
        timestamps = list(self.timestamps)
        data = np.array(list(self.data))
        
        if n_points is not None and len(timestamps) > n_points:
            timestamps = timestamps[-n_points:]
            data = data[-n_points:]
        
        return timestamps, data
    
    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        if len(self.data) == 0:
            return {}
        
        data_array = np.array(list(self.data))
        
        return {
            'mean': float(np.mean(data_array)),
            'std': float(np.std(data_array)),
            'min': float(np.min(data_array)),
            'max': float(np.max(data_array)),
            'last': float(data_array[-1])
        }


class BaseRenderer(ABC):
    """渲染器基类"""
    
    @abstractmethod
    def render(self, data: Dict[str, Any]) -> Any:
        """渲染数据"""
        pass


class MatplotlibRenderer(BaseRenderer):
    """
    Matplotlib渲染器
    
    生成静态图表
    """
    
    def __init__(self, style: str = "dark"):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib not available")
        
        self.style = style
        self.fig_size = (12, 8)
        self.dpi = 100
        
        if style == "dark":
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
    
    def render(self, data: Dict[str, Any]) -> Figure:
        """渲染仪表板"""
        fig = plt.figure(figsize=self.fig_size, dpi=self.dpi)
        
        # 创建子图网格
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 状态时间序列 (左上)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_state_timeseries(ax1, data.get('state_history', []))
        
        # 2. 健康指标 (右上)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_health_gauge(ax2, data.get('health', 1.0))
        
        # 3. RUL预测 (中左)
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_rul(ax3, data.get('rul', None))
        
        # 4. 关键指标 (中右)
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_metrics(ax4, data.get('metrics', []))
        
        # 5. 系统状态 (底部)
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_system_status(ax5, data.get('status', {}))
        
        plt.tight_layout()
        return fig
    
    def _plot_state_timeseries(self, ax, state_history: List[StateVector]) -> None:
        """绘制状态时间序列"""
        if len(state_history) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            return
        
        timestamps = [s.timestamp for s in state_history]
        
        # 绘制每个维度
        n_dims = len(state_history[0].data)
        for i in range(min(n_dims, 5)):  # 最多5个维度
            values = [s.data[i] for s in state_history]
            ax.plot(timestamps, values, label=f'Dim {i}', alpha=0.8)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('State Evolution')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_health_gauge(self, ax, health: float) -> None:
        """绘制健康度仪表盘"""
        # 简化的健康度指示器
        theta = np.linspace(0, np.pi, 100)
        r = 1.0
        
        # 背景弧
        ax.fill_between(np.cos(theta), np.sin(theta), 0, alpha=0.1, color='gray')
        
        # 健康度指示
        health_theta = np.pi * (1 - health)
        ax.arrow(0, 0, 0.8 * np.cos(health_theta), 0.8 * np.sin(health_theta),
                head_width=0.1, head_length=0.05, fc='green' if health > 0.7 else 'orange' if health > 0.4 else 'red',
                ec='white', linewidth=2)
        
        # 标签
        ax.text(0, -0.3, f'{health:.1%}', ha='center', va='center', fontsize=20, fontweight='bold')
        ax.text(0, -0.6, 'Health', ha='center', va='center', fontsize=12)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.8, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('System Health', fontsize=12)
    
    def _plot_rul(self, ax, rul: Optional[RULPrediction]) -> None:
        """绘制RUL预测"""
        if rul is None or len(rul.degradation_trend) == 0:
            ax.text(0.5, 0.5, 'No RUL Data', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 绘制历史趋势
        trend = rul.degradation_trend
        x_hist = np.arange(len(trend))
        ax.plot(x_hist, trend, 'b-', label='Historical', linewidth=2)
        
        # 预测部分
        current = trend[-1]
        rul_cycles = rul.rul_cycles
        lower, upper = rul.confidence_interval
        
        x_future = np.arange(len(trend), len(trend) + int(rul_cycles))
        
        # 简化的线性预测
        slope = (current - trend[0]) / len(trend) if len(trend) > 1 else 0
        future_trend = current + slope * np.arange(len(x_future))
        
        ax.plot(x_future, future_trend, 'r--', label='Predicted', linewidth=2)
        ax.axhline(y=0.2, color='red', linestyle=':', alpha=0.5, label='Failure Threshold')
        
        # 置信区间
        ax.fill_between(x_future, future_trend * 0.8, future_trend * 1.2, alpha=0.2, color='red')
        
        ax.set_xlabel('Cycles')
        ax.set_ylabel('Health Index')
        ax.set_title(f'RUL: {rul_cycles:.0f} cycles')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_metrics(self, ax, metrics: List[MetricValue]) -> None:
        """绘制关键指标"""
        if len(metrics) == 0:
            ax.text(0.5, 0.5, 'No Metrics', ha='center', va='center', transform=ax.transAxes)
            return
        
        names = [m.name for m in metrics]
        values = [m.value for m in metrics]
        colors = ['red' if m.status == 'critical' else 'orange' if m.status == 'warning' else 'green' 
                 for m in metrics]
        
        bars = ax.barh(names, values, color=colors, alpha=0.7)
        ax.set_xlabel('Value')
        ax.set_title('Key Metrics')
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.2f}', 
                   va='center', fontsize=9)
    
    def _plot_system_status(self, ax, status: Dict[str, Any]) -> None:
        """绘制系统状态"""
        ax.axis('off')
        
        status_text = f"""
        System Status: {status.get('state', 'Unknown')}
        Uptime: {status.get('uptime', 0):.1f}s
        Total Steps: {status.get('total_steps', 0)}
        Total Observations: {status.get('total_observations', 0)}
        Sync Success Rate: {status.get('sync_success_rate', 0):.1%}
        """
        
        ax.text(0.1, 0.5, status_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='center', fontfamily='monospace')
    
    def save(self, fig: Figure, filepath: str) -> None:
        """保存图表"""
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        print(f"Dashboard saved to {filepath}")


class PlotlyRenderer(BaseRenderer):
    """
    Plotly渲染器
    
    生成交互式图表
    """
    
    def __init__(self):
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly not available")
    
    def render(self, data: Dict[str, Any]) -> go.Figure:
        """渲染交互式仪表板"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('State Evolution', 'Health', 'RUL Prediction',
                          'Metrics', 'Uncertainty', 'System Status'),
            specs=[[{"colspan": 2}, None, {"type": "indicator"}],
                   [{"colspan": 2}, None, {"type": "bar"}],
                   [{"colspan": 2}, None, {"type": "table"}]]
        )
        
        # 状态时间序列
        state_history = data.get('state_history', [])
        if len(state_history) > 0:
            timestamps = [s.timestamp for s in state_history]
            n_dims = len(state_history[0].data)
            
            for i in range(min(n_dims, 3)):
                values = [s.data[i] for s in state_history]
                fig.add_trace(
                    go.Scatter(x=timestamps, y=values, name=f'Dim {i}'),
                    row=1, col=1
                )
        
        # 健康度指示器
        health = data.get('health', 1.0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=health * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [0, 100]},
                      'bar': {'color': "darkgreen" if health > 0.7 else "orange" if health > 0.4 else "red"},
                      'steps': [
                          {'range': [0, 40], 'color': "red"},
                          {'range': [40, 70], 'color': "orange"},
                          {'range': [70, 100], 'color': "lightgreen"}]}
            ),
            row=1, col=3
        )
        
        fig.update_layout(height=800, showlegend=True)
        return fig


class Dashboard:
    """
    实时监控仪表板
    
    整合数据收集、处理和可视化
    """
    
    def __init__(self, twin: DigitalTwinCore, config: Optional[DashboardConfig] = None):
        self.twin = twin
        self.config = config or DashboardConfig()
        
        # 数据缓冲区
        self.state_buffer = TimeSeriesBuffer(max_size=self.config.max_history_points)
        self.health_buffer = TimeSeriesBuffer(max_size=self.config.max_history_points, n_channels=2)
        self.metrics_buffer: Dict[str, TimeSeriesBuffer] = {}
        
        # 渲染器
        self.renderers: Dict[str, BaseRenderer] = {}
        if MATPLOTLIB_AVAILABLE:
            self.renderers['matplotlib'] = MatplotlibRenderer(self.config.plot_style)
        if PLOTLY_AVAILABLE:
            self.renderers['plotly'] = PlotlyRenderer()
        
        # 状态
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        self._current_health = 1.0
        self._current_rul: Optional[RULPrediction] = None
        
    def register_metric(self, name: str, n_channels: int = 1) -> None:
        """注册指标"""
        self.metrics_buffer[name] = TimeSeriesBuffer(
            max_size=self.config.max_history_points,
            n_channels=n_channels
        )
    
    def update(self) -> None:
        """更新仪表板数据"""
        # 获取当前状态
        state = self.twin.get_current_state()
        if state:
            self.state_buffer.add(state.timestamp, state.data)
        
        # 获取统计信息
        stats = self.twin.get_statistics()
        
        # 更新指标
        for name, buffer in self.metrics_buffer.items():
            if name in stats:
                buffer.add(time.time(), np.array([stats[name]]))
    
    def start(self) -> None:
        """启动仪表板"""
        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop)
        self._update_thread.daemon = True
        self._update_thread.start()
        
        print(f"Dashboard started (refresh rate: {self.config.refresh_rate_hz} Hz)")
    
    def stop(self) -> None:
        """停止仪表板"""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=1.0)
        print("Dashboard stopped")
    
    def _update_loop(self) -> None:
        """更新循环"""
        interval = 1.0 / self.config.refresh_rate_hz
        
        while self._running:
            self.update()
            time.sleep(interval)
    
    def get_data(self) -> Dict[str, Any]:
        """获取当前数据"""
        stats = self.twin.get_statistics()
        
        return {
            'state_history': self.twin.get_state_history(n=200),
            'health': self._current_health,
            'rul': self._current_rul,
            'metrics': self._get_current_metrics(),
            'status': stats
        }
    
    def _get_current_metrics(self) -> List[MetricValue]:
        """获取当前指标"""
        metrics = []
        
        for name, buffer in self.metrics_buffer.items():
            stats = buffer.get_statistics()
            if stats:
                metrics.append(MetricValue(
                    name=name,
                    value=stats['last'],
                    unit='',
                    timestamp=time.time()
                ))
        
        return metrics
    
    def render(self, renderer: str = "matplotlib") -> Any:
        """渲染仪表板"""
        if renderer not in self.renderers:
            raise ValueError(f"Renderer {renderer} not available")
        
        data = self.get_data()
        return self.renderers[renderer].render(data)
    
    def save_snapshot(self, filepath: str, renderer: str = "matplotlib") -> None:
        """保存快照"""
        if renderer == "matplotlib" and MATPLOTLIB_AVAILABLE:
            fig = self.render(renderer)
            self.renderers[renderer].save(fig, filepath)
        else:
            # 保存JSON数据
            data = self.get_data()
            with open(filepath, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'status': data['status']
                }, f, indent=2, default=str)
    
    def set_health(self, health: float) -> None:
        """设置健康度"""
        self._current_health = health
        self.health_buffer.add(time.time(), np.array([health, 1.0]))
    
    def set_rul(self, rul: RULPrediction) -> None:
        """设置RUL预测"""
        self._current_rul = rul


class WebDashboard:
    """
    Web仪表板
    
    基于Flask的Web界面
    """
    
    def __init__(self, dashboard: Dashboard, port: int = 5000):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask not available")
        
        self.dashboard = dashboard
        self.port = port
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """设置路由"""
        
        @self.app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)
        
        @self.app.route('/api/data')
        def api_data():
            return jsonify(self.dashboard.get_data())
        
        @self.app.route('/api/metrics')
        def api_metrics():
            return jsonify([asdict(m) for m in self.dashboard._get_current_metrics()])
    
    def run(self, debug: bool = False) -> None:
        """启动Web服务器"""
        self.app.run(host='0.0.0.0', port=self.port, debug=debug)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Digital Twin Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a2e; color: #eee; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
        .card { background: #16213e; padding: 20px; border-radius: 10px; }
        .card h3 { margin-top: 0; color: #e94560; }
        .metric { font-size: 2em; font-weight: bold; }
        .status-normal { color: #4ecca3; }
        .status-warning { color: #ffc107; }
        .status-critical { color: #e94560; }
        .refresh-btn { background: #e94560; color: white; border: none; padding: 10px 20px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 数字孪生实时监控仪表板</h1>
            <p>Digital Twin Real-Time Monitoring</p>
            <button class="refresh-btn" onclick="location.reload()">刷新</button>
        </div>
        <div class="grid">
            <div class="card">
                <h3>系统状态</h3>
                <div id="system-status">加载中...</div>
            </div>
            <div class="card">
                <h3>健康度</h3>
                <div id="health" class="metric">--</div>
            </div>
            <div class="card">
                <h3>运行时间</h3>
                <div id="uptime" class="metric">--</div>
            </div>
        </div>
    </div>
    <script>
        async function fetchData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                updateUI(data);
            } catch (e) {
                console.error('Failed to fetch data:', e);
            }
        }
        
        function updateUI(data) {
            document.getElementById('health').textContent = 
                (data.health * 100).toFixed(1) + '%';
            document.getElementById('uptime').textContent = 
                (data.status.uptime / 60).toFixed(1) + ' min';
            document.getElementById('system-status').textContent = 
                data.status.state;
        }
        
        fetchData();
        setInterval(fetchData, 1000);
    </script>
</body>
</html>
"""


def demo():
    """演示仪表板功能"""
    print("=" * 60)
    print("实时监控仪表板演示")
    print("=" * 60)
    
    try:
        from twin_core import DigitalTwinCore, TwinConfiguration, StateVector
    except ImportError:
        print("Error: twin_core module not found")
        return
    
    # 创建数字孪生
    config = TwinConfiguration()
    twin = DigitalTwinCore(config)
    
    # 初始化
    initial_state = StateVector(
        timestamp=0.0,
        data=np.random.randn(5) * 0.1,
        metadata={'init': True}
    )
    twin.initialize(initial_state)
    
    # 创建仪表板
    dashboard_config = DashboardConfig(
        refresh_rate_hz=5.0,
        max_history_points=500
    )
    dashboard = Dashboard(twin, dashboard_config)
    
    # 注册指标
    dashboard.register_metric('total_steps')
    dashboard.register_metric('mean_error')
    
    print("\n1. 启动仪表板")
    dashboard.start()
    
    # 模拟运行
    print("\n2. 模拟数据收集...")
    for i in range(50):
        state = StateVector(
            timestamp=i * 0.1,
            data=np.random.randn(5) * 0.1 + np.sin(i * 0.1),
            metadata={'step': i}
        )
        twin._current_state = state
        twin._state_history.append(state)
        twin._stats['total_steps'] = i
        twin._stats['mean_error'] = np.random.rand() * 0.05
        
        dashboard.set_health(1.0 - i * 0.01)
        time.sleep(0.05)
    
    print("\n3. 生成可视化")
    
    if MATPLOTLIB_AVAILABLE:
        print("   使用Matplotlib渲染...")
        fig = dashboard.render('matplotlib')
        
        # 保存
        dashboard.save_snapshot('/tmp/dashboard_demo.png')
        print("   已保存到 /tmp/dashboard_demo.png")
    else:
        print("   Matplotlib不可用，跳过渲染")
    
    # 停止
    dashboard.stop()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    
    return dashboard


if __name__ == "__main__":
    demo()
