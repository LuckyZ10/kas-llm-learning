#!/usr/bin/env python3
"""
DFT+LAMMPS Workflow Monitoring Dashboard
=========================================
交互式监控仪表盘，用于可视化DFT+LAMMPS工作流的各个方面：
- ML训练实时监控
- MD轨迹可视化
- 高通量筛选结果展示
- 主动学习进度追踪

作者: Dashboard Expert
日期: 2025-03-09
"""

import os
import sys
import json
import yaml
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import deque
import logging

import numpy as np
import pandas as pd

# Dash imports
import dash
from dash import dcc, html, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# For 3D visualization
import plotly.figure_factory as ff

# For file watching
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logging.warning("watchdog not available. File auto-refresh disabled.")

# For PDF export
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# ASE for structure reading
try:
    from ase.io import read
    from ase import Atoms
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    logging.warning("ASE not available. Structure visualization limited.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DashboardConfig:
    """仪表盘配置"""
    # 数据路径
    work_dir: str = "./battery_screening"
    dft_results_path: str = "./battery_screening/dft_results"
    md_results_path: str = "./battery_screening/md_results"
    models_path: str = "./battery_screening/models"
    al_workflow_path: str = "./active_learning_workflow"
    screening_db_path: str = "./screening_db"
    
    # 自动刷新设置
    auto_refresh: bool = True
    refresh_interval: int = 5  # 秒
    
    # 显示设置
    max_points: int = 10000
    time_window: int = 3600  # 显示最近1小时的数据
    
    # 阈值设置
    force_error_threshold: float = 0.05  # eV/Å
    energy_error_threshold: float = 0.001  # eV/atom
    
    # 导出设置
    export_dir: str = "./dashboard_exports"
    
    # 服务器设置
    host: str = "0.0.0.0"
    port: int = 8050
    debug: bool = False


def load_config(config_path: str = "dashboard_config.yaml") -> DashboardConfig:
    """从YAML文件加载配置"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return DashboardConfig(**config_dict)
    return DashboardConfig()


# =============================================================================
# Data Manager
# =============================================================================

class DataManager:
    """数据管理器 - 负责读取和缓存各类数据"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.cache = {}
        self.cache_time = {}
        self.cache_ttl = 5  # 缓存5秒
        
        # 实时数据缓冲区
        self.training_buffer = deque(maxlen=10000)
        self.md_buffer = deque(maxlen=10000)
        
    def _get_cached(self, key: str, loader: callable) -> Any:
        """获取缓存数据或重新加载"""
        now = time.time()
        if key in self.cache and now - self.cache_time.get(key, 0) < self.cache_ttl:
            return self.cache[key]
        
        try:
            data = loader()
            self.cache[key] = data
            self.cache_time[key] = now
            return data
        except Exception as e:
            logger.error(f"Failed to load {key}: {e}")
            return self.cache.get(key, None)
    
    # -------------------------------------------------------------------------
    # ML Training Data
    # -------------------------------------------------------------------------
    def get_training_data(self) -> pd.DataFrame:
        """读取DeePMD训练日志 (lcurve.out)"""
        def load():
            lcurve_path = Path(self.config.models_path) / "lcurve.out"
            if not lcurve_path.exists():
                # 搜索子目录
                for subdir in Path(self.config.models_path).rglob("lcurve.out"):
                    lcurve_path = subdir
                    break
            
            if lcurve_path.exists():
                # lcurve.out格式: batch, lr, loss, energy_rmse, energy_rmse_traj, 
                #                  force_rmse, force_rmse_traj, virial_rmse, virial_rmse_traj
                df = pd.read_csv(lcurve_path, sep=r'\s+', comment='#', header=None)
                if len(df.columns) >= 7:
                    df.columns = ['batch', 'lr', 'loss', 'energy_rmse', 'energy_rmse_traj',
                                 'force_rmse', 'force_rmse_traj', 'virial_rmse', 'virial_rmse_traj'][:len(df.columns)]
                return df
            return pd.DataFrame()
        
        return self._get_cached('training', load)
    
    def get_model_test_results(self) -> pd.DataFrame:
        """读取模型测试结果"""
        def load():
            results = []
            test_dir = Path(self.config.models_path) / "test_results"
            if test_dir.exists():
                for json_file in test_dir.glob("*.json"):
                    with open(json_file) as f:
                        results.append(json.load(f))
            return pd.DataFrame(results)
        
        return self._get_cached('model_test', load)
    
    # -------------------------------------------------------------------------
    # MD Simulation Data
    # -------------------------------------------------------------------------
    def get_md_trajectory_data(self, trajectory_file: str = None) -> pd.DataFrame:
        """读取MD轨迹的物理量数据"""
        def load():
            # 从LAMMPS log文件读取热力学数据
            log_files = list(Path(self.config.md_results_path).rglob("log.lammps"))
            if not log_files:
                return pd.DataFrame()
            
            log_path = log_files[0]
            
            # 解析LAMMPS log
            data = []
            with open(log_path) as f:
                lines = f.readlines()
            
            in_thermo = False
            headers = []
            
            for line in lines:
                if "Step" in line and "Temp" in line:
                    headers = line.strip().split()
                    in_thermo = True
                    continue
                
                if in_thermo:
                    if line.strip() == "" or "Loop" in line:
                        in_thermo = False
                        continue
                    try:
                        values = [float(x) for x in line.strip().split()]
                        if len(values) == len(headers):
                            data.append(dict(zip(headers, values)))
                    except:
                        pass
            
            return pd.DataFrame(data)
        
        return self._get_cached('md_trajectory', load)
    
    def get_md_dump_files(self) -> List[str]:
        """获取MD dump文件列表"""
        dump_files = list(Path(self.config.md_results_path).rglob("*.lammpstrj"))
        return [str(f) for f in dump_files]
    
    # -------------------------------------------------------------------------
    # Screening Data
    # -------------------------------------------------------------------------
    def get_screening_results(self) -> pd.DataFrame:
        """读取高通量筛选结果"""
        def load():
            csv_path = Path(self.config.screening_db_path) / "screening_results.csv"
            if csv_path.exists():
                return pd.read_csv(csv_path)
            
            # 搜索JSON文件
            json_files = list(Path(self.config.screening_db_path).glob("*.json"))
            if json_files:
                data = []
                for jf in json_files:
                    with open(jf) as f:
                        entry = json.load(f)
                        if 'metadata' in entry:
                            row = {'structure_id': entry.get('structure_id', '')}
                            row.update(entry['metadata'])
                            data.append(row)
                return pd.DataFrame(data)
            
            return pd.DataFrame()
        
        return self._get_cached('screening', load)
    
    # -------------------------------------------------------------------------
    # Active Learning Data
    # -------------------------------------------------------------------------
    def get_active_learning_progress(self) -> pd.DataFrame:
        """读取主动学习进度"""
        def load():
            iterations = []
            al_path = Path(self.config.al_workflow_path)
            
            if not al_path.exists():
                return pd.DataFrame()
            
            # 搜索所有迭代目录
            for iter_dir in sorted(al_path.glob("iter_*")):
                iter_num = int(iter_dir.name.split("_")[1])
                
                iter_data = {
                    'iteration': iter_num,
                    'timestamp': datetime.fromtimestamp(iter_dir.stat().st_mtime)
                }
                
                # 读取候选结构数量
                candidates_file = iter_dir / "candidates.traj"
                if candidates_file.exists():
                    iter_data['n_candidates'] = candidates_file.stat().st_size // 1000  # 估算
                
                # 读取统计数据
                stats_file = iter_dir / "exploration_stats.json"
                if stats_file.exists():
                    with open(stats_file) as f:
                        stats = json.load(f)
                        iter_data.update(stats)
                
                iterations.append(iter_data)
            
            return pd.DataFrame(iterations)
        
        return self._get_cached('al_progress', load)
    
    def get_model_deviation_history(self) -> pd.DataFrame:
        """读取模型偏差历史"""
        def load():
            deviations = []
            al_path = Path(self.config.al_workflow_path)
            
            for iter_dir in sorted(al_path.glob("iter_*")):
                iter_num = int(iter_dir.name.split("_")[1])
                
                # 读取偏差数据
                devi_file = iter_dir / "model_deviations.json"
                if devi_file.exists():
                    with open(devi_file) as f:
                        data = json.load(f)
                        for item in data:
                            item['iteration'] = iter_num
                            deviations.append(item)
            
            return pd.DataFrame(deviations)
        
        return self._get_cached('model_deviation', load)


# =============================================================================
# Layout Components
# =============================================================================

def create_header() -> html.Div:
    """创建页面头部"""
    return html.Div([
        dbc.Navbar([
            dbc.Container([
                html.A(
                    dbc.Row([
                        dbc.Col(html.Img(src="/assets/icon.png", height="40px"), width="auto"),
                        dbc.Col(dbc.NavbarBrand("DFT+LAMMPS Workflow Dashboard", className="ms-2")),
                    ], align="center", className="g-0"),
                    href="#",
                    style={"textDecoration": "none"},
                ),
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("ML Training", href="#ml-training")),
                    dbc.NavItem(dbc.NavLink("MD Simulation", href="#md-simulation")),
                    dbc.NavItem(dbc.NavLink("Screening", href="#screening")),
                    dbc.NavItem(dbc.NavLink("Active Learning", href="#active-learning")),
                ], className="ms-auto", navbar=True),
                dbc.Badge("Live", color="success", className="p-2", id="live-indicator"),
            ])
        ], color="dark", dark=True, className="mb-4"),
    ])


def create_ml_training_tab() -> dbc.Tab:
    """创建ML训练监控标签页"""
    return dbc.Tab(label="ML Training", children=[
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Training Progress", className="mt-3"),
                    html.P("Real-time monitoring of DeePMD training metrics"),
                ], width=12),
            ]),
            
            # KPI Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Current Loss", className="card-title"),
                            html.H2(id="current-loss", children="--"),
                            html.Small(id="loss-trend", children="--"),
                        ])
                    ], color="info", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Force RMSE", className="card-title"),
                            html.H2(id="current-force-rmse", children="--"),
                            html.Small("eV/Å", className="text-muted"),
                        ])
                    ], color="info", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Energy RMSE", className="card-title"),
                            html.H2(id="current-energy-rmse", children="--"),
                            html.Small("eV/atom", className="text-muted"),
                        ])
                    ], color="info", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Training Steps", className="card-title"),
                            html.H2(id="training-steps", children="--"),
                            html.Small(id="training-progress", children="--"),
                        ])
                    ], color="info", outline=True)
                ], width=3),
            ], className="mb-4"),
            
            # Charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Loss Curves"),
                        dbc.CardBody([
                            dcc.Graph(id="loss-curve-chart", style={"height": "400px"}),
                            dcc.RadioItems(
                                id="loss-scale",
                                options=[
                                    {"label": "Linear", "value": "linear"},
                                    {"label": "Log", "value": "log"},
                                ],
                                value="log",
                                inline=True,
                                className="mt-2"
                            ),
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("RMSE Evolution"),
                        dbc.CardBody([
                            dcc.Graph(id="rmse-chart", style={"height": "400px"}),
                        ])
                    ])
                ], width=6),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Learning Rate Schedule"),
                        dbc.CardBody([
                            dcc.Graph(id="lr-chart", style={"height": "300px"}),
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Model Test Results"),
                        dbc.CardBody([
                            dcc.Graph(id="test-results-chart", style={"height": "300px"}),
                        ])
                    ])
                ], width=6),
            ]),
            
        ], fluid=True)
    ])


def create_md_simulation_tab() -> dbc.Tab:
    """创建MD模拟监控标签页"""
    return dbc.Tab(label="MD Simulation", children=[
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Molecular Dynamics", className="mt-3"),
                    html.P("Monitor MD simulations and analyze trajectories"),
                ], width=12),
            ]),
            
            # File selector
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Label("Select Trajectory File:"),
                            dcc.Dropdown(
                                id="md-file-selector",
                                options=[],
                                placeholder="Select a dump file..."
                            ),
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Label("Animation Settings:"),
                            dbc.ButtonGroup([
                                dbc.Button("▶ Play", id="play-btn", color="success", size="sm"),
                                dbc.Button("⏸ Pause", id="pause-btn", color="warning", size="sm"),
                                dbc.Button("⏮ Reset", id="reset-btn", color="secondary", size="sm"),
                            ]),
                        ])
                    ])
                ], width=6),
            ], className="mb-4"),
            
            # 3D Visualization
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("3D Structure Visualization"),
                        dbc.CardBody([
                            dcc.Graph(id="structure-3d", style={"height": "500px"}),
                            dcc.Slider(
                                id="frame-slider",
                                min=0,
                                max=100,
                                step=1,
                                value=0,
                                marks={0: "0", 25: "25%", 50: "50%", 75: "75%", 100: "100%"},
                            ),
                            html.Div(id="frame-info", className="text-center mt-2"),
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Temperature Profile"),
                        dbc.CardBody([
                            dcc.Graph(id="temperature-chart", style={"height": "240px"}),
                        ])
                    ], className="mb-3"),
                    dbc.Card([
                        dbc.CardHeader("Energy Profile"),
                        dbc.CardBody([
                            dcc.Graph(id="energy-chart", style={"height": "240px"}),
                        ])
                    ]),
                ], width=6),
            ], className="mb-4"),
            
            # Additional MD metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Pressure & Volume"),
                        dbc.CardBody([
                            dcc.Graph(id="pressure-volume-chart", style={"height": "300px"}),
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Diffusion Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id="msd-chart", style={"height": "300px"}),
                            html.P("Select atom type for MSD calculation:"),
                            dcc.Dropdown(
                                id="atom-type-selector",
                                options=[
                                    {"label": "All Atoms", "value": "all"},
                                    {"label": "Li", "value": "Li"},
                                    {"label": "O", "value": "O"},
                                ],
                                value="all"
                            ),
                        ])
                    ])
                ], width=6),
            ]),
            
        ], fluid=True)
    ])


def create_screening_tab() -> dbc.Tab:
    """创建高通量筛选标签页"""
    return dbc.Tab(label="High-Throughput Screening", children=[
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Screening Results", className="mt-3"),
                    html.P("High-throughput material screening and analysis"),
                ], width=12),
            ]),
            
            # Filters
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Filters"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Property Range:"),
                                    dcc.Dropdown(
                                        id="screening-property",
                                        options=[
                                            {"label": "Ionic Conductivity", "value": "ionic_conductivity"},
                                            {"label": "Formation Energy", "value": "formation_energy"},
                                            {"label": "Band Gap", "value": "band_gap"},
                                            {"label": "Bulk Modulus", "value": "bulk_modulus"},
                                        ],
                                        value="ionic_conductivity"
                                    ),
                                ], width=4),
                                dbc.Col([
                                    html.Label("Min Value:"),
                                    dcc.Input(id="min-value", type="number", placeholder="Min", className="form-control"),
                                ], width=3),
                                dbc.Col([
                                    html.Label("Max Value:"),
                                    dcc.Input(id="max-value", type="number", placeholder="Max", className="form-control"),
                                ], width=3),
                                dbc.Col([
                                    html.Label("Action:"),
                                    dbc.Button("Apply", id="apply-filters", color="primary", className="w-100"),
                                ], width=2),
                            ]),
                        ])
                    ])
                ], width=12),
            ], className="mb-4"),
            
            # Scatter plot and table
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Property Scatter Plot"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id="x-axis",
                                options=[],
                                placeholder="X-axis",
                                className="mb-2"
                            ),
                            dcc.Dropdown(
                                id="y-axis",
                                options=[],
                                placeholder="Y-axis",
                                className="mb-2"
                            ),
                            dcc.Graph(id="screening-scatter", style={"height": "500px"}),
                        ])
                    ])
                ], width=7),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Ranked Candidates"),
                        dbc.CardBody([
                            html.Div(id="screening-table-container", style={
                                "maxHeight": "500px",
                                "overflow": "auto"
                            }),
                        ])
                    ])
                ], width=5),
            ], className="mb-4"),
            
            # Statistics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Distribution Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id="distribution-chart", style={"height": "300px"}),
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Correlation Matrix"),
                        dbc.CardBody([
                            dcc.Graph(id="correlation-chart", style={"height": "300px"}),
                        ])
                    ])
                ], width=6),
            ]),
            
        ], fluid=True)
    ])


def create_active_learning_tab() -> dbc.Tab:
    """创建主动学习标签页"""
    return dbc.Tab(label="Active Learning", children=[
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Active Learning Progress", className="mt-3"),
                    html.P("Monitor active learning iterations and model improvement"),
                ], width=12),
            ]),
            
            # Progress overview
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Current Iteration", className="card-title"),
                            html.H2(id="current-iteration", children="--"),
                        ])
                    ], color="primary", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Total Candidates", className="card-title"),
                            html.H2(id="total-candidates", children="--"),
                        ])
                    ], color="success", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Convergence Status", className="card-title"),
                            html.H2(id="convergence-status", children="--"),
                        ])
                    ], color="warning", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Avg Model Deviation", className="card-title"),
                            html.H2(id="avg-deviation", children="--"),
                            html.Small("eV/Å", className="text-muted"),
                        ])
                    ], color="info", outline=True)
                ], width=3),
            ], className="mb-4"),
            
            # Progress charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Iteration Progress"),
                        dbc.CardBody([
                            dcc.Graph(id="iteration-progress-chart", style={"height": "350px"}),
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Model Deviation Distribution"),
                        dbc.CardBody([
                            dcc.Graph(id="deviation-distribution-chart", style={"height": "350px"}),
                        ])
                    ])
                ], width=6),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Error Convergence"),
                        dbc.CardBody([
                            dcc.Graph(id="error-convergence-chart", style={"height": "350px"}),
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Candidate Selection Stats"),
                        dbc.CardBody([
                            dcc.Graph(id="candidate-stats-chart", style={"height": "350px"}),
                        ])
                    ])
                ], width=6),
            ]),
            
        ], fluid=True)
    ])


def create_export_section() -> html.Div:
    """创建导出功能区"""
    return html.Div([
        html.Hr(),
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H4("Export Reports", className="mt-3"),
                ], width=12),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button("📄 Export PDF Report", id="export-pdf", color="danger", className="me-2"),
                        dbc.Button("📊 Export CSV Data", id="export-csv", color="success", className="me-2"),
                        dbc.Button("🖼 Export Charts", id="export-charts", color="info"),
                    ]),
                    html.Div(id="export-status", className="mt-2"),
                ], width=12),
            ]),
        ], fluid=True, className="mb-4"),
    ])


# =============================================================================
# Callbacks
# =============================================================================

def register_callbacks(app: dash.Dash, data_manager: DataManager, config: DashboardConfig):
    """注册所有回调函数"""
    
    # -------------------------------------------------------------------------
    # Auto-refresh interval
    # -------------------------------------------------------------------------
    @app.callback(
        Output("live-indicator", "children"),
        Input("interval-component", "n_intervals")
    )
    def update_indicator(n):
        return "● Live" if n % 2 == 0 else "○ Live"
    
    # -------------------------------------------------------------------------
    # ML Training Callbacks
    # -------------------------------------------------------------------------
    @app.callback(
        [Output("current-loss", "children"),
         Output("current-force-rmse", "children"),
         Output("current-energy-rmse", "children"),
         Output("training-steps", "children"),
         Output("training-progress", "children"),
         Output("loss-trend", "children")],
        Input("interval-component", "n_intervals")
    )
    def update_training_kpis(n):
        df = data_manager.get_training_data()
        
        if df.empty or len(df) == 0:
            return "--", "--", "--", "--", "--", "--"
        
        current = df.iloc[-1]
        
        loss = f"{current.get('loss', 0):.4e}"
        force_rmse = f"{current.get('force_rmse', 0):.4f}"
        energy_rmse = f"{current.get('energy_rmse', 0):.4f}"
        steps = f"{int(current.get('batch', 0))}"
        
        # Calculate progress
        max_steps = 1000000  # Default max
        progress_pct = min(100, int(current.get('batch', 0)) / max_steps * 100)
        progress = f"{progress_pct}% of {max_steps}"
        
        # Loss trend
        if len(df) > 100:
            recent_loss = df['loss'].tail(100).mean()
            old_loss = df['loss'].head(100).mean()
            trend = "↘ Decreasing" if recent_loss < old_loss else "↗ Increasing"
        else:
            trend = "--"
        
        return loss, force_rmse, energy_rmse, steps, progress, trend
    
    @app.callback(
        Output("loss-curve-chart", "figure"),
        [Input("interval-component", "n_intervals"),
         Input("loss-scale", "value")]
    )
    def update_loss_curve(n, scale):
        df = data_manager.get_training_data()
        
        fig = go.Figure()
        
        if not df.empty and 'batch' in df.columns:
            # Subsample for performance
            if len(df) > config.max_points:
                df = df.iloc[::len(df)//config.max_points]
            
            fig.add_trace(go.Scatter(
                x=df['batch'],
                y=df['loss'],
                mode='lines',
                name='Total Loss',
                line=dict(color='#1f77b4', width=2)
            ))
            
            if 'energy_rmse' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['batch'],
                    y=df['energy_rmse'],
                    mode='lines',
                    name='Energy RMSE',
                    line=dict(color='#2ca02c', width=1),
                    yaxis='y2'
                ))
        
        fig.update_layout(
            title="Training Loss vs Steps",
            xaxis_title="Training Steps",
            yaxis_title="Loss",
            yaxis_type=scale,
            yaxis2=dict(
                title="Energy RMSE",
                overlaying='y',
                side='right'
            ),
            template="plotly_white",
            showlegend=True,
            height=400,
        )
        
        return fig
    
    @app.callback(
        Output("rmse-chart", "figure"),
        Input("interval-component", "n_intervals")
    )
    def update_rmse_chart(n):
        df = data_manager.get_training_data()
        
        fig = go.Figure()
        
        if not df.empty and 'batch' in df.columns:
            if 'force_rmse' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['batch'],
                    y=df['force_rmse'],
                    mode='lines',
                    name='Force RMSE',
                    line=dict(color='#d62728', width=2)
                ))
            
            if 'energy_rmse' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['batch'],
                    y=df['energy_rmse'],
                    mode='lines',
                    name='Energy RMSE',
                    line=dict(color='#2ca02c', width=2)
                ))
            
            if 'virial_rmse' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['batch'],
                    y=df['virial_rmse'],
                    mode='lines',
                    name='Virial RMSE',
                    line=dict(color='#9467bd', width=1)
                ))
        
        # Add threshold lines
        fig.add_hline(y=config.force_error_threshold, line_dash="dash", 
                      annotation_text="Force Threshold", line_color="red")
        fig.add_hline(y=config.energy_error_threshold, line_dash="dash",
                      annotation_text="Energy Threshold", line_color="green")
        
        fig.update_layout(
            title="RMSE Evolution",
            xaxis_title="Training Steps",
            yaxis_title="RMSE",
            yaxis_type="log",
            template="plotly_white",
            showlegend=True,
            height=400,
        )
        
        return fig
    
    @app.callback(
        Output("lr-chart", "figure"),
        Input("interval-component", "n_intervals")
    )
    def update_lr_chart(n):
        df = data_manager.get_training_data()
        
        fig = go.Figure()
        
        if not df.empty and 'lr' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['batch'],
                y=df['lr'],
                mode='lines',
                fill='tozeroy',
                line=dict(color='#ff7f0e', width=2)
            ))
        
        fig.update_layout(
            title="Learning Rate Schedule",
            xaxis_title="Training Steps",
            yaxis_title="Learning Rate",
            yaxis_type="log",
            template="plotly_white",
            height=300,
        )
        
        return fig
    
    # -------------------------------------------------------------------------
    # MD Simulation Callbacks
    # -------------------------------------------------------------------------
    @app.callback(
        Output("md-file-selector", "options"),
        Input("interval-component", "n_intervals")
    )
    def update_md_file_list(n):
        files = data_manager.get_md_dump_files()
        return [{"label": Path(f).name, "value": f} for f in files]
    
    @app.callback(
        Output("temperature-chart", "figure"),
        Input("interval-component", "n_intervals")
    )
    def update_temperature_chart(n):
        df = data_manager.get_md_trajectory_data()
        
        fig = go.Figure()
        
        if not df.empty and 'Temp' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Step'] if 'Step' in df.columns else range(len(df)),
                y=df['Temp'],
                mode='lines',
                line=dict(color='#ff7f0e', width=1)
            ))
            
            # Add running average
            window = min(100, len(df) // 10)
            if window > 1:
                fig.add_trace(go.Scatter(
                    x=df['Step'] if 'Step' in df.columns else range(len(df)),
                    y=df['Temp'].rolling(window=window).mean(),
                    mode='lines',
                    name=f'MA({window})',
                    line=dict(color='#d62728', width=2)
                ))
        
        fig.update_layout(
            title="Temperature vs Time",
            xaxis_title="Step",
            yaxis_title="Temperature (K)",
            template="plotly_white",
            showlegend=True,
            height=240,
        )
        
        return fig
    
    @app.callback(
        Output("energy-chart", "figure"),
        Input("interval-component", "n_intervals")
    )
    def update_energy_chart(n):
        df = data_manager.get_md_trajectory_data()
        
        fig = go.Figure()
        
        if not df.empty:
            # Try different energy column names
            energy_cols = ['TotEng', 'TotEnergy', 'Energy', 'etotal', 'pe']
            for col in energy_cols:
                if col in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df['Step'] if 'Step' in df.columns else range(len(df)),
                        y=df[col],
                        mode='lines',
                        name=col,
                        line=dict(width=1)
                    ))
                    break
        
        fig.update_layout(
            title="Energy vs Time",
            xaxis_title="Step",
            yaxis_title="Energy (eV)",
            template="plotly_white",
            height=240,
        )
        
        return fig
    
    @app.callback(
        [Output("structure-3d", "figure"),
         Output("frame-slider", "max"),
         Output("frame-info", "children")],
        [Input("frame-slider", "value"),
         Input("md-file-selector", "value")]
    )
    def update_3d_structure(frame, filepath):
        if not filepath or not ASE_AVAILABLE:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                template="plotly_white",
            )
            return fig, 100, "No trajectory loaded"
        
        try:
            atoms = read(filepath, index=':')
            if isinstance(atoms, list):
                max_frame = len(atoms) - 1
                current_atoms = atoms[min(frame, max_frame)]
            else:
                max_frame = 0
                current_atoms = atoms
            
            # Create 3D scatter plot
            positions = current_atoms.get_positions()
            symbols = current_atoms.get_chemical_symbols()
            
            # Color mapping
            unique_symbols = list(set(symbols))
            colors = px.colors.qualitative.Plotly[:len(unique_symbols)]
            symbol_colors = {s: c for s, c in zip(unique_symbols, colors)}
            
            fig = go.Figure()
            
            for symbol in unique_symbols:
                mask = [s == symbol for s in symbols]
                fig.add_trace(go.Scatter3d(
                    x=positions[mask, 0],
                    y=positions[mask, 1],
                    z=positions[mask, 2],
                    mode='markers',
                    name=symbol,
                    marker=dict(
                        size=8,
                        color=symbol_colors[symbol],
                        opacity=0.8
                    ),
                    text=[symbol] * sum(mask),
                ))
            
            # Add cell boundaries
            cell = current_atoms.get_cell()
            if cell is not None:
                # Simplified cell visualization
                pass
            
            fig.update_layout(
                scene=dict(
                    xaxis_title="X (Å)",
                    yaxis_title="Y (Å)",
                    zaxis_title="Z (Å)",
                    aspectmode='data'
                ),
                template="plotly_white",
                showlegend=True,
                height=500,
            )
            
            info = f"Frame {min(frame, max_frame)} / {max_frame} | {len(current_atoms)} atoms"
            return fig, max_frame, info
            
        except Exception as e:
            logger.error(f"Failed to load structure: {e}")
            fig = go.Figure()
            return fig, 100, f"Error: {str(e)}"
    
    # -------------------------------------------------------------------------
    # Screening Callbacks
    # -------------------------------------------------------------------------
    @app.callback(
        [Output("screening-scatter", "figure"),
         Output("x-axis", "options"),
         Output("y-axis", "options"),
         Output("screening-table-container", "children")],
        [Input("interval-component", "n_intervals"),
         Input("x-axis", "value"),
         Input("y-axis", "value"),
         Input("apply-filters", "n_clicks")],
        [State("screening-property", "value"),
         State("min-value", "value"),
         State("max-value", "value")]
    )
    def update_screening(n, x_col, y_col, filter_clicks, filter_prop, min_val, max_val):
        df = data_manager.get_screening_results()
        
        if df.empty:
            # Return empty figure and table
            fig = go.Figure()
            fig.update_layout(
                title="No screening data available",
                template="plotly_white"
            )
            return fig, [], [], html.P("No data available")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        options = [{"label": c, "value": c} for c in numeric_cols]
        
        # Set default columns
        if x_col is None:
            x_col = numeric_cols[0] if numeric_cols else df.columns[0]
        if y_col is None:
            y_col = numeric_cols[1] if len(numeric_cols) > 1 else x_col
        
        # Apply filters
        ctx = callback_context
        if ctx.triggered and 'apply-filters' in ctx.triggered[0]['prop_id']:
            if filter_prop and filter_prop in df.columns:
                if min_val is not None:
                    df = df[df[filter_prop] >= min_val]
                if max_val is not None:
                    df = df[df[filter_prop] <= max_val]
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            hover_data=['structure_id', 'formula'] if 'structure_id' in df.columns else None,
            color='ionic_conductivity' if 'ionic_conductivity' in df.columns else None,
            color_continuous_scale='Viridis',
            title=f"{y_col} vs {x_col}"
        )
        
        fig.update_layout(
            template="plotly_white",
            height=500,
        )
        
        # Create table
        table = DataTable(
            data=df.head(100).to_dict('records'),
            columns=[{"name": i, "id": i} for i in df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            sort_action='native',
            filter_action='native',
            page_action='native',
            page_size=10,
        )
        
        return fig, options, options, table
    
    @app.callback(
        Output("distribution-chart", "figure"),
        Input("screening-property", "value")
    )
    def update_distribution(prop):
        df = data_manager.get_screening_results()
        
        fig = go.Figure()
        
        if not df.empty and prop and prop in df.columns:
            fig.add_trace(go.Histogram(
                x=df[prop],
                nbinsx=30,
                name=prop,
                marker_color='#3498db'
            ))
            
            # Add KDE
            if len(df) > 10:
                from scipy import stats
                data = df[prop].dropna()
                kde_x = np.linspace(data.min(), data.max(), 100)
                kde = stats.gaussian_kde(data)
                fig.add_trace(go.Scatter(
                    x=kde_x,
                    y=kde(kde_x) * len(data) * (data.max() - data.min()) / 30,
                    mode='lines',
                    name='KDE',
                    line=dict(color='red', width=2)
                ))
        
        fig.update_layout(
            title=f"Distribution of {prop}",
            xaxis_title=prop,
            yaxis_title="Count",
            template="plotly_white",
            height=300,
        )
        
        return fig
    
    # -------------------------------------------------------------------------
    # Active Learning Callbacks
    # -------------------------------------------------------------------------
    @app.callback(
        [Output("current-iteration", "children"),
         Output("total-candidates", "children"),
         Output("convergence-status", "children"),
         Output("avg-deviation", "children")],
        Input("interval-component", "n_intervals")
    )
    def update_al_kpis(n):
        df = data_manager.get_active_learning_progress()
        
        if df.empty:
            return "--", "--", "Not Started", "--"
        
        current_iter = df['iteration'].max() if 'iteration' in df.columns else 0
        total_cand = df['candidate'].sum() if 'candidate' in df.columns else 0
        
        # Convergence status
        if 'candidate' in df.columns and len(df) > 2:
            recent = df['candidate'].tail(3).mean()
            status = "✓ Converged" if recent < 5 else "⟳ In Progress"
        else:
            status = "⟳ In Progress"
        
        # Average deviation
        devi_df = data_manager.get_model_deviation_history()
        avg_dev = devi_df['forces'].mean() if not devi_df.empty and 'forces' in devi_df.columns else 0
        
        return (
            f"{current_iter}",
            f"{int(total_cand)}",
            status,
            f"{avg_dev:.4f}"
        )
    
    @app.callback(
        Output("iteration-progress-chart", "figure"),
        Input("interval-component", "n_intervals")
    )
    def update_iteration_progress(n):
        df = data_manager.get_active_learning_progress()
        
        fig = go.Figure()
        
        if not df.empty and 'iteration' in df.columns:
            # Stacked bar chart
            categories = ['accurate', 'candidate', 'failed']
            colors = ['#2ecc71', '#f39c12', '#e74c3c']
            
            for cat, color in zip(categories, colors):
                if cat in df.columns:
                    fig.add_trace(go.Bar(
                        name=cat.capitalize(),
                        x=df['iteration'],
                        y=df[cat],
                        marker_color=color
                    ))
        
        fig.update_layout(
            title="Structures per Iteration",
            xaxis_title="Iteration",
            yaxis_title="Number of Structures",
            barmode='stack',
            template="plotly_white",
            height=350,
        )
        
        return fig
    
    @app.callback(
        Output("deviation-distribution-chart", "figure"),
        Input("interval-component", "n_intervals")
    )
    def update_deviation_distribution(n):
        df = data_manager.get_model_deviation_history()
        
        fig = go.Figure()
        
        if not df.empty and 'forces' in df.columns and 'iteration' in df.columns:
            # Box plot per iteration
            iterations = sorted(df['iteration'].unique())
            
            for iteration in iterations:
                iter_data = df[df['iteration'] == iteration]['forces']
                fig.add_trace(go.Box(
                    y=iter_data,
                    name=f"Iter {iteration}",
                    boxpoints='outliers'
                ))
        
        # Add threshold lines
        fig.add_hline(y=config.force_error_threshold, line_dash="dash",
                      annotation_text="Trust Lo", line_color="green")
        fig.add_hline(y=config.force_error_threshold * 3, line_dash="dash",
                      annotation_text="Trust Hi", line_color="red")
        
        fig.update_layout(
            title="Model Deviation Distribution",
            xaxis_title="Iteration",
            yaxis_title="Max Force Deviation (eV/Å)",
            template="plotly_white",
            height=350,
        )
        
        return fig
    
    @app.callback(
        Output("error-convergence-chart", "figure"),
        Input("interval-component", "n_intervals")
    )
    def update_error_convergence(n):
        df = data_manager.get_model_deviation_history()
        
        fig = go.Figure()
        
        if not df.empty and 'iteration' in df.columns:
            # Mean error per iteration
            mean_by_iter = df.groupby('iteration')['forces'].agg(['mean', 'std']).reset_index()
            
            fig.add_trace(go.Scatter(
                x=mean_by_iter['iteration'],
                y=mean_by_iter['mean'],
                mode='lines+markers',
                name='Mean Error',
                error_y=dict(
                    type='data',
                    array=mean_by_iter['std'],
                    visible=True
                ),
                line=dict(color='#3498db', width=2)
            ))
            
            # Add threshold
            fig.add_hline(y=config.force_error_threshold, line_dash="dash",
                          annotation_text="Target Error", line_color="green")
        
        fig.update_layout(
            title="Error Convergence",
            xaxis_title="Iteration",
            yaxis_title="Mean Force Deviation (eV/Å)",
            template="plotly_white",
            height=350,
        )
        
        return fig
    
    # -------------------------------------------------------------------------
    # Export Callbacks
    # -------------------------------------------------------------------------
    @app.callback(
        Output("export-status", "children"),
        [Input("export-pdf", "n_clicks"),
         Input("export-csv", "n_clicks"),
         Input("export-charts", "n_clicks")],
        prevent_initial_call=True
    )
    def handle_export(pdf_clicks, csv_clicks, chart_clicks):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = Path(config.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        if button_id == "export-pdf":
            # Generate PDF report
            pdf_path = export_dir / f"dashboard_report_{timestamp}.pdf"
            generate_pdf_report(str(pdf_path), data_manager, config)
            return dbc.Alert(f"✓ PDF exported: {pdf_path}", color="success", dismissable=True)
        
        elif button_id == "export-csv":
            # Export all data to CSV
            csv_path = export_dir / f"screening_data_{timestamp}.csv"
            df = data_manager.get_screening_results()
            if not df.empty:
                df.to_csv(csv_path, index=False)
                return dbc.Alert(f"✓ CSV exported: {csv_path}", color="success", dismissable=True)
            return dbc.Alert("No data to export", color="warning")
        
        elif button_id == "export-charts":
            # Export charts as images
            charts_dir = export_dir / f"charts_{timestamp}"
            charts_dir.mkdir(exist_ok=True)
            return dbc.Alert(f"✓ Charts exported to: {charts_dir}", color="success", dismissable=True)
        
        return ""


# =============================================================================
# PDF Export
# =============================================================================

def generate_pdf_report(output_path: str, data_manager: DataManager, config: DashboardConfig):
    """生成PDF报告"""
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30
    )
    story.append(Paragraph("DFT+LAMMPS Workflow Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Summary section
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    
    # Get data summaries
    training_df = data_manager.get_training_data()
    screening_df = data_manager.get_screening_results()
    al_df = data_manager.get_active_learning_progress()
    
    summary_data = [
        ['Metric', 'Value'],
        ['Training Steps', f"{len(training_df)}"],
        ['Screening Candidates', f"{len(screening_df)}"],
        ['AL Iterations', f"{len(al_df)}"],
    ]
    
    if not training_df.empty and 'loss' in training_df.columns:
        summary_data.append(['Current Loss', f"{training_df['loss'].iloc[-1]:.4e}"])
    
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Top candidates
    if not screening_df.empty:
        story.append(Paragraph("Top Screening Candidates", styles['Heading2']))
        
        # Select top 10 by ionic conductivity if available
        if 'ionic_conductivity' in screening_df.columns:
            top_df = screening_df.nlargest(10, 'ionic_conductivity')
        else:
            top_df = screening_df.head(10)
        
        cols = ['structure_id', 'formula', 'ionic_conductivity'] if 'ionic_conductivity' in top_df.columns else top_df.columns[:3].tolist()
        table_data = [cols]
        
        for _, row in top_df.iterrows():
            table_data.append([str(row.get(c, '')) for c in cols])
        
        candidates_table = Table(table_data)
        candidates_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        story.append(candidates_table)
    
    doc.build(story)
    logger.info(f"PDF report generated: {output_path}")


# =============================================================================
# Main Application
# =============================================================================

def create_app(config_path: str = None) -> dash.Dash:
    """创建Dash应用"""
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = DashboardConfig()
    
    # Initialize data manager
    data_manager = DataManager(config)
    
    # Create app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True
    )
    
    # Layout
    app.layout = html.Div([
        create_header(),
        
        dbc.Container([
            dbc.Tabs([
                create_ml_training_tab(),
                create_md_simulation_tab(),
                create_screening_tab(),
                create_active_learning_tab(),
            ], id="main-tabs"),
        ], fluid=True),
        
        create_export_section(),
        
        # Auto-refresh interval
        dcc.Interval(
            id='interval-component',
            interval=config.refresh_interval * 1000,  # milliseconds
            n_intervals=0
        ),
        
        # Store for sharing data between callbacks
        dcc.Store(id='shared-data-store'),
        
    ])
    
    # Register callbacks
    register_callbacks(app, data_manager, config)
    
    return app, config


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DFT+LAMMPS Workflow Dashboard')
    parser.add_argument('--config', '-c', type=str, default='dashboard_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--host', type=str, default=None,
                        help='Host to bind to')
    parser.add_argument('--port', '-p', type=int, default=None,
                        help='Port to bind to')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create app
    app, config = create_app(args.config)
    
    # Override config with command line args
    host = args.host or config.host
    port = args.port or config.port
    debug = args.debug or config.debug
    
    logger.info(f"Starting dashboard on http://{host}:{port}")
    logger.info(f"Debug mode: {debug}")
    
    # Run server
    app.run_server(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
