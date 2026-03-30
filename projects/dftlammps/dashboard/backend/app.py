"""
DFT/LAMMPS Dashboard - Dash应用主模块
========================================
基于Plotly Dash的交互式仪表板

参考设计理念:
- Material Design: 使用dbc组件实现现代UI
- Reactive UI: 回调驱动数据更新
- Progressive Enhancement: 基础功能不依赖JS
"""

import os
import sys
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd

# Dash imports
import dash
from dash import dcc, html, Input, Output, State, callback_context, MATCH, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import dash_table as dt

# Plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 导入数据管理器
from data_manager import (
    DataManager, DashboardConfig, load_config,
    ASE_AVAILABLE, PYMATGEN_AVAILABLE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 主题与样式定义
# =============================================================================

THEME = {
    'primary': '#1976d2',
    'secondary': '#424242',
    'success': '#4caf50',
    'warning': '#ff9800',
    'danger': '#f44336',
    'info': '#2196f3',
    'light': '#f5f5f5',
    'dark': '#212121',
    'background': '#fafafa',
}

CARD_STYLE = {
    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
    'borderRadius': '8px',
    'border': 'none',
}

HEADER_STYLE = {
    'backgroundColor': THEME['primary'],
    'color': 'white',
    'padding': '1rem 2rem',
    'marginBottom': '1rem',
}

# =============================================================================
# 布局组件工厂函数
# =============================================================================

def create_kpi_card(title: str, value: str, subtitle: str = "", 
                    icon: str = "📊", color: str = "primary") -> dbc.Card:
    """创建KPI卡片"""
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Span(icon, style={'fontSize': '2rem', 'marginRight': '1rem'}),
                html.Div([
                    html.H4(title, className="card-title text-muted", 
                           style={'fontSize': '0.875rem', 'marginBottom': '0.25rem'}),
                    html.H2(value, className=f"text-{color}", 
                           style={'fontSize': '1.75rem', 'fontWeight': 'bold', 'marginBottom': '0'}),
                    html.Small(subtitle, className="text-muted") if subtitle else None,
                ], style={'flex': 1}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
        ]),
        style={**CARD_STYLE, 'borderLeft': f'4px solid {THEME[color]}'},
    )


def create_graph_card(title: str, graph_id: str, 
                     controls: Optional[List] = None,
                     height: str = "400px") -> dbc.Card:
    """创建带控制按钮的图表卡片"""
    header = [
        html.H5(title, className="card-title", style={'margin': 0}),
    ]
    if controls:
        header.append(html.Div(controls, style={'display': 'flex', 'gap': '0.5rem'}))
    
    return dbc.Card([
        dbc.CardHeader(header, style={
            'display': 'flex', 
            'justifyContent': 'space-between',
            'alignItems': 'center',
            'backgroundColor': 'white',
            'borderBottom': '1px solid #e0e0e0'
        }),
        dbc.CardBody(
            dcc.Graph(
                id=graph_id,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                },
                style={'height': height}
            ),
            style={'padding': '1rem'}
        ),
    ], style=CARD_STYLE)


def create_section_header(title: str, description: str = "") -> html.Div:
    """创建章节标题"""
    return html.Div([
        html.H4(title, style={'fontWeight': '600', 'marginBottom': '0.25rem'}),
        html.P(description, style={'color': '#666', 'marginBottom': '1rem'}) if description else None,
        html.Hr(style={'marginBottom': '1.5rem'}),
    ])


# =============================================================================
# 应用布局定义
# =============================================================================

def create_layout(config: DashboardConfig) -> html.Div:
    """创建应用布局"""
    
    # 导航栏
    navbar = dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand([
                html.I(className="fas fa-atom", style={'marginRight': '0.5rem'}),
                "DFT/LAMMPS Dashboard"
            ], href="#", className="ms-2"),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Overview", href="#overview")),
                dbc.NavItem(dbc.NavLink("Training", href="#training")),
                dbc.NavItem(dbc.NavLink("MD Simulation", href="#md")),
                dbc.NavItem(dbc.NavLink("Screening", href="#screening")),
                dbc.NavItem(dbc.NavLink("Structures", href="#structures")),
                dbc.NavItem(dbc.NavLink("Workflows", href="#workflows")),
            ], className="ms-auto", navbar=True),
            dbc.Button(
                [html.I(className="fas fa-sync"), " Refresh"],
                id="refresh-btn",
                color="light",
                size="sm",
                className="ms-2"
            ),
        ], fluid=True),
        color="primary",
        dark=True,
        className="mb-4",
        style={'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}
    )
    
    # 刷新间隔组件
    refresh_interval = dcc.Interval(
        id='interval-component',
        interval=config.refresh_interval * 1000,  # milliseconds
        disabled=not config.auto_refresh
    )
    
    # 存储组件
    store_data = dcc.Store(id='store-data')
    store_config = dcc.Store(id='store-config', data=config.__dict__)
    
    # 标签页内容
    tabs = dbc.Tabs([
        dbc.Tab(label="📊 Overview", tab_id="tab-overview"),
        dbc.Tab(label="🧠 ML Training", tab_id="tab-training"),
        dbc.Tab(label="🔬 MD Simulation", tab_id="tab-md"),
        dbc.Tab(label="🔍 Screening", tab_id="tab-screening"),
        dbc.Tab(label="🏗️ Structures", tab_id="tab-structures"),
        dbc.Tab(label="⚙️ Workflows", tab_id="tab-workflows"),
    ], id="main-tabs", active_tab="tab-overview", className="mb-4")
    
    # 标签页内容区域
    tab_content = html.Div(id="tab-content")
    
    # 状态栏
    status_bar = html.Div(
        id="status-bar",
        style={
            'position': 'fixed',
            'bottom': 0,
            'left': 0,
            'right': 0,
            'backgroundColor': '#f5f5f5',
            'borderTop': '1px solid #ddd',
            'padding': '0.5rem 1rem',
            'fontSize': '0.75rem',
            'display': 'flex',
            'justifyContent': 'space-between',
            'zIndex': 1000,
        }
    )
    
    return html.Div([
        navbar,
        dbc.Container([
            refresh_interval,
            store_data,
            store_config,
            tabs,
            tab_content,
            html.Div(style={'height': '50px'}),  # 底部留白
        ], fluid=True),
        status_bar,
    ])


# =============================================================================
# 标签页内容创建函数
# =============================================================================

def create_overview_tab(data_manager: DataManager) -> html.Div:
    """创建概览标签页"""
    
    # 系统状态卡片
    status_cards = dbc.Row([
        dbc.Col(create_kpi_card(
            "Total Structures", 
            "--", 
            "Available in database",
            "🏗️", "info"
        ), width=3),
        dbc.Col(create_kpi_card(
            "Active Simulations", 
            "--", 
            "Currently running",
            "⚡", "warning"
        ), width=3),
        dbc.Col(create_kpi_card(
            "ML Models", 
            "--", 
            "Trained and ready",
            "🧠", "success"
        ), width=3),
        dbc.Col(create_kpi_card(
            "Screened Materials", 
            "--", 
            "High-throughput results",
            "🔬", "primary"
        ), width=3),
    ], className="mb-4")
    
    # 最近活动
    activity_section = html.Div([
        create_section_header("Recent Activity", "Latest operations and results"),
        dbc.Card([
            dbc.CardBody(id="recent-activity-content", children=[
                html.P("No recent activity", className="text-muted text-center"),
            ])
        ], style=CARD_STYLE),
    ], className="mb-4")
    
    # 快速链接
    quick_links = html.Div([
        create_section_header("Quick Actions"),
        dbc.Row([
            dbc.Col(dbc.Button([
                html.I(className="fas fa-play me-2"),
                "New Workflow"
            ], color="primary", className="w-100"), width=2),
            dbc.Col(dbc.Button([
                html.I(className="fas fa-upload me-2"),
                "Import Data"
            ], color="secondary", className="w-100"), width=2),
            dbc.Col(dbc.Button([
                html.I(className="fas fa-file-export me-2"),
                "Export Report"
            ], color="info", className="w-100"), width=2),
            dbc.Col(dbc.Button([
                html.I(className="fas fa-cog me-2"),
                "Settings"
            ], color="light", className="w-100"), width=2),
        ], className="g-2"),
    ])
    
    return html.Div([status_cards, activity_section, quick_links])


def create_training_tab() -> html.Div:
    """创建ML训练标签页"""
    
    # KPI卡片
    kpi_cards = dbc.Row([
        dbc.Col(html.Div(id="training-kpi-step"), width=3),
        dbc.Col(html.Div(id="training-kpi-loss"), width=3),
        dbc.Col(html.Div(id="training-kpi-energy"), width=3),
        dbc.Col(html.Div(id="training-kpi-force"), width=3),
    ], className="mb-4")
    
    # 损失曲线
    loss_graph = dbc.Row([
        dbc.Col(create_graph_card(
            "Training Loss",
            "training-loss-graph",
            controls=[
                dbc.ButtonGroup([
                    dbc.Button("Linear", id="loss-scale-linear", size="sm", color="primary", outline=True),
                    dbc.Button("Log", id="loss-scale-log", size="sm", color="secondary", outline=True),
                ]),
            ],
            height="400px"
        ), width=8),
        dbc.Col([
            create_graph_card(
                "Learning Rate",
                "lr-graph",
                height="190px"
            ),
            html.Div(style={'height': '20px'}),
            create_graph_card(
                "RMSE Distribution",
                "rmse-dist-graph",
                height="190px"
            ),
        ], width=4),
    ], className="mb-4")
    
    # RMSE曲线
    rmse_graph = dbc.Row([
        dbc.Col(create_graph_card(
            "Energy RMSE",
            "energy-rmse-graph",
            height="300px"
        ), width=6),
        dbc.Col(create_graph_card(
            "Force RMSE",
            "force-rmse-graph",
            height="300px"
        ), width=6),
    ], className="mb-4")
    
    # 控制面板
    control_panel = dbc.Card([
        dbc.CardHeader("Training Controls"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Max Steps"),
                    dbc.Input(id="training-max-steps", type="number", value=1000000),
                ], width=3),
                dbc.Col([
                    dbc.Label("Learning Rate"),
                    dbc.Input(id="training-lr", type="number", value=0.001, step=0.0001),
                ], width=3),
                dbc.Col([
                    dbc.Label("Batch Size"),
                    dbc.Input(id="training-batch-size", type="number", value=4),
                ], width=3),
                dbc.Col([
                    dbc.Label("Actions"),
                    dbc.ButtonGroup([
                        dbc.Button("Start", color="success", id="training-start"),
                        dbc.Button("Pause", color="warning", id="training-pause"),
                        dbc.Button("Stop", color="danger", id="training-stop"),
                    ]),
                ], width=3),
            ]),
        ]),
    ], style=CARD_STYLE, className="mb-4")
    
    return html.Div([
        create_section_header("ML Potential Training", "Monitor neural network training progress"),
        kpi_cards,
        loss_graph,
        rmse_graph,
        control_panel,
    ])


def create_md_tab() -> html.Div:
    """创建MD模拟标签页"""
    
    # 轨迹播放器
    trajectory_player = dbc.Card([
        dbc.CardHeader("Trajectory Player"),
        dbc.CardBody([
            # 3D可视化区域
            html.Div(
                id="structure-3d-view",
                style={
                    'height': '500px',
                    'backgroundColor': '#1a1a2e',
                    'borderRadius': '4px',
                    'position': 'relative',
                },
                children=[
                    html.Div(
                        "3D Structure Viewer",
                        style={
                            'position': 'absolute',
                            'top': '50%',
                            'left': '50%',
                            'transform': 'translate(-50%, -50%)',
                            'color': '#666',
                            'fontSize': '1.25rem',
                        }
                    )
                ]
            ),
            # 播放控制
            html.Div([
                dbc.ButtonGroup([
                    dbc.Button(html.I(className="fas fa-step-backward"), id="traj-first"),
                    dbc.Button(html.I(className="fas fa-play"), id="traj-play"),
                    dbc.Button(html.I(className="fas fa-pause"), id="traj-pause"),
                    dbc.Button(html.I(className="fas fa-step-forward"), id="traj-last"),
                ]),
                dcc.Slider(
                    id='traj-frame-slider',
                    min=0,
                    max=100,
                    step=1,
                    value=0,
                    marks={0: '0', 50: '50', 100: '100'},
                    className="mx-3 flex-grow-1"
                ),
                html.Span(id="traj-frame-display", children="Frame: 0/100"),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '1rem'}),
        ]),
    ], style=CARD_STYLE, className="mb-4")
    
    # 物理量监控
    thermo_graphs = dbc.Row([
        dbc.Col(create_graph_card(
            "Temperature",
            "md-temp-graph",
            height="250px"
        ), width=4),
        dbc.Col(create_graph_card(
            "Energy",
            "md-energy-graph",
            height="250px"
        ), width=4),
        dbc.Col(create_graph_card(
            "Pressure",
            "md-pressure-graph",
            height="250px"
        ), width=4),
    ], className="mb-4")
    
    # MSD分析
    analysis_section = dbc.Row([
        dbc.Col(create_graph_card(
            "Mean Square Displacement",
            "msd-graph",
            controls=[
                dbc.Select(
                    id="msd-atom-type",
                    options=[
                        {"label": "All Atoms", "value": "all"},
                        {"label": "Li", "value": "Li"},
                        {"label": "O", "value": "O"},
                    ],
                    value="all",
                    style={'width': '150px'}
                ),
            ],
            height="300px"
        ), width=6),
        dbc.Col(create_graph_card(
            "Radial Distribution Function",
            "rdf-graph",
            height="300px"
        ), width=6),
    ], className="mb-4")
    
    return html.Div([
        create_section_header("Molecular Dynamics", "Monitor MD simulations and analyze trajectories"),
        trajectory_player,
        thermo_graphs,
        analysis_section,
    ])


def create_screening_tab() -> html.Div:
    """创建高通量筛选标签页"""
    
    # 筛选控制面板
    filter_panel = dbc.Card([
        dbc.CardHeader("Filter Criteria"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Energy Range (eV)"),
                    dcc.RangeSlider(
                        id='filter-energy',
                        min=-10, max=0, step=0.1,
                        value=[-10, 0],
                        marks={-10: '-10', -5: '-5', 0: '0'},
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Label("Band Gap (eV)"),
                    dcc.RangeSlider(
                        id='filter-bandgap',
                        min=0, max=10, step=0.1,
                        value=[0, 10],
                        marks={0: '0', 5: '5', 10: '10'},
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Label("Ionic Conductivity"),
                    dcc.RangeSlider(
                        id='filter-conductivity',
                        min=0, max=1, step=0.01,
                        value=[0, 1],
                        marks={0: '0', 0.5: '0.5', 1: '1'},
                    ),
                ], width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Button("Apply Filters", color="primary", id="apply-filters", className="mt-3"),
                    dbc.Button("Reset", color="secondary", id="reset-filters", className="mt-3 ms-2"),
                    dbc.Button("Export Results", color="info", id="export-screening", className="mt-3 ms-2"),
                ]),
            ]),
        ]),
    ], style=CARD_STYLE, className="mb-4")
    
    # 散点图矩阵
    scatter_section = dbc.Row([
        dbc.Col(create_graph_card(
            "Property Correlation",
            "screening-scatter",
            controls=[
                dbc.Select(
                    id="scatter-x",
                    options=[
                        {"label": "Formation Energy", "value": "formation_energy"},
                        {"label": "Band Gap", "value": "band_gap"},
                        {"label": "Volume", "value": "volume"},
                    ],
                    value="formation_energy",
                    style={'width': '150px'}
                ),
                dbc.Select(
                    id="scatter-y",
                    options=[
                        {"label": "Band Gap", "value": "band_gap"},
                        {"label": "Formation Energy", "value": "formation_energy"},
                        {"label": "Ionic Conductivity", "value": "ionic_conductivity"},
                    ],
                    value="band_gap",
                    style={'width': '150px'}
                ),
                dbc.Checklist(
                    options=[{"label": "Show Pareto Front", "value": "pareto"}],
                    value=["pareto"],
                    id="show-pareto",
                    inline=True,
                ),
            ],
            height="450px"
        ), width=8),
        dbc.Col(create_graph_card(
            "Property Distribution",
            "screening-histogram",
            height="450px"
        ), width=4),
    ], className="mb-4")
    
    # 结果表格
    results_table = dbc.Card([
        dbc.CardHeader("Screening Results"),
        dbc.CardBody(
            dt.DataTable(
                id='screening-table',
                columns=[
                    {"name": "ID", "id": "structure_id"},
                    {"name": "Formula", "id": "formula"},
                    {"name": "Energy (eV)", "id": "formation_energy", "type": "numeric", "format": {"specifier": ".3f"}},
                    {"name": "Band Gap (eV)", "id": "band_gap", "type": "numeric", "format": {"specifier": ".2f"}},
                    {"name": "Conductivity", "id": "ionic_conductivity", "type": "numeric", "format": {"specifier": ".2e"}},
                ],
                data=[],
                sort_action="native",
                filter_action="native",
                page_action="native",
                page_size=10,
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={
                    'backgroundColor': '#f5f5f5',
                    'fontWeight': 'bold',
                    'borderBottom': '2px solid #ddd'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#fafafa'
                    }
                ],
            )
        ),
    ], style=CARD_STYLE)
    
    return html.Div([
        create_section_header("High-Throughput Screening", "Browse and filter materials database"),
        filter_panel,
        scatter_section,
        results_table,
    ])


def create_structures_tab() -> html.Div:
    """创建结构可视化标签页"""
    
    # 文件浏览器
    file_browser = dbc.Card([
        dbc.CardHeader("Structure Files"),
        dbc.CardBody([
            dbc.Select(
                id="structure-file-select",
                options=[],
                placeholder="Select a structure file...",
                className="mb-3"
            ),
            dbc.ListGroup(id="structure-file-list", children=[]),
        ]),
    ], style=CARD_STYLE)
    
    # 3D可视化
    viewer = dbc.Card([
        dbc.CardHeader("Structure Viewer"),
        dbc.CardBody([
            html.Div(
                id="structure-viewer-3d",
                style={
                    'height': '500px',
                    'backgroundColor': '#0d1117',
                    'borderRadius': '4px',
                }
            ),
        ]),
    ], style=CARD_STYLE)
    
    # 结构信息
    info_panel = dbc.Card([
        dbc.CardHeader("Structure Information"),
        dbc.CardBody(id="structure-info", children=[
            html.P("Select a structure to view details", className="text-muted"),
        ]),
    ], style=CARD_STYLE)
    
    return html.Div([
        create_section_header("Structure Visualization", "View crystal structures and molecules"),
        dbc.Row([
            dbc.Col(file_browser, width=3),
            dbc.Col([
                viewer,
                html.Div(style={'height': '1rem'}),
                info_panel,
            ], width=9),
        ]),
    ])


def create_workflows_tab() -> html.Div:
    """创建工作流标签页"""
    
    # 工作流构建器
    builder = dbc.Card([
        dbc.CardHeader("Workflow Builder"),
        dbc.CardBody([
            # 节点库
            html.H6("Available Nodes", className="mb-2"),
            dbc.Row([
                dbc.Col(dbc.Button([
                    html.I(className="fas fa-cube me-2"),
                    "Load Structure"
                ], color="light", size="sm", className="w-100"), width=3),
                dbc.Col(dbc.Button([
                    html.I(className="fas fa-atom me-2"),
                    "DFT Calculation"
                ], color="light", size="sm", className="w-100"), width=3),
                dbc.Col(dbc.Button([
                    html.I(className="fas fa-wave-square me-2"),
                    "ML Potential"
                ], color="light", size="sm", className="w-100"), width=3),
                dbc.Col(dbc.Button([
                    html.I(className="fas fa-running me-2"),
                    "MD Simulation"
                ], color="light", size="sm", className="w-100"), width=3),
            ], className="g-2 mb-3"),
            
            # 画布
            html.Div(
                id="workflow-canvas",
                style={
                    'height': '400px',
                    'backgroundColor': '#f8f9fa',
                    'border': '2px dashed #dee2e6',
                    'borderRadius': '4px',
                    'position': 'relative',
                },
                children=[
                    html.Div(
                        "Drag nodes here to build workflow",
                        style={
                            'position': 'absolute',
                            'top': '50%',
                            'left': '50%',
                            'transform': 'translate(-50%, -50%)',
                            'color': '#adb5bd',
                        }
                    )
                ]
            ),
            
            # 控制按钮
            html.Div([
                dbc.Button("Run Workflow", color="success", className="me-2"),
                dbc.Button("Save", color="primary", className="me-2"),
                dbc.Button("Load", color="secondary", className="me-2"),
                dbc.Button("Clear", color="danger", outline=True),
            ], className="mt-3"),
        ]),
    ], style=CARD_STYLE, className="mb-4")
    
    # 运行中的工作流
    running_workflows = dbc.Card([
        dbc.CardHeader("Running Workflows"),
        dbc.CardBody(id="running-workflows-list", children=[
            html.P("No workflows running", className="text-muted text-center"),
        ]),
    ], style=CARD_STYLE)
    
    return html.Div([
        create_section_header("Workflow Builder", "Design and execute computational workflows"),
        builder,
        running_workflows,
    ])


# =============================================================================
# 回调函数注册
# =============================================================================

def register_callbacks(app: dash.Dash, data_manager: DataManager):
    """注册所有回调函数"""
    
    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "active_tab"),
    )
    def render_tab_content(active_tab):
        """渲染标签页内容"""
        if active_tab == "tab-overview":
            return create_overview_tab(data_manager)
        elif active_tab == "tab-training":
            return create_training_tab()
        elif active_tab == "tab-md":
            return create_md_tab()
        elif active_tab == "tab-screening":
            return create_screening_tab()
        elif active_tab == "tab-structures":
            return create_structures_tab()
        elif active_tab == "tab-workflows":
            return create_workflows_tab()
        return html.Div("Tab not found")
    
    @app.callback(
        Output("store-data", "data"),
        Input("interval-component", "n_intervals"),
        Input("refresh-btn", "n_clicks"),
        prevent_initial_call=False,
    )
    def update_data(n_intervals, n_clicks):
        """定期更新数据"""
        ctx = callback_context
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'training': {},
            'md': {},
            'screening': {},
        }
        
        # 加载训练数据
        training_df = data_manager.training.load_training_log()
        if not training_df.empty:
            data['training'] = data_manager.training.get_training_stats(training_df)
            data['training']['history'] = training_df.to_dict('records')[-1000:]  # 最近1000步
        
        # 加载MD数据
        md_df = data_manager.md.load_md_log()
        if not md_df.empty:
            data['md'] = data_manager.md.get_md_stats(md_df)
            data['md']['history'] = md_df.to_dict('records')[-1000:]
        
        # 加载筛选数据
        screening_df = data_manager.screening.load_screening_results()
        if not screening_df.empty:
            data['screening'] = {
                'count': len(screening_df),
                'columns': list(screening_df.columns),
                'data': screening_df.head(100).to_dict('records'),
            }
        
        return data
    
    @app.callback(
        Output("status-bar", "children"),
        Input("store-data", "data"),
    )
    def update_status_bar(data):
        """更新状态栏"""
        timestamp = data.get('timestamp', datetime.now().isoformat()) if data else datetime.now().isoformat()
        return [
            html.Span(f"Last updated: {timestamp}"),
            html.Span([
                "ASE: ", html.Span("✓" if ASE_AVAILABLE else "✗", 
                                 style={'color': 'green' if ASE_AVAILABLE else 'red'}),
                " | pymatgen: ", html.Span("✓" if PYMATGEN_AVAILABLE else "✗",
                                          style={'color': 'green' if PYMATGEN_AVAILABLE else 'red'}),
            ]),
        ]
    
    # Training Tab 回调
    @app.callback(
        Output("training-kpi-step", "children"),
        Output("training-kpi-loss", "children"),
        Output("training-kpi-energy", "children"),
        Output("training-kpi-force", "children"),
        Input("store-data", "data"),
    )
    def update_training_kpis(data):
        """更新训练KPI"""
        training_data = data.get('training', {}) if data else {}
        
        step_card = create_kpi_card(
            "Current Step",
            str(training_data.get('current_step', 0)),
            f"Total: {training_data.get('total_steps', 0)}",
            "🔄", "info"
        )
        
        loss_card = create_kpi_card(
            "Total Loss",
            f"{training_data.get('current_loss', 0):.4f}",
            "Training loss",
            "📉", "primary"
        )
        
        energy_rmse = training_data.get('energy_rmse')
        energy_card = create_kpi_card(
            "Energy RMSE",
            f"{energy_rmse:.4f}" if energy_rmse else "N/A",
            "eV/atom" if energy_rmse else "",
            "⚡", "success" if (energy_rmse and energy_rmse < 0.01) else "warning"
        )
        
        force_rmse = training_data.get('force_rmse')
        force_card = create_kpi_card(
            "Force RMSE",
            f"{force_rmse:.4f}" if force_rmse else "N/A",
            "eV/Å" if force_rmse else "",
            "💪", "success" if (force_rmse and force_rmse < 0.1) else "warning"
        )
        
        return step_card, loss_card, energy_card, force_card
    
    @app.callback(
        Output("training-loss-graph", "figure"),
        Input("store-data", "data"),
        Input("loss-scale-linear", "n_clicks"),
        Input("loss-scale-log", "n_clicks"),
    )
    def update_loss_graph(data, linear_clicks, log_clicks):
        """更新损失曲线"""
        ctx = callback_context
        
        # 确定刻度类型
        yaxis_type = 'linear'
        if ctx.triggered:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'loss-scale-log':
                yaxis_type = 'log'
        
        fig = go.Figure()
        
        if data and 'training' in data and 'history' in data['training']:
            history = data['training']['history']
            if history:
                df = pd.DataFrame(history)
                
                if 'step' in df.columns and 'total_loss' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df['step'],
                        y=df['total_loss'],
                        mode='lines',
                        name='Total Loss',
                        line=dict(color=THEME['primary'], width=2),
                    ))
        
        fig.update_layout(
            xaxis_title='Step',
            yaxis_title='Loss',
            yaxis_type=yaxis_type,
            template='plotly_white',
            margin=dict(l=50, r=20, t=20, b=50),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
        )
        
        return fig
    
    @app.callback(
        Output("lr-graph", "figure"),
        Output("rmse-dist-graph", "figure"),
        Input("store-data", "data"),
    )
    def update_training_secondary_graphs(data):
        """更新学习率和RMSE分布图"""
        # 学习率图
        lr_fig = go.Figure()
        if data and 'training' in data and 'history' in data['training']:
            history = data['training']['history']
            if history:
                df = pd.DataFrame(history)
                if 'step' in df.columns and 'learning_rate' in df.columns:
                    lr_fig.add_trace(go.Scatter(
                        x=df['step'],
                        y=df['learning_rate'],
                        mode='lines',
                        line=dict(color=THEME['info'], width=2),
                        fill='tozeroy',
                    ))
        
        lr_fig.update_layout(
            xaxis_title='Step',
            yaxis_title='Learning Rate',
            yaxis_type='log',
            template='plotly_white',
            margin=dict(l=40, r=20, t=10, b=40),
            showlegend=False,
        )
        
        # RMSE分布图
        rmse_fig = go.Figure()
        if data and 'training' in data and 'history' in data['training']:
            history = data['training']['history']
            if history:
                df = pd.DataFrame(history)
                if 'energy_rmse' in df.columns:
                    rmse_fig.add_trace(go.Histogram(
                        x=df['energy_rmse'].dropna(),
                        name='Energy RMSE',
                        opacity=0.7,
                        marker_color=THEME['success'],
                    ))
        
        rmse_fig.update_layout(
            xaxis_title='RMSE (eV)',
            yaxis_title='Count',
            template='plotly_white',
            margin=dict(l=40, r=20, t=10, b=40),
            showlegend=False,
        )
        
        return lr_fig, rmse_fig
    
    @app.callback(
        Output("energy-rmse-graph", "figure"),
        Output("force-rmse-graph", "figure"),
        Input("store-data", "data"),
    )
    def update_rmse_graphs(data):
        """更新RMSE曲线"""
        energy_fig = go.Figure()
        force_fig = go.Figure()
        
        if data and 'training' in data and 'history' in data['training']:
            history = data['training']['history']
            if history:
                df = pd.DataFrame(history)
                
                if 'step' in df.columns and 'energy_rmse' in df.columns:
                    energy_fig.add_trace(go.Scatter(
                        x=df['step'],
                        y=df['energy_rmse'],
                        mode='lines',
                        line=dict(color=THEME['success'], width=2),
                    ))
                    # 添加阈值线
                    energy_fig.add_hline(
                        y=0.001, line_dash="dash", 
                        line_color="red", annotation_text="Target"
                    )
                
                if 'step' in df.columns and 'force_rmse' in df.columns:
                    force_fig.add_trace(go.Scatter(
                        x=df['step'],
                        y=df['force_rmse'],
                        mode='lines',
                        line=dict(color=THEME['warning'], width=2),
                    ))
                    force_fig.add_hline(
                        y=0.05, line_dash="dash",
                        line_color="red", annotation_text="Target"
                    )
        
        energy_fig.update_layout(
            title="Energy RMSE",
            xaxis_title='Step',
            yaxis_title='RMSE (eV/atom)',
            template='plotly_white',
            margin=dict(l=50, r=20, t=50, b=50),
            showlegend=False,
        )
        
        force_fig.update_layout(
            title="Force RMSE",
            xaxis_title='Step',
            yaxis_title='RMSE (eV/Å)',
            template='plotly_white',
            margin=dict(l=50, r=20, t=50, b=50),
            showlegend=False,
        )
        
        return energy_fig, force_fig
    
    # MD Tab 回调
    @app.callback(
        Output("md-temp-graph", "figure"),
        Output("md-energy-graph", "figure"),
        Output("md-pressure-graph", "figure"),
        Input("store-data", "data"),
    )
    def update_md_graphs(data):
        """更新MD监控图"""
        figs = []
        
        for col, title, color in [
            ('Temp', 'Temperature (K)', THEME['danger']),
            ('TotEng', 'Total Energy (eV)', THEME['primary']),
            ('Press', 'Pressure (bar)', THEME['info']),
        ]:
            fig = go.Figure()
            
            if data and 'md' in data and 'history' in data['md']:
                history = data['md']['history']
                if history:
                    df = pd.DataFrame(history)
                    if 'Step' in df.columns and col in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df['Step'],
                            y=df[col],
                            mode='lines',
                            line=dict(color=color, width=1.5),
                        ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Step',
                yaxis_title=title.split('(')[0].strip(),
                template='plotly_white',
                margin=dict(l=50, r=20, t=40, b=40),
                showlegend=False,
            )
            figs.append(fig)
        
        return tuple(figs)
    
    # Screening Tab 回调
    @app.callback(
        Output("screening-table", "data"),
        Output("screening-scatter", "figure"),
        Input("store-data", "data"),
        Input("apply-filters", "n_clicks"),
        State("filter-energy", "value"),
        State("filter-bandgap", "value"),
        State("scatter-x", "value"),
        State("scatter-y", "value"),
        State("show-pareto", "value"),
    )
    def update_screening(data, n_clicks, energy_range, bandgap_range, 
                        x_col, y_col, show_pareto):
        """更新筛选结果"""
        table_data = []
        scatter_fig = go.Figure()
        
        if data and 'screening' in data and 'data' in data['screening']:
            df = pd.DataFrame(data['screening']['data'])
            
            if not df.empty:
                # 应用筛选
                if 'formation_energy' in df.columns and energy_range:
                    df = df[
                        (df['formation_energy'] >= energy_range[0]) &
                        (df['formation_energy'] <= energy_range[1])
                    ]
                
                if 'band_gap' in df.columns and bandgap_range:
                    df = df[
                        (df['band_gap'] >= bandgap_range[0]) &
                        (df['band_gap'] <= bandgap_range[1])
                    ]
                
                table_data = df.to_dict('records')
                
                # 散点图
                if x_col in df.columns and y_col in df.columns:
                    scatter_fig.add_trace(go.Scatter(
                        x=df[x_col],
                        y=df[y_col],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=df.get('band_gap', 0),
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title='Band Gap'),
                        ),
                        text=df.get('formula', ''),
                        hovertemplate='<b>%{text}</b><br>' +
                                     f'{x_col}: %{{x:.3f}}<br>' +
                                     f'{y_col}: %{{y:.3f}}<br>' +
                                     '<extra></extra>',
                    ))
                    
                    # Pareto前沿
                    if show_pareto and 'pareto' in show_pareto:
                        pareto_df = data_manager.screening.get_pareto_front(df, x_col, y_col)
                        if not pareto_df.empty:
                            scatter_fig.add_trace(go.Scatter(
                                x=pareto_df[x_col],
                                y=pareto_df[y_col],
                                mode='lines+markers',
                                line=dict(color='red', dash='dash'),
                                marker=dict(size=8, symbol='star'),
                                name='Pareto Front',
                            ))
        
        scatter_fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            template='plotly_white',
            margin=dict(l=60, r=60, t=20, b=60),
            showlegend=True,
        )
        
        return table_data, scatter_fig


# =============================================================================
# 主应用入口
# =============================================================================

def create_app(config: Optional[DashboardConfig] = None) -> dash.Dash:
    """创建Dash应用"""
    config = config or load_config()
    
    # 初始化数据管理器
    data_manager = DataManager(config)
    
    # 创建应用
    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            "https://use.fontawesome.com/releases/v6.3.0/css/all.css",
        ],
        suppress_callback_exceptions=True,
        title="DFT/LAMMPS Dashboard",
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )
    
    # 设置布局
    app.layout = create_layout(config)
    
    # 注册回调
    register_callbacks(app, data_manager)
    
    return app, data_manager


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DFT/LAMMPS Dashboard')
    parser.add_argument('--config', '-c', type=str, default='dashboard_config.yaml',
                       help='Path to config file')
    parser.add_argument('--host', '-H', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', '-p', type=int, default=8050,
                       help='Port to bind to')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    config.host = args.host
    config.port = args.port
    config.debug = args.debug
    
    # 创建应用
    app, data_manager = create_app(config)
    
    logger.info(f"Starting DFT/LAMMPS Dashboard on http://{config.host}:{config.port}")
    
    # 运行应用
    app.run(
        host=config.host,
        port=config.port,
        debug=config.debug,
    )


if __name__ == '__main__':
    main()
