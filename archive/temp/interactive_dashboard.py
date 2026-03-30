#!/usr/bin/env python3
"""
交互式仪表板完整示例
功能：实时监控、Plotly可视化、Dash/Streamlit仪表板
"""

import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timedelta
import json

# 可视化库
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("[WARN] Plotly未安装")

try:
    import dash
    from dash import dcc, html, Input, Output, State
    from dash.dependencies import ALL
    import dash_daq as daq
    HAS_DASH = True
except ImportError:
    HAS_DASH = False
    print("[WARN] Dash未安装")

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    print("[WARN] Streamlit未安装")


class MDDataGenerator:
    """MD模拟数据生成器（模拟实时数据流）"""
    
    def __init__(self):
        self.time = 0
        self.temperature = 300.0
        self.pressure = 1.0
        self.energy = -1000.0
        self.volume = 1000.0
        self.density = 1.0
        
    def get_next_frame(self):
        """生成下一帧数据"""
        self.time += 0.1
        
        # 添加噪声和趋势
        self.temperature = 300 + 20 * np.sin(self.time / 10) + np.random.normal(0, 2)
        self.pressure = 1 + 0.5 * np.sin(self.time / 5) + np.random.normal(0, 0.1)
        self.energy = -1000 + 50 * np.sin(self.time / 20) + np.random.normal(0, 5)
        self.volume = 1000 + 10 * np.sin(self.time / 15) + np.random.normal(0, 1)
        self.density = 1000 / self.volume
        
        return {
            'time': self.time,
            'temperature': self.temperature,
            'pressure': self.pressure,
            'energy': self.energy,
            'volume': self.volume,
            'density': self.density,
            'timestamp': datetime.now()
        }


class PlotlyVisualizer:
    """Plotly可视化工具"""
    
    def __init__(self):
        self.themes = {
            'dark': 'plotly_dark',
            'light': 'plotly_white',
            'seaborn': 'seaborn'
        }
    
    def create_trajectory_plot(self, positions, colors=None, sizes=None):
        """
        创建3D轨迹图
        """
        if not HAS_PLOTLY:
            print("[ERROR] Plotly未安装")
            return None
        
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        
        if colors is None:
            colors = z
        if sizes is None:
            sizes = 5
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title='Property')
            ),
            hovertemplate='<b>Atom %{pointNumber}</b><br>' +
                         'X: %{x:.3f}<br>' +
                         'Y: %{y:.3f}<br>' +
                         'Z: %{z:.3f}<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title='Molecular Configuration',
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='data'
            ),
            template=self.themes['light']
        )
        
        return fig
    
    def create_timeseries_plot(self, data_dict, title='Time Series'):
        """
        创建时间序列图
        """
        if not HAS_PLOTLY:
            return None
        
        fig = make_subplots(
            rows=len(data_dict), cols=1,
            shared_xaxes=True,
            subplot_titles=list(data_dict.keys())
        )
        
        for i, (name, df) in enumerate(data_dict.items(), 1):
            if 'time' in df.columns:
                x_col = 'time'
                y_cols = [c for c in df.columns if c != 'time']
            else:
                x_col = df.index
                y_cols = df.columns
            
            for y_col in y_cols:
                fig.add_trace(
                    go.Scatter(
                        x=df[x_col], 
                        y=df[y_col],
                        name=f"{name}_{y_col}",
                        line=dict(width=1.5)
                    ),
                    row=i, col=1
                )
        
        fig.update_layout(
            title=title,
            height=300 * len(data_dict),
            showlegend=True,
            template=self.themes['light']
        )
        
        return fig
    
    def create_correlation_heatmap(self, df):
        """创建相关性热力图"""
        if not HAS_PLOTLY:
            return None
        
        corr = df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Property Correlation Matrix',
            template=self.themes['light']
        )
        
        return fig
    
    def create_rdf_plot(self, r, g_r):
        """创建径向分布函数图"""
        if not HAS_PLOTLY:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=r, y=g_r,
            mode='lines',
            name='g(r)',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_hline(y=1, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title='Radial Distribution Function',
            xaxis_title='r (Å)',
            yaxis_title='g(r)',
            template=self.themes['light']
        )
        
        return fig
    
    def create_histogram(self, data, bins=50, title='Distribution'):
        """创建直方图"""
        if not HAS_PLOTLY:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=bins,
            opacity=0.7,
            marker_color='blue'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Value',
            yaxis_title='Count',
            template=self.themes['light']
        )
        
        return fig
    
    def create_dashboard_figure(self, data):
        """创建综合仪表板图形"""
        if not HAS_PLOTLY:
            return None
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Temperature', 'Pressure', 'Energy',
                'Volume', 'Density', 'Phase Space'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"type": "scatter"}]
            ]
        )
        
        # 温度
        fig.add_trace(
            go.Scatter(x=data['time'], y=data['temperature'], 
                      name='Temperature', line=dict(color='red')),
            row=1, col=1
        )
        
        # 压力
        fig.add_trace(
            go.Scatter(x=data['time'], y=data['pressure'], 
                      name='Pressure', line=dict(color='blue')),
            row=1, col=2
        )
        
        # 能量
        fig.add_trace(
            go.Scatter(x=data['time'], y=data['energy'], 
                      name='Energy', line=dict(color='green')),
            row=1, col=3
        )
        
        # 体积
        fig.add_trace(
            go.Scatter(x=data['time'], y=data['volume'], 
                      name='Volume', line=dict(color='purple')),
            row=2, col=1
        )
        
        # 密度
        fig.add_trace(
            go.Scatter(x=data['time'], y=data['density'], 
                      name='Density', line=dict(color='orange')),
            row=2, col=2
        )
        
        # 相空间 (温度 vs 密度)
        fig.add_trace(
            go.Scatter(x=data['temperature'], y=data['density'], 
                      mode='markers',
                      marker=dict(color=data['time'], colorscale='Viridis'),
                      name='Phase Space'),
            row=2, col=3
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="MD Simulation Dashboard",
            template=self.themes['light']
        )
        
        return fig


class DashDashboard:
    """Dash仪表板应用"""
    
    def __init__(self):
        if not HAS_DASH:
            print("[ERROR] Dash未安装")
            return
        
        self.app = dash.Dash(__name__)
        self.data_generator = MDDataGenerator()
        self.data_buffer = deque(maxlen=1000)
        
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """设置布局"""
        self.app.layout = html.Div([
            html.H1('MD Simulation Real-time Dashboard', 
                   style={'textAlign': 'center', 'color': '#2c3e50'}),
            
            # 控制面板
            html.Div([
                html.Div([
                    html.H3('Simulation Controls'),
                    html.Button('Start', id='start-btn', n_clicks=0),
                    html.Button('Pause', id='pause-btn', n_clicks=0),
                    html.Button('Reset', id='reset-btn', n_clicks=0),
                    
                    html.Br(), html.Br(),
                    
                    html.Label('Update Interval (ms):'),
                    dcc.Slider(
                        id='interval-slider',
                        min=100, max=2000, step=100, value=500,
                        marks={i: str(i) for i in range(100, 2001, 500)}
                    ),
                    
                    html.Br(),
                    
                    html.Label('Display Window:'),
                    dcc.Dropdown(
                        id='window-dropdown',
                        options=[
                            {'label': '100 points', 'value': 100},
                            {'label': '500 points', 'value': 500},
                            {'label': '1000 points', 'value': 1000}
                        ],
                        value=500
                    )
                ], style={'width': '20%', 'display': 'inline-block', 'padding': '20px'}),
                
                # 实时指标
                html.Div([
                    html.H3('Current Status'),
                    html.Div(id='temperature-display', 
                            children='Temperature: -- K',
                            style={'fontSize': 20, 'color': 'red'}),
                    html.Div(id='pressure-display', 
                            children='Pressure: -- bar',
                            style={'fontSize': 20, 'color': 'blue'}),
                    html.Div(id='energy-display', 
                            children='Energy: -- eV',
                            style={'fontSize': 20, 'color': 'green'})
                ], style={'width': '20%', 'display': 'inline-block', 'padding': '20px'})
            ]),
            
            # 图表区域
            html.Div([
                dcc.Tabs([
                    dcc.Tab(label='Time Series', children=[
                        dcc.Graph(id='timeseries-graph', style={'height': '600px'})
                    ]),
                    dcc.Tab(label='Phase Space', children=[
                        dcc.Graph(id='phasespace-graph', style={'height': '600px'})
                    ]),
                    dcc.Tab(label='Distribution', children=[
                        dcc.Graph(id='distribution-graph', style={'height': '600px'})
                    ])
                ])
            ]),
            
            # 更新间隔
            dcc.Interval(id='interval-component', interval=500, n_intervals=0),
            
            # 存储状态
            dcc.Store(id='running-state', data=False)
        ])
    
    def _setup_callbacks(self):
        """设置回调函数"""
        
        @self.app.callback(
            Output('running-state', 'data'),
            Output('interval-component', 'interval'),
            Input('start-btn', 'n_clicks'),
            Input('pause-btn', 'n_clicks'),
            Input('interval-slider', 'value')
        )
        def control_simulation(start, pause, interval):
            ctx = dash.callback_context
            if not ctx.triggered:
                return False, interval
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'start-btn':
                return True, interval
            elif button_id == 'pause-btn':
                return False, interval
            else:
                return dash.no_update, interval
        
        @self.app.callback(
            Output('timeseries-graph', 'figure'),
            Output('phasespace-graph', 'figure'),
            Output('distribution-graph', 'figure'),
            Output('temperature-display', 'children'),
            Output('pressure-display', 'children'),
            Output('energy-display', 'children'),
            Input('interval-component', 'n_intervals'),
            Input('window-dropdown', 'value'),
            State('running-state', 'data')
        )
        def update_graphs(n, window, running):
            if running:
                # 生成新数据
                new_data = self.data_generator.get_next_frame()
                self.data_buffer.append(new_data)
            
            if len(self.data_buffer) == 0:
                return {}, {}, {}, 'Temperature: -- K', 'Pressure: -- bar', 'Energy: -- eV'
            
            # 转换为DataFrame
            df = pd.DataFrame(list(self.data_buffer))
            
            # 限制显示窗口
            if len(df) > window:
                df = df.iloc[-window:]
            
            # 创建图表
            visualizer = PlotlyVisualizer()
            
            # 时间序列图
            timeseries_fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                subplot_titles=('Temperature (K)', 'Pressure (bar)', 'Energy (eV)')
            )
            
            timeseries_fig.add_trace(
                go.Scatter(x=df['time'], y=df['temperature'], 
                          line=dict(color='red')),
                row=1, col=1
            )
            timeseries_fig.add_trace(
                go.Scatter(x=df['time'], y=df['pressure'], 
                          line=dict(color='blue')),
                row=2, col=1
            )
            timeseries_fig.add_trace(
                go.Scatter(x=df['time'], y=df['energy'], 
                          line=dict(color='green')),
                row=3, col=1
            )
            
            timeseries_fig.update_layout(height=600, showlegend=False)
            
            # 相空间图
            phasespace_fig = go.Figure(data=go.Scatter(
                x=df['temperature'], 
                y=df['pressure'],
                mode='markers+lines',
                marker=dict(color=df['time'], colorscale='Viridis', showscale=True)
            ))
            phasespace_fig.update_layout(
                xaxis_title='Temperature (K)',
                yaxis_title='Pressure (bar)'
            )
            
            # 分布图
            distribution_fig = make_subplots(rows=1, cols=3, 
                                            subplot_titles=('Temperature', 'Pressure', 'Energy'))
            distribution_fig.add_trace(go.Histogram(x=df['temperature']), row=1, col=1)
            distribution_fig.add_trace(go.Histogram(x=df['pressure']), row=1, col=2)
            distribution_fig.add_trace(go.Histogram(x=df['energy']), row=1, col=3)
            
            # 当前值
            latest = df.iloc[-1]
            temp_str = f"Temperature: {latest['temperature']:.2f} K"
            press_str = f"Pressure: {latest['pressure']:.3f} bar"
            energy_str = f"Energy: {latest['energy']:.2f} eV"
            
            return timeseries_fig, phasespace_fig, distribution_fig, temp_str, press_str, energy_str
    
    def run(self, debug=True, port=8050):
        """运行Dash应用"""
        if not HAS_DASH:
            print("[ERROR] Dash未安装，无法启动仪表板")
            return
        
        print(f"[INFO] 启动Dash仪表板: http://localhost:{port}")
        self.app.run_server(debug=debug, port=port, host='0.0.0.0')


class StreamlitDashboard:
    """Streamlit仪表板"""
    
    def __init__(self):
        self.data_generator = MDDataGenerator()
        self.data_buffer = deque(maxlen=1000)
    
    def run(self):
        """运行Streamlit应用"""
        if not HAS_STREAMLIT:
            print("[ERROR] Streamlit未安装")
            return
        
        st.set_page_config(page_title="MD Analysis", layout="wide")
        
        # 标题
        st.title('🔬 Molecular Dynamics Analysis Dashboard')
        
        # 侧边栏
        st.sidebar.header('Simulation Settings')
        
        if st.sidebar.button('Generate Data'):
            for _ in range(100):
                self.data_buffer.append(self.data_generator.get_next_frame())
        
        if st.sidebar.button('Clear Data'):
            self.data_buffer.clear()
        
        analysis_type = st.sidebar.selectbox(
            'Analysis Type',
            ['Real-time Monitoring', 'RDF Analysis', 'MSD Analysis', 'Structure Analysis']
        )
        
        # 主面板
        if len(self.data_buffer) == 0:
            st.info('Click "Generate Data" to start')
            return
        
        df = pd.DataFrame(list(self.data_buffer))
        
        if analysis_type == 'Real-time Monitoring':
            self._show_realtime(df)
        elif analysis_type == 'RDF Analysis':
            self._show_rdf()
        elif analysis_type == 'MSD Analysis':
            self._show_msd()
        elif analysis_type == 'Structure Analysis':
            self._show_structure()
    
    def _show_realtime(self, df):
        """显示实时监控"""
        # 指标卡片
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric('Temperature', f"{df['temperature'].iloc[-1]:.2f} K", 
                     f"{df['temperature'].iloc[-1] - df['temperature'].iloc[-2]:.2f}")
        with col2:
            st.metric('Pressure', f"{df['pressure'].iloc[-1]:.3f} bar",
                     f"{df['pressure'].iloc[-1] - df['pressure'].iloc[-2]:.3f}")
        with col3:
            st.metric('Energy', f"{df['energy'].iloc[-1]:.2f} eV",
                     f"{df['energy'].iloc[-1] - df['energy'].iloc[-2]:.2f}")
        with col4:
            st.metric('Density', f"{df['density'].iloc[-1]:.4f} g/cm³")
        
        # 图表
        st.subheader('Time Series')
        col1, col2 = st.columns(2)
        
        with col1:
            st.line_chart(df[['time', 'temperature']].set_index('time'))
            st.line_chart(df[['time', 'pressure']].set_index('time'))
        
        with col2:
            st.line_chart(df[['time', 'energy']].set_index('time'))
            st.line_chart(df[['time', 'density']].set_index('time'))
        
        # 数据表
        st.subheader('Raw Data')
        st.dataframe(df.tail(20))
        
        # 下载按钮
        csv = df.to_csv(index=False)
        st.download_button('Download CSV', csv, 'md_data.csv', 'text/csv')
    
    def _show_rdf(self):
        """显示RDF分析"""
        st.subheader('Radial Distribution Function')
        
        # 模拟RDF数据
        r = np.linspace(0, 10, 500)
        g_r = 1 + np.exp(-(r-3)**2/0.5) + 0.5*np.exp(-(r-5.5)**2/1)
        
        rdf_df = pd.DataFrame({'r': r, 'g(r)': g_r})
        st.line_chart(rdf_df.set_index('r'))
        
        st.write('Peak positions: 3.0 Å, 5.5 Å')
    
    def _show_msd(self):
        """显示MSD分析"""
        st.subheader('Mean Square Displacement')
        
        # 模拟MSD数据
        t = np.linspace(0, 100, 100)
        msd = 4 * t + np.random.normal(0, 5, 100)
        
        msd_df = pd.DataFrame({'time': t, 'MSD': msd})
        st.line_chart(msd_df.set_index('time'))
        
        # 扩散系数
        D = np.polyfit(t, msd, 1)[0] / 6
        st.write(f'Diffusion coefficient: {D:.4e} cm²/s')
    
    def _show_structure(self):
        """显示结构分析"""
        st.subheader('Molecular Structure')
        st.info('3D visualization would be shown here with py3dmol or nglview')


# 独立可视化示例
if __name__ == '__main__':
    print("=== 交互式仪表板模块 ===\n")
    
    # 创建示例数据
    data_gen = MDDataGenerator()
    data = []
    for _ in range(500):
        data.append(data_gen.get_next_frame())
    df = pd.DataFrame(data)
    
    # Plotly可视化示例
    print("1. 创建Plotly可视化...")
    visualizer = PlotlyVisualizer()
    
    # 3D轨迹图
    positions = np.random.randn(100, 3) * 5
    fig = visualizer.create_trajectory_plot(positions)
    if fig:
        fig.write_html('trajectory_3d.html')
        print("   - 3D轨迹图: trajectory_3d.html")
    
    # 时间序列
    data_dict = {
        'Temperature': df[['time', 'temperature']],
        'Pressure': df[['time', 'pressure']],
        'Energy': df[['time', 'energy']]
    }
    fig = visualizer.create_timeseries_plot(data_dict)
    if fig:
        fig.write_html('timeseries.html')
        print("   - 时间序列: timeseries.html")
    
    # 仪表板
    fig = visualizer.create_dashboard_figure(df)
    if fig:
        fig.write_html('dashboard.html')
        print("   - 仪表板: dashboard.html")
    
    print("\n2. 启动Dash仪表板...")
    print("   运行: python -c \"from dashboard import DashDashboard; DashDashboard().run()\"")
    
    print("\n3. 启动Streamlit仪表板...")
    print("   运行: streamlit run dashboard.py")
