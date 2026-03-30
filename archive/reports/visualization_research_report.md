# 可视化与后处理工具研究报告
# 启动时间: 2026-03-08 16:31 GMT+8

---

## 模块1: OVITO高级分析 (进行中)

### 1.1 核心功能概述
OVITO (Open Visualization Tool) 是一个用于原子和分子模拟数据可视化和分析的科学软件。

#### 安装
```bash
pip install ovito
# 或conda
conda install -c conda-forge ovito
```

#### Python API基础
```python
from ovito.io import import_file
from ovito.modifiers import *
from ovito.vis import *

# 加载LAMMPS dump文件
pipeline = import_file("trajectory.dump")

# 添加修改器
pipeline.modifiers.append(ComputePropertyModifier(output_property='Color', expressions=['Position.X / 10']))
```

### 1.2 缺陷识别技术

#### 位错分析 (Dislocation Analysis - DXA)
```python
from ovito.modifiers import DislocationAnalysisModifier, CommonNeighborAnalysisModifier

# 晶体结构识别
pipeline.modifiers.append(CommonNeighborAnalysisModifier())

# 位错分析
dxa = DislocationAnalysisModifier()
dxa.input_crystal_structure = DislocationAnalysisModifier.CrystalType.FCC
dxa.output_filename = "dislocations.txt"
pipeline.modifiers.append(dxa)
```

#### 空位和间隙识别
```python
from ovito.modifiers import IdentifyDiamondModifier, ClusterAnalysisModifier

# 缺陷识别
pipeline.modifiers.append(IdentifyDiamondModifier())

# 团簇分析
cluster = ClusterAnalysisModifier(cutoff=3.5)
pipeline.modifiers.append(cluster)

# 获取数据
output = pipeline.compute()
defects = output.attributes['IdentifyDiamond.counts']
```

#### Wigner-Seitz缺陷分析
```python
from ovito.modifiers import WignerSeitzAnalysisModifier

# Wigner-Seitz分析
ws = WignerSeitzAnalysisModifier()
ws.reference = import_file("reference.crystal")
ws.eliminate_cell_deformation = True
pipeline.modifiers.append(ws)

# 输出结果
output = pipeline.compute()
vacancies = output.attributes['WignerSeitz.vacancy_count']
interstitials = output.attributes['WignerSeitz.interstitial_count']
```

### 1.3 晶体结构分析

#### 常见邻居分析 (CNA)
```python
from ovito.modifiers import CommonNeighborAnalysisModifier

cna = CommonNeighborAnalysisModifier()
cna.mode = CommonNeighborAnalysisModifier.Mode.FixedCutoff
cna.cutoff = 3.5
pipeline.modifiers.append(cna)

# 获取结构类型
output = pipeline.compute()
structure_types = output.particles['Structure Type']
# 0=其他, 1=FCC, 2=HCP, 3=BCC, 4=ICO
fcc_count = np.sum(structure_types == 1)
```

#### 径向分布函数 (RDF)
```python
from ovito.modifiers import CoordinationAnalysisModifier

rdf = CoordinationAnalysisModifier(
    cutoff=10.0,
    number_of_bins=100
)
pipeline.modifiers.append(rdf)

# 获取RDF数据
output = pipeline.compute()
rdf_table = output.tables['coordination-rdf'].xy()
r = rdf_table[:, 0]  # 距离
g_r = rdf_table[:, 1]  # g(r)
```

### 1.4 局部结构分析

#### Voronoi分析
```python
from ovito.modifiers import VoronoiAnalysisModifier

voronoi = VoronoiAnalysisModifier(
    compute_indices=True,
    use_radii=False,
    edge_threshold=0.1
)
pipeline.modifiers.append(voronoi)

# 分析局部配位
output = pipeline.compute()
volume = output.particles['Atomic Volume']
indices = output.particles['Voronoi Index']
```

#### 局部分子取向分析
```python
from ovito.modifiers import CalculateLocalStructureModifier

# 取向张量分析
local_structure = CalculateLocalStructureModifier()
pipeline.modifiers.append(local_structure)

output = pipeline.compute()
orientations = output.particles['Orientation']
```

### 1.5 高级渲染和输出
```python
from ovito.vis import Viewport, RenderSettings

# 创建渲染设置
settings = RenderSettings(
    filename="output.png",
    size=(1920, 1080),
    background_color=(1,1,1)
)

# 渲染
viewport = Viewport(type=Viewport.Type.Ortho)
viewport.render(settings)
```

### 1.6 批处理分析脚本示例
```python
import os
from ovito.io import import_file
from ovito.modifiers import *

# 批量处理多个dump文件
for i in range(100):
    filename = f"dump.{i}0000"
    if not os.path.exists(filename):
        continue
    
    pipeline = import_file(filename)
    
    # 添加分析链
    pipeline.modifiers.append(CommonNeighborAnalysisModifier())
    pipeline.modifiers.append(DislocationAnalysisModifier())
    
    # 导出结果
    export_file(pipeline, f"analysis_{i}.txt", "txt",
                columns=["Particle Identifier", "Position.X", "Position.Y", 
                        "Position.Z", "Structure Type"])
```

---

## 模块2: VMD分子可视化 (进行中)

### 2.1 核心功能与安装
```bash
# Ubuntu/Debian
sudo apt-get install vmd

# 或从官网下载
# https://www.ks.uiuc.edu/Research/vmd/
```

### 2.2 Python接口 - VMD Python API
```python
# 通过vmd-python包
import molecule
import atomsel
import vmdnumpy as vnp

# 加载分子结构
molid = molecule.load('pdb', 'protein.pdb')
molid2 = molecule.load('dcd', 'trajectory.dcd')

# 选择原子
sel = atomsel.atomsel('protein and resid 1 to 50')
ca_atoms = atomsel.atomsel('name CA')
```

### 2.3 生物分子可视化
```python
# 二级结构分析
import secondary_structure

# 自动识别二级结构
secondary_structure.ssrecalc(molid)

# 获取二级结构
ss = vnp.atomselect(molid, 'all', 'structure')
# H=helix, E=sheet, T=turn, C=coil
```

#### 蛋白质表面分析
```tcl
# VMD Tcl脚本示例
mol new protein.pdb
mol addfile trajectory.dcd type dcd first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all

# 显示表面
mol representation Surf 1.4 0
mol color Name
mol selection "protein"
mol addrep top

# 测量RMSD
set ref [atomselect top "protein" frame 0]
set comp [atomselect top "protein"]
set num_frames [molinfo top get numframes]

for {set i 0} {$i < $num_frames} {incr i} {
    $comp frame $i
    set rmsd [measure rmsd $comp $ref]
    puts "$i $rmsd"
}
```

### 2.4 聚合物轨迹分析
```python
# 回转半径分析
import numpy as np

def radius_of_gyration(molid, selection='all'):
    """计算回转半径"""
    sel = atomsel.atomsel(selection, molid=molid)
    coords = np.array(sel.get('coords'))
    masses = np.array(sel.get('mass'))
    
    # 质心
    center = np.average(coords, axis=0, weights=masses)
    
    # 回转半径
    Rg = np.sqrt(np.average(np.sum((coords - center)**2, axis=1), weights=masses))
    return Rg

# 分析整个轨迹
n_frames = molecule.numframes(molid)
rg_values = []

for frame in range(n_frames):
    molecule.set_frame(molid, frame)
    rg = radius_of_gyration(molid, 'polymer')
    rg_values.append(rg)
```

#### 端到端距离分析
```python
def end_to_end_distance(molid, chain_id=0):
    """计算聚合物端到端距离"""
    sel_start = atomsel.atomsel(f'resid 1 and segid P{chain_id}')
    sel_end = atomsel.atomsel(f'resid 100 and segid P{chain_id}')
    
    start_coords = np.array(sel_start.get('coords'))[0]
    end_coords = np.array(sel_end.get('coords'))[0]
    
    distance = np.linalg.norm(end_coords - start_coords)
    return distance
```

### 2.5 氢键分析
```python
# VMD氢键分析
import hbonds

# 查找氢键
hb = hbonds.HBonds()
hb.set_mol(molid)
hb.find_hbonds()

# 获取结果
hbonds_data = hb.get_hbonds()
```

### 2.6 大规模轨迹分析
```python
# 使用MDAnalysis作为VMD的补充
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align

# 加载轨迹
u = mda.Universe('system.psf', 'trajectory.dcd')

# RMSD分析
ref = u.select_atoms('protein')
mobile = u.select_atoms('protein')

R = rms.RMSD(mobile, ref, select='backbone', ref_frame=0)
R.run()

# 输出结果
print(R.results.rmsd)
```

### 2.7 VMD渲染设置
```tcl
# 高质量渲染
render TachyonInternal scene.dat -aasamples 12 -res 3840 2160
exec ./tachyon_LINUXAMD64 scene.dat -format TARGA -o output.tga -res 3840 2160 -aasamples 12
```

---

## 模块3: Pymatgen结构分析 (进行中)

### 3.1 安装与基础
```bash
pip install pymatgen
# 完整安装
pip install pymatgen[optional]
```

### 3.2 晶体结构处理
```python
from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar, Vasprun
from pymatgen.analysis.structure_matcher import StructureMatcher

# 从POSCAR加载
structure = Structure.from_file('POSCAR')

# 创建结构
lattice = Lattice.cubic(4.2)
structure = Structure(lattice, ['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]])

# 结构匹配
matcher = StructureMatcher()
are_same = matcher.fit(struct1, struct2)
```

### 3.3 能带结构分析
```python
from pymatgen.io.vasp import Vasprun
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.electronic_structure.bandstructure import BandStructure

# 读取VASP计算结果
vasprun = Vasprun('vasprun.xml')
bs = vasprun.get_band_structure(line_mode=True)

# 绘制能带结构
plotter = BSPlotter(bs)
plotter.get_plot(vbm_cbm_marker=True).show()

# 获取带隙
eg = bs.get_band_gap()
print(f"Band gap: {eg['energy']} eV")
print(f"Direct: {eg['direct']}")

# 有效质量计算
from pymatgen.electronic_structure.effective_mass import get_fitting_data

vbm_data = get_fitting_data(bs, 'p', 0.001)  # 空穴
print(f"Hole effective mass: {vbm_data['effective_mass']}")
```

### 3.4 态密度 (DOS) 分析
```python
from pymatgen.electronic_structure.plotter import DosPlotter
from pymatgen.electronic_structure.dos import CompleteDos

# 读取DOS数据
dos = vasprun.complete_dos

# 绘制DOS
plotter = DosPlotter()
plotter.add_dos("Total", dos)
plotter.add_dos("O-p", dos.get_spd_dos()['p'])
plotter.add_dos("Ti-d", dos.get_element_spd_dos('Ti')['d'])
plotter.get_plot().show()

# 投影态密度
pdos = dos.get_site_orbital_dos(structure[0], 'd')
```

### 3.5 相图分析
```python
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen.ext.matproj import MPRester

# 从Materials Project获取数据
with MPRester("YOUR_API_KEY") as mpr:
    entries = mpr.get_entries_in_chemsys(['Li', 'Fe', 'O'])

# 构建相图
pd = PhaseDiagram(entries)

# 绘制相图
plotter = PDPlotter(pd)
plotter.get_plot().show()

# 稳定性分析
decomp_energy = pd.get_decomp_and_e_above_hull(entry)
print(f"Hull energy: {decomp_energy[1]} eV/atom")
```

### 3.6 材料性质计算
```python
from pymatgen.analysis.elasticity import ElasticTensor
from pymatgen.analysis.diffusion_analyzer import DiffusionAnalyzer

# 弹性张量
elastic_tensor = ElasticTensor.from_vasp_voigt('ELASTIC')
bulk_modulus = elastic_tensor.k_voigt
shear_modulus = elastic_tensor.g_voigt
youngs_modulus = elastic_tensor.y_mod

# 扩散分析
diffusion_analyzer = DiffusionAnalyzer.from_vaspruns(
    ['vasprun_1.xml', 'vasprun_2.xml'],
    specie='Li',
    smoothed=False
)
print(f"Diffusion coefficient: {diffusion_analyzer.diffusivity}")
```

### 3.7 高通量计算工作流
```python
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
from pymatgen.ext.matproj import MPRester

# 创建VASP输入
structure = Structure.from_file('POSCAR')
relax_set = MPRelaxSet(structure)
relax_set.write_input('relax_calc')

# 静态计算
static_set = MPStaticSet(structure)
static_set.write_input('static_calc')
```

### 3.8 晶体结构可视化
```python
from pymatgen.vis.structure_vtk import StructureVis

# 3D可视化
vis = StructureVis()
vis.set_structure(structure)
vis.show()
```

---

## 模块4: 交互式仪表板 (进行中)

### 4.1 Plotly基础
```python
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# 3D散点图 - 原子位置
fig = go.Figure(data=[go.Scatter3d(
    x=atoms_x, y=atoms_y, z=atoms_z,
    mode='markers',
    marker=dict(
        size=5,
        color=temperature,
        colorscale='Viridis',
        opacity=0.8
    ),
    text=atom_labels,
    hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>T: %{marker.color:.1f}K<extra></extra>'
)])

fig.update_layout(
    title='Molecular Dynamics Trajectory',
    scene=dict(
        xaxis_title='X (Å)',
        yaxis_title='Y (Å)',
        zaxis_title='Z (Å)',
        aspectmode='data'
    )
)
fig.show()
```

### 4.2 实时数据可视化
```python
import plotly.graph_objects as go
from collections import deque
import time

# 实时轨迹监控
class RealTimeTrajectoryMonitor:
    def __init__(self, max_points=1000):
        self.fig = go.FigureWidget()
        self.energy_history = deque(maxlen=max_points)
        self.temp_history = deque(maxlen=max_points)
        self.time_history = deque(maxlen=max_points)
        
    def add_frame(self, energy, temperature, time_step):
        self.energy_history.append(energy)
        self.temp_history.append(temperature)
        self.time_history.append(time_step)
        
        with self.fig.batch_update():
            self.fig.data[0].y = list(self.energy_history)
            self.fig.data[0].x = list(self.time_history)
            self.fig.data[1].y = list(self.temp_history)
            self.fig.data[1].x = list(self.time_history)

# 使用示例
monitor = RealTimeTrajectoryMonitor()
```

### 4.3 Dash仪表板
```python
import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('MD Simulation Dashboard'),
    
    # 控制面板
    html.Div([
        dcc.Dropdown(
            id='property-selector',
            options=[
                {'label': 'Temperature', 'value': 'temp'},
                {'label': 'Energy', 'value': 'energy'},
                {'label': 'Pressure', 'value': 'pressure'},
                {'label': 'Density', 'value': 'density'}
            ],
            value='temp'
        ),
        dcc.Slider(
            id='frame-slider',
            min=0,
            max=1000,
            step=1,
            value=0,
            marks={i: str(i) for i in range(0, 1001, 100)}
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),
    
    # 图表区域
    html.Div([
        dcc.Graph(id='trajectory-plot'),
        dcc.Graph(id='property-plot'),
    ]),
    
    # 自动更新
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
])

@callback(
    Output('trajectory-plot', 'figure'),
    Output('property-plot', 'figure'),
    Input('frame-slider', 'value'),
    Input('property-selector', 'value'),
    Input('interval-component', 'n_intervals')
)
def update_plots(frame, property_name, n):
    # 更新轨迹图
    traj_fig = create_trajectory_figure(frame)
    
    # 更新属性图
    prop_fig = create_property_timeseries(property_name)
    
    return traj_fig, prop_fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
```

### 4.4 交互式分子查看器
```python
import dash_bio as dashbio

# 3D分子可视化
app.layout = html.Div([
    dashbio.Molecule3dViewer(
        id='mol-viewer',
        modelData={
            'atoms': [
                {'name': 'CA', 'coords': [0, 0, 0], 'element': 'C', 'residue_index': 1},
                {'name': 'N', 'coords': [1.5, 0, 0], 'element': 'N', 'residue_index': 1},
                # ...
            ],
            'bonds': [
                {'atom1_index': 0, 'atom2_index': 1},
                # ...
            ]
        },
        styles={
            'stick': {'radius': 0.2},
            'sphere': {'scale': 0.3}
        }
    )
])
```

### 4.5 Streamlit快速仪表板
```python
import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="MD Analysis", layout="wide")

# 侧边栏
st.sidebar.header("Simulation Controls")
trajectory_file = st.sidebar.file_uploader("Upload Trajectory", type=['dcd', 'xtc'])
analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    ["RDF", "MSD", "RMSF", "Hydrogen Bonds"]
)

# 主面板
col1, col2 = st.columns(2)

with col1:
    st.subheader("Trajectory View")
    # 3D轨迹可视化
    st.plotly_chart(create_trajectory_plot(), use_container_width=True)

with col2:
    st.subheader("Analysis Results")
    if analysis_type == "RDF":
        rdf_data = calculate_rdf(trajectory_file)
        fig = px.line(rdf_data, x='r', y='g_r', title='Radial Distribution Function')
        st.plotly_chart(fig)
    elif analysis_type == "MSD":
        msd_data = calculate_msd(trajectory_file)
        fig = px.scatter(msd_data, x='time', y='msd', title='Mean Square Displacement')
        st.plotly_chart(fig)
```

### 4.6 实时监控工作流
```python
import asyncio
import websockets
import json
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# WebSocket实时数据接收
class RealTimeDataReceiver:
    def __init__(self):
        self.data_buffer = []
        
    async def connect(self, uri):
        async with websockets.connect(uri) as websocket:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                self.data_buffer.append(data)
                
# Dash应用集成实时数据
app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='live-graph'),
    dcc.Interval(id='interval', interval=1000)
])

@app.callback(
    Output('live-graph', 'figure'),
    Input('interval', 'n_intervals')
)
def update_graph(n):
    # 从缓冲区获取最新数据
    df = pd.DataFrame(receiver.data_buffer[-1000:])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['temperature'],
        mode='lines', name='Temperature'
    ))
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['pressure'],
        mode='lines', name='Pressure'
    ))
    
    return fig
```

---

## 模块5: 3D可视化 (进行中)

### 5.1 PyVista基础
```python
import pyvista as pv
import numpy as np

# 创建点云
points = np.random.rand(1000, 3) * 10
point_cloud = pv.PolyData(points)

# 添加标量场
point_cloud['Temperature'] = np.random.rand(1000) * 300 + 273

# 可视化
plotter = pv.Plotter()
plotter.add_mesh(point_cloud, render_points_as_spheres=True, 
                 point_size=10, cmap='viridis')
plotter.add_scalar_bar(title='Temperature (K)')
plotter.show()
```

### 5.2 分子轨迹可视化
```python
import pyvista as pv
from pyvista import themes

pv.set_plot_theme(themes.DocumentTheme())

def visualize_trajectory(trajectory_frames, atomic_numbers):
    """可视化MD轨迹"""
    plotter = pv.Plotter(window_size=[1920, 1080])
    
    # 原子颜色映射（CPK颜色）
    cpk_colors = {
        1: 'white',   # H
        6: 'gray',    # C
        7: 'blue',    # N
        8: 'red',     # O
        16: 'yellow', # S
    }
    
    # 原子半径映射（范德华半径，单位Å）
    vdw_radii = {
        1: 1.2, 6: 1.7, 7: 1.55, 8: 1.52, 16: 1.8
    }
    
    for i, frame in enumerate(trajectory_frames):
        for j, (pos, z) in enumerate(zip(frame, atomic_numbers)):
            sphere = pv.Sphere(radius=vdw_radii.get(z, 1.5), center=pos)
            color = cpk_colors.get(z, 'green')
            plotter.add_mesh(sphere, color=color, smooth_shading=True)
    
    plotter.add_axes()
    plotter.add_bounding_box()
    plotter.show()

# 使用示例
# visualize_trajectory(trajectory_data, atomic_nums)
```

### 5.3 体积数据可视化
```python
import pyvista as pv

# 电子密度可视化
def plot_electron_density(cube_file):
    """从Gaussian Cube文件可视化电子密度"""
    grid = pv.UniformGrid()
    # 加载cube数据...
    
    # 创建体积渲染
    plotter = pv.Plotter()
    
    # 等值面
    contours = grid.contour([0.05, 0.1, 0.2])
    plotter.add_mesh(contours, cmap='viridis', opacity=0.5)
    
    # 体积渲染
    plotter.add_volume(grid, cmap='plasma', opacity='sigmoid')
    
    plotter.show()

# 电荷密度可视化
def plot_charge_density(structure, charge_data):
    """可视化电荷密度分布"""
    # 创建网格
    x = np.linspace(0, structure.lattice.a, 50)
    y = np.linspace(0, structure.lattice.b, 50)
    z = np.linspace(0, structure.lattice.c, 50)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 插值电荷数据
    grid = pv.StructuredGrid(X, Y, Z)
    grid['charge'] = charge_data.flatten()
    
    # 切片可视化
    slices = grid.slice_orthogonal()
    
    plotter = pv.Plotter()
    plotter.add_mesh(slices, cmap='RdBu_r', clim=[-1, 1])
    plotter.show()
```

### 5.4 Blender分子渲染
```python
import bpy
import numpy as np

# 清除场景
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# 原子属性
ATOM_PROPERTIES = {
    'H': {'radius': 0.5, 'color': (1, 1, 1)},
    'C': {'radius': 0.7, 'color': (0.3, 0.3, 0.3)},
    'N': {'radius': 0.65, 'color': (0, 0, 0.8)},
    'O': {'radius': 0.6, 'color': (0.8, 0, 0)},
    'S': {'radius': 1.0, 'color': (1, 0.8, 0)},
}

def create_atom_element(name, radius, color):
    """创建原子材质和网格"""
    # 创建材质
    mat = bpy.data.materials.new(name=f"{name}_material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Roughness'].default_value = 0.2
    bsdf.inputs['Metallic'].default_value = 0.1
    
    # 创建球体
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=32, ring_count=16)
    atom = bpy.context.active_object
    atom.data.materials.append(mat)
    
    return atom

def create_bond(start, end, radius=0.15):
    """创建化学键"""
    # 计算位置和旋转
    direction = np.array(end) - np.array(start)
    length = np.linalg.norm(direction)
    mid_point = (np.array(start) + np.array(end)) / 2
    
    # 创建圆柱体
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length)
    bond = bpy.context.active_object
    bond.location = tuple(mid_point)
    
    # 旋转对齐
    rot_quat = direction.to_track_quat('Z', 'Y')
    bond.rotation_euler = rot_quat.to_euler()
    
    return bond

def render_molecule_pdb(pdb_file, output_path="molecule_render.png"):
    """渲染PDB分子结构"""
    import Bio.PDB
    
    parser = Bio.PDB.PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    
    # 创建原子
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    element = atom.element
                    coord = atom.coord
                    
                    if element in ATOM_PROPERTIES:
                        props = ATOM_PROPERTIES[element]
                        atom_obj = create_atom_element(element, props['radius'], props['color'])
                        atom_obj.location = tuple(coord / 10)  # 转换为Blender单位
    
    # 设置相机
    bpy.ops.object.camera_add(location=(10, -10, 5))
    camera = bpy.context.active_object
    camera.rotation_euler = (1.1, 0, 0.785)
    bpy.context.scene.camera = camera
    
    # 设置灯光
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    light = bpy.context.active_object
    light.data.energy = 3
    
    # 渲染设置
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 3840
    scene.render.resolution_y = 2160
    scene.render.resolution_percentage = 100
    scene.render.filepath = output_path
    
    # 渲染
    bpy.ops.render.render(write_still=True)
```

### 5.5 高级PyVista可视化
```python
import pyvista as pv
import numpy as np

# 创建动画轨迹
plotter = pv.Plotter(off_screen=True)
plotter.open_movie('trajectory.mp4', framerate=30)

# 为每一帧生成图像
for frame_idx in range(n_frames):
    plotter.clear()
    
    # 添加原子
    positions = trajectory[frame_idx]
    for pos, element in zip(positions, elements):
        sphere = pv.Sphere(radius=atomic_radii[element], center=pos)
        plotter.add_mesh(sphere, color=cpk_colors[element], smooth_shading=True)
    
    # 添加单元格边界
    box = pv.Box(bounds=[0, 10, 0, 10, 0, 10])
    plotter.add_mesh(box, style='wireframe', color='gray')
    
    # 添加时间标签
    plotter.add_text(f"Frame: {frame_idx}", position='upper_left', font_size=20)
    
    plotter.write_frame()

plotter.close()

# 流线可视化（用于流场）
def plot_streamlines(velocity_field, seeds):
    """可视化速度场流线"""
    plotter = pv.Plotter()
    
    # 创建流线
    streamlines = velocity_field.streamlines(
        vectors='velocity',
        source_radius=5,
        n_points=100,
        integration_direction='both'
    )
    
    plotter.add_mesh(streamlines, cmap='plasma', line_width=2)
    plotter.add_mesh(velocity_field.glyph(orient='velocity', scale='velocity', factor=0.1))
    plotter.show()
```

### 5.6 多物理场可视化
```python
# 温度场+应力场联合可视化
plotter = pv.Plotter(shape=(1, 2))

# 温度场
plotter.subplot(0, 0)
plotter.add_mesh(mesh, scalars='temperature', cmap='hot', show_edges=True)
plotter.add_title('Temperature Field')

# 应力场
plotter.subplot(0, 1)
plotter.add_mesh(mesh, scalars='von_mises_stress', cmap='coolwarm', show_edges=True)
plotter.add_title('Von Mises Stress')

plotter.show()
```

---

## 综合集成示例

### 完整分析工作流
```python
# 一站式分析脚本
import ovito
from ovito.io import import_file
from ovito.modifiers import *
import pyvista as pv
import plotly.graph_objects as go
import pandas as pd

class MDAnalysisPipeline:
    """MD分析完整工作流"""
    
    def __init__(self, trajectory_file):
        self.pipeline = import_file(trajectory_file)
        self.results = {}
        
    def run_structure_analysis(self):
        """结构分析"""
        self.pipeline.modifiers.append(CommonNeighborAnalysisModifier())
        self.pipeline.modifiers.append(VoronoiAnalysisModifier())
        
    def run_defect_analysis(self):
        """缺陷分析"""
        self.pipeline.modifiers.append(DislocationAnalysisModifier())
        self.pipeline.modifiers.append(WignerSeitzAnalysisModifier())
        
    def export_results(self, output_dir):
        """导出所有结果"""
        # 数据表
        df = pd.DataFrame(self.results)
        df.to_csv(f"{output_dir}/analysis_results.csv")
        
        # 可视化
        self._create_visualizations(output_dir)
        
    def _create_visualizations(self, output_dir):
        """创建可视化"""
        # Plotly图表
        fig = make_subplots(rows=2, cols=2)
        # ... 添加轨迹
        fig.write_html(f"{output_dir}/dashboard.html")
        
        # PyVista 3D可视化
        plotter = pv.Plotter(off_screen=True)
        # ... 创建3D场景
        plotter.screenshot(f"{output_dir}/snapshot.png", scale=2)

# 使用示例
# pipeline = MDAnalysisPipeline('trajectory.dump')
# pipeline.run_structure_analysis()
# pipeline.export_results('./output')
```

---

## 研究时间线

| 时间 | 模块 | 状态 |
|------|------|------|
| 16:31-17:00 | 模块1: OVITO高级分析 | ✅ 完成 |
| 17:00-17:30 | 模块2: VMD分子可视化 | ✅ 完成 |
| 17:30-18:00 | 模块3: Pymatgen结构分析 | ✅ 完成 |
| 18:00-18:30 | 模块4: 交互式仪表板 | ✅ 完成 |
| 18:30-19:00 | 模块5: 3D可视化 | ✅ 完成 |

---

## 附录：依赖安装清单

```bash
# 核心依赖
pip install ovito pymatgen vmd-python MDAnalysis

# 可视化
pip install plotly dash dash-bio streamlit pyvista trame

# 科学计算
pip install numpy scipy pandas scikit-learn

# Blender Python API (在Blender内)
# import bpy
```

---

*报告生成时间: 2026-03-08*
*研究状态: 所有模块完成*
