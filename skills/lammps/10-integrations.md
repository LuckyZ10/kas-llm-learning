# 10. LAMMPS与其他工具集成

> LAMMPS与Python、ASE、OVITO等工具的深度集成

---

## 目录
- [Python接口](#python接口)
- [ASE集成](#ase集成)
- [OVITO编程](#ovito编程)
- [自动化工作流](#自动化工作流)
- [可视化管道](#可视化管道)

---

## Python接口

### PyLAMMPS高级用法

```python
from lammps import lammps
import numpy as np

class LAMMPSController:
    """LAMMPS高级控制器"""
    
    def __init__(self, cmdargs=None):
        self.lmp = lammps(cmdargs=cmdargs)
        self.natoms = 0
    
    def create_lattice(self, style, params, nx, ny, nz):
        """创建晶格"""
        commands = f"""
        lattice {style} {params}
        region box block 0 {nx} 0 {ny} 0 {nz}
        create_box 1 box
        create_atoms 1 box
        """
        self.lmp.commands_string(commands)
        self.natoms = self.lmp.get_natoms()
    
    def set_potential(self, pair_style, pair_coeff):
        """设置势函数"""
        self.lmp.command(f"pair_style {pair_style}")
        self.lmp.command(f"pair_coeff {pair_coeff}")
    
    def get_trajectory(self):
        """获取当前构型"""
        x = self.lmp.gather_atoms("x", 1, 3)
        return np.array(x).reshape(-1, 3)
    
    def set_positions(self, positions):
        """设置原子位置"""
        x = positions.flatten().tolist()
        self.lmp.scatter_atoms("x", 1, 3, x)
    
    def run_md(self, nsteps, thermo_freq=100):
        """运行MD"""
        self.lmp.command(f"thermo {thermo_freq}")
        self.lmp.command(f"run {nsteps}")
    
    def get_forces(self):
        """获取力"""
        f = self.lmp.gather_atoms("f", 1, 3)
        return np.array(f).reshape(-1, 3)
    
    def get_potential_energy(self):
        """获取势能"""
        return self.lmp.extract_compute("thermo_pe", 0, 0)

# 使用示例
ctrl = LAMMPSController()
ctrl.create_lattice("fcc", 3.52, 10, 10, 10)
ctrl.set_potential("eam/alloy", "* * Ni_u3.eam.alloy Ni")

# 自定义MD循环
for step in range(1000):
    ctrl.run_md(1)
    pos = ctrl.get_trajectory()
    forces = ctrl.get_forces()
    
    # 自定义分析
    if step % 100 == 0:
        print(f"Step {step}: PE = {ctrl.get_potential_energy()}")
```

### 实时可视化

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LiveLAMMPSVisualizer:
    """实时LAMMPS可视化"""
    
    def __init__(self, lmp_controller):
        self.ctrl = lmp_controller
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.scatter = None
    
    def update(self, frame):
        """更新函数"""
        # 运行一步
        self.ctrl.run_md(10)
        
        # 获取位置
        pos = self.ctrl.get_trajectory()
        
        # 更新绘图
        if self.scatter is None:
            self.scatter = self.ax.scatter(pos[:, 0], pos[:, 1], c='blue', s=50)
            self.ax.set_xlim(pos[:, 0].min() - 5, pos[:, 0].max() + 5)
            self.ax.set_ylim(pos[:, 1].min() - 5, pos[:, 1].max() + 5)
        else:
            self.scatter.set_offsets(pos[:, :2])
        
        return self.scatter,
    
    def run(self, n_frames):
        """运行动画"""
        anim = FuncAnimation(
            self.fig, self.update, frames=n_frames,
            interval=50, blit=True
        )
        plt.show()
```

---

## ASE集成

### ASE-LAMMPS接口

```python
from ase.calculators.lammpsrun import LAMMPS
from ase import Atoms
from ase.optimize import BFGS
from ase.md.langevin import Langevin
from ase import units

# 创建ASE calculator
calc = LAMMPS(
    command='lmp',
    pair_style='eam/alloy',
    pair_coeff=['* * Cu_u3.eam.alloy Cu'],
    specorder=['Cu'],
    keep_alive=True
)

# 创建结构
atoms = Atoms(
    'Cu108',
    positions=...,
    cell=[20, 20, 20],
    pbc=True
)
atoms.calc = calc

# 优化
opt = BFGS(atoms)
opt.run(fmax=0.01)

# MD
dyn = Langevin(atoms, 1*units.fs, 300*units.kB, 0.002)
dyn.run(10000)

# 分析
from ase.geometry.analysis import Analysis
analyzer = Analysis([atoms])
rdf = analyzer.get_rdf(rmax=10.0, nbins=100)
```

### 批量计算

```python
from ase.build import bulk, surface
from ase.io import write
from concurrent.futures import ProcessPoolExecutor

def calculate_bulk_modulus(element, lattice='fcc'):
    """计算体模量"""
    atoms = bulk(element, lattice, a=3.6)
    atoms.calc = LAMMPS(...)
    
    # 体积扫描
    volumes = []
    energies = []
    
    for scale in np.linspace(0.9, 1.1, 20):
        atoms_scaled = atoms.copy()
        atoms_scaled.set_cell(atoms.cell * scale, scale_atoms=True)
        volumes.append(atoms_scaled.get_volume())
        energies.append(atoms_scaled.get_potential_energy())
    
    # Birch-Murnaghan拟合
    from ase.eos import EquationOfState
    eos = EquationOfState(volumes, energies)
    v0, e0, B = eos.fit()
    
    return B / units.kJ * 1e24  # GPa

# 并行计算
with ProcessPoolExecutor(max_workers=4) as executor:
    elements = ['Cu', 'Ag', 'Au', 'Ni', 'Pd', 'Pt']
    results = executor.map(calculate_bulk_modulus, elements)
```

---

## OVITO编程

### Python脚本修饰符

```python
from ovito.io import import_file
from ovito.modifiers import *
from ovito.pipeline import StaticSource
from ovito.data import DataCollection
import numpy as np

# 创建管道
pipeline = import_file("simulation.dump")

# 添加修饰符
# 1. CNA分析
cna = CommonNeighborAnalysisModifier(cutoff=3.5)
pipeline.modifiers.append(cna)

# 2. 选择特定结构
defect_select = SelectExpressionModifier(
    expression="StructureType != 1")  # 非FCC
pipeline.modifiers.append(defect_select)

# 3. 团簇分析
cluster = ClusterAnalysisModifier(
    cutoff=3.5,
    only_selected=True,
    sort_by_size=True
)
pipeline.modifiers.append(cluster)

# 4. 颜色编码
color_mod = ColorCodingModifier(
    property="Cluster",
    gradient=ColorCodingModifier.Rainbow()
)
pipeline.modifiers.append(color_mod)

# 计算特定帧
data = pipeline.compute(100)

# 导出
data = pipeline.compute()
from ovito.io import export_file
export_file(data, "output.xyz", "xyz")
export_file(data, "output.png", "image", size=(800, 600))
```

### 自定义分析修饰符

```python
from ovito.data import *
import numpy as np

def compute_local_strength(frame, data):
    """
    自定义修饰符: 计算局部强度
    """
    # 获取位置和力
    positions = data.particles.positions
    forces = data.particles.forces
    
    # 计算局部应力
    stress = np.zeros(len(positions))
    
    # 邻居查找
    finder = CutoffNeighborFinder(3.5, data)
    
    for i in range(len(positions)):
        for j in finder.find(i):
            # 计算应力贡献
            stress[i] += np.linalg.norm(forces[i])
    
    # 创建新属性
    strength = data.particles_.create_property('LocalStrength', data=stress)
    
    return

# 添加到管道
from ovito.modifiers import PythonScriptModifier
pipeline.modifiers.append(PythonScriptModifier(
    function=compute_local_strength
))
```

---

## 自动化工作流

### DVC数据版本控制

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python prepare_system.py --output data/prepared
    deps:
      - prepare_system.py
    outs:
      - data/prepared
  
  equilibrate:
    cmd: python run_equilibration.py --input data/prepared --output data/equilibrated
    deps:
      - run_equilibration.py
      - data/prepared
    outs:
      - data/equilibrated
  
  production:
    cmd: python run_production.py --input data/equilibrated --output data/production
    deps:
      - run_production.py
      - data/equilibrated
    outs:
      - data/production
  
  analyze:
    cmd: python analyze_results.py --input data/production --output results/
    deps:
      - analyze_results.py
      - data/production
    outs:
      - results/
    metrics:
      - results/metrics.json:
          cache: false
    plots:
      - results/diffusion_coefficient.png
      - results/rdf.png
```

### Snakemake工作流

```python
# Snakefile
configfile: "config.yaml"

rule all:
    input:
        "results/final_report.html"

rule prepare_system:
    output:
        "data/{system}/initial.data"
    script:
        "scripts/prepare_system.py"

rule minimize:
    input:
        "data/{system}/initial.data"
    output:
        "data/{system}/minimized.data",
        "data/{system}/minimized.log"
    shell:
        "lmp -in scripts/minimize.in -var data_file {input} -log {output[1]}"

rule equilibrate:
    input:
        "data/{system}/minimized.data"
    output:
        "data/{system}/equilibrated.restart"
    shell:
        "lmp -in scripts/equilibrate.in -var restart_file {input}"

rule production:
    input:
        "data/{system}/equilibrated.restart"
    output:
        "data/{system}/production.dump",
        "data/{system}/production.log"
    shell:
        "mpirun -np 8 lmp -in scripts/production.in -var restart_file {input}"

rule analyze:
    input:
        dump="data/{system}/production.dump",
        log="data/{system}/production.log"
    output:
        "results/{system}/analysis_complete"
    script:
        "scripts/analyze.py"

rule report:
    input:
        expand("results/{system}/analysis_complete", system=config["systems"])
    output:
        "results/final_report.html"
    script:
        "scripts/generate_report.py"
```

---

## 可视化管道

### 自动化渲染

```python
# render_pipeline.py
import subprocess
from pathlib import Path

def render_trajectory(dump_file, output_dir, style='cinematic'):
    """自动渲染轨迹"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 使用OVITO渲染
    ovito_script = f"""
from ovito.io import import_file
from ovito.modifiers import *
from ovito.vis import *

pipeline = import_file("{dump_file}")

# 添加修饰符
pipeline.modifiers.append(CommonNeighborAnalysisModifier())

# 设置渲染
vp = Viewport()
vp.type = Viewport.Type.Perspective
vp.camera_pos = (50, 50, 100)
vp.camera_dir = (0, 0, -1)

# 渲染所有帧
for frame in range(pipeline.source.num_frames):
    vp.render_image(
        filename="{output_dir}/frame_%04d.png" % frame,
        size=(1920, 1080),
        frame=frame
    )
"""
    
    with open('/tmp/render.py', 'w') as f:
        f.write(ovito_script)
    
    subprocess.run(['ovitos', '/tmp/render.py'])
    
    # 组合成视频
    subprocess.run([
        'ffmpeg', '-y',
        '-framerate', '30',
        '-i', f'{output_dir}/frame_%04d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        f'{output_dir}/movie.mp4'
    ])

# 批量处理
for dump_file in Path('simulations').glob('*.dump'):
    render_trajectory(dump_file, f"movies/{dump_file.stem}")
```

### 交互式仪表板

```python
# dashboard.py
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("LAMMPS Simulation Dashboard"),
    
    dcc.Dropdown(
        id='simulation-select',
        options=[...],
        value='default'
    ),
    
    dcc.Graph(id='trajectory-plot'),
    dcc.Graph(id='energy-plot'),
    dcc.Graph(id='rdf-plot'),
    
    dcc.Slider(
        id='frame-slider',
        min=0,
        max=1000,
        step=1,
        value=0
    ),
    
    html.Div(id='metrics-display')
])

@app.callback(
    Output('trajectory-plot', 'figure'),
    Input('frame-slider', 'value'),
    Input('simulation-select', 'value')
)
def update_trajectory(frame, sim_name):
    # 加载数据
    positions = load_frame(sim_name, frame)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='markers',
        marker=dict(size=2)
    )])
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
```

---

## 快速集成模板

### 最小工作流

```bash
#!/bin/bash
# quick_workflow.sh

# 1. 准备
python -c "
from ase.build import bulk
from ase.io import write
atoms = bulk('Cu', 'fcc', a=3.6)
atoms *= (10, 10, 10)
write('Cu.data', atoms, format='lammps-data')
"

# 2. 运行
cat > run.in << 'EOF'
units metal
atom_style atomic
read_data Cu.data
pair_style eam/alloy
pair_coeff * * Cu_u3.eam.alloy Cu
fix 1 all nvt temp 300 300 $(100.0*dt)
dump 1 all custom 1000 dump.lammpstrj id type x y z
run 100000
EOF

lmp -in run.in

# 3. 分析
python -c "
from ase.io import read
from ase.geometry.analysis import Analysis
traj = read('dump.lammpstrj', index=':')
analyzer = Analysis(traj)
rdf = analyzer.get_rdf(rmax=10.0, nbins=100)
print('RDF calculated')
"
```
