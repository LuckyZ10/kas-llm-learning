"""
相场-DFT多尺度耦合模块 - 快速入门
=================================

本模块提供从微观(DFT/MD)到介观(相场)的完整多尺度模拟工具链。

安装要求
--------
必需:
    - Python >= 3.8
    - NumPy >= 1.20
    - SciPy >= 1.7

可选:
    - CuPy (GPU加速)
    - ASE (DFT/MD接口)
    - Pymatgen (材料分析)
    - Matplotlib (可视化)

基本用法
--------

1. Cahn-Hilliard模拟:

    from phase_field import CahnHilliardSolver, CahnHilliardConfig
    
    config = CahnHilliardConfig(nx=128, ny=128, M=1.0, kappa=1.0)
    solver = CahnHilliardSolver(config)
    solver.initialize_fields(seed=42)
    result = solver.run(n_steps=5000)

2. SEI生长模拟:

    from phase_field import SEIGrowthSimulator, SEIConfig
    
    config = SEIConfig(nx=128, ny=64, component_names=['organic', 'Li2CO3', 'LiF'])
    simulator = SEIGrowthSimulator(config)
    simulator.initialize_fields()
    result = simulator.run()
    
    sei_props = simulator.get_sei_properties()
    print(f"SEI厚度: {sei_props['thickness']:.2f} nm")

3. 完整工作流:

    from phase_field import PhaseFieldWorkflow, WorkflowConfig
    
    config = WorkflowConfig(
        dft_output_path="./vasp_results",
        run_dft=True,
        run_phase_field=True
    )
    workflow = PhaseFieldWorkflow(config)
    results = workflow.run()

模块结构
--------

phase_field/
├── core/              # 核心物理模型
│   ├── cahn_hilliard.py      # Cahn-Hilliard方程
│   ├── allen_cahn.py         # Allen-Cahn方程
│   ├── electrochemical.py    # 电化学相场
│   └── mechanochemistry.py   # 力学-化学耦合
├── solvers/           # 数值求解器
│   ├── finite_difference.py  # 有限差分
│   ├── finite_element.py     # 有限元
│   ├── gpu_solver.py         # GPU加速
│   ├── parallel_solver.py    # 并行求解
│   └── adaptive_mesh.py      # 自适应网格
├── coupling/          # 多尺度耦合
│   ├── dft_coupling.py       # DFT耦合
│   ├── md_coupling.py        # MD耦合
│   └── parameter_transfer.py # 参数传递
├── applications/      # 应用模块
│   ├── sei_growth.py         # SEI生长
│   ├── precipitation.py      # 沉淀相演化
│   ├── grain_boundary.py     # 晶界迁移
│   └── catalyst_reconstruction.py  # 催化剂重构
├── workflow.py        # 自动化工作流
└── examples/          # 使用示例

更多信息
--------
- 详细文档: README.md
- 实施报告: IMPLEMENTATION_REPORT.md
- 测试套件: tests/test_models.py
"""

__docformat__ = "numpy"
