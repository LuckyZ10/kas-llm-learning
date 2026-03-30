#!/bin/bash
# setup.sh - 可视化与后处理工具安装脚本

echo "========================================"
echo "可视化与后处理工具安装脚本"
echo "========================================"

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 创建虚拟环境
echo ""
echo "[1/6] 创建虚拟环境..."
python3 -m venv md_visualization_env
source md_visualization_env/bin/activate

# 升级pip
echo ""
echo "[2/6] 升级pip..."
pip install --upgrade pip

# 安装科学计算基础库
echo ""
echo "[3/6] 安装科学计算库..."
pip install numpy scipy pandas scikit-learn matplotlib

# 安装可视化库
echo ""
echo "[4/6] 安装可视化库..."
pip install plotly dash dash-bio streamlit

# 安装3D可视化库
echo ""
echo "[5/6] 安装3D可视化库..."
pip install pyvista vtk trame

# 安装分子模拟分析库
echo ""
echo "[6/6] 安装分子模拟分析库..."

# OVITO
pip install ovito

# MDAnalysis
pip install MDAnalysis MDAnalysisTests

# Pymatgen
pip install pymatgen

# RDKit (化学信息学)
pip install rdkit

# ASE (原子模拟环境)
pip install ase

# 可选：安装Jupyter
echo ""
echo "[可选] 安装Jupyter..."
pip install jupyter jupyterlab ipywidgets

# 可选：安装Blender Python API (如果Blender已安装)
# pip install bpy

echo ""
echo "========================================"
echo "安装完成!"
echo "========================================"
echo ""
echo "激活环境: source md_visualization_env/bin/activate"
echo ""
echo "模块清单:"
echo "  ✓ OVITO - 原子和分子模拟数据可视化"
echo "  ✓ MDAnalysis - 分子动力学轨迹分析"
echo "  ✓ Pymatgen - 材料分析"
echo "  ✓ Plotly/Dash - 交互式可视化"
echo "  ✓ PyVista - 3D科学可视化"
echo "  ✓ Streamlit - 快速仪表板"
echo ""
echo "运行示例:"
echo "  python ovito_defect_analysis.py"
echo "  python vmd_biomolecular_analysis.py"
echo "  python pymatgen_materials_analysis.py"
echo "  python interactive_dashboard.py"
echo "  python 3d_visualization.py"
echo "  python integrated_pipeline.py"
echo ""
