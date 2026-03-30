#!/bin/bash
# Docker入口脚本
# 初始化环境并启动服务

set -e

# 打印欢迎信息
echo "========================================"
echo "  DFT+LAMMPS Framework Docker"
echo "  多尺度材料计算环境"
echo "========================================"
echo ""

# 检查GPU可用性
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU支持已启用"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
    echo "⚠ GPU支持未启用（运行CPU模式）"
fi

echo ""

# 检查VASP
if [ -f "/opt/vasp/bin/vasp_std" ]; then
    echo "✓ VASP已配置"
else
    echo "⚠ VASP未配置（需要挂载二进制文件）"
    echo "  使用方法: -v /path/to/vasp:/opt/vasp/bin"
fi

# 检查各组件版本
echo ""
echo "已安装组件:"
echo "  Python: $(python3 --version 2>/dev/null || echo 'N/A')"
echo "  LAMMPS: $(lmp -h 2>&1 | head -1 || echo 'N/A')"
echo "  DeepMD: $(dp --version 2>/dev/null || echo 'N/A')"
echo "  NEP: $(nep 2>&1 | head -1 || echo 'N/A')"
echo "  Quantum ESPRESSO: $(pw.x --version 2>&1 | head -1 || echo 'N/A')"
echo ""

# 设置环境变量
export PYTHONPATH="/workspace/dft_lammps_research:${PYTHONPATH}"
export PATH="/opt/lammps/bin:/opt/vasp/bin:/opt/qe-7.2/bin:/opt/ovito/bin:${PATH}"

# 如果传入了命令，执行它
if [ $# -eq 0 ]; then
    echo "启动交互式shell..."
    echo "提示: 运行 'python /workspace/dft_lammps_research/examples/demo_workflow.py' 查看演示"
    echo ""
    exec /bin/bash
else
    exec "$@"
fi
