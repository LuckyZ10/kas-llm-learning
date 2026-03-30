#!/bin/bash
# DFT-LAMMPS API Gateway启动脚本

set -e

echo "🚀 DFT-LAMMPS API Gateway 启动脚本"
echo "====================================="

# 检查环境
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 未安装"; exit 1; }
command -v redis-cli >/dev/null 2>&1 || { echo "⚠️ Redis 未安装，请先安装Redis"; }

# 设置默认值
export API_HOST=${API_HOST:-0.0.0.0}
export API_PORT=${API_PORT:-8000}
export WORKERS=${WORKERS:-4}
export LOG_LEVEL=${LOG_LEVEL:-info}

# 检查虚拟环境
if [ -d "venv" ]; then
    echo "✅ 激活虚拟环境..."
    source venv/bin/activate
else
    echo "⚠️ 虚拟环境不存在，使用系统Python"
fi

# 检查依赖
echo "📦 检查依赖..."
python3 -c "import fastapi" 2>/dev/null || {
    echo "📥 安装依赖..."
    pip install -r dftlammps/api_gateway/deployments/docker/requirements.txt
}

# 检查Redis连接
echo "🔍 检查Redis连接..."
if redis-cli ping >/dev/null 2>&1; then
    echo "✅ Redis连接正常"
else
    echo "⚠️ Redis未启动，尝试启动Redis..."
    redis-server --daemonize yes
    sleep 2
    if redis-cli ping >/dev/null 2>&1; then
        echo "✅ Redis已启动"
    else
        echo "❌ 无法启动Redis，请手动启动"
        exit 1
    fi
fi

echo ""
echo "🌐 启动API服务..."
echo "   地址: http://${API_HOST}:${API_PORT}"
echo "   文档: http://${API_HOST}:${API_PORT}/docs"
echo ""

# 启动服务
exec uvicorn dftlammps.api_gateway.api.main:app \
    --host ${API_HOST} \
    --port ${API_PORT} \
    --workers ${WORKERS} \
    --log-level ${LOG_LEVEL} \
    --reload
