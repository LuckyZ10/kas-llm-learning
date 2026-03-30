#!/bin/bash
# 启动 KAS Cloud API Server

echo "🚀 Starting KAS Cloud API Server..."

# 安装依赖（如果需要）
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# 初始化数据库
echo "🗄️ Initializing database..."
python3 -c "from kas_cloud.database import init_db; init_db()"

# 启动服务器
echo "🌐 Server starting at http://localhost:8000"
echo "📚 API docs at http://localhost:8000/docs"
echo ""
uvicorn kas_cloud.main:app --host 0.0.0.0 --port 8000 --reload
