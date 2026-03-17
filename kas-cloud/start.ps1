# KAS Cloud API Server 启动脚本 (Windows PowerShell)
# 用法: .\start.ps1

Write-Host "🚀 Starting KAS Cloud API Server..." -ForegroundColor Cyan

# 检查 Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command python3 -ErrorAction SilentlyContinue
}
if (-not $python) {
    Write-Host "❌ Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

Write-Host "📦 Python found: $($python.Source)" -ForegroundColor Green

# 创建虚拟环境（如果不存在）
if (-not (Test-Path "venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    & $python.Source -m venv venv
}

# 激活虚拟环境
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
. .\venv\Scripts\Activate.ps1

# 安装依赖
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
pip install -q -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# 初始化数据库
Write-Host "🗄️  Initializing database..." -ForegroundColor Yellow
python -c "from database import init_db; init_db()"

# 启动服务器
Write-Host "🌐 Server starting at http://localhost:8000" -ForegroundColor Green
Write-Host "📚 API docs at http://localhost:8000/docs" -ForegroundColor Green
Write-Host "" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
