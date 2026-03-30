# DFT-LAMMPS API Gateway

DFT-LAMMPS 计算平台 API 网关 - Phase 69

## 功能特性

- **认证授权**：JWT令牌和API Key双重认证机制
- **DFT计算**：支持VASP、Quantum ESPRESSO、ABACUS、CP2K等多种DFT代码
- **MD模拟**：支持NEP、DeepMD、ReaxFF等机器学习势函数
- **ML训练**：机器学习势函数训练和管理
- **高通量筛选**：多目标优化、Pareto前沿、聚类分析
- **任务队列**：基于Celery的分布式异步任务处理
- **限流监控**：API限流、Prometheus指标监控
- **容器部署**：Docker Compose一键部署

## 快速开始

### 环境要求

- Python 3.11+
- Redis 7+
- PostgreSQL 15+
- Docker & Docker Compose (可选)

### 本地安装

```bash
# 克隆仓库
git clone https://github.com/your-org/dftlammps.git
cd dftlammps

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r dftlammps/api_gateway/deployments/docker/requirements.txt

# 启动Redis
redis-server

# 启动API服务
uvicorn dftlammps.api_gateway.api.main:app --reload
```

### Docker部署

```bash
cd dftlammps/api_gateway/deployments/docker

# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f api

# 停止服务
docker-compose down
```

## API文档

启动服务后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## 认证方式

### 1. OAuth2 (JWT)

```bash
# 获取访问令牌
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=demo&password=demo123"

# 使用令牌访问API
curl -X GET "http://localhost:8000/api/v1/auth/me" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### 2. API Key

```bash
curl -X GET "http://localhost:8000/api/v1/dft/codes" \
  -H "X-API-Key: dftlammps_demo_api_key_12345"
```

## 使用示例

### 提交DFT计算

```python
import requests

# 提交DFT计算任务
response = requests.post(
    "http://localhost:8000/api/v1/dft/calculate",
    headers={"X-API-Key": "dftlammps_demo_api_key_12345"},
    json={
        "name": "Li3PS4_optimization",
        "calculation_type": "relax",
        "code": "vasp",
        "structure": {
            "lattice": [[...], [...], [...]],
            "elements": ["Li", "P", "S"],
            "positions": [[...], [...], [...]]
        },
        "parameters": {
            "encut": 520,
            "kpoints": [3, 3, 3]
        }
    }
)

task_id = response.json()["task_id"]
print(f"Task submitted: {task_id}")

# 查询任务状态
status = requests.get(
    f"http://localhost:8000/api/v1/tasks/{task_id}/status",
    headers={"X-API-Key": "dftlammps_demo_api_key_12345"}
)
print(status.json())
```

### 提交MD模拟

```python
response = requests.post(
    "http://localhost:8000/api/v1/md/simulate",
    headers={"X-API-Key": "dftlammps_demo_api_key_12345"},
    json={
        "name": "Li_diffusion",
        "simulation_type": "nvt",
        "potential": "nep",
        "structure": {...},
        "temperature": 300,
        "n_steps": 100000,
        "time_step": 1.0
    }
)
```

### 高通量筛选

```python
response = requests.post(
    "http://localhost:8000/api/v1/screening/run",
    headers={"X-API-Key": "dftlammps_demo_api_key_12345"},
    json={
        "name": "cathode_screening",
        "dataset": {"structures": [...]},
        "criteria": [
            {"property_name": "voltage", "min_value": 3.5, "weight": 1.0},
            {"property_name": "capacity", "min_value": 200, "weight": 1.0}
        ],
        "method": "multi_objective",
        "top_k": 10
    }
)
```

## 默认用户

| 用户名 | 密码 | 角色 | API Key |
|--------|------|------|---------|
| admin | admin123 | admin | - |
| demo | demo123 | researcher | - |
| api_client | api123 | api_client | dftlammps_demo_api_key_12345 |

## 项目结构

```
dftlammps/api_gateway/
├── api/
│   ├── main.py              # FastAPI应用入口
│   └── routes/              # API路由
│       ├── auth.py          # 认证路由
│       ├── users.py         # 用户管理
│       ├── tasks.py         # 任务管理
│       ├── dft.py           # DFT计算
│       ├── md.py            # MD模拟
│       ├── ml.py            # ML训练
│       ├── screening.py     # 高通量筛选
│       ├── files.py         # 文件管理
│       └── system.py        # 系统管理
├── auth/                    # 认证授权模块
│   ├── security.py          # JWT/OAuth2实现
│   └── permissions.py       # RBAC权限控制
├── models/                  # 数据模型
│   └── schemas.py           # Pydantic模型
├── tasks/                   # Celery任务
│   ├── celery_app.py        # Celery配置
│   ├── dft_tasks.py         # DFT任务
│   ├── md_tasks.py          # MD任务
│   ├── ml_tasks.py          # ML任务
│   ├── screening_tasks.py   # 筛选任务
│   └── analysis_tasks.py    # 分析任务
├── monitoring/              # 监控模块
│   ├── rate_limiter.py      # 限流器
│   └── metrics.py           # 指标收集
├── utils/                   # 工具函数
│   ├── helpers.py           # 辅助函数
│   └── exceptions.py        # 异常处理
├── tests/                   # 测试
│   ├── test_api.py          # API测试
│   └── conftest.py          # 测试配置
└── deployments/             # 部署配置
    └── docker/
        ├── docker-compose.yml
        ├── Dockerfile.api
        ├── Dockerfile.worker
        └── nginx.conf
```

## 监控

- **Flower**: http://localhost:5555 (Celery监控)
- **Prometheus指标**: http://localhost:8000/api/v1/system/metrics
- **健康检查**: http://localhost:8000/health

## 测试

```bash
# 运行所有测试
pytest dftlammps/api_gateway/tests/ -v

# 运行特定测试
pytest dftlammps/api_gateway/tests/test_api.py::TestAuth -v

# 带覆盖率测试
pytest --cov=dftlammps.api_gateway dftlammps/api_gateway/tests/
```

## 配置

通过环境变量配置：

```bash
# 安全
export SECRET_KEY="your-secret-key"
export ACCESS_TOKEN_EXPIRE_MINUTES=30

# 数据库
export DATABASE_URL="postgresql://user:pass@localhost/dftlammps"
export REDIS_URL="redis://localhost:6379/0"

# Celery
export CELERY_BROKER_URL="redis://localhost:6379/0"
export CELERY_RESULT_BACKEND="redis://localhost:6379/0"

# 限流
export RATE_LIMIT_PER_MINUTE=100
```

## 许可证

MIT License
