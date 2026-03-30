"""
FastAPI主应用
"""

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import uuid
import logging

# 导入路由
from .routes import auth, tasks, dft, md, ml, screening, files, system, users

# 导入监控
from ..monitoring import RateLimitMiddleware, system_monitor, api_metrics

# 导入异常处理
from ..utils import setup_exception_handlers

# 导入配置
from ..models.schemas import APIResponse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动
    logger.info("Starting DFT-LAMMPS API Gateway")
    system_monitor.start(interval=60)
    yield
    # 关闭
    logger.info("Shutting down DFT-LAMMPS API Gateway")
    system_monitor.stop()


def create_app() -> FastAPI:
    """创建FastAPI应用"""
    app = FastAPI(
        title="DFT-LAMMPS API Gateway",
        description="""
        DFT-LAMMPS 计算平台 API 网关
        
        ## 功能
        
        * **认证授权** - JWT和API Key认证
        * **DFT计算** - VASP、QE、ABACUS等DFT计算
        * **MD模拟** - NEP、DeepMD、ReaxFF等分子动力学模拟
        * **ML训练** - 机器学习势函数训练
        * **高通量筛选** - 材料高通量计算筛选
        * **任务管理** - 异步任务队列管理
        
        ## 认证方式
        
        1. **OAuth2** - 通过 `/api/v1/auth/login` 获取JWT令牌
        2. **API Key** - 在请求头中添加 `X-API-Key: your-api-key`
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应限制域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # GZip压缩
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # 限流中间件
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=100,
        window_seconds=60,
        strategy="sliding_window",
    )
    
    # 请求ID中间件
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        # 添加请求ID到响应头
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        # 记录API指标
        api_metrics.record_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            duration=duration,
        )
        
        return response
    
    # 设置异常处理器
    setup_exception_handlers(app)
    
    # 注册路由
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["认证"])
    app.include_router(users.router, prefix="/api/v1/users", tags=["用户管理"])
    app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["任务管理"])
    app.include_router(dft.router, prefix="/api/v1/dft", tags=["DFT计算"])
    app.include_router(md.router, prefix="/api/v1/md", tags=["MD模拟"])
    app.include_router(ml.router, prefix="/api/v1/ml", tags=["ML训练"])
    app.include_router(screening.router, prefix="/api/v1/screening", tags=["高通量筛选"])
    app.include_router(files.router, prefix="/api/v1/files", tags=["文件管理"])
    app.include_router(system.router, prefix="/api/v1/system", tags=["系统管理"])
    
    # 根路由
    @app.get("/", response_model=APIResponse)
    async def root():
        return APIResponse(
            success=True,
            message="DFT-LAMMPS API Gateway",
            data={
                "version": "1.0.0",
                "docs": "/docs",
                "health": "/api/v1/system/health",
            }
        )
    
    # 健康检查
    @app.get("/health", tags=["健康检查"])
    async def health_check():
        return {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": time.time(),
        }
    
    return app


# 创建应用实例
app = create_app()
