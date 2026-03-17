"""
KAS Cloud API Server
FastAPI 主入口
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from database import init_db
from users import router as users_router
from market import router as market_router

# 创建应用
app = FastAPI(
    title="KAS Cloud API",
    description="Kimi Agent Studio 云端市场 API",
    version="0.1.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    """启动时初始化"""
    init_db()


@app.get("/")
def root():
    """根路径"""
    return {
        "name": "KAS Cloud API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "users": "/api/v1/users",
            "market": "/api/v1/market"
        }
    }


@app.get("/health")
def health():
    """健康检查"""
    return {"status": "healthy"}


# 注册路由
app.include_router(users_router)
app.include_router(market_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
