"""
KAS Web 服务 - FastAPI 后端
提供 REST API 和 WebSocket 支持
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

from kas.core.config import get_config
from kas.core.models import Agent
from kas.core.ingestion import ingest_project
from kas.core.fusion import fuse_agents
from kas.core.chat import ChatEngine
from kas.core.stats import AnalyticsDatabase
from kas.core.market import get_market
from kas.core.cloud_market import get_cloud_client
from kas.core.knowledge import get_knowledge_base
from kas.core.workflow import get_workflow_engine
from kas.core.abtest import get_abtest_engine

app = FastAPI(title="KAS Web API", version="0.2.0")

# 存储 WebSocket 连接
active_connections: List[WebSocket] = []


# ========== Pydantic 模型 ==========

class AgentCreate(BaseModel):
    name: str
    source_path: str
    description: Optional[str] = None


class AgentResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    created_at: str
    capabilities: List[str]


class ChatRequest(BaseModel):
    message: str
    use_mock: bool = False


class ChatResponse(BaseModel):
    response: str
    timestamp: str


class AgentList(BaseModel):
    agents: List[AgentResponse]
    total: int


class StatsResponse(BaseModel):
    total_conversations: int
    total_agents: int
    daily_stats: List[Dict]


# ========== API 路由 ==========

@app.get("/")
async def root():
    """API 根路径"""
    return {
        "name": "KAS Web API",
        "version": "0.2.0",
        "status": "running"
    }


@app.get("/api/agents", response_model=AgentList)
async def list_agents():
    """获取 Agent 列表"""
    config = get_config()
    agents_dir = Path(config.agents_dir)
    
    agents = []
    if agents_dir.exists():
        for agent_path in agents_dir.iterdir():
            if agent_path.is_dir() and (agent_path / "agent.yaml").exists():
                try:
                    agent = Agent.load(agent_path)
                    agents.append(AgentResponse(
                        id=agent.name,
                        name=agent.name,
                        description=agent.description,
                        created_at=agent.created_at or datetime.now().isoformat(),
                        capabilities=agent.capabilities
                    ))
                except Exception:
                    pass
    
    return AgentList(agents=agents, total=len(agents))


@app.get("/api/agents/{agent_name}")
async def get_agent(agent_name: str):
    """获取 Agent 详情"""
    config = get_config()
    agent_path = Path(config.agents_dir) / agent_name
    
    if not agent_path.exists():
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        agent = Agent.load(agent_path)
        return {
            "id": agent.name,
            "name": agent.name,
            "description": agent.description,
            "created_at": agent.created_at,
            "source_projects": agent.source_projects,
            "capabilities": agent.capabilities,
            "system_prompt": agent.system_prompt[:500] + "..." if len(agent.system_prompt) > 500 else agent.system_prompt
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agents/{agent_name}/chat")
async def chat_with_agent(agent_name: str, request: ChatRequest):
    """与 Agent 对话"""
    try:
        chat = ChatEngine(agent_name)
        response = chat.run(request.message, use_mock=request.use_mock)
        return ChatResponse(response=response, timestamp=datetime.now().isoformat())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agents/ingest")
async def ingest_agent(data: AgentCreate):
    """创建新 Agent"""
    try:
        result = ingest_project(
            source_path=data.source_path,
            name=data.name,
            description=data.description
        )
        return {
            "success": True,
            "agent_name": result['name'],
            "capabilities_count": len(result['capabilities'])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """获取统计数据"""
    db = AnalyticsDatabase()
    
    # 获取每日统计
    daily = db.get_daily_stats(days=7)
    
    # 获取 Agent 数量
    config = get_config()
    agents_dir = Path(config.agents_dir)
    agent_count = len([d for d in agents_dir.glob("*") if d.is_dir()]) if agents_dir.exists() else 0
    
    # 总对话数
    cursor = db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM conversations")
    conv_count = cursor.fetchone()[0]
    
    return {
        "total_conversations": conv_count,
        "total_agents": agent_count,
        "daily_stats": [
            {
                "date": s.date,
                "conversations": s.total_conversations,
                "tokens": s.total_tokens
            }
            for s in daily
        ]
    }


@app.get("/api/market/search")
async def market_search(q: str = Query(..., min_length=1)):
    """搜索市场"""
    try:
        client = get_cloud_client()
        if client.is_available():
            results = client.search(q)
        else:
            # 本地市场
            market = get_market()
            results = market.search(q)
        return {"success": True, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/list")
async def market_list():
    """列出市场所有 Agent"""
    try:
        market = get_market()
        agents = market.list_agents()
        return {"success": True, "agents": agents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/market/install/{agent_name}")
async def market_install(agent_name: str):
    """从市场安装 Agent"""
    try:
        market = get_market()
        result = market.install(agent_name)
        return {"success": True, "message": f"Agent {agent_name} 安装成功", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/market/download/{package_id}")
async def market_download(package_id: str):
    """从云端市场下载 Agent"""
    try:
        client = get_cloud_client()
        if not client.is_available():
            raise HTTPException(status_code=503, detail="云端市场不可用")
        
        result = client.download(package_id)
        return {"success": True, "message": "下载成功", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workflows")
async def list_workflows():
    """列出工作流"""
    engine = get_workflow_engine()
    workflows = engine.list_workflows()
    return {"workflows": workflows}


@app.get("/api/workflows/{name}")
async def get_workflow(name: str):
    """获取工作流详情"""
    engine = get_workflow_engine()
    wf = engine.load(name)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return wf.to_dict()


@app.get("/api/abtests")
async def list_abtests():
    """列出 A/B 测试"""
    engine = get_abtest_engine()
    tests = engine.list_tests()
    return {"tests": [t.to_dict() for t in tests]}


@app.get("/api/abtests/{test_id}")
async def get_abtest(test_id: str):
    """获取 A/B 测试详情"""
    engine = get_abtest_engine()
    stats = engine.get_stats(test_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Test not found")
    return stats


# ========== WebSocket 路由 ==========

@app.websocket("/ws/chat/{agent_name}")
async def websocket_chat(websocket: WebSocket, agent_name: str):
    """WebSocket 实时对话"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        chat = ChatEngine(agent_name)
        
        while True:
            # 接收消息
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 发送思考中
            await websocket.send_json({
                "type": "thinking",
                "content": "..."
            })
            
            # 生成回复
            response = chat.run(message.get("text", ""), use_mock=message.get("mock", False))
            
            # 发送回复
            await websocket.send_json({
                "type": "response",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "content": str(e)
        })
        active_connections.remove(websocket)


# ========== 静态文件 ==========

# 检查是否存在构建好的前端文件
frontend_static = Path(__file__).parent / "static"
if frontend_static.exists():
    app.mount("/", StaticFiles(directory=str(frontend_static), html=True), name="static")


def run_web(host: str = "0.0.0.0", port: int = 3000, reload: bool = False):
    """启动 Web 服务"""
    print(f"🌐 KAS Web starting at http://{host}:{port}")
    uvicorn.run("kas.web.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_web()
