"""
测试配置
"""

import pytest
from fastapi.testclient import TestClient
from dftlammps.api_gateway.api.main import create_app


@pytest.fixture
def app():
    """创建应用fixture"""
    return create_app()


@pytest.fixture
def client(app):
    """创建测试客户端fixture"""
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    """获取认证头部"""
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "demo", "password": "demo123"}
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def api_key_headers():
    """API Key认证头部"""
    return {"X-API-Key": "dftlammps_demo_api_key_12345"}
