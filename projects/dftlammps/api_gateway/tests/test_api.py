"""
API Gateway测试
"""

import pytest
from fastapi.testclient import TestClient
from dftlammps.api_gateway.api.main import create_app

# 创建测试客户端
app = create_app()
client = TestClient(app)


class TestHealth:
    """健康检查测试"""
    
    def test_health_check(self):
        """测试健康检查端点"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    def test_root_endpoint(self):
        """测试根端点"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestAuth:
    """认证测试"""
    
    def test_login_success(self):
        """测试成功登录"""
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "demo", "password": "demo123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
    
    def test_login_failure(self):
        """测试失败登录"""
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "demo", "password": "wrong_password"}
        )
        assert response.status_code == 401
    
    def test_get_current_user(self):
        """测试获取当前用户"""
        # 先登录
        login_response = client.post(
            "/api/v1/auth/login",
            data={"username": "demo", "password": "demo123"}
        )
        token = login_response.json()["access_token"]
        
        # 获取用户信息
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "demo"


class TestDFT:
    """DFT计算测试"""
    
    def test_list_dft_codes(self):
        """测试获取DFT代码列表"""
        # 使用API Key认证
        response = client.get(
            "/api/v1/dft/codes",
            headers={"X-API-Key": "dftlammps_demo_api_key_12345"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "codes" in data
        assert len(data["codes"]) > 0
    
    def test_submit_dft_calculation(self):
        """测试提交DFT计算"""
        response = client.post(
            "/api/v1/dft/calculate",
            headers={"X-API-Key": "dftlammps_demo_api_key_12345"},
            json={
                "name": "test_calculation",
                "calculation_type": "scf",
                "code": "vasp",
                "structure": {"elements": ["Li", "P", "S"], "positions": [[0, 0, 0]]},
                "parameters": {"encut": 520},
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"


class TestTasks:
    """任务管理测试"""
    
    def test_list_tasks(self):
        """测试获取任务列表"""
        response = client.get(
            "/api/v1/tasks",
            headers={"X-API-Key": "dftlammps_demo_api_key_12345"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
    
    def test_create_task(self):
        """测试创建任务"""
        response = client.post(
            "/api/v1/tasks",
            headers={"X-API-Key": "dftlammps_demo_api_key_12345"},
            json={
                "name": "test_task",
                "task_type": "dft_calculation",
                "input_data": {"test": True},
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["status"] == "queued"


class TestMD:
    """MD模拟测试"""
    
    def test_list_potentials(self):
        """测试获取势能列表"""
        response = client.get(
            "/api/v1/md/potentials",
            headers={"X-API-Key": "dftlammps_demo_api_key_12345"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "potentials" in data
    
    def test_submit_md_simulation(self):
        """测试提交MD模拟"""
        response = client.post(
            "/api/v1/md/simulate",
            headers={"X-API-Key": "dftlammps_demo_api_key_12345"},
            json={
                "name": "test_md",
                "simulation_type": "nvt",
                "potential": "nep",
                "structure": {"elements": ["Li", "S"]},
                "temperature": 300,
                "n_steps": 1000,
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data


class TestML:
    """ML训练测试"""
    
    def test_list_models(self):
        """测试获取模型列表"""
        response = client.get(
            "/api/v1/ml/models",
            headers={"X-API-Key": "dftlammps_demo_api_key_12345"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
    
    def test_submit_training(self):
        """测试提交训练任务"""
        response = client.post(
            "/api/v1/ml/train",
            headers={"X-API-Key": "dftlammps_demo_api_key_12345"},
            json={
                "name": "test_training",
                "model_type": "nep",
                "training_data": {"n_structures": 100},
                "hyperparameters": {"learning_rate": 0.001},
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data


class TestSystem:
    """系统管理测试"""
    
    def test_get_metrics(self):
        """测试获取指标"""
        response = client.get(
            "/api/v1/system/metrics",
            headers={"X-API-Key": "dftlammps_demo_api_key_12345"}
        )
        # 需要SYSTEM_READ权限
        assert response.status_code in [200, 403]
    
    def test_get_stats(self):
        """测试获取系统统计"""
        response = client.get(
            "/api/v1/system/stats",
            headers={"X-API-Key": "dftlammps_demo_api_key_12345"}
        )
        assert response.status_code in [200, 403]


class TestRateLimit:
    """限流测试"""
    
    def test_rate_limit_headers(self):
        """测试限流响应头"""
        response = client.get("/health")
        assert response.status_code == 200
        # 检查限流头部
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
