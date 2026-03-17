"""
测试脚本 - 验证 KAS Cloud API 可以加载
"""
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from kas_cloud.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

# 测试根路径
response = client.get("/")
print(f"Root: {response.status_code}")
print(f"Response: {response.json()}")

# 测试健康检查
response = client.get("/health")
print(f"Health: {response.status_code}")
print(f"Response: {response.json()}")

print("\n✅ KAS Cloud API 基础测试通过！")
print("📚 API 文档: http://localhost:8000/docs")
print("🔧 启动命令: cd kas-cloud && bash start.sh")
