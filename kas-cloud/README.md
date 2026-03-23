# KAS Cloud API

Klaw Agent Studio 云端市场后端服务

## 功能

- **用户系统**: 注册、登录、API Key 管理
- **Agent 市场**: 发布、搜索、下载、评分
- **文件存储**: .kas-agent 包管理

## 快速开始

```bash
cd kas-cloud
bash start.sh
```

## API 端点

### 用户
- `POST /api/v1/users/register` - 注册
- `POST /api/v1/users/login` - 登录
- `GET /api/v1/users/me` - 获取用户信息

### 市场
- `POST /api/v1/market/publish` - 发布 Agent
- `GET /api/v1/market/search` - 搜索 Agent
- `GET /api/v1/market/packages/{id}` - 获取详情
- `POST /api/v1/market/packages/{id}/download` - 下载
- `POST /api/v1/market/packages/{id}/rate` - 评分

## 文档

启动后访问: http://localhost:8000/docs
