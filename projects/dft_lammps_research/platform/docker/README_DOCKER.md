# DFT+LAMMPS Framework Docker 使用指南

本目录包含DFT+LAMMPS多尺度材料计算框架的完整Docker运行环境。

## 包含组件

| 组件 | 版本 | 说明 |
|------|------|------|
| **Python** | 3.10 | 科学计算环境 |
| **LAMMPS** | 2Aug2023 | 分子动力学模拟（支持GPU） |
| **DeepMD-kit** | 2.2.4 | 深度势能训练 |
| **NEP** | latest | 神经演化势（GPUMD） |
| **Quantum ESPRESSO** | 7.2 | 第一性原理计算 |
| **VASP** | 用户自备 | 商业软件，需自行挂载 |
| **MACE** | 0.3.0 | 等变神经网络势 |
| **ASE** | 3.22.1 | 原子模拟环境 |
| **Pymatgen** | 2023.7.17 | 材料分析库 |
| **Jupyter Lab** | 4.0 | 交互式开发环境 |
| **OVITO** | 3.9.4 | 原子可视化 |

## 快速开始

### 1. 构建镜像

```bash
cd docker
docker-compose build
```

或者使用Docker直接构建：

```bash
docker build -t dft-lammps-framework:latest -f docker/Dockerfile .
```

### 2. 启动服务

#### 方式一：交互式Shell

```bash
docker-compose up -d dft-lammps
docker-compose exec dft-lammps /bin/bash
```

#### 方式二：Jupyter Lab

```bash
docker-compose up -d jupyter
```

访问 http://localhost:8888 打开Jupyter界面。

#### 方式三：监控Dashboard

```bash
docker-compose up -d dashboard
```

访问 http://localhost:8050 查看监控面板。

### 3. 运行演示

```bash
# 进入容器
docker-compose exec dft-lammps /bin/bash

# 运行演示工作流
cd /workspace/dft_lammps_research
python examples/demo_workflow.py

# 查看输出
ls -la demo_output/
```

## 配置说明

### VASP配置（重要）

VASP是商业软件，需要用户自行提供二进制文件和许可证。

#### 方法一：挂载二进制文件

```bash
docker run -it \
  -v /path/to/your/vasp/bin:/opt/vasp/bin:ro \
  -v /path/to/your/vasp/potentials:/opt/vasp/potentials:ro \
  dft-lammps-framework:latest
```

#### 方法二：修改docker-compose.yml

编辑 `docker-compose.yml`，取消注释并修改以下行：

```yaml
volumes:
  # ... 其他卷 ...
  - /path/to/your/vasp/bin:/opt/vasp/bin:ro
  - /path/to/your/vasp/potentials:/opt/vasp/potentials:ro
```

### Materials Project API Key

设置环境变量以使用Materials Project数据库：

```bash
export MP_API_KEY=your_api_key_here
docker-compose up -d
```

或者在 `docker-compose.yml` 中设置：

```yaml
environment:
  - MP_API_KEY=your_api_key_here
```

### GPU支持

确保已安装NVIDIA Docker运行时：

```bash
# 检查NVIDIA运行时
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

如果GPU不可用，容器将自动切换到CPU模式。

## 常用命令

### 查看容器状态

```bash
docker-compose ps
```

### 查看日志

```bash
# 所有服务
docker-compose logs

# 特定服务
docker-compose logs -f dft-lammps
docker-compose logs -f jupyter
docker-compose logs -f dashboard
```

### 停止服务

```bash
docker-compose down
```

### 完全清理（包括数据卷）

```bash
docker-compose down -v
docker system prune
```

### 进入运行中的容器

```bash
docker-compose exec dft-lammps /bin/bash
```

### 执行单次命令

```bash
docker-compose run --rm dft-lammps python examples/demo_workflow.py
```

## 目录结构

```
docker/
├── Dockerfile              # 镜像构建文件
├── docker-compose.yml      # 服务编排配置
├── entrypoint.sh           # 容器入口脚本
├── README_DOCKER.md        # 本文件
└── outputs/                # 输出目录（自动创建）
```

## 性能优化

### 调整CPU线程数

编辑 `docker-compose.yml`：

```yaml
environment:
  - OMP_NUM_THREADS=8
  - OPENBLAS_NUM_THREADS=8
```

### 内存限制

```yaml
deploy:
  resources:
    limits:
      memory: 32G
    reservations:
      memory: 16G
```

### GPU选择

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0', '1']  # 指定GPU
          capabilities: [gpu]
```

## 故障排除

### 问题：无法连接Jupyter

检查容器是否运行：
```bash
docker-compose ps
```

查看Jupyter日志：
```bash
docker-compose logs jupyter
```

### 问题：VASP命令未找到

确保已正确挂载VASP二进制文件：
```bash
docker-compose exec dft-lammps ls -la /opt/vasp/bin/
```

### 问题：GPU不可用

检查NVIDIA运行时：
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

如果没有GPU，容器会自动切换到CPU模式。

### 问题：权限错误

文件可能以root用户创建。修复权限：
```bash
docker-compose exec dft-lammps chown -R researcher:researcher /workspace
```

## 自定义镜像

### 添加更多Python包

编辑 `Dockerfile`，在适当位置添加：

```dockerfile
RUN pip3 install --no-cache-dir your-package-name
```

### 安装额外软件

```dockerfile
RUN apt-get update && apt-get install -y \
    your-package \
    && rm -rf /var/lib/apt/lists/*
```

然后重新构建：
```bash
docker-compose build --no-cache
```

## 示例工作流

### 运行完整工作流

```bash
# 进入容器
docker-compose exec dft-lammps /bin/bash

# 设置工作目录
cd /workspace

# 运行示例
python dft_lammps_research/examples/quick_start/simple_workflow.py
```

### 批量筛选

```bash
python dft_lammps_research/battery_screening_pipeline.py \
    --config dft_lammps_research/screening_config.yaml
```

### 监控仪表板

```bash
python dft_lammps_research/monitoring_dashboard.py
```

## 网络配置

服务间可通过服务名互相访问：

- `dft-lammps`: 主计算服务
- `jupyter`: Jupyter Lab (端口8888)
- `dashboard`: 监控面板 (端口8050)
- `slurm-master`: Slurm测试环境

## 数据持久化

以下数据会持久保存：

- `/workspace/data`: 计算数据
- `/home/researcher/.jupyter`: Jupyter配置
- `./outputs`: 输出文件

## 安全提示

1. 不要在镜像中存储VASP许可证文件
2. 使用环境变量传递敏感信息（API keys等）
3. 生产环境应移除Jupyter的tokenless访问
4. 定期更新基础镜像以获取安全补丁

## 许可证

- Docker配置: MIT License
- VASP: 需要商业许可证
- Quantum ESPRESSO: GPL
- LAMMPS: GPL
- DeepMD-kit: LGPL

## 获取帮助

- 项目主页: https://github.com/yourusername/dft-lammps-framework
- 文档: https://your-docs-site.com
- 问题反馈: https://github.com/yourusername/dft-lammps-framework/issues
