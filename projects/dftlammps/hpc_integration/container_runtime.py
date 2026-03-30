#!/usr/bin/env python3
"""
container_runtime.py
===================
容器运行时模块

支持的容器技术：
- Singularity (HPC首选)
- Docker (rootless模式)
- Podman (可选)

功能：
- 镜像拉取
- 容器执行
- 绑定挂载管理
- GPU支持
- 多架构支持
"""

import os
import re
import json
import time
import logging
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class ContainerEngine(Enum):
    """容器引擎类型"""
    SINGULARITY = "singularity"
    APPTAINER = "apptainer"  # Singularity的新名称
    DOCKER = "docker"
    PODMAN = "podman"


@dataclass
class ContainerImage:
    """容器镜像"""
    name: str
    tag: str = "latest"
    registry: Optional[str] = None
    digest: Optional[str] = None
    local_path: Optional[Path] = None
    
    @property
    def full_name(self) -> str:
        """完整镜像名"""
        if self.registry:
            return f"{self.registry}/{self.name}:{self.tag}"
        return f"{self.name}:{self.tag}"
    
    @property
    def sif_name(self) -> str:
        """Singularity镜像文件名"""
        safe_name = self.name.replace("/", "_")
        return f"{safe_name}_{self.tag}.sif"
    
    @classmethod
    def from_string(cls, image_string: str) -> "ContainerImage":
        """从字符串解析镜像"""
        # 格式: registry/name:tag 或 name:tag
        if ":" in image_string:
            image_part, tag = image_string.rsplit(":", 1)
        else:
            image_part, tag = image_string, "latest"
        
        # 解析registry
        if "/" in image_part and "." in image_part.split("/")[0]:
            registry, name = image_part.split("/", 1)
        else:
            registry, name = None, image_part
        
        return cls(name=name, tag=tag, registry=registry)


@dataclass
class ContainerConfig:
    """容器配置"""
    # 基本配置
    image: ContainerImage
    engine: ContainerEngine = ContainerEngine.SINGULARITY
    
    # 资源限制
    cpus: Optional[int] = None
    memory_gb: Optional[float] = None
    gpus: Optional[Union[int, List[int]]] = None
    
    # 挂载
    bind_mounts: Dict[str, str] = field(default_factory=dict)  # host:container
    volumes: List[str] = field(default_factory=list)
    
    # 环境
    environment: Dict[str, str] = field(default_factory=dict)
    workdir: Optional[str] = None
    
    # 网络
    network: str = "host"  # host, bridge, none
    
    # 权限
    privileged: bool = False
    user: Optional[str] = None
    
    # 其他选项
    cleanup: bool = True
    debug: bool = False


@dataclass
class ContainerExecutionResult:
    """容器执行结果"""
    success: bool
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    container_id: Optional[str] = None


class ContainerRuntime(ABC):
    """容器运行时基类"""
    
    def __init__(self, config: ContainerConfig):
        self.config = config
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查运行时是否可用"""
        pass
    
    @abstractmethod
    def pull_image(self, force: bool = False) -> bool:
        """拉取镜像"""
        pass
    
    @abstractmethod
    def build_command(
        self,
        command: List[str],
        interactive: bool = False,
        detach: bool = False
    ) -> List[str]:
        """构建容器执行命令"""
        pass
    
    @abstractmethod
    def run(
        self,
        command: List[str],
        working_dir: Path = None,
        interactive: bool = False,
        timeout: int = None
    ) -> ContainerExecutionResult:
        """运行容器"""
        pass
    
    @abstractmethod
    def exec_in_container(
        self,
        container_id: str,
        command: List[str]
    ) -> ContainerExecutionResult:
        """在运行中的容器内执行命令"""
        pass
    
    @abstractmethod
    def cleanup(self, container_id: str = None):
        """清理容器资源"""
        pass
    
    def _run_subprocess(
        self,
        command: List[str],
        working_dir: Path = None,
        timeout: int = None,
        capture_output: bool = True
    ) -> Tuple[int, str, str]:
        """运行子进程"""
        logger.debug(f"Executing: {' '.join(command)}")
        
        result = subprocess.run(
            command,
            cwd=working_dir,
            capture_output=capture_output,
            text=True,
            timeout=timeout
        )
        
        return result.returncode, result.stdout, result.stderr


class SingularityRuntime(ContainerRuntime):
    """Singularity/Apptainer运行时"""
    
    def __init__(self, config: ContainerConfig, image_cache_dir: Path = None):
        super().__init__(config)
        self.image_cache_dir = image_cache_dir or Path.home() / ".singularity" / "cache"
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 检测可用的命令
        self._command = self._detect_command()
    
    def _detect_command(self) -> str:
        """检测可用的Singularity/Apptainer命令"""
        for cmd in ["apptainer", "singularity"]:
            result = subprocess.run(
                ["which", cmd],
                capture_output=True
            )
            if result.returncode == 0:
                return cmd
        
        return "singularity"  # 默认
    
    def is_available(self) -> bool:
        """检查Singularity是否可用"""
        result = subprocess.run(
            [self._command, "version"],
            capture_output=True
        )
        return result.returncode == 0
    
    def pull_image(self, force: bool = False) -> bool:
        """拉取Singularity镜像"""
        image_path = self.image_cache_dir / self.config.image.sif_name
        
        # 检查是否已存在
        if image_path.exists() and not force:
            logger.info(f"Image already exists: {image_path}")
            self.config.image.local_path = image_path
            return True
        
        # 拉取镜像
        cmd = [
            self._command,
            "pull",
            str(image_path),
            self.config.image.full_name
        ]
        
        logger.info(f"Pulling image: {self.config.image.full_name}")
        
        code, stdout, stderr = self._run_subprocess(cmd, timeout=600)
        
        if code == 0:
            self.config.image.local_path = image_path
            logger.info(f"Image pulled successfully: {image_path}")
            return True
        else:
            logger.error(f"Failed to pull image: {stderr}")
            return False
    
    def build_command(
        self,
        command: List[str],
        interactive: bool = False,
        detach: bool = False
    ) -> List[str]:
        """构建Singularity执行命令"""
        cmd = [self._command, "exec"]
        
        # 绑定挂载
        for host_path, container_path in self.config.bind_mounts.items():
            cmd.extend(["--bind", f"{host_path}:{container_path}"])
        
        # GPU支持
        if self.config.gpus:
            if isinstance(self.config.gpus, list):
                gpu_opt = ",".join(map(str, self.config.gpus))
                cmd.extend(["--nv", f"--nvccli", f"--nvidia", gpu_opt])
            else:
                cmd.extend(["--nv"])
        
        # 环境变量
        for key, value in self.config.environment.items():
            cmd.extend(["--env", f"{key}={value}"])
        
        # 工作目录
        if self.config.workdir:
            cmd.extend(["--pwd", self.config.workdir])
        
        # 网络
        if self.config.network == "host":
            cmd.append("--net")
        
        # 调试
        if self.config.debug:
            cmd.append("--debug")
        
        # 镜像
        if self.config.image.local_path:
            cmd.append(str(self.config.image.local_path))
        else:
            cmd.append(self.config.image.full_name)
        
        # 执行的命令
        cmd.extend(command)
        
        return cmd
    
    def run(
        self,
        command: List[str],
        working_dir: Path = None,
        interactive: bool = False,
        timeout: int = None
    ) -> ContainerExecutionResult:
        """运行Singularity容器"""
        start_time = time.time()
        
        # 确保镜像存在
        if not self.pull_image():
            return ContainerExecutionResult(
                success=False,
                return_code=-1,
                stderr="Failed to pull image"
            )
        
        # 构建命令
        cmd = self.build_command(command, interactive)
        
        # 执行
        code, stdout, stderr = self._run_subprocess(
            cmd, working_dir, timeout
        )
        
        duration = time.time() - start_time
        
        return ContainerExecutionResult(
            success=code == 0,
            return_code=code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration
        )
    
    def exec_in_container(
        self,
        container_id: str,
        command: List[str]
    ) -> ContainerExecutionResult:
        """Singularity不支持在运行中的实例内执行（不同于Docker）"""
        # Singularity实例功能较为有限
        logger.warning("Singularity exec_in_container not fully supported")
        return ContainerExecutionResult(
            success=False,
            stderr="Not supported"
        )
    
    def cleanup(self, container_id: str = None):
        """Singularity清理（主要是缓存管理）"""
        # Singularity会自动清理，这里可以添加缓存清理逻辑
        pass


class DockerRuntime(ContainerRuntime):
    """Docker运行时（rootless模式）"""
    
    def __init__(self, config: ContainerConfig):
        super().__init__(config)
        self._container_id: Optional[str] = None
    
    def is_available(self) -> bool:
        """检查Docker是否可用"""
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True
        )
        return result.returncode == 0
    
    def pull_image(self, force: bool = False) -> bool:
        """拉取Docker镜像"""
        cmd = ["docker", "pull", self.config.image.full_name]
        
        logger.info(f"Pulling image: {self.config.image.full_name}")
        
        code, stdout, stderr = self._run_subprocess(cmd, timeout=600)
        
        if code == 0:
            logger.info(f"Image pulled successfully")
            return True
        else:
            logger.error(f"Failed to pull image: {stderr}")
            return False
    
    def build_command(
        self,
        command: List[str],
        interactive: bool = False,
        detach: bool = False
    ) -> List[str]:
        """构建Docker执行命令"""
        cmd = ["docker", "run"]
        
        # 后台运行
        if detach:
            cmd.append("-d")
        
        # 交互式
        if interactive:
            cmd.extend(["-it"])
        
        # 清理
        if self.config.cleanup:
            cmd.append("--rm")
        
        # 名称
        cmd.extend(["--name", f"dft_job_{int(time.time())}"])
        
        # 资源限制
        if self.config.cpus:
            cmd.extend(["--cpus", str(self.config.cpus)])
        
        if self.config.memory_gb:
            cmd.extend(["--memory", f"{self.config.memory_gb}g"])
        
        # GPU
        if self.config.gpus:
            if isinstance(self.config.gpus, list):
                gpu_str = ",".join([f"device={g}" for g in self.config.gpus])
            else:
                gpu_str = f"all"
            cmd.extend(["--gpus", f"'\"{gpu_str}\"'"])
        
        # 挂载
        for host_path, container_path in self.config.bind_mounts.items():
            cmd.extend(["-v", f"{host_path}:{container_path}"])
        
        # 环境变量
        for key, value in self.config.environment.items():
            cmd.extend(["-e", f"{key}={value}"])
        
        # 工作目录
        if self.config.workdir:
            cmd.extend(["-w", self.config.workdir])
        
        # 网络
        cmd.extend(["--network", self.config.network])
        
        # 用户
        if self.config.user:
            cmd.extend(["-u", self.config.user])
        
        # 特权模式
        if self.config.privileged:
            cmd.append("--privileged")
        
        # 镜像
        cmd.append(self.config.image.full_name)
        
        # 命令
        if command:
            cmd.extend(command)
        
        return cmd
    
    def run(
        self,
        command: List[str],
        working_dir: Path = None,
        interactive: bool = False,
        timeout: int = None
    ) -> ContainerExecutionResult:
        """运行Docker容器"""
        start_time = time.time()
        
        # 拉取镜像
        if not self.pull_image():
            return ContainerExecutionResult(
                success=False,
                return_code=-1,
                stderr="Failed to pull image"
            )
        
        # 构建命令
        cmd = self.build_command(command, interactive)
        
        # 执行
        code, stdout, stderr = self._run_subprocess(
            cmd, working_dir, timeout
        )
        
        duration = time.time() - start_time
        
        # 解析容器ID（如果是后台运行）
        container_id = stdout.strip() if code == 0 else None
        self._container_id = container_id
        
        return ContainerExecutionResult(
            success=code == 0,
            return_code=code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            container_id=container_id
        )
    
    def exec_in_container(
        self,
        container_id: str,
        command: List[str]
    ) -> ContainerExecutionResult:
        """在运行中的容器内执行命令"""
        cmd = ["docker", "exec", container_id] + command
        
        code, stdout, stderr = self._run_subprocess(cmd)
        
        return ContainerExecutionResult(
            success=code == 0,
            return_code=code,
            stdout=stdout,
            stderr=stderr
        )
    
    def cleanup(self, container_id: str = None):
        """清理Docker容器"""
        cid = container_id or self._container_id
        
        if cid:
            # 停止容器
            subprocess.run(
                ["docker", "stop", cid],
                capture_output=True
            )
            # 删除容器
            subprocess.run(
                ["docker", "rm", cid],
                capture_output=True
            )
            logger.debug(f"Cleaned up container: {cid}")


def pull_image(
    image_string: str,
    engine: ContainerEngine = ContainerEngine.SINGULARITY,
    cache_dir: Path = None
) -> bool:
    """
    便捷函数：拉取镜像
    
    Args:
        image_string: 镜像字符串 (如 "docker://ubuntu:20.04")
        engine: 容器引擎
        cache_dir: 缓存目录
    
    Returns:
        是否成功
    """
    image = ContainerImage.from_string(image_string)
    
    config = ContainerImage(image=image, engine=engine)
    
    if engine == ContainerEngine.SINGULARITY or engine == ContainerEngine.APPTAINER:
        runtime = SingularityRuntime(config, cache_dir)
    elif engine == ContainerEngine.DOCKER:
        runtime = DockerRuntime(config)
    else:
        raise ValueError(f"Unsupported engine: {engine}")
    
    return runtime.pull_image()


def detect_available_runtime() -> Optional[ContainerEngine]:
    """
    检测可用的容器运行时
    
    Returns:
        可用的容器引擎或None
    """
    # 优先Singularity/Apptainer（HPC环境）
    for cmd, engine in [("apptainer", ContainerEngine.APPTAINER),
                        ("singularity", ContainerEngine.SINGULARITY)]:
        result = subprocess.run(["which", cmd], capture_output=True)
        if result.returncode == 0:
            return engine
    
    # 其次Docker
    result = subprocess.run(["docker", "version"], capture_output=True)
    if result.returncode == 0:
        return ContainerEngine.DOCKER
    
    # 最后Podman
    result = subprocess.run(["podman", "version"], capture_output=True)
    if result.returncode == 0:
        return ContainerEngine.PODMAN
    
    return None


def get_runtime(
    config: ContainerConfig,
    image_cache_dir: Path = None
) -> ContainerRuntime:
    """
    获取合适的运行时实例
    
    Args:
        config: 容器配置
        image_cache_dir: 镜像缓存目录
    
    Returns:
        ContainerRuntime实例
    """
    if config.engine == ContainerEngine.SINGULARITY or config.engine == ContainerEngine.APPTAINER:
        return SingularityRuntime(config, image_cache_dir)
    elif config.engine == ContainerEngine.DOCKER:
        return DockerRuntime(config)
    else:
        raise ValueError(f"Unsupported container engine: {config.engine}")
