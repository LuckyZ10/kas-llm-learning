#!/usr/bin/env python3
"""
Real-time Global Simulation
实时全球模拟系统 - 分布式实时分子动力学与量子计算模拟

Features:
- Distributed simulation orchestration
- Real-time visualization streaming
- Multi-scale simulation coupling
- Resource-aware task scheduling
- Live result aggregation and analysis
- Global simulation dashboard

Author: DFTLammps Platform
Version: 2.0.0
"""

import asyncio
import json
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any, Union, Set, Tuple
from enum import Enum, auto
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
import random
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimulationType(Enum):
    """模拟类型"""
    MOLECULAR_DYNAMICS = "md"          # 分子动力学
    QUANTUM_CHEMISTRY = "qc"           # 量子化学
    MONTE_CARLO = "mc"                 # 蒙特卡洛
    FINITE_ELEMENT = "fe"              # 有限元
    MULTISCALE = "multiscale"          # 多尺度
    MACHINE_LEARNING = "ml"            # 机器学习势


class SimulationStatus(Enum):
    """模拟状态"""
    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SimulationDomain(Enum):
    """模拟区域"""
    ASIA = "asia"
    EUROPE = "europe"
    NORTH_AMERICA = "north_america"
    SOUTH_AMERICA = "south_america"
    AFRICA = "africa"
    OCEANIA = "oceania"


@dataclass
class SimulationParameters:
    """模拟参数"""
    # 基本参数
    temperature: float = 300.0  # K
    pressure: float = 1.0  # bar
    timestep: float = 1.0  # fs
    
    # 运行参数
    n_steps: int = 10000
    n_equilibration: int = 1000
    
    # 输出参数
    output_frequency: int = 100
    trajectory_format: str = "xyz"
    
    # 算法参数
    ensemble: str = "NPT"  # NVE, NVT, NPT
    integrator: str = "velocity_verlet"
    thermostat: str = "nose_hoover"
    barostat: Optional[str] = " Parrinello_rahman"
    
    # 力场参数
    force_field: str = "reaxff"
    charge_method: str = "qeq"
    
    # 并行参数
    n_mpi: int = 1
    n_omp: int = 1
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SimulationJob:
    """模拟作业"""
    job_id: str
    name: str
    simulation_type: SimulationType
    description: str = ""
    
    # 输入
    input_structure: Optional[str] = None
    input_file_path: Optional[str] = None
    parameters: SimulationParameters = field(default_factory=SimulationParameters)
    
    # 资源要求
    required_cores: int = 1
    required_memory_gb: float = 4.0
    required_gpus: int = 0
    estimated_duration_hours: float = 1.0
    
    # 状态
    status: SimulationStatus = SimulationStatus.QUEUED
    
    # 分配
    assigned_domain: Optional[SimulationDomain] = None
    assigned_node: Optional[str] = None
    
    # 进度
    current_step: int = 0
    total_steps: int = 0
    progress_percent: float = 0.0
    
    # 时间
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 结果
    output_files: List[str] = field(default_factory=list)
    trajectory_data: List[Dict] = field(default_factory=list)
    analysis_results: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.job_id:
            self.job_id = hashlib.md5(
                f"{self.name}-{time.time()}".encode()
            ).hexdigest()[:16]
        
        self.total_steps = self.parameters.n_steps
    
    def update_progress(self, step: int):
        """更新进度"""
        self.current_step = step
        self.progress_percent = (step / max(self.total_steps, 1)) * 100


@dataclass
class SimulationFrame:
    """模拟帧数据"""
    job_id: str
    step: int
    timestamp: datetime
    
    # 原子数据
    positions: np.ndarray = field(default_factory=lambda: np.array([]))
    velocities: Optional[np.ndarray] = None
    forces: Optional[np.ndarray] = None
    
    # 系统属性
    temperature: float = 0.0
    pressure: float = 0.0
    potential_energy: float = 0.0
    kinetic_energy: float = 0.0
    total_energy: float = 0.0
    
    # 元数据
    box_vectors: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "step": self.step,
            "timestamp": self.timestamp.isoformat(),
            "temperature": self.temperature,
            "pressure": self.pressure,
            "potential_energy": self.potential_energy,
            "kinetic_energy": self.kinetic_energy,
            "total_energy": self.total_energy,
        }


@dataclass
class SimulationNode:
    """模拟计算节点"""
    node_id: str
    domain: SimulationDomain
    name: str
    location: str
    
    # 资源
    total_cores: int
    total_memory_gb: float
    total_gpus: int = 0
    gpu_type: Optional[str] = None
    
    # 状态
    available_cores: int = 0
    available_memory_gb: float = 0.0
    available_gpus: int = 0
    active_jobs: int = 0
    
    # 性能
    avg_job_duration: float = 0.0
    success_rate: float = 1.0
    
    # 网络
    latency_to_regions: Dict[str, float] = field(default_factory=dict)
    bandwidth_mbps: float = 1000.0
    
    def __post_init__(self):
        self.available_cores = self.total_cores
        self.available_memory_gb = self.total_memory_gb
        self.available_gpus = self.total_gpus
    
    def can_accommodate(self, job: SimulationJob) -> bool:
        """检查是否能容纳作业"""
        return (
            self.available_cores >= job.required_cores and
            self.available_memory_gb >= job.required_memory_gb and
            self.available_gpus >= job.required_gpus
        )
    
    def allocate(self, job: SimulationJob):
        """分配资源"""
        self.available_cores -= job.required_cores
        self.available_memory_gb -= job.required_memory_gb
        self.available_gpus -= job.required_gpus
        self.active_jobs += 1
    
    def release(self, job: SimulationJob):
        """释放资源"""
        self.available_cores += job.required_cores
        self.available_memory_gb += job.required_memory_gb
        self.available_gpus += job.required_gpus
        self.active_jobs -= 1


@dataclass
class RealTimeStream:
    """实时数据流"""
    stream_id: str
    job_id: str
    
    # 订阅者
    subscribers: List[Callable] = field(default_factory=list)
    
    # 缓冲
    frame_buffer: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # 统计
    frames_sent: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)
    
    def subscribe(self, callback: Callable):
        """订阅流"""
        self.subscribers.append(callback)
    
    async def publish_frame(self, frame: SimulationFrame):
        """发布帧"""
        self.frame_buffer.append(frame)
        
        for callback in self.subscribers:
            try:
                await callback(frame)
            except Exception as e:
                logger.error(f"Stream callback error: {e}")
        
        self.frames_sent += 1


class SimulationEngine:
    """模拟引擎 (模拟 LAMMPS/Quantum ESPRESSO 等)"""
    
    def __init__(self):
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self.job_streams: Dict[str, RealTimeStream] = {}
    
    async def run_simulation(self, job: SimulationJob) -> bool:
        """运行模拟"""
        if job.job_id in self.running_jobs:
            return False
        
        task = asyncio.create_task(self._execute_job(job))
        self.running_jobs[job.job_id] = task
        
        return True
    
    async def _execute_job(self, job: SimulationJob):
        """执行模拟 (模拟)"""
        try:
            job.status = SimulationStatus.RUNNING
            job.started_at = datetime.utcnow()
            
            # 创建实时流
            stream = RealTimeStream(
                stream_id=f"stream-{job.job_id}",
                job_id=job.job_id
            )
            self.job_streams[job.job_id] = stream
            
            # 模拟运行
            n_atoms = 100  # 假设100个原子
            
            for step in range(job.parameters.n_steps):
                # 检查是否取消
                if job.status == SimulationStatus.CANCELLED:
                    break
                
                # 模拟计算时间
                await asyncio.sleep(0.001)  # 每步1ms
                
                # 生成模拟数据
                frame = self._generate_frame(job, step, n_atoms)
                
                # 按频率输出
                if step % job.parameters.output_frequency == 0:
                    job.trajectory_data.append(frame.to_dict())
                    await stream.publish_frame(frame)
                
                # 更新进度
                job.update_progress(step)
                
                # 每100步记录一次
                if step % 100 == 0:
                    logger.debug(f"Job {job.job_id}: step {step}/{job.total_steps}")
            
            # 完成
            if job.status != SimulationStatus.CANCELLED:
                job.status = SimulationStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                
                # 执行分析
                job.analysis_results = self._analyze_results(job)
                
                logger.info(f"Job {job.job_id} completed")
            
        except Exception as e:
            job.status = SimulationStatus.FAILED
            logger.error(f"Job {job.job_id} failed: {e}")
        
        finally:
            if job.job_id in self.running_jobs:
                del self.running_jobs[job.job_id]
            if job.job_id in self.job_streams:
                del self.job_streams[job.job_id]
    
    def _generate_frame(self, job: SimulationJob, step: int, n_atoms: int) -> SimulationFrame:
        """生成模拟帧 (模拟数据)"""
        # 生成随机位置
        positions = np.random.randn(n_atoms, 3) * 10
        
        # 模拟能量
        base_energy = -1000.0
        noise = np.random.randn() * 10
        potential_energy = base_energy + noise + np.sin(step * 0.01) * 5
        kinetic_energy = 50 + np.random.randn() * 5
        
        return SimulationFrame(
            job_id=job.job_id,
            step=step,
            timestamp=datetime.utcnow(),
            positions=positions,
            temperature=job.parameters.temperature + np.random.randn() * 5,
            pressure=job.parameters.pressure + np.random.randn() * 0.5,
            potential_energy=potential_energy,
            kinetic_energy=kinetic_energy,
            total_energy=potential_energy + kinetic_energy
        )
    
    def _analyze_results(self, job: SimulationJob) -> Dict:
        """分析结果"""
        if not job.trajectory_data:
            return {}
        
        energies = [f.get("total_energy", 0) for f in job.trajectory_data]
        temperatures = [f.get("temperature", 0) for f in job.trajectory_data]
        
        return {
            "avg_energy": sum(energies) / len(energies),
            "energy_std": np.std(energies) if len(energies) > 1 else 0,
            "avg_temperature": sum(temperatures) / len(temperatures),
            "temperature_stability": np.std(temperatures) if len(temperatures) > 1 else 0,
            "n_frames": len(job.trajectory_data)
        }
    
    async def cancel_job(self, job_id: str) -> bool:
        """取消作业"""
        task = self.running_jobs.get(job_id)
        if task:
            task.cancel()
            return True
        return False
    
    def get_stream(self, job_id: str) -> Optional[RealTimeStream]:
        """获取实时流"""
        return self.job_streams.get(job_id)


class GlobalScheduler:
    """全局调度器"""
    
    def __init__(self):
        self.nodes: Dict[str, SimulationNode] = {}
        self.jobs: Dict[str, SimulationJob] = {}
        self.queue: deque = deque()
        self._lock = asyncio.Lock()
    
    def register_node(self, node: SimulationNode):
        """注册节点"""
        self.nodes[node.node_id] = node
        logger.info(f"Registered node: {node.name} ({node.domain.value})")
    
    async def submit_job(self, job: SimulationJob) -> str:
        """提交作业"""
        async with self._lock:
            self.jobs[job.job_id] = job
            self.queue.append(job)
        
        logger.info(f"Job submitted: {job.name} ({job.job_id})")
        return job.job_id
    
    async def schedule(self) -> List[Tuple[SimulationJob, SimulationNode]]:
        """调度作业"""
        scheduled = []
        
        async with self._lock:
            to_remove = []
            
            for job in list(self.queue):
                # 查找合适的节点
                node = self._find_best_node(job)
                
                if node:
                    # 分配
                    node.allocate(job)
                    job.assigned_node = node.node_id
                    job.assigned_domain = node.domain
                    job.status = SimulationStatus.PREPARING
                    
                    scheduled.append((job, node))
                    to_remove.append(job)
            
            for job in to_remove:
                self.queue.remove(job)
        
        return scheduled
    
    def _find_best_node(self, job: SimulationJob) -> Optional[SimulationNode]:
        """查找最佳节点"""
        candidates = []
        
        for node in self.nodes.values():
            if node.can_accommodate(job):
                # 计算分数
                score = self._calculate_node_score(node, job)
                candidates.append((node, score))
        
        if not candidates:
            return None
        
        # 选择分数最高的
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def _calculate_node_score(self, node: SimulationNode, job: SimulationJob) -> float:
        """计算节点分数"""
        score = 0.0
        
        # 资源匹配度
        score += (node.available_cores / max(job.required_cores, 1)) * 10
        score += (node.available_memory_gb / max(job.required_memory_gb, 1)) * 10
        
        # GPU匹配
        if job.required_gpus > 0 and node.available_gpus >= job.required_gpus:
            score += 50
        
        # 负载均衡 (负载越低分数越高)
        load_ratio = node.active_jobs / max(len(self.jobs), 1)
        score += (1 - load_ratio) * 20
        
        # 成功率
        score += node.success_rate * 10
        
        return score
    
    async def release_job(self, job_id: str):
        """释放作业资源"""
        job = self.jobs.get(job_id)
        if job and job.assigned_node:
            node = self.nodes.get(job.assigned_node)
            if node:
                node.release(job)
    
    def get_queue_stats(self) -> Dict:
        """获取队列统计"""
        return {
            "queued": len(self.queue),
            "total_jobs": len(self.jobs),
            "active_nodes": len([
                n for n in self.nodes.values()
                if n.active_jobs > 0
            ]),
            "available_cores": sum(
                n.available_cores for n in self.nodes.values()
            ),
            "available_gpus": sum(
                n.available_gpus for n in self.nodes.values()
            )
        }


class ResultAggregator:
    """结果聚合器"""
    
    def __init__(self):
        self.results: Dict[str, Dict] = {}
        self.global_metrics: Dict[str, List[float]] = defaultdict(list)
    
    def aggregate(self, job: SimulationJob):
        """聚合作业结果"""
        self.results[job.job_id] = {
            "job_id": job.job_id,
            "name": job.name,
            "status": job.status.value,
            "analysis": job.analysis_results,
            "duration_seconds": (
                (job.completed_at - job.started_at).total_seconds()
                if job.completed_at and job.started_at else 0
            )
        }
        
        # 更新全局指标
        if job.analysis_results:
            for key, value in job.analysis_results.items():
                if isinstance(value, (int, float)):
                    self.global_metrics[key].append(value)
    
    def get_global_stats(self) -> Dict:
        """获取全局统计"""
        return {
            "completed_simulations": len(self.results),
            "avg_duration": np.mean([
                r["duration_seconds"] for r in self.results.values()
                if r["duration_seconds"] > 0
            ]) if self.results else 0,
            "metrics": {
                key: {
                    "avg": np.mean(values),
                    "std": np.std(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values)
                }
                for key, values in self.global_metrics.items()
                if values
            }
        }


class GlobalSimulationOrchestrator:
    """全球模拟编排器"""
    
    def __init__(self):
        self.scheduler = GlobalScheduler()
        self.engine = SimulationEngine()
        self.aggregator = ResultAggregator()
        
        self._running = False
        self._scheduling_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动编排器"""
        self._running = True
        self._scheduling_task = asyncio.create_task(self._scheduling_loop())
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Global simulation orchestrator started")
    
    async def stop(self):
        """停止编排器"""
        self._running = False
        
        if self._scheduling_task:
            self._scheduling_task.cancel()
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        logger.info("Global simulation orchestrator stopped")
    
    async def _scheduling_loop(self):
        """调度循环"""
        while self._running:
            try:
                # 调度作业
                scheduled = await self.scheduler.schedule()
                
                for job, node in scheduled:
                    # 启动模拟
                    await self.engine.run_simulation(job)
                
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduling error: {e}")
                await asyncio.sleep(5)
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self._running:
            try:
                # 检查完成的作业
                for job_id, job in list(self.scheduler.jobs.items()):
                    if job.status in [SimulationStatus.COMPLETED, SimulationStatus.FAILED]:
                        # 释放资源
                        await self.scheduler.release_job(job_id)
                        # 聚合结果
                        self.aggregator.aggregate(job)
                
                await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def submit_simulation(
        self,
        name: str,
        simulation_type: SimulationType,
        parameters: SimulationParameters,
        required_cores: int = 4,
        required_memory_gb: float = 16.0
    ) -> str:
        """提交模拟"""
        job = SimulationJob(
            job_id="",
            name=name,
            simulation_type=simulation_type,
            parameters=parameters,
            required_cores=required_cores,
            required_memory_gb=required_memory_gb,
            total_steps=parameters.n_steps
        )
        
        return await self.scheduler.submit_job(job)
    
    async def subscribe_to_stream(
        self,
        job_id: str,
        callback: Callable[[SimulationFrame], Any]
    ) -> bool:
        """订阅实时流"""
        stream = self.engine.get_stream(job_id)
        if stream:
            stream.subscribe(callback)
            return True
        return False
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """获取作业状态"""
        job = self.scheduler.jobs.get(job_id)
        if not job:
            return None
        
        return {
            "job_id": job.job_id,
            "name": job.name,
            "status": job.status.value,
            "progress": job.progress_percent,
            "assigned_node": job.assigned_node,
            "assigned_domain": job.assigned_domain.value if job.assigned_domain else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "analysis": job.analysis_results
        }
    
    def get_global_status(self) -> Dict:
        """获取全局状态"""
        return {
            "scheduler": self.scheduler.get_queue_stats(),
            "results": self.aggregator.get_global_stats(),
            "active_streams": len(self.engine.job_streams),
            "nodes": [
                {
                    "node_id": n.node_id,
                    "name": n.name,
                    "domain": n.domain.value,
                    "active_jobs": n.active_jobs,
                    "available_cores": n.available_cores,
                    "available_gpus": n.available_gpus
                }
                for n in self.scheduler.nodes.values()
            ]
        }


# 示例使用
async def demo():
    """实时全球模拟演示"""
    
    orchestrator = GlobalSimulationOrchestrator()
    
    # 注册全球节点
    print("=== 注册全球计算节点 ===")
    
    nodes = [
        SimulationNode(
            node_id="na-hpc-01",
            domain=SimulationDomain.NORTH_AMERICA,
            name="NERSC Cori",
            location="Berkeley, USA",
            total_cores=1000,
            total_memory_gb=4000,
            total_gpus=100,
            gpu_type="NVIDIA V100"
        ),
        SimulationNode(
            node_id="eu-hpc-01",
            domain=SimulationDomain.EUROPE,
            name="CSCS Piz Daint",
            location="Lugano, Switzerland",
            total_cores=2000,
            total_memory_gb=8000,
            total_gpus=200,
            gpu_type="NVIDIA A100"
        ),
        SimulationNode(
            node_id="asia-hpc-01",
            domain=SimulationDomain.ASIA,
            name="RIKEN Fugaku",
            location="Kobe, Japan",
            total_cores=5000,
            total_memory_gb=20000,
            total_gpus=0
        ),
    ]
    
    for node in nodes:
        orchestrator.scheduler.register_node(node)
        print(f"  {node.name} ({node.domain.value}):")
        print(f"    Cores: {node.total_cores}, Memory: {node.total_memory_gb}GB, GPUs: {node.total_gpus}")
    
    # 启动编排器
    await orchestrator.start()
    
    try:
        # 提交模拟作业
        print("\n=== 提交模拟作业 ===")
        
        jobs_config = [
            {
                "name": "Li-ion Battery Cathode MD",
                "type": SimulationType.MOLECULAR_DYNAMICS,
                "parameters": SimulationParameters(
                    temperature=300,
                    pressure=1,
                    n_steps=5000,
                    output_frequency=100,
                    ensemble="NPT"
                ),
                "cores": 64,
                "memory": 128
            },
            {
                "name": "Catalyst Surface DFT",
                "type": SimulationType.QUANTUM_CHEMISTRY,
                "parameters": SimulationParameters(
                    n_steps=1000,
                    output_frequency=50
                ),
                "cores": 128,
                "memory": 256,
                "gpus": 4
            },
            {
                "name": "Polymer Chain MC",
                "type": SimulationType.MONTE_CARLO,
                "parameters": SimulationParameters(
                    temperature=400,
                    n_steps=10000,
                    output_frequency=200
                ),
                "cores": 32,
                "memory": 64
            },
        ]
        
        job_ids = []
        for config in jobs_config:
            jid = await orchestrator.submit_simulation(
                config["name"],
                config["type"],
                config["parameters"],
                config["cores"],
                config["memory"]
            )
            job_ids.append(jid)
            print(f"  Submitted: {config['name']} ({jid[:8]}...)")
            print(f"    Type: {config['type'].value}, Cores: {config['cores']}")
        
        # 设置实时流订阅
        print("\n=== 实时流订阅 ===")
        
        async def stream_callback(frame: SimulationFrame):
            if frame.step % 500 == 0:
                print(f"  [{frame.job_id[:8]}] Step {frame.step}: "
                      f"E={frame.total_energy:.2f}, T={frame.temperature:.1f}K")
        
        for jid in job_ids:
            success = await orchestrator.subscribe_to_stream(jid, stream_callback)
            if success:
                print(f"  Subscribed to stream: {jid[:8]}...")
        
        # 等待模拟完成
        print("\n=== 模拟运行中 ===")
        
        for _ in range(30):  # 等待30秒
            await asyncio.sleep(1)
            
            # 打印状态
            if _ % 5 == 0:
                status = orchestrator.get_global_status()
                print(f"  Time {_}s: {status['scheduler']['queued']} queued, "
                      f"{status['active_streams']} active streams")
        
        # 查看结果
        print("\n=== 模拟结果 ===")
        
        for jid in job_ids:
            status = orchestrator.get_job_status(jid)
            if status:
                print(f"\n  {status['name']}:")
                print(f"    Status: {status['status']}")
                print(f"    Progress: {status['progress']:.1f}%")
                print(f"    Domain: {status['assigned_domain']}")
                
                if status['analysis']:
                    print(f"    Analysis:")
                    for key, value in status['analysis'].items():
                        if isinstance(value, (int, float)):
                            print(f"      {key}: {value:.4f}")
        
        # 全局统计
        print("\n=== 全局统计 ===")
        global_status = orchestrator.get_global_status()
        print(f"  Total simulations: {global_status['results']['completed_simulations']}")
        print(f"  Average duration: {global_status['results']['avg_duration']:.1f}s")
        
        if global_status['results']['metrics']:
            print("\n  Aggregated metrics:")
            for metric, stats in global_status['results']['metrics'].items():
                print(f"    {metric}: avg={stats['avg']:.4f}, std={stats['std']:.4f}")
        
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(demo())
