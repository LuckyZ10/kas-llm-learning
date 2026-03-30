#!/usr/bin/env python3
"""
Edge Deployment Module
边缘部署模块 - 提供边缘节点管理、低延迟推理和数据本地化

Features:
- Edge node management and discovery
- Low-latency inference at the edge
- Data locality enforcement
- Edge-cloud synchronization
- Bandwidth-aware task distribution

Author: DFTLammps Platform
Version: 2.0.0
"""

import asyncio
import json
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any, Union, Set, Tuple
from enum import Enum, auto
from collections import defaultdict, deque
import logging
from datetime import datetime, timedelta
import random
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EdgeNodeStatus(Enum):
    """边缘节点状态"""
    OFFLINE = "offline"
    ONLINE = "online"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"


class InferenceEngine(Enum):
    """推理引擎类型"""
    TENSORRT = "tensorrt"
    ONNX_RUNTIME = "onnx_runtime"
    TFLITE = "tflite"
    PYTORCH_MOBILE = "pytorch_mobile"
    COREML = "coreml"
    OPENVINO = "openvino"


class DataLocalityPolicy(Enum):
    """数据本地化策略"""
    STRICT = "strict"      # 数据必须留在本地
    PREFERRED = "preferred"  # 优先本地处理
    BALANCED = "balanced"    # 平衡考虑
    FLEXIBLE = "flexible"    # 灵活，可传输到云端


@dataclass
class GeoLocation:
    """地理位置信息"""
    lat: float
    lon: float
    altitude: float = 0.0
    accuracy: float = 10.0  # meters
    
    def distance_to(self, other: 'GeoLocation') -> float:
        """计算与另一个位置的距离 (km)"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # 地球半径 (km)
        lat1, lon1 = radians(self.lat), radians(self.lon)
        lat2, lon2 = radians(other.lat), radians(other.lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def to_dict(self) -> Dict:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "altitude": self.altitude,
            "accuracy": self.accuracy
        }


@dataclass
class HardwareCapabilities:
    """硬件能力信息"""
    cpu_cores: int
    cpu_frequency_ghz: float
    memory_gb: float
    storage_gb: float
    has_gpu: bool = False
    gpu_model: Optional[str] = None
    gpu_memory_gb: float = 0.0
    has_npu: bool = False  # Neural Processing Unit
    npu_tops: float = 0.0  # Tera Operations Per Second
    has_fpga: bool = False
    network_bandwidth_mbps: float = 100.0
    network_latency_ms: float = 10.0
    
    def inference_score(self, engine: InferenceEngine) -> float:
        """计算推理性能分数"""
        score = self.cpu_cores * self.cpu_frequency_ghz * 10
        
        if self.has_gpu and engine in [InferenceEngine.TENSORRT, InferenceEngine.ONNX_RUNTIME]:
            gpu_score = self.gpu_memory_gb * 50
            if "A100" in (self.gpu_model or ""):
                gpu_score *= 5
            elif "H100" in (self.gpu_model or ""):
                gpu_score *= 8
            elif "T4" in (self.gpu_model or ""):
                gpu_score *= 1.5
            score += gpu_score
        
        if self.has_npu:
            score += self.npu_tops * 100
        
        return score
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ModelInfo:
    """模型信息"""
    model_id: str
    name: str
    version: str
    framework: str
    engine: InferenceEngine
    input_shape: List[int]
    output_shape: List[int]
    model_size_mb: float
    latency_ms_p50: float  # 50th percentile latency
    latency_ms_p99: float  # 99th percentile latency
    throughput_qps: float  # Queries per second
    accuracy: float  # Model accuracy metric
    quantization: Optional[str] = None  # int8, fp16, etc.
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EdgeNode:
    """边缘节点定义"""
    node_id: str
    name: str
    location: GeoLocation
    hardware: HardwareCapabilities
    status: EdgeNodeStatus = EdgeNodeStatus.OFFLINE
    data_locality_policy: DataLocalityPolicy = DataLocalityPolicy.PREFERRED
    
    # 运行时信息
    current_load: float = 0.0  # 0-1
    active_tasks: int = 0
    last_heartbeat: Optional[datetime] = None
    uptime_seconds: float = 0.0
    
    # 数据位置
    local_data_paths: List[str] = field(default_factory=list)
    data_replication_factor: int = 2
    
    # 网络信息
    connected_gateway: Optional[str] = None
    peers: List[str] = field(default_factory=list)  # 相邻节点ID
    
    # 元数据
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = hashlib.md5(
                f"{self.name}-{time.time()}".encode()
            ).hexdigest()[:16]
    
    def is_healthy(self) -> bool:
        """检查节点健康状态"""
        if self.status in [EdgeNodeStatus.OFFLINE, EdgeNodeStatus.MAINTENANCE]:
            return False
        
        if self.last_heartbeat:
            elapsed = (datetime.utcnow() - self.last_heartbeat).total_seconds()
            if elapsed > 60:  # 超过60秒无心跳
                return False
        
        return True
    
    def available_capacity(self) -> float:
        """获取可用容量 (0-1)"""
        return 1.0 - self.current_load
    
    def can_handle_task(self, required_qps: float, model: ModelInfo) -> bool:
        """检查是否能处理任务"""
        if not self.is_healthy():
            return False
        
        if self.current_load > 0.9:
            return False
        
        # 检查硬件兼容性
        score = self.hardware.inference_score(model.engine)
        min_score = required_qps * model.latency_ms_p50 / 1000
        
        return score >= min_score
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "location": self.location.to_dict(),
            "hardware": self.hardware.to_dict(),
            "status": self.status.value,
            "current_load": self.current_load,
            "active_tasks": self.active_tasks,
            "is_healthy": self.is_healthy(),
            "available_capacity": self.available_capacity(),
            "tags": self.tags
        }


@dataclass
class InferenceRequest:
    """推理请求"""
    request_id: str
    model_id: str
    input_data: Any
    input_location: Optional[GeoLocation] = None
    priority: int = 5
    max_latency_ms: float = 100.0
    min_confidence: float = 0.8
    require_local_processing: bool = False
    data_size_mb: float = 0.0
    
    # QoS要求
    required_accuracy: Optional[float] = None
    encryption_required: bool = False
    audit_log_required: bool = False
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = hashlib.md5(
                f"{time.time()}-{random.randint(0,1000000)}".encode()
            ).hexdigest()[:16]


@dataclass
class InferenceResult:
    """推理结果"""
    request_id: str
    node_id: str
    model_id: str
    output_data: Any
    confidence: float
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict = field(default_factory=dict)


@dataclass
class DataPlacementPolicy:
    """数据放置策略"""
    policy_id: str
    data_pattern: str  # 数据模式，如 "sensor-*", "user-{region}"
    primary_nodes: List[str]  # 主节点
    replica_nodes: List[str]  # 副本节点
    retention_days: int = 30
    compression: str = "lz4"
    encryption: str = "aes-256"
    access_pattern: str = "hot"  # hot, warm, cold


class EdgeNodeManager:
    """边缘节点管理器"""
    
    def __init__(self):
        self.nodes: Dict[str, EdgeNode] = {}
        self._lock = asyncio.Lock()
        self._heartbeat_interval = 30  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable] = []
    
    def register_callback(self, callback: Callable):
        """注册节点状态变化回调"""
        self._callbacks.append(callback)
    
    async def start(self):
        """启动管理器"""
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Edge node manager started")
    
    async def stop(self):
        """停止管理器"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                await self._check_nodes_health()
                await asyncio.sleep(self._heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)
    
    async def _check_nodes_health(self):
        """检查节点健康状态"""
        async with self._lock:
            for node_id, node in self.nodes.items():
                if node.last_heartbeat:
                    elapsed = (datetime.utcnow() - node.last_heartbeat).total_seconds()
                    
                    if elapsed > 120:  # 超过2分钟无心跳
                        if node.status != EdgeNodeStatus.OFFLINE:
                            old_status = node.status
                            node.status = EdgeNodeStatus.OFFLINE
                            await self._notify_status_change(node, old_status)
                    elif elapsed > 60:  # 超过1分钟
                        if node.status == EdgeNodeStatus.ONLINE:
                            old_status = node.status
                            node.status = EdgeNodeStatus.DEGRADED
                            await self._notify_status_change(node, old_status)
    
    async def _notify_status_change(self, node: EdgeNode, old_status: EdgeNodeStatus):
        """通知状态变化"""
        logger.info(f"Node {node.node_id} status changed: {old_status.value} -> {node.status.value}")
        for callback in self._callbacks:
            try:
                await callback(node, old_status)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def register_node(self, node: EdgeNode) -> bool:
        """注册边缘节点"""
        async with self._lock:
            self.nodes[node.node_id] = node
            node.status = EdgeNodeStatus.ONLINE
            node.last_heartbeat = datetime.utcnow()
            logger.info(f"Registered edge node: {node.node_id} ({node.name})")
            return True
    
    async def unregister_node(self, node_id: str) -> bool:
        """注销边缘节点"""
        async with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logger.info(f"Unregistered edge node: {node_id}")
                return True
            return False
    
    async def update_heartbeat(self, node_id: str, metrics: Dict) -> bool:
        """更新节点心跳"""
        async with self._lock:
            if node_id not in self.nodes:
                return False
            
            node = self.nodes[node_id]
            node.last_heartbeat = datetime.utcnow()
            node.current_load = metrics.get("load", node.current_load)
            node.active_tasks = metrics.get("active_tasks", node.active_tasks)
            
            # 恢复在线状态
            if node.status == EdgeNodeStatus.OFFLINE or node.status == EdgeNodeStatus.DEGRADED:
                old_status = node.status
                node.status = EdgeNodeStatus.ONLINE
                await self._notify_status_change(node, old_status)
            
            return True
    
    def get_node(self, node_id: str) -> Optional[EdgeNode]:
        """获取节点信息"""
        return self.nodes.get(node_id)
    
    def find_nearest_nodes(
        self, 
        location: GeoLocation, 
        max_distance_km: float = 100.0,
        min_capacity: float = 0.1
    ) -> List[Tuple[EdgeNode, float]]:
        """查找最近的节点"""
        results = []
        
        for node in self.nodes.values():
            if not node.is_healthy():
                continue
            
            if node.available_capacity() < min_capacity:
                continue
            
            distance = node.location.distance_to(location)
            if distance <= max_distance_km:
                results.append((node, distance))
        
        results.sort(key=lambda x: x[1])
        return results
    
    def find_nodes_by_region(
        self, 
        region: str,
        status: Optional[EdgeNodeStatus] = None
    ) -> List[EdgeNode]:
        """按区域查找节点"""
        results = []
        for node in self.nodes.values():
            if region in node.tags.get("region", ""):
                if status is None or node.status == status:
                    results.append(node)
        return results
    
    def get_cluster_stats(self) -> Dict:
        """获取集群统计信息"""
        total_nodes = len(self.nodes)
        healthy_nodes = sum(1 for n in self.nodes.values() if n.is_healthy())
        
        status_counts = defaultdict(int)
        for node in self.nodes.values():
            status_counts[node.status.value] += 1
        
        total_capacity = sum(n.hardware.cpu_cores for n in self.nodes.values())
        available_capacity = sum(
            n.hardware.cpu_cores * n.available_capacity() 
            for n in self.nodes.values() if n.is_healthy()
        )
        
        return {
            "total_nodes": total_nodes,
            "healthy_nodes": healthy_nodes,
            "status_distribution": dict(status_counts),
            "total_capacity_cores": total_capacity,
            "available_capacity_cores": round(available_capacity, 2),
            "average_load": round(
                sum(n.current_load for n in self.nodes.values()) / max(total_nodes, 1), 
                2
            ),
            "total_active_tasks": sum(n.active_tasks for n in self.nodes.values())
        }


class LatencyAwareScheduler:
    """延迟感知调度器"""
    
    def __init__(self, node_manager: EdgeNodeManager):
        self.node_manager = node_manager
        self.latency_cache: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.model_deployments: Dict[str, List[str]] = defaultdict(list)  # model_id -> node_ids
    
    async def schedule_inference(
        self,
        request: InferenceRequest,
        available_models: Dict[str, ModelInfo]
    ) -> Optional[Tuple[EdgeNode, ModelInfo]]:
        """
        调度推理请求
        
        Returns: (选中的节点, 模型信息) 或 None
        """
        model = available_models.get(request.model_id)
        if not model:
            logger.error(f"Model {request.model_id} not found")
            return None
        
        # 查找候选节点
        candidates = self._find_candidates(request, model)
        
        if not candidates:
            logger.warning(f"No suitable node found for request {request.request_id}")
            return None
        
        # 评分并选择最佳节点
        best_node = self._select_best_node(candidates, request, model)
        
        return (best_node, model) if best_node else None
    
    def _find_candidates(
        self, 
        request: InferenceRequest, 
        model: ModelInfo
    ) -> List[EdgeNode]:
        """查找候选节点"""
        candidates = []
        
        # 基于请求位置查找最近节点
        if request.input_location:
            nearest = self.node_manager.find_nearest_nodes(
                request.input_location,
                max_distance_km=500.0
            )
            
            for node, distance in nearest:
                # 检查节点是否部署了该模型
                if node.node_id not in self.model_deployments.get(model.model_id, []):
                    # 需要检查是否可以部署
                    if not self._can_deploy_model(node, model):
                        continue
                
                # 检查是否可以处理
                required_qps = 1.0 / (request.max_latency_ms / 1000)
                if node.can_handle_task(required_qps, model):
                    candidates.append(node)
        
        # 如果没有位置信息或附近没有节点，考虑所有健康节点
        if not candidates:
            for node in self.node_manager.nodes.values():
                if node.is_healthy() and node.available_capacity() > 0.2:
                    candidates.append(node)
        
        return candidates
    
    def _can_deploy_model(self, node: EdgeNode, model: ModelInfo) -> bool:
        """检查节点是否可以部署模型"""
        # 检查存储空间
        if model.model_size_mb > node.hardware.storage_gb * 1024 * 0.1:  # 最多使用10%存储
            return False
        
        # 检查推理引擎兼容性
        engine_score = node.hardware.inference_score(model.engine)
        if engine_score < model.throughput_qps * model.latency_ms_p50 / 1000:
            return False
        
        return True
    
    def _select_best_node(
        self,
        candidates: List[EdgeNode],
        request: InferenceRequest,
        model: ModelInfo
    ) -> Optional[EdgeNode]:
        """选择最佳节点"""
        
        def score_node(node: EdgeNode) -> float:
            score = 0.0
            
            # 延迟分数 (越低越好)
            if request.input_location:
                distance = node.location.distance_to(request.input_location)
                # 假设每100km约增加5ms延迟
                estimated_latency = distance * 0.05
                latency_score = max(0, 100 - estimated_latency)
                score += latency_score * 0.4
            
            # 容量分数
            capacity_score = node.available_capacity() * 100
            score += capacity_score * 0.3
            
            # 模型已部署分数 (避免冷启动)
            if node.node_id in self.model_deployments.get(model.model_id, []):
                score += 20  # 模型已部署的奖励
            
            # 硬件匹配分数
            hw_score = node.hardware.inference_score(model.engine)
            score += min(hw_score / 100, 50) * 0.3
            
            return score
        
        # 按分数排序
        candidates.sort(key=score_node, reverse=True)
        return candidates[0] if candidates else None
    
    async def deploy_model(self, model: ModelInfo, node_ids: Optional[List[str]] = None) -> List[str]:
        """
        部署模型到边缘节点
        
        Returns: 成功部署的节点ID列表
        """
        deployed = []
        
        if node_ids:
            target_nodes = [self.node_manager.get_node(nid) for nid in node_ids]
        else:
            # 自动选择节点 (选择负载最低的)
            target_nodes = sorted(
                [n for n in self.node_manager.nodes.values() if n.is_healthy()],
                key=lambda n: n.current_load
            )[:5]  # 部署到前5个节点
        
        for node in target_nodes:
            if node and self._can_deploy_model(node, model):
                # 模拟部署
                logger.info(f"Deploying model {model.model_id} to node {node.node_id}")
                await asyncio.sleep(0.5)  # 模拟部署延迟
                
                if model.model_id not in self.model_deployments:
                    self.model_deployments[model.model_id] = []
                self.model_deployments[model.model_id].append(node.node_id)
                deployed.append(node.node_id)
        
        return deployed
    
    async def undeploy_model(self, model_id: str, node_ids: Optional[List[str]] = None) -> bool:
        """从边缘节点卸载模型"""
        if model_id not in self.model_deployments:
            return False
        
        if node_ids:
            self.model_deployments[model_id] = [
                nid for nid in self.model_deployments[model_id]
                if nid not in node_ids
            ]
        else:
            del self.model_deployments[model_id]
        
        return True


class DataLocalityManager:
    """数据本地化管理器"""
    
    def __init__(self, node_manager: EdgeNodeManager):
        self.node_manager = node_manager
        self.policies: Dict[str, DataPlacementPolicy] = {}
        self.data_registry: Dict[str, Dict[str, Any]] = {}  # data_id -> metadata
        self._replication_queue: asyncio.Queue = asyncio.Queue()
        self._sync_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动数据管理器"""
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info("Data locality manager started")
    
    async def stop(self):
        """停止数据管理器"""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
    
    async def _sync_loop(self):
        """数据同步循环"""
        while True:
            try:
                await self._process_replication_queue()
                await asyncio.sleep(60)  # 每分钟检查一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync error: {e}")
                await asyncio.sleep(10)
    
    async def _process_replication_queue(self):
        """处理复制队列"""
        while not self._replication_queue.empty():
            try:
                task = self._replication_queue.get_nowait()
                await self._replicate_data(task)
            except asyncio.QueueEmpty:
                break
    
    async def _replicate_data(self, task: Dict):
        """复制数据"""
        data_id = task["data_id"]
        source_node = task["source_node"]
        target_nodes = task["target_nodes"]
        
        logger.info(f"Replicating data {data_id} from {source_node} to {target_nodes}")
        # 实际实现会执行数据复制
        await asyncio.sleep(1)
        
        # 更新注册表
        if data_id in self.data_registry:
            self.data_registry[data_id]["replicas"] = target_nodes
    
    def register_policy(self, policy: DataPlacementPolicy):
        """注册数据放置策略"""
        self.policies[policy.policy_id] = policy
        logger.info(f"Registered data placement policy: {policy.policy_id}")
    
    def find_data_location(self, data_id: str) -> Optional[List[str]]:
        """查找数据位置"""
        if data_id in self.data_registry:
            return self.data_registry[data_id].get("replicas", [])
        return None
    
    def find_nearest_data_replica(
        self, 
        data_id: str, 
        location: GeoLocation
    ) -> Optional[Tuple[str, float]]:
        """查找最近的数据副本"""
        replicas = self.find_data_location(data_id)
        if not replicas:
            return None
        
        nearest = None
        min_distance = float('inf')
        
        for node_id in replicas:
            node = self.node_manager.get_node(node_id)
            if node:
                distance = node.location.distance_to(location)
                if distance < min_distance:
                    min_distance = distance
                    nearest = node_id
        
        return (nearest, min_distance) if nearest else None
    
    async def place_data(
        self,
        data_id: str,
        data_size_mb: float,
        source_location: GeoLocation,
        policy_id: Optional[str] = None
    ) -> List[str]:
        """
        放置数据到边缘节点
        
        Returns: 存储数据的节点ID列表
        """
        # 查找最近的节点
        nearest_nodes = self.node_manager.find_nearest_nodes(
            source_location,
            max_distance_km=200.0
        )
        
        if not nearest_nodes:
            logger.warning(f"No edge nodes found near data source")
            return []
        
        # 选择主节点和副本节点
        primary_node = nearest_nodes[0][0]
        replica_count = min(2, len(nearest_nodes) - 1)
        replica_nodes = [n[0] for n in nearest_nodes[1:replica_count+1]]
        
        # 注册数据
        self.data_registry[data_id] = {
            "primary": primary_node.node_id,
            "replicas": [n.node_id for n in replica_nodes] + [primary_node.node_id],
            "size_mb": data_size_mb,
            "created_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Data {data_id} placed on {primary_node.node_id} with {len(replica_nodes)} replicas")
        
        return self.data_registry[data_id]["replicas"]
    
    def should_process_locally(
        self,
        data_id: str,
        node_id: str,
        policy: DataLocalityPolicy = DataLocalityPolicy.PREFERRED
    ) -> bool:
        """判断数据是否应该在本地处理"""
        if policy == DataLocalityPolicy.STRICT:
            # 严格模式：数据必须在该节点
            replicas = self.find_data_location(data_id) or []
            return node_id in replicas
        
        elif policy == DataLocalityPolicy.PREFERRED:
            # 优先模式：如果在该节点或相邻节点则本地处理
            replicas = self.find_data_location(data_id) or []
            if node_id in replicas:
                return True
            
            node = self.node_manager.get_node(node_id)
            if node:
                for replica_node_id in replicas:
                    if replica_node_id in node.peers:
                        return True
            return False
        
        elif policy == DataLocalityPolicy.BALANCED:
            # 平衡模式：考虑网络成本
            return True  # 简化实现
        
        else:  # FLEXIBLE
            return False  # 可以传输到云端


class EdgeInferenceEngine:
    """边缘推理引擎"""
    
    def __init__(
        self, 
        node_manager: EdgeNodeManager,
        scheduler: LatencyAwareScheduler,
        data_manager: DataLocalityManager
    ):
        self.node_manager = node_manager
        self.scheduler = scheduler
        self.data_manager = data_manager
        self.models: Dict[str, ModelInfo] = {}
        self._inference_stats: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0,
            "total_latency": 0.0,
            "errors": 0
        })
    
    def register_model(self, model: ModelInfo):
        """注册模型"""
        self.models[model.model_id] = model
        logger.info(f"Registered model: {model.name} (v{model.version})")
    
    async def inference(self, request: InferenceRequest) -> InferenceResult:
        """
        执行推理
        
        完整的推理流程：
        1. 调度到合适的边缘节点
        2. 检查数据位置
        3. 执行推理
        4. 返回结果
        """
        start_time = time.time()
        
        # 1. 调度
        schedule_result = await self.scheduler.schedule_inference(request, self.models)
        
        if not schedule_result:
            # 无法调度到边缘，考虑云端回退
            logger.warning(f"Falling back to cloud for request {request.request_id}")
            return await self._cloud_fallback(request)
        
        node, model = schedule_result
        
        # 2. 检查数据本地性
        # 实际实现会检查数据是否已在节点上，如不在可能需要预取
        
        # 3. 执行推理 (模拟)
        try:
            result = await self._execute_inference(node, model, request)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # 更新统计
            self._inference_stats[model.model_id]["count"] += 1
            self._inference_stats[model.model_id]["total_latency"] += latency_ms
            
            return InferenceResult(
                request_id=request.request_id,
                node_id=node.node_id,
                model_id=model.model_id,
                output_data=result,
                confidence=random.uniform(0.85, 0.99),
                latency_ms=latency_ms,
                metadata={
                    "scheduled_node": node.name,
                    "model_version": model.version,
                    "inference_engine": model.engine.value
                }
            )
            
        except Exception as e:
            self._inference_stats[model.model_id]["errors"] += 1
            logger.error(f"Inference error on node {node.node_id}: {e}")
            raise
    
    async def _execute_inference(
        self, 
        node: EdgeNode, 
        model: ModelInfo, 
        request: InferenceRequest
    ) -> Any:
        """在指定节点执行推理 (模拟)"""
        # 模拟推理延迟
        base_latency = model.latency_ms_p50
        load_factor = 1 + node.current_load * 0.5  # 负载越高延迟越高
        actual_latency = base_latency * load_factor * random.uniform(0.9, 1.1)
        
        await asyncio.sleep(actual_latency / 1000)
        
        # 返回模拟输出
        return {
            "predictions": [random.random() for _ in range(10)],
            "inference_time_ms": actual_latency
        }
    
    async def _cloud_fallback(self, request: InferenceRequest) -> InferenceResult:
        """云端回退处理"""
        start_time = time.time()
        
        # 模拟云端推理 (延迟更高)
        await asyncio.sleep(0.5)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return InferenceResult(
            request_id=request.request_id,
            node_id="cloud",
            model_id=request.model_id,
            output_data={"predictions": [random.random() for _ in range(10)]},
            confidence=random.uniform(0.85, 0.99),
            latency_ms=latency_ms,
            metadata={"fallback": "cloud", "reason": "no_edge_node_available"}
        )
    
    def get_inference_stats(self) -> Dict:
        """获取推理统计信息"""
        stats = {}
        for model_id, data in self._inference_stats.items():
            count = data["count"]
            stats[model_id] = {
                "total_inferences": count,
                "avg_latency_ms": round(data["total_latency"] / max(count, 1), 2),
                "error_count": data["errors"],
                "error_rate": round(data["errors"] / max(count, 1), 4)
            }
        return stats
    
    async def batch_inference(
        self,
        requests: List[InferenceRequest],
        max_concurrent: int = 10
    ) -> List[InferenceResult]:
        """批量推理"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def infer_with_limit(req: InferenceRequest) -> InferenceResult:
            async with semaphore:
                return await self.inference(req)
        
        results = await asyncio.gather(
            *[infer_with_limit(req) for req in requests],
            return_exceptions=True
        )
        
        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch inference error for request {requests[i].request_id}: {result}")
                # 创建错误结果
                processed_results.append(InferenceResult(
                    request_id=requests[i].request_id,
                    node_id="error",
                    model_id=requests[i].model_id,
                    output_data=None,
                    confidence=0.0,
                    latency_ms=0.0,
                    metadata={"error": str(result)}
                ))
            else:
                processed_results.append(result)
        
        return processed_results


class EdgeCloudSync:
    """边缘-云同步管理器"""
    
    def __init__(
        self, 
        node_manager: EdgeNodeManager,
        data_manager: DataLocalityManager
    ):
        self.node_manager = node_manager
        self.data_manager = data_manager
        self.sync_policies: Dict[str, Dict] = {}
        self.pending_syncs: asyncio.Queue = asyncio.Queue()
    
    async def sync_model_to_edge(
        self,
        model: ModelInfo,
        target_regions: List[str],
        priority: int = 5
    ) -> List[str]:
        """将模型同步到边缘节点"""
        deployed_nodes = []
        
        for region in target_regions:
            nodes = self.node_manager.find_nodes_by_region(region)
            
            for node in nodes:
                if not node.is_healthy():
                    continue
                
                # 检查存储空间
                if model.model_size_mb > node.hardware.storage_gb * 1024 * 0.2:
                    continue
                
                # 添加到同步队列
                await self.pending_syncs.put({
                    "type": "model",
                    "model_id": model.model_id,
                    "node_id": node.node_id,
                    "priority": priority
                })
                
                deployed_nodes.append(node.node_id)
        
        return deployed_nodes
    
    async def sync_data_to_cloud(
        self,
        data_id: str,
        source_node_id: str,
        aggregation_level: str = "raw"  # raw, aggregated, summary
    ) -> bool:
        """将边缘数据同步到云端"""
        node = self.node_manager.get_node(source_node_id)
        if not node:
            return False
        
        # 根据聚合级别决定传输策略
        if aggregation_level == "raw":
            # 完整数据传输
            bandwidth_limit = node.hardware.network_bandwidth_mbps * 0.3
        elif aggregation_level == "aggregated":
            # 聚合数据 (约10%大小)
            bandwidth_limit = node.hardware.network_bandwidth_mbps * 0.1
        else:  # summary
            # 摘要数据 (约1%大小)
            bandwidth_limit = node.hardware.network_bandwidth_mbps * 0.05
        
        logger.info(
            f"Syncing data {data_id} from {source_node_id} to cloud "
            f"(level={aggregation_level}, bw_limit={bandwidth_limit:.2f} Mbps)"
        )
        
        # 模拟同步
        await asyncio.sleep(1)
        return True
    
    async def configure_sync_policy(
        self,
        node_id: str,
        sync_interval_seconds: int = 300,
        max_batch_size_mb: float = 100.0,
        compression: str = "lz4",
        encryption: bool = True
    ):
        """配置节点的同步策略"""
        self.sync_policies[node_id] = {
            "sync_interval": sync_interval_seconds,
            "max_batch_size_mb": max_batch_size_mb,
            "compression": compression,
            "encryption": encryption,
            "last_sync": None
        }


# 边缘部署协调器
class EdgeDeploymentOrchestrator:
    """边缘部署编排器"""
    
    def __init__(self):
        self.node_manager = EdgeNodeManager()
        self.scheduler = LatencyAwareScheduler(self.node_manager)
        self.data_manager = DataLocalityManager(self.node_manager)
        self.inference_engine = EdgeInferenceEngine(
            self.node_manager, self.scheduler, self.data_manager
        )
        self.sync_manager = EdgeCloudSync(self.node_manager, self.data_manager)
        self._running = False
    
    async def start(self):
        """启动边缘部署编排器"""
        self._running = True
        await self.node_manager.start()
        await self.data_manager.start()
        logger.info("Edge deployment orchestrator started")
    
    async def stop(self):
        """停止边缘部署编排器"""
        self._running = False
        await self.data_manager.stop()
        await self.node_manager.stop()
        logger.info("Edge deployment orchestrator stopped")
    
    async def register_edge_node(self, node: EdgeNode) -> bool:
        """注册边缘节点"""
        return await self.node_manager.register_node(node)
    
    async def deploy_model_to_edge(
        self,
        model: ModelInfo,
        target_node_ids: Optional[List[str]] = None,
        target_regions: Optional[List[str]] = None
    ) -> List[str]:
        """部署模型到边缘"""
        # 注册模型
        self.inference_engine.register_model(model)
        
        # 确定目标节点
        if target_node_ids:
            node_ids = target_node_ids
        elif target_regions:
            node_ids = []
            for region in target_regions:
                nodes = self.node_manager.find_nodes_by_region(region)
                node_ids.extend([n.node_id for n in nodes])
        else:
            # 自动选择
            node_ids = None
        
        # 部署
        deployed = await self.scheduler.deploy_model(model, node_ids)
        
        logger.info(f"Model {model.model_id} deployed to {len(deployed)} edge nodes")
        return deployed
    
    async def run_inference(self, request: InferenceRequest) -> InferenceResult:
        """运行边缘推理"""
        return await self.inference_engine.inference(request)
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            "nodes": self.node_manager.get_cluster_stats(),
            "inference": self.inference_engine.get_inference_stats(),
            "models_deployed": list(self.scheduler.model_deployments.keys()),
            "data_policies": len(self.data_manager.policies)
        }


# 示例使用
async def demo():
    """边缘部署演示"""
    
    # 创建编排器
    orchestrator = EdgeDeploymentOrchestrator()
    await orchestrator.start()
    
    try:
        # 注册边缘节点
        nodes_data = [
            ("edge-tokyo-01", 35.6895, 139.6917, "Tokyo"),
            ("edge-singapore-01", 1.3521, 103.8198, "Singapore"),
            ("edge-frankfurt-01", 50.1109, 8.6821, "Frankfurt"),
            ("edge-oregon-01", 45.5152, -122.6784, "Oregon"),
            ("edge-mumbai-01", 19.0760, 72.8777, "Mumbai"),
        ]
        
        for name, lat, lon, region in nodes_data:
            node = EdgeNode(
                node_id="",
                name=name,
                location=GeoLocation(lat=lat, lon=lon),
                hardware=HardwareCapabilities(
                    cpu_cores=8,
                    cpu_frequency_ghz=2.5,
                    memory_gb=32,
                    storage_gb=500,
                    has_gpu=True,
                    gpu_model="NVIDIA T4",
                    gpu_memory_gb=16,
                    network_bandwidth_mbps=1000,
                    network_latency_ms=5
                ),
                tags={"region": region, "tier": "1"}
            )
            await orchestrator.register_edge_node(node)
        
        # 创建模型
        model = ModelInfo(
            model_id="molecule-predictor-v1",
            name="Molecule Property Predictor",
            version="1.0.0",
            framework="pytorch",
            engine=InferenceEngine.TENSORRT,
            input_shape=[1, 128, 128, 3],
            output_shape=[1, 100],
            model_size_mb=250,
            latency_ms_p50=15,
            latency_ms_p99=30,
            throughput_qps=50,
            accuracy=0.94,
            quantization="fp16"
        )
        
        # 部署模型到边缘
        print("=== 部署模型到边缘 ===")
        deployed = await orchestrator.deploy_model_to_edge(
            model,
            target_regions=["Tokyo", "Singapore", "Frankfurt", "Oregon"]
        )
        print(f"  Deployed to {len(deployed)} nodes: {deployed}")
        
        # 模拟推理请求
        print("\n=== 边缘推理测试 ===")
        
        test_locations = [
            ("User A (Tokyo)", 35.6762, 139.6503),
            ("User B (Singapore)", 1.2966, 103.7764),
            ("User C (London)", 51.5074, -0.1278),
        ]
        
        for user_name, lat, lon in test_locations:
            request = InferenceRequest(
                request_id="",
                model_id=model.model_id,
                input_data={"molecule_smiles": "CCO"},
                input_location=GeoLocation(lat=lat, lon=lon),
                max_latency_ms=50
            )
            
            result = await orchestrator.run_inference(request)
            print(f"  {user_name}: latency={result.latency_ms:.1f}ms, "
                  f"node={result.node_id[:8]}..., confidence={result.confidence:.2%}")
        
        # 批量推理测试
        print("\n=== 批量推理测试 ===")
        batch_requests = [
            InferenceRequest(
                request_id="",
                model_id=model.model_id,
                input_data={"id": i},
                input_location=GeoLocation(lat=35.0 + i*0.1, lon=139.0 + i*0.1),
                max_latency_ms=100
            )
            for i in range(10)
        ]
        
        batch_results = await orchestrator.inference_engine.batch_inference(
            batch_requests, max_concurrent=5
        )
        
        latencies = [r.latency_ms for r in batch_results if r.node_id != "error"]
        if latencies:
            print(f"  Batch of 10: avg={sum(latencies)/len(latencies):.1f}ms, "
                  f"min={min(latencies):.1f}ms, max={max(latencies):.1f}ms")
        
        # 系统状态
        print("\n=== 系统状态 ===")
        status = orchestrator.get_system_status()
        print(f"  Nodes: {status['nodes']['total_nodes']} total, "
              f"{status['nodes']['healthy_nodes']} healthy")
        print(f"  Average load: {status['nodes']['average_load']:.2%}")
        print(f"  Models deployed: {status['models_deployed']}")
        
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(demo())
