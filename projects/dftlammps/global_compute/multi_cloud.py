#!/usr/bin/env python3
"""
Multi-Cloud Orchestration Module
多云编排模块 - 提供AWS/Azure/GCP的统一接口

Features:
- Unified cloud provider interface
- Cross-region scheduling
- Cost optimization
- Auto-failover between clouds
- Resource pooling and allocation

Author: DFTLammps Platform
Version: 2.0.0
"""

import asyncio
import json
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from enum import Enum, auto
from collections import defaultdict
import logging
from datetime import datetime, timedelta
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """支持的云服务提供商"""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ALIBABA = "alibaba"
    TENCENT = "tencent"


class InstanceType(Enum):
    """实例类型"""
    CPU = auto()
    GPU = auto()
    MEMORY = auto()
    STORAGE = auto()
    FPGA = auto()


class RegionTier(Enum):
    """区域层级"""
    TIER_1 = 1  # 主要区域
    TIER_2 = 2  # 次要区域
    TIER_3 = 3  # 边缘区域


@dataclass
class CloudRegion:
    """云区域定义"""
    provider: CloudProvider
    region_code: str
    region_name: str
    tier: RegionTier
    lat: float
    lon: float
    available_instance_types: List[InstanceType] = field(default_factory=list)
    carbon_intensity: float = 0.0  # gCO2/kWh
    electricity_cost: float = 0.0  # $/kWh
    network_bandwidth: float = 10.0  # Gbps
    
    def distance_to(self, lat: float, lon: float) -> float:
        """计算与指定坐标的距离 (km)"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # 地球半径 (km)
        lat1, lon1 = radians(self.lat), radians(self.lon)
        lat2, lon2 = radians(lat), radians(lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c


@dataclass
class InstanceSpec:
    """实例规格"""
    instance_type: InstanceType
    vcpus: int
    memory_gb: float
    gpu_count: int = 0
    gpu_type: Optional[str] = None
    storage_gb: float = 100.0
    storage_type: str = "ssd"
    spot: bool = False  # 是否使用 spot/preemptible 实例
    
    def to_dict(self) -> Dict:
        return {
            "instance_type": self.instance_type.name,
            "vcpus": self.vcpus,
            "memory_gb": self.memory_gb,
            "gpu_count": self.gpu_count,
            "gpu_type": self.gpu_type,
            "storage_gb": self.storage_gb,
            "storage_type": self.storage_type,
            "spot": self.spot
        }


@dataclass
class PricingInfo:
    """定价信息"""
    provider: CloudProvider
    region: str
    instance_spec: InstanceSpec
    on_demand_price: float  # $/hour
    spot_price: Optional[float] = None  # $/hour
    reserved_1y_price: Optional[float] = None
    reserved_3y_price: Optional[float] = None
    currency: str = "USD"
    
    def get_effective_price(self, use_spot: bool = False) -> float:
        if use_spot and self.spot_price:
            return self.spot_price
        return self.on_demand_price


@dataclass
class ComputeTask:
    """计算任务定义"""
    task_id: str
    instance_spec: InstanceSpec
    region_preferences: List[str] = field(default_factory=list)
    priority: int = 5  # 1-10, 10为最高
    estimated_duration: float = 3600  # 秒
    data_locality_requirements: Dict[str, str] = field(default_factory=dict)
    compliance_requirements: List[str] = field(default_factory=list)
    max_cost: Optional[float] = None
    deadline: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = hashlib.md5(
                f"{time.time()}-{random.randint(0, 1000000)}".encode()
            ).hexdigest()[:16]


@dataclass
class ProvisionedInstance:
    """已调配的实例"""
    instance_id: str
    provider: CloudProvider
    region: str
    instance_spec: InstanceSpec
    public_ip: Optional[str] = None
    private_ip: Optional[str] = None
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)
    cost_per_hour: float = 0.0
    ssh_key: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    async def terminate(self):
        """终止实例"""
        self.status = "terminating"
        logger.info(f"Terminating instance {self.instance_id} on {self.provider.value}")
        # 实际实现会调用云提供商API
        await asyncio.sleep(0.5)
        self.status = "terminated"


class CloudProviderInterface(ABC):
    """云提供商抽象接口"""
    
    def __init__(self, provider: CloudProvider, credentials: Dict):
        self.provider = provider
        self.credentials = credentials
        self.regions: List[CloudRegion] = []
        self._initialized = False
    
    @abstractmethod
    async def initialize(self):
        """初始化连接"""
        pass
    
    @abstractmethod
    async def list_regions(self) -> List[CloudRegion]:
        """列出可用区域"""
        pass
    
    @abstractmethod
    async def get_pricing(self, instance_spec: InstanceSpec, region: str) -> PricingInfo:
        """获取定价信息"""
        pass
    
    @abstractmethod
    async def provision_instance(
        self, 
        instance_spec: InstanceSpec, 
        region: str,
        **kwargs
    ) -> ProvisionedInstance:
        """创建实例"""
        pass
    
    @abstractmethod
    async def terminate_instance(self, instance_id: str, region: str) -> bool:
        """终止实例"""
        pass
    
    @abstractmethod
    async def list_instances(self, region: Optional[str] = None) -> List[ProvisionedInstance]:
        """列出实例"""
        pass
    
    @abstractmethod
    async def get_instance_metrics(self, instance_id: str, region: str) -> Dict:
        """获取实例指标"""
        pass


class AWSProvider(CloudProviderInterface):
    """AWS 云提供商实现"""
    
    def __init__(self, credentials: Dict):
        super().__init__(CloudProvider.AWS, credentials)
        self.ec2_client = None
        self.pricing_client = None
    
    async def initialize(self):
        """初始化 AWS 连接"""
        try:
            import boto3
            session = boto3.Session(
                aws_access_key_id=self.credentials.get("access_key_id"),
                aws_secret_access_key=self.credentials.get("secret_access_key"),
                region_name=self.credentials.get("default_region", "us-east-1")
            )
            self.ec2_client = session.client("ec2")
            self.pricing_client = session.client("pricing", region_name="us-east-1")
            self._initialized = True
            logger.info("AWS provider initialized successfully")
        except ImportError:
            logger.warning("boto3 not installed, using mock AWS provider")
            self._initialized = True
    
    async def list_regions(self) -> List[CloudRegion]:
        """列出 AWS 区域"""
        aws_regions = [
            CloudRegion(CloudProvider.AWS, "us-east-1", "N. Virginia", RegionTier.TIER_1, 39.0, -77.5),
            CloudRegion(CloudProvider.AWS, "us-west-2", "Oregon", RegionTier.TIER_1, 45.5, -122.0),
            CloudRegion(CloudProvider.AWS, "eu-west-1", "Ireland", RegionTier.TIER_1, 53.3, -6.3),
            CloudRegion(CloudProvider.AWS, "ap-northeast-1", "Tokyo", RegionTier.TIER_1, 35.7, 139.7),
            CloudRegion(CloudProvider.AWS, "ap-southeast-1", "Singapore", RegionTier.TIER_2, 1.3, 103.8),
            CloudRegion(CloudProvider.AWS, "eu-central-1", "Frankfurt", RegionTier.TIER_1, 50.1, 8.7),
            CloudRegion(CloudProvider.AWS, "ap-south-1", "Mumbai", RegionTier.TIER_2, 19.1, 72.9),
            CloudRegion(CloudProvider.AWS, "sa-east-1", "São Paulo", RegionTier.TIER_3, -23.5, -46.6),
        ]
        self.regions = aws_regions
        return aws_regions
    
    async def get_pricing(self, instance_spec: InstanceSpec, region: str) -> PricingInfo:
        """获取 AWS 定价 (模拟)"""
        # 基于实例类型的基础定价
        base_price = self._calculate_base_price(instance_spec)
        
        # 区域价格调整
        region_multiplier = {
            "us-east-1": 1.0,
            "us-west-2": 1.05,
            "eu-west-1": 1.15,
            "eu-central-1": 1.18,
            "ap-northeast-1": 1.20,
            "ap-southeast-1": 1.12,
            "ap-south-1": 1.08,
            "sa-east-1": 1.35,
        }.get(region, 1.0)
        
        on_demand = base_price * region_multiplier
        spot = on_demand * 0.3 if instance_spec.spot else None
        
        return PricingInfo(
            provider=CloudProvider.AWS,
            region=region,
            instance_spec=instance_spec,
            on_demand_price=round(on_demand, 4),
            spot_price=round(spot, 4) if spot else None
        )
    
    def _calculate_base_price(self, spec: InstanceSpec) -> float:
        """计算基础价格"""
        price = spec.vcpus * 0.04 + spec.memory_gb * 0.005
        if spec.gpu_count > 0:
            gpu_price = {"v100": 2.5, "a100": 3.5, "h100": 5.0, "t4": 0.35}
            price += spec.gpu_count * gpu_price.get(spec.gpu_type or "t4", 1.0)
        return price
    
    async def provision_instance(
        self, 
        instance_spec: InstanceSpec, 
        region: str,
        **kwargs
    ) -> ProvisionedInstance:
        """创建 EC2 实例"""
        instance_id = f"i-{hashlib.md5(str(time.time()).encode()).hexdigest()[:17]}"
        
        # 获取定价
        pricing = await self.get_pricing(instance_spec, region)
        
        instance = ProvisionedInstance(
            instance_id=instance_id,
            provider=CloudProvider.AWS,
            region=region,
            instance_spec=instance_spec,
            public_ip=f"54.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
            private_ip=f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
            status="running",
            cost_per_hour=pricing.get_effective_price(instance_spec.spot),
            ssh_key=kwargs.get("ssh_key_name"),
            metadata={
                "launch_time": datetime.utcnow().isoformat(),
                "instance_type": self._get_instance_type_name(instance_spec),
                "spot": instance_spec.spot
            }
        )
        
        logger.info(f"Provisioned AWS instance {instance_id} in {region}")
        return instance
    
    def _get_instance_type_name(self, spec: InstanceSpec) -> str:
        """获取 AWS 实例类型名称"""
        if spec.gpu_count > 0:
            if spec.gpu_type == "v100":
                return f"p3.{spec.gpu_count}xlarge"
            elif spec.gpu_type == "a100":
                return f"p4d.{spec.gpu_count}xlarge"
            return "g4dn.xlarge"
        elif spec.vcpus <= 2:
            return "t3.medium"
        elif spec.vcpus <= 4:
            return "t3.large"
        elif spec.vcpus <= 8:
            return "c5.2xlarge"
        else:
            return f"c5.{spec.vcpus//4}xlarge"
    
    async def terminate_instance(self, instance_id: str, region: str) -> bool:
        """终止 EC2 实例"""
        logger.info(f"Terminating AWS instance {instance_id} in {region}")
        return True
    
    async def list_instances(self, region: Optional[str] = None) -> List[ProvisionedInstance]:
        """列出 EC2 实例"""
        return []
    
    async def get_instance_metrics(self, instance_id: str, region: str) -> Dict:
        """获取 CloudWatch 指标"""
        return {
            "cpu_utilization": random.uniform(10, 90),
            "memory_utilization": random.uniform(20, 80),
            "network_in": random.uniform(1000, 100000),
            "network_out": random.uniform(1000, 50000)
        }


class AzureProvider(CloudProviderInterface):
    """Azure 云提供商实现"""
    
    def __init__(self, credentials: Dict):
        super().__init__(CloudProvider.AZURE, credentials)
    
    async def initialize(self):
        """初始化 Azure 连接"""
        try:
            from azure.identity import ClientSecretCredential
            from azure.mgmt.compute import ComputeManagementClient
            
            credential = ClientSecretCredential(
                tenant_id=self.credentials.get("tenant_id"),
                client_id=self.credentials.get("client_id"),
                client_secret=self.credentials.get("client_secret")
            )
            self.compute_client = ComputeManagementClient(
                credential, 
                self.credentials.get("subscription_id")
            )
            self._initialized = True
            logger.info("Azure provider initialized successfully")
        except ImportError:
            logger.warning("Azure SDK not installed, using mock Azure provider")
            self._initialized = True
    
    async def list_regions(self) -> List[CloudRegion]:
        """列出 Azure 区域"""
        azure_regions = [
            CloudRegion(CloudProvider.AZURE, "eastus", "East US", RegionTier.TIER_1, 37.0, -79.0),
            CloudRegion(CloudProvider.AZURE, "westeurope", "West Europe", RegionTier.TIER_1, 52.4, 4.9),
            CloudRegion(CloudProvider.AZURE, "southeastasia", "Southeast Asia", RegionTier.TIER_1, 1.3, 103.8),
            CloudRegion(CloudProvider.AZURE, "japaneast", "Japan East", RegionTier.TIER_1, 35.7, 139.7),
            CloudRegion(CloudProvider.AZURE, "westus2", "West US 2", RegionTier.TIER_1, 47.2, -119.5),
            CloudRegion(CloudProvider.AZURE, "northeurope", "North Europe", RegionTier.TIER_1, 53.3, -6.3),
            CloudRegion(CloudProvider.AZURE, "centralindia", "Central India", RegionTier.TIER_2, 18.5, 73.9),
            CloudRegion(CloudProvider.AZURE, "brazilsouth", "Brazil South", RegionTier.TIER_3, -23.5, -46.6),
        ]
        self.regions = azure_regions
        return azure_regions
    
    async def get_pricing(self, instance_spec: InstanceSpec, region: str) -> PricingInfo:
        """获取 Azure 定价 (模拟)"""
        base_price = self._calculate_base_price(instance_spec)
        
        region_multiplier = {
            "eastus": 1.0,
            "westus2": 1.02,
            "westeurope": 1.18,
            "northeurope": 1.15,
            "southeastasia": 1.15,
            "japaneast": 1.22,
            "centralindia": 1.10,
            "brazilsouth": 1.40,
        }.get(region, 1.0)
        
        on_demand = base_price * region_multiplier
        spot = on_demand * 0.25 if instance_spec.spot else None
        
        return PricingInfo(
            provider=CloudProvider.AZURE,
            region=region,
            instance_spec=instance_spec,
            on_demand_price=round(on_demand, 4),
            spot_price=round(spot, 4) if spot else None
        )
    
    def _calculate_base_price(self, spec: InstanceSpec) -> float:
        """计算基础价格"""
        price = spec.vcpus * 0.042 + spec.memory_gb * 0.006
        if spec.gpu_count > 0:
            gpu_price = {"v100": 2.8, "a100": 3.8, "h100": 5.5, "t4": 0.40}
            price += spec.gpu_count * gpu_price.get(spec.gpu_type or "t4", 1.1)
        return price
    
    async def provision_instance(
        self, 
        instance_spec: InstanceSpec, 
        region: str,
        **kwargs
    ) -> ProvisionedInstance:
        """创建 Azure VM"""
        instance_id = f"vm-{hashlib.md5(str(time.time()).encode()).hexdigest()[:12]}"
        
        pricing = await self.get_pricing(instance_spec, region)
        
        instance = ProvisionedInstance(
            instance_id=instance_id,
            provider=CloudProvider.AZURE,
            region=region,
            instance_spec=instance_spec,
            public_ip=f"20.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
            private_ip=f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
            status="running",
            cost_per_hour=pricing.get_effective_price(instance_spec.spot),
            metadata={
                "launch_time": datetime.utcnow().isoformat(),
                "vm_size": self._get_vm_size(instance_spec),
                "priority": "Spot" if instance_spec.spot else "Regular"
            }
        )
        
        logger.info(f"Provisioned Azure VM {instance_id} in {region}")
        return instance
    
    def _get_vm_size(self, spec: InstanceSpec) -> str:
        """获取 Azure VM 大小"""
        if spec.gpu_count > 0:
            if spec.gpu_type == "v100":
                return "Standard_NC6s_v3"
            elif spec.gpu_type == "a100":
                return "Standard_ND96asr_v4"
            return "Standard_NC4as_T4_v3"
        elif spec.vcpus <= 2:
            return "Standard_D2s_v5"
        elif spec.vcpus <= 4:
            return "Standard_D4s_v5"
        else:
            return f"Standard_D{spec.vcpus}s_v5"
    
    async def terminate_instance(self, instance_id: str, region: str) -> bool:
        """终止 Azure VM"""
        logger.info(f"Terminating Azure VM {instance_id} in {region}")
        return True
    
    async def list_instances(self, region: Optional[str] = None) -> List[ProvisionedInstance]:
        return []
    
    async def get_instance_metrics(self, instance_id: str, region: str) -> Dict:
        return {
            "cpu_utilization": random.uniform(10, 90),
            "memory_utilization": random.uniform(20, 80),
            "disk_read": random.uniform(1000, 50000),
            "disk_write": random.uniform(1000, 30000)
        }


class GCPProvider(CloudProviderInterface):
    """Google Cloud Platform 提供商实现"""
    
    def __init__(self, credentials: Dict):
        super().__init__(CloudProvider.GCP, credentials)
    
    async def initialize(self):
        """初始化 GCP 连接"""
        try:
            from google.cloud import compute_v1
            from google.oauth2 import service_account
            
            credentials = service_account.Credentials.from_service_account_info(
                self.credentials.get("service_account_key")
            )
            self.instances_client = compute_v1.InstancesClient(credentials=credentials)
            self._initialized = True
            logger.info("GCP provider initialized successfully")
        except ImportError:
            logger.warning("Google Cloud SDK not installed, using mock GCP provider")
            self._initialized = True
    
    async def list_regions(self) -> List[CloudRegion]:
        """列出 GCP 区域"""
        gcp_regions = [
            CloudRegion(CloudProvider.GCP, "us-central1", "Iowa", RegionTier.TIER_1, 41.3, -93.6),
            CloudRegion(CloudProvider.GCP, "us-west1", "Oregon", RegionTier.TIER_1, 45.5, -122.0),
            CloudRegion(CloudProvider.GCP, "europe-west1", "Belgium", RegionTier.TIER_1, 50.4, 3.8),
            CloudRegion(CloudProvider.GCP, "asia-northeast1", "Tokyo", RegionTier.TIER_1, 35.7, 139.7),
            CloudRegion(CloudProvider.GCP, "asia-southeast1", "Singapore", RegionTier.TIER_1, 1.3, 103.8),
            CloudRegion(CloudProvider.GCP, "europe-west4", "Netherlands", RegionTier.TIER_1, 53.4, 6.8),
            CloudRegion(CloudProvider.GCP, "asia-south1", "Mumbai", RegionTier.TIER_2, 19.1, 72.9),
            CloudRegion(CloudProvider.GCP, "southamerica-east1", "São Paulo", RegionTier.TIER_3, -23.5, -46.6),
        ]
        self.regions = gcp_regions
        return gcp_regions
    
    async def get_pricing(self, instance_spec: InstanceSpec, region: str) -> PricingInfo:
        """获取 GCP 定价 (模拟)"""
        base_price = self._calculate_base_price(instance_spec)
        
        region_multiplier = {
            "us-central1": 1.0,
            "us-west1": 1.02,
            "europe-west1": 1.12,
            "europe-west4": 1.15,
            "asia-northeast1": 1.20,
            "asia-southeast1": 1.12,
            "asia-south1": 1.10,
            "southamerica-east1": 1.38,
        }.get(region, 1.0)
        
        on_demand = base_price * region_multiplier
        spot = on_demand * 0.20 if instance_spec.spot else None  # GCP preemptible
        
        return PricingInfo(
            provider=CloudProvider.GCP,
            region=region,
            instance_spec=instance_spec,
            on_demand_price=round(on_demand, 4),
            spot_price=round(spot, 4) if spot else None
        )
    
    def _calculate_base_price(self, spec: InstanceSpec) -> float:
        """计算基础价格"""
        price = spec.vcpus * 0.038 + spec.memory_gb * 0.005
        if spec.gpu_count > 0:
            gpu_price = {"v100": 2.6, "a100": 3.6, "h100": 5.2, "t4": 0.32}
            price += spec.gpu_count * gpu_price.get(spec.gpu_type or "t4", 1.0)
        return price
    
    async def provision_instance(
        self, 
        instance_spec: InstanceSpec, 
        region: str,
        **kwargs
    ) -> ProvisionedInstance:
        """创建 GCP Compute Engine 实例"""
        zone = f"{region}-a"
        instance_id = f"gcp-{hashlib.md5(str(time.time()).encode()).hexdigest()[:12]}"
        
        pricing = await self.get_pricing(instance_spec, region)
        
        instance = ProvisionedInstance(
            instance_id=instance_id,
            provider=CloudProvider.GCP,
            region=region,
            instance_spec=instance_spec,
            public_ip=f"34.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
            private_ip=f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
            status="running",
            cost_per_hour=pricing.get_effective_price(instance_spec.spot),
            metadata={
                "launch_time": datetime.utcnow().isoformat(),
                "machine_type": self._get_machine_type(instance_spec),
                "preemptible": instance_spec.spot,
                "zone": zone
            }
        )
        
        logger.info(f"Provisioned GCP instance {instance_id} in {region}")
        return instance
    
    def _get_machine_type(self, spec: InstanceSpec) -> str:
        """获取 GCP 机器类型"""
        if spec.gpu_count > 0:
            if spec.gpu_type == "a100":
                return f"a2-highgpu-{spec.gpu_count}g"
            elif spec.gpu_type == "v100":
                return f"n1-standard-{spec.vcpus}"
            return f"n1-standard-{spec.vcpus}"
        else:
            return f"c2-standard-{spec.vcpus}"
    
    async def terminate_instance(self, instance_id: str, region: str) -> bool:
        """终止 GCP 实例"""
        logger.info(f"Terminating GCP instance {instance_id} in {region}")
        return True
    
    async def list_instances(self, region: Optional[str] = None) -> List[ProvisionedInstance]:
        return []
    
    async def get_instance_metrics(self, instance_id: str, region: str) -> Dict:
        return {
            "cpu_utilization": random.uniform(10, 90),
            "memory_utilization": random.uniform(20, 80),
            "sent_bytes_count": random.uniform(1000000, 100000000),
            "received_bytes_count": random.uniform(1000000, 50000000)
        }


class CostOptimizer:
    """成本优化器 - 提供智能成本优化策略"""
    
    def __init__(self):
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.spot_interruption_rates: Dict[str, float] = {}
    
    async def optimize_task_placement(
        self,
        task: ComputeTask,
        providers: List[CloudProviderInterface],
        optimization_strategy: str = "cost"  # cost, performance, balanced, carbon
    ) -> List[Tuple[CloudProvider, str, float, PricingInfo]]:
        """
        优化任务放置
        
        Returns: 按优先级排序的 (provider, region, score, pricing) 列表
        """
        options = []
        
        for provider in providers:
            if not provider._initialized:
                await provider.initialize()
            
            regions = await provider.list_regions()
            
            for region in regions:
                # 检查区域偏好
                if task.region_preferences and region.region_code not in task.region_preferences:
                    continue
                
                # 检查合规要求
                if task.compliance_requirements:
                    if not self._check_compliance(region, task.compliance_requirements):
                        continue
                
                try:
                    pricing = await provider.get_pricing(task.instance_spec, region.region_code)
                    
                    # 计算分数
                    score = self._calculate_placement_score(
                        task, region, pricing, optimization_strategy
                    )
                    
                    options.append((provider.provider, region.region_code, score, pricing))
                    
                except Exception as e:
                    logger.warning(f"Failed to get pricing for {provider.provider.value}/{region.region_code}: {e}")
        
        # 按分数排序
        options.sort(key=lambda x: x[2], reverse=True)
        return options
    
    def _calculate_placement_score(
        self,
        task: ComputeTask,
        region: CloudRegion,
        pricing: PricingInfo,
        strategy: str
    ) -> float:
        """计算放置分数 (越高越好)"""
        
        if strategy == "cost":
            # 成本优化：价格越低越好
            price = pricing.get_effective_price(task.instance_spec.spot)
            max_price = task.max_cost or (price * 10)
            if price > max_price:
                return 0
            return (max_price - price) / max_price * 100
        
        elif strategy == "performance":
            # 性能优化：网络带宽越高越好
            return region.network_bandwidth * 10
        
        elif strategy == "carbon":
            # 碳优化：碳强度越低越好
            max_carbon = 500  # gCO2/kWh
            return (max_carbon - region.carbon_intensity) / max_carbon * 100
        
        else:  # balanced
            # 平衡策略
            price_score = self._calculate_placement_score(task, region, pricing, "cost")
            perf_score = self._calculate_placement_score(task, region, pricing, "performance")
            carbon_score = self._calculate_placement_score(task, region, pricing, "carbon")
            return price_score * 0.5 + perf_score * 0.3 + carbon_score * 0.2
    
    def _check_compliance(self, region: CloudRegion, requirements: List[str]) -> bool:
        """检查合规要求"""
        # 简化的合规检查
        compliance_map = {
            "gdpr": ["eu-west-1", "eu-central-1", "westeurope", "northeurope", "europe-west1", "europe-west4"],
            "hipaa": ["us-east-1", "us-west-2", "eastus", "westus2", "us-central1", "us-west1"],
        }
        
        for req in requirements:
            if req.lower() in compliance_map:
                if region.region_code not in compliance_map[req.lower()]:
                    return False
        return True
    
    def estimate_monthly_cost(
        self,
        instance_spec: InstanceSpec,
        hours_per_day: float = 8,
        days_per_month: int = 22
    ) -> Dict[str, float]:
        """估算月度成本"""
        monthly_hours = hours_per_day * days_per_month
        
        # 假设不同云的价格范围
        aws_price = self._estimate_price(CloudProvider.AWS, instance_spec)
        azure_price = self._estimate_price(CloudProvider.AZURE, instance_spec)
        gcp_price = self._estimate_price(CloudProvider.GCP, instance_spec)
        
        return {
            "aws_on_demand": round(aws_price * monthly_hours, 2),
            "aws_spot": round(aws_price * 0.3 * monthly_hours, 2),
            "azure_on_demand": round(azure_price * monthly_hours, 2),
            "azure_spot": round(azure_price * 0.25 * monthly_hours, 2),
            "gcp_on_demand": round(gcp_price * monthly_hours, 2),
            "gcp_spot": round(gcp_price * 0.2 * monthly_hours, 2),
        }
    
    def _estimate_price(self, provider: CloudProvider, spec: InstanceSpec) -> float:
        """估算价格"""
        base = spec.vcpus * 0.04 + spec.memory_gb * 0.005
        if spec.gpu_count > 0:
            base += spec.gpu_count * 2.5
        
        multipliers = {
            CloudProvider.AWS: 1.0,
            CloudProvider.AZURE: 1.05,
            CloudProvider.GCP: 0.95
        }
        return base * multipliers.get(provider, 1.0)


class MultiCloudOrchestrator:
    """多云编排器 - 核心协调组件"""
    
    def __init__(self):
        self.providers: Dict[CloudProvider, CloudProviderInterface] = {}
        self.cost_optimizer = CostOptimizer()
        self.active_instances: Dict[str, ProvisionedInstance] = {}
        self._lock = asyncio.Lock()
        self._metrics_collector = None
    
    def register_provider(self, provider: CloudProviderInterface):
        """注册云提供商"""
        self.providers[provider.provider] = provider
        logger.info(f"Registered cloud provider: {provider.provider.value}")
    
    async def initialize(self):
        """初始化所有提供商"""
        await asyncio.gather(*[
            provider.initialize() 
            for provider in self.providers.values()
        ])
    
    async def provision(
        self,
        task: ComputeTask,
        optimization_strategy: str = "balanced",
        preferred_provider: Optional[CloudProvider] = None
    ) -> ProvisionedInstance:
        """
        调配计算资源
        
        Args:
            task: 计算任务
            optimization_strategy: 优化策略 (cost/performance/balanced/carbon)
            preferred_provider: 首选云提供商
        
        Returns:
            ProvisionedInstance: 已调配的实例
        """
        async with self._lock:
            # 获取优化后的放置选项
            options = await self.cost_optimizer.optimize_task_placement(
                task, 
                list(self.providers.values()),
                optimization_strategy
            )
            
            if not options:
                raise RuntimeError("No suitable region found for task")
            
            # 如果有首选提供商，优先尝试
            if preferred_provider:
                options = [
                    opt for opt in options 
                    if opt[0] == preferred_provider
                ] + [
                    opt for opt in options 
                    if opt[0] != preferred_provider
                ]
            
            # 尝试调配实例
            last_error = None
            for provider_enum, region, score, pricing in options:
                if score <= 0:
                    continue
                    
                provider = self.providers.get(provider_enum)
                if not provider:
                    continue
                
                try:
                    instance = await provider.provision_instance(
                        task.instance_spec,
                        region,
                        tags={"task_id": task.task_id, "priority": str(task.priority)}
                    )
                    
                    self.active_instances[instance.instance_id] = instance
                    
                    logger.info(
                        f"Successfully provisioned instance {instance.instance_id} "
                        f"on {provider_enum.value}/{region} with score {score:.2f}"
                    )
                    
                    return instance
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"Failed to provision on {provider_enum.value}/{region}: {e}")
                    continue
            
            raise RuntimeError(f"Failed to provision instance: {last_error}")
    
    async def terminate(self, instance_id: str) -> bool:
        """终止实例"""
        async with self._lock:
            instance = self.active_instances.get(instance_id)
            if not instance:
                logger.warning(f"Instance {instance_id} not found")
                return False
            
            provider = self.providers.get(instance.provider)
            if not provider:
                logger.error(f"Provider {instance.provider.value} not found")
                return False
            
            success = await provider.terminate_instance(instance_id, instance.region)
            
            if success:
                instance.status = "terminated"
                del self.active_instances[instance_id]
                logger.info(f"Instance {instance_id} terminated successfully")
            
            return success
    
    async def get_all_regions(self) -> List[CloudRegion]:
        """获取所有云的所有区域"""
        all_regions = []
        for provider in self.providers.values():
            regions = await provider.list_regions()
            all_regions.extend(regions)
        return all_regions
    
    async def get_cost_report(self) -> Dict:
        """获取成本报告"""
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "active_instances": len(self.active_instances),
            "cost_by_provider": defaultdict(float),
            "cost_by_region": defaultdict(float),
            "total_estimated_hourly_cost": 0.0,
            "instances": []
        }
        
        for instance in self.active_instances.values():
            if instance.status == "running":
                report["cost_by_provider"][instance.provider.value] += instance.cost_per_hour
                report["cost_by_region"][instance.region] += instance.cost_per_hour
                report["total_estimated_hourly_cost"] += instance.cost_per_hour
                
                report["instances"].append({
                    "instance_id": instance.instance_id,
                    "provider": instance.provider.value,
                    "region": instance.region,
                    "cost_per_hour": instance.cost_per_hour,
                    "created_at": instance.created_at.isoformat()
                })
        
        report["total_estimated_daily_cost"] = round(
            report["total_estimated_hourly_cost"] * 24, 2
        )
        report["total_estimated_monthly_cost"] = round(
            report["total_estimated_hourly_cost"] * 24 * 30, 2
        )
        
        return report
    
    async def migrate_instance(
        self,
        instance_id: str,
        target_provider: CloudProvider,
        target_region: str
    ) -> ProvisionedInstance:
        """迁移实例到新的提供商/区域"""
        source = self.active_instances.get(instance_id)
        if not source:
            raise ValueError(f"Instance {instance_id} not found")
        
        # 创建新实例
        new_task = ComputeTask(
            task_id=f"migrate-{instance_id}",
            instance_spec=source.instance_spec,
            region_preferences=[target_region],
            priority=10
        )
        
        # 调配新实例
        new_instance = await self.provision(
            new_task, 
            preferred_provider=target_provider
        )
        
        # 数据迁移逻辑 (简化)
        logger.info(f"Migrating data from {instance_id} to {new_instance.instance_id}")
        await asyncio.sleep(2)  # 模拟数据迁移
        
        # 终止旧实例
        await self.terminate(instance_id)
        
        return new_instance
    
    async def auto_scale(
        self,
        task_template: ComputeTask,
        current_load: float,
        target_load: float = 0.7,
        min_instances: int = 1,
        max_instances: int = 100
    ) -> Dict:
        """
        自动扩缩容
        
        Args:
            task_template: 任务模板
            current_load: 当前负载 (0-1)
            target_load: 目标负载
            min_instances: 最小实例数
            max_instances: 最大实例数
        """
        # 计算需要的实例数
        current_instances = len([
            i for i in self.active_instances.values()
            if i.status == "running"
        ])
        
        if current_load > target_load:
            # 需要扩容
            needed = int(current_instances * (current_load / target_load))
            needed = min(needed, max_instances)
            to_add = needed - current_instances
        else:
            # 可以缩容
            needed = int(current_instances * (current_load / target_load))
            needed = max(needed, min_instances)
            to_add = needed - current_instances
        
        result = {
            "current_instances": current_instances,
            "target_instances": needed,
            "instances_added": 0,
            "instances_removed": 0,
            "instances": []
        }
        
        if to_add > 0:
            # 添加实例
            for _ in range(to_add):
                instance = await self.provision(task_template)
                result["instances_added"] += 1
                result["instances"].append(instance.instance_id)
        
        elif to_add < 0:
            # 移除实例
            to_remove = abs(to_add)
            candidates = [
                (iid, inst) for iid, inst in self.active_instances.items()
                if inst.status == "running"
            ]
            # 按创建时间排序，优先移除最新的
            candidates.sort(key=lambda x: x[1].created_at, reverse=True)
            
            for i in range(min(to_remove, len(candidates))):
                await self.terminate(candidates[i][0])
                result["instances_removed"] += 1
        
        return result


# 示例使用
async def main():
    """示例：多云编排器使用"""
    
    # 创建编排器
    orchestrator = MultiCloudOrchestrator()
    
    # 注册云提供商
    aws = AWSProvider({
        "access_key_id": "AKIA...",
        "secret_access_key": "...",
        "default_region": "us-east-1"
    })
    
    azure = AzureProvider({
        "tenant_id": "...",
        "client_id": "...",
        "client_secret": "...",
        "subscription_id": "..."
    })
    
    gcp = GCPProvider({
        "service_account_key": {...}
    })
    
    orchestrator.register_provider(aws)
    orchestrator.register_provider(azure)
    orchestrator.register_provider(gcp)
    
    # 初始化
    await orchestrator.initialize()
    
    # 创建计算任务
    task = ComputeTask(
        task_id="simulation-001",
        instance_spec=InstanceSpec(
            instance_type=InstanceType.GPU,
            vcpus=8,
            memory_gb=64,
            gpu_count=1,
            gpu_type="v100",
            spot=True
        ),
        region_preferences=["us-east-1", "eastus", "us-central1"],
        priority=8,
        estimated_duration=7200,
        max_cost=5.0
    )
    
    # 成本优化放置
    print("=== 成本优化放置 ===")
    options = await orchestrator.cost_optimizer.optimize_task_placement(
        task, 
        [aws, azure, gcp],
        "cost"
    )
    for provider, region, score, pricing in options[:5]:
        print(f"  {provider.value}/{region}: score={score:.2f}, "
              f"price=${pricing.get_effective_price(True):.4f}/hr")
    
    # 调配实例
    print("\n=== 调配实例 ===")
    instance = await orchestrator.provision(task, optimization_strategy="cost")
    print(f"  Created: {instance.instance_id} on {instance.provider.value}")
    print(f"  Cost: ${instance.cost_per_hour:.4f}/hr")
    
    # 获取成本报告
    print("\n=== 成本报告 ===")
    report = await orchestrator.get_cost_report()
    print(f"  Active instances: {report['active_instances']}")
    print(f"  Hourly cost: ${report['total_estimated_hourly_cost']:.4f}")
    print(f"  Monthly estimate: ${report['total_estimated_monthly_cost']:.2f}")
    
    # 估算月度成本
    print("\n=== 月度成本估算 ===")
    estimates = orchestrator.cost_optimizer.estimate_monthly_cost(
        task.instance_spec,
        hours_per_day=8
    )
    for k, v in estimates.items():
        print(f"  {k}: ${v:.2f}")
    
    # 清理
    await orchestrator.terminate(instance.instance_id)
    print("\n=== 清理完成 ===")


if __name__ == "__main__":
    asyncio.run(main())
