#!/usr/bin/env python3
"""
A/B Testing Module
A/B测试模块 - 提供实验管理、流量分配和统计分析

Features:
- Multi-variant experiment management
- Traffic splitting and user assignment
- Statistical significance testing
- Automatic winner selection
- Feature flags integration
- Real-time experiment monitoring

Author: DFTLammps Platform
Version: 2.0.0
"""

import asyncio
import json
import hashlib
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any, Union, Set, Tuple
from enum import Enum, auto
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean, stdev
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """实验状态"""
    DRAFT = "draft"              # 草稿
    RUNNING = "running"          # 运行中
    PAUSED = "paused"            # 暂停
    COMPLETED = "completed"      # 已完成
    ARCHIVED = "archived"        # 已归档


class AssignmentMethod(Enum):
    """用户分配方法"""
    RANDOM = "random"            # 完全随机
    USER_ID = "user_id"          # 基于用户ID哈希
    SESSION = "session"          # 基于会话
    COOKIE = "cookie"            # 基于Cookie
    CONSISTENT = "consistent"    # 一致性哈希


class MetricType(Enum):
    """指标类型"""
    CONVERSION = "conversion"    # 转化率
    COUNT = "count"              # 计数
    SUM = "sum"                  # 求和
    AVERAGE = "average"          # 平均值
    RETENTION = "retention"      # 留存率
    CUSTOM = "custom"            # 自定义


@dataclass
class Variant:
    """实验变体"""
    variant_id: str
    name: str
    description: str = ""
    traffic_percentage: float = 50.0  # 流量百分比
    
    # 配置参数
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # 统计
    users_assigned: int = 0
    events_recorded: int = 0
    
    def __post_init__(self):
        if not self.variant_id:
            self.variant_id = hashlib.md5(
                f"{self.name}-{time.time()}".encode()
            ).hexdigest()[:16]


@dataclass
class ExperimentMetric:
    """实验指标"""
    metric_id: str
    name: str
    metric_type: MetricType
    description: str = ""
    
    # 对于转化指标
    event_name: str = ""
    
    # 目标方向
    higher_is_better: bool = True
    
    # 最小可检测效应 (MDE)
    minimum_detectable_effect: float = 0.05  # 5%
    
    def __post_init__(self):
        if not self.metric_id:
            self.metric_id = hashlib.md5(
                f"{self.name}-{time.time()}".encode()
            ).hexdigest()[:16]


@dataclass
class Experiment:
    """实验定义"""
    experiment_id: str
    name: str
    description: str
    
    # 变体
    control_variant: Variant
    treatment_variants: List[Variant]
    
    # 指标
    primary_metric: ExperimentMetric
    secondary_metrics: List[ExperimentMetric] = field(default_factory=list)
    guardrail_metrics: List[ExperimentMetric] = field(default_factory=list)
    
    # 实验配置
    assignment_method: AssignmentMethod = AssignmentMethod.USER_ID
    traffic_allocation: float = 100.0  # 参与实验的流量百分比
    
    # 时间安排
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # 状态
    status: ExperimentStatus = ExperimentStatus.DRAFT
    
    # 目标人群
    target_segments: List[str] = field(default_factory=list)  # 目标用户群体
    exclusion_segments: List[str] = field(default_factory=list)  # 排除的用户群体
    
    # 自动决策
    auto_stop_enabled: bool = True
    auto_stop_confidence: float = 0.95  # 置信度
    max_sample_size: Optional[int] = None
    
    # 元数据
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = hashlib.md5(
                f"{self.name}-{time.time()}".encode()
            ).hexdigest()[:16]
    
    def get_all_variants(self) -> List[Variant]:
        """获取所有变体"""
        return [self.control_variant] + self.treatment_variants
    
    def get_variant(self, variant_id: str) -> Optional[Variant]:
        """获取指定变体"""
        for v in self.get_all_variants():
            if v.variant_id == variant_id:
                return v
        return None


@dataclass
class UserAssignment:
    """用户分配记录"""
    user_id: str
    experiment_id: str
    variant_id: str
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricObservation:
    """指标观察值"""
    experiment_id: str
    variant_id: str
    user_id: str
    metric_id: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VariantStats:
    """变体统计"""
    variant_id: str
    variant_name: str
    
    # 样本量
    sample_size: int = 0
    
    # 指标统计
    metric_sum: float = 0.0
    metric_count: int = 0
    
    # 转化统计 (用于转化率)
    conversions: int = 0
    
    def get_mean(self) -> float:
        """获取平均值"""
        if self.metric_count == 0:
            return 0.0
        return self.metric_sum / self.metric_count
    
    def get_conversion_rate(self) -> float:
        """获取转化率"""
        if self.sample_size == 0:
            return 0.0
        return self.conversions / self.sample_size


@dataclass
class StatisticalResult:
    """统计结果"""
    control_stats: VariantStats
    treatment_stats: VariantStats
    
    # 相对提升
    relative_lift: float = 0.0
    
    # 置信区间
    confidence_interval_lower: float = 0.0
    confidence_interval_upper: float = 0.0
    
    # 统计显著性
    p_value: float = 1.0
    is_statistically_significant: bool = False
    
    # 检验功效 (Power)
    power: float = 0.0
    
    # 建议
    recommendation: str = ""  # "continue", "stop_control", "stop_treatment", "inconclusive"


class StatisticalEngine:
    """统计引擎"""
    
    @staticmethod
    def calculate_conversion_stats(
        control_conversions: int,
        control_sample: int,
        treatment_conversions: int,
        treatment_sample: int,
        confidence_level: float = 0.95
    ) -> StatisticalResult:
        """
        计算转化率统计检验 (Z检验)
        """
        # 转化率
        p_control = control_conversions / max(control_sample, 1)
        p_treatment = treatment_conversions / max(treatment_sample, 1)
        
        # 相对提升
        relative_lift = (p_treatment - p_control) / max(p_control, 0.0001)
        
        # 合并方差
        p_pooled = (control_conversions + treatment_conversions) / max(
            control_sample + treatment_sample, 1
        )
        se = math.sqrt(
            p_pooled * (1 - p_pooled) * (1/control_sample + 1/treatment_sample)
        )
        
        # Z分数
        z_score = (p_treatment - p_control) / max(se, 0.0001)
        
        # P值 (双侧检验)
        p_value = 2 * (1 - StatisticalEngine._normal_cdf(abs(z_score)))
        
        # 置信区间
        z_critical = StatisticalEngine._z_critical(confidence_level)
        margin = z_critical * se
        
        ci_lower = relative_lift - margin / max(p_control, 0.0001)
        ci_upper = relative_lift + margin / max(p_control, 0.0001)
        
        # 显著性判断
        is_significant = p_value < (1 - confidence_level)
        
        # 建议
        if is_significant and relative_lift > 0:
            recommendation = "stop_treatment"
        elif is_significant and relative_lift < 0:
            recommendation = "stop_control"
        else:
            recommendation = "continue"
        
        return StatisticalResult(
            control_stats=VariantStats(
                variant_id="control",
                variant_name="Control",
                sample_size=control_sample,
                conversions=control_conversions
            ),
            treatment_stats=VariantStats(
                variant_id="treatment",
                variant_name="Treatment",
                sample_size=treatment_sample,
                conversions=treatment_conversions
            ),
            relative_lift=round(relative_lift, 4),
            confidence_interval_lower=round(ci_lower, 4),
            confidence_interval_upper=round(ci_upper, 4),
            p_value=round(p_value, 4),
            is_statistically_significant=is_significant,
            recommendation=recommendation
        )
    
    @staticmethod
    def calculate_continuous_stats(
        control_values: List[float],
        treatment_values: List[float],
        confidence_level: float = 0.95
    ) -> StatisticalResult:
        """
        计算连续变量统计检验 (t检验简化版)
        """
        n1, n2 = len(control_values), len(treatment_values)
        
        if n1 < 2 or n2 < 2:
            return StatisticalResult(
                control_stats=VariantStats("control", "Control", n1),
                treatment_stats=VariantStats("treatment", "Treatment", n2),
                recommendation="continue"
            )
        
        mean1, mean2 = mean(control_values), mean(treatment_values)
        
        # 相对提升
        relative_lift = (mean2 - mean1) / max(abs(mean1), 0.0001)
        
        # 标准差
        std1 = stdev(control_values) if n1 > 1 else 0
        std2 = stdev(treatment_values) if n2 > 1 else 0
        
        # 标准误
        se = math.sqrt(std1**2/n1 + std2**2/n2)
        
        # t统计量
        t_stat = (mean2 - mean1) / max(se, 0.0001)
        
        # 简化的P值估计
        p_value = min(1.0, 2 / (1 + abs(t_stat)**2))
        
        is_significant = p_value < (1 - confidence_level)
        
        return StatisticalResult(
            control_stats=VariantStats(
                variant_id="control",
                variant_name="Control",
                sample_size=n1,
                metric_sum=sum(control_values),
                metric_count=n1
            ),
            treatment_stats=VariantStats(
                variant_id="treatment",
                variant_name="Treatment",
                sample_size=n2,
                metric_sum=sum(treatment_values),
                metric_count=n2
            ),
            relative_lift=round(relative_lift, 4),
            p_value=round(p_value, 4),
            is_statistically_significant=is_significant,
            recommendation="stop_treatment" if is_significant and relative_lift > 0 else "continue"
        )
    
    @staticmethod
    def sample_size_calculation(
        baseline_rate: float,
        mde: float,
        alpha: float = 0.05,
        power: float = 0.8
    ) -> int:
        """
        计算所需样本量
        
        Args:
            baseline_rate: 基准转化率
            mde: 最小可检测效应 (相对值)
            alpha: 显著性水平
            power: 检验功效
        """
        p1 = baseline_rate
        p2 = baseline_rate * (1 + mde)
        
        z_alpha = StatisticalEngine._z_critical(1 - alpha/2)
        z_beta = StatisticalEngine._z_critical(power)
        
        p_avg = (p1 + p2) / 2
        
        n = (
            (z_alpha * math.sqrt(2 * p_avg * (1 - p_avg)) +
             z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        ) / (p1 - p2) ** 2
        
        return int(math.ceil(n))
    
    @staticmethod
    def _normal_cdf(x: float) -> float:
        """标准正态分布累积分布函数"""
        import math
        return (1 + math.erf(x / math.sqrt(2))) / 2
    
    @staticmethod
    def _z_critical(confidence: float) -> float:
        """获取Z临界值 (简化近似)"""
        # 常用临界值
        z_values = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
        return z_values.get(confidence, 1.96)


class AssignmentEngine:
    """分配引擎"""
    
    def __init__(self):
        self.user_assignments: Dict[str, Dict[str, UserAssignment]] = defaultdict(dict)
        self._lock = asyncio.Lock()
    
    async def assign_user(
        self,
        user_id: str,
        experiment: Experiment,
        user_attributes: Optional[Dict] = None
    ) -> Optional[Variant]:
        """
        分配用户到实验变体
        
        Returns: 分配的变体，如果用户不应参与实验则返回None
        """
        # 检查实验状态
        if experiment.status != ExperimentStatus.RUNNING:
            return None
        
        # 检查用户是否已分配
        existing = self.user_assignments[user_id].get(experiment.experiment_id)
        if existing:
            return experiment.get_variant(existing.variant_id)
        
        # 检查目标人群
        if not self._check_eligibility(user_id, user_attributes, experiment):
            return None
        
        # 计算分配
        variant = self._calculate_assignment(user_id, experiment)
        
        if variant:
            async with self._lock:
                # 记录分配
                assignment = UserAssignment(
                    user_id=user_id,
                    experiment_id=experiment.experiment_id,
                    variant_id=variant.variant_id,
                    attributes=user_attributes or {}
                )
                self.user_assignments[user_id][experiment.experiment_id] = assignment
                variant.users_assigned += 1
        
        return variant
    
    def _check_eligibility(
        self,
        user_id: str,
        attributes: Optional[Dict],
        experiment: Experiment
    ) -> bool:
        """检查用户资格"""
        # 检查排除条件
        for segment in experiment.exclusion_segments:
            if self._is_in_segment(user_id, segment, attributes):
                return False
        
        # 检查目标条件
        if experiment.target_segments:
            for segment in experiment.target_segments:
                if self._is_in_segment(user_id, segment, attributes):
                    return True
            return False
        
        return True
    
    def _is_in_segment(
        self,
        user_id: str,
        segment: str,
        attributes: Optional[Dict]
    ) -> bool:
        """检查用户是否在指定用户群"""
        # 简化实现
        return True
    
    def _calculate_assignment(
        self,
        user_id: str,
        experiment: Experiment
    ) -> Optional[Variant]:
        """计算用户分配"""
        variants = experiment.get_all_variants()
        
        # 首先检查是否进入实验
        if random.random() * 100 > experiment.traffic_allocation:
            return None
        
        # 根据分配方法选择
        if experiment.assignment_method == AssignmentMethod.RANDOM:
            # 随机分配
            weights = [v.traffic_percentage for v in variants]
            return random.choices(variants, weights=weights)[0]
        
        elif experiment.assignment_method == AssignmentMethod.USER_ID:
            # 基于用户ID的确定性分配
            hash_input = f"{experiment.experiment_id}:{user_id}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            
            cumulative = 0
            for variant in variants:
                cumulative += variant.traffic_percentage
                if (hash_value % 100) < cumulative:
                    return variant
        
        elif experiment.assignment_method == AssignmentMethod.CONSISTENT:
            # 一致性哈希
            return self._consistent_hash(user_id, variants)
        
        # 默认随机
        return random.choice(variants)
    
    def _consistent_hash(self, user_id: str, variants: List[Variant]) -> Variant:
        """一致性哈希"""
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return variants[hash_value % len(variants)]
    
    def get_user_variant(
        self,
        user_id: str,
        experiment_id: str
    ) -> Optional[str]:
        """获取用户被分配的变体"""
        assignment = self.user_assignments[user_id].get(experiment_id)
        return assignment.variant_id if assignment else None


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.observations: List[MetricObservation] = []
        self.variant_stats: Dict[str, Dict[str, VariantStats]] = defaultdict(
            lambda: defaultdict(lambda: VariantStats("", ""))
        )
        self.user_conversions: Dict[str, Set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()
    
    async def record_event(
        self,
        experiment_id: str,
        variant_id: str,
        user_id: str,
        metric: ExperimentMetric,
        value: float = 1.0,
        metadata: Optional[Dict] = None
    ):
        """记录事件"""
        observation = MetricObservation(
            experiment_id=experiment_id,
            variant_id=variant_id,
            user_id=user_id,
            metric_id=metric.metric_id,
            value=value,
            metadata=metadata or {}
        )
        
        async with self._lock:
            self.observations.append(observation)
            
            # 更新统计
            key = f"{experiment_id}:{variant_id}"
            stats = self.variant_stats[experiment_id][metric.metric_id]
            stats.variant_id = variant_id
            
            if metric.metric_type == MetricType.CONVERSION:
                # 确保每个用户只计算一次转化
                user_key = f"{experiment_id}:{metric.metric_id}:{user_id}"
                if user_key not in self.user_conversions:
                    self.user_conversions[user_key].add(variant_id)
                    stats.conversions += int(value)
            
            stats.metric_sum += value
            stats.metric_count += 1
    
    def get_variant_stats(
        self,
        experiment_id: str,
        metric_id: str
    ) -> Dict[str, VariantStats]:
        """获取变体统计"""
        return dict(self.variant_stats[experiment_id].get(metric_id, {}))
    
    def get_user_values(
        self,
        experiment_id: str,
        variant_id: str,
        metric_id: str
    ) -> List[float]:
        """获取用户的所有观察值"""
        values = []
        for obs in self.observations:
            if (obs.experiment_id == experiment_id and
                obs.variant_id == variant_id and
                obs.metric_id == metric_id):
                values.append(obs.value)
        return values


class ABTestOrchestrator:
    """A/B测试编排器"""
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.assignment_engine = AssignmentEngine()
        self.metrics_collector = MetricsCollector()
        self.statistical_engine = StatisticalEngine()
        
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """启动编排器"""
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("A/B Test orchestrator started")
    
    async def stop(self):
        """停止编排器"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
        logger.info("A/B Test orchestrator stopped")
    
    async def _monitoring_loop(self):
        """监控循环 - 检查实验状态"""
        while self._running:
            try:
                await self._check_experiments()
                await asyncio.sleep(60)  # 每分钟检查一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _check_experiments(self):
        """检查实验状态"""
        for exp in self.experiments.values():
            if exp.status != ExperimentStatus.RUNNING:
                continue
            
            # 检查是否到达结束日期
            if exp.end_date and datetime.utcnow() > exp.end_date:
                await self.stop_experiment(exp.experiment_id, "schedule")
                continue
            
            # 检查自动停止条件
            if exp.auto_stop_enabled:
                should_stop = await self._check_auto_stop(exp)
                if should_stop:
                    await self.stop_experiment(exp.experiment_id, "auto")
    
    async def _check_auto_stop(self, experiment: Experiment) -> bool:
        """检查是否应该自动停止"""
        # 检查样本量
        if experiment.max_sample_size:
            total_assigned = sum(
                v.users_assigned for v in experiment.get_all_variants()
            )
            if total_assigned >= experiment.max_sample_size:
                return True
        
        # 检查统计显著性
        for treatment in experiment.treatment_variants:
            result = await self.get_statistical_result(
                experiment.experiment_id,
                experiment.control_variant.variant_id,
                treatment.variant_id,
                experiment.primary_metric.metric_id
            )
            
            if result and result.is_statistically_significant:
                if abs(result.relative_lift) > experiment.primary_metric.minimum_detectable_effect:
                    return True
        
        return False
    
    async def create_experiment(self, experiment: Experiment) -> str:
        """创建实验"""
        self.experiments[experiment.experiment_id] = experiment
        logger.info(f"Created experiment: {experiment.name} ({experiment.experiment_id})")
        return experiment.experiment_id
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """启动实验"""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return False
        
        exp.status = ExperimentStatus.RUNNING
        exp.start_date = datetime.utcnow()
        
        logger.info(f"Started experiment: {exp.name}")
        return True
    
    async def stop_experiment(self, experiment_id: str, reason: str = "manual") -> bool:
        """停止实验"""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return False
        
        exp.status = ExperimentStatus.COMPLETED
        exp.end_date = datetime.utcnow()
        
        logger.info(f"Stopped experiment {exp.name}: {reason}")
        return True
    
    async def get_variant_for_user(
        self,
        user_id: str,
        experiment_id: str,
        user_attributes: Optional[Dict] = None
    ) -> Optional[Variant]:
        """获取用户应该看到的变体"""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return None
        
        return await self.assignment_engine.assign_user(
            user_id, exp, user_attributes
        )
    
    async def track_event(
        self,
        experiment_id: str,
        variant_id: str,
        user_id: str,
        metric_id: str,
        value: float = 1.0
    ):
        """追踪事件"""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return
        
        # 查找指标
        metric = None
        if exp.primary_metric.metric_id == metric_id:
            metric = exp.primary_metric
        else:
            for m in exp.secondary_metrics + exp.guardrail_metrics:
                if m.metric_id == metric_id:
                    metric = m
                    break
        
        if metric:
            await self.metrics_collector.record_event(
                experiment_id, variant_id, user_id, metric, value
            )
    
    async def get_statistical_result(
        self,
        experiment_id: str,
        control_variant_id: str,
        treatment_variant_id: str,
        metric_id: str
    ) -> Optional[StatisticalResult]:
        """获取统计结果"""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return None
        
        # 获取统计数据
        control_stats = self.metrics_collector.get_variant_stats(
            experiment_id, metric_id
        ).get(control_variant_id)
        
        treatment_stats = self.metrics_collector.get_variant_stats(
            experiment_id, metric_id
        ).get(treatment_variant_id)
        
        if not control_stats or not treatment_stats:
            return None
        
        # 查找指标类型
        metric = exp.primary_metric
        if metric.metric_id != metric_id:
            for m in exp.secondary_metrics + exp.guardrail_metrics:
                if m.metric_id == metric_id:
                    metric = m
                    break
        
        # 计算统计结果
        if metric.metric_type == MetricType.CONVERSION:
            return self.statistical_engine.calculate_conversion_stats(
                control_stats.conversions,
                control_stats.sample_size,
                treatment_stats.conversions,
                treatment_stats.sample_size,
                confidence_level=exp.auto_stop_confidence
            )
        else:
            control_values = self.metrics_collector.get_user_values(
                experiment_id, control_variant_id, metric_id
            )
            treatment_values = self.metrics_collector.get_user_values(
                experiment_id, treatment_variant_id, metric_id
            )
            
            return self.statistical_engine.calculate_continuous_stats(
                control_values,
                treatment_values,
                confidence_level=exp.auto_stop_confidence
            )
    
    def get_experiment_report(self, experiment_id: str) -> Optional[Dict]:
        """获取实验报告"""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return None
        
        return {
            "experiment": {
                "id": exp.experiment_id,
                "name": exp.name,
                "status": exp.status.value,
                "start_date": exp.start_date.isoformat() if exp.start_date else None,
                "end_date": exp.end_date.isoformat() if exp.end_date else None,
            },
            "variants": [
                {
                    "id": v.variant_id,
                    "name": v.name,
                    "users_assigned": v.users_assigned,
                    "traffic_percentage": v.traffic_percentage
                }
                for v in exp.get_all_variants()
            ],
            "primary_metric": {
                "id": exp.primary_metric.metric_id,
                "name": exp.primary_metric.name,
                "type": exp.primary_metric.metric_type.value
            }
        }


# 示例使用
async def demo():
    """A/B测试演示"""
    
    orchestrator = ABTestOrchestrator()
    await orchestrator.start()
    
    try:
        # 创建实验
        print("=== 创建A/B测试实验 ===")
        
        experiment = Experiment(
            experiment_id="",
            name="New Algorithm Optimization",
            description="测试新的分子模拟算法优化",
            control_variant=Variant(
                variant_id="",
                name="Control",
                description="当前算法",
                traffic_percentage=50,
                parameters={"algorithm_version": "v1"}
            ),
            treatment_variants=[
                Variant(
                    variant_id="",
                    name="Treatment A",
                    description="新算法 - 优化版",
                    traffic_percentage=25,
                    parameters={"algorithm_version": "v2", "optimization": "level1"}
                ),
                Variant(
                    variant_id="",
                    name="Treatment B",
                    description="新算法 - 激进版",
                    traffic_percentage=25,
                    parameters={"algorithm_version": "v2", "optimization": "level2"}
                )
            ],
            primary_metric=ExperimentMetric(
                metric_id="",
                name="Simulation Convergence Rate",
                metric_type=MetricType.CONVERSION,
                description="模拟成功收敛的比例",
                minimum_detectable_effect=0.05
            ),
            secondary_metrics=[
                ExperimentMetric(
                    metric_id="",
                    name="Average Simulation Time",
                    metric_type=MetricType.AVERAGE,
                    description="平均模拟时间",
                    higher_is_better=False
                )
            ],
            traffic_allocation=100.0,
            auto_stop_enabled=True,
            auto_stop_confidence=0.95
        )
        
        exp_id = await orchestrator.create_experiment(experiment)
        print(f"  Created experiment: {experiment.name} ({exp_id[:8]}...)")
        print(f"    Variants: {len(experiment.get_all_variants())}")
        print(f"    Primary metric: {experiment.primary_metric.name}")
        
        # 启动实验
        await orchestrator.start_experiment(exp_id)
        print("  Experiment started")
        
        # 模拟用户分配
        print("\n=== 用户分配测试 ===")
        
        users = [f"user_{i}" for i in range(1000)]
        assignment_counts = defaultdict(int)
        
        for user_id in users:
            variant = await orchestrator.get_variant_for_user(user_id, exp_id)
            if variant:
                assignment_counts[variant.name] += 1
        
        print("  Assignment distribution:")
        for name, count in assignment_counts.items():
            percentage = count / len(users) * 100
            print(f"    {name}: {count} users ({percentage:.1f}%)")
        
        # 模拟事件追踪
        print("\n=== 模拟事件追踪 ===")
        
        # 模拟转化事件
        for user_id in users[:500]:  # 前500个用户
            variant = await orchestrator.get_variant_for_user(user_id, exp_id)
            if variant:
                # 控制组转化率 20%
                # 实验组转化率 25%
                if variant.name == "Control":
                    converted = random.random() < 0.20
                else:
                    converted = random.random() < 0.25
                
                await orchestrator.track_event(
                    exp_id,
                    variant.variant_id,
                    user_id,
                    experiment.primary_metric.metric_id,
                    1.0 if converted else 0.0
                )
        
        print(f"  Tracked events for 500 users")
        
        # 统计分析
        print("\n=== 统计分析 ===")
        
        for treatment in experiment.treatment_variants:
            result = await orchestrator.get_statistical_result(
                exp_id,
                experiment.control_variant.variant_id,
                treatment.variant_id,
                experiment.primary_metric.metric_id
            )
            
            if result:
                print(f"\n  {treatment.name} vs Control:")
                print(f"    Control: {result.control_stats.conversions}/{result.control_stats.sample_size} "
                      f"({result.control_stats.get_conversion_rate():.2%})")
                print(f"    Treatment: {result.treatment_stats.conversions}/{result.treatment_stats.sample_size} "
                      f"({result.treatment_stats.get_conversion_rate():.2%})")
                print(f"    Relative lift: {result.relative_lift:+.2%}")
                print(f"    95% CI: [{result.confidence_interval_lower:+.2%}, {result.confidence_interval_upper:+.2%}]")
                print(f"    P-value: {result.p_value:.4f}")
                print(f"    Significant: {result.is_statistically_significant}")
                print(f"    Recommendation: {result.recommendation}")
        
        # 样本量计算
        print("\n=== 样本量计算 ===")
        
        required_n = StatisticalEngine.sample_size_calculation(
            baseline_rate=0.20,
            mde=0.05,
            alpha=0.05,
            power=0.8
        )
        print(f"  Required sample size per variant: {required_n:,}")
        print(f"  Total required: {required_n * 2:,}")
        
        # 实验报告
        print("\n=== 实验报告 ===")
        report = orchestrator.get_experiment_report(exp_id)
        print(f"  Experiment: {report['experiment']['name']}")
        print(f"  Status: {report['experiment']['status']}")
        print(f"  Variants:")
        for v in report['variants']:
            print(f"    - {v['name']}: {v['users_assigned']} users")
        
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(demo())
