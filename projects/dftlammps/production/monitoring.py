#!/usr/bin/env python3
"""
Production Monitoring Module
生产监控告警模块 - 提供全面的系统监控和智能告警

Features:
- Multi-dimensional metrics collection
- Intelligent alerting with anomaly detection
- Custom dashboard support
- SLA monitoring
- Predictive alerting
- Multi-channel notifications

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
from datetime import datetime, timedelta
import statistics
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"      # 累加计数器
    GAUGE = "gauge"          # 瞬时值
    HISTOGRAM = "histogram"  # 分布直方图
    SUMMARY = "summary"      # 摘要统计


class AlertSeverity(Enum):
    """告警严重级别"""
    CRITICAL = "critical"    # 严重 - 立即处理
    HIGH = "high"            # 高 - 1小时内处理
    MEDIUM = "medium"        # 中 - 4小时内处理
    LOW = "low"              # 低 - 24小时内处理
    INFO = "info"            # 信息 - 无需处理


class AlertState(Enum):
    """告警状态"""
    PENDING = "pending"      # 待触发
    FIRING = "firing"        # 已触发
    RESOLVED = "resolved"    # 已解决
    SUPPRESSED = "suppressed"  # 已抑制


class NotificationChannel(Enum):
    """通知渠道"""
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    SMS = "sms"
    DISCORD = "discord"


@dataclass
class Metric:
    """指标数据点"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels
        }


@dataclass
class TimeSeries:
    """时间序列数据"""
    metric_name: str
    labels: Dict[str, str]
    values: deque = field(default_factory=lambda: deque(maxlen=10000))
    
    def add(self, value: float, timestamp: Optional[datetime] = None):
        """添加数据点"""
        self.values.append({
            "timestamp": timestamp or datetime.utcnow(),
            "value": value
        })
    
    def get_range(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[Dict]:
        """获取时间范围内的数据"""
        result = []
        for point in self.values:
            if start and point["timestamp"] < start:
                continue
            if end and point["timestamp"] > end:
                continue
            result.append(point)
        return result
    
    def get_stats(self, window_seconds: int = 300) -> Dict:
        """获取统计信息"""
        cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)
        recent = [p["value"] for p in self.values if p["timestamp"] > cutoff]
        
        if not recent:
            return {"count": 0}
        
        return {
            "count": len(recent),
            "min": min(recent),
            "max": max(recent),
            "avg": statistics.mean(recent),
            "std": statistics.stdev(recent) if len(recent) > 1 else 0,
            "p50": statistics.median(recent),
            "p95": sorted(recent)[int(len(recent) * 0.95)] if len(recent) >= 20 else max(recent),
            "p99": sorted(recent)[int(len(recent) * 0.99)] if len(recent) >= 100 else max(recent)
        }


@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    description: str
    
    # 查询条件
    metric_name: str
    query: str  # 查询表达式
    labels_filter: Dict[str, str] = field(default_factory=dict)
    
    # 告警条件
    operator: str = ">"  # >, <, >=, <=, ==, !=
    threshold: float = 0.0
    duration_seconds: int = 60  # 持续多久触发
    
    # 告警属性
    severity: AlertSeverity = AlertSeverity.MEDIUM
    auto_resolve: bool = True
    
    # 通知配置
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    escalation_policy: Optional[str] = None
    
    # 抑制规则
    suppress_alerts: List[str] = field(default_factory=list)
    silence_duration_minutes: int = 0
    
    # 元数据
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.rule_id:
            self.rule_id = hashlib.md5(
                f"{self.name}-{time.time()}".encode()
            ).hexdigest()[:16]


@dataclass
class Alert:
    """告警实例"""
    alert_id: str
    rule_id: str
    rule_name: str
    severity: AlertSeverity
    state: AlertState
    
    # 告警内容
    summary: str
    description: str
    value: float
    threshold: float
    labels: Dict[str, str]
    
    # 时间戳
    starts_at: datetime
    ends_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    # 通知记录
    notifications_sent: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = hashlib.md5(
                f"{self.rule_id}-{time.time()}".encode()
            ).hexdigest()[:16]
    
    def duration_seconds(self) -> float:
        """告警持续时间"""
        end = self.resolved_at or self.ends_at or datetime.utcnow()
        return (end - self.starts_at).total_seconds()
    
    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "state": self.state.value,
            "summary": self.summary,
            "description": self.description,
            "value": self.value,
            "threshold": self.threshold,
            "labels": self.labels,
            "starts_at": self.starts_at.isoformat(),
            "duration_seconds": self.duration_seconds()
        }


@dataclass
class SLITarget:
    """SLI/SLO 目标"""
    name: str
    metric_name: str
    target_percentage: float  # 如 99.9
    time_window_days: int = 30
    
    # 计算方式
    good_events_query: str = ""
    total_events_query: str = ""


@dataclass
class SLIResult:
    """SLI 计算结果"""
    sli_name: str
    target_percentage: float
    actual_percentage: float
    status: str  # "meeting", "at_risk", "breached"
    error_budget_remaining: float  # 百分比
    burn_rate: float  # 错误预算消耗速度


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, retention_hours: int = 24):
        self.time_series: Dict[str, TimeSeries] = {}
        self.retention_hours = retention_hours
        self._lock = asyncio.Lock()
    
    def _get_key(self, metric_name: str, labels: Dict[str, str]) -> str:
        """生成时间序列键"""
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{metric_name}:{label_str}"
    
    async def record(self, metric: Metric):
        """记录指标"""
        async with self._lock:
            key = self._get_key(metric.name, metric.labels)
            
            if key not in self.time_series:
                self.time_series[key] = TimeSeries(
                    metric_name=metric.name,
                    labels=metric.labels
                )
            
            self.time_series[key].add(metric.value, metric.timestamp)
    
    async def record_many(self, metrics: List[Metric]):
        """批量记录指标"""
        for metric in metrics:
            await self.record(metric)
    
    def query(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[Dict]:
        """查询指标"""
        results = []
        
        for key, ts in self.time_series.items():
            if ts.metric_name == metric_name:
                if labels:
                    # 检查标签匹配
                    match = all(
                        ts.labels.get(k) == v
                        for k, v in labels.items()
                    )
                    if not match:
                        continue
                
                points = ts.get_range(start, end)
                for point in points:
                    results.append({
                        **point,
                        "labels": ts.labels
                    })
        
        return sorted(results, key=lambda x: x["timestamp"])
    
    def query_stats(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
        window_seconds: int = 300
    ) -> Dict[str, Dict]:
        """查询统计信息"""
        results = {}
        
        for key, ts in self.time_series.items():
            if ts.metric_name == metric_name:
                if labels:
                    match = all(
                        ts.labels.get(k) == v
                        for k, v in labels.items()
                    )
                    if not match:
                        continue
                
                results[key] = ts.get_stats(window_seconds)
        
        return results
    
    def aggregate(
        self,
        metric_name: str,
        aggregation: str = "avg",  # avg, sum, min, max, count
        group_by: Optional[List[str]] = None,
        window_seconds: int = 300
    ) -> Dict:
        """聚合查询"""
        # 获取所有匹配的时间序列
        series_list = [
            ts for ts in self.time_series.values()
            if ts.metric_name == metric_name
        ]
        
        if not series_list:
            return {"result": None}
        
        # 计算每个系列的最新值
        values = []
        for ts in series_list:
            stats = ts.get_stats(window_seconds)
            if stats["count"] > 0:
                values.append(stats.get(aggregation, stats["avg"]))
        
        if not values:
            return {"result": None}
        
        # 执行最终聚合
        if aggregation == "sum":
            result = sum(values)
        elif aggregation == "avg":
            result = statistics.mean(values)
        elif aggregation == "min":
            result = min(values)
        elif aggregation == "max":
            result = max(values)
        elif aggregation == "count":
            result = len(values)
        else:
            result = statistics.mean(values)
        
        return {
            "result": result,
            "series_count": len(series_list),
            "data_points": sum(len(ts.values) for ts in series_list)
        }
    
    async def cleanup(self):
        """清理过期数据"""
        cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)
        
        async with self._lock:
            for ts in self.time_series.values():
                # 清理超过保留期的数据
                while ts.values and ts.values[0]["timestamp"] < cutoff:
                    ts.values.popleft()


class AlertManager:
    """告警管理器"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: Dict[str, Alert] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        
        # 通知处理器
        self.notification_handlers: Dict[NotificationChannel, Callable] = {}
        
        # 评估状态
        self.rule_eval_state: Dict[str, Dict] = defaultdict(lambda: {
            "first_triggered": None,
            "last_evaluated": None
        })
    
    def register_notification_handler(
        self,
        channel: NotificationChannel,
        handler: Callable[[Alert], Any]
    ):
        """注册通知处理器"""
        self.notification_handlers[channel] = handler
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """移除告警规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False
    
    async def evaluate_rules(self):
        """评估所有告警规则"""
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            try:
                await self._evaluate_rule(rule)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
    
    async def _evaluate_rule(self, rule: AlertRule):
        """评估单个规则"""
        # 查询当前指标值
        stats = self.metrics_collector.query_stats(
            rule.metric_name,
            rule.labels_filter,
            window_seconds=rule.duration_seconds
        )
        
        if not stats:
            return
        
        # 获取聚合值 (使用平均值)
        current_value = None
        for key, s in stats.items():
            if s["count"] > 0:
                current_value = s["avg"]
                break
        
        if current_value is None:
            return
        
        # 评估条件
        triggered = self._check_condition(
            current_value, rule.threshold, rule.operator
        )
        
        state = self.rule_eval_state[rule.rule_id]
        now = datetime.utcnow()
        
        if triggered:
            if state["first_triggered"] is None:
                state["first_triggered"] = now
            
            # 检查是否满足持续时间
            elapsed = (now - state["first_triggered"]).total_seconds()
            
            if elapsed >= rule.duration_seconds:
                # 触发告警
                await self._fire_alert(rule, current_value)
        else:
            # 重置触发状态
            state["first_triggered"] = None
            
            # 检查是否有活动告警需要解决
            await self._maybe_resolve_alert(rule, current_value)
        
        state["last_evaluated"] = now
    
    def _check_condition(self, value: float, threshold: float, operator: str) -> bool:
        """检查条件"""
        ops = {
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        return ops.get(operator, lambda a, b: False)(value, threshold)
    
    async def _fire_alert(self, rule: AlertRule, value: float):
        """触发告警"""
        # 检查是否已有相同规则的活动告警
        alert_key = f"{rule.rule_id}:{json.dumps(rule.labels_filter, sort_keys=True)}"
        
        if alert_key in self.active_alerts:
            # 更新现有告警
            return
        
        # 创建新告警
        alert = Alert(
            alert_id="",
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            state=AlertState.FIRING,
            summary=f"{rule.name}: {value:.2f} {rule.operator} {rule.threshold}",
            description=rule.description,
            value=value,
            threshold=rule.threshold,
            labels=rule.labels_filter,
            starts_at=datetime.utcnow()
        )
        
        self.alerts[alert.alert_id] = alert
        self.active_alerts[alert_key] = alert
        
        logger.warning(f"Alert firing: {alert.summary}")
        
        # 发送通知
        await self._send_notifications(alert, rule)
    
    async def _maybe_resolve_alert(self, rule: AlertRule, value: float):
        """尝试解决告警"""
        if not rule.auto_resolve:
            return
        
        alert_key = f"{rule.rule_id}:{json.dumps(rule.labels_filter, sort_keys=True)}"
        
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.state = AlertState.RESOLVED
            alert.resolved_at = datetime.utcnow()
            alert.ends_at = datetime.utcnow()
            
            del self.active_alerts[alert_key]
            self.alert_history.append(alert)
            
            logger.info(f"Alert resolved: {alert.rule_name}")
    
    async def _send_notifications(self, alert: Alert, rule: AlertRule):
        """发送通知"""
        for channel in rule.notification_channels:
            handler = self.notification_handlers.get(channel)
            if handler:
                try:
                    await handler(alert)
                    alert.notifications_sent.append({
                        "channel": channel.value,
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": "sent"
                    })
                except Exception as e:
                    logger.error(f"Failed to send notification to {channel}: {e}")
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """获取活动告警"""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.severity.value, reverse=True)
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """确认告警"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = user
            logger.info(f"Alert {alert_id} acknowledged by {user}")
            return True
        return False
    
    def get_alert_stats(self) -> Dict:
        """获取告警统计"""
        return {
            "active": len(self.active_alerts),
            "by_severity": {
                severity.value: len([
                    a for a in self.active_alerts.values()
                    if a.severity == severity
                ])
                for severity in AlertSeverity
            },
            "total_history": len(self.alert_history)
        }


class SLIMonitor:
    """SLI/SLO 监控器"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.targets: Dict[str, SLITarget] = {}
        self.results: Dict[str, SLIResult] = {}
    
    def add_target(self, target: SLITarget):
        """添加SLI目标"""
        self.targets[target.name] = target
        logger.info(f"Added SLI target: {target.name} ({target.target_percentage}%)")
    
    async def calculate_sli(self, target_name: str) -> Optional[SLIResult]:
        """计算SLI"""
        target = self.targets.get(target_name)
        if not target:
            return None
        
        # 查询good events和total events
        # 简化实现：从指标中计算
        good_stats = self.metrics_collector.aggregate(
            target.good_events_query or target.metric_name,
            aggregation="sum",
            window_seconds=target.time_window_days * 86400
        )
        
        total_stats = self.metrics_collector.aggregate(
            target.total_events_query or f"{target.metric_name}_total",
            aggregation="sum",
            window_seconds=target.time_window_days * 86400
        )
        
        good = good_stats.get("result", 0) or 0
        total = total_stats.get("result", 1) or 1
        
        actual_percentage = (good / total) * 100 if total > 0 else 100
        
        # 确定状态
        if actual_percentage >= target.target_percentage:
            status = "meeting"
        elif actual_percentage >= target.target_percentage * 0.95:
            status = "at_risk"
        else:
            status = "breached"
        
        # 计算错误预算
        error_budget_remaining = max(0, target.target_percentage - (100 - actual_percentage))
        
        result = SLIResult(
            sli_name=target.name,
            target_percentage=target.target_percentage,
            actual_percentage=round(actual_percentage, 3),
            status=status,
            error_budget_remaining=round(error_budget_remaining, 3),
            burn_rate=0.0  # 简化实现
        )
        
        self.results[target_name] = result
        return result
    
    async def calculate_all(self) -> Dict[str, SLIResult]:
        """计算所有SLI"""
        for name in self.targets.keys():
            await self.calculate_sli(name)
        return self.results


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.baselines: Dict[str, Dict] = {}
    
    async def detect_anomalies(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
        sensitivity: float = 2.0  # 标准差倍数
    ) -> List[Dict]:
        """检测异常"""
        stats = self.metrics_collector.query_stats(metric_name, labels, window_seconds=3600)
        
        anomalies = []
        
        for key, s in stats.items():
            if s["count"] < 10:
                continue
            
            mean = s["avg"]
            std = s["std"]
            
            # 获取最新值
            ts = self.metrics_collector.time_series.get(key)
            if not ts or not ts.values:
                continue
            
            latest = ts.values[-1]["value"]
            
            # 检查是否异常
            z_score = abs(latest - mean) / max(std, 0.0001)
            
            if z_score > sensitivity:
                anomalies.append({
                    "metric": metric_name,
                    "labels": ts.labels,
                    "latest_value": latest,
                    "expected_range": [mean - sensitivity * std, mean + sensitivity * std],
                    "z_score": round(z_score, 2),
                    "severity": "high" if z_score > sensitivity * 2 else "medium"
                })
        
        return anomalies


class MonitoringDashboard:
    """监控仪表盘"""
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alert_manager: AlertManager,
        sli_monitor: SLIMonitor
    ):
        self.metrics = metrics_collector
        self.alerts = alert_manager
        self.sli = sli_monitor
        
        self.dashboards: Dict[str, Dict] = {}
    
    def create_dashboard(self, name: str, widgets: List[Dict]) -> str:
        """创建仪表盘"""
        dashboard_id = hashlib.md5(f"{name}-{time.time()}".encode()).hexdigest()[:16]
        
        self.dashboards[dashboard_id] = {
            "id": dashboard_id,
            "name": name,
            "widgets": widgets,
            "created_at": datetime.utcnow().isoformat()
        }
        
        return dashboard_id
    
    def get_dashboard_data(self, dashboard_id: str) -> Optional[Dict]:
        """获取仪表盘数据"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return None
        
        # 获取每个widget的数据
        widgets_data = []
        for widget in dashboard["widgets"]:
            widget_type = widget.get("type")
            
            if widget_type == "metric":
                data = self.metrics.query(
                    widget["metric_name"],
                    widget.get("labels"),
                    start=datetime.utcnow() - timedelta(hours=widget.get("hours", 1))
                )
                widgets_data.append({**widget, "data": data})
            
            elif widget_type == "alert_list":
                alerts = self.alerts.get_active_alerts()
                widgets_data.append({**widget, "data": [a.to_dict() for a in alerts]})
            
            elif widget_type == "sli":
                sli_data = self.sli.results.get(widget.get("sli_name"), {})
                widgets_data.append({**widget, "data": sli_data})
            
            else:
                widgets_data.append(widget)
        
        return {
            **dashboard,
            "widgets": widgets_data
        }


class MonitoringOrchestrator:
    """监控编排器"""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.alerts = AlertManager(self.metrics)
        self.sli = SLIMonitor(self.metrics)
        self.anomalies = AnomalyDetector(self.metrics)
        self.dashboards = MonitoringDashboard(self.metrics, self.alerts, self.sli)
        
        self._running = False
        self._eval_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动监控"""
        self._running = True
        self._eval_task = asyncio.create_task(self._evaluation_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Monitoring orchestrator started")
    
    async def stop(self):
        """停止监控"""
        self._running = False
        
        if self._eval_task:
            self._eval_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        logger.info("Monitoring orchestrator stopped")
    
    async def _evaluation_loop(self):
        """告警评估循环"""
        while self._running:
            try:
                await self.alerts.evaluate_rules()
                await asyncio.sleep(15)  # 每15秒评估一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Evaluation error: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_loop(self):
        """清理循环"""
        while self._running:
            try:
                await self.metrics.cleanup()
                await asyncio.sleep(3600)  # 每小时清理一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(300)
    
    async def record_metric(self, name: str, value: float, **labels):
        """记录指标"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels
        )
        await self.metrics.record(metric)
    
    def add_alert_rule(self, **kwargs) -> str:
        """添加告警规则"""
        rule = AlertRule(**kwargs)
        self.alerts.add_rule(rule)
        return rule.rule_id
    
    def get_system_health(self) -> Dict:
        """获取系统健康状态"""
        return {
            "metrics": {
                "total_series": len(self.metrics.time_series),
                "data_points": sum(
                    len(ts.values) for ts in self.metrics.time_series.values()
                )
            },
            "alerts": self.alerts.get_alert_stats(),
            "slis": {
                name: {
                    "target": r.target_percentage,
                    "actual": r.actual_percentage,
                    "status": r.status
                }
                for name, r in self.sli.results.items()
            }
        }


# 示例使用
async def demo():
    """监控告警演示"""
    
    orchestrator = MonitoringOrchestrator()
    await orchestrator.start()
    
    try:
        # 注册通知处理器
        async def slack_handler(alert: Alert):
            print(f"  [SLACK] {alert.severity.value.upper()}: {alert.summary}")
        
        async def pagerduty_handler(alert: Alert):
            print(f"  [PAGERDUTY] {alert.severity.value.upper()}: {alert.summary}")
        
        orchestrator.alerts.register_notification_handler(
            NotificationChannel.SLACK, slack_handler
        )
        orchestrator.alerts.register_notification_handler(
            NotificationChannel.PAGERDUTY, pagerduty_handler
        )
        
        # 添加告警规则
        print("=== 配置告警规则 ===")
        
        rules = [
            AlertRule(
                rule_id="",
                name="High CPU Usage",
                description="CPU使用率超过80%",
                metric_name="cpu_usage_percent",
                operator=">",
                threshold=80.0,
                duration_seconds=60,
                severity=AlertSeverity.HIGH,
                notification_channels=[NotificationChannel.SLACK]
            ),
            AlertRule(
                rule_id="",
                name="High Memory Usage",
                description="内存使用率超过90%",
                metric_name="memory_usage_percent",
                operator=">",
                threshold=90.0,
                duration_seconds=120,
                severity=AlertSeverity.CRITICAL,
                notification_channels=[NotificationChannel.PAGERDUTY, NotificationChannel.SLACK]
            ),
            AlertRule(
                rule_id="",
                name="High Error Rate",
                description="错误率超过5%",
                metric_name="error_rate_percent",
                operator=">",
                threshold=5.0,
                duration_seconds=30,
                severity=AlertSeverity.CRITICAL,
                notification_channels=[NotificationChannel.PAGERDUTY]
            ),
        ]
        
        for rule in rules:
            rid = orchestrator.add_alert_rule(**{k: v for k, v in asdict(rule).items()})
            print(f"  Added: {rule.name} ({rid[:8]}...)")
        
        # 模拟指标数据
        print("\n=== 模拟系统指标 ===")
        
        for i in range(30):
            # 正常范围的指标
            await orchestrator.record_metric(
                "cpu_usage_percent",
                random.uniform(30, 70),
                host=f"server-{i % 5}",
                region="us-east"
            )
            
            await orchestrator.record_metric(
                "memory_usage_percent",
                random.uniform(40, 80),
                host=f"server-{i % 5}"
            )
            
            await orchestrator.record_metric(
                "request_latency_ms",
                random.uniform(10, 100),
                endpoint="/api/predict"
            )
            
            # 每10次产生一个异常值
            if i % 10 == 0:
                await orchestrator.record_metric(
                    "cpu_usage_percent",
                    random.uniform(85, 95),  # 触发告警
                    host="server-0",
                    region="us-east"
                )
            
            await asyncio.sleep(0.1)
        
        # 手动触发评估
        await orchestrator.alerts.evaluate_rules()
        
        # 查看告警
        print("\n=== 当前告警 ===")
        active_alerts = orchestrator.alerts.get_active_alerts()
        if active_alerts:
            for alert in active_alerts:
                print(f"  [{alert.severity.value.upper()}] {alert.summary}")
        else:
            print("  No active alerts")
        
        # 添加SLI目标
        print("\n=== 配置SLI/SLO ===")
        
        sli_targets = [
            SLITarget(
                name="api-availability",
                metric_name="request_success",
                target_percentage=99.9,
                time_window_days=30
            ),
            SLITarget(
                name="api-latency",
                metric_name="request_latency_ms",
                target_percentage=99.0,  # 99%请求在目标时间内
                time_window_days=7
            ),
        ]
        
        for target in sli_targets:
            orchestrator.sli.add_target(target)
            print(f"  Added: {target.name} (target: {target.target_percentage}%)")
        
        # 模拟成功请求数据
        for i in range(1000):
            await orchestrator.record_metric(
                "request_success",
                1,  # 成功
                endpoint="/api/predict"
            )
        
        # 计算SLI
        await orchestrator.sli.calculate_all()
        
        print("\n=== SLI状态 ===")
        for name, result in orchestrator.sli.results.items():
            print(f"  {name}:")
            print(f"    Target: {result.target_percentage}%")
            print(f"    Actual: {result.actual_percentage}%")
            print(f"    Status: {result.status}")
        
        # 异常检测
        print("\n=== 异常检测 ===")
        
        # 添加一些异常值
        for _ in range(5):
            await orchestrator.record_metric(
                "request_latency_ms",
                random.uniform(500, 1000),  # 异常高延迟
                endpoint="/api/predict"
            )
        
        anomalies = await orchestrator.anomalies.detect_anomalies(
            "request_latency_ms",
            sensitivity=2.5
        )
        
        if anomalies:
            for a in anomalies:
                print(f"  Anomaly detected: {a['metric']}")
                print(f"    Value: {a['latest_value']:.2f}")
                print(f"    Expected: {a['expected_range'][0]:.2f} - {a['expected_range'][1]:.2f}")
        else:
            print("  No anomalies detected")
        
        # 系统健康状态
        print("\n=== 系统健康状态 ===")
        health = orchestrator.get_system_health()
        print(f"  Metrics: {health['metrics']['total_series']} series, "
              f"{health['metrics']['data_points']} data points")
        print(f"  Active alerts: {health['alerts']['active']}")
        print(f"  Alerts by severity: {health['alerts']['by_severity']}")
        
        # 创建仪表盘
        print("\n=== 创建监控仪表盘 ===")
        
        dashboard_id = orchestrator.dashboards.create_dashboard(
            name="System Overview",
            widgets=[
                {"type": "metric", "metric_name": "cpu_usage_percent", "title": "CPU Usage"},
                {"type": "metric", "metric_name": "memory_usage_percent", "title": "Memory Usage"},
                {"type": "alert_list", "title": "Active Alerts"},
                {"type": "sli", "sli_name": "api-availability", "title": "API Availability"},
            ]
        )
        print(f"  Dashboard created: {dashboard_id}")
        
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(demo())
