#!/usr/bin/env python3
"""
Log Analysis Module
日志分析模块 - 提供分布式日志收集、分析和洞察

Features:
- Distributed log collection
- Real-time log streaming
- Pattern detection and anomaly identification
- Full-text search
- Log aggregation and metrics extraction
- Automated alerting from logs
- Compliance and audit logging

Author: DFTLammps Platform
Version: 2.0.0
"""

import asyncio
import json
import re
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any, Union, Set, Tuple
from enum import Enum, auto
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AUDIT = "AUDIT"  # 审计日志


class LogSource(Enum):
    """日志来源"""
    APPLICATION = "application"
    SYSTEM = "system"
    SECURITY = "security"
    ACCESS = "access"
    AUDIT = "audit"
    PERFORMANCE = "performance"


@dataclass
class LogEntry:
    """日志条目"""
    timestamp: datetime
    level: LogLevel
    message: str
    source: LogSource
    
    # 上下文信息
    service: str
    host: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    # 结构化数据
    fields: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "source": self.source.value,
            "service": self.service,
            "host": self.host,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "fields": self.fields,
            "tags": self.tags
        }
    
    @classmethod
    def from_string(
        cls,
        log_line: str,
        format_pattern: Optional[str] = None
    ) -> Optional['LogEntry']:
        """从字符串解析日志条目"""
        # 简化实现 - 实际使用解析器
        try:
            # 尝试JSON解析
            data = json.loads(log_line)
            return cls(
                timestamp=datetime.fromisoformat(data.get("timestamp", "").replace('Z', '+00:00')),
                level=LogLevel(data.get("level", "INFO")),
                message=data.get("message", ""),
                source=LogSource(data.get("source", "application")),
                service=data.get("service", "unknown"),
                host=data.get("host", "unknown"),
                **{k: v for k, v in data.items() if k not in 
                   ["timestamp", "level", "message", "source", "service", "host"]}
            )
        except:
            # 简单解析
            return cls(
                timestamp=datetime.utcnow(),
                level=LogLevel.INFO,
                message=log_line,
                source=LogSource.APPLICATION,
                service="unknown",
                host="unknown"
            )


@dataclass
class LogPattern:
    """日志模式"""
    pattern_id: str
    name: str
    regex: str
    description: str
    
    # 提取的字段
    fields: List[str] = field(default_factory=list)
    
    # 示例
    sample_logs: List[str] = field(default_factory=list)
    
    # 统计
    match_count: int = 0
    last_match: Optional[datetime] = None
    
    def matches(self, log_message: str) -> Optional[Dict]:
        """检查日志是否匹配模式"""
        match = re.search(self.regex, log_message)
        if match:
            self.match_count += 1
            self.last_match = datetime.utcnow()
            
            result = {"matched": True, "pattern_id": self.pattern_id}
            
            # 提取字段
            for i, field in enumerate(self.fields):
                if i < len(match.groups()):
                    result[field] = match.group(i + 1)
            
            result.update(match.groupdict())
            return result
        
        return None


@dataclass
class LogQuery:
    """日志查询"""
    query_id: str
    
    # 查询条件
    level: Optional[LogLevel] = None
    source: Optional[LogSource] = None
    service: Optional[str] = None
    host: Optional[str] = None
    trace_id: Optional[str] = None
    message_contains: Optional[str] = None
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    
    # 结构化查询
    field_filters: Dict[str, Any] = field(default_factory=dict)
    
    # 分页
    limit: int = 100
    offset: int = 0
    
    # 排序
    sort_by: str = "timestamp"
    sort_desc: bool = True


@dataclass
class LogAggregation:
    """日志聚合结果"""
    dimension: str
    buckets: List[Dict]
    total_count: int
    time_range: Tuple[datetime, datetime]


@dataclass
class LogAlertRule:
    """日志告警规则"""
    rule_id: str
    name: str
    description: str
    
    # 匹配条件
    level: Optional[LogLevel] = None
    source: Optional[LogSource] = None
    message_pattern: Optional[str] = None  # 正则表达式
    field_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # 阈值
    threshold_count: int = 1
    threshold_window_seconds: int = 60
    
    # 动作
    notification_channels: List[str] = field(default_factory=list)
    auto_create_ticket: bool = False
    
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


class LogParser:
    """日志解析器"""
    
    COMMON_PATTERNS = {
        "nginx_access": re.compile(
            r'(?P<ip>\S+) - - \[(?P<time>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) (?P<protocol>\S+)" '
            r'(?P<status>\d+) (?P<bytes>\d+) "(?P<referer>[^"]*)" "(?P<user_agent>[^"]*)"'
        ),
        "django": re.compile(
            r'\[(?P<time>[^\]]+)\] (?P<level>\w+): (?P<message>.*)'
        ),
        "syslog": re.compile(
            r'(?P<month>\w{3})\s+(?P<day>\d+)\s+(?P<time>\d{2}:\d{2}:\d{2})\s+'
            r'(?P<host>\S+)\s+(?P<process>\S+):\s*(?P<message>.*)'
        ),
    }
    
    def __init__(self):
        self.custom_patterns: Dict[str, re.Pattern] = {}
    
    def add_pattern(self, name: str, pattern: str):
        """添加自定义模式"""
        self.custom_patterns[name] = re.compile(pattern)
    
    def parse(self, log_line: str, parser_type: str = "auto") -> Optional[Dict]:
        """解析日志行"""
        
        # 尝试JSON解析
        try:
            return {"type": "json", "data": json.loads(log_line)}
        except json.JSONDecodeError:
            pass
        
        # 尝试结构化模式匹配
        if parser_type == "auto" or parser_type in self.COMMON_PATTERNS:
            patterns = [parser_type] if parser_type != "auto" else self.COMMON_PATTERNS.keys()
            
            for pattern_name in patterns:
                pattern = self.COMMON_PATTERNS.get(pattern_name)
                if pattern:
                    match = pattern.match(log_line)
                    if match:
                        return {"type": pattern_name, "data": match.groupdict()}
        
        # 尝试自定义模式
        for name, pattern in self.custom_patterns.items():
            match = pattern.match(log_line)
            if match:
                return {"type": name, "data": match.groupdict()}
        
        # 返回原始内容
        return {"type": "raw", "data": {"message": log_line}}


class LogStorage:
    """日志存储 (简化版，实际使用Elasticsearch/Loki等)"""
    
    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
        self.logs: deque = deque(maxlen=100000)  # 内存中保留最近10万条
        self.indexes: Dict[str, Dict] = defaultdict(lambda: defaultdict(set))
        self._lock = asyncio.Lock()
    
    async def store(self, entry: LogEntry):
        """存储日志条目"""
        async with self._lock:
            self.logs.append(entry)
            
            # 更新索引
            idx = len(self.logs) - 1
            self.indexes["level"][entry.level.value].add(idx)
            self.indexes["source"][entry.source.value].add(idx)
            self.indexes["service"][entry.service].add(idx)
            self.indexes["host"][entry.host].add(idx)
            
            if entry.trace_id:
                self.indexes["trace_id"][entry.trace_id].add(idx)
            
            for tag in entry.tags:
                self.indexes["tag"][tag].add(idx)
    
    async def store_batch(self, entries: List[LogEntry]):
        """批量存储"""
        for entry in entries:
            await self.store(entry)
    
    async def query(self, query: LogQuery) -> List[LogEntry]:
        """查询日志"""
        results = []
        
        # 使用索引快速过滤
        candidate_indices = None
        
        if query.level:
            candidate_indices = self.indexes["level"][query.level.value]
        
        if query.source:
            source_indices = self.indexes["source"][query.source.value]
            candidate_indices = candidate_indices & source_indices if candidate_indices else source_indices
        
        if query.service:
            service_indices = self.indexes["service"][query.service]
            candidate_indices = candidate_indices & service_indices if candidate_indices else service_indices
        
        if query.trace_id:
            trace_indices = self.indexes["trace_id"][query.trace_id]
            candidate_indices = candidate_indices & trace_indices if candidate_indices else trace_indices
        
        # 如果没有索引可用，扫描所有日志
        indices_to_scan = candidate_indices if candidate_indices else range(len(self.logs))
        
        for idx in indices_to_scan:
            if idx >= len(self.logs):
                continue
            
            entry = self.logs[idx]
            
            # 应用过滤条件
            if query.level and entry.level != query.level:
                continue
            
            if query.source and entry.source != query.source:
                continue
            
            if query.service and entry.service != query.service:
                continue
            
            if query.host and entry.host != query.host:
                continue
            
            if query.time_range_start and entry.timestamp < query.time_range_start:
                continue
            
            if query.time_range_end and entry.timestamp > query.time_range_end:
                continue
            
            if query.message_contains and query.message_contains not in entry.message:
                continue
            
            if query.field_filters:
                match = all(
                    entry.fields.get(k) == v
                    for k, v in query.field_filters.items()
                )
                if not match:
                    continue
            
            results.append(entry)
        
        # 排序
        reverse = query.sort_desc
        if query.sort_by == "timestamp":
            results.sort(key=lambda x: x.timestamp, reverse=reverse)
        
        # 分页
        start = query.offset
        end = query.offset + query.limit
        
        return results[start:end]
    
    async def aggregate(
        self,
        dimension: str,
        time_range_start: Optional[datetime] = None,
        time_range_end: Optional[datetime] = None
    ) -> LogAggregation:
        """按维度聚合"""
        buckets = defaultdict(int)
        total = 0
        
        for entry in self.logs:
            if time_range_start and entry.timestamp < time_range_start:
                continue
            if time_range_end and entry.timestamp > time_range_end:
                continue
            
            key = None
            if dimension == "level":
                key = entry.level.value
            elif dimension == "source":
                key = entry.source.value
            elif dimension == "service":
                key = entry.service
            elif dimension == "host":
                key = entry.host
            elif dimension == "hour":
                key = entry.timestamp.strftime("%Y-%m-%d %H:00")
            
            if key:
                buckets[key] += 1
                total += 1
        
        bucket_list = [
            {"key": k, "count": v}
            for k, v in sorted(buckets.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return LogAggregation(
            dimension=dimension,
            buckets=bucket_list,
            total_count=total,
            time_range=(
                time_range_start or datetime.utcnow() - timedelta(hours=1),
                time_range_end or datetime.utcnow()
            )
        )
    
    async def get_trace(self, trace_id: str) -> List[LogEntry]:
        """获取完整链路跟踪"""
        indices = self.indexes["trace_id"].get(trace_id, set())
        
        entries = []
        for idx in sorted(indices):
            if idx < len(self.logs):
                entries.append(self.logs[idx])
        
        entries.sort(key=lambda x: x.timestamp)
        return entries


class LogAnalyzer:
    """日志分析器"""
    
    def __init__(self, storage: LogStorage):
        self.storage = storage
        self.patterns: Dict[str, LogPattern] = {}
        self.anomaly_baselines: Dict[str, Dict] = {}
    
    def add_pattern(self, pattern: LogPattern):
        """添加日志模式"""
        self.patterns[pattern.pattern_id] = pattern
    
    async def analyze_patterns(self, time_window_seconds: int = 3600) -> Dict[str, int]:
        """分析日志模式匹配情况"""
        results = defaultdict(int)
        
        cutoff = datetime.utcnow() - timedelta(seconds=time_window_seconds)
        
        for entry in self.storage.logs:
            if entry.timestamp < cutoff:
                continue
            
            for pattern in self.patterns.values():
                if pattern.matches(entry.message):
                    results[pattern.name] += 1
        
        return dict(results)
    
    async def detect_anomalies(
        self,
        metric_name: str,
        time_window_seconds: int = 3600
    ) -> List[Dict]:
        """检测异常"""
        # 简化的异常检测：基于历史统计
        
        cutoff = datetime.utcnow() - timedelta(seconds=time_window_seconds)
        
        # 按小时统计
        hourly_counts = defaultdict(int)
        for entry in self.storage.logs:
            if entry.timestamp < cutoff:
                continue
            hour = entry.timestamp.strftime("%Y-%m-%d %H")
            hourly_counts[hour] += 1
        
        if len(hourly_counts) < 2:
            return []
        
        values = list(hourly_counts.values())
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = variance ** 0.5
        
        anomalies = []
        for hour, count in hourly_counts.items():
            z_score = abs(count - mean) / max(std, 0.1)
            if z_score > 2:  # 超过2个标准差
                anomalies.append({
                    "hour": hour,
                    "count": count,
                    "expected": round(mean, 2),
                    "z_score": round(z_score, 2),
                    "type": "high" if count > mean else "low"
                })
        
        return sorted(anomalies, key=lambda x: x["z_score"], reverse=True)
    
    async def extract_metrics(self, time_window_seconds: int = 300) -> Dict[str, List[Dict]]:
        """从日志中提取指标"""
        metrics = defaultdict(list)
        
        cutoff = datetime.utcnow() - timedelta(seconds=time_window_seconds)
        
        for entry in self.storage.logs:
            if entry.timestamp < cutoff:
                continue
            
            # 提取数字字段作为指标
            for field, value in entry.fields.items():
                if isinstance(value, (int, float)):
                    metrics[field].append({
                        "timestamp": entry.timestamp.isoformat(),
                        "value": value,
                        "service": entry.service
                    })
        
        return dict(metrics)
    
    async def error_analysis(self, time_window_seconds: int = 3600) -> Dict:
        """错误分析"""
        cutoff = datetime.utcnow() - timedelta(seconds=time_window_seconds)
        
        errors = []
        error_patterns = defaultdict(int)
        
        for entry in self.storage.logs:
            if entry.timestamp < cutoff:
                continue
            
            if entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                errors.append(entry)
                
                # 提取错误模式 (简化：取前50个字符)
                pattern = entry.message[:50]
                error_patterns[pattern] += 1
        
        # 统计每个服务的错误
        errors_by_service = defaultdict(int)
        for entry in errors:
            errors_by_service[entry.service] += 1
        
        return {
            "total_errors": len(errors),
            "error_rate": len(errors) / max(len(self.storage.logs), 1),
            "top_patterns": sorted(
                error_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "errors_by_service": dict(errors_by_service)
        }


class LogStreamHandler:
    """日志流处理器"""
    
    def __init__(self, storage: LogStorage):
        self.storage = storage
        self.subscribers: Dict[str, Callable] = {}
        self.filters: Dict[str, LogQuery] = {}
    
    def subscribe(
        self,
        subscription_id: str,
        query: LogQuery,
        callback: Callable[[LogEntry], Any]
    ):
        """订阅日志流"""
        self.subscribers[subscription_id] = callback
        self.filters[subscription_id] = query
        logger.info(f"Log stream subscription added: {subscription_id}")
    
    def unsubscribe(self, subscription_id: str):
        """取消订阅"""
        if subscription_id in self.subscribers:
            del self.subscribers[subscription_id]
            del self.filters[subscription_id]
    
    async def process_entry(self, entry: LogEntry):
        """处理新日志条目"""
        # 存储
        await self.storage.store(entry)
        
        # 推送给订阅者
        for sub_id, callback in self.subscribers.items():
            query = self.filters[sub_id]
            
            # 检查是否匹配
            if query.level and entry.level != query.level:
                continue
            if query.source and entry.source != query.source:
                continue
            if query.service and entry.service != query.service:
                continue
            if query.host and entry.host != query.host:
                continue
            
            try:
                await callback(entry)
            except Exception as e:
                logger.error(f"Error in log stream callback: {e}")


class LogAlertManager:
    """日志告警管理器"""
    
    def __init__(self, storage: LogStorage, analyzer: LogAnalyzer):
        self.storage = storage
        self.analyzer = analyzer
        self.rules: Dict[str, LogAlertRule] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self._eval_state: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0,
            "first_match": None,
            "matches": []
        })
    
    def add_rule(self, rule: LogAlertRule):
        """添加告警规则"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added log alert rule: {rule.name}")
    
    async def evaluate(self, entry: LogEntry):
        """评估日志条目是否触发告警"""
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # 检查匹配条件
            if rule.level and entry.level != rule.level:
                continue
            
            if rule.source and entry.source != rule.source:
                continue
            
            if rule.message_pattern:
                if not re.search(rule.message_pattern, entry.message):
                    continue
            
            if rule.field_conditions:
                if not all(
                    entry.fields.get(k) == v
                    for k, v in rule.field_conditions.items()
                ):
                    continue
            
            # 匹配成功
            state = self._eval_state[rule.rule_id]
            state["count"] += 1
            state["matches"].append(entry)
            
            if state["first_match"] is None:
                state["first_match"] = entry.timestamp
            
            # 检查是否超过阈值
            window_start = datetime.utcnow() - timedelta(
                seconds=rule.threshold_window_seconds
            )
            
            # 清理过期匹配
            state["matches"] = [
                e for e in state["matches"]
                if e.timestamp >= window_start
            ]
            state["count"] = len(state["matches"])
            
            if state["count"] >= rule.threshold_count:
                await self._trigger_alert(rule, state["matches"])
                # 重置状态
                state["count"] = 0
                state["first_match"] = None
                state["matches"] = []
    
    async def _trigger_alert(self, rule: LogAlertRule, entries: List[LogEntry]):
        """触发告警"""
        alert = {
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "description": rule.description,
            "triggered_at": datetime.utcnow().isoformat(),
            "match_count": len(entries),
            "sample_entries": [e.to_dict() for e in entries[:3]]
        }
        
        self.alert_history.append(alert)
        logger.warning(f"Log alert triggered: {rule.name} ({len(entries)} matches)")
        
        # 发送通知
        for channel in rule.notification_channels:
            logger.info(f"  Notifying {channel}")


class LogOrchestrator:
    """日志编排器"""
    
    def __init__(self):
        self.storage = LogStorage()
        self.parser = LogParser()
        self.analyzer = LogAnalyzer(self.storage)
        self.stream = LogStreamHandler(self.storage)
        self.alerts = LogAlertManager(self.storage, self.analyzer)
        
        # 统计
        self.stats = {
            "total_logs_processed": 0,
            "logs_per_second": 0,
            "last_calculation": datetime.utcnow()
        }
    
    async def ingest(self, log_line: str, **metadata) -> LogEntry:
        """摄取单条日志"""
        # 解析
        parsed = self.parser.parse(log_line)
        
        # 创建日志条目
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=metadata.get("level", LogLevel.INFO),
            message=parsed.get("data", {}).get("message", log_line),
            source=metadata.get("source", LogSource.APPLICATION),
            service=metadata.get("service", "unknown"),
            host=metadata.get("host", "unknown"),
            fields=parsed.get("data", {})
        )
        
        # 处理流
        await self.stream.process_entry(entry)
        
        # 评估告警
        await self.alerts.evaluate(entry)
        
        # 更新统计
        self.stats["total_logs_processed"] += 1
        
        return entry
    
    async def ingest_batch(self, log_lines: List[str], **metadata) -> List[LogEntry]:
        """批量摄取日志"""
        entries = []
        for line in log_lines:
            entry = await self.ingest(line, **metadata)
            entries.append(entry)
        return entries
    
    async def query(self, **kwargs) -> List[LogEntry]:
        """查询日志"""
        query = LogQuery(query_id="", **kwargs)
        return await self.storage.query(query)
    
    async def tail(self, count: int = 100) -> List[LogEntry]:
        """获取最新日志"""
        query = LogQuery(
            query_id="",
            limit=count,
            sort_by="timestamp",
            sort_desc=True
        )
        return await self.storage.query(query)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        now = datetime.utcnow()
        elapsed = (now - self.stats["last_calculation"]).total_seconds()
        
        if elapsed > 0:
            self.stats["logs_per_second"] = round(
                self.stats["total_logs_processed"] / elapsed, 2
            )
        
        return {
            **self.stats,
            "storage_size": len(self.storage.logs),
            "indexed_fields": len(self.storage.indexes)
        }


# 示例使用
async def demo():
    """日志分析演示"""
    
    orchestrator = LogOrchestrator()
    
    # 添加日志模式
    print("=== 配置日志模式 ===")
    patterns = [
        LogPattern(
            pattern_id="1",
            name="Database Error",
            regex=r"(?i)(database|db).*error",
            description="数据库错误",
            fields=["error_type"]
        ),
        LogPattern(
            pattern_id="2",
            name="API Request",
            regex=r"(\w+)\s+(\S+)\s+(\d+)ms",
            description="API请求",
            fields=["method", "path", "duration_ms"]
        ),
    ]
    
    for pattern in patterns:
        orchestrator.analyzer.add_pattern(pattern)
        print(f"  Added pattern: {pattern.name}")
    
    # 添加告警规则
    print("\n=== 配置日志告警规则 ===")
    alert_rules = [
        LogAlertRule(
            rule_id="1",
            name="High Error Rate",
            description="5分钟内错误日志超过10条",
            level=LogLevel.ERROR,
            threshold_count=10,
            threshold_window_seconds=300,
            notification_channels=["slack", "pagerduty"]
        ),
        LogAlertRule(
            rule_id="2",
            name="Database Connection Failure",
            description="数据库连接失败",
            message_pattern=r"(?i)connection.*failed",
            threshold_count=1,
            threshold_window_seconds=60,
            notification_channels=["pagerduty"]
        ),
    ]
    
    for rule in alert_rules:
        orchestrator.alerts.add_rule(rule)
        print(f"  Added alert rule: {rule.name}")
    
    # 模拟日志数据
    print("\n=== 模拟日志摄取 ===")
    
    services = ["api-gateway", "simulation-service", "molecule-predictor", "data-processor"]
    hosts = ["server-01", "server-02", "server-03"]
    
    # 生成正常日志
    for i in range(100):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": random.choice(["INFO", "DEBUG"]),
            "message": f"Processing job {i}: completed successfully",
            "service": random.choice(services),
            "host": random.choice(hosts),
            "fields": {"job_id": i, "duration_ms": random.randint(10, 500)}
        }
        await orchestrator.ingest(
            json.dumps(log_entry),
            level=LogLevel.INFO,
            service=log_entry["service"],
            host=log_entry["host"]
        )
    
    # 生成错误日志
    for i in range(15):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "ERROR",
            "message": random.choice([
                "Database connection failed: timeout",
                "API request failed: 500 Internal Server Error",
                "Simulation failed to converge",
                "Database query timeout after 30s"
            ]),
            "service": random.choice(services),
            "host": random.choice(hosts)
        }
        await orchestrator.ingest(
            json.dumps(log_entry),
            level=LogLevel.ERROR,
            service=log_entry["service"],
            host=log_entry["host"]
        )
    
    print(f"  Ingested {orchestrator.stats['total_logs_processed']} log entries")
    
    # 查询日志
    print("\n=== 日志查询 ===")
    
    # 查询所有ERROR级别日志
    error_logs = await orchestrator.query(level=LogLevel.ERROR, limit=10)
    print(f"  Found {len(error_logs)} ERROR logs")
    for log in error_logs[:3]:
        print(f"    [{log.timestamp.strftime('%H:%M:%S')}] {log.service}: {log.message[:60]}...")
    
    # 按服务查询
    service_logs = await orchestrator.query(service="simulation-service", limit=5)
    print(f"\n  Found {len(service_logs)} logs from simulation-service")
    
    # 日志聚合
    print("\n=== 日志聚合 ===")
    
    level_agg = await orchestrator.storage.aggregate("level")
    print("  By Level:")
    for bucket in level_agg.buckets:
        print(f"    {bucket['key']}: {bucket['count']}")
    
    service_agg = await orchestrator.storage.aggregate("service")
    print("\n  By Service:")
    for bucket in service_agg.buckets:
        print(f"    {bucket['key']}: {bucket['count']}")
    
    # 错误分析
    print("\n=== 错误分析 ===")
    error_analysis = await orchestrator.analyzer.error_analysis(time_window_seconds=3600)
    print(f"  Total errors: {error_analysis['total_errors']}")
    print(f"  Error rate: {error_analysis['error_rate']:.2%}")
    print("  Top error patterns:")
    for pattern, count in error_analysis['top_patterns'][:3]:
        print(f"    {count}x: {pattern[:50]}...")
    
    # 模式分析
    print("\n=== 日志模式匹配 ===")
    pattern_matches = await orchestrator.analyzer.analyze_patterns()
    for name, count in pattern_matches.items():
        print(f"  {name}: {count} matches")
    
    # 异常检测
    print("\n=== 异常检测 ===")
    # 添加更多样化的数据以支持异常检测
    for _ in range(50):
        await orchestrator.ingest(
            json.dumps({"level": "INFO", "message": "heartbeat", "timestamp": datetime.utcnow().isoformat()}),
            level=LogLevel.INFO,
            service="monitoring"
        )
    
    anomalies = await orchestrator.analyzer.detect_anomalies("count", 3600)
    if anomalies:
        for a in anomalies[:3]:
            print(f"  Anomaly at {a['hour']}: {a['count']} logs (expected {a['expected']})")
    else:
        print("  No significant anomalies detected")
    
    # 最新日志
    print("\n=== 最新日志 (tail -10) ===")
    recent_logs = await orchestrator.tail(10)
    for log in recent_logs:
        print(f"  [{log.timestamp.strftime('%H:%M:%S')}] [{log.level.value}] {log.service}: {log.message[:50]}...")
    
    # 统计
    print("\n=== 系统统计 ===")
    stats = orchestrator.get_stats()
    print(f"  Total processed: {stats['total_logs_processed']}")
    print(f"  Storage size: {stats['storage_size']}")
    print(f"  Indexed fields: {stats['indexed_fields']}")


if __name__ == "__main__":
    asyncio.run(demo())
