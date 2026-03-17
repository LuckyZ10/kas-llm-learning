"""
KAS Core - Statistics & Analytics
统计面板 - 可视化 Agent 表现

功能:
- 对话统计
- 质量趋势
- 能力使用频率
- Token 消耗
- 图表生成
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict


# 统计数据库
STATS_DIR = Path.home() / '.kas' / 'stats'
STATS_DB = STATS_DIR / 'analytics.db'


@dataclass
class ConversationMetrics:
    """对话指标"""
    conversation_id: str
    agent_name: str
    timestamp: str
    message_count: int
    response_time_avg: float  # 平均响应时间(秒)
    token_input: int
    token_output: int
    user_rating: int  # 1-5 星评价


@dataclass
class DailyStats:
    """每日统计"""
    date: str
    total_conversations: int
    total_messages: int
    avg_response_time: float
    total_tokens: int
    unique_users: int


class AnalyticsDatabase:
    """分析数据库"""
    
    def __init__(self):
        STATS_DIR.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(STATS_DB))
        self._init_tables()
    
    def _init_tables(self):
        """初始化表结构"""
        cursor = self.conn.cursor()
        
        # 对话记录
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                message_count INTEGER DEFAULT 0,
                response_time_avg REAL DEFAULT 0,
                token_input INTEGER DEFAULT 0,
                token_output INTEGER DEFAULT 0,
                user_rating INTEGER DEFAULT 0
            )
        ''')
        
        # 能力使用记录
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS capability_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                capability TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                success BOOLEAN DEFAULT 1
            )
        ''')
        
        # 进化记录
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evolutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                version_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                quality_score REAL DEFAULT 0,
                generation INTEGER DEFAULT 0
            )
        ''')
        
        self.conn.commit()
    
    def record_conversation(self, metrics: ConversationMetrics):
        """记录对话"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO conversations 
            (id, agent_name, timestamp, message_count, response_time_avg,
             token_input, token_output, user_rating)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.conversation_id,
            metrics.agent_name,
            metrics.timestamp,
            metrics.message_count,
            metrics.response_time_avg,
            metrics.token_input,
            metrics.token_output,
            metrics.user_rating
        ))
        self.conn.commit()
    
    def record_capability_usage(self, agent_name: str, capability: str, success: bool = True):
        """记录能力使用"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO capability_usage (agent_name, capability, timestamp, success)
            VALUES (?, ?, ?, ?)
        ''', (agent_name, capability, datetime.now().isoformat(), success))
        self.conn.commit()
    
    def record_evolution(self, agent_name: str, version_id: str, quality_score: float, generation: int):
        """记录进化"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO evolutions (agent_name, version_id, timestamp, quality_score, generation)
            VALUES (?, ?, ?, ?, ?)
        ''', (agent_name, version_id, datetime.now().isoformat(), quality_score, generation))
        self.conn.commit()
    
    def get_agent_stats(self, agent_name: str, days: int = 30) -> Dict:
        """获取 Agent 统计"""
        cursor = self.conn.cursor()
        since = (datetime.now() - timedelta(days=days)).isoformat()
        
        # 对话统计
        cursor.execute('''
            SELECT COUNT(*), AVG(message_count), AVG(response_time_avg),
                   SUM(token_input), SUM(token_output), AVG(user_rating)
            FROM conversations
            WHERE agent_name = ? AND timestamp > ?
        ''', (agent_name, since))
        
        row = cursor.fetchone()
        conv_stats = {
            'total_conversations': row[0] or 0,
            'avg_messages': round(row[1] or 0, 1),
            'avg_response_time': round(row[2] or 0, 2),
            'total_tokens': (row[3] or 0) + (row[4] or 0),
            'avg_rating': round(row[5] or 0, 1)
        }
        
        # 能力使用统计
        cursor.execute('''
            SELECT capability, COUNT(*), SUM(CASE WHEN success THEN 1 ELSE 0 END)
            FROM capability_usage
            WHERE agent_name = ? AND timestamp > ?
            GROUP BY capability
        ''', (agent_name, since))
        
        capability_stats = {}
        for row in cursor.fetchall():
            capability_stats[row[0]] = {
                'count': row[1],
                'success_rate': round(row[2] / row[1] * 100, 1) if row[1] > 0 else 0
            }
        
        # 进化历史
        cursor.execute('''
            SELECT version_id, timestamp, quality_score, generation
            FROM evolutions
            WHERE agent_name = ?
            ORDER BY timestamp DESC
            LIMIT 10
        ''', (agent_name,))
        
        evolution_history = []
        for row in cursor.fetchall():
            evolution_history.append({
                'version_id': row[0],
                'timestamp': row[1],
                'quality_score': row[2],
                'generation': row[3]
            })
        
        return {
            'conversations': conv_stats,
            'capabilities': capability_stats,
            'evolution_history': evolution_history
        }
    
    def get_daily_stats(self, days: int = 7) -> List[DailyStats]:
        """获取每日统计"""
        cursor = self.conn.cursor()
        
        stats = []
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            start = f"{date}T00:00:00"
            end = f"{date}T23:59:59"
            
            cursor.execute('''
                SELECT COUNT(*), SUM(message_count), AVG(response_time_avg),
                       SUM(token_input + token_output), COUNT(DISTINCT agent_name)
                FROM conversations
                WHERE timestamp >= ? AND timestamp <= ?
            ''', (start, end))
            
            row = cursor.fetchone()
            stats.append(DailyStats(
                date=date,
                total_conversations=row[0] or 0,
                total_messages=row[1] or 0,
                avg_response_time=round(row[2] or 0, 2),
                total_tokens=row[3] or 0,
                unique_users=row[4] or 0
            ))
        
        return list(reversed(stats))


class StatsDashboard:
    """统计面板"""
    
    def __init__(self):
        self.db = AnalyticsDatabase()
    
    def show_agent_stats(self, agent_name: str, days: int = 30) -> str:
        """显示 Agent 统计"""
        stats = self.db.get_agent_stats(agent_name, days)
        conv = stats['conversations']
        caps = stats['capabilities']
        evo = stats['evolution_history']
        
        lines = [
            f"\n📊 {agent_name} 统计报告 (最近 {days} 天)",
            "=" * 60,
            "\n💬 对话统计:",
            f"  总对话数: {conv['total_conversations']}",
            f"  平均消息数: {conv['avg_messages']}",
            f"  平均响应时间: {conv['avg_response_time']}s",
            f"  总 Token 消耗: {conv['total_tokens']:,}",
            f"  平均评分: {conv['avg_rating']}/5.0",
        ]
        
        if caps:
            lines.append("\n💪 能力使用:")
            for cap, data in sorted(caps.items(), key=lambda x: x[1]['count'], reverse=True):
                bar = "█" * int(data['success_rate'] / 10)
                lines.append(f"  {cap:20s} {bar} {data['count']}次 ({data['success_rate']}%)")
        
        if evo:
            lines.append("\n🧬 进化历史:")
            for e in evo[:5]:
                score_bar = "⭐" * int(e['quality_score'] / 20)
                lines.append(f"  Gen{e['generation']:2d} {score_bar} {e['quality_score']:.0f}%")
        
        return "\n".join(lines)
    
    def show_overview(self) -> str:
        """显示总览"""
        daily = self.db.get_daily_stats(7)
        
        lines = [
            "\n📈 KAS 使用概览 (最近 7 天)",
            "=" * 60,
        ]
        
        total_conv = sum(d.total_conversations for d in daily)
        total_tokens = sum(d.total_tokens for d in daily)
        
        lines.extend([
            f"\n📊 总计:",
            f"  总对话数: {total_conv}",
            f"  总 Token: {total_tokens:,}",
            "\n📅 每日趋势:",
        ])
        
        for d in daily:
            bar = "█" * min(d.total_conversations, 20)
            lines.append(f"  {d.date} {bar} {d.total_conversations}对话")
        
        return "\n".join(lines)
    
    def generate_ascii_chart(self, values: List[float], labels: List[str], width: int = 40) -> str:
        """生成 ASCII 图表"""
        if not values:
            return "无数据"
        
        max_val = max(values) if max(values) > 0 else 1
        lines = []
        
        for label, value in zip(labels, values):
            bar_len = int((value / max_val) * width)
            bar = "█" * bar_len
            lines.append(f"{label:10s} {bar} {value:.1f}")
        
        return "\n".join(lines)


# 便捷函数
def get_dashboard() -> StatsDashboard:
    """获取统计面板实例"""
    return StatsDashboard()


def record_conversation(agent_name: str, message_count: int, 
                       response_time: float, tokens: int = 0):
    """记录对话（便捷函数）"""
    db = AnalyticsDatabase()
    metrics = ConversationMetrics(
        conversation_id=f"{agent_name}_{datetime.now().timestamp()}",
        agent_name=agent_name,
        timestamp=datetime.now().isoformat(),
        message_count=message_count,
        response_time_avg=response_time,
        token_input=tokens // 2,
        token_output=tokens // 2,
        user_rating=0
    )
    db.record_conversation(metrics)


if __name__ == "__main__":
    print("📊 测试统计面板")
    
    dashboard = StatsDashboard()
    
    # 模拟数据
    print("\n📝 添加模拟数据...")
    record_conversation("ChatPetAgent", 10, 1.5, 500)
    record_conversation("ChatPetAgent", 5, 2.0, 300)
    
    # 显示统计
    print(dashboard.show_agent_stats("ChatPetAgent", days=7))
    print(dashboard.show_overview())
