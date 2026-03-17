"""
KAS Dashboard - Web 统计面板
Flask + Chart.js 实现
"""
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, render_template, jsonify
from kas.core.stats import StatsDashboard, AnalyticsDatabase

app = Flask(__name__)

# 配置
app.config['SECRET_KEY'] = os.getenv('DASHBOARD_SECRET_KEY', 'kas-dashboard-dev')


def get_db() -> AnalyticsDatabase:
    """获取数据库实例"""
    return AnalyticsDatabase()


def get_stats_data() -> Dict:
    """获取统计数据"""
    db = get_db()
    
    # 获取最近 7 天数据
    daily_stats_raw = db.get_daily_stats(days=7)
    
    # 转换 DailyStats 对象
    daily_stats = [
        {
            'date': s.date,
            'conversations': s.total_conversations,
            'messages': s.total_messages,
            'tokens': s.total_tokens,
            'quality_score': min(100, s.total_conversations * 10)  # 简单评分
        }
        for s in daily_stats_raw
    ]
    
    # 获取所有 Agent 列表（从对话记录中提取）
    cursor = db.conn.cursor()
    cursor.execute('SELECT DISTINCT agent_name FROM conversations')
    agents = [row[0] for row in cursor.fetchall()]
    
    # Agent 统计
    agent_stats = []
    for agent_name in agents:
        stats = db.get_agent_stats(agent_name, days=30)
        agent_stats.append({
            'name': agent_name,
            'conversations': stats['conversations']['total_conversations'],
            'messages': stats['conversations']['avg_messages'],
            'rating': stats['conversations']['avg_rating'],
            'tokens': stats['conversations']['total_tokens']
        })
    
    # 能力使用统计
    cursor.execute('''
        SELECT capability, COUNT(*) as count 
        FROM capability_usage 
        GROUP BY capability 
        ORDER BY count DESC 
        LIMIT 10
    ''')
    capability_stats = [
        {'name': row[0], 'usage_count': row[1]}
        for row in cursor.fetchall()
    ]
    
    return {
        'daily': daily_stats,
        'agents': agent_stats,
        'capabilities': capability_stats,
        'summary': {
            'total_conversations': sum(s.total_conversations for s in daily_stats_raw),
            'total_agents': len(agent_stats),
            'avg_quality': sum(d['quality_score'] for d in daily_stats) / len(daily_stats) if daily_stats else 0
        }
    }


@app.route('/')
def index():
    """主页面"""
    return render_template('dashboard.html')


@app.route('/api/stats')
def api_stats():
    """API: 获取统计数据"""
    try:
        data = get_stats_data()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/agents')
def api_agents():
    """API: 获取 Agent 列表"""
    try:
        db = get_db()
        cursor = db.conn.cursor()
        cursor.execute('SELECT DISTINCT agent_name FROM conversations')
        agents = [{'name': row[0]} for row in cursor.fetchall()]
        return jsonify({'success': True, 'agents': agents})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/conversations')
def api_conversations():
    """API: 获取最近对话"""
    try:
        db = get_db()
        cursor = db.conn.cursor()
        cursor.execute('''
            SELECT id, agent_name, timestamp, message_count, 
                   token_input + token_output as tokens
            FROM conversations
            ORDER BY timestamp DESC
            LIMIT 50
        ''')
        conversations = [
            {
                'id': row[0],
                'agent_name': row[1],
                'timestamp': row[2],
                'message_count': row[3],
                'tokens': row[4]
            }
            for row in cursor.fetchall()
        ]
        return jsonify({'success': True, 'conversations': conversations})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def run_dashboard(host='127.0.0.1', port=8080, debug=False):
    """启动仪表板"""
    print(f"🚀 KAS Dashboard starting at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_dashboard()
