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
from kas.core.stats import get_dashboard, AnalyticsDatabase
from kas.core.config import get_config

app = Flask(__name__)

# 配置
app.config['SECRET_KEY'] = os.getenv('DASHBOARD_SECRET_KEY', 'kas-dashboard-dev')


def get_stats_data() -> Dict:
    """获取统计数据"""
    dashboard = get_dashboard()
    
    # 获取最近 7 天数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    daily_stats = dashboard.get_daily_stats(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    # Agent 统计
    agent_stats = dashboard.get_agent_stats()
    
    # 能力使用统计
    capability_stats = dashboard.get_capability_stats()
    
    return {
        'daily': daily_stats,
        'agents': agent_stats,
        'capabilities': capability_stats,
        'summary': {
            'total_conversations': sum(d['conversations'] for d in daily_stats),
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
        dashboard = get_dashboard()
        agents = dashboard.get_agent_stats()
        return jsonify({'success': True, 'agents': agents})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/conversations')
def api_conversations():
    """API: 获取最近对话"""
    try:
        dashboard = get_dashboard()
        # 获取最近 50 条对话
        conversations = dashboard.get_recent_conversations(limit=50)
        return jsonify({'success': True, 'conversations': conversations})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def run_dashboard(host='127.0.0.1', port=8080, debug=False):
    """启动仪表板"""
    print(f"🚀 KAS Dashboard starting at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_dashboard()
