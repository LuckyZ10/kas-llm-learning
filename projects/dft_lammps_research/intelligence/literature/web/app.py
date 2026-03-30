"""
Flask Web界面
"""

import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path

# 添加父目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from literature_survey.fetcher import LiteratureFetcher
from literature_survey.analysis.topic_modeling import TopicModeler
from literature_survey.analysis.trend_analysis import TrendAnalyzer
from literature_survey.analysis.knowledge_graph import KnowledgeGraphBuilder
from literature_survey.generator.review_generator import ReviewGenerator
from literature_survey.alert.alert_system import AlertSystem, TrendMonitor
from literature_survey.config.database import DatabaseManager
from literature_survey.config.models import AlertSubscription
from literature_survey.config.settings import DATA_DIR, REPORTS_DIR


def create_app():
    """创建Flask应用"""
    app = Flask(__name__)
    app.secret_key = 'literature_survey_secret_key'
    
    # 初始化组件
    db = DatabaseManager()
    fetcher = LiteratureFetcher(db)
    review_gen = ReviewGenerator(db)
    alert_system = AlertSystem(db)
    trend_monitor = TrendMonitor(db)
    
    # ========== 页面路由 ==========
    
    @app.route('/')
    def index():
        """首页"""
        stats = {
            "total_papers": db.get_paper_count(),
            "sources": ["arXiv", "PubMed", "CrossRef", "Semantic Scholar"]
        }
        return render_template('index.html', stats=stats)
    
    @app.route('/search')
    def search_page():
        """搜索页面"""
        return render_template('search.html')
    
    @app.route('/analysis')
    def analysis_page():
        """分析页面"""
        return render_template('analysis.html')
    
    @app.route('/review')
    def review_page():
        """综述页面"""
        return render_template('review.html')
    
    @app.route('/alerts')
    def alerts_page():
        """预警页面"""
        subscriptions = alert_system.db.get_subscriptions(active_only=True)
        notifications = alert_system.get_notifications(unread_only=True)
        return render_template('alerts.html', 
                             subscriptions=subscriptions,
                             notifications=notifications)
    
    @app.route('/knowledge_graph')
    def knowledge_graph_page():
        """知识图谱页面"""
        return render_template('knowledge_graph.html')
    
    # ========== API路由 ==========
    
    @app.route('/api/search', methods=['POST'])
    def api_search():
        """API: 搜索论文"""
        data = request.json
        query = data.get('query', '')
        sources = data.get('sources', ['arxiv'])
        max_results = data.get('max_results', 50)
        
        try:
            papers = fetcher.search(
                query=query,
                sources=sources,
                max_results=max_results,
                save_to_db=True
            )
            
            return jsonify({
                'success': True,
                'count': len(papers),
                'papers': [p.to_dict() for p in papers]
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/papers', methods=['GET'])
    def api_get_papers():
        """API: 获取论文列表"""
        limit = request.args.get('limit', 100, type=int)
        query = request.args.get('query', '')
        
        try:
            if query:
                papers = db.search_papers(query=query, limit=limit)
            else:
                # 获取最近的论文
                from datetime import timedelta
                date_from = datetime.now() - timedelta(days=365)
                papers = db.get_papers_by_date_range(date_from, datetime.now())
                papers = papers[:limit]
            
            return jsonify({
                'success': True,
                'count': len(papers),
                'papers': [p.to_dict() for p in papers]
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/paper/<paper_id>')
    def api_get_paper(paper_id):
        """API: 获取单篇论文"""
        paper = db.get_paper(paper_id)
        
        if paper:
            return jsonify({'success': True, 'paper': paper.to_dict()})
        else:
            return jsonify({'success': False, 'error': 'Paper not found'}), 404
    
    @app.route('/api/analyze/topics', methods=['POST'])
    def api_analyze_topics():
        """API: 主题分析"""
        data = request.json
        paper_ids = data.get('paper_ids', [])
        n_topics = data.get('n_topics', 10)
        
        try:
            papers = db.get_papers_by_ids(paper_ids)
            
            if not papers:
                return jsonify({'success': False, 'error': 'No papers found'}), 404
            
            modeler = TopicModeler(n_topics=n_topics)
            topics = modeler.fit(papers)
            
            return jsonify({
                'success': True,
                'topics': topics,
                'topic_distribution': modeler.get_topic_distribution()
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/analyze/trends', methods=['POST'])
    def api_analyze_trends():
        """API: 趋势分析"""
        data = request.json
        paper_ids = data.get('paper_ids', [])
        
        try:
            papers = db.get_papers_by_ids(paper_ids)
            
            if not papers:
                return jsonify({'success': False, 'error': 'No papers found'}), 404
            
            analyzer = TrendAnalyzer()
            trends = analyzer.analyze(papers)
            hot_topics = analyzer.get_hot_topics(papers)
            
            return jsonify({
                'success': True,
                'trends': [t.to_dict() for t in trends],
                'hot_topics': hot_topics
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/analyze/knowledge_graph', methods=['POST'])
    def api_knowledge_graph():
        """API: 知识图谱"""
        data = request.json
        paper_ids = data.get('paper_ids', [])
        min_cooccurrence = data.get('min_cooccurrence', 3)
        
        try:
            papers = db.get_papers_by_ids(paper_ids)
            
            if not papers:
                return jsonify({'success': False, 'error': 'No papers found'}), 404
            
            builder = KnowledgeGraphBuilder(min_cooccurrence=min_cooccurrence)
            graph = builder.build_from_papers(papers)
            
            return jsonify({
                'success': True,
                'graph': graph,
                'd3_format': builder.to_d3(),
                'cytoscape_format': builder.to_cytoscape()
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/generate_review', methods=['POST'])
    def api_generate_review():
        """API: 生成综述"""
        data = request.json
        paper_ids = data.get('paper_ids', [])
        title = data.get('title')
        
        try:
            papers = db.get_papers_by_ids(paper_ids)
            
            if not papers:
                return jsonify({'success': False, 'error': 'No papers found'}), 404
            
            review = review_gen.generate_review(papers, title=title)
            
            return jsonify({
                'success': True,
                'review': review.to_dict()
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/export_review/<review_id>', methods=['GET'])
    def api_export_review(review_id):
        """API: 导出综述"""
        format_type = request.args.get('format', 'markdown')
        
        try:
            review = db.get_review(review_id)
            
            if not review:
                return jsonify({'success': False, 'error': 'Review not found'}), 404
            
            # 确保报告目录存在
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            
            if format_type == 'markdown':
                output_path = REPORTS_DIR / f"review_{review_id}.md"
                review_gen.export_markdown(review, str(output_path))
            elif format_type == 'html':
                output_path = REPORTS_DIR / f"review_{review_id}.html"
                review_gen.export_html(review, str(output_path))
            else:
                return jsonify({'success': False, 'error': 'Unsupported format'}), 400
            
            return send_file(output_path, as_attachment=True)
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/alerts/subscriptions', methods=['GET'])
    def api_get_subscriptions():
        """API: 获取订阅列表"""
        subscriptions = alert_system.db.get_subscriptions(active_only=True)
        
        return jsonify({
            'success': True,
            'subscriptions': [s.to_dict() for s in subscriptions]
        })
    
    @app.route('/api/alerts/subscriptions', methods=['POST'])
    def api_create_subscription():
        """API: 创建订阅"""
        data = request.json
        
        try:
            subscription = alert_system.create_subscription(
                name=data['name'],
                keywords=data['keywords'],
                authors=data.get('authors', []),
                journals=data.get('journals', []),
                min_citations=data.get('min_citations', 0),
                notification_email=data.get('notification_email')
            )
            
            return jsonify({
                'success': True,
                'subscription': subscription.to_dict()
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/alerts/check', methods=['POST'])
    def api_check_alerts():
        """API: 手动检查预警"""
        try:
            notifications = alert_system.run_check()
            
            return jsonify({
                'success': True,
                'new_notifications': len(notifications),
                'notifications': [n.to_dict() for n in notifications]
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/alerts/weekly_digest', methods=['GET'])
    def api_weekly_digest():
        """API: 获取周报"""
        try:
            digest = alert_system.generate_weekly_digest()
            
            return jsonify({
                'success': True,
                'digest': digest
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/stats')
    def api_stats():
        """API: 获取统计信息"""
        try:
            total_papers = db.get_paper_count()
            
            # 获取近期论文
            from datetime import timedelta
            recent_date = datetime.now() - timedelta(days=30)
            recent_papers = db.get_papers_by_date_range(recent_date, datetime.now())
            
            return jsonify({
                'success': True,
                'stats': {
                    'total_papers': total_papers,
                    'recent_papers': len(recent_papers),
                    'sources': ['arXiv', 'PubMed', 'CrossRef', 'Semantic Scholar']
                }
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/monitor/topic', methods=['POST'])
    def api_monitor_topic():
        """API: 监控主题趋势"""
        data = request.json
        topic = data.get('topic', '')
        period = data.get('period_months', 6)
        
        try:
            result = trend_monitor.monitor_topic(topic, period)
            
            return jsonify({
                'success': True,
                'monitoring': result
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
