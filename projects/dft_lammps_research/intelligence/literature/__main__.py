"""
命令行界面
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from literature_survey import (
    LiteratureFetcher, 
    TopicModeler, 
    TrendAnalyzer,
    KnowledgeGraphBuilder,
    ReviewGenerator,
    AlertSystem,
    DatabaseManager
)


def main():
    parser = argparse.ArgumentParser(
        description='智能文献综述与研究趋势分析系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 搜索文献
  python -m literature_survey search "density functional theory battery" --max-results 50
  
  # 生成综述
  python -m literature_survey review --query "solid electrolyte" --output review.md
  
  # 启动Web界面
  python -m literature_survey web --port 5000
  
  # 检查预警
  python -m literature_survey alerts check
  
  # 获取周报
  python -m literature_survey alerts digest
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 搜索命令
    search_parser = subparsers.add_parser('search', help='搜索文献')
    search_parser.add_argument('query', help='搜索查询')
    search_parser.add_argument('--sources', nargs='+', 
                              choices=['arxiv', 'pubmed', 'crossref', 'semanticscholar'],
                              default=['arxiv'],
                              help='数据源')
    search_parser.add_argument('--max-results', type=int, default=50, help='最大结果数')
    search_parser.add_argument('--days', type=int, help='限制在N天内')
    search_parser.add_argument('--save', action='store_true', default=True, help='保存到数据库')
    
    # 综述命令
    review_parser = subparsers.add_parser('review', help='生成综述')
    review_parser.add_argument('--query', help='搜索查询')
    review_parser.add_argument('--paper-ids', nargs='+', help='论文ID列表')
    review_parser.add_argument('--title', help='综述标题')
    review_parser.add_argument('--output', '-o', required=True, help='输出文件路径')
    review_parser.add_argument('--format', choices=['markdown', 'html'], default='markdown', 
                              help='输出格式')
    
    # 分析命令
    analysis_parser = subparsers.add_parser('analyze', help='分析文献')
    analysis_parser.add_argument('--query', help='搜索查询')
    analysis_parser.add_argument('--paper-ids', nargs='+', help='论文ID列表')
    analysis_parser.add_argument('--topics', action='store_true', help='主题分析')
    analysis_parser.add_argument('--trends', action='store_true', help='趋势分析')
    analysis_parser.add_argument('--graph', action='store_true', help='知识图谱')
    analysis_parser.add_argument('--output', '-o', help='输出文件路径')
    
    # 预警命令
    alert_parser = subparsers.add_parser('alerts', help='预警管理')
    alert_subparsers = alert_parser.add_subparsers(dest='alert_command', help='预警命令')
    
    # 创建订阅
    subscribe_parser = alert_subparsers.add_parser('subscribe', help='创建订阅')
    subscribe_parser.add_argument('--name', required=True, help='订阅名称')
    subscribe_parser.add_argument('--keywords', nargs='+', required=True, help='关键词')
    subscribe_parser.add_argument('--email', help='通知邮箱')
    
    # 检查预警
    alert_subparsers.add_parser('check', help='检查预警')
    
    # 周报
    alert_subparsers.add_parser('digest', help='生成周报')
    
    # 列出订阅
    alert_subparsers.add_parser('list', help='列出所有订阅')
    
    # Web界面命令
    web_parser = subparsers.add_parser('web', help='启动Web界面')
    web_parser.add_argument('--host', default='0.0.0.0', help='主机地址')
    web_parser.add_argument('--port', type=int, default=5000, help='端口号')
    web_parser.add_argument('--debug', action='store_true', help='调试模式')
    
    # 统计命令
    subparsers.add_parser('stats', help='显示统计信息')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 执行命令
    if args.command == 'search':
        cmd_search(args)
    elif args.command == 'review':
        cmd_review(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'alerts':
        cmd_alerts(args)
    elif args.command == 'web':
        cmd_web(args)
    elif args.command == 'stats':
        cmd_stats(args)


def cmd_search(args):
    """搜索文献"""
    print(f"正在搜索: {args.query}")
    print(f"数据源: {', '.join(args.sources)}")
    
    fetcher = LiteratureFetcher()
    
    date_from = None
    if args.days:
        date_from = datetime.now() - timedelta(days=args.days)
    
    papers = fetcher.search(
        query=args.query,
        sources=args.sources,
        max_results=args.max_results,
        date_from=date_from,
        save_to_db=args.save
    )
    
    print(f"\n找到 {len(papers)} 篇论文:\n")
    
    for i, paper in enumerate(papers[:20], 1):
        print(f"{i}. {paper.title}")
        print(f"   作者: {paper.get_author_names()[:80]}...")
        print(f"   日期: {paper.publication_date.strftime('%Y-%m-%d')}")
        print(f"   来源: {paper.source}")
        if paper.citation_count > 0:
            print(f"   引用: {paper.citation_count}")
        print()
    
    if len(papers) > 20:
        print(f"... 还有 {len(papers) - 20} 篇论文")


def cmd_review(args):
    """生成综述"""
    db = DatabaseManager()
    generator = ReviewGenerator(db)
    
    papers = []
    
    if args.paper_ids:
        papers = db.get_papers_by_ids(args.paper_ids)
    elif args.query:
        print(f"搜索文献: {args.query}")
        fetcher = LiteratureFetcher(db)
        papers = fetcher.search(args.query, max_results=100)
    else:
        print("错误: 请提供 --query 或 --paper-ids")
        return
    
    if not papers:
        print("未找到论文")
        return
    
    print(f"生成综述，基于 {len(papers)} 篇论文...")
    
    review = generator.generate_review(
        papers=papers,
        title=args.title,
        query=args.query or "Manual Review"
    )
    
    # 导出
    if args.format == 'markdown':
        generator.export_markdown(review, args.output)
    else:
        generator.export_html(review, args.output)
    
    print(f"\n综述已生成: {args.output}")
    print(f"标题: {review.title}")
    print(f"论文数: {len(review.papers)}")
    print(f"主题: {', '.join(review.topics[:5])}")


def cmd_analyze(args):
    """分析文献"""
    db = DatabaseManager()
    
    papers = []
    if args.paper_ids:
        papers = db.get_papers_by_ids(args.paper_ids)
    elif args.query:
        fetcher = LiteratureFetcher(db)
        papers = fetcher.search(args.query, max_results=100)
    else:
        print("错误: 请提供 --query 或 --paper-ids")
        return
    
    if not papers:
        print("未找到论文")
        return
    
    print(f"分析 {len(papers)} 篇论文...\n")
    
    results = []
    
    if args.topics or (not args.trends and not args.graph):
        print("=" * 50)
        print("主题分析")
        print("=" * 50)
        
        modeler = TopicModeler()
        topics = modeler.fit(papers)
        
        print(f"\n识别到 {len(topics)} 个主题:")
        for topic_id, topic_name in topics.items():
            print(f"  - {topic_name}")
        
        results.append(f"Topics: {len(topics)}")
    
    if args.trends or (not args.topics and not args.graph):
        print("\n" + "=" * 50)
        print("趋势分析")
        print("=" * 50)
        
        analyzer = TrendAnalyzer()
        trends = analyzer.analyze(papers)
        hot_topics = analyzer.get_hot_topics(papers)
        
        print(f"\n热门主题:")
        for topic, score in hot_topics[:5]:
            print(f"  - {topic}: {score:.1f}")
        
        results.append(f"Trends analyzed: {len(trends)}")
    
    if args.graph:
        print("\n" + "=" * 50)
        print("知识图谱")
        print("=" * 50)
        
        builder = KnowledgeGraphBuilder()
        graph = builder.build_from_papers(papers)
        
        print(f"\n图谱统计:")
        print(f"  节点数: {len(graph['nodes'])}")
        print(f"  边数: {len(graph['edges'])}")
        
        # 导出图谱
        if args.output:
            builder.export_graphml(args.output)
            print(f"\n图谱已导出: {args.output}")
        
        results.append(f"Graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
    
    print("\n" + "=" * 50)
    print("分析完成")
    print("=" * 50)
    print("\n".join(results))


def cmd_alerts(args):
    """预警管理"""
    db = DatabaseManager()
    alert_system = AlertSystem(db)
    
    if args.alert_command == 'subscribe':
        subscription = alert_system.create_subscription(
            name=args.name,
            keywords=args.keywords,
            notification_email=args.email
        )
        print(f"订阅创建成功: {subscription.id}")
        print(f"名称: {subscription.name}")
        print(f"关键词: {', '.join(subscription.keywords)}")
    
    elif args.alert_command == 'check':
        print("检查新论文...")
        notifications = alert_system.run_check()
        
        if notifications:
            print(f"\n发现 {len(notifications)} 条新通知:")
            for n in notifications:
                print(f"\n[{n.type}] {n.message}")
                for paper in n.papers[:5]:
                    print(f"  - {paper.title}")
        else:
            print("没有新通知")
    
    elif args.alert_command == 'digest':
        print("生成周报...")
        digest = alert_system.generate_weekly_digest()
        
        print(f"\n周报: {digest['period_start'].strftime('%Y-%m-%d')} 至 {digest['period_end'].strftime('%Y-%m-%d')}")
        print(f"\n新论文总数: {digest['summary']['total_new_papers']}")
        print(f"活跃订阅数: {digest['summary']['total_subscriptions']}")
        
        if 'hot_topics' in digest:
            print(f"\n热门主题:")
            for topic, count in digest['hot_topics'][:5]:
                print(f"  - {topic}: {count} 篇论文")
    
    elif args.alert_command == 'list':
        subscriptions = alert_system.db.get_subscriptions(active_only=False)
        
        print(f"共有 {len(subscriptions)} 个订阅:\n")
        for s in subscriptions:
            status = "✓ 活跃" if s.is_active else "✗ 停用"
            print(f"[{status}] {s.name}")
            print(f"  ID: {s.id}")
            print(f"  关键词: {', '.join(s.keywords)}")
            print(f"  创建时间: {s.created_at.strftime('%Y-%m-%d')}")
            if s.last_check:
                print(f"  最后检查: {s.last_check.strftime('%Y-%m-%d %H:%M')}")
            print()
    
    else:
        print("请指定子命令: subscribe, check, digest, list")


def cmd_web(args):
    """启动Web界面"""
    try:
        from literature_survey.web.app import create_app
        
        app = create_app()
        print(f"启动Web服务器...")
        print(f"访问地址: http://{args.host}:{args.port}")
        print("按 Ctrl+C 停止服务器")
        
        app.run(host=args.host, port=args.port, debug=args.debug)
    
    except ImportError as e:
        print(f"启动失败: {e}")
        print("请确保安装了Flask: pip install flask")


def cmd_stats(args):
    """显示统计信息"""
    db = DatabaseManager()
    
    print("=" * 50)
    print("文献综述系统统计")
    print("=" * 50)
    
    print(f"\n数据库:")
    print(f"  论文总数: {db.get_paper_count()}")
    
    # 获取最近论文
    recent_date = datetime.now() - timedelta(days=30)
    recent_papers = db.get_papers_by_date_range(recent_date, datetime.now())
    print(f"  近30天新增: {len(recent_papers)}")
    
    # 订阅统计
    alert_system = AlertSystem(db)
    subscriptions = alert_system.db.get_subscriptions(active_only=False)
    active_subs = [s for s in subscriptions if s.is_active]
    
    print(f"\n预警系统:")
    print(f"  订阅总数: {len(subscriptions)}")
    print(f"  活跃订阅: {len(active_subs)}")
    
    # 通知统计
    notifications = alert_system.get_notifications(unread_only=True)
    print(f"  未读通知: {len(notifications)}")


if __name__ == '__main__':
    main()
