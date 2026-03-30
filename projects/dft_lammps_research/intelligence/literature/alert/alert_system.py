"""
实时预警系统
监控新论文、引用提醒、研究动态
"""

import uuid
import hashlib
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict

from ..config.models import Paper, AlertSubscription, AlertNotification
from ..config.database import DatabaseManager
from ..config.settings import ALERT_CONFIG
from ..fetcher import LiteratureFetcher


class AlertSystem:
    """预警系统"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()
        self.fetcher = LiteratureFetcher(self.db)
    
    def create_subscription(
        self,
        name: str,
        keywords: List[str],
        authors: Optional[List[str]] = None,
        journals: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        min_citations: int = 0,
        notification_email: Optional[str] = None
    ) -> AlertSubscription:
        """
        创建订阅
        
        Args:
            name: 订阅名称
            keywords: 关键词列表
            authors: 关注作者
            journals: 关注期刊
            categories: 关注分类
            min_citations: 最小引用数
            notification_email: 通知邮箱
        
        Returns:
            订阅对象
        """
        subscription = AlertSubscription(
            id=str(uuid.uuid4()),
            name=name,
            keywords=keywords,
            authors=authors or [],
            journals=journals or [],
            categories=categories or [],
            min_citations=min_citations,
            notification_email=notification_email,
            created_at=datetime.now()
        )
        
        self.db.save_subscription(subscription)
        print(f"订阅创建成功: {name} ({subscription.id})")
        
        return subscription
    
    def check_new_papers(
        self,
        subscription: AlertSubscription,
        days: int = 7
    ) -> List[Paper]:
        """
        检查新论文
        
        Args:
            subscription: 订阅
            days: 检查天数
        
        Returns:
            新论文列表
        """
        date_from = datetime.now() - timedelta(days=days)
        
        # 构建查询
        query_parts = []
        
        # 关键词查询
        if subscription.keywords:
            keyword_query = " OR ".join([f'"{kw}"' for kw in subscription.keywords])
            query_parts.append(f"({keyword_query})")
        
        # 作者查询
        if subscription.authors:
            author_query = " OR ".join([f'"{a}"' for a in subscription.authors])
            query_parts.append(f"({author_query})")
        
        if not query_parts:
            return []
        
        query = " AND ".join(query_parts)
        
        # 搜索新论文
        print(f"检查订阅 '{subscription.name}' 的新论文...")
        new_papers = self.fetcher.search(
            query=query,
            max_results=ALERT_CONFIG["max_new_papers"],
            date_from=date_from
        )
        
        # 过滤已存在的论文
        existing_ids = set()
        if subscription.last_check:
            # 获取上次检查后的新论文
            existing_papers = self.db.search_papers(
                date_from=subscription.last_check,
                limit=1000
            )
            existing_ids = {p.id for p in existing_papers}
        
        truly_new = [p for p in new_papers if p.id not in existing_ids]
        
        # 额外过滤
        filtered = self._filter_papers(truly_new, subscription)
        
        return filtered
    
    def _filter_papers(
        self,
        papers: List[Paper],
        subscription: AlertSubscription
    ) -> List[Paper]:
        """根据订阅条件过滤论文"""
        filtered = []
        
        for paper in papers:
            # 检查引用数
            if paper.citation_count < subscription.min_citations:
                continue
            
            # 检查期刊
            if subscription.journals and paper.journal:
                if not any(j.lower() in paper.journal.lower() 
                          for j in subscription.journals):
                    continue
            
            # 检查分类
            if subscription.categories and paper.categories:
                if not any(c in paper.categories for c in subscription.categories):
                    continue
            
            filtered.append(paper)
        
        return filtered
    
    def check_citation_alerts(
        self,
        paper_ids: List[str],
        threshold: int = 10
    ) -> List[Dict[str, Any]]:
        """
        检查引用提醒
        
        Args:
            paper_ids: 监控的论文ID
            threshold: 引用数阈值
        
        Returns:
            引用提醒列表
        """
        alerts = []
        
        for paper_id in paper_ids:
            paper = self.db.get_paper(paper_id)
            if not paper:
                continue
            
            # 获取最新引用信息
            citing_papers = self.fetcher.fetch_citations(paper)
            
            # 检查是否有新增的显著引用
            if citing_papers:
                new_citations = [
                    p for p in citing_papers
                    if p.citation_count >= threshold
                ]
                
                if new_citations:
                    alerts.append({
                        "paper": paper,
                        "new_citations": new_citations,
                        "total_new": len(citing_papers)
                    })
        
        return alerts
    
    def generate_weekly_digest(
        self,
        subscriptions: Optional[List[AlertSubscription]] = None
    ) -> Dict[str, Any]:
        """
        生成周报
        
        Args:
            subscriptions: 订阅列表，None则使用所有活跃订阅
        
        Returns:
            周报内容
        """
        if subscriptions is None:
            subscriptions = self.db.get_subscriptions(active_only=True)
        
        digest = {
            "period_start": datetime.now() - timedelta(days=7),
            "period_end": datetime.now(),
            "subscriptions": [],
            "summary": {
                "total_new_papers": 0,
                "total_subscriptions": len(subscriptions)
            }
        }
        
        all_new_papers = []
        
        for subscription in subscriptions:
            new_papers = self.check_new_papers(subscription, days=7)
            
            if new_papers:
                sub_digest = {
                    "subscription_name": subscription.name,
                    "new_papers_count": len(new_papers),
                    "top_papers": [
                        {
                            "id": p.id,
                            "title": p.title,
                            "authors": p.get_author_names(),
                            "date": p.publication_date.isoformat()
                        }
                        for p in sorted(new_papers, 
                                      key=lambda x: x.citation_count, 
                                      reverse=True)[:5]
                    ]
                }
                
                digest["subscriptions"].append(sub_digest)
                all_new_papers.extend(new_papers)
        
        # 去重
        seen_ids = set()
        unique_papers = []
        for p in all_new_papers:
            if p.id not in seen_ids:
                seen_ids.add(p.id)
                unique_papers.append(p)
        
        digest["summary"]["total_new_papers"] = len(unique_papers)
        
        # 热门主题
        if unique_papers:
            topics = defaultdict(int)
            for p in unique_papers:
                for t in p.topics:
                    topics[t] += 1
            
            digest["hot_topics"] = sorted(topics.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True)[:10]
        
        return digest
    
    def run_check(self) -> List[AlertNotification]:
        """
        运行检查并生成通知
        
        Returns:
            通知列表
        """
        subscriptions = self.db.get_subscriptions(active_only=True)
        notifications = []
        
        for subscription in subscriptions:
            # 检查新论文
            new_papers = self.check_new_papers(subscription)
            
            if new_papers:
                notification = AlertNotification(
                    id=str(uuid.uuid4()),
                    subscription_id=subscription.id,
                    type="new_paper",
                    papers=new_papers,
                    message=f"Found {len(new_papers)} new papers matching your subscription '{subscription.name}'"
                )
                
                self.db.save_notification(notification)
                notifications.append(notification)
            
            # 更新最后检查时间
            subscription.last_check = datetime.now()
            self.db.save_subscription(subscription)
        
        return notifications
    
    def get_notifications(
        self,
        unread_only: bool = True,
        subscription_id: Optional[str] = None
    ) -> List[AlertNotification]:
        """
        获取通知
        
        Args:
            unread_only: 仅未读
            subscription_id: 订阅ID过滤
        
        Returns:
            通知列表
        """
        if unread_only:
            return self.db.get_unread_notifications()
        
        # 获取所有通知
        # 这里简化处理，实际应该实现完整的查询
        return []
    
    def mark_as_read(self, notification_id: str) -> bool:
        """标记通知为已读"""
        return self.db.mark_notification_read(notification_id)
    
    def delete_subscription(self, subscription_id: str) -> bool:
        """删除订阅"""
        try:
            subscription = self.db.get_subscription(subscription_id)
            if subscription:
                subscription.is_active = False
                return self.db.save_subscription(subscription)
            return False
        except:
            return False
    
    def get_subscription_stats(self, subscription_id: str) -> Dict[str, Any]:
        """获取订阅统计"""
        subscription = None
        for sub in self.db.get_subscriptions(active_only=False):
            if sub.id == subscription_id:
                subscription = sub
                break
        
        if not subscription:
            return {}
        
        return {
            "id": subscription.id,
            "name": subscription.name,
            "keywords": subscription.keywords,
            "created_at": subscription.created_at.isoformat(),
            "last_check": subscription.last_check.isoformat() if subscription.last_check else None,
            "is_active": subscription.is_active
        }


class CitationTracker:
    """引用追踪器"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()
        self.fetcher = LiteratureFetcher(self.db)
    
    def track_paper(self, paper_id: str) -> Dict[str, Any]:
        """
        追踪论文引用
        
        Args:
            paper_id: 论文ID
        
        Returns:
            追踪结果
        """
        paper = self.db.get_paper(paper_id)
        if not paper:
            return {"error": "Paper not found"}
        
        # 获取引用
        citing_papers = self.fetcher.fetch_citations(paper)
        
        return {
            "paper_id": paper_id,
            "title": paper.title,
            "current_citations": paper.citation_count,
            "tracked_at": datetime.now().isoformat(),
            "citing_papers": [
                {
                    "id": p.id,
                    "title": p.title,
                    "authors": p.get_author_names(),
                    "date": p.publication_date.isoformat()
                }
                for p in citing_papers
            ]
        }
    
    def get_citation_history(self, paper_id: str) -> List[Dict]:
        """
        获取引用历史
        
        注意：这需要保存历史数据，这里简化实现
        """
        paper = self.db.get_paper(paper_id)
        if not paper:
            return []
        
        return [{
            "date": paper.fetched_at.isoformat(),
            "citation_count": paper.citation_count
        }]


class TrendMonitor:
    """趋势监控器"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()
    
    def monitor_topic(
        self,
        topic: str,
        period_months: int = 6
    ) -> Dict[str, Any]:
        """
        监控主题趋势
        
        Args:
            topic: 主题
            period_months: 监控周期
        
        Returns:
            监控结果
        """
        from_date = datetime.now() - timedelta(days=period_months * 30)
        
        # 搜索相关论文
        papers = self.db.search_papers(
            query=topic,
            date_from=from_date,
            limit=1000
        )
        
        # 按月统计
        monthly_stats = defaultdict(lambda: {"count": 0, "citations": 0})
        
        for paper in papers:
            month_key = paper.publication_date.strftime("%Y-%m")
            monthly_stats[month_key]["count"] += 1
            monthly_stats[month_key]["citations"] += paper.citation_count
        
        # 计算趋势
        months = sorted(monthly_stats.keys())
        if len(months) >= 2:
            first_month = monthly_stats[months[0]]["count"]
            last_month = monthly_stats[months[-1]]["count"]
            
            if first_month > 0:
                growth_rate = (last_month - first_month) / first_month
            else:
                growth_rate = 0
        else:
            growth_rate = 0
        
        return {
            "topic": topic,
            "period_months": period_months,
            "total_papers": len(papers),
            "monthly_stats": dict(monthly_stats),
            "trend": "increasing" if growth_rate > 0.1 else "decreasing" if growth_rate < -0.1 else "stable",
            "growth_rate": growth_rate
        }
