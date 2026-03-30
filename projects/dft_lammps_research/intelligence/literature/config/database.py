"""
数据库管理模块
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from ..config.models import Paper, Author, LiteratureReview, AlertSubscription, AlertNotification
from ..config.settings import DATABASE_CONFIG


class DatabaseManager:
    """SQLite数据库管理器"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DATABASE_CONFIG["path"]
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_db(self):
        """初始化数据库表"""
        with self._get_connection() as conn:
            # 论文表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors TEXT NOT NULL,
                    abstract TEXT,
                    publication_date TEXT,
                    journal TEXT,
                    doi TEXT,
                    arxiv_id TEXT,
                    pmid TEXT,
                    url TEXT,
                    pdf_url TEXT,
                    keywords TEXT,
                    full_text TEXT,
                    sections TEXT,
                    citation_count INTEGER DEFAULT 0,
                    reference_count INTEGER DEFAULT 0,
                    references_list TEXT,
                    categories TEXT,
                    topics TEXT,
                    methods TEXT,
                    software TEXT,
                    datasets TEXT,
                    source TEXT,
                    fetched_at TEXT,
                    updated_at TEXT,
                    sentiment_score REAL,
                    importance_score REAL
                )
            """)
            
            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_date ON papers(publication_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source)")
            
            # 综述表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    query TEXT,
                    created_at TEXT,
                    paper_ids TEXT,
                    topics TEXT,
                    trends TEXT,
                    methods TEXT,
                    gaps TEXT,
                    summary TEXT,
                    sections TEXT,
                    total_papers INTEGER,
                    date_range TEXT
                )
            """)
            
            # 订阅表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    keywords TEXT,
                    authors TEXT,
                    journals TEXT,
                    categories TEXT,
                    min_citations INTEGER,
                    created_at TEXT,
                    last_check TEXT,
                    is_active INTEGER DEFAULT 1,
                    notification_email TEXT,
                    webhook_url TEXT
                )
            """)
            
            # 通知表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY,
                    subscription_id TEXT,
                    type TEXT,
                    paper_ids TEXT,
                    created_at TEXT,
                    is_read INTEGER DEFAULT 0,
                    message TEXT
                )
            """)
    
    # 论文相关操作
    def save_paper(self, paper: Paper) -> bool:
        """保存论文"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO papers (
                        id, title, authors, abstract, publication_date, journal,
                        doi, arxiv_id, pmid, url, pdf_url, keywords, full_text,
                        sections, citation_count, reference_count, references_list,
                        categories, topics, methods, software, datasets, source,
                        fetched_at, updated_at, sentiment_score, importance_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    paper.id,
                    paper.title,
                    json.dumps([a.to_dict() for a in paper.authors]),
                    paper.abstract,
                    paper.publication_date.isoformat(),
                    paper.journal,
                    paper.doi,
                    paper.arxiv_id,
                    paper.pmid,
                    paper.url,
                    paper.pdf_url,
                    json.dumps(paper.keywords),
                    paper.full_text,
                    json.dumps(paper.sections),
                    paper.citation_count,
                    paper.reference_count,
                    json.dumps(paper.references),
                    json.dumps(paper.categories),
                    json.dumps(paper.topics),
                    json.dumps(paper.methods),
                    json.dumps(paper.software),
                    json.dumps(paper.datasets),
                    paper.source,
                    paper.fetched_at.isoformat(),
                    paper.updated_at.isoformat(),
                    paper.sentiment_score,
                    paper.importance_score
                ))
            return True
        except Exception as e:
            print(f"保存论文失败: {e}")
            return False
    
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """获取论文"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM papers WHERE id = ?", (paper_id,)
            ).fetchone()
            if row:
                return self._row_to_paper(row)
            return None
    
    def get_papers_by_ids(self, paper_ids: List[str]) -> List[Paper]:
        """批量获取论文"""
        if not paper_ids:
            return []
        
        placeholders = ','.join(['?' for _ in paper_ids])
        with self._get_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM papers WHERE id IN ({placeholders})",
                paper_ids
            ).fetchall()
            return [self._row_to_paper(row) for row in rows]
    
    def search_papers(
        self,
        query: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        authors: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Paper]:
        """搜索论文"""
        conditions = []
        params = []
        
        if query:
            conditions.append("(title LIKE ? OR abstract LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])
        
        if keywords:
            keyword_conditions = []
            for kw in keywords:
                keyword_conditions.append("keywords LIKE ?")
                params.append(f"%{kw}%")
            conditions.append(f"({' OR '.join(keyword_conditions)})")
        
        if date_from:
            conditions.append("publication_date >= ?")
            params.append(date_from.isoformat())
        
        if date_to:
            conditions.append("publication_date <= ?")
            params.append(date_to.isoformat())
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with self._get_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM papers WHERE {where_clause} ORDER BY publication_date DESC LIMIT ?",
                params + [limit]
            ).fetchall()
            return [self._row_to_paper(row) for row in rows]
    
    def _row_to_paper(self, row: sqlite3.Row) -> Paper:
        """将数据库行转换为Paper对象"""
        return Paper(
            id=row["id"],
            title=row["title"],
            authors=[Author(**a) for a in json.loads(row["authors"])],
            abstract=row["abstract"],
            publication_date=datetime.fromisoformat(row["publication_date"]),
            journal=row["journal"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            pmid=row["pmid"],
            url=row["url"],
            pdf_url=row["pdf_url"],
            keywords=json.loads(row["keywords"] or "[]"),
            full_text=row["full_text"],
            sections=json.loads(row["sections"] or "{}"),
            citation_count=row["citation_count"],
            reference_count=row["reference_count"],
            references=json.loads(row["references_list"] or "[]"),
            categories=json.loads(row["categories"] or "[]"),
            topics=json.loads(row["topics"] or "[]"),
            methods=json.loads(row["methods"] or "[]"),
            software=json.loads(row["software"] or "[]"),
            datasets=json.loads(row["datasets"] or "[]"),
            source=row["source"],
            fetched_at=datetime.fromisoformat(row["fetched_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            sentiment_score=row["sentiment_score"],
            importance_score=row["importance_score"]
        )
    
    def get_paper_count(self) -> int:
        """获取论文总数"""
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM papers").fetchone()
            return row["count"] if row else 0
    
    def get_papers_by_date_range(
        self,
        start: datetime,
        end: datetime
    ) -> List[Paper]:
        """获取日期范围内的论文"""
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM papers 
                   WHERE publication_date >= ? AND publication_date <= ?
                   ORDER BY publication_date DESC""",
                (start.isoformat(), end.isoformat())
            ).fetchall()
            return [self._row_to_paper(row) for row in rows]
    
    # 综述相关操作
    def save_review(self, review: LiteratureReview) -> bool:
        """保存综述"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO reviews (
                        id, title, query, created_at, paper_ids, topics,
                        trends, methods, gaps, summary, sections, total_papers, date_range
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    review.id,
                    review.title,
                    review.query,
                    review.created_at.isoformat(),
                    json.dumps([p.id for p in review.papers]),
                    json.dumps(review.topics),
                    json.dumps([t.to_dict() for t in review.trends]),
                    json.dumps([m.to_dict() for m in review.methods]),
                    json.dumps([g.to_dict() for g in review.gaps]),
                    review.summary,
                    json.dumps(review.sections),
                    len(review.papers),
                    json.dumps(review.date_range)
                ))
            return True
        except Exception as e:
            print(f"保存综述失败: {e}")
            return False
    
    def get_review(self, review_id: str) -> Optional[LiteratureReview]:
        """获取综述"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM reviews WHERE id = ?", (review_id,)
            ).fetchone()
            if row:
                return self._row_to_review(row)
            return None
    
    def _row_to_review(self, row: sqlite3.Row) -> LiteratureReview:
        """将数据库行转换为LiteratureReview对象"""
        paper_ids = json.loads(row["paper_ids"] or "[]")
        papers = self.get_papers_by_ids(paper_ids)
        
        return LiteratureReview(
            id=row["id"],
            title=row["title"],
            query=row["query"],
            created_at=datetime.fromisoformat(row["created_at"]),
            papers=papers,
            topics=json.loads(row["topics"] or "[]"),
            trends=[ResearchTrend(**t) for t in json.loads(row["trends"] or "[]")],
            methods=[MethodComparison(**m) for m in json.loads(row["methods"] or "[]")],
            gaps=[ResearchGap(**g) for g in json.loads(row["gaps"] or "[]")],
            summary=row["summary"],
            sections=json.loads(row["sections"] or "{}"),
            total_papers=row["total_papers"],
            date_range=tuple(json.loads(row["date_range"] or "[null, null]"))
        )
    
    # 订阅相关操作
    def save_subscription(self, subscription: AlertSubscription) -> bool:
        """保存订阅"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO subscriptions (
                        id, name, keywords, authors, journals, categories,
                        min_citations, created_at, last_check, is_active,
                        notification_email, webhook_url
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    subscription.id,
                    subscription.name,
                    json.dumps(subscription.keywords),
                    json.dumps(subscription.authors),
                    json.dumps(subscription.journals),
                    json.dumps(subscription.categories),
                    subscription.min_citations,
                    subscription.created_at.isoformat(),
                    subscription.last_check.isoformat() if subscription.last_check else None,
                    int(subscription.is_active),
                    subscription.notification_email,
                    subscription.webhook_url
                ))
            return True
        except Exception as e:
            print(f"保存订阅失败: {e}")
            return False
    
    def get_subscriptions(self, active_only: bool = True) -> List[AlertSubscription]:
        """获取所有订阅"""
        query = "SELECT * FROM subscriptions"
        if active_only:
            query += " WHERE is_active = 1"
        
        with self._get_connection() as conn:
            rows = conn.execute(query).fetchall()
            return [self._row_to_subscription(row) for row in rows]
    
    def _row_to_subscription(self, row: sqlite3.Row) -> AlertSubscription:
        """将数据库行转换为AlertSubscription对象"""
        return AlertSubscription(
            id=row["id"],
            name=row["name"],
            keywords=json.loads(row["keywords"] or "[]"),
            authors=json.loads(row["authors"] or "[]"),
            journals=json.loads(row["journals"] or "[]"),
            categories=json.loads(row["categories"] or "[]"),
            min_citations=row["min_citations"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_check=datetime.fromisoformat(row["last_check"]) if row["last_check"] else None,
            is_active=bool(row["is_active"]),
            notification_email=row["notification_email"],
            webhook_url=row["webhook_url"]
        )
    
    # 通知相关操作
    def save_notification(self, notification: AlertNotification) -> bool:
        """保存通知"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO notifications (
                        id, subscription_id, type, paper_ids, created_at, is_read, message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    notification.id,
                    notification.subscription_id,
                    notification.type,
                    json.dumps([p.id for p in notification.papers]),
                    notification.created_at.isoformat(),
                    int(notification.is_read),
                    notification.message
                ))
            return True
        except Exception as e:
            print(f"保存通知失败: {e}")
            return False
    
    def get_unread_notifications(self) -> List[AlertNotification]:
        """获取未读通知"""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM notifications WHERE is_read = 0 ORDER BY created_at DESC"
            ).fetchall()
            return [self._row_to_notification(row) for row in rows]
    
    def _row_to_notification(self, row: sqlite3.Row) -> AlertNotification:
        """将数据库行转换为AlertNotification对象"""
        paper_ids = json.loads(row["paper_ids"] or "[]")
        papers = self.get_papers_by_ids(paper_ids)
        
        return AlertNotification(
            id=row["id"],
            subscription_id=row["subscription_id"],
            type=row["type"],
            papers=papers,
            created_at=datetime.fromisoformat(row["created_at"]),
            is_read=bool(row["is_read"]),
            message=row["message"]
        )
    
    def mark_notification_read(self, notification_id: str) -> bool:
        """标记通知为已读"""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE notifications SET is_read = 1 WHERE id = ?",
                    (notification_id,)
                )
            return True
        except Exception as e:
            print(f"标记通知失败: {e}")
            return False
