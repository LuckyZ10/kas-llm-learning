"""
trend_detector.py
趋势检测 - 新兴方法识别

分析论文数据, 识别研究热点和新兴方法趋势。

References:
- 2024进展: AI驱动的研究趋势预测
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json


@dataclass
class Trend:
    """趋势数据结构"""
    keyword: str
    category: str
    frequency: int
    growth_rate: float
    start_date: datetime
    momentum: float  # 趋势动量
    related_terms: List[str] = field(default_factory=list)
    papers: List[str] = field(default_factory=list)


@dataclass
class TrendReport:
    """趋势报告"""
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    emerging_trends: List[Trend]
    declining_trends: List[Trend]
    stable_topics: List[Trend]
    category_breakdown: Dict[str, List[Trend]]


class TrendDetector:
    """
    趋势检测器
    
    分析论文数据, 识别新兴研究趋势
    """
    
    def __init__(self):
        # 历史数据存储
        self.keyword_history: Dict[str, List[Tuple[datetime, int]]] = defaultdict(list)
        self.paper_metadata: List[Dict] = []
        
        # 类别映射
        self.category_map = {
            'method': ['neural network', 'transformer', 'graph neural network', 'diffusion model',
                      'active learning', 'Bayesian optimization', 'reinforcement learning',
                      'contrastive learning', 'self-supervised', 'federated learning'],
            'material': ['perovskite', 'battery', 'catalyst', 'semiconductor', 'superconductor',
                        '2D material', 'MOF', 'quantum dot', 'solid electrolyte'],
            'application': ['energy storage', 'solar cell', 'fuel cell', 'carbon capture',
                           'water splitting', 'CO2 reduction', 'drug delivery'],
            'software': ['VASP', 'Quantum ESPRESSO', 'PyTorch', 'TensorFlow', 'JAX',
                        'ASE', 'Pymatgen', 'DeepMD'],
            'technique': ['DFT', 'molecular dynamics', 'Monte Carlo', 'machine learning potential',
                         'high-throughput', 'active learning', 'transfer learning']
        }
    
    def add_papers(self, papers: List[Dict]):
        """添加论文数据"""
        for paper in papers:
            self.paper_metadata.append(paper)
            
            # 提取日期和关键词
            pub_date = datetime.fromisoformat(paper.get('published', datetime.now().isoformat()))
            keywords = paper.get('keywords_matched', [])
            
            for keyword in keywords:
                self.keyword_history[keyword].append((pub_date, 1))
    
    def detect_trends(
        self,
        window_months: int = 6,
        min_frequency: int = 5
    ) -> TrendReport:
        """
        检测趋势
        
        Args:
            window_months: 分析窗口 (月)
            min_frequency: 最小出现频率
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30 * window_months)
        
        # 按时间段统计关键词
        period_stats = self._compute_period_stats(start_date, end_date)
        
        # 计算增长率和动量
        trends = []
        for keyword, stats in period_stats.items():
            if stats['total'] < min_frequency:
                continue
            
            growth_rate = self._calculate_growth_rate(stats['timeline'])
            momentum = self._calculate_momentum(stats['timeline'])
            
            category = self._classify_keyword(keyword)
            
            trend = Trend(
                keyword=keyword,
                category=category,
                frequency=stats['total'],
                growth_rate=growth_rate,
                start_date=stats['first_seen'],
                momentum=momentum,
                related_terms=self._find_related_terms(keyword),
                papers=stats['papers']
            )
            trends.append(trend)
        
        # 分类趋势
        emerging = [t for t in trends if t.growth_rate > 0.3 and t.momentum > 0.5]
        declining = [t for t in trends if t.growth_rate < -0.2]
        stable = [t for t in trends if -0.1 <= t.growth_rate <= 0.1]
        
        # 按类别分组
        category_breakdown = defaultdict(list)
        for trend in trends:
            category_breakdown[trend.category].append(trend)
        
        # 排序
        emerging.sort(key=lambda x: x.growth_rate, reverse=True)
        declining.sort(key=lambda x: x.growth_rate)
        stable.sort(key=lambda x: x.frequency, reverse=True)
        
        return TrendReport(
            generated_at=datetime.now(),
            period_start=start_date,
            period_end=end_date,
            emerging_trends=emerging,
            declining_trends=declining,
            stable_topics=stable,
            category_breakdown=dict(category_breakdown)
        )
    
    def _compute_period_stats(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Dict]:
        """计算时间段统计"""
        stats = defaultdict(lambda: {
            'total': 0,
            'timeline': [],
            'first_seen': end_date,
            'papers': []
        })
        
        for paper in self.paper_metadata:
            pub_date = datetime.fromisoformat(paper.get('published', datetime.now().isoformat()))
            
            if not (start_date <= pub_date <= end_date):
                continue
            
            for keyword in paper.get('keywords_matched', []):
                stats[keyword]['total'] += 1
                stats[keyword]['timeline'].append((pub_date, 1))
                stats[keyword]['first_seen'] = min(stats[keyword]['first_seen'], pub_date)
                stats[keyword]['papers'].append(paper.get('id', 'unknown'))
        
        return stats
    
    def _calculate_growth_rate(self, timeline: List[Tuple[datetime, int]]) -> float:
        """计算增长率"""
        if len(timeline) < 2:
            return 0.0
        
        # 按月分组
        monthly_counts = defaultdict(int)
        for date, count in timeline:
            month_key = date.strftime('%Y-%m')
            monthly_counts[month_key] += count
        
        months = sorted(monthly_counts.keys())
        if len(months) < 2:
            return 0.0
        
        # 计算前半段和后半段的平均值
        mid = len(months) // 2
        early_avg = np.mean([monthly_counts[m] for m in months[:mid]])
        late_avg = np.mean([monthly_counts[m] for m in months[mid:]])
        
        if early_avg == 0:
            return 1.0 if late_avg > 0 else 0.0
        
        growth = (late_avg - early_avg) / early_avg
        return growth
    
    def _calculate_momentum(self, timeline: List[Tuple[datetime, int]]) -> float:
        """计算趋势动量 (近期加速程度)"""
        if len(timeline) < 3:
            return 0.0
        
        # 按周分组
        weekly_counts = defaultdict(int)
        for date, count in timeline:
            week_key = date.strftime('%Y-%W')
            weekly_counts[week_key] += count
        
        weeks = sorted(weekly_counts.keys())
        if len(weeks) < 3:
            return 0.0
        
        # 计算近期变化率
        recent = weekly_counts[weeks[-1]] + weekly_counts.get(weeks[-2], 0)
        older = sum(weekly_counts[w] for w in weeks[:-2]) / max(len(weeks) - 2, 1)
        
        if older == 0:
            return 0.5
        
        momentum = (recent / 2 - older) / (older + 1)
        return np.clip(momentum, -1, 1)
    
    def _classify_keyword(self, keyword: str) -> str:
        """分类关键词"""
        for category, terms in self.category_map.items():
            if keyword.lower() in [t.lower() for t in terms]:
                return category
        return 'other'
    
    def _find_related_terms(self, keyword: str, top_k: int = 5) -> List[str]:
        """查找相关术语 (共现分析)"""
        # 找到包含该关键词的论文
        co_occurrence = Counter()
        
        for paper in self.paper_metadata:
            keywords = paper.get('keywords_matched', [])
            if keyword in keywords:
                for kw in keywords:
                    if kw != keyword:
                        co_occurrence[kw] += 1
        
        return [term for term, _ in co_occurrence.most_common(top_k)]
    
    def predict_future_trends(
        self,
        trend: Trend,
        months_ahead: int = 6
    ) -> Dict:
        """预测未来趋势"""
        # 简单线性外推
        current_freq = trend.frequency
        growth = trend.growth_rate
        
        predicted_freq = current_freq * (1 + growth) ** months_ahead
        
        return {
            'keyword': trend.keyword,
            'current_frequency': current_freq,
            'predicted_frequency': int(predicted_freq),
            'confidence': min(trend.momentum + 0.5, 1.0),
            'prediction_basis': f"Growth rate: {growth:.2%}, Momentum: {trend.momentum:.2f}"
        }
    
    def generate_trend_report(self, report: TrendReport) -> str:
        """生成趋势报告"""
        output = f"""# Research Trend Report
Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M')}
Period: {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}

## Emerging Trends (Top 10)
| Rank | Keyword | Category | Growth Rate | Momentum | Frequency |
|------|---------|----------|-------------|----------|----------|
"""
        
        for i, trend in enumerate(report.emerging_trends[:10], 1):
            output += f"| {i} | {trend.keyword} | {trend.category} | "
            output += f"{trend.growth_rate:+.1%} | {trend.momentum:.2f} | {trend.frequency} |\n"
        
        output += f"\n## Hot Topics by Category\n"
        
        for category, trends in sorted(report.category_breakdown.items()):
            output += f"\n### {category.upper()}\n"
            for trend in sorted(trends, key=lambda x: x.frequency, reverse=True)[:5]:
                output += f"- **{trend.keyword}**: {trend.frequency} papers, "
                output += f"growth {trend.growth_rate:+.1%}\n"
        
        output += f"\n## Declining Topics\n"
        for trend in report.declining_trends[:5]:
            output += f"- {trend.keyword}: {trend.growth_rate:.1%}\n"
        
        return output
    
    def identify_hot_combinations(self, top_k: int = 10) -> List[Tuple[str, str, int]]:
        """识别热门组合 (两个关键词共现)"""
        combinations = Counter()
        
        for paper in self.paper_metadata:
            keywords = paper.get('keywords_matched', [])
            for i, kw1 in enumerate(keywords):
                for kw2 in keywords[i+1:]:
                    pair = tuple(sorted([kw1, kw2]))
                    combinations[pair] += 1
        
        return [(pair[0], pair[1], count) for pair, count in combinations.most_common(top_k)]


def demo():
    """演示"""
    print("=" * 60)
    print("Trend Detector Demo")
    print("=" * 60)
    
    detector = TrendDetector()
    
    # 模拟论文数据
    import random
    
    demo_papers = []
    base_date = datetime.now() - timedelta(days=180)
    
    keywords_pool = [
        'transformer', 'graph neural network', 'diffusion model', 'active learning',
        'perovskite', 'battery', 'catalyst', '2D material',
        'high-throughput', 'machine learning potential', 'Bayesian optimization'
    ]
    
    # 生成模拟论文
    for i in range(200):
        # 模拟趋势: 后期论文更多关注transformer和diffusion model
        date = base_date + timedelta(days=random.randint(0, 180))
        
        # 前期更多传统关键词
        if i < 100:
            selected = random.sample(keywords_pool[:6], k=random.randint(2, 4))
        else:
            # 后期更多新兴关键词
            selected = random.sample(keywords_pool[3:8], k=random.randint(2, 4))
        
        demo_papers.append({
            'id': f'paper_{i}',
            'published': date.isoformat(),
            'keywords_matched': selected
        })
    
    # 添加数据
    detector.add_papers(demo_papers)
    
    print(f"\nAdded {len(demo_papers)} papers")
    
    # 检测趋势
    print("\nDetecting trends...")
    report = detector.detect_trends(window_months=6, min_frequency=3)
    
    # 输出报告
    print("\n" + "=" * 60)
    print("Trend Report")
    print("=" * 60)
    
    print(f"\nPeriod: {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}")
    print(f"\nEmerging trends: {len(report.emerging_trends)}")
    print(f"Declining trends: {len(report.declining_trends)}")
    print(f"Stable topics: {len(report.stable_topics)}")
    
    print("\n--- Top Emerging Trends ---")
    for i, trend in enumerate(report.emerging_trends[:5], 1):
        print(f"\n{i}. {trend.keyword}")
        print(f"   Category: {trend.category}")
        print(f"   Growth rate: {trend.growth_rate:+.1%}")
        print(f"   Momentum: {trend.momentum:.2f}")
        print(f"   Frequency: {trend.frequency}")
        print(f"   Related: {', '.join(trend.related_terms[:3])}")
        
        # 预测
        prediction = detector.predict_future_trends(trend, months_ahead=3)
        print(f"   Prediction: {prediction['predicted_frequency']} papers in 3 months")
    
    # 热门组合
    print("\n--- Hot Keyword Combinations ---")
    hot_combos = detector.identify_hot_combinations(top_k=5)
    for kw1, kw2, count in hot_combos:
        print(f"  {kw1} + {kw2}: {count} papers")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
