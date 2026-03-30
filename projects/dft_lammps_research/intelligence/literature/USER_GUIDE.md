# 智能文献综述系统使用指南

## 系统概述

智能文献综述系统是一个自动化的研究辅助工具，能够：
- 从多个学术数据源抓取文献
- 自动分析研究主题和趋势
- 生成结构化综述报告
- 提供实时研究动态预警

## 快速开始

### 1. 安装依赖

```bash
cd /root/.openclaw/workspace/dft_lammps_research/literature_survey
pip install -r requirements.txt
```

### 2. 运行演示

```bash
cd /root/.openclaw/workspace/dft_lammps_research
python3 literature_survey/demo.py
```

### 3. 启动Web界面

```bash
cd /root/.openclaw/workspace/dft_lammps_research
python3 -m literature_survey web --port 5000
```

然后访问 http://localhost:5000

## 命令行使用

### 搜索文献

```bash
# 从arXiv搜索
python3 -m literature_survey search "density functional theory battery" \
    --sources arxiv --max-results 50

# 多源搜索
python3 -m literature_survey search "machine learning potential" \
    --sources arxiv crossref semanticscholar \
    --max-results 100
```

### 生成综述

```bash
# 基于查询生成综述
python3 -m literature_survey review \
    --query "solid electrolyte DFT" \
    --title "DFT Studies of Solid Electrolytes" \
    --output review.md --format markdown

# 基于特定论文生成
python3 -m literature_survey review \
    --paper-ids arxiv:2401.12345 arxiv:2401.12346 \
    --output review.html --format html
```

### 分析文献

```bash
# 主题分析
python3 -m literature_survey analyze \
    --query "battery materials" \
    --topics --output topics.txt

# 趋势分析
python3 -m literature_survey analyze \
    --query "lithium battery" \
    --trends --output trends.txt

# 知识图谱
python3 -m literature_survey analyze \
    --query "molecular dynamics" \
    --graph --output graph.graphml

# 完整分析
python3 -m literature_survey analyze \
    --query "DFT MD materials" \
    --topics --trends --graph \
    --output analysis_results
```

### 预警管理

```bash
# 创建订阅
python3 -m literature_survey alerts subscribe \
    --name "Battery Research" \
    --keywords "lithium battery" "solid electrolyte" "DFT" \
    --email researcher@example.com

# 列出所有订阅
python3 -m literature_survey alerts list

# 手动检查新论文
python3 -m literature_survey alerts check

# 生成周报
python3 -m literature_survey alerts digest
```

## Python API使用

### 基础用法

```python
import sys
sys.path.insert(0, '/root/.openclaw/workspace/dft_lammps_research')

from literature_survey import LiteratureFetcher, ReviewGenerator
from literature_survey.config.database import DatabaseManager

# 初始化组件
db = DatabaseManager()
fetcher = LiteratureFetcher(db)

# 搜索文献
papers = fetcher.search(
    query="density functional theory battery",
    sources=["arxiv", "semanticscholar"],
    max_results=50
)

print(f"找到 {len(papers)} 篇论文")
for paper in papers[:5]:
    print(f"- {paper.title}")
```

### 生成综述

```python
from literature_survey import ReviewGenerator

generator = ReviewGenerator()

# 生成综述
review = generator.generate_review(
    papers=papers,
    title="DFT Studies in Battery Research",
    query="density functional theory battery"
)

# 导出为Markdown
generator.export_markdown(review, "my_review.md")

# 导出为HTML
generator.export_html(review, "my_review.html")
```

### 主题建模

```python
from literature_survey.analysis.topic_modeling import TopicModeler

modeler = TopicModeler(method="simple", n_topics=5)
topics = modeler.fit(papers)

print("识别到的主题:")
for topic_id, topic_name in topics.items():
    print(f"  - {topic_name}")
```

### 趋势分析

```python
from literature_survey.analysis.trend_analysis import TrendAnalyzer

analyzer = TrendAnalyzer(window_size=2)
trends = analyzer.analyze(papers)

# 获取热门主题
hot_topics = analyzer.get_hot_topics(papers, recent_years=3)
print("热门主题:")
for topic, score in hot_topics:
    print(f"  - {topic}: {score:.1f}")

# 预测未来趋势
predictions = analyzer.predict_trends(trends, forecast_years=2)
```

### 知识图谱

```python
from literature_survey.analysis.knowledge_graph import KnowledgeGraphBuilder

builder = KnowledgeGraphBuilder(min_cooccurrence=3)
graph = builder.build_from_papers(papers)

print(f"节点数: {len(graph['nodes'])}")
print(f"边数: {len(graph['edges'])}")

# 导出为D3.js格式
d3_data = builder.to_d3()

# 导出为GraphML
builder.export_graphml("knowledge_graph.graphml")

# 查找相关概念
related = builder.get_related_concepts("DFT", top_n=10)
```

### 方法提取

```python
from literature_survey.analysis.method_extraction import MethodExtractor

extractor = MethodExtractor()

# 分析单篇论文
for paper in papers[:3]:
    info = extractor.extract_from_paper(paper)
    print(f"\n论文: {paper.title}")
    print(f"  方法: {info['methods']}")
    print(f"  软件: {info['software']}")
    print(f"  数据集: {info['datasets']}")

# 分析所有论文的方法使用情况
comparisons = extractor.analyze_methods(papers)
for comp in comparisons[:5]:
    print(f"{comp.method_name}: {comp.paper_count} 篇论文")
```

### 预警系统

```python
from literature_survey import AlertSystem

alert_system = AlertSystem()

# 创建订阅
subscription = alert_system.create_subscription(
    name="My Research Interest",
    keywords=["machine learning", "potential", "battery"],
    notification_email="user@example.com"
)

# 运行检查
notifications = alert_system.run_check()
for notification in notifications:
    print(f"[{notification.type}] {notification.message}")
    for paper in notification.papers:
        print(f"  - {paper.title}")

# 获取周报
digest = alert_system.generate_weekly_digest()
print(f"本周新论文: {digest['summary']['total_new_papers']}")
```

## Web界面功能

### 1. 文献搜索 (/search)
- 输入关键词搜索文献
- 选择数据源（arXiv、PubMed等）
- 设置最大结果数
- 查看搜索结果并选择论文

### 2. 智能分析 (/analysis)
- **主题建模**: 识别研究主题
- **趋势分析**: 分析研究热点演变
- **知识图谱**: 可视化概念关系
- **方法分析**: 提取方法信息

### 3. 综述生成 (/review)
- 选择要包含的论文
- 输入综述标题
- 生成结构化综述
- 导出为Markdown或HTML

### 4. 实时预警 (/alerts)
- 创建关键词订阅
- 查看通知列表
- 生成和查看周报
- 管理订阅

### 5. 知识图谱 (/knowledge_graph)
- 可视化概念网络
- 探索概念关系
- 发现研究热点

## 配置说明

### 修改配置

编辑 `literature_survey/config/settings.py`:

```python
# 修改搜索关键词
SEARCH_KEYWORDS = {
    "your_topic": ["keyword1", "keyword2"]
}

# 修改分析参数
ANALYSIS_CONFIG = {
    "topic_modeling": {
        "n_topics": 15,  # 增加主题数量
        "method": "lda"  # 使用LDA替代BERTopic
    }
}

# 修改预警设置
ALERT_CONFIG = {
    "check_interval": 3600,  # 每小时检查一次
    "max_new_papers": 100
}
```

### 环境变量

```bash
# PubMed API Key
export PUBMED_API_KEY="your_key_here"

# Semantic Scholar API Key
export SEMANTIC_SCHOLAR_API_KEY="your_key_here"
```

## 注意事项

1. **API限制**: 部分数据源有请求频率限制
2. **存储**: 论文数据存储在 `literature_survey/data/` 目录
3. **PDF下载**: 自动下载的PDF存储在 `literature_survey/data/papers/`
4. **网络**: 确保可以访问外部学术数据源

## 故障排除

### 导入错误
```bash
cd /root/.openclaw/workspace/dft_lammps_research
python3 -c "import literature_survey"
```

### 数据库问题
```bash
# 删除数据库重新初始化
rm literature_survey/data/literature.db
```

### 网络问题
```bash
# 测试数据源连接
python3 -c "
from literature_survey.fetcher.arxiv_fetcher import ArxivFetcher
f = ArxivFetcher()
papers = f.search('test', max_results=5)
print(f'Connected: {len(papers)} papers')
"
```

## 更多帮助

查看 `literature_survey/README.md` 获取完整文档。
