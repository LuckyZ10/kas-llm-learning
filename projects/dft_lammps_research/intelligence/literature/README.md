# 智能文献综述与研究趋势分析系统

自动跟踪研究前沿，提取关键方法，生成结构化综述的智能系统。

## 功能特点

### 1. 文献抓取与解析
- **多源抓取**: 支持 arXiv、PubMed、CrossRef、Semantic Scholar
- **PDF解析**: 全文提取、章节识别、元数据提取
- **元数据提取**: 标题、作者、摘要、关键词、引用

### 2. 智能分析
- **主题建模**: 使用BERTopic或LDA识别研究主题
- **趋势分析**: 时间序列分析研究热点演变
- **方法提取**: 自动识别方法、软件、数据集
- **知识图谱**: 构建概念关系网络

### 3. 综述生成
- **自动摘要**: 提取关键发现
- **对比分析**: 方法对比、性能对比
- **结构化报告**: 生成Markdown/HTML综述文档
- **研究空白识别**: 发现未被充分探索的方向

### 4. 实时预警
- **关键词订阅**: 自定义关注的主题
- **新论文推送**: 自动发现相关新论文
- **引用提醒**: 追踪论文引用情况
- **研究动态周报**: 定期汇总研究动态

## 安装

```bash
# 安装依赖
pip install -r literature_survey/requirements.txt

# 安装可选依赖（用于高级功能）
pip install bertopic sentence-transformers networkx
```

## 快速开始

### 命令行使用

```bash
# 搜索文献
python -m literature_survey search "density functional theory battery" --max-results 50

# 生成综述
python -m literature_survey review --query "solid electrolyte" --output review.md

# 启动Web界面
python -m literature_survey web --port 5000
```

### Python API

```python
from literature_survey import LiteratureFetcher, ReviewGenerator

# 搜索文献
fetcher = LiteratureFetcher()
papers = fetcher.search("DFT molecular dynamics", max_results=100)

# 生成综述
generator = ReviewGenerator()
review = generator.generate_review(papers, title="DFT-MD Review")
```

## Web界面

启动Web服务器后访问 `http://localhost:5000`:

- **首页**: 系统概览和功能导航
- **文献搜索**: 多源搜索和筛选
- **智能分析**: 主题建模、趋势分析、知识图谱
- **综述生成**: 生成和导出综述报告
- **实时预警**: 管理订阅和查看通知
- **知识图谱**: 可视化概念关系

## 项目结构

```
literature_survey/
├── config/              # 配置和模型
│   ├── settings.py      # 系统配置
│   ├── models.py        # 数据模型
│   └── database.py      # 数据库管理
├── fetcher/             # 文献抓取器
│   ├── arxiv_fetcher.py
│   ├── pubmed_fetcher.py
│   ├── crossref_fetcher.py
│   ├── semantic_scholar_fetcher.py
│   └── __init__.py
├── parser/              # PDF解析
│   └── pdf_parser.py
├── analysis/            # 分析引擎
│   ├── topic_modeling.py
│   ├── trend_analysis.py
│   ├── method_extraction.py
│   └── knowledge_graph.py
├── generator/           # 综述生成
│   └── review_generator.py
├── alert/               # 实时预警
│   └── alert_system.py
├── web/                 # Web界面
│   ├── app.py
│   ├── templates/
│   └── static/
├── data/                # 数据存储
│   ├── papers/          # PDF文件
│   ├── cache/           # 缓存
│   └── reports/         # 生成的报告
├── tests/               # 测试
├── __init__.py
├── __main__.py
└── requirements.txt
```

## 配置

编辑 `config/settings.py` 自定义：

- 数据源配置
- 搜索关键词
- 分析参数
- 报告格式

## 使用示例

### 搜索DFT电池材料相关文献

```bash
python -m literature_survey search "density functional theory battery materials" \
    --sources arxiv crossref \
    --max-results 100
```

### 分析最近一个月的研究趋势

```bash
python -m literature_survey analyze \
    --query "solid electrolyte" \
    --topics --trends --graph \
    --output analysis_results
```

### 创建订阅获取新论文通知

```bash
python -m literature_survey alerts subscribe \
    --name "Battery Research" \
    --keywords "lithium battery" "solid electrolyte" \
    --email researcher@example.com
```

## API参考

### LiteratureFetcher

```python
fetcher = LiteratureFetcher()

# 搜索
papers = fetcher.search(
    query="machine learning potential",
    sources=["arxiv", "semanticscholar"],
    max_results=100
)

# 获取最近论文
recent = fetcher.fetch_recent(days=7)
```

### ReviewGenerator

```python
generator = ReviewGenerator()

# 生成综述
review = generator.generate_review(
    papers=papers,
    title="My Literature Review"
)

# 导出
generator.export_markdown(review, "review.md")
generator.export_html(review, "review.html")
```

### AlertSystem

```python
alert_system = AlertSystem()

# 创建订阅
sub = alert_system.create_subscription(
    name="My Research",
    keywords=["DFT", "MD"]
)

# 检查新论文
notifications = alert_system.run_check()
```

## 许可证

MIT License
