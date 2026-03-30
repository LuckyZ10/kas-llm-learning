#!/usr/bin/env python3
"""
智能文献综述系统演示脚本
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from literature_survey.config.models import Paper, Author, LiteratureReview
from literature_survey.config.database import DatabaseManager
from literature_survey.analysis.topic_modeling import TopicModeler, KeywordExtractor
from literature_survey.analysis.trend_analysis import TrendAnalyzer
from literature_survey.analysis.knowledge_graph import KnowledgeGraphBuilder
from literature_survey.analysis.method_extraction import MethodExtractor
from datetime import datetime


def demo_models():
    """演示数据模型"""
    print("=" * 60)
    print("演示1: 数据模型")
    print("=" * 60)
    
    # 创建作者
    authors = [
        Author(name="张三", affiliation="清华大学", email="zhang@tsinghua.edu.cn"),
        Author(name="李四", affiliation="北京大学", orcid="0000-0001-2345-6789")
    ]
    
    # 创建论文
    paper = Paper(
        id="arxiv:2401.12345",
        title="Machine Learning Potentials for Battery Materials: A Review",
        authors=authors,
        abstract="""This review paper discusses the application of machine learning 
        potentials in battery materials research. We cover various methodologies 
        including neural network potentials, Gaussian approximation potentials, 
        and moment tensor potentials. The review focuses on solid electrolyte 
        materials and lithium-ion battery cathodes.""",
        publication_date=datetime(2024, 1, 15),
        journal="npj Computational Materials",
        doi="10.1038/s41524-024-01234-5",
        arxiv_id="2401.12345",
        url="https://arxiv.org/abs/2401.12345",
        keywords=["machine learning", "battery", "DFT", "molecular dynamics"],
        categories=["cond-mat.mtrl-sci"],
        methods=["DFT", "Machine Learning", "Molecular Dynamics"],
        software=["VASP", "LAMMPS", "ASE"],
        citation_count=25,
        source="arxiv"
    )
    
    print(f"✓ 论文创建成功")
    print(f"  标题: {paper.title}")
    print(f"  作者: {paper.get_author_names()}")
    print(f"  日期: {paper.publication_date.strftime('%Y-%m-%d')}")
    print(f"  引用: {paper.citation_count}")
    print()
    
    return paper


def demo_database():
    """演示数据库功能"""
    print("=" * 60)
    print("演示2: 数据库操作")
    print("=" * 60)
    
    db = DatabaseManager()
    
    # 创建示例论文
    papers = []
    for i in range(5):
        paper = Paper(
            id=f"demo:{i}",
            title=f"Demo Paper {i+1}: DFT Study of Material Properties",
            authors=[Author(name=f"Author {i+1}")],
            abstract=f"This is a demo abstract for paper {i+1}. " * 5,
            publication_date=datetime(2023 + i, 1, 1),
            citation_count=i * 10,
            source="demo"
        )
        papers.append(paper)
        db.save_paper(paper)
    
    print(f"✓ 保存了 {len(papers)} 篇论文到数据库")
    print(f"  当前论文总数: {db.get_paper_count()}")
    
    # 搜索论文
    results = db.search_papers(query="DFT", limit=10)
    print(f"  搜索 'DFT': 找到 {len(results)} 篇论文")
    
    print()
    return papers


def demo_analysis(papers):
    """演示分析功能"""
    print("=" * 60)
    print("演示3: 智能分析")
    print("=" * 60)
    
    # 为论文添加更多数据用于分析
    for i, paper in enumerate(papers):
        paper.keywords = ["DFT", "battery", "machine learning", "MD"][i % 4:4]
        paper.topics = []
        paper.full_text = f"Full text content for paper {i+1}. " * 20
    
    # 关键词提取
    print("\n1. 关键词提取")
    extractor = KeywordExtractor(top_n=10)
    keywords = extractor.extract(papers)
    print(f"   提取的关键词:")
    for kw, count in keywords[:5]:
        print(f"   - {kw}: {count}")
    
    # 主题建模（简单模式）
    print("\n2. 主题建模")
    modeler = TopicModeler(method="simple", n_topics=2)
    topics = modeler.fit(papers)
    print(f"   识别到的主题:")
    for topic_id, topic_name in topics.items():
        print(f"   - {topic_name}")
    
    # 趋势分析
    print("\n3. 趋势分析")
    analyzer = TrendAnalyzer(window_size=1)
    trends = analyzer.analyze(papers)
    print(f"   分析了 {len(trends)} 个趋势点")
    
    # 方法提取
    print("\n4. 方法提取")
    method_extractor = MethodExtractor()
    for paper in papers[:2]:
        methods = method_extractor.extract_from_paper(paper)
        print(f"   论文 '{paper.title[:30]}...'")
        print(f"   - 方法: {', '.join(methods.get('methods', [])[:3])}")
        print(f"   - 软件: {', '.join(methods.get('software', [])[:3])}")
    
    print()


def demo_knowledge_graph(papers):
    """演示知识图谱"""
    print("=" * 60)
    print("演示4: 知识图谱")
    print("=" * 60)
    
    # 添加更多元数据用于图谱构建
    for i, paper in enumerate(papers):
        paper.keywords = ["DFT", "battery", "electrolyte", "Li", "MD"]
        paper.topics = ["Energy Storage", "Computational Materials"]
        paper.methods = ["DFT", "MD"]
        paper.software = ["VASP", "LAMMPS"]
    
    builder = KnowledgeGraphBuilder(min_cooccurrence=1)
    graph = builder.build_from_papers(papers)
    
    print(f"✓ 知识图谱构建完成")
    print(f"  节点数: {len(graph['nodes'])}")
    print(f"  边数: {len(graph['edges'])}")
    
    if graph['nodes']:
        print(f"\n  节点示例:")
        for node in graph['nodes'][:5]:
            print(f"  - {node['id']} ({node['type']}): 出现 {node['frequency']} 次")
    
    if graph['edges']:
        print(f"\n  边示例:")
        for edge in graph['edges'][:5]:
            print(f"  - {edge['source']} -- {edge['target']}: {edge['weight']}")
    
    # 查找相关概念
    if graph['nodes']:
        main_node = graph['nodes'][0]['id']
        related = builder.get_related_concepts(main_node, top_n=5)
        if related:
            print(f"\n  与 '{main_node}' 相关的概念:")
            for concept, weight in related:
                print(f"  - {concept}: {weight}")
    
    print()


def demo_review_generation():
    """演示综述生成"""
    print("=" * 60)
    print("演示5: 综述生成")
    print("=" * 60)
    
    # 创建示例综述
    review = LiteratureReview(
        id="review:demo",
        title="Demo Literature Review: DFT in Battery Research",
        query="density functional theory battery",
        created_at=datetime.now(),
        papers=[],
        topics=["Energy Storage", "Computational Methods"],
        summary="This is a demo review summarizing recent advances..."
    )
    
    print(f"✓ 综述对象创建成功")
    print(f"  标题: {review.title}")
    print(f"  查询: {review.query}")
    print(f"  主题: {', '.join(review.topics)}")
    print(f"  创建时间: {review.created_at.strftime('%Y-%m-%d %H:%M')}")
    print()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("智能文献综述系统演示")
    print("=" * 60 + "\n")
    
    # 运行各个演示
    try:
        paper = demo_models()
        papers = demo_database()
        demo_analysis(papers)
        demo_knowledge_graph(papers)
        demo_review_generation()
        
        print("=" * 60)
        print("演示完成！")
        print("=" * 60)
        print("\n系统组件全部正常工作。")
        print("使用 'python -m literature_survey web' 启动Web界面。")
        
    except Exception as e:
        print(f"\n✗ 演示出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
