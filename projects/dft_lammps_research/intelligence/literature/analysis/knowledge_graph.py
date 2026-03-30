"""
知识图谱模块
构建概念关系网络
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
import json

from ..config.models import Paper


class KnowledgeGraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self, min_cooccurrence: int = 3):
        """
        Args:
            min_cooccurrence: 最小共现次数
        """
        self.min_cooccurrence = min_cooccurrence
        self.nodes = {}  # 节点
        self.edges = defaultdict(int)  # 边权重
        self.node_types = {}  # 节点类型
    
    def build_from_papers(self, papers: List[Paper]) -> Dict:
        """
        从论文构建知识图谱
        
        Args:
            papers: 论文列表
        
        Returns:
            知识图谱字典
        """
        # 提取实体
        entities_per_paper = []
        for paper in papers:
            entities = self._extract_entities(paper)
            entities_per_paper.append(entities)
        
        # 统计节点
        all_entities = Counter()
        for entities in entities_per_paper:
            for entity, entity_type in entities.items():
                all_entities[entity] += 1
                if entity not in self.node_types:
                    self.node_types[entity] = entity_type
        
        # 筛选频繁出现的实体
        frequent_entities = {
            entity for entity, count in all_entities.items()
            if count >= self.min_cooccurrence
        }
        
        # 构建节点
        for entity in frequent_entities:
            self.nodes[entity] = {
                "id": entity,
                "type": self.node_types.get(entity, "unknown"),
                "frequency": all_entities[entity]
            }
        
        # 构建边（共现关系）
        for entities in entities_per_paper:
            entity_list = [
                e for e in entities.keys()
                if e in frequent_entities
            ]
            
            for i, e1 in enumerate(entity_list):
                for e2 in entity_list[i+1:]:
                    if e1 != e2:
                        edge_key = tuple(sorted([e1, e2]))
                        self.edges[edge_key] += 1
        
        # 过滤弱边
        self.edges = {
            k: v for k, v in self.edges.items()
            if v >= self.min_cooccurrence
        }
        
        return self.to_dict()
    
    def _extract_entities(self, paper: Paper) -> Dict[str, str]:
        """
        从论文提取实体
        
        Returns:
            实体名称到类型的映射
        """
        entities = {}
        text = f"{paper.title} {paper.abstract}"
        if paper.full_text:
            text += f" {paper.full_text[:10000]}"
        
        text_lower = text.lower()
        
        # 方法实体
        methods = [
            "DFT", "density functional theory", "molecular dynamics", "MD",
            "machine learning", "deep learning", "neural network", "Monte Carlo",
            "molecular dynamics simulation", "ab initio", "first principles",
            "cluster expansion", "CALPHAD", "kinetic Monte Carlo"
        ]
        for method in methods:
            if method.lower() in text_lower:
                entities[method] = "method"
        
        # 软件实体
        software = [
            "VASP", "Quantum ESPRESSO", "Gaussian", "LAMMPS", "GROMACS",
            "NAMD", "ASE", "Pymatgen", "CP2K", "ABINIT", "CASTEP"
        ]
        for sw in software:
            if sw.lower() in text_lower or sw in text:
                entities[sw] = "software"
        
        # 材料实体
        materials = [
            "lithium", "Li", "sodium", "Na", "potassium", "K",
            "oxide", "sulfide", "phosphate", "silicate",
            "electrolyte", "cathode", "anode", "electrode",
            "battery", "supercapacitor", "fuel cell"
        ]
        for mat in materials:
            if re.search(r'\b' + re.escape(mat) + r'\b', text_lower):
                entities[mat] = "material"
        
        # 性质实体
        properties = [
            "band gap", "electronic structure", "density of states",
            "formation energy", "diffusion coefficient", "ionic conductivity",
            "elastic modulus", "hardness", "melting point"
        ]
        for prop in properties:
            if prop in text_lower:
                entities[prop] = "property"
        
        # 添加论文关键词
        for keyword in paper.keywords:
            if keyword not in entities and len(keyword) > 2:
                entities[keyword] = "keyword"
        
        # 添加主题
        for topic in paper.topics:
            if topic not in entities:
                entities[topic] = "topic"
        
        return entities
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "nodes": list(self.nodes.values()),
            "edges": [
                {
                    "source": edge[0],
                    "target": edge[1],
                    "weight": weight
                }
                for edge, weight in self.edges.items()
            ]
        }
    
    def to_cytoscape(self) -> List[Dict]:
        """转换为Cytoscape.js格式"""
        elements = []
        
        # 节点
        for entity, data in self.nodes.items():
            elements.append({
                "data": {
                    "id": entity,
                    "label": entity,
                    "type": data["type"],
                    "frequency": data["frequency"]
                }
            })
        
        # 边
        for (source, target), weight in self.edges.items():
            elements.append({
                "data": {
                    "id": f"{source}-{target}",
                    "source": source,
                    "target": target,
                    "weight": weight
                }
            })
        
        return elements
    
    def to_d3(self) -> Dict:
        """转换为D3.js格式"""
        nodes = []
        node_ids = {}
        
        for i, (entity, data) in enumerate(self.nodes.items()):
            nodes.append({
                "id": i,
                "name": entity,
                "type": data["type"],
                "value": data["frequency"]
            })
            node_ids[entity] = i
        
        links = []
        for (source, target), weight in self.edges.items():
            if source in node_ids and target in node_ids:
                links.append({
                    "source": node_ids[source],
                    "target": node_ids[target],
                    "value": weight
                })
        
        return {"nodes": nodes, "links": links}
    
    def get_related_concepts(self, concept: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        获取相关概念
        
        Args:
            concept: 查询概念
            top_n: 返回数量
        
        Returns:
            相关概念和关联强度
        """
        related = []
        
        for (source, target), weight in self.edges.items():
            if source == concept:
                related.append((target, weight))
            elif target == concept:
                related.append((source, weight))
        
        return sorted(related, key=lambda x: x[1], reverse=True)[:top_n]
    
    def find_communities(self) -> Dict[str, List[str]]:
        """
        发现概念社区（简单连通分量）
        
        Returns:
            社区字典
        """
        # 构建邻接表
        adjacency = defaultdict(set)
        for (source, target), _ in self.edges.items():
            adjacency[source].add(target)
            adjacency[target].add(source)
        
        # 寻找连通分量
        visited = set()
        communities = {}
        community_id = 0
        
        for node in self.nodes:
            if node not in visited:
                community = []
                stack = [node]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        community.append(current)
                        stack.extend(adjacency[current] - visited)
                
                if len(community) >= 3:  # 只保留较大的社区
                    communities[f"community_{community_id}"] = community
                    community_id += 1
        
        return communities
    
    def get_central_concepts(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        获取中心概念（度中心性）
        
        Returns:
            中心概念和度数
        """
        degrees = Counter()
        
        for (source, target), weight in self.edges.items():
            degrees[source] += weight
            degrees[target] += weight
        
        return degrees.most_common(top_n)
    
    def export_graphml(self, output_path: str):
        """导出为GraphML格式"""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
            '  <key id="type" for="node" attr.name="type" attr.type="string"/>',
            '  <key id="frequency" for="node" attr.name="frequency" attr.type="int"/>',
            '  <key id="weight" for="edge" attr.name="weight" attr.type="int"/>',
            '  <graph id="G" edgedefault="undirected">'
        ]
        
        # 节点
        for entity, data in self.nodes.items():
            escaped = entity.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            lines.append(f'    <node id="{escaped}">')
            lines.append(f'      <data key="type">{data["type"]}</data>')
            lines.append(f'      <data key="frequency">{data["frequency"]}</data>')
            lines.append('    </node>')
        
        # 边
        for (source, target), weight in self.edges.items():
            s = source.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            t = target.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            lines.append(f'    <edge source="{s}" target="{t}">')
            lines.append(f'      <data key="weight">{weight}</data>')
            lines.append('    </edge>')
        
        lines.append('  </graph>')
        lines.append('</graphml>')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


class ConceptEvolutionTracker:
    """概念演变追踪器"""
    
    def __init__(self):
        self.year_graphs = {}
    
    def track_concept(
        self,
        papers: List[Paper],
        concept: str
    ) -> Dict[int, Dict]:
        """
        追踪概念演变
        
        Args:
            papers: 论文列表
            concept: 追踪的概念
        
        Returns:
            年度演变数据
        """
        # 按年份分组
        year_papers = defaultdict(list)
        for paper in papers:
            year = paper.publication_date.year
            year_papers[year].append(paper)
        
        evolution = {}
        
        for year in sorted(year_papers.keys()):
            papers_in_year = year_papers[year]
            
            # 构建当年图谱
            builder = KnowledgeGraphBuilder(min_cooccurrence=1)
            builder.build_from_papers(papers_in_year)
            
            # 获取概念相关信息
            if concept in builder.nodes:
                related = builder.get_related_concepts(concept, top_n=5)
                evolution[year] = {
                    "frequency": builder.nodes[concept]["frequency"],
                    "related_concepts": related,
                    "total_papers": len(papers_in_year)
                }
        
        return evolution
