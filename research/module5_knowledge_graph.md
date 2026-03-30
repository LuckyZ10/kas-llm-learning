# 材料数据库与自动化工具研究报告
## 模块5: 数据挖掘与知识图谱构建

**研究时间**: 2026-03-08 16:55+

---

## 1. 知识图谱基础架构

### 1.1 核心概念
知识图谱(Knowledge Graph)是一种用图结构表示知识的语义网络，由**实体-关系-实体**三元组组成。

```
┌─────────────────────────────────────────────────────────┐
│                    知识图谱结构                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│    [Material] ──hasProperty──> [BandGap]               │
│         │                         ↑                     │
│    containsElement           predictedBy               │
│         │                    [CGCNN_Model]              │
│         ↓                                               │
│    [Element: Si] ──belongsTo──> [Group IV]             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 1.2 技术栈对比
| 技术 | 用途 | 特点 |
|------|------|------|
| **Neo4j** | 图数据库 | Cypher查询，GDS算法库 |
| **RDF/OWL** | 语义网 | W3C标准，SPARQL查询 |
| **Apache Jena** | 语义框架 | 推理能力，TDB存储 |
| **JanusGraph** | 分布式图 | 大规模，多后端 |

---

## 2. Neo4j图数据库

### 2.1 核心特性
- **属性图模型**: 节点和关系都可带有属性
- **Cypher查询语言**: 声明式图查询语言
- **图数据科学(GDS)**: 内置 centrality/community detection算法
- **GraphRAG**: 与LLM集成，支持检索增强生成

### 2.2 Cypher查询示例
```cypher
// 创建材料节点
CREATE (m:Material {
  formula: "SiO2",
  space_group: "P3221",
  band_gap: 9.0
})

// 创建关系
CREATE (e:Element {symbol: "Si", number: 14})
CREATE (m)-[:CONTAINS {count: 1}]->(e)

// 复杂查询：找所有含Si的半导体
MATCH (m:Material)-[:CONTAINS]->(e:Element {symbol: "Si"})
WHERE m.band_gap > 0 AND m.band_gap < 3
RETURN m.formula, m.band_gap
ORDER BY m.band_gap DESC
```

### 2.3 图数据科学算法
```python
from graphdatascience import GraphDataScience

gds = GraphDataScience("bolt://localhost:7687", auth=("neo4j", "password"))

# 创建投影
gds.graph.project.cypher(
    "materials-graph",
    "MATCH (m:Material) RETURN id(m) AS id",
    "MATCH (m1:Material)-[:SIMILAR]->(m2:Material) RETURN id(m1) AS source, id(m2) AS target"
)

# 运行社区检测
gds.louvain.stream("materials-graph")
```

---

## 3. 语义网技术 (RDF/OWL)

### 3.1 RDF三元组
```turtle
@prefix mat: <http://example.org/materials/> .
@prefix ex: <http://example.org/> .

mat:mp-1234 a ex:Material ;
    ex:formula "GaAs" ;
    ex:hasProperty mat:bandgap-1 ;
    ex:containsElement ex:Ga, ex:As .

mat:bandgap-1 a ex:BandGap ;
    ex:value 1.42 ;
    ex:unit "eV" .
```

### 3.2 OWL本体定义
```owl
:Material rdf:type owl:Class .
:Element rdf:type owl:Class .
:Property rdf:type owl:Class .

:containsElement rdf:type owl:ObjectProperty ;
    rdfs:domain :Material ;
    rdfs:range :Element .

:hasProperty rdf:type owl:ObjectProperty ;
    rdfs:domain :Material ;
    rdfs:range :Property .

:bandGap rdf:type owl:DatatypeProperty ;
    rdfs:domain :ElectronicProperty ;
    rdfs:range xsd:float .
```

---

## 4. FAIR数据原则

### 4.1 四大原则
| 原则 | 含义 | 实施要点 |
|------|------|---------|
| **Findable** | 可发现 | 全局唯一标识符(PID), 丰富元数据 |
| **Accessible** | 可访问 | 标准协议, 身份验证机制 |
| **Interoperable** | 可互操作 | 标准格式, 受控词表, 本体 |
| **Reusable** | 可重用 | 明确许可, 详细溯源信息 |

### 4.2 材料科学FAIR实践 (NOMAD示例)
```
Findability: 唯一名称 + 人类可读描述
Accessibility: RESTful API + PID
Interoperability: 可扩展模式 + 本体链接
Reusability: 模块化层次结构
```

---

## 5. 材料知识图谱构建流程

### 5.1 数据抽取与转换
```python
# 从Materials Project抽取数据构建知识图谱
from pymatgen.ext.matproj import MPRester
from py2neo import Graph, Node, Relationship

graph = Graph("bolt://localhost:7687", auth=("neo4j", "pass"))
mpr = MPRester("API_KEY")

# 获取数据
data = mpr.query(criteria={}, properties=["material_id", "formula", "band_gap"])

# 创建节点和关系
tx = graph.begin()
for item in data:
    material = Node("Material", 
                   id=item["material_id"],
                   formula=item["formula"],
                   band_gap=item["band_gap"])
    tx.create(material)
graph.commit(tx)
```

### 5.2 知识融合
- **实体对齐**: 识别不同数据源中的同一实体
- **关系推理**: 基于本体规则推断隐含关系
- **冲突解决**: 处理多源数据的不一致性

---

## 6. 应用案例

### 6.1 材料发现推荐
```cypher
// 基于相似性推荐替代材料
MATCH (m:Material {formula: "LiCoO2"})-[:SIMILAR*1..3]-(candidate:Material)
WHERE candidate.band_gap > 0
RETURN candidate.formula, 
       candidate.energy_density,
       count(*) AS similarity_score
ORDER BY similarity_score DESC
LIMIT 10
```

### 6.2 多跳推理
```
问题: "哪些元素常用于锂离子电池正极材料?"

查询路径:
[Application:Li-ion_battery] <-usedIn- [Material] -containsElement-> [Element]

结果聚合:
- Co (出现在85%的材料中)
- Ni (出现在72%的材料中)
- Mn (出现在68%的材料中)
```

---

**模块5研究完成**
