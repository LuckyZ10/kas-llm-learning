"""
Example 1: Basic Knowledge Base Usage
======================================
演示知识库模块的基本使用方法。
"""

import sys
sys.path.insert(0, '/root/.openclaw/workspace')

from dftlammps.knowledge_base import (
    create_knowledge_base,
    KnowledgeBuilder,
    KnowledgeAPI,
    APIConfig
)


def example_1_basic_storage():
    """示例1: 基本存储操作"""
    print("=" * 60)
    print("示例1: 基本存储操作")
    print("=" * 60)
    
    # 创建配置 (使用本地存储)
    config = APIConfig(
        mongodb_host="localhost",
        mongodb_port=27017,
        vector_provider="local",
        vector_dimension=128
    )
    
    # 初始化知识库
    with create_knowledge_base(config) as kb:
        print(f"\n✓ 知识库初始化完成")
        print(f"  健康状态: {kb.health_check()}")
        
        # 存储材料数据
        material = {
            "formula": "Li3PS4",
            "name": "Lithium Thiophosphate",
            "structure_type": "orthorhombic",
            "space_group": "Pnma",
            "properties": {
                "band_gap": 2.5,
                "ionic_conductivity": 1.6e-4,
                "density": 1.83
            }
        }
        
        record_id = kb.store_document("materials", material, tags=["solid_electrolyte", "battery"])
        print(f"\n✓ 存储材料数据")
        print(f"  记录ID: {record_id}")
        
        # 存储计算结果
        calculation = {
            "material": {"formula": "Li3PS4"},
            "method": {
                "name": "VASP",
                "type": "DFT",
                "xc": "PBE",
                "encut": 520
            },
            "results": {
                "total_energy": -156.234,
                "band_gap": 2.45,
                "lattice_a": 8.52,
                "lattice_b": 6.12,
                "lattice_c": 12.47
            }
        }
        
        calc_id = kb.store_document("calculations", calculation)
        print(f"\n✓ 存储计算结果")
        print(f"  记录ID: {calc_id}")


def example_2_knowledge_graph():
    """示例2: 知识图谱构建"""
    print("\n" + "=" * 60)
    print("示例2: 知识图谱构建")
    print("=" * 60)
    
    # 创建知识构建器
    builder = KnowledgeBuilder()
    
    # 添加材料实体
    li3ps4 = builder.add_entity(
        entity_type="Material",
        name="Li3PS4",
        properties={
            "formula": "Li3PS4",
            "structure_type": "orthorhombic",
            "space_group": "Pnma"
        }
    )
    print(f"\n✓ 创建材料实体")
    print(f"  ID: {li3ps4['id']}")
    print(f"  名称: {li3ps4['name']}")
    
    # 添加元素实体
    li = builder.add_entity("Element", "Li", {"atomic_number": 3, "group": 1})
    p = builder.add_entity("Element", "P", {"atomic_number": 15, "group": 15})
    s = builder.add_entity("Element", "S", {"atomic_number": 16, "group": 16})
    
    print(f"\n✓ 创建元素实体")
    print(f"  元素: Li, P, S")
    
    # 添加组成关系
    builder.add_relation("HAS_ELEMENT", li3ps4["id"], li["id"], {"count": 3})
    builder.add_relation("HAS_ELEMENT", li3ps4["id"], p["id"], {"count": 1})
    builder.add_relation("HAS_ELEMENT", li3ps4["id"], s["id"], {"count": 4})
    
    print(f"\n✓ 添加组成关系")
    
    # 添加性质实体
    band_gap = builder.add_entity(
        entity_type="Property",
        name="band_gap",
        properties={"value": 2.45, "unit": "eV"}
    )
    
    builder.add_relation("HAS_PROPERTY", li3ps4["id"], band_gap["id"])
    
    print(f"\n✓ 添加性质实体")
    
    # 获取知识图谱
    kg = builder.get_knowledge_graph()
    print(f"\n✓ 知识图谱统计")
    print(f"  实体数量: {len(kg['entities'])}")
    print(f"  关系数量: {len(kg['relations'])}")


def example_3_version_control():
    """示例3: 版本控制"""
    print("\n" + "=" * 60)
    print("示例3: 版本控制")
    print("=" * 60)
    
    from dftlammps.knowledge_base import create_version_control
    
    # 创建版本控制系统
    vc = create_version_control()
    
    # 提交版本
    v1 = vc.commit(
        data={
            "material": "Li3PS4",
            "energy": -156.234,
            "kpoints": [4, 4, 4]
        },
        message="Initial DFT calculation",
        author="researcher1",
        calculation_type="DFT",
        parameters={"encut": 500, "kspacing": 0.2},
        results={"energy": -156.234}
    )
    print(f"\n✓ 提交初始版本")
    print(f"  版本ID: {v1.id[:8]}")
    print(f"  消息: {v1.message}")
    
    # 修改参数后提交新版本
    v2 = vc.commit(
        data={
            "material": "Li3PS4",
            "energy": -157.123,
            "kpoints": [6, 6, 6]
        },
        message="Increase k-point density",
        author="researcher1",
        calculation_type="DFT",
        parameters={"encut": 500, "kspacing": 0.15},
        results={"energy": -157.123}
    )
    print(f"\n✓ 提交新版本")
    print(f"  版本ID: {v2.id[:8]}")
    print(f"  消息: {v2.message}")
    
    # 打标签
    vc.tag(v2.id, "v1.0", "Stable calculation result", "researcher1")
    print(f"\n✓ 为版本打标签")
    print(f"  标签: v1.0")
    
    # 比较版本差异
    diff = vc.diff(v1.id, v2.id)
    print(f"\n✓ 版本差异")
    print(f"  修改: {len(diff.modified)} 项")
    print(f"  新增: {len(diff.added)} 项")
    for path, change in diff.modified.items():
        print(f"    - {path}: {change['old']} -> {change['new']}")
    
    # 查看历史
    print(f"\n✓ 提交历史")
    for version in vc.log(limit=5):
        print(f"  [{version.id[:8]}] {version.message}")


def example_4_semantic_search():
    """示例4: 语义搜索"""
    print("\n" + "=" * 60)
    print("示例4: 语义搜索")
    print("=" * 60)
    
    from dftlammps.knowledge_base import (
        VectorConfig, create_vector_store, EmbeddingProvider,
        SearchConfig, SemanticSearch, SearchQuery, SearchMode
    )
    
    # 创建向量存储
    config = VectorConfig(provider="local", dimension=128)
    store = create_vector_store(config)
    store.connect()
    store.create_collection("materials", 128)
    
    # 设置嵌入提供者
    embedding_provider = EmbeddingProvider()
    store.set_embedding_provider(embedding_provider)
    
    print(f"\n✓ 向量存储初始化完成")
    
    # 准备材料数据
    materials = [
        {
            "id": "mat1",
            "content": "Li3PS4 is a solid electrolyte material for lithium batteries with high ionic conductivity",
            "formula": "Li3PS4",
            "type": "solid_electrolyte"
        },
        {
            "id": "mat2",
            "content": "LiCoO2 is a cathode material for lithium-ion batteries",
            "formula": "LiCoO2",
            "type": "cathode"
        },
        {
            "id": "mat3",
            "content": "Silicon is a promising anode material for next-generation batteries",
            "formula": "Si",
            "type": "anode"
        },
        {
            "id": "mat4",
            "content": "LLZO is a garnet-type solid electrolyte with excellent conductivity",
            "formula": "Li7La3Zr2O12",
            "type": "solid_electrolyte"
        }
    ]
    
    # 向量化并存储
    texts = [m["content"] for m in materials]
    ids = [m["id"] for m in materials]
    metadata = [{"formula": m["formula"], "type": m["type"]} for m in materials]
    
    store.upsert_texts(texts, ids, metadata)
    print(f"\n✓ 存储了 {len(materials)} 个材料")
    
    # 创建搜索引擎
    search_config = SearchConfig(
        vector_store=store,
        embedding_provider=embedding_provider
    )
    search_engine = SemanticSearch(search_config)
    
    # 执行语义搜索
    query = SearchQuery(
        query="solid electrolyte for batteries",
        mode=SearchMode.SEMANTIC,
        top_k=3
    )
    
    results = search_engine.search(query)
    
    print(f"\n✓ 语义搜索结果")
    print(f"  查询: '{query.query}'")
    for i, result in enumerate(results, 1):
        print(f"  {i}. [{result.id}] 相似度: {result.score:.3f}")
        print(f"     内容: {result.content[:60]}...")


def example_5_data_pipeline():
    """示例5: 数据管道"""
    print("\n" + "=" * 60)
    print("示例5: 数据管道")
    print("=" * 60)
    
    from dftlammps.knowledge_base import create_knowledge_api, APIConfig
    
    # 创建知识库API
    config = APIConfig(
        vector_provider="local",
        vector_dimension=128
    )
    
    with create_knowledge_api(config) as api:
        print(f"\n✓ 知识库API初始化")
        
        # 摄入计算结果
        calculation = {
            "type": "DFT",
            "material": {
                "formula": "LiFePO4",
                "structure_type": "olivine"
            },
            "method": {
                "name": "VASP",
                "xc": "PBE+U",
                "encut": 520
            },
            "parameters": {
                "kpoints": [6, 6, 6],
                "sigma": 0.05
            },
            "results": {
                "total_energy": -285.456,
                "band_gap": 3.5,
                "magnetic_moment": 3.8
            }
        }
        
        record_id = api.pipeline.ingest_calculation(calculation)
        print(f"\n✓ 摄入计算结果")
        print(f"  记录ID: {record_id}")
        
        # 查询文档
        results = api.query_documents("calculations", limit=5)
        print(f"\n✓ 查询结果")
        print(f"  找到 {len(results)} 个计算")
        for r in results:
            print(f"    - {r.get('material', {}).get('formula', 'unknown')}")


def run_all_examples():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("DFT-LAMMPS Knowledge Base Examples")
    print("=" * 60)
    
    try:
        example_2_knowledge_graph()
    except Exception as e:
        print(f"示例2出错: {e}")
    
    try:
        example_3_version_control()
    except Exception as e:
        print(f"示例3出错: {e}")
    
    try:
        example_4_semantic_search()
    except Exception as e:
        print(f"示例4出错: {e}")
    
    try:
        example_5_data_pipeline()
    except Exception as e:
        print(f"示例5出错: {e}")
    
    print("\n" + "=" * 60)
    print("所有示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
