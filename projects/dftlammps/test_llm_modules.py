#!/usr/bin/env python3
"""
LLM集成模块测试脚本
==================
验证所有模块功能正常运行。

Author: DFT-LAMMPS Team
"""

import sys
import traceback


def test_materials_gpt():
    """测试MaterialsGPT模块"""
    print("Testing MaterialsGPT...")
    try:
        from llm_interface.materials_gpt import (
            MaterialsGPT, MaterialEntity, ExperimentDesign, 
            ComputationParameters, TaskType
        )
        
        # 创建实例
        gpt = MaterialsGPT()
        
        # 测试文献提取
        text = "CsPbI3 perovskite has a band gap of 1.73 eV."
        knowledge = gpt.extract_knowledge_from_literature(text, "Test")
        assert isinstance(knowledge, dict)
        
        # 测试参数推荐
        params = gpt.recommend_computation_parameters(
            calc_type="scf", material="Si", software="VASP"
        )
        assert isinstance(params, ComputationParameters)
        assert params.task_type == "scf"
        
        # 测试实验设计
        design = gpt.design_experiment(
            objective="Test experiment",
            materials=["Fe", "O"]
        )
        assert isinstance(design, ExperimentDesign)
        
        print("  ✓ MaterialsGPT tests passed")
        return True
    except Exception as e:
        print(f"  ✗ MaterialsGPT tests failed: {e}")
        traceback.print_exc()
        return False


def test_code_generator():
    """测试CodeGenerator模块"""
    print("Testing CodeGenerator...")
    try:
        from llm_interface.code_generator import (
            CodeGenerator, CodeLanguage, CalculationType, GeneratedCode
        )
        
        # 创建实例
        gen = CodeGenerator()
        
        # 测试代码生成
        code = gen.generate_from_description(
            description="VASP SCF for Si",
            language=CodeLanguage.VASP,
            calc_type=CalculationType.SCF
        )
        assert isinstance(code, GeneratedCode)
        assert code.language == CodeLanguage.VASP
        
        # 测试错误诊断
        fixes = gen.diagnose_error(
            error_log="Error EDDDAV: Call to ZHEGV failed",
            language=CodeLanguage.VASP,
            current_input=""
        )
        assert len(fixes) > 0
        
        # 测试工作流构建
        workflow = gen.build_workflow(
            workflow_type="test",
            stages=[{"name": "step1", "language": "vasp", "calc_type": "scf"}]
        )
        assert "stages" in workflow
        
        print("  ✓ CodeGenerator tests passed")
        return True
    except Exception as e:
        print(f"  ✗ CodeGenerator tests failed: {e}")
        traceback.print_exc()
        return False


def test_chat_assistant():
    """测试ChatAssistant模块"""
    print("Testing ChatAssistant...")
    try:
        from llm_interface.chat_assistant import (
            ChatAssistant, ExpertiseLevel, QAResponse
        )
        
        # 创建实例
        assistant = ChatAssistant()
        
        # 测试聊天
        response = assistant.chat("Hello!", session_id="test")
        assert isinstance(response, QAResponse)
        assert len(response.answer) > 0
        
        # 测试诊断
        diagnosis = assistant.diagnose(
            error_log="ZHEGV failed",
            software="vasp"
        )
        assert diagnosis.problem_type is not None
        
        # 测试指导
        guidance = assistant.get_guidance("scf_initialization", "vasp")
        assert len(guidance) > 0
        
        print("  ✓ ChatAssistant tests passed")
        return True
    except Exception as e:
        print(f"  ✗ ChatAssistant tests failed: {e}")
        traceback.print_exc()
        return False


def test_knowledge_graph():
    """测试KnowledgeGraph模块"""
    print("Testing KnowledgeGraph...")
    try:
        from knowledge_graph.kg_core import (
            KnowledgeGraph, Entity, EntityType, Relation, RelationType
        )
        
        # 创建实例
        kg = KnowledgeGraph()
        
        # 测试知识提取
        text = "Fe3O4 is a magnetic material with band gap 0.1 eV."
        stats = kg.ingest_text(text, "Test Paper")
        assert isinstance(stats, dict)
        assert stats["entities_added"] >= 0
        
        # 测试查询
        results = kg.query("Fe")
        assert isinstance(results, dict)
        
        # 测试实体添加
        entity = Entity(
            id="test_001",
            name="TestMaterial",
            entity_type=EntityType.MATERIAL
        )
        kg.add_entity(entity)
        assert "test_001" in kg.entities
        
        print("  ✓ KnowledgeGraph tests passed")
        return True
    except Exception as e:
        print(f"  ✗ KnowledgeGraph tests failed: {e}")
        traceback.print_exc()
        return False


def test_knowledge_graph_init():
    """测试知识图谱初始化"""
    print("Testing KnowledgeGraph initialization...")
    try:
        from knowledge_graph.kg_init import initialize_materials_ontology
        from knowledge_graph.kg_core import KnowledgeGraph
        
        kg = KnowledgeGraph()
        initialize_materials_ontology(kg)
        
        # 检查是否添加了实体
        assert len(kg.entities) > 0
        
        # 检查元素是否添加
        fe_entities = kg.find_entities_by_name("Fe")
        assert len(fe_entities) > 0
        
        print("  ✓ KnowledgeGraph initialization tests passed")
        return True
    except Exception as e:
        print(f"  ✗ KnowledgeGraph initialization tests failed: {e}")
        traceback.print_exc()
        return False


def test_application_examples():
    """测试应用案例"""
    print("Testing Application Examples...")
    try:
        from llm_interface.application_examples import (
            LiteratureMiningExample,
            NLWorkflowDesignExample,
            SmartLabNotebookExample
        )
        
        # 测试文献挖掘
        lit_example = LiteratureMiningExample()
        knowledge = lit_example.demonstrate_literature_mining()
        assert isinstance(knowledge, list)
        
        # 测试工作流设计
        nl_example = NLWorkflowDesignExample()
        workflows = nl_example.demonstrate_nl_workflow()
        assert isinstance(workflows, list)
        
        # 测试智能日记
        notebook_example = SmartLabNotebookExample()
        data = notebook_example.demonstrate_smart_notebook()
        assert isinstance(data, dict)
        
        print("  ✓ Application Examples tests passed")
        return True
    except Exception as e:
        print(f"  ✗ Application Examples tests failed: {e}")
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("=" * 70)
    print("DFT-LAMMPS LLM集成模块测试")
    print("=" * 70)
    print()
    
    tests = [
        test_materials_gpt,
        test_code_generator,
        test_chat_assistant,
        test_knowledge_graph,
        test_knowledge_graph_init,
        test_application_examples
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test crashed: {e}")
            results.append(False)
        print()
    
    # 打印结果
    print("=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("\n✓ 所有测试通过!")
        return 0
    else:
        print(f"\n✗ {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
