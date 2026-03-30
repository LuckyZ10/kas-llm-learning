"""
自主材料科学家应用案例
========================
演示如何使用自主智能体系统进行材料发现和研究。
"""

import asyncio
import sys
from datetime import datetime
from typing import Dict, Any, List

# 添加父目录到路径
sys.path.insert(0, '/root/.openclaw/workspace')

from dftlammps.autonomous_agent.agent_core import (
    AutonomousAgent, Goal, Task, TaskStatus, Reflection,
    DFTTool, LAMMPTool, StructureAnalysisTool
)
from dftlammps.autonomous_agent.experiment_planner import (
    ExperimentPlanner, ExperimentalVariable, ResourceType
)
from dftlammps.autonomous_agent.literature_agent import LiteratureAgent


class AutonomousCatalystDiscovery:
    """
    自主发现新催化剂
    
    演示如何使用自主智能体系统自动发现和优化催化剂。
    """
    
    def __init__(self):
        self.agent = AutonomousAgent("CatalystDiscoveryAgent")
        self.planner = ExperimentPlanner()
        self.literature_agent = LiteratureAgent()
        self.results_history: List[Dict] = []
        
        # 注册工具
        self._setup_tools()
        
    def _setup_tools(self):
        """设置工具"""
        self.agent.register_tool(DFTTool())
        self.agent.register_tool(LAMMPTool())
        self.agent.register_tool(StructureAnalysisTool())
    
    async def discover_catalyst(self, target_reaction: str, 
                                performance_target: Dict[str, float]) -> Dict:
        """
        自主发现催化剂
        
        完整的催化剂自主发现流程。
        """
        print("=" * 70)
        print("自主催化剂发现系统")
        print("=" * 70)
        print(f"目标反应: {target_reaction}")
        print(f"性能目标: {performance_target}")
        print()
        
        discovery_report = {
            "target_reaction": target_reaction,
            "performance_target": performance_target,
            "start_time": datetime.now().isoformat(),
            "phases": []
        }
        
        # ========== 阶段1: 文献调研 ==========
        print("\n[阶段 1] 文献调研与知识获取")
        print("-" * 50)
        
        phase1 = await self._literature_research(target_reaction)
        discovery_report["phases"].append({
            "name": "Literature Research",
            "result": phase1
        })
        
        # ========== 阶段2: 假设生成 ==========
        print("\n[阶段 2] 生成科学假设")
        print("-" * 50)
        
        phase2 = await self._generate_hypotheses(target_reaction, phase1)
        discovery_report["phases"].append({
            "name": "Hypothesis Generation",
            "result": phase2
        })
        
        # ========== 阶段3: 实验规划 ==========
        print("\n[阶段 3] 规划计算实验")
        print("-" * 50)
        
        phase3 = await self._plan_experiments(phase2)
        discovery_report["phases"].append({
            "name": "Experiment Planning",
            "result": phase3
        })
        
        # ========== 阶段4: 自主执行 ==========
        print("\n[阶段 4] 自主执行与优化")
        print("-" * 50)
        
        phase4 = await self._autonomous_execution(
            target_reaction, performance_target, phase3
        )
        discovery_report["phases"].append({
            "name": "Autonomous Execution",
            "result": phase4
        })
        
        # ========== 阶段5: 结果分析与报告 ==========
        print("\n[阶段 5] 分析与报告生成")
        print("-" * 50)
        
        phase5 = await self._generate_report(discovery_report)
        discovery_report["phases"].append({
            "name": "Analysis and Reporting",
            "result": phase5
        })
        
        discovery_report["end_time"] = datetime.now().isoformat()
        discovery_report["summary"] = phase5
        
        print("\n" + "=" * 70)
        print("催化剂发现流程完成!")
        print("=" * 70)
        
        return discovery_report
    
    async def _literature_research(self, target_reaction: str) -> Dict:
        """文献调研"""
        print("  → 搜索相关文献...")
        
        # 搜索文献
        papers = await self.literature_agent.search_literature(
            query=f"{target_reaction} catalyst DFT",
            max_results=15
        )
        
        print(f"     找到 {len(papers)} 篇相关文献")
        
        # 提取知识
        print("  → 提取关键知识...")
        knowledge_units = await self.literature_agent.extract_knowledge(papers)
        print(f"     提取 {len(knowledge_units)} 个知识单元")
        
        # 整合知识
        integration = await self.literature_agent.integrate_knowledge()
        
        # 识别研究空白
        gaps = await self.literature_agent.identify_research_gaps()
        
        return {
            "papers_reviewed": len(papers),
            "knowledge_extracted": len(knowledge_units),
            "integration_result": integration,
            "research_gaps": [g.to_dict() for g in gaps[:3]]
        }
    
    async def _generate_hypotheses(self, target_reaction: str, 
                                    literature_data: Dict) -> Dict:
        """生成假设"""
        print("  → 基于文献生成假设...")
        
        hypotheses = self.planner.hypothesis_generator.generate_hypotheses(
            research_question=f"发现{target_reaction}的高效催化剂",
            num_hypotheses=5
        )
        
        print(f"     生成 {len(hypotheses)} 个候选假设")
        
        for i, h in enumerate(hypotheses, 1):
            print(f"     {i}. {h.statement[:50]}...")
            print(f"        置信度: {h.confidence:.2f}, 优先级: {h.priority:.2f}")
        
        return {
            "hypotheses": [
                {
                    "id": h.id,
                    "statement": h.statement,
                    "confidence": h.confidence,
                    "priority": h.priority,
                    "novelty": h.novelty_score
                }
                for h in hypotheses
            ]
        }
    
    async def _plan_experiments(self, hypothesis_data: Dict) -> Dict:
        """规划实验"""
        print("  → 创建实验计划...")
        
        plan = await self.planner.create_experiment_plan(
            research_question="基于DFT计算筛选催化剂",
            available_resources={
                ResourceType.CPU: 200,
                ResourceType.GPU: 20,
                ResourceType.MEMORY: 2000
            },
            constraints={
                "max_budget": 10000,
                "num_hypotheses": 3
            }
        )
        
        print(f"     计划包含 {len(plan['experiments'])} 个实验")
        print(f"     预计总成本: ${plan['total_estimated_cost']:.2f}")
        print(f"     预计总时长: {plan['total_duration']:.1f} 小时")
        
        return {
            "plan_id": plan["id"],
            "experiment_count": len(plan["experiments"]),
            "estimated_cost": plan["total_estimated_cost"],
            "estimated_duration": plan["total_duration"],
            "risk_score": plan["overall_risk_score"]
        }
    
    async def _autonomous_execution(self, target_reaction: str,
                                     performance_target: Dict[str, float],
                                     plan_data: Dict) -> Dict:
        """自主执行"""
        print("  → 启动自主执行循环...")
        
        # 设置目标
        goal_desc = f"发现{target_reaction}催化剂，过电位<{performance_target.get('overpotential', 0.3)}V"
        
        # 执行
        execution_result = await self.agent.run(
            goal_description=goal_desc,
            criteria={
                "requires_experiment": True,
                **performance_target
            }
        )
        
        print(f"     完成任务数: {execution_result['tasks_completed']}")
        print(f"     成功率: {execution_result['success_rate']:.2%}")
        print(f"     目标达成: {execution_result['goal_achieved']}")
        
        return {
            "tasks_completed": execution_result['tasks_completed'],
            "success_rate": execution_result['success_rate'],
            "goal_achieved": execution_result['goal_achieved'],
            "reflection_insights": len(execution_result['reflection'].insights)
        }
    
    async def _generate_report(self, discovery_data: Dict) -> Dict:
        """生成报告"""
        print("  → 生成发现报告...")
        
        summary = {
            "total_phases": len(discovery_data["phases"]),
            "papers_reviewed": discovery_data["phases"][0]["result"]["papers_reviewed"],
            "hypotheses_generated": len(discovery_data["phases"][1]["result"]["hypotheses"]),
            "experiments_planned": discovery_data["phases"][2]["result"]["experiment_count"],
            "goal_achieved": discovery_data["phases"][3]["result"]["goal_achieved"]
        }
        
        print("\n  发现摘要:")
        print(f"     - 文献调研: {summary['papers_reviewed']} 篇")
        print(f"     - 生成假设: {summary['hypotheses_generated']} 个")
        print(f"     - 规划实验: {summary['experiments_planned']} 个")
        print(f"     - 目标达成: {'是' if summary['goal_achieved'] else '否'}")
        
        return summary


class AutonomousSynthesisOptimization:
    """
    自主优化合成路线
    
    演示如何使用自主智能体系统优化材料合成。
    """
    
    def __init__(self):
        self.agent = AutonomousAgent("SynthesisOptimizationAgent")
        self.planner = ExperimentPlanner()
        self.optimization_history: List[Dict] = []
        
    async def optimize_synthesis(self, target_material: str,
                                  target_properties: Dict[str, Any]) -> Dict:
        """
        优化合成路线
        """
        print("\n" + "=" * 70)
        print("自主合成路线优化系统")
        print("=" * 70)
        print(f"目标材料: {target_material}")
        print(f"目标性质: {target_properties}")
        
        optimization_report = {
            "target_material": target_material,
            "target_properties": target_properties,
            "iterations": []
        }
        
        # 迭代优化
        for iteration in range(3):  # 模拟3轮迭代
            print(f"\n[迭代 {iteration + 1}/3]")
            print("-" * 50)
            
            iter_result = await self._optimization_iteration(
                iteration, target_material, target_properties
            )
            optimization_report["iterations"].append(iter_result)
            
            # 检查是否达到目标
            if iter_result.get("meets_target", False):
                print(f"  ✓ 达到目标，提前结束!")
                break
        
        # 生成最终报告
        final_report = await self._generate_synthesis_report(optimization_report)
        
        print("\n" + "=" * 70)
        print("合成路线优化完成!")
        print("=" * 70)
        
        return final_report
    
    async def _optimization_iteration(self, iteration: int,
                                       target_material: str,
                                       target_properties: Dict) -> Dict:
        """单次优化迭代"""
        print(f"  → 设计合成参数...")
        
        # 定义变量
        variables = [
            ExperimentalVariable(
                name="temperature",
                type="continuous",
                range=(300, 1500),
                importance=0.9
            ),
            ExperimentalVariable(
                name="pressure",
                type="continuous",
                range=(0.1, 100),
                importance=0.7
            ),
            ExperimentalVariable(
                name="time",
                type="continuous",
                range=(1, 48),
                importance=0.6
            ),
            ExperimentalVariable(
                name="precursor_ratio",
                type="continuous",
                range=(0.5, 2.0),
                importance=0.8
            )
        ]
        
        # 生成设计
        design_matrix = self.planner.design_optimizer._latin_hypercube_sampling(
            variables, num_samples=5
        )
        
        print(f"     设计 {len(design_matrix)} 组实验条件")
        
        # 模拟执行
        print(f"  → 执行计算实验...")
        results = []
        for i, params in enumerate(design_matrix):
            # 模拟结果
            result = self._simulate_synthesis(params, target_properties)
            results.append({
                "params": params,
                "result": result,
                "score": result.get("score", 0)
            })
            print(f"     实验 {i+1}: 得分 = {result.get('score', 0):.3f}")
        
        # 找到最佳
        best = max(results, key=lambda x: x["score"])
        print(f"  → 最佳结果: 得分 = {best['score']:.3f}")
        
        meets_target = best["score"] >= 0.8
        
        return {
            "iteration": iteration + 1,
            "designs_tested": len(design_matrix),
            "best_score": best["score"],
            "best_params": best["params"],
            "meets_target": meets_target
        }
    
    def _simulate_synthesis(self, params: Dict, 
                            target: Dict) -> Dict:
        """模拟合成结果"""
        import numpy as np
        
        # 简化的模拟
        temp_score = 1.0 - abs(params.get("temperature", 800) - 800) / 1000
        press_score = 1.0 - abs(params.get("pressure", 1) - 1) / 50
        time_score = 1.0 - abs(params.get("time", 12) - 12) / 20
        ratio_score = 1.0 - abs(params.get("precursor_ratio", 1) - 1) / 1
        
        overall_score = (temp_score + press_score + time_score + ratio_score) / 4
        overall_score += np.random.randn() * 0.05  # 添加噪声
        
        return {
            "score": max(0, min(1, overall_score)),
            "purity": max(0, min(100, 80 + overall_score * 20)),
            "yield": max(0, min(100, 60 + overall_score * 30)),
            "crystal_quality": max(0, min(10, overall_score * 10))
        }
    
    async def _generate_synthesis_report(self, data: Dict) -> Dict:
        """生成合成报告"""
        iterations = data["iterations"]
        
        best_iteration = max(iterations, key=lambda x: x["best_score"])
        
        report = {
            "target_material": data["target_material"],
            "total_iterations": len(iterations),
            "best_score": best_iteration["best_score"],
            "optimal_parameters": best_iteration["best_params"],
            "success": best_iteration["meets_target"],
            "improvement": best_iteration["best_score"] - iterations[0]["best_score"] if iterations else 0
        }
        
        print("\n  优化报告:")
        print(f"     迭代次数: {report['total_iterations']}")
        print(f"     最佳得分: {report['best_score']:.3f}")
        print(f"     性能提升: {report['improvement']:.3f}")
        print(f"     最优参数:")
        for param, value in report['optimal_parameters'].items():
            print(f"       - {param}: {value:.2f}")
        
        return report


class AutonomousResearchReporter:
    """
    自主撰写研究报告
    
    自动分析研究结果并生成报告。
    """
    
    def __init__(self):
        self.literature_agent = LiteratureAgent()
        
    async def generate_research_report(self, 
                                        research_topic: str,
                                        experimental_data: List[Dict]) -> str:
        """
        生成研究报告
        """
        print("\n" + "=" * 70)
        print("自主研究报告生成系统")
        print("=" * 70)
        print(f"研究主题: {research_topic}")
        
        # 1. 文献背景
        print("\n[1] 收集文献背景...")
        literature_review = await self.literature_agent.generate_literature_review(
            research_topic
        )
        
        # 2. 分析实验数据
        print("\n[2] 分析实验数据...")
        data_analysis = self._analyze_experimental_data(experimental_data)
        
        # 3. 生成报告
        print("\n[3] 撰写报告...")
        report = self._compile_report(
            research_topic, literature_review, data_analysis
        )
        
        print("\n" + "=" * 70)
        print("报告生成完成!")
        print("=" * 70)
        
        return report
    
    def _analyze_experimental_data(self, data: List[Dict]) -> Dict:
        """分析实验数据"""
        if not data:
            return {"error": "No data provided"}
        
        # 统计摘要
        scores = [d.get("score", 0) for d in data]
        
        import numpy as np
        
        analysis = {
            "sample_count": len(data),
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "max_score": np.max(scores),
            "min_score": np.min(scores),
            "best_config": max(data, key=lambda x: x.get("score", 0))
        }
        
        return analysis
    
    def _compile_report(self, topic: str, literature: str, 
                        analysis: Dict) -> str:
        """编译报告"""
        report = f"""
{'=' * 70}
                      研究报告: {topic}
{'=' * 70}

生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{'=' * 70}
一、文献综述
{'=' * 70}

{literature}

{'=' * 70}
二、实验数据分析
{'=' * 70}

实验样本数: {analysis.get('sample_count', 0)}
平均得分: {analysis.get('mean_score', 0):.3f} ± {analysis.get('std_score', 0):.3f}
最佳得分: {analysis.get('max_score', 0):.3f}

最佳配置:
"""
        
        best_config = analysis.get('best_config', {})
        if 'params' in best_config:
            for param, value in best_config['params'].items():
                report += f"  - {param}: {value:.3f}\n"
        
        report += f"""
{'=' * 70}
三、结论与展望
{'=' * 70}

本研究通过自主智能体系统对{topic}进行了系统性研究。
通过结合文献调研、假设生成、实验规划和自主执行，
我们实现了研究流程的自动化和智能化。

主要发现:
1. 系统成功分析了现有文献，识别了关键研究空白
2. 生成了多个可检验的科学假设
3. 优化了实验设计，提高了研究效率
4. 通过多轮迭代，逐步接近性能目标

未来工作:
- 扩展搜索空间，探索更多材料组合
- 集成更多实验表征技术
- 建立完整的知识图谱，支持更复杂的推理

{'=' * 70}
报告结束
{'=' * 70}
"""
        
        return report


async def main():
    """主函数"""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  自主材料科学家演示  ".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70 + "\n")
    
    # 示例1: 自主发现催化剂
    print("\n" + "▶" * 35)
    print("示例 1: 自主发现水分解催化剂")
    print("▶" * 35)
    
    catalyst_discovery = AutonomousCatalystDiscovery()
    
    catalyst_report = await catalyst_discovery.discover_catalyst(
        target_reaction="Water Splitting HER",
        performance_target={
            "overpotential": 0.2,  # V
            "stability_hours": 1000,
            "cost_per_gram": 10.0
        }
    )
    
    # 示例2: 自主优化合成路线
    print("\n" + "▶" * 35)
    print("示例 2: 自主优化钙钛矿合成")
    print("▶" * 35)
    
    synthesis_optimization = AutonomousSynthesisOptimization()
    
    synthesis_report = await synthesis_optimization.optimize_synthesis(
        target_material="SrTiO3 Perovskite",
        target_properties={
            "band_gap": 3.2,
            "crystallinity": "high",
            "purity": ">99%"
        }
    )
    
    # 示例3: 自主撰写研究报告
    print("\n" + "▶" * 35)
    print("示例 3: 自主撰写研究报告")
    print("▶" * 35)
    
    research_reporter = AutonomousResearchReporter()
    
    # 准备示例实验数据
    example_data = [
        {"params": {"T": 800, "P": 1}, "score": 0.75},
        {"params": {"T": 900, "P": 2}, "score": 0.82},
        {"params": {"T": 850, "P": 1.5}, "score": 0.88},
    ]
    
    report = await research_reporter.generate_research_report(
        research_topic="Catalyst Discovery via Autonomous Agents",
        experimental_data=example_data
    )
    
    # 保存报告
    with open("autonomous_research_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("\n报告已保存到: autonomous_research_report.txt")
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("所有演示完成!")
    print("=" * 70)
    print("\n摘要:")
    print(f"  1. 催化剂发现: {'成功' if catalyst_report['phases'][3]['result']['goal_achieved'] else '进行中'}")
    print(f"  2. 合成优化: 最佳得分 = {synthesis_report.get('best_score', 0):.3f}")
    print(f"  3. 报告生成: 已完成")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
