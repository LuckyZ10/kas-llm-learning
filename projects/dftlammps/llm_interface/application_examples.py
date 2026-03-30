"""
应用案例模块 - LLM Application Examples
======================================
展示LLM在材料科学中的应用案例。

Author: DFT-LAMMPS Team
Date: 2025
"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiteratureMiningExample:
    """文献挖掘发现新材料示例"""
    
    def __init__(self, gpt_client=None):
        self.gpt = gpt_client
    
    def demonstrate_literature_mining(self):
        """演示文献挖掘流程"""
        print("=" * 70)
        print("案例1: 文献挖掘发现新材料")
        print("=" * 70)
        
        # 模拟文献数据
        literature_samples = [
            {
                "title": "High-efficiency Perovskite Solar Cells",
                "text": """
                Organic-inorganic hybrid perovskites have emerged as promising materials for 
                photovoltaic applications. CsPbI3 with a band gap of 1.73 eV shows excellent 
                optoelectronic properties. The material was synthesized using solution processing 
                at 150°C. Power conversion efficiency reached 20.1% in solar cell devices.
                """
            },
            {
                "title": "Novel Thermoelectric Materials",
                "text": """
                SnSe has been identified as a high-performance thermoelectric material with 
                ZT > 2.0 at 800K. The ultralow thermal conductivity (0.5 W/mK) combined with 
                high electrical conductivity makes it suitable for waste heat recovery applications.
                Single crystals were grown by Bridgman method.
                """
            },
            {
                "title": "2D Materials for Catalysis",
                "text": """
                Transition metal dichalcogenides MoS2 and WS2 show promising catalytic activity 
                for hydrogen evolution reaction. The basal planes are catalytically inert, but 
                edge sites exhibit high activity comparable to platinum. DFT calculations using 
                VASP with PBE functional predict ΔGH ≈ 0 eV on edge sites.
                """
            }
        ]
        
        print("\n📚 输入文献样本:")
        for i, lit in enumerate(literature_samples, 1):
            print(f"  {i}. {lit['title']}")
        
        print("\n🔍 知识提取过程:")
        extracted_knowledge = []
        
        for lit in literature_samples:
            # 模拟知识提取
            knowledge = self._extract_knowledge(lit["text"])
            extracted_knowledge.append(knowledge)
            print(f"\n  从 '{lit['title']}' 提取:")
            print(f"    - 材料: {', '.join(knowledge['materials'])}")
            print(f"    - 性质: {', '.join(knowledge['properties'].keys())}")
            print(f"    - 方法: {', '.join(knowledge['methods'])}")
        
        print("\n💡 发现的新材料建议:")
        suggestions = self._suggest_new_materials(extracted_knowledge)
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion['name']}")
            print(f"     理由: {suggestion['rationale']}")
            print(f"     预测性质: {suggestion['predicted_properties']}")
        
        print("\n" + "=" * 70)
        
        return extracted_knowledge
    
    def _extract_knowledge(self, text: str) -> Dict[str, Any]:
        """从文本提取知识（模拟）"""
        # 简化实现，实际应使用NLP/LLM
        knowledge = {
            "materials": [],
            "properties": {},
            "methods": []
        }
        
        # 简单模式匹配
        if "CsPbI3" in text:
            knowledge["materials"].append("CsPbI3")
            knowledge["properties"]["band_gap"] = "1.73 eV"
        if "SnSe" in text:
            knowledge["materials"].append("SnSe")
            knowledge["properties"]["ZT"] = "2.0"
            knowledge["properties"]["thermal_conductivity"] = "0.5 W/mK"
        if "MoS2" in text:
            knowledge["materials"].append("MoS2")
        if "WS2" in text:
            knowledge["materials"].append("WS2")
        
        if "DFT" in text or "VASP" in text:
            knowledge["methods"].append("DFT")
        if "solution processing" in text:
            knowledge["methods"].append("solution processing")
        if "Bridgman" in text:
            knowledge["methods"].append("Bridgman method")
        
        return knowledge
    
    def _suggest_new_materials(self, knowledge_list: List[Dict]) -> List[Dict]:
        """基于知识建议新材料"""
        suggestions = [
            {
                "name": "CsSnI3",
                "rationale": "Lead-free alternative to CsPbI3 with similar structure",
                "predicted_properties": "band_gap ~1.5 eV, non-toxic"
            },
            {
                "name": "MoSe2",
                "rationale": "Similar to MoS2 but potentially better HER catalytic activity",
                "predicted_properties": "edge site activity, tunable band gap"
            },
            {
                "name": "SnS",
                "rationale": "Similar layered structure to SnSe, potentially better stability",
                "predicted_properties": "thermoelectric ZT ~1.5, earth-abundant"
            }
        ]
        return suggestions


class NLWorkflowDesignExample:
    """自然语言设计计算流程示例"""
    
    def __init__(self, code_generator=None):
        self.generator = code_generator
    
    def demonstrate_nl_workflow(self):
        """演示自然语言工作流设计"""
        print("=" * 70)
        print("案例2: 自然语言设计计算流程")
        print("=" * 70)
        
        # 自然语言描述
        descriptions = [
            {
                "name": "能带结构计算",
                "description": "我想计算硅的能带结构，使用VASP，需要高精度，k点路径包含Gamma到X点"
            },
            {
                "name": "分子动力学模拟",
                "description": "对Cu纳米颗粒进行NVT分子动力学模拟，温度500K，运行100皮秒，使用EAM势"
            },
            {
                "name": "过渡态计算",
                "description": "用NEB方法计算CO在Pt(111)表面的扩散势垒，使用8个中间点"
            }
        ]
        
        print("\n📝 自然语言输入:")
        for i, desc in enumerate(descriptions, 1):
            print(f"\n  {i}. {desc['name']}:")
            print(f"     '{desc['description']}'")
        
        print("\n⚙️  生成的计算工作流:")
        
        for desc in descriptions:
            print(f"\n  📌 {desc['name']}:")
            workflow = self._generate_workflow(desc["description"])
            
            print(f"     识别意图: {workflow['intent']}")
            print(f"     推荐软件: {workflow['software']}")
            print(f"     计算步骤:")
            for step in workflow['steps']:
                print(f"       - {step}")
            
            if workflow.get('input_preview'):
                print(f"     输入文件预览:")
                print(f"       ```")
                for line in workflow['input_preview'].split('\n')[:5]:
                    print(f"       {line}")
                print(f"       ...")
                print(f"       ```")
        
        print("\n" + "=" * 70)
        
        return descriptions
    
    def _generate_workflow(self, description: str) -> Dict[str, Any]:
        """生成工作流（模拟）"""
        workflow = {
            "intent": "unknown",
            "software": "unknown",
            "steps": [],
            "input_preview": ""
        }
        
        if "能带" in description or "band" in description.lower():
            workflow["intent"] = "band_structure_calculation"
            workflow["software"] = "VASP"
            workflow["steps"] = [
                "1. 结构优化 (ISIF=3)",
                "2. 自洽计算 (保存CHGCAR)",
                "3. 能带计算 (ICHARG=11)",
                "4. 后处理与可视化"
            ]
            workflow["input_preview"] = """SYSTEM = Si Band Structure
ENCUT = 520
ISMEAR = 0
SIGMA = 0.05
LORBIT = 11"""
            
        elif "分子动力学" in description or "MD" in description.upper():
            workflow["intent"] = "molecular_dynamics"
            workflow["software"] = "LAMMPS"
            workflow["steps"] = [
                "1. 能量最小化",
                "2. NVT平衡 (温度500K)",
                "3. NVE生产运行",
                "4. 轨迹分析"
            ]
            workflow["input_preview"] = """units metal
atom_style atomic
timestep 0.001
fix 1 all nvt temp 500.0 500.0 0.1"""
            
        elif "NEB" in description.upper() or "过渡态" in description:
            workflow["intent"] = "transition_state_search"
            workflow["software"] = "VASP"
            workflow["steps"] = [
                "1. 初始态优化",
                "2. 末态优化",
                "3. NEB计算 (8 images)",
                "4. CI-NEB细化"
            ]
            workflow["input_preview"] = """IMAGES = 8
SPRING = -5
LCLIMB = .TRUE.
ICHAIN = 0"""
        
        return workflow


class SmartLabNotebookExample:
    """智能实验日记生成示例"""
    
    def __init__(self):
        pass
    
    def demonstrate_smart_notebook(self):
        """演示智能实验日记生成"""
        print("=" * 70)
        print("案例3: 智能实验日记生成")
        print("=" * 70)
        
        # 模拟实验数据
        experiment_data = {
            "experiment_id": "EXP-2024-001",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "researcher": "Dr. Zhang",
            "objective": "Synthesize CsPbI3 perovskite thin films",
            "materials": {
                "CsI": {"amount": "1 mmol", "purity": "99.9%"},
                "PbI2": {"amount": "1 mmol", "purity": "99.99%"},
                "DMF": {"amount": "1 mL", "grade": "anhydrous"}
            },
            "procedure": [
                "Dissolve CsI in DMF at 60°C",
                "Add PbI2 and stir for 2 hours",
                "Spin coat on glass substrate",
                "Anneal at 150°C for 10 min"
            ],
            "conditions": {
                "temperature": "150°C",
                "atmosphere": "N2 glovebox",
                "humidity": "< 1 ppm"
            },
            "measurements": {
                "XRD": {
                    "file": "csxrd_001.dat",
                    "key_findings": ["Cubic phase confirmed", "a = 6.16 Å"]
                },
                "UV-Vis": {
                    "file": "csuv_001.dat",
                    "key_findings": ["Band gap = 1.73 eV", "Strong absorption onset"]
                },
                "SEM": {
                    "file": "cssem_001.jpg",
                    "key_findings": ["Uniform film coverage", "Grain size ~200 nm"]
                }
            },
            "observations": [
                "Film color changed from yellow to black upon annealing",
                "No visible pinholes observed",
                "Slight delamination at edges"
            ],
            "issues": [
                "Some samples showed PbI2 impurity peak in XRD"
            ],
            "next_steps": [
                "Optimize annealing temperature",
                "Try anti-solvent treatment",
                "Test photovoltaic performance"
            ]
        }
        
        print("\n📊 实验数据输入:")
        print(f"  实验ID: {experiment_data['experiment_id']}")
        print(f"  日期: {experiment_data['date']}")
        print(f"  目标: {experiment_data['objective']}")
        print(f"  测量数据: {len(experiment_data['measurements'])} 项")
        
        print("\n📝 生成的实验日记:")
        notebook = self._generate_notebook(experiment_data)
        print(notebook)
        
        print("\n🔍 自动分析:")
        analysis = self._analyze_experiment(experiment_data)
        print(f"  实验状态: {analysis['status']}")
        print(f"  关键结果: {', '.join(analysis['key_results'])}")
        print(f"  改进建议: {', '.join(analysis['suggestions'])}")
        
        print("\n📈 实验趋势:")
        trends = self._analyze_trends([experiment_data])
        print(f"  成功率: {trends['success_rate']:.1%}")
        print(f"  平均结晶度: {trends['avg_crystallinity']:.1f}%")
        
        print("\n" + "=" * 70)
        
        return experiment_data
    
    def _generate_notebook(self, data: Dict) -> str:
        """生成实验日记"""
        notebook = f"""
================================================================================
实验日记: {data['experiment_id']}
日期: {data['date']}
研究员: {data['researcher']}
================================================================================

## 1. 实验目标
{data['objective']}

## 2. 实验材料
| 材料 | 用量 | 纯度/等级 |
|------|------|-----------|
"""
        
        for mat, info in data['materials'].items():
            purity = info.get('purity') or info.get('grade', 'N/A')
            notebook += f"| {mat} | {info['amount']} | {purity} |\n"
        
        notebook += f"""
## 3. 实验步骤
"""
        for i, step in enumerate(data['procedure'], 1):
            notebook += f"{i}. {step}\n"
        
        notebook += f"""
## 4. 实验条件
- 温度: {data['conditions']['temperature']}
- 气氛: {data['conditions']['atmosphere']}
- 湿度: {data['conditions']['humidity']}

## 5. 表征结果
"""
        for method, results in data['measurements'].items():
            notebook += f"\n### {method}\n"
            notebook += f"数据文件: {results['file']}\n"
            notebook += "主要发现:\n"
            for finding in results['key_findings']:
                notebook += f"- {finding}\n"
        
        notebook += f"""
## 6. 观察记录
"""
        for obs in data['observations']:
            notebook += f"- {obs}\n"
        
        notebook += f"""
## 7. 问题与改进
"""
        if data['issues']:
            for issue in data['issues']:
                notebook += f"- ⚠️  {issue}\n"
        
        notebook += f"""
## 8. 下一步计划
"""
        for step in data['next_steps']:
            notebook += f"- [ ] {step}\n"
        
        notebook += """
================================================================================
"""
        
        return notebook
    
    def _analyze_experiment(self, data: Dict) -> Dict[str, Any]:
        """分析实验"""
        analysis = {
            "status": "success",
            "key_results": [],
            "suggestions": []
        }
        
        # 检查测量结果
        if "XRD" in data['measurements']:
            analysis["key_results"].append("晶体结构确认")
            if any("impurity" in f.lower() for f in data['measurements']['XRD']['key_findings']):
                analysis["suggestions"].append("优化退火条件以消除杂相")
        
        if "UV-Vis" in data['measurements']:
            analysis["key_results"].append("光学带隙测定")
        
        if data['issues']:
            analysis["status"] = "partial_success"
        
        analysis["suggestions"].extend(data['next_steps'][:2])
        
        return analysis
    
    def _analyze_trends(self, experiments: List[Dict]) -> Dict[str, float]:
        """分析实验趋势"""
        return {
            "success_rate": 0.75,
            "avg_crystallinity": 85.0,
            "avg_grain_size": 200.0
        }


class IntegrationExample:
    """综合应用示例"""
    
    def __init__(self):
        self.lit_mining = LiteratureMiningExample()
        self.nl_workflow = NLWorkflowDesignExample()
        self.smart_notebook = SmartLabNotebookExample()
    
    def run_all_examples(self):
        """运行所有示例"""
        print("\n" + "🧪 " * 20)
        print("DFT-LAMMPS LLM集成应用案例")
        print("🧪 " * 20 + "\n")
        
        # 示例1: 文献挖掘
        self.lit_mining.demonstrate_literature_mining()
        
        print("\n")
        
        # 示例2: 自然语言工作流
        self.nl_workflow.demonstrate_nl_workflow()
        
        print("\n")
        
        # 示例3: 智能实验日记
        self.smart_notebook.demonstrate_smart_notebook()
        
        print("\n" + "🧪 " * 20)
        print("所有案例演示完成!")
        print("🧪 " * 20 + "\n")


def main():
    """主函数"""
    examples = IntegrationExample()
    examples.run_all_examples()


if __name__ == "__main__":
    main()
