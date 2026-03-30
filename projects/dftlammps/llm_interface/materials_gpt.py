"""
材料科学GPT模块 - Materials Science GPT
=====================================
用于材料知识提取、实验设计建议、计算参数推荐和结果解释的LLM接口。

Author: DFT-LAMMPS Team
Date: 2025
"""

import os
import json
import re
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """LLM任务类型枚举"""
    LITERATURE_EXTRACTION = "literature_extraction"
    EXPERIMENTAL_DESIGN = "experimental_design"
    PARAMETER_RECOMMENDATION = "parameter_recommendation"
    RESULT_INTERPRETATION = "result_interpretation"
    REPORT_GENERATION = "report_generation"
    MATERIAL_DISCOVERY = "material_discovery"


@dataclass
class MaterialEntity:
    """材料实体数据结构"""
    name: str
    formula: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    structure: Optional[str] = None
    synthesis_method: Optional[str] = None
    applications: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


@dataclass
class ComputationParameters:
    """计算参数推荐数据结构"""
    task_type: str
    software: str
    functional: Optional[str] = None
    basis_set: Optional[str] = None
    kpoints: Optional[str] = None
    encut: Optional[float] = None
    smearing: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)
    justification: str = ""


@dataclass
class ExperimentDesign:
    """实验设计建议数据结构"""
    title: str
    objective: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcomes: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)


class LLMProvider:
    """LLM提供商基类"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "default"):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model
        self._client = None
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """生成文本的抽象方法"""
        raise NotImplementedError
    
    async def generate_async(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """异步生成文本"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, self.generate, prompt, temperature, max_tokens)


class OpenAIProvider(LLMProvider):
    """OpenAI API提供商"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        super().__init__(api_key, model)
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            logger.warning("openai package not installed, using mock mode")
            self.client = None
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """使用OpenAI API生成文本"""
        if not self.client:
            return self._mock_generate(prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a materials science expert AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._mock_generate(prompt)
    
    def _mock_generate(self, prompt: str) -> str:
        """模拟生成（用于测试或无API密钥时）"""
        logger.info("Using mock generation mode")
        return f"[MOCK RESPONSE] Generated content based on: {prompt[:100]}..."


class LocalLLMProvider(LLMProvider):
    """本地LLM提供商（用于私有化部署）"""
    
    def __init__(self, model_path: str = "", endpoint: str = "http://localhost:8000"):
        super().__init__(model="local")
        self.model_path = model_path
        self.endpoint = endpoint
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """调用本地LLM服务"""
        try:
            import requests
            response = requests.post(
                f"{self.endpoint}/generate",
                json={
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get("text", "")
            else:
                return f"[ERROR] Local LLM service returned {response.status_code}"
        except Exception as e:
            logger.error(f"Local LLM error: {e}")
            return self._mock_generate(prompt)
    
    def _mock_generate(self, prompt: str) -> str:
        return f"[MOCK LOCAL LLM] Generated content based on: {prompt[:100]}..."


class MaterialsGPT:
    """
    材料科学GPT主类
    
    提供文献知识提取、实验设计、参数推荐和报告生成等功能。
    """
    
    def __init__(self, provider: Optional[LLMProvider] = None):
        """
        初始化Materials GPT
        
        Args:
            provider: LLM提供商实例，默认使用OpenAI
        """
        self.provider = provider or OpenAIProvider()
        self.history: List[Dict[str, Any]] = []
        self.templates = self._load_templates()
        logger.info("MaterialsGPT initialized successfully")
    
    def _load_templates(self) -> Dict[str, str]:
        """加载提示词模板"""
        return {
            TaskType.LITERATURE_EXTRACTION.value: """You are a materials science expert. Analyze the following text and extract materials-related information.

TEXT:
{text}

Please extract and format the following:
1. Materials mentioned (with chemical formulas if available)
2. Key properties discussed
3. Synthesis methods described
4. Applications mentioned
5. Important findings and conclusions
6. Computational/theoretical methods used

Format as JSON with these keys: materials, properties, methods, applications, findings""",

            TaskType.EXPERIMENTAL_DESIGN.value: """You are a materials science experimental design expert. Design an experiment based on the following requirements.

OBJECTIVE: {objective}
CONSTRAINTS: {constraints}
MATERIALS: {materials}

Please provide:
1. Detailed experimental procedure (step-by-step)
2. Required materials and equipment
3. Key parameters to control
4. Expected outcomes and how to measure them
5. Potential risks and safety considerations
6. Alternative approaches if primary method fails

Format as structured JSON.""",

            TaskType.PARAMETER_RECOMMENDATION.value: """You are a computational materials science expert. Recommend appropriate simulation parameters.

CALCULATION TYPE: {calc_type}
MATERIAL: {material}
PROPERTY TARGET: {property_target}
SOFTWARE PREFERENCE: {software}

Based on best practices in the field, recommend:
1. Exchange-correlation functional
2. Basis set / pseudopotentials
3. K-point grid density
4. Energy cutoff (eV)
5. Smearing method and width
6. Convergence criteria
7. Any special considerations

Provide justification for each recommendation.""",

            TaskType.RESULT_INTERPRETATION.value: """You are a materials characterization expert. Interpret the following computational/experimental results.

RESULTS DATA:
{results}

CONTEXT:
{context}

Please provide:
1. Key observations from the data
2. Physical/chemical interpretation
3. Comparison with expected values or literature
4. Potential sources of error or uncertainty
5. Recommendations for further analysis
6. Scientific significance of findings

Be specific and cite relevant physics/chemistry principles.""",

            TaskType.REPORT_GENERATION.value: """You are a technical writer specializing in materials science. Generate a comprehensive report.

REPORT TYPE: {report_type}
DATA PROVIDED:
{data}

REQUIREMENTS:
{requirements}

Generate a well-structured report including:
1. Executive summary
2. Introduction/Background
3. Methodology
4. Results and Analysis
5. Discussion
6. Conclusions and Recommendations
7. References (if applicable)

Use professional scientific writing style."""
        }
    
    def extract_knowledge_from_literature(
        self,
        text: str,
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        从文献文本中提取材料知识
        
        Args:
            text: 文献文本内容
            source: 文献来源（可选）
            
        Returns:
            提取的结构化知识
        """
        prompt = self.templates[TaskType.LITERATURE_EXTRACTION.value].format(text=text)
        response = self.provider.generate(prompt, temperature=0.3)
        
        try:
            # 尝试解析JSON响应
            knowledge = self._extract_json(response)
            if source:
                knowledge["source"] = source
            
            # 保存到历史
            self.history.append({
                "task": TaskType.LITERATURE_EXTRACTION.value,
                "source": source,
                "input_length": len(text),
                "result": knowledge
            })
            
            return knowledge
        except Exception as e:
            logger.error(f"Failed to parse knowledge extraction: {e}")
            return {"raw_response": response, "source": source}
    
    def design_experiment(
        self,
        objective: str,
        materials: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        budget: Optional[str] = None
    ) -> ExperimentDesign:
        """
        设计材料实验
        
        Args:
            objective: 实验目标
            materials: 可用材料列表
            constraints: 约束条件
            budget: 预算限制
            
        Returns:
            实验设计方案
        """
        materials_str = ", ".join(materials) if materials else "Not specified"
        constraints_str = json.dumps(constraints, indent=2) if constraints else "None specified"
        
        prompt = self.templates[TaskType.EXPERIMENTAL_DESIGN.value].format(
            objective=objective,
            materials=materials_str,
            constraints=constraints_str
        )
        
        response = self.provider.generate(prompt, temperature=0.7)
        
        try:
            design_data = self._extract_json(response)
            design = ExperimentDesign(
                title=design_data.get("title", "Untitled Experiment"),
                objective=objective,
                steps=design_data.get("steps", []),
                parameters=design_data.get("parameters", {}),
                expected_outcomes=design_data.get("expected_outcomes", []),
                risks=design_data.get("risks", []),
                alternatives=design_data.get("alternatives", [])
            )
            
            self.history.append({
                "task": TaskType.EXPERIMENTAL_DESIGN.value,
                "objective": objective,
                "design": design
            })
            
            return design
        except Exception as e:
            logger.error(f"Failed to parse experiment design: {e}")
            return ExperimentDesign(
                title="Error in Design",
                objective=objective,
                steps=[{"error": response}]
            )
    
    def recommend_computation_parameters(
        self,
        calc_type: str,
        material: str,
        property_target: Optional[str] = None,
        software: str = "VASP",
        precision: str = "normal"
    ) -> ComputationParameters:
        """
        推荐计算参数
        
        Args:
            calc_type: 计算类型（如"band structure", "geometry optimization"等）
            material: 材料名称或化学式
            property_target: 目标性质
            software: 使用的软件
            precision: 精度要求（low/normal/high）
            
        Returns:
            计算参数推荐
        """
        prompt = self.templates[TaskType.PARAMETER_RECOMMENDATION.value].format(
            calc_type=calc_type,
            material=material,
            property_target=property_target or "general properties",
            software=software
        )
        
        response = self.provider.generate(prompt, temperature=0.5)
        
        # 解析参数
        params = self._parse_parameters(response)
        
        comp_params = ComputationParameters(
            task_type=calc_type,
            software=software,
            functional=params.get("functional"),
            basis_set=params.get("basis_set"),
            kpoints=params.get("kpoints"),
            encut=params.get("encut"),
            smearing=params.get("smearing"),
            additional_params=params.get("additional", {}),
            justification=params.get("justification", "")
        )
        
        self.history.append({
            "task": TaskType.PARAMETER_RECOMMENDATION.value,
            "calc_type": calc_type,
            "material": material,
            "params": comp_params
        })
        
        return comp_params
    
    def interpret_results(
        self,
        results: Union[str, Dict[str, Any]],
        context: Optional[str] = None,
        result_type: str = "computational"
    ) -> str:
        """
        解释计算或实验结果
        
        Args:
            results: 结果数据（文本或结构化数据）
            context: 实验背景
            result_type: 结果类型
            
        Returns:
            结果解释
        """
        if isinstance(results, dict):
            results_str = json.dumps(results, indent=2)
        else:
            results_str = results
        
        prompt = self.templates[TaskType.RESULT_INTERPRETATION.value].format(
            results=results_str,
            context=context or "General analysis"
        )
        
        interpretation = self.provider.generate(prompt, temperature=0.6)
        
        self.history.append({
            "task": TaskType.RESULT_INTERPRETATION.value,
            "result_type": result_type,
            "interpretation": interpretation
        })
        
        return interpretation
    
    def generate_report(
        self,
        report_type: str,
        data: Dict[str, Any],
        requirements: Optional[str] = None,
        format_style: str = "scientific"
    ) -> str:
        """
        生成科学报告
        
        Args:
            report_type: 报告类型（"research", "progress", "final"等）
            data: 报告数据
            requirements: 特殊要求
            format_style: 格式风格
            
        Returns:
            生成的报告文本
        """
        data_str = json.dumps(data, indent=2)
        
        prompt = self.templates[TaskType.REPORT_GENERATION.value].format(
            report_type=report_type,
            data=data_str,
            requirements=requirements or "Standard scientific report format"
        )
        
        report = self.provider.generate(prompt, temperature=0.7, max_tokens=4000)
        
        self.history.append({
            "task": TaskType.REPORT_GENERATION.value,
            "report_type": report_type,
            "report_length": len(report)
        })
        
        return report
    
    def discover_new_materials(
        self,
        target_properties: List[str],
        constraints: Optional[Dict[str, Any]] = None,
        known_materials: Optional[List[str]] = None
    ) -> List[MaterialEntity]:
        """
        基于性质要求发现潜在新材料
        
        Args:
            target_properties: 目标性质列表
            constraints: 约束条件（成本、毒性等）
            known_materials: 已知类似材料
            
        Returns:
            潜在新材料列表
        """
        constraints_str = json.dumps(constraints) if constraints else "No specific constraints"
        known_str = ", ".join(known_materials) if known_materials else "None provided"
        
        prompt = f"""You are a materials discovery expert. Based on the target properties and constraints, suggest potential new materials to investigate.

TARGET PROPERTIES: {', '.join(target_properties)}
CONSTRAINTS: {constraints_str}
KNOWN SIMILAR MATERIALS: {known_str}

Please suggest 3-5 potential materials that might exhibit the desired properties. For each material:
1. Propose a chemical formula or composition
2. Explain why it might have the target properties
3. Suggest a synthesis approach
4. Identify key characterization needed
5. Note any potential challenges

Format as JSON array with fields: name, formula, predicted_properties, synthesis_suggestion, characterization_needed, challenges"""
        
        response = self.provider.generate(prompt, temperature=0.8)
        
        try:
            materials_data = self._extract_json(response)
            if not isinstance(materials_data, list):
                materials_data = [materials_data]
            
            materials = []
            for m in materials_data:
                material = MaterialEntity(
                    name=m.get("name", "Unknown"),
                    formula=m.get("formula", ""),
                    properties=m.get("predicted_properties", {}),
                    synthesis_method=m.get("synthesis_suggestion"),
                    applications=target_properties
                )
                materials.append(material)
            
            self.history.append({
                "task": TaskType.MATERIAL_DISCOVERY.value,
                "target_properties": target_properties,
                "suggested_materials": [m.name for m in materials]
            })
            
            return materials
        except Exception as e:
            logger.error(f"Failed to parse material discovery: {e}")
            return []
    
    def _extract_json(self, text: str) -> Any:
        """从文本中提取JSON"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except:
            pass
        
        # 尝试从代码块中提取
        json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # 尝试从数组代码块中提取
        json_match = re.search(r'```(?:json)?\s*(\[.*\])\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # 返回原始文本
        return {"raw_text": text}
    
    def _parse_parameters(self, response: str) -> Dict[str, Any]:
        """解析参数推荐响应"""
        params = {}
        
        # 尝试提取JSON
        try:
            data = self._extract_json(response)
            if isinstance(data, dict):
                return data
        except:
            pass
        
        # 手动解析关键参数
        patterns = {
            "functional": r"(?:functional|exchange-correlation)[\s:]+([^\n,]+)",
            "basis_set": r"(?:basis set|pseudopotential)[\s:]+([^\n,]+)",
            "kpoints": r"(?:k-point|kpoint)[\s:]+([^\n,]+)",
            "encut": r"(?:energy cutoff|ENCUT)[\s:]+(\d+)",
            "smearing": r"(?:smearing|broadening)[\s:]+([^\n,]+)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if key == "encut":
                    try:
                        value = int(value)
                    except:
                        pass
                params[key] = value
        
        params["justification"] = response[:500] + "..." if len(response) > 500 else response
        
        return params
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取操作历史"""
        return self.history
    
    def clear_history(self):
        """清除历史记录"""
        self.history = []
    
    def export_knowledge_base(self, filepath: str):
        """导出知识库到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, default=str)
        logger.info(f"Knowledge base exported to {filepath}")


class LiteratureMiner:
    """文献挖掘器 - 批量处理文献"""
    
    def __init__(self, gpt: MaterialsGPT):
        self.gpt = gpt
        self.knowledge_base: List[Dict[str, Any]] = []
    
    def mine_from_texts(
        self,
        texts: List[Dict[str, str]],
        batch_size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        批量挖掘文献知识
        
        Args:
            texts: 文献列表，每项包含text和source
            batch_size: 批处理大小
            
        Returns:
            提取的知识列表
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            for item in batch:
                knowledge = self.gpt.extract_knowledge_from_literature(
                    text=item["text"],
                    source=item.get("source")
                )
                results.append({
                    "source": item.get("source"),
                    "knowledge": knowledge
                })
        
        self.knowledge_base.extend(results)
        return results
    
    def find_materials_by_property(
        self,
        property_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """根据性质搜索材料"""
        matches = []
        
        for entry in self.knowledge_base:
            knowledge = entry.get("knowledge", {})
            properties = knowledge.get("properties", {})
            
            if property_name.lower() in str(properties).lower():
                matches.append(entry)
        
        return matches
    
    def export_to_graph_format(self, filepath: str):
        """导出为知识图谱格式"""
        nodes = []
        edges = []
        
        for entry in self.knowledge_base:
            source = entry.get("source", "unknown")
            knowledge = entry.get("knowledge", {})
            
            # 创建材料节点
            materials = knowledge.get("materials", [])
            for mat in materials:
                node_id = f"mat_{len(nodes)}"
                nodes.append({
                    "id": node_id,
                    "type": "material",
                    "name": mat if isinstance(mat, str) else mat.get("name", "unknown"),
                    "source": source
                })
        
        graph_data = {"nodes": nodes, "edges": edges}
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
        
        logger.info(f"Graph data exported to {filepath}")


# 便捷函数
def create_default_gpt() -> MaterialsGPT:
    """创建默认的MaterialsGPT实例"""
    provider = OpenAIProvider()
    return MaterialsGPT(provider)


def quick_extract(text: str, source: Optional[str] = None) -> Dict[str, Any]:
    """快速提取文献知识"""
    gpt = create_default_gpt()
    return gpt.extract_knowledge_from_literature(text, source)


def quick_design_experiment(objective: str, **kwargs) -> ExperimentDesign:
    """快速设计实验"""
    gpt = create_default_gpt()
    return gpt.design_experiment(objective, **kwargs)


def quick_interpret_results(results: Union[str, Dict], context: Optional[str] = None) -> str:
    """快速解释结果"""
    gpt = create_default_gpt()
    return gpt.interpret_results(results, context)


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("MaterialsGPT Test")
    print("=" * 60)
    
    # 创建实例（使用mock模式）
    gpt = MaterialsGPT()
    
    # 测试文献提取
    sample_text = """
    We synthesized a new perovskite material CsPbI3 using solution processing.
    The material exhibits a band gap of 1.73 eV and high thermal stability up to 300°C.
    XRD analysis confirmed the cubic structure with lattice parameter a = 6.1 Å.
    """
    
    print("\n1. Testing literature extraction...")
    knowledge = gpt.extract_knowledge_from_literature(sample_text, "Sample Paper 2024")
    print(f"Extracted knowledge keys: {list(knowledge.keys())}")
    
    # 测试参数推荐
    print("\n2. Testing parameter recommendation...")
    params = gpt.recommend_computation_parameters(
        calc_type="band structure",
        material="CsPbI3",
        software="VASP"
    )
    print(f"Recommended software: {params.software}")
    print(f"Task type: {params.task_type}")
    
    # 测试实验设计
    print("\n3. Testing experiment design...")
    design = gpt.design_experiment(
        objective="Synthesize stable perovskite solar cell material",
        materials=["CsI", "PbI2", "MABr"],
        constraints={"temperature": "< 200°C", "atmosphere": "N2"}
    )
    print(f"Design title: {design.title}")
    print(f"Number of steps: {len(design.steps)}")
    
    # 测试报告生成
    print("\n4. Testing report generation...")
    report_data = {
        "experiment": "Perovskite synthesis",
        "results": {"yield": "85%", "purity": "98%"},
        "methods": ["Solution processing", "Annealing"]
    }
    report = gpt.generate_report("experimental", report_data)
    print(f"Report length: {len(report)} characters")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
