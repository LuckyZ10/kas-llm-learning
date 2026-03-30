"""
llm_materials_design.py
大语言模型材料设计

利用GPT-4、Llama等大语言模型直接进行材料设计和分子生成。
支持: 文本到材料、材料描述生成、性质预测、合成路径规划等。

References:
- Jablonka et al. (2023) "14 Examples of How LLMs Can Transform Materials Science"
- Zaki et al. (2024) "Large Language Models for Chemistry and Materials Science"
- 2024-2025: GPT-4 + 多模态用于晶体结构理解
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import json
import re
from collections import defaultdict
import numpy as np


@dataclass
class MaterialDescription:
    """材料描述数据结构"""
    name: str
    formula: str
    crystal_system: str
    space_group: str
    properties: Dict[str, float]
    applications: List[str]
    synthesis_method: str
    description: str


@dataclass  
class LLMResponse:
    """LLM响应结构"""
    text: str
    structured_data: Optional[Dict] = None
    confidence: float = 0.0
    tokens_used: int = 0


class MaterialsLLMInterface:
    """
    材料科学大语言模型接口
    
    封装与各种LLM的交互, 提供材料科学专用的提示工程
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        
        # 材料科学系统提示
        self.system_prompt = """You are an expert materials scientist AI assistant. 
Your knowledge spans:
- Crystallography and solid-state physics
- Computational materials science (DFT, MD, ML potentials)
- Materials synthesis and characterization
- Battery materials, catalysts, semiconductors, superconductors
- Structure-property relationships

When providing information:
1. Be scientifically accurate and cite relevant theories
2. Provide specific chemical formulas and crystal structures when applicable
3. Include quantitative estimates when possible
4. Suggest relevant characterization methods
5. Consider practical synthesis feasibility

You can output structured data in JSON format when requested."""
        
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        output_format: str = "text"
    ) -> LLMResponse:
        """
        生成LLM响应
        
        Args:
            prompt: 用户提示
            system_prompt: 可选的系统提示覆盖
            output_format: 'text', 'json', 'structured'
        """
        # 这里模拟LLM调用 - 实际使用时连接OpenAI/Anthropic等API
        # 为了演示, 我们实现一个基于规则的模拟响应器
        
        if self.model_name.startswith("mock"):
            return self._mock_generate(prompt, output_format)
        
        # 实际API调用 (需要安装openai等库)
        try:
            return self._call_openai_api(prompt, system_prompt, output_format)
        except Exception as e:
            print(f"API call failed: {e}, falling back to mock")
            return self._mock_generate(prompt, output_format)
    
    def _mock_generate(self, prompt: str, output_format: str) -> LLMResponse:
        """模拟LLM响应 - 用于演示和测试"""
        
        # 解析提示意图
        if "design" in prompt.lower() or "suggest" in prompt.lower():
            response = self._mock_design_material(prompt)
        elif "property" in prompt.lower() or "predict" in prompt.lower():
            response = self._mock_predict_properties(prompt)
        elif "synthesis" in prompt.lower():
            response = self._mock_synthesis_route(prompt)
        elif "compare" in prompt.lower():
            response = self._mock_compare_materials(prompt)
        else:
            response = self._mock_general_response(prompt)
        
        # 解析结构化数据
        structured = None
        if output_format == "json":
            try:
                # 尝试从响应中提取JSON
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    structured = json.loads(json_match.group())
            except:
                pass
        
        return LLMResponse(
            text=response,
            structured_data=structured,
            confidence=0.75 + np.random.rand() * 0.2,
            tokens_used=len(response.split())
        )
    
    def _mock_design_material(self, prompt: str) -> str:
        """模拟材料设计响应"""
        materials_db = {
            "battery cathode": {
                "formula": "LiNi₀.₅Mn₁.₅O₄",
                "structure": "Spinel Fd-3m",
                "properties": {"capacity": "147 mAh/g", "voltage": "4.7 V", "stability": "good"},
                "rationale": "High voltage spinel structure with stable Mn framework"
            },
            "catalyst": {
                "formula": "Fe-N-C",
                "structure": "Porous carbon with Fe-Nx sites",
                "properties": {"ORR_activity": "high", "stability": "excellent"},
                "rationale": "Single-atom Fe sites provide optimal binding energy"
            },
            "superconductor": {
                "formula": "La₃Ni₂O₇",
                "structure": "Ruddlesden-Popper tetragonal",
                "properties": {"Tc": "80 K at high pressure", "mechanism": "bilayer coupling"},
                "rationale": "Recent discovery of nickelate high-Tc superconductor"
            }
        }
        
        # 匹配关键词
        for key, mat in materials_db.items():
            if key in prompt.lower():
                return f"""Based on your requirements for a {key}, I suggest:

Material: {mat['formula']}
Crystal Structure: {mat['structure']}

Key Properties:
{chr(10).join(f"- {k}: {v}" for k, v in mat['properties'].items())}

Design Rationale: {mat['rationale']}

Synthesis Consideration: Solid-state reaction at 900°C followed by annealing.

{{"formula": "{mat['formula']}", "application": "{key}", "confidence": 0.85}}"""
        
        return "I suggest exploring perovskite oxides for this application due to their structural flexibility and tunable properties."
    
    def _mock_predict_properties(self, prompt: str) -> str:
        """模拟性质预测响应"""
        # 从提示中提取化学式
        formula_match = re.search(r'([A-Z][a-z]?\d*)+', prompt)
        formula = formula_match.group() if formula_match else "Unknown"
        
        return f"""Property Prediction for {formula}:

Electronic Properties:
- Band gap: ~2.5 eV (DFT-PBE estimate)
- DOS at Fermi level: Moderate
- Expected conductivity: Semiconductor

Mechanical Properties:
- Bulk modulus: ~150 GPa (estimated)
- Elastic anisotropy: Low to moderate

Thermal Properties:
- Thermal conductivity: 5-10 W/mK (estimated)
- Debye temperature: ~400 K

Note: These are estimates based on similar compounds. DFT calculations recommended for accuracy.

{{"formula": "{formula}", "band_gap": 2.5, "bulk_modulus": 150, "confidence": 0.7}}"""
    
    def _mock_synthesis_route(self, prompt: str) -> str:
        """模拟合成路径响应"""
        return """Proposed Synthesis Route:

1. Precursor Preparation:
   - Use high-purity metal nitrates or acetates
   - Stoichiometric ratio with 2% excess of volatile cations

2. Solution Processing:
   - Dissolve in deionized water or citric acid solution
   - Add chelating agent (citric acid or EDTA)
   - pH adjustment to 7-8 with ammonia

3. Gel Formation:
   - Evaporate at 80°C with stirring
   - Form transparent gel

4. Decomposition:
   - Heat to 350°C at 2°C/min
   - Hold for 4 hours
   - Organic decomposition

5. Crystallization:
   - Calcine at target temperature (800-1000°C)
   - Hold time: 12-24 hours
   - Heating/cooling rate: 5°C/min

6. Post-treatment:
   - Pelletizing and sintering if needed
   - Controlled atmosphere if oxygen-sensitive

Expected phase purity: >95%
Particle size: 1-5 μm

{"synthesis_method": "sol-gel", "temperature_range": "350-1000°C", "expected_purity": 0.95}"""
    
    def _mock_compare_materials(self, prompt: str) -> str:
        """模拟材料对比响应"""
        return """Comparative Analysis:

LiCoO₂ (LCO):
- Advantages: High energy density, good cycle life
- Disadvantages: Cobalt cost, safety concerns
- Best for: Premium electronics

LiFePO₄ (LFP):
- Advantages: Excellent safety, low cost, long cycle life
- Disadvantages: Lower energy density
- Best for: Electric vehicles, grid storage

LiNixMnyCo2O2 (NMC):
- Advantages: Balanced performance, tunable composition
- Disadvantages: Thermal stability concerns at high Ni
- Best for: Long-range EVs

Recommendation: For your application requiring [high power + safety], consider NMC-532 or LFP depending on energy density requirements.

{"recommendation": "NMC-532", "primary_factor": "energy_density", "confidence": 0.8}"""
    
    def _mock_general_response(self, prompt: str) -> str:
        """通用响应"""
        return f"""Analysis complete. Based on current materials science knowledge and recent literature:

The query involves considerations of structure-property relationships, synthesis feasibility, and application requirements. Key factors to consider:

1. Structural stability under operating conditions
2. Electronic properties alignment with application needs
3. Scalability of synthesis methods
4. Cost and sustainability factors

For a more detailed analysis, please provide specific requirements such as target properties, operating conditions, and constraints.

{{"response_type": "general", "requires_more_info": true}}"""
    
    def _call_openai_api(
        self,
        prompt: str,
        system_prompt: Optional[str],
        output_format: str
    ) -> LLMResponse:
        """调用OpenAI API - 需要安装openai库"""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt or self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            text = response.choices[0].message.content
            tokens = response.usage.total_tokens
            
            # 解析JSON
            structured = None
            if output_format == "json":
                try:
                    structured = json.loads(text)
                except:
                    pass
            
            return LLMResponse(
                text=text,
                structured_data=structured,
                confidence=0.9,
                tokens_used=tokens
            )
        except ImportError:
            raise RuntimeError("openai package not installed")


class StructureToTextEncoder(nn.Module):
    """
    结构到文本编码器
    
    将晶体结构编码为LLM可理解的文本描述
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        max_atoms: int = 100
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_atoms = max_atoms
        
        # 原子特征嵌入
        self.atom_embed = nn.Embedding(100, hidden_dim)
        
        # 位置编码器
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 结构特征提取
        self.structure_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                batch_first=True
            ),
            num_layers=4
        )
        
    def encode(self, structure: Dict[str, torch.Tensor]) -> str:
        """
        将结构编码为文本描述
        
        Returns: 自然语言描述
        """
        atom_types = structure['atom_types']
        frac_coords = structure['frac_coords']
        lengths = structure.get('lengths', torch.tensor([10., 10., 10.]))
        angles = structure.get('angles', torch.tensor([90., 90., 90.]))
        
        # 构建描述
        elements = self._get_element_names(atom_types)
        formula = self._formula_from_elements(elements)
        
        # 结构特征
        crystal_system = self._determine_crystal_system(angles)
        space_group = "Unknown"  # 简化
        
        # 计算密度
        volume = self._compute_volume(lengths, angles)
        density = self._estimate_density(elements, volume)
        
        description = f"""Crystal Structure Description:

Chemical Formula: {formula}
Number of Atoms: {len(atom_types)}
Crystal System: {crystal_system}
Unit Cell Volume: {volume:.2f} Å³
Estimated Density: {density:.3f} g/cm³

Lattice Parameters:
a = {lengths[0]:.4f} Å, b = {lengths[1]:.4f} Å, c = {lengths[2]:.4f} Å
α = {angles[0]:.2f}°, β = {angles[1]:.2f}°, γ = {angles[2]:.2f}°

Atomic Positions (fractional coordinates):
"""
        
        for i, (elem, coords) in enumerate(zip(elements, frac_coords)):
            description += f"  {i+1}. {elem}: ({coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f})\n"
        
        return description
    
    def _get_element_names(self, atom_types: torch.Tensor) -> List[str]:
        """原子序数转元素名"""
        element_symbols = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn'
        ]
        return [element_symbols.get(int(z), 'X') for z in atom_types]
    
    def _formula_from_elements(self, elements: List[str]) -> str:
        """从元素列表生成化学式"""
        counts = defaultdict(int)
        for e in elements:
            counts[e] += 1
        
        formula = ""
        for elem, count in sorted(counts.items()):
            if count == 1:
                formula += elem
            else:
                formula += f"{elem}{count}"
        return formula
    
    def _determine_crystal_system(self, angles: torch.Tensor) -> str:
        """根据角度判断晶系"""
        a, b, c = angles
        if torch.allclose(angles, torch.tensor([90., 90., 90.]), atol=1):
            return "Cubic or Orthorhombic"
        elif torch.allclose(a, b) and abs(c - 120) < 1:
            return "Hexagonal"
        elif not torch.allclose(angles, torch.tensor([90., 90., 90.]), atol=1):
            return "Triclinic or Monoclinic"
        return "Unknown"
    
    def _compute_volume(
        self,
        lengths: torch.Tensor,
        angles: torch.Tensor
    ) -> float:
        """计算晶胞体积"""
        a, b, c = lengths
        alpha, beta, gamma = angles * np.pi / 180
        
        volume = a * b * c * torch.sqrt(
            1 - torch.cos(alpha)**2 - torch.cos(beta)**2 - torch.cos(gamma)**2
            + 2 * torch.cos(alpha) * torch.cos(beta) * torch.cos(gamma)
        )
        return volume.item()
    
    def _estimate_density(self, elements: List[str], volume: float) -> float:
        """估算密度"""
        # 简化估算
        avg_mass = 40  # g/mol 平均原子量
        num_atoms = len(elements)
        mass = num_atoms * avg_mass / 6.022e23  # g
        volume_cm3 = volume * 1e-24  # cm³
        return mass / volume_cm3 if volume_cm3 > 0 else 0


class TextToStructureDecoder(nn.Module):
    """
    文本到结构解码器
    
    将LLM生成的文本描述解析为结构化数据
    """
    
    def __init__(self):
        super().__init__()
        
    def parse_formula(self, formula: str) -> Dict[str, int]:
        """解析化学式"""
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)
        
        composition = {}
        for elem, count in matches:
            composition[elem] = int(count) if count else 1
        
        return composition
    
    def parse_lattice_params(self, text: str) -> Dict[str, float]:
        """从文本解析晶格参数"""
        params = {}
        
        # 匹配 a, b, c
        for param in ['a', 'b', 'c']:
            match = re.search(rf'{param}\s*=\s*([\d.]+)', text)
            if match:
                params[param] = float(match.group(1))
        
        # 匹配 angles
        for angle in ['alpha', 'beta', 'gamma']:
            match = re.search(rf'{angle}\s*=\s*([\d.]+)', text, re.IGNORECASE)
            if match:
                params[angle] = float(match.group(1))
        
        return params
    
    def parse_positions(self, text: str) -> List[Tuple[str, float, float, float]]:
        """解析原子位置"""
        positions = []
        
        # 匹配位置行
        pattern = r'(\w+)\s*[:\-]?\s*\(?\s*([\d.-]+)[,\s]+([\d.-]+)[,\s]+([\d.-]+)\s*\)?'
        matches = re.findall(pattern, text)
        
        for elem, x, y, z in matches:
            positions.append((elem, float(x), float(y), float(z)))
        
        return positions


class LLMMaterialsAgent:
    """
    LLM材料设计智能体
    
    整合LLM能力进行端到端材料设计工作流
    """
    
    def __init__(
        self,
        llm_interface: MaterialsLLMInterface,
        structure_encoder: Optional[StructureToTextEncoder] = None
    ):
        self.llm = llm_interface
        self.encoder = structure_encoder or StructureToTextEncoder()
        self.decoder = TextToStructureDecoder()
        
        # 对话历史
        self.conversation_history = []
        
    def design_for_application(
        self,
        application: str,
        constraints: Optional[Dict] = None
    ) -> MaterialDescription:
        """
        为特定应用设计材料
        
        Args:
            application: 应用场景描述
            constraints: 约束条件 (成本、环境、温度等)
        """
        # 构建提示
        prompt = f"""Design a new material for the following application:

Application: {application}

Constraints:
{self._format_constraints(constraints)}

Please provide:
1. Suggested chemical formula
2. Crystal structure (space group if known)
3. Expected properties
4. Potential applications
5. Synthesis approach
6. Advantages over existing materials

Output as JSON with keys: formula, structure, properties, applications, synthesis, advantages"""
        
        response = self.llm.generate(prompt, output_format="json")
        
        # 解析响应
        data = response.structured_data or self._extract_material_data(response.text)
        
        return MaterialDescription(
            name=data.get('formula', 'Unknown'),
            formula=data.get('formula', 'Unknown'),
            crystal_system=data.get('structure', {}).get('system', 'Unknown'),
            space_group=data.get('structure', {}).get('space_group', 'Unknown'),
            properties=data.get('properties', {}),
            applications=data.get('applications', []),
            synthesis_method=data.get('synthesis', ''),
            description=response.text
        )
    
    def predict_properties(
        self,
        formula: str,
        structure: Optional[Dict] = None,
        property_types: List[str] = None
    ) -> Dict[str, Any]:
        """预测材料性质"""
        prop_types = property_types or ['electronic', 'mechanical', 'thermal']
        
        prompt = f"""Predict the following properties for material {formula}:

Requested Properties: {', '.join(prop_types)}

{f"Structure Information:\n{self.encoder.encode(structure)}" if structure else ""}

Please provide quantitative estimates with uncertainty ranges and confidence levels.

Output as JSON with property names as keys."""
        
        response = self.llm.generate(prompt, output_format="json")
        return response.structured_data or {}
    
    def suggest_synthesis(
        self,
        formula: str,
        available_methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """建议合成路线"""
        prompt = f"""Suggest synthesis route for {formula}.

{f"Available Methods: {', '.join(available_methods)}" if available_methods else ""}

Provide detailed steps including:
1. Precursor materials
2. Temperature profile
3. Atmosphere requirements
4. Expected timeline
5. Critical parameters
6. Potential challenges

Output as JSON with these sections."""
        
        response = self.llm.generate(prompt, output_format="json")
        return response.structured_data or {}
    
    def screen_candidates(
        self,
        candidates: List[str],
        target_properties: Dict[str, Tuple[float, float]]
    ) -> List[Dict[str, Any]]:
        """
        筛选候选材料
        
        Args:
            candidates: 候选材料列表
            target_properties: 目标属性范围 {property: (min, max)}
        """
        prompt = f"""Screen the following candidate materials:

Candidates: {', '.join(candidates)}

Target Property Ranges:
{chr(10).join(f"- {p}: {r[0]} to {r[1]}" for p, r in target_properties.items())}

For each candidate, estimate the properties and rank by suitability.
Provide reasoning for top candidates.

Output as JSON array with ranking and scores."""
        
        response = self.llm.generate(prompt, output_format="json")
        return response.structured_data or []
    
    def _format_constraints(self, constraints: Optional[Dict]) -> str:
        """格式化约束条件"""
        if not constraints:
            return "None specified"
        return chr(10).join(f"- {k}: {v}" for k, v in constraints.items())
    
    def _extract_material_data(self, text: str) -> Dict:
        """从文本提取材料数据"""
        # 简单的规则提取
        data = {}
        
        # 提取化学式
        formula_match = re.search(r'[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*', text)
        if formula_match:
            data['formula'] = formula_match.group()
        
        return data
    
    def chat(self, message: str) -> str:
        """交互式对话"""
        # 添加上下文
        context = "\n".join([
            f"User: {h['user']}\nAssistant: {h['assistant']}"
            for h in self.conversation_history[-5:]
        ])
        
        full_prompt = f"{context}\n\nUser: {message}\nAssistant:"
        
        response = self.llm.generate(full_prompt)
        
        # 保存历史
        self.conversation_history.append({
            'user': message,
            'assistant': response.text
        })
        
        return response.text


class MultiModalMaterialsModel(nn.Module):
    """
    多模态材料模型
    
    整合文本、结构、光谱等多种模态
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        structure_dim: int = 256,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        # 各模态编码器
        self.text_encoder = nn.Linear(text_dim, hidden_dim)
        self.structure_encoder = nn.Linear(structure_dim, hidden_dim)
        
        # 融合层
        self.fusion = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # 预测头
        self.property_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 10)  # 10种性质
        )
        
    def forward(
        self,
        text_features: torch.Tensor,
        structure_features: torch.Tensor
    ) -> torch.Tensor:
        """多模态前向传播"""
        # 编码各模态
        text_emb = self.text_encoder(text_features)
        struct_emb = self.structure_encoder(structure_features)
        
        # 拼接
        combined = torch.stack([text_emb, struct_emb], dim=1)
        
        # 跨模态注意力
        fused, _ = self.fusion(combined, combined, combined)
        
        # 全局池化
        global_feat = fused.mean(dim=1)
        
        # 预测
        return self.property_predictor(global_feat)


if __name__ == "__main__":
    print("=" * 60)
    print("LLM Materials Design Demo")
    print("=" * 60)
    
    # 初始化组件
    llm = MaterialsLLMInterface(model_name="mock-gpt-4")
    agent = LLMMaterialsAgent(llm)
    
    # 测试1: 材料设计
    print("\n1. Material Design for Battery Cathode")
    print("-" * 40)
    
    material = agent.design_for_application(
        application="High voltage lithium-ion battery cathode",
        constraints={
            "max_cost": "$50/kg",
            "min_cycle_life": "1000 cycles",
            "operating_voltage": "> 4.5 V"
        }
    )
    
    print(f"Suggested Material: {material.formula}")
    print(f"Crystal System: {material.crystal_system}")
    print(f"Expected Properties: {material.properties}")
    
    # 测试2: 性质预测
    print("\n2. Property Prediction")
    print("-" * 40)
    
    properties = agent.predict_properties(
        formula="LiFePO4",
        property_types=['band_gap', 'ionic_conductivity', 'thermal_stability']
    )
    print(f"Predicted properties: {properties}")
    
    # 测试3: 合成建议
    print("\n3. Synthesis Route Suggestion")
    print("-" * 40)
    
    synthesis = agent.suggest_synthesis("LiNi0.5Mn1.5O4")
    print(f"Synthesis method: {synthesis.get('synthesis_method', 'N/A')}")
    
    # 测试4: 结构编码
    print("\n4. Structure to Text Encoding")
    print("-" * 40)
    
    encoder = StructureToTextEncoder()
    
    # 模拟结构
    test_structure = {
        'atom_types': torch.tensor([3, 8, 8, 8, 8]),  # Li O4
        'frac_coords': torch.tensor([
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75]
        ]),
        'lengths': torch.tensor([4.0, 4.0, 4.0]),
        'angles': torch.tensor([90., 90., 90.])
    }
    
    description = encoder.encode(test_structure)
    print(description[:500] + "...")
    
    # 测试5: 对话
    print("\n5. Interactive Chat")
    print("-" * 40)
    
    response = agent.chat("What are the advantages of solid-state batteries?")
    print(f"Q: What are the advantages of solid-state batteries?")
    print(f"A: {response[:300]}...")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("Key capabilities:")
    print("- Text-based material design")
    print("- Property prediction from formulas")
    print("- Synthesis route planning")
    print("- Structure-text conversion")
    print("=" * 60)
