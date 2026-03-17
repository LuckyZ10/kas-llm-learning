"""
KAS Core - Fusion Engine
简单优先的Agent合体引擎
"""
from typing import List, Dict, Optional
import copy

from .models import Agent, Capability


class FusionStrategy:
    """合体策略"""
    UNION = "union"           # 并集 - 保留所有能力
    INTERSECTION = "intersect" # 交集 - 只保留共同能力
    DOMINANT = "dominant"     # 主导 - 以第一个Agent为准
    SYNTHESIS = "synthesis"   # 合成 - 合并并去重


class FusionEngine:
    """Agent合体引擎 - 简化版"""
    
    def __init__(self):
        self.strategies = {
            FusionStrategy.UNION: self._union,
            FusionStrategy.INTERSECTION: self._intersection,
            FusionStrategy.DOMINANT: self._dominant,
            FusionStrategy.SYNTHESIS: self._synthesis,
        }
    
    def fuse(
        self,
        agents: List[Agent],
        strategy: str = FusionStrategy.SYNTHESIS,
        new_name: Optional[str] = None
    ) -> Agent:
        """
        合体多个Agent
        
        Args:
            agents: Agent列表
            strategy: 合体策略
            new_name: 新Agent名称
        
        Returns:
            合体后的Agent
        """
        if len(agents) < 2:
            raise ValueError("Need at least 2 agents to fuse")
        
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # 执行合体
        fusion_func = self.strategies[strategy]
        result = fusion_func(agents)
        
        # 设置新名称
        if new_name:
            result.name = new_name
        else:
            result.name = f"Fused_{'_'.join([a.name for a in agents])}"
        
        result.description = f"Fused agent combining capabilities from: {', '.join([a.name for a in agents])}"
        
        return result
    
    def _union(self, agents: List[Agent]) -> Agent:
        """并集策略 - 保留所有能力"""
        result = copy.deepcopy(agents[0])
        
        all_capabilities = []
        capability_types = set()
        
        for agent in agents:
            for cap in agent.capabilities:
                if cap.type not in capability_types:
                    all_capabilities.append(cap)
                    capability_types.add(cap.type)
        
        result.capabilities = all_capabilities
        result.system_prompt = self._merge_prompts_union([a.system_prompt for a in agents])
        result.model_config = self._merge_configs([a.model_config for a in agents])
        
        return result
    
    def _intersection(self, agents: List[Agent]) -> Agent:
        """交集策略 - 只保留共同能力"""
        result = copy.deepcopy(agents[0])
        
        # 找出所有Agent都有的能力类型
        common_types = set(agents[0].capabilities)
        for agent in agents[1:]:
            agent_types = set(agent.capabilities)
            common_types = common_types.intersection(agent_types)
        
        # 保留共同能力（取平均置信度）
        result.capabilities = list(common_types)
        result.system_prompt = agents[0].system_prompt  # 使用第一个的prompt
        
        return result
    
    def _dominant(self, agents: List[Agent]) -> Agent:
        """主导策略 - 以第一个Agent为准，补充其他Agent的独特能力"""
        result = copy.deepcopy(agents[0])
        
        # 添加其他Agent的独特能力
        existing_types = {cap.type for cap in result.capabilities}
        
        for agent in agents[1:]:
            for cap in agent.capabilities:
                if cap.type not in existing_types:
                    result.capabilities.append(cap)
                    existing_types.add(cap.type)
        
        return result
    
    def _synthesis(self, agents: List[Agent]) -> Agent:
        """合成策略 - 智能合并"""
        result = copy.deepcopy(agents[0])
        
        # 合并所有独特能力
        all_capabilities = {}
        
        for agent in agents:
            for cap in agent.capabilities:
                if cap.type not in all_capabilities:
                    all_capabilities[cap.type] = cap
                else:
                    # 如果已存在，提升置信度
                    existing = all_capabilities[cap.type]
                    existing.confidence = min(existing.confidence + 0.1, 0.95)
        
        result.capabilities = list(all_capabilities.values())
        
        # 合并prompts
        result.system_prompt = self._merge_prompts_synthesis([a.system_prompt for a in agents])
        
        # 合并配置（取平均）
        result.model_config = self._merge_configs([a.model_config for a in agents])
        
        return result
    
    def _merge_prompts_union(self, prompts: List[str]) -> str:
        """合并Prompts - 并集方式"""
        sections = []
        
        for i, prompt in enumerate(prompts):
            sections.append(f"## Source Agent {i+1}\n{prompt}")
        
        merged = "You are a versatile assistant combining multiple expertises:\n\n"
        merged += "\n\n".join(sections)
        
        return merged
    
    def _merge_prompts_synthesis(self, prompts: List[str]) -> str:
        """合并Prompts - 合成方式（去重）"""
        # 简单合并，去除重复句子
        all_lines = []
        seen = set()
        
        for prompt in prompts:
            for line in prompt.split('\n'):
                line = line.strip()
                if line and line not in seen:
                    all_lines.append(line)
                    seen.add(line)
        
        return '\n'.join(all_lines)
    
    def _merge_configs(self, configs: List[Dict]) -> Dict:
        """合并模型配置 - 取平均值"""
        if not configs:
            return {}
        
        merged = {}
        
        # 数值参数取平均
        numeric_keys = ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty']
        for key in numeric_keys:
            values = [c.get(key) for c in configs if c.get(key) is not None]
            if values:
                merged[key] = sum(values) / len(values)
        
        # max_tokens取最大
        max_tokens = [c.get('max_tokens') for c in configs if c.get('max_tokens') is not None]
        if max_tokens:
            merged['max_tokens'] = max(max_tokens)
        
        return merged
    
    def detect_emergent_capabilities(self, agents: List[Agent]) -> List[Dict]:
        """
        检测可能涌现的新能力
        
        Returns:
            可能的新能力列表
        """
        emergent = []
        
        # 简单规则：某些能力组合会产生新能力
        capability_types = set()
        for agent in agents:
            for cap in agent.capabilities:
                capability_types.add(cap.type.value)
        
        # 检测组合
        if 'code_review' in capability_types and 'test_generation' in capability_types:
            emergent.append({
                'name': 'Test-Driven Development Guide',
                'description': 'Can guide users through TDD workflow',
                'confidence': 0.75
            })
        
        if 'documentation' in capability_types and 'code_review' in capability_types:
            emergent.append({
                'name': 'Self-Documenting Code Review',
                'description': 'Reviews code with documentation quality in mind',
                'confidence': 0.7
            })
        
        if 'architecture' in capability_types and 'refactoring' in capability_types:
            emergent.append({
                'name': 'Architectural Refactoring',
                'description': 'Can suggest high-level architectural improvements',
                'confidence': 0.8
            })
        
        return emergent


def fuse_agents(
    agent_paths: List[str],
    strategy: str = 'synthesis',
    new_name: Optional[str] = None
) -> Dict:
    """
    简单的Agent合体接口
    
    Args:
        agent_paths: Agent路径列表
        strategy: 合体策略
        new_name: 新名称
    
    Returns:
        {
            'agent': 合体后的Agent,
            'emergent_capabilities': 涌现能力,
            'strategy': 使用的策略
        }
    """
    import yaml
    from pathlib import Path
    
    # 加载Agents
    agents = []
    for path in agent_paths:
        agent_file = Path(path) / 'agent.yaml'
        if agent_file.exists():
            with open(agent_file, 'r') as f:
                data = yaml.safe_load(f)
                agents.append(Agent.from_dict(data))
    
    if len(agents) < 2:
        raise ValueError("Need at least 2 valid agents")
    
    # 合体
    engine = FusionEngine()
    fused_agent = engine.fuse(agents, strategy, new_name)
    
    # 检测涌现能力
    emergent = engine.detect_emergent_capabilities(agents)
    
    # 保存
    from .ingestion import IngestionEngine
    ingestion = IngestionEngine()
    output_path = ingestion.save_agent(fused_agent, Path.home() / '.kas' / 'agents')
    
    return {
        'agent': fused_agent,
        'emergent_capabilities': emergent,
        'strategy': strategy,
        'output_path': output_path
    }
