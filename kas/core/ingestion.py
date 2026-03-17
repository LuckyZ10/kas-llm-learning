"""
KAS Core - Ingestion Engine
简单优先的代码吞食引擎
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Set
import yaml

from .models import Agent, Capability, CapabilityType, DEFAULT_MODEL_CONFIG


class IngestionEngine:
    """代码吞食引擎 - 简化版"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        
        # 文件类型映射到能力
        self.file_type_capabilities = {
            '.py': [CapabilityType.CODE_REVIEW, CapabilityType.REFACTORING],
            '.js': [CapabilityType.CODE_REVIEW],
            '.ts': [CapabilityType.CODE_REVIEW, CapabilityType.ARCHITECTURE],
            '.test.py': [CapabilityType.TEST_GENERATION],
            '.spec.js': [CapabilityType.TEST_GENERATION],
            '.md': [CapabilityType.DOCUMENTATION],
            '.rst': [CapabilityType.DOCUMENTATION],
        }
    
    def ingest(self, project_path: str, agent_name: Optional[str] = None) -> Agent:
        """
        吞食项目，提取Agent
        
        Args:
            project_path: 项目路径
            agent_name: Agent名称（可选）
        
        Returns:
            Agent对象
        """
        path = Path(project_path)
        
        if not path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")
        
        # 1. 分析项目结构
        project_info = self._analyze_structure(path)
        
        # 2. 提取代码样本
        code_samples = self._extract_code_samples(path)
        
        # 3. 识别能力（简单规则）
        capabilities = self._detect_capabilities(project_info, code_samples)
        
        # 4. 生成系统提示词
        system_prompt = self._generate_system_prompt(capabilities, project_info)
        
        # 5. 创建Agent
        agent = Agent(
            name=agent_name or path.name,
            description=f"Agent extracted from {path.name}",
            capabilities=capabilities,
            system_prompt=system_prompt,
            model_config=DEFAULT_MODEL_CONFIG,
            created_from=str(path.absolute())
        )
        
        return agent
    
    def _analyze_structure(self, path: Path) -> Dict:
        """分析项目结构"""
        stats = {
            'total_files': 0,
            'file_types': {},
            'has_tests': False,
            'has_docs': False,
            'has_config': False,
        }
        
        for file_path in path.rglob('*'):
            if file_path.is_file():
                stats['total_files'] += 1
                suffix = file_path.suffix.lower()
                stats['file_types'][suffix] = stats['file_types'].get(suffix, 0) + 1
                
                # 检测测试文件
                if 'test' in file_path.name.lower() or 'spec' in file_path.name.lower():
                    stats['has_tests'] = True
                
                # 检测文档
                if suffix in ['.md', '.rst', '.txt']:
                    stats['has_docs'] = True
                
                # 检测配置
                if file_path.name in ['package.json', 'requirements.txt', 'Cargo.toml', 'pom.xml']:
                    stats['has_config'] = True
        
        return stats
    
    def _extract_code_samples(self, path: Path, max_samples: int = 5) -> List[str]:
        """提取代表性代码样本"""
        samples = []
        
        # 优先提取主要代码文件
        for file_path in path.rglob('*'):
            if len(samples) >= max_samples:
                break
            
            if file_path.is_file() and file_path.suffix in ['.py', '.js', '.ts', '.java', '.go']:
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    # 只取前100行
                    lines = content.split('\n')[:100]
                    samples.append(f"# File: {file_path.name}\n" + '\n'.join(lines))
                except Exception:
                    continue
        
        return samples
    
    def _detect_capabilities(self, project_info: Dict, code_samples: List[str]) -> List[Capability]:
        """检测能力 - 基于简单规则"""
        capabilities = []
        
        # 基于文件类型
        for suffix, count in project_info['file_types'].items():
            if suffix in self.file_type_capabilities:
                for cap_type in self.file_type_capabilities[suffix]:
                    cap = Capability(
                        name=f"{cap_type.value.replace('_', ' ').title()}",
                        type=cap_type,
                        description=f"Detected from {count} {suffix} files",
                        confidence=min(0.5 + count * 0.05, 0.95)
                    )
                    # 去重
                    if not any(c.type == cap.type for c in capabilities):
                        capabilities.append(cap)
        
        # 基于代码特征
        all_code = '\n'.join(code_samples)
        
        if 'class ' in all_code or 'def ' in all_code:
            if not any(c.type == CapabilityType.ARCHITECTURE for c in capabilities):
                capabilities.append(Capability(
                    name="Architecture Understanding",
                    type=CapabilityType.ARCHITECTURE,
                    description="Detected class/function definitions",
                    confidence=0.7
                ))
        
        if 'try:' in all_code or 'except' in all_code or 'catch' in all_code:
            if not any(c.type == CapabilityType.DEBUGGING for c in capabilities):
                capabilities.append(Capability(
                    name="Error Handling",
                    type=CapabilityType.DEBUGGING,
                    description="Detected error handling patterns",
                    confidence=0.75
                ))
        
        # 基于项目特征
        if project_info['has_tests']:
            if not any(c.type == CapabilityType.TEST_GENERATION for c in capabilities):
                capabilities.append(Capability(
                    name="Test Generation",
                    type=CapabilityType.TEST_GENERATION,
                    description="Test files detected in project",
                    confidence=0.8
                ))
        
        if project_info['has_docs']:
            if not any(c.type == CapabilityType.DOCUMENTATION for c in capabilities):
                capabilities.append(Capability(
                    name="Documentation",
                    type=CapabilityType.DOCUMENTATION,
                    description="Documentation files detected",
                    confidence=0.75
                ))
        
        return capabilities
    
    def _generate_system_prompt(self, capabilities: List[Capability], project_info: Dict) -> str:
        """生成系统提示词 - 模板化"""
        capability_list = '\n'.join([
            f"- {cap.name}: {cap.description}"
            for cap in capabilities
        ])
        
        prompt = f"""You are an expert software development assistant with the following capabilities:

{capability_list}

When helping users:
1. Apply best practices from the codebase patterns you've learned
2. Focus on code quality and maintainability
3. Provide clear explanations for your suggestions
4. Consider the project's architecture and conventions

You should be helpful, professional, and concise in your responses.
"""
        return prompt.strip()
    
    def save_agent(self, agent: Agent, output_dir: str) -> str:
        """保存Agent到目录"""
        output_path = Path(output_dir) / agent.name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存agent.yaml
        agent_yaml = output_path / 'agent.yaml'
        with open(agent_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(agent.to_dict(), f, default_flow_style=False, allow_unicode=True)
        
        # 保存system_prompt.txt
        prompt_file = output_path / 'system_prompt.txt'
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(agent.system_prompt)
        
        return str(output_path)


def ingest_project(project_path: str, agent_name: Optional[str] = None, llm_client=None) -> Dict:
    """
    简单的项目吞食接口
    
    Returns:
        {
            'agent': Agent对象,
            'output_path': 保存路径,
            'stats': 统计信息
        }
    """
    engine = IngestionEngine(llm_client)
    
    # 吞食项目
    agent = engine.ingest(project_path, agent_name)
    
    # 保存
    import os
    kas_dir = Path.home() / '.kas' / 'agents'
    output_path = engine.save_agent(agent, kas_dir)
    
    return {
        'agent': agent,
        'output_path': output_path,
        'stats': {
            'capabilities_found': len(agent.capabilities),
            'agent_name': agent.name,
        }
    }
