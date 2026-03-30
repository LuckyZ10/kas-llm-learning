"""
auto_importer.py
自动将新方法集成到平台

自动分析论文方法, 生成可集成代码并验证。

References:
- 2024进展: AI辅助代码生成和集成
"""

import re
import ast
import inspect
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class CodeTemplate:
    """代码模板"""
    name: str
    description: str
    template_code: str
    required_params: List[str]
    optional_params: Dict[str, Any]
    dependencies: List[str]


@dataclass
class IntegrationPlan:
    """集成计划"""
    method_name: str
    source_paper: str
    target_module: str
    generated_code: str
    validation_tests: List[str]
    integration_status: str = "pending"  # pending, testing, integrated, failed


class AutoImporter:
    """
    自动导入器
    
    将新方法自动集成到平台
    """
    
    def __init__(self, platform_base_path: str = "dftlammps"):
        self.platform_base = platform_base_path
        
        # 代码模板库
        self.templates = self._load_templates()
        
        # 集成历史
        self.integration_history: List[IntegrationPlan] = []
        
    def _load_templates(self) -> Dict[str, CodeTemplate]:
        """加载代码模板"""
        templates = {}
        
        # GNN模型模板
        templates['gnn_model'] = CodeTemplate(
            name="GNN Model",
            description="图神经网络模型模板",
            template_code='''
class {{model_name}}(nn.Module):
    """
    {{description}}
    
    Source: {{paper_reference}}
    """
    
    def __init__(
        self,
        num_elements: int = 100,
        hidden_dim: int = {{hidden_dim}},
        num_layers: int = {{num_layers}},
        cutoff: float = {{cutoff}}
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 原子嵌入
        self.atom_embed = nn.Embedding(num_elements, hidden_dim)
        
        # 消息传递层
        self.layers = nn.ModuleList([
            {{message_passing_layer}}(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, atomic_numbers, positions, edge_index):
        # 实现前向传播
        h = self.atom_embed(atomic_numbers)
        
        for layer in self.layers:
            h = layer(h, edge_index)
        
        return self.output(h)
''',
            required_params=['model_name', 'description', 'paper_reference'],
            optional_params={'hidden_dim': 128, 'num_layers': 3, 'cutoff': 5.0},
            dependencies=['torch', 'torch.nn']
        )
        
        # 势函数模板
        templates['potential_model'] = CodeTemplate(
            name="ML Potential",
            description="机器学习势函数模板",
            template_code='''
class {{model_name}}(nn.Module):
    """
    {{description}}
    
    Source: {{paper_reference}}
    """
    
    def __init__(self, num_elements=100, hidden_dim={{hidden_dim}}):
        super().__init__()
        self.atom_embed = nn.Embedding(num_elements, hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data):
        positions = data.positions.requires_grad_(True)
        atomic_numbers = data.atomic_numbers
        
        h = self.atom_embed(atomic_numbers - 1)
        atomic_energies = self.network(h).squeeze(-1)
        total_energy = atomic_energies.sum()
        
        forces = -torch.autograd.grad(
            total_energy, positions, create_graph=True
        )[0]
        
        return {'energy': total_energy, 'forces': forces}
''',
            required_params=['model_name', 'description', 'paper_reference'],
            optional_params={'hidden_dim': 64},
            dependencies=['torch']
        )
        
        # 数据集模板
        templates['dataset'] = CodeTemplate(
            name="Dataset Loader",
            description="材料数据集加载器模板",
            template_code='''
class {{dataset_name}}(Dataset):
    """
    {{description}}
    
    Source: {{paper_reference}}
    """
    
    def __init__(self, root: str, split: str = 'train'):
        super().__init__()
        self.root = root
        self.split = split
        self.data = self._load_data()
    
    def _load_data(self):
        # 实现数据加载
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
''',
            required_params=['dataset_name', 'description', 'paper_reference'],
            optional_params={},
            dependencies=['torch.utils.data']
        )
        
        return templates
    
    def analyze_method(
        self,
        paper_analysis: Dict,
        method_description: str
    ) -> Dict[str, Any]:
        """
        分析方法描述, 提取关键信息
        """
        analysis = {
            'method_type': None,
            'key_parameters': {},
            'architecture_hints': [],
            'suggested_template': None
        }
        
        method_lower = method_description.lower()
        
        # 识别方法类型
        if any(kw in method_lower for kw in ['graph neural network', 'gnn', 'message passing']):
            analysis['method_type'] = 'gnn'
            analysis['suggested_template'] = 'gnn_model'
        
        elif any(kw in method_lower for kw in ['potential', 'force field', 'interatomic']):
            analysis['method_type'] = 'potential'
            analysis['suggested_template'] = 'potential_model'
        
        elif any(kw in method_lower for kw in ['dataset', 'benchmark', 'database']):
            analysis['method_type'] = 'dataset'
            analysis['suggested_template'] = 'dataset'
        
        # 提取参数
        param_patterns = [
            (r'hidden\s*dimension\s*(?:of|is|=)\s*(\d+)', 'hidden_dim'),
            (r'hidden\s*size\s*(?:of|is|=)\s*(\d+)', 'hidden_dim'),
            (r'(\d+)\s*layers?', 'num_layers'),
            (r'cutoff\s*(?:of|is|=)\s*([\d.]+)', 'cutoff'),
            (r'learning\s*rate\s*(?:of|is|=)\s*([\d.e-]+)', 'learning_rate'),
        ]
        
        for pattern, param_name in param_patterns:
            match = re.search(pattern, method_description, re.IGNORECASE)
            if match:
                value = match.group(1)
                try:
                    analysis['key_parameters'][param_name] = int(value)
                except ValueError:
                    try:
                        analysis['key_parameters'][param_name] = float(value)
                    except ValueError:
                        analysis['key_parameters'][param_name] = value
        
        return analysis
    
    def generate_code(
        self,
        method_analysis: Dict,
        paper_info: Dict
    ) -> str:
        """
        基于分析生成代码
        """
        template_name = method_analysis.get('suggested_template', 'gnn_model')
        template = self.templates.get(template_name)
        
        if not template:
            return "# No suitable template found"
        
        # 准备参数
        params = {
            'model_name': paper_info.get('method_name', 'NewModel'),
            'description': paper_info.get('description', 'Auto-generated model'),
            'paper_reference': paper_info.get('paper_id', 'Unknown'),
        }
        
        # 添加提取的参数
        params.update(method_analysis.get('key_parameters', {}))
        
        # 添加默认参数
        for key, value in template.optional_params.items():
            if key not in params:
                params[key] = value
        
        # 生成代码
        code = template.template_code
        for key, value in params.items():
            placeholder = '{{' + key + '}}'
            code = code.replace(placeholder, str(value))
        
        return code
    
    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """
        验证生成的代码
        
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        
        # 语法检查
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return False, errors
        
        # 检查必要的导入
        required_imports = ['torch', 'torch.nn']
        for imp in required_imports:
            if imp not in code:
                errors.append(f"Missing import: {imp}")
        
        # 检查类定义
        if 'class ' not in code:
            errors.append("No class definition found")
        
        # 检查前向传播
        if 'def forward(' not in code:
            errors.append("No forward method found")
        
        return len(errors) == 0, errors
    
    def generate_tests(self, code: str, class_name: str) -> str:
        """生成单元测试"""
        test_code = f'''
import torch
import pytest
from {class_name} import {class_name}

class Test{class_name}:
    def test_initialization(self):
        model = {class_name}()
        assert model is not None
    
    def test_forward_pass(self):
        model = {class_name}()
        # Create dummy input
        batch_size = 4
        num_atoms = 10
        atomic_numbers = torch.randint(1, 20, (num_atoms,))
        positions = torch.randn(num_atoms, 3)
        edge_index = torch.randint(0, num_atoms, (2, 50))
        
        output = model(atomic_numbers, positions, edge_index)
        assert output is not None
    
    def test_gradient_flow(self):
        model = {class_name}()
        atomic_numbers = torch.randint(1, 20, (5,))
        positions = torch.randn(5, 3, requires_grad=True)
        edge_index = torch.randint(0, 5, (2, 20))
        
        output = model(atomic_numbers, positions, edge_index)
        loss = output.sum()
        loss.backward()
        
        assert positions.grad is not None

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
'''
        return test_code
    
    def create_integration_plan(
        self,
        paper_info: Dict,
        generated_code: str,
        validation_tests: str
    ) -> IntegrationPlan:
        """创建集成计划"""
        # 确定目标模块
        method_type = paper_info.get('method_type', 'model')
        
        target_modules = {
            'gnn': 'dftlammps/frontier',
            'potential': 'dftlammps/frontier',
            'dataset': 'dftlammps/data',
            'diffusion': 'dftlammps/frontier'
        }
        
        target_module = target_modules.get(method_type, 'dftlammps/frontier')
        
        plan = IntegrationPlan(
            method_name=paper_info.get('method_name', 'NewMethod'),
            source_paper=paper_info.get('paper_id', 'Unknown'),
            target_module=target_module,
            generated_code=generated_code,
            validation_tests=[validation_tests],
            integration_status="pending"
        )
        
        self.integration_history.append(plan)
        return plan
    
    def auto_integrate(
        self,
        paper_analysis: Dict,
        method_description: str
    ) -> IntegrationPlan:
        """
        自动集成新方法 (主入口)
        """
        # 1. 分析方法
        method_analysis = self.analyze_method(paper_analysis, method_description)
        
        # 2. 生成代码
        code = self.generate_code(method_analysis, paper_analysis)
        
        # 3. 验证代码
        is_valid, errors = self.validate_code(code)
        
        if not is_valid:
            code = f"# Validation failed:\n# " + "\n# ".join(errors) + "\n\n" + code
        
        # 4. 生成测试
        class_name = paper_analysis.get('method_name', 'NewModel')
        tests = self.generate_tests(code, class_name)
        
        # 5. 创建集成计划
        plan = self.create_integration_plan(
            paper_analysis,
            code,
            tests
        )
        
        plan.integration_status = "validated" if is_valid else "failed"
        
        return plan
    
    def get_integration_summary(self) -> Dict:
        """获取集成摘要"""
        status_counts = defaultdict(int)
        for plan in self.integration_history:
            status_counts[plan.integration_status] += 1
        
        return {
            'total_attempts': len(self.integration_history),
            'status_breakdown': dict(status_counts),
            'successful_integrations': [
                {
                    'method': p.method_name,
                    'paper': p.source_paper,
                    'module': p.target_module
                }
                for p in self.integration_history
                if p.integration_status == 'integrated'
            ]
        }


def demo():
    """演示"""
    print("=" * 60)
    print("Auto Importer Demo")
    print("=" * 60)
    
    importer = AutoImporter()
    
    # 模拟论文信息
    paper_info = {
        'paper_id': 'arXiv:2401.12345',
        'method_name': 'TransformerGNN',
        'description': 'A transformer-based graph neural network for crystal property prediction',
        'method_type': 'gnn'
    }
    
    # 模拟方法描述
    method_description = """
    We propose TransformerGNN, a novel architecture combining transformer attention
    with message passing neural networks. The model uses a hidden dimension of 256
    with 6 layers and a cutoff of 6.0 Angstroms.
    
    The architecture consists of:
    - Atom embedding layer (256 dimensions)
    - 6 transformer-GNN layers
    - Global attention pooling
    - Output prediction head
    """
    
    print("\n1. Analyzing method...")
    analysis = importer.analyze_method(paper_info, method_description)
    print(f"   Method type: {analysis['method_type']}")
    print(f"   Key parameters: {analysis['key_parameters']}")
    print(f"   Suggested template: {analysis['suggested_template']}")
    
    print("\n2. Generating code...")
    code = importer.generate_code(analysis, paper_info)
    print(f"   Generated {len(code)} characters of code")
    
    print("\n3. Validating code...")
    is_valid, errors = importer.validate_code(code)
    print(f"   Valid: {is_valid}")
    if errors:
        print(f"   Errors: {errors}")
    
    print("\n4. Creating integration plan...")
    plan = importer.auto_integrate(paper_info, method_description)
    print(f"   Method: {plan.method_name}")
    print(f"   Target module: {plan.target_module}")
    print(f"   Status: {plan.integration_status}")
    
    print("\n5. Generated code preview:")
    print("-" * 40)
    print(code[:500] + "..." if len(code) > 500 else code)
    
    print("\n6. Generated test preview:")
    print("-" * 40)
    print(plan.validation_tests[0][:500] + "...")
    
    # 集成摘要
    print("\n" + "=" * 60)
    print("Integration Summary")
    summary = importer.get_integration_summary()
    print(f"Total attempts: {summary['total_attempts']}")
    print(f"Status: {summary['status_breakdown']}")
    
    print("\n" + "=" * 60)
    print("Auto Importer Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
