"""
paper_analyzer.py
论文结构分析 - 方法提取、代码复现评估

自动分析学术论文, 提取关键方法信息, 评估代码可复现性。

References:
- 2024进展: LLM辅助论文分析和代码提取
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import ast


@dataclass
class MethodSection:
    """方法章节"""
    section_name: str
    content: str
    techniques: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    equations: List[str] = field(default_factory=list)


@dataclass
class CodeSnippet:
    """代码片段"""
    language: str
    code: str
    context: str = ""  # 代码周围的文本
    is_complete: bool = False  # 是否是完整可运行代码


@dataclass
class PaperAnalysis:
    """论文分析结果"""
    title: str
    authors: List[str]
    methods: List[MethodSection]
    code_snippets: List[CodeSnippet]
    reproducibility_score: float = 0.0
    has_code_repo: bool = False
    code_repo_url: Optional[str] = None
    datasets_mentioned: List[str] = field(default_factory=list)
    software_mentioned: List[str] = field(default_factory=list)


class PaperAnalyzer:
    """
    论文分析器
    
    提取论文中的方法信息和代码相关内容
    """
    
    # 技术关键词库
    TECHNIQUE_KEYWORDS = {
        'ml': [
            'machine learning', 'deep learning', 'neural network',
            'transformer', 'BERT', 'GPT', 'CNN', 'RNN', 'LSTM',
            'graph neural network', 'GNN', 'message passing',
            'attention', 'self-attention', 'multi-head attention',
            'diffusion model', 'generative model', 'VAE', 'GAN'
        ],
        'dft': [
            'density functional theory', 'DFT', 'Kohn-Sham',
            'PBE', 'LDA', 'GGA', 'hybrid functional',
            'HSE', 'B3LYP', 'SCAN', 'meta-GGA'
        ],
        'md': [
            'molecular dynamics', 'MD', 'LAMMPS', 'GROMACS',
            'NAMD', 'force field', 'interatomic potential',
            'reactive force field', 'ReaxFF'
        ],
        'sampling': [
            'Monte Carlo', 'MC', 'Metropolis', 'kinetic Monte Carlo',
            'kMC', 'umbrella sampling', 'metadynamics',
            'enhanced sampling', 'free energy calculation'
        ],
        'optimization': [
            'Bayesian optimization', 'active learning', 'genetic algorithm',
            'particle swarm', 'gradient descent', 'Adam', 'SGD',
            'hyperparameter optimization', 'neural architecture search'
        ],
        'materials': [
            'high-throughput', 'materials genome', 'AFLOW', 'Materials Project',
            'OQMD', 'NOMAD', 'crystal structure prediction',
            'structure relaxation', 'phonon calculation'
        ]
    }
    
    # 代码仓库域名
    REPO_DOMAINS = [
        'github.com', 'gitlab.com', 'bitbucket.org',
        'zenodo.org', 'figshare.com'
    ]
    
    def __init__(self):
        self.analysis_results: List[PaperAnalysis] = []
    
    def analyze_paper(self, paper_text: str, paper_title: str = "", authors: List[str] = None) -> PaperAnalysis:
        """
        分析单篇论文
        
        Args:
            paper_text: 论文全文文本
            paper_title: 论文标题
            authors: 作者列表
        """
        authors = authors or []
        
        # 提取方法章节
        methods = self._extract_methods(paper_text)
        
        # 提取代码片段
        code_snippets = self._extract_code_snippets(paper_text)
        
        # 查找代码仓库链接
        repo_url, has_repo = self._find_code_repo(paper_text)
        
        # 提取数据集和软件
        datasets = self._extract_datasets(paper_text)
        software = self._extract_software(paper_text)
        
        # 计算可复现性分数
        reproducibility = self._calculate_reproducibility(
            methods, code_snippets, has_repo, datasets, software
        )
        
        analysis = PaperAnalysis(
            title=paper_title,
            authors=authors,
            methods=methods,
            code_snippets=code_snippets,
            reproducibility_score=reproducibility,
            has_code_repo=has_repo,
            code_repo_url=repo_url,
            datasets_mentioned=datasets,
            software_mentioned=software
        )
        
        self.analysis_results.append(analysis)
        return analysis
    
    def _extract_methods(self, text: str) -> List[MethodSection]:
        """提取方法章节"""
        methods = []
        
        # 常见方法章节标题
        method_headers = [
            r'(?i)methods?\s*$',
            r'(?i)methodology\s*$',
            r'(?i)computational\s+details?\s*$',
            r'(?i)theoretical\s+background\s*$',
            r'(?i)model\s+and\s+methods?\s*$',
            r'(?i)implementation\s+details?\s*$'
        ]
        
        # 分割章节 (简化实现)
        sections = re.split(r'\n\s*\d+\.\s+', text)
        
        for section in sections:
            # 检查是否是方法章节
            is_method = False
            for pattern in method_headers:
                if re.search(pattern, section[:200], re.IGNORECASE):
                    is_method = True
                    break
            
            if is_method or 'computational' in section[:100].lower():
                # 提取技术
                techniques = self._identify_techniques(section)
                
                # 提取参数
                parameters = self._extract_parameters(section)
                
                # 提取方程
                equations = self._extract_equations(section)
                
                method = MethodSection(
                    section_name="Method",
                    content=section[:2000],  # 限制长度
                    techniques=techniques,
                    parameters=parameters,
                    equations=equations
                )
                methods.append(method)
        
        return methods
    
    def _identify_techniques(self, text: str) -> List[str]:
        """识别使用的技术"""
        techniques = []
        text_lower = text.lower()
        
        for category, keywords in self.TECHNIQUE_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    techniques.append(keyword)
        
        return list(set(techniques))
    
    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        """提取方法参数"""
        parameters = {}
        
        # 常见参数模式
        param_patterns = [
            r'(?:cutoff|cut-off)\s*[=:]\s*([\d.]+)\s*(\w+)',
            r'(?:learning\s+rate|lr)\s*[=:]\s*([\d.e-]+)',
            r'(?:batch\s+size)\s*[=:]\s*(\d+)',
            r'(?:epoch|iteration)\s*[=:]\s*(\d+)',
            r'(?:temperature)\s*[=:]\s*([\d.]+)\s*(K|°C)',
            r'(?:pressure)\s*[=:]\s*([\d.]+)\s*(GPa|atm)',
            r'(?:k-point|kpoint)\s*(?:mesh|grid)?\s*[=:]\s*([\dx\s]+)',
            r'(?:energy\s+cutoff)\s*[=:]\s*([\d.]+)\s*(eV|Ry|Ha)'
        ]
        
        for pattern in param_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                param_name = match.group(0).split('=')[0].split(':')[0].strip()
                param_value = match.group(1)
                parameters[param_name] = param_value
        
        return parameters
    
    def _extract_equations(self, text: str) -> List[str]:
        """提取方程"""
        equations = []
        
        # LaTeX方程模式
        equation_patterns = [
            r'\$\$(.*?)\$\$',
            r'\\begin\{equation\}(.*?)\\end\{equation\}',
            r'\\\[(.*?)\\\]',
            r'\$(.*?)\$'
        ]
        
        for pattern in equation_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            equations.extend(matches[:5])  # 限制数量
        
        return equations
    
    def _extract_code_snippets(self, text: str) -> List[CodeSnippet]:
        """提取代码片段"""
        snippets = []
        
        # 代码块模式
        code_patterns = [
            (r'```(\w+)?\n(.*?)```', 'markdown'),
            (r'<code>(.*?)</code>', 'html'),
            (r'(?:import|from)\s+(\w+)', 'python_import')
        ]
        
        # Python代码检测
        python_patterns = [
            r'import\s+\w+',
            r'from\s+\w+\s+import',
            r'def\s+\w+\s*\(',
            r'class\s+\w+',
            r'for\s+\w+\s+in\s+',
            r'if\s+__name__\s*==\s*[\'"]__main__[\'"]'
        ]
        
        # 查找Python代码块
        lines = text.split('\n')
        code_block = []
        in_code = False
        
        for line in lines:
            is_code_line = any(re.search(p, line) for p in python_patterns)
            
            if is_code_line:
                in_code = True
                code_block.append(line)
            elif in_code:
                if len(code_block) >= 3:  # 至少3行
                    code_text = '\n'.join(code_block)
                    snippets.append(CodeSnippet(
                        language='python',
                        code=code_text,
                        is_complete=self._check_code_complete(code_text)
                    ))
                code_block = []
                in_code = False
        
        return snippets
    
    def _check_code_complete(self, code: str) -> bool:
        """检查代码是否完整可运行"""
        try:
            ast.parse(code)
            return True
        except:
            return False
    
    def _find_code_repo(self, text: str) -> Tuple[Optional[str], bool]:
        """查找代码仓库链接"""
        for domain in self.REPO_DOMAINS:
            pattern = rf'https?://{re.escape(domain)}/[^\s\)\]\>]+'
            match = re.search(pattern, text)
            if match:
                return match.group(), True
        
        return None, False
    
    def _extract_datasets(self, text: str) -> List[str]:
        """提取提到的数据集"""
        datasets = []
        
        known_datasets = [
            'Materials Project', 'MP', 'AFLOW', 'OQMD',
            'NOMAD', 'COD', 'ICSD', 'PubChem', 'ZINC',
            'QM9', 'MD17', 'ANI-1', 'OC20', 'OC22'
        ]
        
        for dataset in known_datasets:
            if dataset.lower() in text.lower():
                datasets.append(dataset)
        
        return datasets
    
    def _extract_software(self, text: str) -> List[str]:
        """提取提到的软件"""
        software = []
        
        known_software = [
            'VASP', 'Quantum ESPRESSO', 'Gaussian', 'ORCA',
            'LAMMPS', 'GROMACS', 'NAMD', 'Amber',
            'ASE', 'Pymatgen', 'OVITO', 'VESTA',
            'PyTorch', 'TensorFlow', 'JAX', 'Keras',
            'scikit-learn', 'PyG', 'DGL', 'SchNetPack',
            'DeepMD', 'OpenMX', 'Siesta', 'CP2K'
        ]
        
        for sw in known_software:
            if sw.lower() in text.lower():
                software.append(sw)
        
        return software
    
    def _calculate_reproducibility(
        self,
        methods: List[MethodSection],
        code_snippets: List[CodeSnippet],
        has_repo: bool,
        datasets: List[str],
        software: List[str]
    ) -> float:
        """计算可复现性分数"""
        score = 0.0
        
        # 代码仓库 (最高权重)
        if has_repo:
            score += 0.4
        
        # 代码片段
        if code_snippets:
            score += 0.2
            # 完整代码加分
            if any(s.is_complete for s in code_snippets):
                score += 0.1
        
        # 详细方法
        if methods:
            total_params = sum(len(m.parameters) for m in methods)
            if total_params >= 5:
                score += 0.15
        
        # 数据集公开
        if datasets:
            score += 0.1
        
        # 软件开源
        open_source_sw = ['PyTorch', 'TensorFlow', 'ASE', 'Pymatgen', 'LAMMPS']
        if any(sw in open_source_sw for sw in software):
            score += 0.05
        
        return min(1.0, score)
    
    def assess_reproducibility(self, analysis: PaperAnalysis) -> Dict[str, Any]:
        """详细评估可复现性"""
        assessment = {
            'overall_score': analysis.reproducibility_score,
            'grade': self._score_to_grade(analysis.reproducibility_score),
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        if analysis.has_code_repo:
            assessment['strengths'].append("Code repository available")
        else:
            assessment['weaknesses'].append("No code repository found")
            assessment['recommendations'].append("Provide link to code repository")
        
        if analysis.code_snippets:
            assessment['strengths'].append("Code snippets in paper")
        
        if analysis.datasets_mentioned:
            assessment['strengths'].append(f"Datasets mentioned: {analysis.datasets_mentioned}")
        
        if analysis.software_mentioned:
            assessment['strengths'].append(f"Software specified: {len(analysis.software_mentioned)} tools")
        
        return assessment
    
    def _score_to_grade(self, score: float) -> str:
        """分数转等级"""
        if score >= 0.8:
            return "A (Highly Reproducible)"
        elif score >= 0.6:
            return "B (Reproducible)"
        elif score >= 0.4:
            return "C (Partially Reproducible)"
        elif score >= 0.2:
            return "D (Difficult to Reproduce)"
        else:
            return "F (Not Reproducible)"
    
    def generate_summary(self, analysis: PaperAnalysis) -> str:
        """生成分析摘要"""
        summary = f"""# Paper Analysis Summary

## {analysis.title}
**Authors**: {', '.join(analysis.authors[:3])}{' et al.' if len(analysis.authors) > 3 else ''}

### Reproducibility Assessment
- **Score**: {analysis.reproducibility_score:.2f}/1.0
- **Grade**: {self._score_to_grade(analysis.reproducibility_score)}
- **Code Repository**: {'✓ ' + analysis.code_repo_url if analysis.has_code_repo else '✗ Not found'}

### Methods Identified ({len(analysis.methods)} sections)
"""
        
        all_techniques = set()
        all_params = {}
        
        for method in analysis.methods:
            all_techniques.update(method.techniques)
            all_params.update(method.parameters)
        
        summary += f"\n**Techniques**: {', '.join(sorted(all_techniques)[:10])}\n"
        
        summary += f"\n**Key Parameters**:\n"
        for param, value in list(all_params.items())[:10]:
            summary += f"- {param}: {value}\n"
        
        summary += f"\n### Code & Software\n"
        summary += f"- **Code snippets**: {len(analysis.code_snippets)}\n"
        summary += f"- **Complete code blocks**: {sum(1 for s in analysis.code_snippets if s.is_complete)}\n"
        summary += f"- **Software mentioned**: {', '.join(analysis.software_mentioned[:10])}\n"
        
        summary += f"\n### Data\n"
        summary += f"- **Datasets**: {', '.join(analysis.datasets_mentioned) if analysis.datasets_mentioned else 'None mentioned'}\n"
        
        return summary


def demo():
    """演示"""
    print("=" * 60)
    print("Paper Analyzer Demo")
    print("=" * 60)
    
    # 示例论文文本
    sample_paper = """
    Title: Deep Learning for Crystal Structure Prediction: A Transformer Approach
    
    Authors: John Smith, Jane Doe, Alice Wang
    
    Abstract: We present a novel transformer-based approach for predicting crystal structures 
    from chemical composition. Our method achieves state-of-the-art results on the Materials Project dataset.
    
    1. Introduction
    Crystal structure prediction is a fundamental challenge in materials science...
    
    2. Methods
    
    2.1 Model Architecture
    We use a transformer encoder with 8 attention heads and 6 layers.
    The embedding dimension is 512.
    
    2.2 Training
    We train with Adam optimizer with learning rate = 1e-4 and batch size = 32.
    Training for 100 epochs on 4 NVIDIA V100 GPUs.
    
    Code implementation:
    ```python
    import torch
    import torch.nn as nn
    
    class CrystalTransformer(nn.Module):
        def __init__(self, d_model=512, nhead=8):
            super().__init__()
            self.encoder = nn.TransformerEncoderLayer(d_model, nhead)
        
        def forward(self, x):
            return self.encoder(x)
    
    if __name__ == '__main__':
        model = CrystalTransformer()
    ```
    
    2.3 DFT Calculations
    We use VASP with PBE functional, energy cutoff = 520 eV, and k-point mesh 8x8x8.
    
    3. Results
    Our model achieves 85% accuracy on structure prediction.
    
    Code available at: https://github.com/example/crystal-transformer
    
    Data from Materials Project and OQMD databases.
    """
    
    # 分析论文
    analyzer = PaperAnalyzer()
    analysis = analyzer.analyze_paper(
        sample_paper,
        paper_title="Deep Learning for Crystal Structure Prediction",
        authors=["John Smith", "Jane Doe", "Alice Wang"]
    )
    
    # 输出分析结果
    print("\nAnalysis Results:")
    print("-" * 40)
    
    print(f"\nTitle: {analysis.title}")
    print(f"Methods found: {len(analysis.methods)}")
    print(f"Code snippets: {len(analysis.code_snippets)}")
    print(f"Code repository: {analysis.code_repo_url or 'Not found'}")
    print(f"Reproducibility score: {analysis.reproducibility_score:.2f}")
    
    print(f"\nTechniques identified:")
    for method in analysis.methods:
        for tech in method.techniques[:5]:
            print(f"  - {tech}")
    
    print(f"\nSoftware mentioned: {analysis.software_mentioned}")
    print(f"Datasets mentioned: {analysis.datasets_mentioned}")
    
    # 可复现性评估
    print("\n" + "-" * 40)
    assessment = analyzer.assess_reproducibility(analysis)
    print(f"\nReproducibility Grade: {assessment['grade']}")
    print(f"\nStrengths:")
    for s in assessment['strengths']:
        print(f"  ✓ {s}")
    
    print(f"\nRecommendations:")
    for r in assessment['recommendations']:
        print(f"  → {r}")
    
    # 生成完整摘要
    print("\n" + "=" * 60)
    print("Full Summary:")
    print("=" * 60)
    print(analyzer.generate_summary(analysis))
    
    print("\nDemo completed!")


if __name__ == "__main__":
    demo()
