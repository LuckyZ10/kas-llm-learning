"""
方法提取模块
自动识别论文中的方法、软件、数据集
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter

from ..config.models import Paper, MethodComparison


class MethodExtractor:
    """方法提取器"""
    
    # 计算软件列表
    SOFTWARE_PATTERNS = {
        "VASP": r'\bVASP\b',
        "Quantum ESPRESSO": r'Quantum\s+ESPRESSO',
        "Gaussian": r'\bGaussian\b(?!\s+process)',
        "ABINIT": r'\bABINIT\b',
        "CASTEP": r'\bCASTEP\b',
        "GPAW": r'\bGPAW\b',
        "Siesta": r'\bSiesta\b',
        "WIEN2k": r'\bWIEN2k\b',
        "LAMMPS": r'\bLAMMPS\b',
        "GROMACS": r'\bGROMACS\b',
        "NAMD": r'\bNAMD\b',
        "OpenMM": r'\bOpenMM\b',
        "ASE": r'\bASE\b',
        "Pymatgen": r'\bPymatgen\b',
        "OVITO": r'\bOVITO\b',
        "Amber": r'\bAmber\b',
        "CP2K": r'\bCP2K\b',
        "ATK": r'\bATK\b',
        "MedeA": r'\bMedeA\b',
        "Materials Studio": r'Materials\s+Studio',
        "COMSOL": r'\bCOMSOL\b',
        "Lammps": r'\blammps\b'
    }
    
    # 计算方法列表
    METHOD_PATTERNS = {
        "DFT": r'\bDFT\b|\bdensity\s+functional\b',
        "DFT-D": r'\bDFT-D\d?\b',
        "vdW-DF": r'\bvdW-DF\d?\b',
        "SCAN": r'\bSCAN\b(?!\s+for)',
        "PBE": r'\bPBE\b|\bPBEsol\b',
        "HSE06": r'\bHSE06?\b',
        "B3LYP": r'\bB3LYP\b',
        "GW": r'\bGW\b(?!\s+end)',
        "AIMD": r'\bAIMD\b|\bab\s*initio\s+MD\b',
        "DFT-MD": r'\bDFT[-\s]?MD\b',
        "Classical MD": r'\bclassical\s+MD\b|\bclassical\s+molecular\s+dynamics\b',
        "NEB": r'\bNEB\b|\bnudged\s+elastic\s+band\b',
        "CI-NEB": r'\bCI[-\s]?NEB\b',
        "Metadynamics": r'\bmetadynamics\b',
        "Umbrella Sampling": r'\bumbrella\s+sampling\b',
        "Cluster Expansion": r'\bcluster\s+expansion\b',
        "CALPHAD": r'\bCALPHAD\b',
        "KMC": r'\bKMC\b|\bkinetic\s+Monte\s+Carlo\b',
        "Monte Carlo": r'\bMonte\s+Carlo\b',
        "Machine Learning": r'\bmachine\s+learning\b',
        "Deep Learning": r'\bdeep\s+learning\b',
        "Neural Network": r'\bneural\s+network\b'
    }
    
    # 数据集和基准
    DATASET_PATTERNS = {
        "Materials Project": r'Materials\s+Project',
        "AFLOW": r'\bAFLOW\b',
        "OQMD": r'\bOQMD\b',
        "NIST": r'\bNIST\b',
        "MP": r'\bMP\b(?!\s+for)',
    }
    
    # 性能指标
    METRIC_PATTERNS = {
        "accuracy": r'\baccuracy\b|\baccurate\b',
        "precision": r'\bprecision\b|\bprecise\b',
        "efficiency": r'\befficiency\b|\befficient\b',
        "speed": r'\bspeed\b|\bfast\b',
        "scalability": r'\bscalability\b|\bscalable\b'
    }
    
    def __init__(self):
        self.software_regex = {k: re.compile(v, re.IGNORECASE) 
                              for k, v in self.SOFTWARE_PATTERNS.items()}
        self.method_regex = {k: re.compile(v, re.IGNORECASE) 
                            for k, v in self.METHOD_PATTERNS.items()}
        self.dataset_regex = {k: re.compile(v, re.IGNORECASE) 
                             for k, v in self.DATASET_PATTERNS.items()}
    
    def extract_from_paper(self, paper: Paper) -> Dict[str, List[str]]:
        """
        从单篇论文提取方法信息
        
        Returns:
            提取的方法信息字典
        """
        # 合并文本
        text = f"{paper.title} {paper.abstract}"
        if paper.full_text:
            text += f" {paper.full_text}"
        
        # 提取各类信息
        software = self._extract_software(text)
        methods = self._extract_methods(text)
        datasets = self._extract_datasets(text)
        
        # 提取计算细节
        details = self._extract_computational_details(text)
        
        return {
            "software": list(software),
            "methods": list(methods),
            "datasets": list(datasets),
            "details": details
        }
    
    def _extract_software(self, text: str) -> Set[str]:
        """提取软件"""
        found = set()
        for name, pattern in self.software_regex.items():
            if pattern.search(text):
                found.add(name)
        return found
    
    def _extract_methods(self, text: str) -> Set[str]:
        """提取方法"""
        found = set()
        for name, pattern in self.method_regex.items():
            if pattern.search(text):
                found.add(name)
        return found
    
    def _extract_datasets(self, text: str) -> Set[str]:
        """提取数据集"""
        found = set()
        for name, pattern in self.dataset_regex.items():
            if pattern.search(text):
                found.add(name)
        return found
    
    def _extract_computational_details(self, text: str) -> Dict[str, str]:
        """提取计算细节"""
        details = {}
        
        # 能量截断
        cutoff_match = re.search(
            r'(energy\s+cutoff|kinetic\s+energy\s+cutoff|ENCUT)[\s:=]+(\d+)\s*(eV|Ry|Ha|Hartree)',
            text,
            re.IGNORECASE
        )
        if cutoff_match:
            details["energy_cutoff"] = f"{cutoff_match.group(2)} {cutoff_match.group(3)}"
        
        # k点网格
        kpoint_match = re.search(
            r'(\d+)\s*[×x×,]\s*(\d+)\s*[×x×,]\s*(\d+)\s*(k[-\s]?point|k[-\s]?mesh|Monkhorst)',
            text,
            re.IGNORECASE
        )
        if kpoint_match:
            details["kpoint_grid"] = f"{kpoint_match.group(1)}×{kpoint_match.group(2)}×{kpoint_match.group(3)}"
        
        # 收敛标准
        conv_match = re.search(
            r'(convergence\s+criterion|tolerance)[\s:=]+([\d.eE-]+)\s*(eV|Ry|Ha)?',
            text,
            re.IGNORECASE
        )
        if conv_match:
            details["convergence"] = conv_match.group(2)
        
        # 时间步长（MD）
        timestep_match = re.search(
            r'(time\s*step|timestep|dt)[\s:=]+([\d.]+)\s*(fs|femtosecond)',
            text,
            re.IGNORECASE
        )
        if timestep_match:
            details["timestep"] = f"{timestep_match.group(2)} fs"
        
        # 温度
        temp_match = re.search(
            r'(temperature|at\s+)([\d.]+)\s*K',
            text,
            re.IGNORECASE
        )
        if temp_match:
            details["temperature"] = f"{temp_match.group(2)} K"
        
        return details
    
    def analyze_methods(
        self,
        papers: List[Paper]
    ) -> List[MethodComparison]:
        """
        分析方法使用情况
        
        Returns:
            方法对比列表
        """
        method_stats = defaultdict(lambda: {
            "count": 0,
            "papers": [],
            "software": Counter(),
            "datasets": Counter()
        })
        
        for paper in papers:
            info = self.extract_from_paper(paper)
            
            for method in info["methods"]:
                method_stats[method]["count"] += 1
                method_stats[method]["papers"].append(paper.id)
                
                for sw in info["software"]:
                    method_stats[method]["software"][sw] += 1
                
                for ds in info["datasets"]:
                    method_stats[method]["datasets"][ds] += 1
        
        comparisons = []
        for method, stats in method_stats.items():
            comparison = MethodComparison(
                method_name=method,
                paper_count=stats["count"],
                datasets_used=list(stats["datasets"].keys()),
                software_used=list(stats["software"].keys()),
                pros=self._infer_pros(method, stats),
                cons=self._infer_cons(method, stats)
            )
            comparisons.append(comparison)
        
        return sorted(comparisons, key=lambda x: x.paper_count, reverse=True)
    
    def _infer_pros(self, method: str, stats: Dict) -> List[str]:
        """推断方法优点"""
        pros = []
        
        if stats["count"] > 10:
            pros.append("Widely adopted in the community")
        
        if method in ["DFT", "PBE", "GGA"]:
            pros.append("Computationally efficient")
            pros.append("Well-established methodology")
        
        if method in ["HSE06", "SCAN", "GW"]:
            pros.append("High accuracy for electronic properties")
        
        if method in ["Machine Learning", "Deep Learning", "Neural Network"]:
            pros.append("Fast prediction once trained")
            pros.append("Can handle large systems")
        
        if method in ["AIMD", "DFT-MD"]:
            pros.append("Ab initio accuracy")
            pros.append("No empirical parameters needed")
        
        if method in ["NEB", "CI-NEB"]:
            pros.append("Standard method for barrier calculations")
        
        return pros
    
    def _infer_cons(self, method: str, stats: Dict) -> List[str]:
        """推断方法缺点"""
        cons = []
        
        if method in ["DFT", "PBE", "GGA"]:
            cons.append("Band gap underestimation")
            cons.append("Weak van der Waals treatment")
        
        if method in ["HSE06", "SCAN", "GW"]:
            cons.append("Computationally expensive")
        
        if method in ["Machine Learning", "Deep Learning"]:
            cons.append("Requires large training datasets")
            cons.append("Limited transferability")
        
        if method in ["Classical MD", "Force Field"]:
            cons.append("Accuracy depends on force field quality")
            cons.append("Limited to specific chemical systems")
        
        if method in ["AIMD", "DFT-MD"]:
            cons.append("Limited simulation time scales")
            cons.append("High computational cost")
        
        return cons
    
    def extract_comparison_table(self, text: str) -> List[Dict]:
        """从文本提取方法对比表"""
        comparisons = []
        
        # 匹配对比描述
        patterns = [
            r'(\w+(?:\s+\w+)?)\s+(?:is|provides|offers)\s+(\w+)\s+(?:than|compared to)\s+(\w+)',
            r'compared\s+with\s+(\w+),\s+(\w+)\s+(?:is|has|shows)\s+(\w+)',
            r'(\w+)\s+outperform[s]?\s+(\w+)\s+(?:in|for|with)\s+([^,.]+)'
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                comparisons.append({
                    "method1": match.group(1),
                    "comparison": match.group(2),
                    "method2": match.group(3) if len(match.groups()) > 2 else None,
                    "context": match.group(0)
                })
        
        return comparisons


class PerformanceExtractor:
    """性能指标提取器"""
    
    # 性能指标模式
    METRIC_PATTERNS = {
        "accuracy": [
            r'accuracy\s*(?:of|is|=)\s*([\d.]+%?|\d+\.\d+)',
            r'([\d.]+%?)\s+accuracy'
        ],
        "rmse": [
            r'RMSE\s*(?:of|is|=)\s*([\d.]+)\s*(eV|meV|kcal|kJ)?',
            r'root\s+mean\s+square\s+error\s*:?\s*([\d.]+)'
        ],
        "mae": [
            r'MAE\s*(?:of|is|=)\s*([\d.]+)\s*(eV|meV|kcal|kJ)?',
            r'mean\s+absolute\s+error\s*:?\s*([\d.]+)'
        ],
        "r2": [
            r'R[²^22]\s*(?:of|is|=)\s*([\d.]+)',
            r'R-squared\s*:?\s*([\d.]+)'
        ],
        "computational_time": [
            r'(?:takes?|requires?|costs?)\s+([\d.]+)\s*(s|sec|min|hour|day)',
            r'computational\s+time\s*:?\s*([\d.]+)\s*(s|sec|min|hour)'
        ]
    }
    
    def extract_metrics(self, text: str) -> Dict[str, List[str]]:
        """提取性能指标"""
        metrics = defaultdict(list)
        
        for metric_name, patterns in self.METRIC_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    metrics[metric_name].append(match.group(0))
        
        return dict(metrics)
    
    def compare_performance(
        self,
        papers: List[Paper],
        metric_name: str
    ) -> List[Tuple[str, float]]:
        """
        比较论文性能
        
        Returns:
            论文ID和性能值列表
        """
        results = []
        
        for paper in papers:
            text = f"{paper.abstract} {paper.full_text or ''}"
            metrics = self.extract_metrics(text)
            
            if metric_name in metrics and metrics[metric_name]:
                # 尝试提取数值
                for value_str in metrics[metric_name]:
                    try:
                        # 提取数字
                        numbers = re.findall(r'\d+\.?\d*', value_str)
                        if numbers:
                            value = float(numbers[0])
                            results.append((paper.id, value))
                            break
                    except:
                        continue
        
        # 排序（假设越小越好）
        return sorted(results, key=lambda x: x[1])
