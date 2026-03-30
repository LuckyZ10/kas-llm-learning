"""
XRD数据分析模块 - XRD Analysis Module

实现：
- XRD数据解析
- 物相识别
- 晶体结构分析
- Rietveld精修接口
- 计算-实验对比
"""

import os
import json
import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from pathlib import Path
from scipy import signal, optimize
from scipy.interpolate import interp1d
import warnings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class XRDPeak:
    """XRD峰数据"""
    two_theta: float
    intensity: float
    fwhm: float  # 半高宽
    d_spacing: float
    hkl: Optional[Tuple[int, int, int]] = None
    phase: Optional[str] = None
    area: float = 0.0
    background: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "two_theta": self.two_theta,
            "intensity": self.intensity,
            "fwhm": self.fwhm,
            "d_spacing": self.d_spacing,
            "hkl": self.hkl,
            "phase": self.phase,
            "area": self.area
        }


@dataclass
class XRDPattern:
    """XRD衍射图谱"""
    two_theta: np.ndarray
    intensity: np.ndarray
    wavelength: float = 1.5406  # Cu K-alpha, Angstrom
    sample_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    peaks: List[XRDPeak] = field(default_factory=list)
    phases: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if len(self.two_theta) != len(self.intensity):
            raise ValueError("two_theta and intensity must have same length")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "two_theta": self.two_theta.tolist(),
            "intensity": self.intensity.tolist(),
            "wavelength": self.wavelength,
            "sample_name": self.sample_name,
            "metadata": self.metadata,
            "peaks": [p.to_dict() for p in self.peaks],
            "phases": self.phases
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'XRDPattern':
        """从字典创建"""
        pattern = cls(
            two_theta=np.array(data["two_theta"]),
            intensity=np.array(data["intensity"]),
            wavelength=data.get("wavelength", 1.5406),
            sample_name=data.get("sample_name", ""),
            metadata=data.get("metadata", {})
        )
        pattern.peaks = [XRDPeak(**p) for p in data.get("peaks", [])]
        pattern.phases = data.get("phases", [])
        return pattern
    
    def calculate_d_spacing(self, two_theta: float) -> float:
        """计算晶面间距 d = λ / (2sinθ)"""
        theta_rad = np.radians(two_theta / 2)
        return self.wavelength / (2 * np.sin(theta_rad))


@dataclass
class PhaseIdentification:
    """物相识别结果"""
    phase_name: str
    formula: str
    space_group: str
    match_score: float  # 0-1匹配分数
    matched_peaks: List[int] = field(default_factory=list)  # 匹配峰索引
    lattice_params: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "formula": self.formula,
            "space_group": self.space_group,
            "match_score": self.match_score,
            "lattice_params": self.lattice_params,
            "confidence": self.confidence
        }


class XRDParser:
    """XRD数据解析器"""
    
    def __init__(self):
        self.parsers = {
            ".xy": self._parse_xy,
            ".xye": self._parse_xy,
            ".dat": self._parse_dat,
            ".raw": self._parse_brucker_raw,
            ".ras": self._parse_rigaku,
            ".csv": self._parse_csv,
            ".txt": self._parse_txt,
            ".xrdml": self._parse_panalytical,
            ".gsas": self._parse_gsas
        }
    
    def parse(self, filepath: str, file_format: Optional[str] = None) -> XRDPattern:
        """解析XRD数据文件"""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # 自动检测格式
        if file_format is None:
            file_format = path.suffix.lower()
        
        parser = self.parsers.get(file_format)
        if parser is None:
            # 尝试通用解析
            parser = self._parse_generic
        
        return parser(filepath)
    
    def _parse_xy(self, filepath: str) -> XRDPattern:
        """解析XY格式"""
        two_theta = []
        intensity = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('%'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        two_theta.append(float(parts[0]))
                        intensity.append(float(parts[1]))
                    except ValueError:
                        continue
        
        return XRDPattern(
            two_theta=np.array(two_theta),
            intensity=np.array(intensity),
            sample_name=Path(filepath).stem
        )
    
    def _parse_dat(self, filepath: str) -> XRDPattern:
        """解析DAT格式"""
        return self._parse_xy(filepath)
    
    def _parse_csv(self, filepath: str) -> XRDPattern:
        """解析CSV格式"""
        import csv
        
        two_theta = []
        intensity = []
        
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2 and not row[0].startswith('#'):
                    try:
                        two_theta.append(float(row[0]))
                        intensity.append(float(row[1]))
                    except ValueError:
                        continue
        
        return XRDPattern(
            two_theta=np.array(two_theta),
            intensity=np.array(intensity),
            sample_name=Path(filepath).stem
        )
    
    def _parse_txt(self, filepath: str) -> XRDPattern:
        """解析TXT格式（通用）"""
        two_theta = []
        intensity = []
        wavelength = 1.5406
        
        with open(filepath, 'r') as f:
            content = f.read()
            
            # 尝试提取波长信息
            if 'wavelength' in content.lower():
                import re
                match = re.search(r'wavelength[:\s]+([\d.]+)', content, re.IGNORECASE)
                if match:
                    wavelength = float(match.group(1))
            
            # 解析数据行
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        val1 = float(parts[0])
                        val2 = float(parts[1])
                        # 判断哪个是2θ
                        if 0 <= val1 <= 180 and val2 >= 0:
                            two_theta.append(val1)
                            intensity.append(val2)
                    except ValueError:
                        continue
        
        return XRDPattern(
            two_theta=np.array(two_theta),
            intensity=np.array(intensity),
            wavelength=wavelength,
            sample_name=Path(filepath).stem
        )
    
    def _parse_brucker_raw(self, filepath: str) -> XRDPattern:
        """解析Bruker RAW格式"""
        # Bruker RAW是二进制格式，这里简化为文本解析
        # 实际实现需要使用专门的库
        logger.warning("Bruker RAW parsing is simplified, use official software for best results")
        return self._parse_generic(filepath)
    
    def _parse_rigaku(self, filepath: str) -> XRDPattern:
        """解析Rigaku格式"""
        return self._parse_xy(filepath)
    
    def _parse_panalytical(self, filepath: str) -> XRDPattern:
        """解析PANalytical XRDML格式"""
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # 提取数据
            ns = {'xrdml': 'http://www.xrdml.com/XRDMeasurement/1.0'}
            
            positions = root.find('.//xrdml:positions', ns)
            intensities = root.find('.//xrdml:intensities', ns)
            
            if positions is not None and intensities is not None:
                two_theta = np.array([float(x) for x in positions.text.split()])
                intensity = np.array([float(x) for x in intensities.text.split()])
                
                return XRDPattern(
                    two_theta=two_theta,
                    intensity=intensity,
                    sample_name=Path(filepath).stem
                )
        except Exception as e:
            logger.error(f"Error parsing XRDML: {e}")
        
        return self._parse_generic(filepath)
    
    def _parse_gsas(self, filepath: str) -> XRDPattern:
        """解析GSAS格式"""
        # GSAS格式较为复杂，这里简化处理
        return self._parse_generic(filepath)
    
    def _parse_generic(self, filepath: str) -> XRDPattern:
        """通用解析（尝试多种格式）"""
        # 尝试XY格式
        try:
            return self._parse_xy(filepath)
        except:
            pass
        
        # 尝试CSV格式
        try:
            return self._parse_csv(filepath)
        except:
            pass
        
        # 最后尝试TXT格式
        return self._parse_txt(filepath)


class PeakDetector:
    """XRD峰检测器"""
    
    def __init__(self, 
                 prominence: float = 0.05,
                 width: Optional[Tuple[float, float]] = None,
                 min_intensity: float = 0.01):
        self.prominence = prominence
        self.width = width or (0.05, 2.0)  # 2θ范围
        self.min_intensity = min_intensity
        
    def detect_peaks(self, pattern: XRDPattern, 
                     background_subtract: bool = True) -> List[XRDPeak]:
        """检测衍射峰"""
        intensity = pattern.intensity.copy()
        
        # 背景扣除
        if background_subtract:
            background = self._estimate_background(pattern)
            intensity = intensity - background
            intensity[intensity < 0] = 0
        
        # 归一化
        intensity_normalized = intensity / np.max(intensity)
        
        # 检测峰
        peaks_indices, properties = signal.find_peaks(
            intensity_normalized,
            prominence=self.prominence,
            width=self.width
        )
        
        peaks = []
        for i, idx in enumerate(peaks_indices):
            # 计算FWHM
            fwhm = self._calculate_fwhm(pattern.two_theta, intensity, idx)
            
            # 计算峰面积
            area = self._calculate_peak_area(pattern.two_theta, intensity, idx, fwhm)
            
            peak = XRDPeak(
                two_theta=pattern.two_theta[idx],
                intensity=intensity[idx],
                fwhm=fwhm,
                d_spacing=pattern.calculate_d_spacing(pattern.two_theta[idx]),
                area=area,
                background=background[idx] if background_subtract else 0
            )
            
            peaks.append(peak)
        
        # 按强度排序
        peaks.sort(key=lambda p: p.intensity, reverse=True)
        
        return peaks
    
    def _estimate_background(self, pattern: XRDPattern, 
                            iterations: int = 10) -> np.ndarray:
        """使用SNIP算法估计背景"""
        intensity = pattern.intensity.copy()
        
        for _ in range(iterations):
            new_intensity = intensity.copy()
            for i in range(1, len(intensity) - 1):
                avg = (intensity[i-1] + intensity[i+1]) / 2
                new_intensity[i] = min(intensity[i], avg)
            intensity = new_intensity
        
        return intensity
    
    def _calculate_fwhm(self, two_theta: np.ndarray, 
                       intensity: np.ndarray, 
                       peak_idx: int) -> float:
        """计算半高宽"""
        peak_intensity = intensity[peak_idx]
        half_max = peak_intensity / 2
        
        # 找到左右半高点
        left_idx = peak_idx
        right_idx = peak_idx
        
        while left_idx > 0 and intensity[left_idx] > half_max:
            left_idx -= 1
        
        while right_idx < len(intensity) - 1 and intensity[right_idx] > half_max:
            right_idx += 1
        
        # 线性插值更精确的位置
        if left_idx > 0 and left_idx < len(intensity) - 1:
            t1 = two_theta[left_idx]
            t2 = two_theta[left_idx + 1]
            i1 = intensity[left_idx]
            i2 = intensity[left_idx + 1]
            if i2 != i1:
                left_theta = t1 + (t2 - t1) * (half_max - i1) / (i2 - i1)
            else:
                left_theta = two_theta[left_idx]
        else:
            left_theta = two_theta[left_idx]
        
        if right_idx > 0 and right_idx < len(intensity) - 1:
            t1 = two_theta[right_idx - 1]
            t2 = two_theta[right_idx]
            i1 = intensity[right_idx - 1]
            i2 = intensity[right_idx]
            if i2 != i1:
                right_theta = t1 + (t2 - t1) * (half_max - i1) / (i2 - i1)
            else:
                right_theta = two_theta[right_idx]
        else:
            right_theta = two_theta[right_idx]
        
        return right_theta - left_theta
    
    def _calculate_peak_area(self, two_theta: np.ndarray,
                            intensity: np.ndarray,
                            peak_idx: int,
                            fwhm: float) -> float:
        """计算峰面积"""
        # 使用3倍FWHM作为积分范围
        window = int(3 * fwhm / (two_theta[1] - two_theta[0]))
        left = max(0, peak_idx - window)
        right = min(len(two_theta), peak_idx + window)
        
        return np.trapz(intensity[left:right], two_theta[left:right])


class PhaseIdentifier:
    """物相识别器"""
    
    def __init__(self, database_path: Optional[str] = None):
        self.database = self._load_database(database_path)
        
    def _load_database(self, path: Optional[str]) -> Dict[str, Any]:
        """加载物相数据库"""
        # 简化的内置数据库
        default_db = {
            "Li3PS4": {
                "formula": "Li3PS4",
                "space_group": "Pmn21",
                "lattice_params": {"a": 7.71, "b": 6.14, "c": 6.47},
                "peaks": [
                    {"hkl": (1,0,1), "two_theta": 17.5, "intensity": 100},
                    {"hkl": (1,1,1), "two_theta": 20.2, "intensity": 80},
                    {"hkl": (2,0,0), "two_theta": 25.1, "intensity": 60}
                ]
            },
            "Li7P3S11": {
                "formula": "Li7P3S11",
                "space_group": "P21/n",
                "lattice_params": {"a": 12.49, "b": 5.88, "c": 12.49, "beta": 109.1},
                "peaks": [
                    {"hkl": (2,0,0), "two_theta": 14.2, "intensity": 100},
                    {"hkl": (1,1,1), "two_theta": 18.5, "intensity": 70}
                ]
            },
            "Li2S": {
                "formula": "Li2S",
                "space_group": "Fm-3m",
                "lattice_params": {"a": 5.71},
                "peaks": [
                    {"hkl": (1,1,1), "two_theta": 26.5, "intensity": 100},
                    {"hkl": (2,0,0), "two_theta": 30.8, "intensity": 80}
                ]
            },
            "Si": {
                "formula": "Si",
                "space_group": "Fd-3m",
                "lattice_params": {"a": 5.43},
                "peaks": [
                    {"hkl": (1,1,1), "two_theta": 28.44, "intensity": 100},
                    {"hkl": (2,2,0), "two_theta": 47.30, "intensity": 60}
                ]
            },
            "TiO2_rutile": {
                "formula": "TiO2",
                "space_group": "P42/mnm",
                "lattice_params": {"a": 4.59, "c": 2.96},
                "peaks": [
                    {"hkl": (1,1,0), "two_theta": 27.45, "intensity": 100},
                    {"hkl": (1,0,1), "two_theta": 36.09, "intensity": 50}
                ]
            },
            "TiO2_anatase": {
                "formula": "TiO2",
                "space_group": "I41/amd",
                "lattice_params": {"a": 3.78, "c": 9.51},
                "peaks": [
                    {"hkl": (1,0,1), "two_theta": 25.28, "intensity": 100},
                    {"hkl": (0,0,4), "two_theta": 37.80, "intensity": 20}
                ]
            }
        }
        
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                loaded_db = json.load(f)
                default_db.update(loaded_db)
        
        return default_db
    
    def identify_phases(self, pattern: XRDPattern, 
                       top_n: int = 5,
                       min_match_score: float = 0.3) -> List[PhaseIdentification]:
        """识别物相"""
        # 检测峰
        detector = PeakDetector()
        if not pattern.peaks:
            pattern.peaks = detector.detect_peaks(pattern)
        
        results = []
        
        for phase_name, phase_data in self.database.items():
            match_score, matched_peaks = self._match_phase(pattern, phase_data)
            
            if match_score >= min_match_score:
                result = PhaseIdentification(
                    phase_name=phase_name,
                    formula=phase_data["formula"],
                    space_group=phase_data.get("space_group", "Unknown"),
                    match_score=match_score,
                    matched_peaks=matched_peaks,
                    lattice_params=phase_data.get("lattice_params", {}),
                    confidence=match_score
                )
                results.append(result)
        
        # 按匹配分数排序
        results.sort(key=lambda r: r.match_score, reverse=True)
        
        return results[:top_n]
    
    def _match_phase(self, pattern: XRDPattern, 
                    phase_data: Dict[str, Any]) -> Tuple[float, List[int]]:
        """匹配单个物相"""
        db_peaks = phase_data.get("peaks", [])
        pattern_peaks = pattern.peaks
        
        if not db_peaks or not pattern_peaks:
            return 0.0, []
        
        matched = []
        total_intensity = 0
        matched_intensity = 0
        
        for db_peak in db_peaks:
            db_2theta = db_peak["two_theta"]
            db_intensity = db_peak.get("intensity", 100)
            total_intensity += db_intensity
            
            # 在实验图谱中寻找匹配的峰
            for i, peak in enumerate(pattern_peaks):
                if abs(peak.two_theta - db_2theta) < 0.5:  # 允许0.5度误差
                    matched.append(i)
                    matched_intensity += db_intensity
                    break
        
        # 计算匹配分数
        if total_intensity > 0:
            match_score = matched_intensity / total_intensity
        else:
            match_score = 0.0
        
        # 考虑匹配峰数量
        peak_match_ratio = len(matched) / len(db_peaks)
        match_score = (match_score + peak_match_ratio) / 2
        
        return match_score, matched
    
    def calculate_lattice_parameters(self, pattern: XRDPattern,
                                    phase: PhaseIdentification) -> Dict[str, float]:
        """根据峰位计算晶格参数"""
        # 简化实现，实际需要精修算法
        lattice_params = {}
        
        if phase.space_group in ["Fm-3m", "Fd-3m", "Im-3m"]:
            # 立方晶系: 1/d² = (h²+k²+l²)/a²
            a_values = []
            for peak in pattern.peaks:
                if peak.hkl:
                    h, k, l = peak.hkl
                    d = peak.d_spacing
                    a = d * np.sqrt(h**2 + k**2 + l**2)
                    a_values.append(a)
            
            if a_values:
                lattice_params["a"] = np.mean(a_values)
                lattice_params["a_std"] = np.std(a_values)
        
        return lattice_params


class XRDAnalyzer:
    """XRD综合分析器"""
    
    def __init__(self):
        self.parser = XRDParser()
        self.detector = PeakDetector()
        self.identifier = PhaseIdentifier()
    
    def analyze(self, filepath: str, 
                file_format: Optional[str] = None) -> Dict[str, Any]:
        """完整分析XRD数据"""
        # 解析数据
        pattern = self.parser.parse(filepath, file_format)
        
        # 检测峰
        pattern.peaks = self.detector.detect_peaks(pattern)
        
        # 识别物相
        phases = self.identifier.identify_phases(pattern)
        pattern.phases = [p.phase_name for p in phases]
        
        # 计算统计信息
        stats = self._calculate_statistics(pattern)
        
        return {
            "pattern": pattern.to_dict(),
            "phases": [p.to_dict() for p in phases],
            "statistics": stats,
            "summary": self._generate_summary(pattern, phases)
        }
    
    def _calculate_statistics(self, pattern: XRDPattern) -> Dict[str, Any]:
        """计算统计信息"""
        return {
            "num_data_points": len(pattern.two_theta),
            "two_theta_range": [float(pattern.two_theta[0]), 
                               float(pattern.two_theta[-1])],
            "max_intensity": float(np.max(pattern.intensity)),
            "num_peaks": len(pattern.peaks),
            "avg_fwhm": np.mean([p.fwhm for p in pattern.peaks]) if pattern.peaks else 0,
            "avg_intensity": float(np.mean(pattern.intensity))
        }
    
    def _generate_summary(self, pattern: XRDPattern, 
                         phases: List[PhaseIdentification]) -> str:
        """生成分析摘要"""
        summary = f"XRD Analysis Summary for {pattern.sample_name}\n"
        summary += f"Detected {len(pattern.peaks)} peaks\n"
        
        if phases:
            summary += f"Identified {len(phases)} phase(s):\n"
            for i, phase in enumerate(phases[:3], 1):
                summary += f"  {i}. {phase.phase_name} ({phase.formula}) - "
                summary += f"Match: {phase.match_score:.2%}\n"
        else:
            summary += "No phases identified with confidence > 30%\n"
        
        return summary
    
    def compare_patterns(self, pattern1: XRDPattern, 
                        pattern2: XRDPattern) -> Dict[str, float]:
        """比较两个XRD图谱"""
        # 插值到相同网格
        common_2theta = np.linspace(
            max(pattern1.two_theta[0], pattern2.two_theta[0]),
            min(pattern1.two_theta[-1], pattern2.two_theta[-1]),
            1000
        )
        
        f1 = interp1d(pattern1.two_theta, pattern1.intensity, 
                     kind='cubic', bounds_error=False, fill_value=0)
        f2 = interp1d(pattern2.two_theta, pattern2.intensity,
                     kind='cubic', bounds_error=False, fill_value=0)
        
        int1 = f1(common_2theta)
        int2 = f2(common_2theta)
        
        # 归一化
        int1 = int1 / np.max(int1) if np.max(int1) > 0 else int1
        int2 = int2 / np.max(int2) if np.max(int2) > 0 else int2
        
        # 计算差异指标
        correlation = np.corrcoef(int1, int2)[0, 1]
        rmse = np.sqrt(np.mean((int1 - int2)**2))
        
        # 计算Rwp (Rietveld weighted pattern)
        diff = int1 - int2
        Rwp = np.sqrt(np.sum(diff**2) / np.sum(int1**2))
        
        return {
            "correlation": float(correlation),
            "rmse": float(rmse),
            "Rwp": float(Rwp),
            "similarity": float((correlation + 1) / 2)  # 归一化到0-1
        }


class RietveldRefinement:
    """Rietveld精修接口"""
    
    def __init__(self):
        self.initial_params = {}
        
    def setup_refinement(self, pattern: XRDPattern, 
                        phases: List[PhaseIdentification]) -> Dict[str, Any]:
        """设置精修参数"""
        params = {
            "phases": [],
            "background": {
                "type": "chebyshev",
                "coefficients": [1.0, 0.1, 0.01]
            },
            "instrument": {
                "wavelength": pattern.wavelength,
                "zero_shift": 0.0
            },
            "profile": {
                "type": "pseudo_voigt",
                "U": 0.01,
                "V": -0.01,
                "W": 0.005
            }
        }
        
        for phase in phases:
            phase_params = {
                "name": phase.phase_name,
                "space_group": phase.space_group,
                "lattice_params": phase.lattice_params,
                "scale_factor": 1.0,
                "fractional_coordinates": {}
            }
            params["phases"].append(phase_params)
        
        return params
    
    def generate_gsas_script(self, params: Dict[str, Any], 
                            output_path: str):
        """生成GSAS-II脚本"""
        script = """
# GSAS-II Rietveld Refinement Script
import GSASIIscriptable as G2sc

# Load project
gsas_project = G2sc.G2Project(new=True)

# Add histogram
hist = gsas_project.add_histogram('data.xy', fmthint='XYE')

# Add phase(s)
"""
        for phase in params["phases"]:
            script += f"""
phase = gsas_project.add_phase(
    phasefile='{phase['name']}.cif',
    phasename='{phase['name']}',
    histograms=[hist]
)
"""
        
        script += """
# Set up refinement
refinement = {
    'set': {
        'Background': {'refine': True},
        'Instrument Parameters': {'refine': True},
        'Phase': {'scale': True, 'cell': True}
    }
}

# Run refinement
gsas_project.do_refinements(refinement)

# Save results
gsas_project.save()
"""
        
        with open(output_path, 'w') as f:
            f.write(script)
    
    def generate_fullprof_script(self, params: Dict[str, Any],
                                output_path: str):
        """生成FullProf脚本"""
        pcr_content = f"""COMM {params['phases'][0]['name'] if params['phases'] else 'Unknown'}
!
! Rietveld Refinement Input File
!
Cell {params['phases'][0]['lattice_params'].get('a', 5.0):.4f}
     {params['phases'][0]['lattice_params'].get('b', 5.0):.4f}
     {params['phases'][0]['lattice_params'].get('c', 5.0):.4f}
     {params['phases'][0]['lattice_params'].get('alpha', 90.0):.3f}
     {params['phases'][0]['lattice_params'].get('beta', 90.0):.3f}
     {params['phases'][0]['lattice_params'].get('gamma', 90.0):.3f}
!
 wavelength {params['instrument']['wavelength']:.5f}
!
"""
        
        with open(output_path, 'w') as f:
            f.write(pcr_content)


# ==================== 主入口函数 ====================

def analyze_xrd(filepath: str, 
               file_format: Optional[str] = None) -> Dict[str, Any]:
    """分析XRD文件"""
    analyzer = XRDAnalyzer()
    return analyzer.analyze(filepath, file_format)


def compare_xrd(exp_file: str, calc_file: str) -> Dict[str, float]:
    """比较实验和计算XRD"""
    analyzer = XRDAnalyzer()
    
    pattern1 = analyzer.parser.parse(exp_file)
    pattern2 = analyzer.parser.parse(calc_file)
    
    return analyzer.compare_patterns(pattern1, pattern2)


def identify_phases(filepath: str,
                   database_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """识别物相"""
    parser = XRDParser()
    detector = PeakDetector()
    identifier = PhaseIdentifier(database_path)
    
    pattern = parser.parse(filepath)
    pattern.peaks = detector.detect_peaks(pattern)
    phases = identifier.identify_phases(pattern)
    
    return [p.to_dict() for p in phases]


# 示例用法
if __name__ == "__main__":
    # 模拟数据
    two_theta = np.linspace(10, 80, 700)
    # 模拟Li3PS4的衍射峰
    intensity = np.zeros_like(two_theta)
    peaks_2theta = [17.5, 20.2, 25.1, 30.5, 35.8]
    for p in peaks_2theta:
        intensity += 100 * np.exp(-((two_theta - p) / 0.3)**2)
    intensity += np.random.normal(0, 2, len(two_theta))
    
    pattern = XRDPattern(two_theta=two_theta, intensity=intensity, sample_name="Li3PS4_sample")
    
    # 分析
    analyzer = XRDAnalyzer()
    pattern.peaks = analyzer.detector.detect_peaks(pattern)
    phases = analyzer.identifier.identify_phases(pattern)
    
    print(f"Detected {len(pattern.peaks)} peaks")
    print(f"\nIdentified phases:")
    for phase in phases:
        print(f"  - {phase.phase_name}: {phase.match_score:.2%} match")
