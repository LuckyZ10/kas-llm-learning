"""
TEM数据分析模块 - TEM Analysis Module

实现：
- TEM图像解析
- 高分辨像(HRTEM)分析
- 选区电子衍射(SAED)分析
- 晶格条纹分析
- 计算-实验对比
"""

import os
import json
import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from scipy import ndimage, fftpack, signal
from scipy.optimize import curve_fit
import warnings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("OpenCV not available, some features may be limited")


@dataclass
class LatticeFringe:
    """晶格条纹数据"""
    spacing: float  # nm
    orientation: float  # degrees
    intensity: float
    coherence_length: float  # nm
    hkl: Optional[Tuple[int, int, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spacing": self.spacing,
            "orientation": self.orientation,
            "intensity": self.intensity,
            "coherence_length": self.coherence_length,
            "hkl": self.hkl
        }


@dataclass
class DiffractionSpot:
    """衍射斑点数据"""
    x: float  # 像素坐标
    y: float
    intensity: float
    radius: float  # 倒易空间距离
    hkl: Optional[Tuple[int, int, int]] = None
    d_spacing: Optional[float] = None  # nm
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "intensity": self.intensity,
            "radius": self.radius,
            "hkl": self.hkl,
            "d_spacing": self.d_spacing
        }


@dataclass
class SAEDPattern:
    """选区电子衍射图谱"""
    image: np.ndarray
    spots: List[DiffractionSpot] = field(default_factory=list)
    center: Tuple[float, float] = (0.0, 0.0)
    calibration: float = 1.0  # nm^-1/pixel
    wavelength: float = 0.00251  # 200kV电子波长，nm
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": self.center,
            "calibration": self.calibration,
            "num_spots": len(self.spots),
            "spots": [s.to_dict() for s in self.spots]
        }


@dataclass
class HRTEMImage:
    """高分辨TEM图像"""
    image: np.ndarray
    pixel_size: float  # nm/pixel
    accelerating_voltage: float  # kV
    defocus: float = 0.0  # nm
    spherical_aberration: float = 1.0  # mm
    chromatic_aberration: float = 1.0  # mm
    lattice_fringes: List[LatticeFringe] = field(default_factory=list)
    fourier_transform: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape": self.image.shape,
            "pixel_size": self.pixel_size,
            "accelerating_voltage": self.accelerating_voltage,
            "defocus": self.defocus,
            "num_fringes": len(self.lattice_fringes),
            "fringes": [f.to_dict() for f in self.lattice_fringes]
        }


@dataclass
class CrystalInfo:
    """晶体信息"""
    phase_name: str
    space_group: str
    lattice_type: str
    lattice_params: Dict[str, float] = field(default_factory=dict)
    zone_axis: Optional[Tuple[int, int, int]] = None
    orientation_matrix: Optional[np.ndarray] = None
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "space_group": self.space_group,
            "lattice_type": self.lattice_type,
            "lattice_params": self.lattice_params,
            "zone_axis": self.zone_axis,
            "confidence": self.confidence
        }


class TEMImageLoader:
    """TEM图像加载器"""
    
    def __init__(self):
        self.supported_formats = ['.tif', '.tiff', '.dm3', '.dm4', 
                                  '.ser', '.emi', '.mrc', '.jpg', '.png']
    
    def load(self, filepath: str, 
             image_type: str = "auto") -> Union[HRTEMImage, SAEDPattern]:
        """加载TEM图像"""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        suffix = path.suffix.lower()
        
        # 自动检测图像类型
        if image_type == "auto":
            image_type = self._detect_image_type(filepath)
        
        if image_type == "hrtem":
            return self._load_hrtem(filepath, suffix)
        elif image_type == "saed":
            return self._load_saed(filepath, suffix)
        else:
            raise ValueError(f"Unknown image type: {image_type}")
    
    def _detect_image_type(self, filepath: str) -> str:
        """检测图像类型"""
        # 根据文件名或元数据推断
        path = Path(filepath)
        name = path.stem.lower()
        
        if 'diffraction' in name or 'saed' in name or 'sed' in name:
            return "saed"
        elif 'hrtem' in name or 'hr' in name or 'lattice' in name:
            return "hrtem"
        else:
            # 默认假设为HRTEM
            return "hrtem"
    
    def _load_hrtem(self, filepath: str, suffix: str) -> HRTEMImage:
        """加载HRTEM图像"""
        if suffix in ['.dm3', '.dm4']:
            return self._load_digital_micrograph_hrtem(filepath)
        elif suffix in ['.tif', '.tiff', '.jpg', '.png']:
            return self._load_standard_hrtem(filepath)
        elif suffix == '.mrc':
            return self._load_mrc_hrtem(filepath)
        else:
            raise ValueError(f"Unsupported HRTEM format: {suffix}")
    
    def _load_saed(self, filepath: str, suffix: str) -> SAEDPattern:
        """加载SAED图像"""
        if suffix in ['.dm3', '.dm4']:
            return self._load_digital_micrograph_saed(filepath)
        elif suffix in ['.tif', '.tiff', '.jpg', '.png']:
            return self._load_standard_saed(filepath)
        else:
            raise ValueError(f"Unsupported SAED format: {suffix}")
    
    def _load_digital_micrograph_hrtem(self, filepath: str) -> HRTEMImage:
        """从Digital Micrograph加载HRTEM"""
        try:
            import ncempy.io.dm
            
            dm_file = ncempy.io.dm.dmReader(filepath)
            image = dm_file['data']
            
            # 获取像素大小
            pixel_size = dm_file.get('pixelSize', [0.01])[0] * 1e9  # nm
            
            # 提取元数据
            metadata = dm_file.get('metadata', {})
            voltage = metadata.get('Voltage', 200.0)
            
            return HRTEMImage(
                image=image,
                pixel_size=pixel_size,
                accelerating_voltage=voltage
            )
        except ImportError:
            logger.error("ncempy not available")
            raise
    
    def _load_digital_micrograph_saed(self, filepath: str) -> SAEDPattern:
        """从Digital Micrograph加载SAED"""
        try:
            import ncempy.io.dm
            
            dm_file = ncempy.io.dm.dmReader(filepath)
            image = dm_file['data']
            
            # 获取校准
            pixel_size = dm_file.get('pixelSize', [1.0])[0]
            
            return SAEDPattern(
                image=image,
                calibration=1.0 / (pixel_size * 1e9)  # nm^-1/pixel
            )
        except ImportError:
            logger.error("ncempy not available")
            raise
    
    def _load_standard_hrtem(self, filepath: str) -> HRTEMImage:
        """加载标准图像格式HRTEM"""
        try:
            from PIL import Image
            
            with Image.open(filepath) as img:
                image = np.array(img)
                if len(image.shape) == 3:
                    image = np.mean(image, axis=2)
            
            return HRTEMImage(
                image=image,
                pixel_size=0.1,  # 默认值
                accelerating_voltage=200.0
            )
        except ImportError:
            raise RuntimeError("PIL required for image loading")
    
    def _load_standard_saed(self, filepath: str) -> SAEDPattern:
        """加载标准图像格式SAED"""
        try:
            from PIL import Image
            
            with Image.open(filepath) as img:
                image = np.array(img)
                if len(image.shape) == 3:
                    image = np.mean(image, axis=2)
            
            return SAEDPattern(
                image=image,
                calibration=0.1  # 默认值
            )
        except ImportError:
            raise RuntimeError("PIL required for image loading")
    
    def _load_mrc_hrtem(self, filepath: str) -> HRTEMImage:
        """加载MRC格式HRTEM"""
        try:
            import mrcfile
            
            with mrcfile.open(filepath) as mrc:
                image = mrc.data
                
                # 获取像素大小
                pixel_size = mrc.voxel_size.x * 1e9  # nm
            
            return HRTEMImage(
                image=image,
                pixel_size=pixel_size,
                accelerating_voltage=200.0
            )
        except ImportError:
            logger.error("mrcfile not available")
            raise


class LatticeAnalyzer:
    """晶格分析器"""
    
    def __init__(self):
        self.min_spacing = 0.05  # nm
        self.max_spacing = 2.0  # nm
        
    def analyze(self, hrtem: HRTEMImage) -> List[LatticeFringe]:
        """分析晶格条纹"""
        # 计算FFT
        fft = self._calculate_fft(hrtem.image)
        hrtem.fourier_transform = fft
        
        # 检测FFT峰
        peaks = self._detect_fft_peaks(fft)
        
        # 转换为晶格条纹
        fringes = []
        for peak in peaks:
            # 计算晶格间距
            spacing = 1.0 / (peak['radius'] / hrtem.pixel_size)  # nm
            
            if self.min_spacing < spacing < self.max_spacing:
                fringe = LatticeFringe(
                    spacing=spacing,
                    orientation=peak['angle'],
                    intensity=peak['intensity'],
                    coherence_length=self._estimate_coherence_length(
                        hrtem.image, peak
                    )
                )
                fringes.append(fringe)
        
        hrtem.lattice_fringes = fringes
        return fringes
    
    def _calculate_fft(self, image: np.ndarray) -> np.ndarray:
        """计算FFT"""
        # 窗函数
        window = np.hanning(image.shape[0])[:, None] * np.hanning(image.shape[1])[None, :]
        
        # FFT
        fft = fftpack.fft2(image * window)
        fft_shifted = fftpack.fftshift(fft)
        
        # 功率谱
        power = np.abs(fft_shifted) ** 2
        
        return power
    
    def _detect_fft_peaks(self, fft: np.ndarray, 
                         num_peaks: int = 10) -> List[Dict[str, Any]]:
        """检测FFT峰值"""
        # 中心化
        cy, cx = fft.shape[0] // 2, fft.shape[1] // 2
        
        # 排除中心低频
        mask_radius = min(fft.shape) // 20
        Y, X = np.ogrid[:fft.shape[0], :fft.shape[1]]
        center_mask = ((X - cx)**2 + (Y - cy)**2) > mask_radius**2
        
        fft_masked = fft.copy()
        fft_masked[~center_mask] = 0
        
        # 寻找局部最大值
        peaks = []
        
        for _ in range(num_peaks * 2):  # 多找一些，然后过滤
            max_idx = np.unravel_index(np.argmax(fft_masked), fft_masked.shape)
            
            if fft_masked[max_idx] == 0:
                break
            
            y, x = max_idx
            
            # 计算极坐标
            dy = y - cy
            dx = x - cx
            radius = np.sqrt(dx**2 + dy**2)
            angle = np.degrees(np.arctan2(dy, dx))
            
            peaks.append({
                'y': y,
                'x': x,
                'radius': radius,
                'angle': angle,
                'intensity': float(fft[max_idx])
            })
            
            # 抑制邻域
            suppression_radius = 10
            y_range = slice(max(0, y - suppression_radius), 
                          min(fft.shape[0], y + suppression_radius))
            x_range = slice(max(0, x - suppression_radius),
                          min(fft.shape[1], x + suppression_radius))
            fft_masked[y_range, x_range] = 0
        
        # 按强度排序并选择前num_peaks个
        peaks.sort(key=lambda p: p['intensity'], reverse=True)
        return peaks[:num_peaks]
    
    def _estimate_coherence_length(self, image: np.ndarray,
                                   peak: Dict[str, Any]) -> float:
        """估计相干长度"""
        # 基于FFT峰宽的估计
        # 简化实现
        return 5.0  # nm，默认值
    
    def index_fringes(self, hrtem: HRTEMImage, 
                     crystal_structure: Any) -> None:
        """为晶格条纹标定米勒指数"""
        # 获取理论d间距
        theoretical_spacings = self._get_theoretical_spacings(crystal_structure)
        
        for fringe in hrtem.lattice_fringes:
            # 找到最接近的理论间距
            best_match = None
            best_error = float('inf')
            
            for hkl, d_theory in theoretical_spacings.items():
                error = abs(fringe.spacing - d_theory) / d_theory
                if error < 0.1 and error < best_error:  # 10%容差
                    best_error = error
                    best_match = hkl
            
            if best_match:
                fringe.hkl = best_match


class DiffractionAnalyzer:
    """衍射分析器"""
    
    def __init__(self):
        self.min_spot_intensity = 0.05
        
    def analyze(self, saed: SAEDPattern) -> List[DiffractionSpot]:
        """分析SAED图谱"""
        image = saed.image.astype(float)
        
        # 预处理
        image = self._preprocess(image)
        
        # 检测中心
        center = self._find_center(image)
        saed.center = center
        
        # 检测斑点
        spots = self._detect_spots(image, center)
        
        # 计算d间距
        for spot in spots:
            spot.d_spacing = 1.0 / (spot.radius * saed.calibration)
        
        saed.spots = spots
        return spots
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理"""
        # 背景扣除
        background = ndimage.gaussian_filter(image, sigma=20)
        image = image - background
        image[image < 0] = 0
        
        # 归一化
        if np.max(image) > 0:
            image = image / np.max(image)
        
        return image
    
    def _find_center(self, image: np.ndarray) -> Tuple[float, float]:
        """找到衍射图中心"""
        # 方法1: 质心法
        y_coords, x_coords = np.indices(image.shape)
        
        threshold = np.max(image) * 0.1
        mask = image > threshold
        
        if np.sum(mask) > 0:
            cy = np.sum(y_coords * image) / np.sum(image)
            cx = np.sum(x_coords * image) / np.sum(image)
        else:
            cy, cx = image.shape[0] / 2, image.shape[1] / 2
        
        return (cx, cy)
    
    def _detect_spots(self, image: np.ndarray, 
                     center: Tuple[float, float]) -> List[DiffractionSpot]:
        """检测衍射斑点"""
        spots = []
        
        if HAS_CV2:
            # 使用OpenCV检测
            spots = self._detect_spots_cv2(image, center)
        else:
            # 使用scipy检测
            spots = self._detect_spots_scipy(image, center)
        
        return spots
    
    def _detect_spots_cv2(self, image: np.ndarray,
                         center: Tuple[float, float]) -> List[DiffractionSpot]:
        """使用OpenCV检测斑点"""
        # 转换为8位图像
        img_8bit = (image * 255).astype(np.uint8)
        
        # 二值化
        _, binary = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        spots = []
        for i, contour in enumerate(contours):
            # 计算矩
            M = cv2.moments(contour)
            
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                
                # 计算半径（倒易空间距离）
                dx = cx - center[0]
                dy = cy - center[1]
                radius = np.sqrt(dx**2 + dy**2)
                
                # 计算强度
                intensity = float(image[int(cy), int(cx)])
                
                if intensity > self.min_spot_intensity:
                    spot = DiffractionSpot(
                        x=cx,
                        y=cy,
                        intensity=intensity,
                        radius=radius
                    )
                    spots.append(spot)
        
        return spots
    
    def _detect_spots_scipy(self, image: np.ndarray,
                           center: Tuple[float, float]) -> List[DiffractionSpot]:
        """使用scipy检测斑点"""
        from scipy.signal import peak_local_max
        
        # 寻找局部最大值
        coordinates = peak_local_max(
            image,
            min_distance=10,
            threshold_abs=self.min_spot_intensity
        )
        
        spots = []
        for y, x in coordinates:
            dx = x - center[0]
            dy = y - center[1]
            radius = np.sqrt(dx**2 + dy**2)
            
            # 排除中心透射斑
            if radius > 10:
                spot = DiffractionSpot(
                    x=float(x),
                    y=float(y),
                    intensity=float(image[y, x]),
                    radius=radius
                )
                spots.append(spot)
        
        return spots
    
    def index_pattern(self, saed: SAEDPattern,
                     crystal_structure: Any) -> CrystalInfo:
        """为衍射图谱标定指数"""
        # 获取理论衍射斑点
        theoretical_spots = self._calculate_theoretical_spots(crystal_structure)
        
        # 匹配
        matched_spots = self._match_spots(saed.spots, theoretical_spots)
        
        # 确定晶带轴
        zone_axis = self._determine_zone_axis(matched_spots)
        
        # 计算置信度
        confidence = len(matched_spots) / max(len(saed.spots), len(theoretical_spots))
        
        crystal_info = CrystalInfo(
            phase_name=getattr(crystal_structure, 'formula', 'Unknown'),
            space_group=getattr(crystal_structure, 'space_group', 'Unknown'),
            lattice_type=getattr(crystal_structure, 'lattice_type', 'Unknown'),
            zone_axis=zone_axis,
            confidence=confidence
        )
        
        return crystal_info


class TEMAnalyzer:
    """TEM综合分析器"""
    
    def __init__(self):
        self.image_loader = TEMImageLoader()
        self.lattice_analyzer = LatticeAnalyzer()
        self.diffraction_analyzer = DiffractionAnalyzer()
    
    def analyze_hrtem(self, filepath: str) -> Dict[str, Any]:
        """分析HRTEM图像"""
        hrtem = self.image_loader.load(filepath, "hrtem")
        
        # 晶格分析
        fringes = self.lattice_analyzer.analyze(hrtem)
        
        return {
            "image_info": hrtem.to_dict(),
            "num_fringes": len(fringes),
            "fringes": [f.to_dict() for f in fringes],
            "mean_spacing": np.mean([f.spacing for f in fringes]) if fringes else 0
        }
    
    def analyze_saed(self, filepath: str) -> Dict[str, Any]:
        """分析SAED图谱"""
        saed = self.image_loader.load(filepath, "saed")
        
        # 衍射分析
        spots = self.diffraction_analyzer.analyze(saed)
        
        return {
            "pattern_info": saed.to_dict(),
            "num_spots": len(spots),
            "spots": [s.to_dict() for s in spots]
        }
    
    def compare_with_simulation(self, hrtem: HRTEMImage,
                               simulated_image: np.ndarray) -> Dict[str, float]:
        """与模拟HRTEM图像对比"""
        comparison = {}
        
        # FFT对比
        exp_fft = self.lattice_analyzer._calculate_fft(hrtem.image)
        sim_fft = self.lattice_analyzer._calculate_fft(simulated_image)
        
        # 归一化
        exp_fft = exp_fft / np.max(exp_fft)
        sim_fft = sim_fft / np.max(sim_fft)
        
        # 计算相关性
        correlation = np.corrcoef(exp_fft.flatten(), sim_fft.flatten())[0, 1]
        comparison["fft_correlation"] = float(correlation)
        
        # 实空间对比
        comparison["real_space_correlation"] = float(
            np.corrcoef(hrtem.image.flatten(), simulated_image.flatten())[0, 1]
        )
        
        return comparison


# ==================== 主入口函数 ====================

def analyze_tem_hrtem(filepath: str) -> Dict[str, Any]:
    """分析HRTEM图像"""
    analyzer = TEMAnalyzer()
    return analyzer.analyze_hrtem(filepath)


def analyze_tem_saed(filepath: str) -> Dict[str, Any]:
    """分析SAED图谱"""
    analyzer = TEMAnalyzer()
    return analyzer.analyze_saed(filepath)


def compare_hrtem_simulation(exp_image: str, 
                            sim_image: np.ndarray) -> Dict[str, float]:
    """对比实验和模拟HRTEM"""
    analyzer = TEMAnalyzer()
    hrtem = analyzer.image_loader.load(exp_image, "hrtem")
    return analyzer.compare_with_simulation(hrtem, sim_image)


# 示例用法
if __name__ == "__main__":
    # 创建模拟HRTEM图像
    size = 512
    image = np.random.rand(size, size) * 50
    
    # 添加晶格条纹
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    
    # 正交条纹
    spacing = 10  # 像素
    fringe1 = 100 * np.sin(2 * np.pi * X / spacing)
    fringe2 = 100 * np.sin(2 * np.pi * Y / spacing)
    
    image += fringe1 + fringe2
    image += np.random.normal(0, 10, (size, size))
    
    hrtem = HRTEMImage(
        image=image,
        pixel_size=0.1,  # nm
        accelerating_voltage=200.0
    )
    
    # 分析
    lattice_analyzer = LatticeAnalyzer()
    fringes = lattice_analyzer.analyze(hrtem)
    
    print(f"Detected {len(fringes)} lattice fringes")
    for fringe in fringes[:5]:
        print(f"  Spacing: {fringe.spacing:.3f} nm, "
              f"Orientation: {fringe.orientation:.1f}°")
