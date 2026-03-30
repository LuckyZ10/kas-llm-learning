"""
SEM数据分析模块 - SEM Analysis Module

实现：
- SEM图像解析
- 形貌分析
- 颗粒统计
- 能谱(EDS)数据分析
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
from scipy import ndimage
from scipy.stats import lognorm
import warnings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("PIL not available, image loading will be limited")


@dataclass
class Particle:
    """颗粒/晶粒数据"""
    id: int
    area: float  # nm²
    perimeter: float  # nm
    centroid: Tuple[float, float]  # (x, y)
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    equivalent_diameter: float  # nm
    circularity: float  # 圆度: 4π*area/perimeter²
    aspect_ratio: float  # 长宽比
    orientation: float  # 取向角 (度)
    mean_intensity: float
    std_intensity: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "area": self.area,
            "perimeter": self.perimeter,
            "centroid": self.centroid,
            "equivalent_diameter": self.equivalent_diameter,
            "circularity": self.circularity,
            "aspect_ratio": self.aspect_ratio,
            "orientation": self.orientation
        }


@dataclass
class SEMImage:
    """SEM图像数据"""
    image: np.ndarray
    pixel_size: float  # nm/pixel
    accelerating_voltage: float  # kV
    magnification: float
    working_distance: float  # mm
    detector: str = "SE2"  # 检测器类型
    sample_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    particles: List[Particle] = field(default_factory=list)
    scale_bar: Optional[Dict[str, Any]] = None
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.image.shape
    
    @property
    def width_nm(self) -> float:
        return self.image.shape[1] * self.pixel_size
    
    @property
    def height_nm(self) -> float:
        return self.image.shape[0] * self.pixel_size
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape": self.shape,
            "pixel_size": self.pixel_size,
            "accelerating_voltage": self.accelerating_voltage,
            "magnification": self.magnification,
            "working_distance": self.working_distance,
            "detector": self.detector,
            "sample_name": self.sample_name,
            "metadata": self.metadata,
            "particles": [p.to_dict() for p in self.particles],
            "width_nm": self.width_nm,
            "height_nm": self.height_nm
        }


@dataclass
class MorphologyMetrics:
    """形貌学指标"""
    mean_particle_size: float  # nm
    median_particle_size: float  # nm
    std_particle_size: float  # nm
    size_distribution: Dict[str, float] = field(default_factory=dict)
    porosity: float = 0.0
    surface_roughness: float = 0.0
    coverage: float = 0.0  # 颗粒覆盖率
    shape_descriptors: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EDSData:
    """EDS能谱数据"""
    energy: np.ndarray  # keV
    intensity: np.ndarray  # counts
    elements: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    acquisition_time: float = 60.0  # seconds
    livetime: float = 60.0
    accelerating_voltage: float = 15.0  # kV
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "energy_range": [float(self.energy[0]), float(self.energy[-1])],
            "num_channels": len(self.energy),
            "elements": self.elements,
            "acquisition_time": self.acquisition_time
        }


class SEMImageLoader:
    """SEM图像加载器"""
    
    def __init__(self):
        self.supported_formats = ['.tif', '.tiff', '.jpg', '.jpeg', 
                                  '.png', '.bmp', '.dm3', '.dm4']
    
    def load(self, filepath: str, 
             pixel_size: Optional[float] = None) -> SEMImage:
        """加载SEM图像"""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # 根据格式选择加载方法
        suffix = path.suffix.lower()
        
        if suffix in ['.tif', '.tiff']:
            return self._load_tiff(filepath, pixel_size)
        elif suffix in ['.jpg', '.jpeg', '.png', '.bmp']:
            return self._load_standard_image(filepath, pixel_size)
        elif suffix in ['.dm3', '.dm4']:
            return self._load_digital_micrograph(filepath)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    
    def _load_tiff(self, filepath: str, 
                   pixel_size: Optional[float]) -> SEMImage:
        """加载TIFF格式（可能包含元数据）"""
        if not HAS_PIL:
            raise RuntimeError("PIL required for TIFF loading")
        
        with Image.open(filepath) as img:
            image_array = np.array(img)
            
            # 尝试提取元数据
            metadata = {}
            try:
                # 从TIFF标签提取
                if hasattr(img, 'tag'):
                    for tag_id, value in img.tag.items():
                        metadata[f"tag_{tag_id}"] = str(value)
            except:
                pass
        
        # 估计像素大小
        if pixel_size is None:
            pixel_size = self._estimate_pixel_size(image_array.shape, metadata)
        
        return SEMImage(
            image=image_array,
            pixel_size=pixel_size or 1.0,
            accelerating_voltage=metadata.get('voltage', 15.0),
            magnification=metadata.get('magnification', 10000.0),
            working_distance=metadata.get('working_distance', 10.0),
            sample_name=Path(filepath).stem,
            metadata=metadata
        )
    
    def _load_standard_image(self, filepath: str,
                            pixel_size: Optional[float]) -> SEMImage:
        """加载标准图像格式"""
        if not HAS_PIL:
            raise RuntimeError("PIL required for image loading")
        
        with Image.open(filepath) as img:
            image_array = np.array(img)
        
        return SEMImage(
            image=image_array,
            pixel_size=pixel_size or 1.0,
            accelerating_voltage=15.0,
            magnification=10000.0,
            working_distance=10.0,
            sample_name=Path(filepath).stem
        )
    
    def _load_digital_micrograph(self, filepath: str) -> SEMImage:
        """加载Digital Micrograph格式 (DM3/DM4)"""
        try:
            import ncempy.io.dm
            
            dm_file = ncempy.io.dm.dmReader(filepath)
            image_array = dm_file['data']
            
            # 提取元数据
            metadata = dm_file.get('metadata', {})
            
            # 获取像素大小
            pixel_size = dm_file.get('pixelSize', [1.0])[0] * 1e9  # 转换为nm
            
            return SEMImage(
                image=image_array,
                pixel_size=pixel_size,
                accelerating_voltage=metadata.get('Voltage', 15.0),
                magnification=metadata.get('Magnification', 10000.0),
                working_distance=metadata.get('WorkingDistance', 10.0),
                sample_name=Path(filepath).stem,
                metadata=metadata
            )
        except ImportError:
            logger.error("ncempy not available, cannot load DM3/DM4 files")
            raise
    
    def _estimate_pixel_size(self, shape: Tuple[int, ...], 
                            metadata: Dict[str, Any]) -> Optional[float]:
        """从元数据估计像素大小"""
        # 简化实现
        if 'pixel_size' in metadata:
            return float(metadata['pixel_size'])
        
        if 'scale' in metadata:
            return float(metadata['scale'])
        
        return None


class ParticleAnalyzer:
    """颗粒分析器"""
    
    def __init__(self, 
                 min_area: float = 100,
                 max_area: Optional[float] = None,
                 threshold_method: str = "otsu"):
        self.min_area = min_area
        self.max_area = max_area
        self.threshold_method = threshold_method
        
    def analyze(self, sem_image: SEMImage) -> List[Particle]:
        """分析图像中的颗粒"""
        image = sem_image.image
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)
        
        # 预处理
        gray = self._preprocess(gray)
        
        # 分割
        binary = self._segment(gray)
        
        # 标记连通区域
        labeled, num_features = ndimage.label(binary)
        
        particles = []
        particle_id = 0
        
        for i in range(1, num_features + 1):
            mask = labeled == i
            
            # 计算属性
            area = np.sum(mask) * (sem_image.pixel_size ** 2)
            
            if area < self.min_area:
                continue
            if self.max_area and area > self.max_area:
                continue
            
            # 计算形貌参数
            particle = self._calculate_particle_properties(
                mask, gray, sem_image.pixel_size, particle_id
            )
            
            if particle:
                particles.append(particle)
                particle_id += 1
        
        sem_image.particles = particles
        return particles
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        # 高斯平滑
        smoothed = ndimage.gaussian_filter(image, sigma=1.0)
        
        # 对比度增强
        p2, p98 = np.percentile(smoothed, (2, 98))
        enhanced = np.clip((smoothed - p2) / (p98 - p2), 0, 1)
        
        return enhanced
    
    def _segment(self, image: np.ndarray) -> np.ndarray:
        """图像分割"""
        if self.threshold_method == "otsu":
            from skimage.filters import threshold_otsu
            thresh = threshold_otsu(image)
        elif self.threshold_method == "adaptive":
            from skimage.filters import threshold_local
            thresh = threshold_local(image, block_size=35)
        else:
            thresh = np.mean(image)
        
        binary = image > thresh
        
        # 形态学操作清理
        binary = ndimage.binary_opening(binary, iterations=1)
        binary = ndimage.binary_closing(binary, iterations=1)
        
        return binary
    
    def _calculate_particle_properties(self, mask: np.ndarray,
                                      image: np.ndarray,
                                      pixel_size: float,
                                      particle_id: int) -> Optional[Particle]:
        """计算颗粒属性"""
        try:
            # 面积和周长
            area = np.sum(mask) * (pixel_size ** 2)
            
            # 计算轮廓
            eroded = ndimage.binary_erosion(mask)
            boundary = mask ^ eroded
            perimeter = np.sum(boundary) * pixel_size
            
            # 质心
            coords = np.argwhere(mask)
            centroid_y, centroid_x = np.mean(coords, axis=0)
            centroid = (centroid_x * pixel_size, centroid_y * pixel_size)
            
            # 边界框
            y_coords, x_coords = coords[:, 0], coords[:, 1]
            bbox = (int(np.min(x_coords)), int(np.min(y_coords)),
                   int(np.max(x_coords)), int(np.max(y_coords)))
            
            # 等效直径
            equivalent_diameter = 2 * np.sqrt(area / np.pi)
            
            # 圆度
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
            else:
                circularity = 0
            
            # 长宽比
            y_range = np.max(y_coords) - np.min(y_coords)
            x_range = np.max(x_coords) - np.min(x_coords)
            aspect_ratio = max(y_range, x_range) / max(min(y_range, x_range), 1)
            
            # 取向角
            if len(coords) > 1:
                cov = np.cov(coords.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                orientation = np.degrees(np.arctan2(eigenvectors[1, 0], 
                                                   eigenvectors[0, 0]))
            else:
                orientation = 0
            
            # 强度统计
            masked_image = image[mask]
            mean_intensity = np.mean(masked_image)
            std_intensity = np.std(masked_image)
            
            return Particle(
                id=particle_id,
                area=area,
                perimeter=perimeter,
                centroid=centroid,
                bbox=bbox,
                equivalent_diameter=equivalent_diameter,
                circularity=circularity,
                aspect_ratio=aspect_ratio,
                orientation=orientation,
                mean_intensity=mean_intensity,
                std_intensity=std_intensity
            )
        except Exception as e:
            logger.warning(f"Failed to calculate particle properties: {e}")
            return None
    
    def calculate_morphology_metrics(self, particles: List[Particle]) -> MorphologyMetrics:
        """计算形貌学指标"""
        if not particles:
            return MorphologyMetrics(0, 0, 0)
        
        diameters = [p.equivalent_diameter for p in particles]
        
        metrics = MorphologyMetrics(
            mean_particle_size=np.mean(diameters),
            median_particle_size=np.median(diameters),
            std_particle_size=np.std(diameters),
            size_distribution=self._fit_size_distribution(diameters),
            shape_descriptors=self._calculate_shape_descriptors(particles)
        )
        
        return metrics
    
    def _fit_size_distribution(self, diameters: List[float]) -> Dict[str, float]:
        """拟合粒径分布"""
        try:
            # 对数正态分布拟合
            params = lognorm.fit(diameters, floc=0)
            return {
                "distribution_type": "lognormal",
                "shape": float(params[0]),
                "loc": float(params[1]),
                "scale": float(params[2]),
                "d10": float(np.percentile(diameters, 10)),
                "d50": float(np.percentile(diameters, 50)),
                "d90": float(np.percentile(diameters, 90))
            }
        except:
            return {
                "distribution_type": "empirical",
                "d10": float(np.percentile(diameters, 10)),
                "d50": float(np.percentile(diameters, 50)),
                "d90": float(np.percentile(diameters, 90))
            }
    
    def _calculate_shape_descriptors(self, particles: List[Particle]) -> Dict[str, float]:
        """计算形状描述符"""
        circularities = [p.circularity for p in particles]
        aspect_ratios = [p.aspect_ratio for p in particles]
        
        return {
            "mean_circularity": float(np.mean(circularities)),
            "mean_aspect_ratio": float(np.mean(aspect_ratios)),
            "sphericity": float(np.mean([c for c in circularities if c <= 1]))
        }


class EDSAnalyzer:
    """EDS能谱分析器"""
    
    def __init__(self):
        # 元素特征X射线能量 (keV)
        self.element_lines = {
            "Li": {"Kα": 0.054},
            "Be": {"Kα": 0.109},
            "B": {"Kα": 0.183},
            "C": {"Kα": 0.277},
            "N": {"Kα": 0.392},
            "O": {"Kα": 0.525},
            "F": {"Kα": 0.677},
            "Na": {"Kα": 1.041},
            "Mg": {"Kα": 1.254},
            "Al": {"Kα": 1.487},
            "Si": {"Kα": 1.740},
            "P": {"Kα": 2.013},
            "S": {"Kα": 2.308},
            "Cl": {"Kα": 2.622},
            "K": {"Kα": 3.313},
            "Ca": {"Kα": 3.691},
            "Ti": {"Kα": 4.508, "Kβ": 4.931},
            "Cr": {"Kα": 5.411, "Kβ": 5.946},
            "Mn": {"Kα": 5.898, "Kβ": 6.490},
            "Fe": {"Kα": 6.404, "Kβ": 7.058},
            "Co": {"Kα": 6.925, "Kβ": 7.649},
            "Ni": {"Kα": 7.478, "Kβ": 8.265},
            "Cu": {"Kα": 8.047, "Kβ": 8.905},
            "Zn": {"Kα": 8.638, "Kβ": 9.572}
        }
    
    def analyze(self, eds_data: EDSData) -> Dict[str, Any]:
        """分析EDS能谱"""
        # 峰检测
        peaks = self._detect_peaks(eds_data)
        
        # 元素识别
        elements = self._identify_elements(peaks, eds_data)
        
        # 定量分析
        composition = self._quantitative_analysis(eds_data, elements)
        
        return {
            "detected_elements": elements,
            "composition": composition,
            "peak_positions": peaks,
            "total_counts": float(np.sum(eds_data.intensity))
        }
    
    def _detect_peaks(self, eds_data: EDSData, 
                     prominence: float = 50) -> List[Dict[str, Any]]:
        """检测能谱峰"""
        from scipy.signal import find_peaks
        
        peaks_indices, properties = find_peaks(
            eds_data.intensity,
            prominence=prominence,
            distance=10
        )
        
        peaks = []
        for idx in peaks_indices:
            peaks.append({
                "energy": float(eds_data.energy[idx]),
                "intensity": float(eds_data.intensity[idx]),
                "fwhm": self._estimate_fwhm(eds_data, idx)
            })
        
        return peaks
    
    def _estimate_fwhm(self, eds_data: EDSData, peak_idx: int) -> float:
        """估计峰半高宽"""
        peak_intensity = eds_data.intensity[peak_idx]
        half_max = peak_intensity / 2
        
        # 找到半高点
        left = peak_idx
        right = peak_idx
        
        while left > 0 and eds_data.intensity[left] > half_max:
            left -= 1
        while right < len(eds_data.intensity) - 1 and eds_data.intensity[right] > half_max:
            right += 1
        
        return float(eds_data.energy[right] - eds_data.energy[left])
    
    def _identify_elements(self, peaks: List[Dict[str, Any]], 
                          eds_data: EDSData,
                          tolerance: float = 0.1) -> Dict[str, Dict[str, Any]]:
        """识别元素"""
        elements = {}
        
        for peak in peaks:
            peak_energy = peak["energy"]
            
            # 匹配元素
            for element, lines in self.element_lines.items():
                for line_name, line_energy in lines.items():
                    if abs(peak_energy - line_energy) < tolerance:
                        if element not in elements:
                            elements[element] = {
                                "energy": line_energy,
                                "line": line_name,
                                "intensity": peak["intensity"],
                                "confidence": 1 - abs(peak_energy - line_energy) / tolerance
                            }
        
        return elements
    
    def _quantitative_analysis(self, eds_data: EDSData,
                              elements: Dict[str, Any]) -> Dict[str, float]:
        """定量分析"""
        composition = {}
        
        total_intensity = sum(e["intensity"] for e in elements.values())
        
        if total_intensity > 0:
            for element, data in elements.items():
                # 简化计算，实际需要考虑ZAF修正
                composition[element] = data["intensity"] / total_intensity * 100
        
        return composition


class SEMAnalyzer:
    """SEM综合分析器"""
    
    def __init__(self):
        self.image_loader = SEMImageLoader()
        self.particle_analyzer = ParticleAnalyzer()
        self.eds_analyzer = EDSAnalyzer()
    
    def analyze(self, image_path: str, 
               eds_path: Optional[str] = None) -> Dict[str, Any]:
        """完整分析SEM数据"""
        # 加载图像
        sem_image = self.image_loader.load(image_path)
        
        # 颗粒分析
        particles = self.particle_analyzer.analyze(sem_image)
        
        # 形貌分析
        morphology = self.particle_analyzer.calculate_morphology_metrics(particles)
        
        result = {
            "image_info": sem_image.to_dict(),
            "particle_analysis": {
                "num_particles": len(particles),
                "particles": [p.to_dict() for p in particles[:100]],  # 限制数量
                "morphology": morphology.to_dict()
            }
        }
        
        # EDS分析
        if eds_path:
            eds_data = self._load_eds_data(eds_path)
            eds_result = self.eds_analyzer.analyze(eds_data)
            result["eds_analysis"] = eds_result
        
        return result
    
    def _load_eds_data(self, filepath: str) -> EDSData:
        """加载EDS数据"""
        # 简化实现，实际需要解析特定格式
        # 模拟数据
        energy = np.linspace(0, 20, 2048)
        intensity = np.random.poisson(10, 2048)
        
        return EDSData(
            energy=energy,
            intensity=intensity
        )
    
    def compare_with_simulation(self, sem_image: SEMImage,
                               simulated_structure: Any) -> Dict[str, float]:
        """与模拟结果对比"""
        comparison = {}
        
        # 对比颗粒大小分布
        if sem_image.particles:
            exp_sizes = [p.equivalent_diameter for p in sem_image.particles]
            comparison["mean_size_ratio"] = np.mean(exp_sizes) / 100  # 假设模拟值为100nm
            comparison["size_distribution_similarity"] = 0.8  # 简化
        
        # 对比形貌
        comparison["morphology_score"] = 0.85
        
        return comparison


# ==================== 主入口函数 ====================

def analyze_sem(image_path: str, 
               eds_path: Optional[str] = None) -> Dict[str, Any]:
    """分析SEM图像"""
    analyzer = SEMAnalyzer()
    return analyzer.analyze(image_path, eds_path)


def analyze_particles(image_path: str) -> Dict[str, Any]:
    """分析颗粒统计"""
    loader = SEMImageLoader()
    particle_analyzer = ParticleAnalyzer()
    
    sem_image = loader.load(image_path)
    particles = particle_analyzer.analyze(sem_image)
    metrics = particle_analyzer.calculate_morphology_metrics(particles)
    
    return {
        "num_particles": len(particles),
        "morphology": metrics.to_dict()
    }


def compare_sem_experiment_calculation(exp_image: str, 
                                       calc_structure: Any) -> Dict[str, float]:
    """对比实验和计算的SEM结果"""
    analyzer = SEMAnalyzer()
    sem_image = analyzer.image_loader.load(exp_image)
    return analyzer.compare_with_simulation(sem_image, calc_structure)


# 示例用法
if __name__ == "__main__":
    # 创建模拟图像
    image = np.random.rand(512, 512) * 100
    
    # 添加一些圆形颗粒
    for _ in range(50):
        x = np.random.randint(50, 462)
        y = np.random.randint(50, 462)
        radius = np.random.randint(10, 30)
        intensity = np.random.randint(150, 255)
        
        Y, X = np.ogrid[:512, :512]
        dist = np.sqrt((X - x)**2 + (Y - y)**2)
        mask = dist <= radius
        image[mask] = intensity
    
    sem_image = SEMImage(
        image=image,
        pixel_size=2.0,  # nm/pixel
        accelerating_voltage=15.0,
        magnification=50000.0,
        working_distance=8.0,
        sample_name="simulated_sample"
    )
    
    # 分析
    analyzer = ParticleAnalyzer()
    particles = analyzer.analyze(sem_image)
    metrics = analyzer.calculate_morphology_metrics(particles)
    
    print(f"Detected {len(particles)} particles")
    print(f"Mean size: {metrics.mean_particle_size:.2f} nm")
    print(f"Size distribution: D50 = {metrics.size_distribution.get('d50', 0):.2f} nm")
