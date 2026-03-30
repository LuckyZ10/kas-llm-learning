"""
Matminer特征库集成 (Matminer Integration)
集成Matminer工具包进行材料特征提取

作者: DFT-LAMMPS Team
版本: 1.0.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from collections import defaultdict
import warnings


@dataclass
class MatminerConfig:
    """Matminer配置"""
    # 特征类别
    use_composition_features: bool = True
    use_structure_features: bool = True
    use_bandstructure_features: bool = False
    use_dos_features: bool = False
    
    # 特征选择
    feature_selection: str = "all"  # all, reduced, custom
    max_features: int = 200
    
    # 数据清洗
    drop_nan: bool = True
    fill_nan: bool = False
    fill_value: float = 0.0


class MatminerFeatureExtractor:
    """Matminer特征提取器"""
    
    def __init__(self, config: MatminerConfig = None):
        self.config = config or MatminerConfig()
        self.featurizers = {}
        self.feature_names = []
        self._init_featurizers()
    
    def _init_featurizers(self):
        """初始化特征提取器"""
        try:
            from matminer.featurizers.composition import (
                ElementProperty, Stoichiometry, ValenceOrbital,
                IonProperty, OxidationStates
            )
            from matminer.featurizers.structure import (
                DensityFeatures, StructuralHeterogeneity,
                ChemicalOrdering, StructureComposition
            )
            
            self.matminer_available = True
            
            # 成分特征
            if self.config.use_composition_features:
                self.featurizers["composition"] = [
                    ElementProperty.from_preset("magpie"),
                    Stoichiometry(),
                    ValenceOrbital(props=["avg", "std", "max", "min"]),
                    IonProperty(),
                    OxidationStates()
                ]
            
            # 结构特征
            if self.config.use_structure_features:
                self.featurizers["structure"] = [
                    DensityFeatures(),
                    StructuralHeterogeneity(),
                    ChemicalOrdering(),
                    StructureComposition()
                ]
            
        except ImportError:
            warnings.warn("Matminer not available. Using fallback implementations.")
            self.matminer_available = False
            self._init_fallback_featurizers()
    
    def _init_fallback_featurizers(self):
        """初始化备用特征提取器"""
        # 当Matminer不可用时使用简化实现
        pass
    
    def extract_composition_features(
        self,
        composition: Union[str, Dict[str, float]]
    ) -> np.ndarray:
        """
        提取成分特征
        
        Args:
            composition: 成分，可以是化学式字符串或元素比例字典
        
        Returns:
            特征向量
        """
        if self.matminer_available:
            return self._extract_with_matminer_composition(composition)
        else:
            return self._extract_composition_fallback(composition)
    
    def _extract_with_matminer_composition(
        self,
        composition: Union[str, Dict[str, float]]
    ) -> np.ndarray:
        """使用Matminer提取成分特征"""
        try:
            from pymatgen.core import Composition as PMGComposition
            
            # 转换为pymatgen Composition
            if isinstance(composition, str):
                comp = PMGComposition(composition)
            else:
                comp = PMGComposition(composition)
            
            features = []
            for featurizer in self.featurizers.get("composition", []):
                try:
                    feat = featurizer.featurize(comp)
                    features.extend(feat)
                except Exception as e:
                    warnings.warn(f"Featurizer error: {e}")
                    features.extend([0.0] * len(featurizer.feature_labels()))
            
            return np.array(features)
            
        except Exception as e:
            warnings.warn(f"Matminer extraction failed: {e}")
            return self._extract_composition_fallback(composition)
    
    def _extract_composition_fallback(
        self,
        composition: Union[str, Dict[str, float]]
    ) -> np.ndarray:
        """备用的成分特征提取"""
        # 简化实现
        if isinstance(composition, str):
            # 解析化学式
            comp_dict = self._parse_formula(composition)
        else:
            comp_dict = composition
        
        features = []
        
        # 基础统计特征
        values = np.array(list(comp_dict.values()))
        values = values / values.sum()  # 归一化
        
        features.extend([
            len(comp_dict),  # 元素种类数
            np.mean(values),  # 平均比例
            np.std(values),   # 比例标准差
            np.max(values),   # 最大比例
            np.min(values[values > 0]),  # 最小非零比例
            -np.sum(values * np.log(values + 1e-10)),  # 混合熵
        ])
        
        # 元素属性 (简化)
        element_properties = self._get_element_properties()
        
        weighted_props = defaultdict(list)
        for elem, frac in comp_dict.items():
            if elem in element_properties:
                for prop, value in element_properties[elem].items():
                    weighted_props[prop].append(value * frac)
        
        for prop_values in weighted_props.values():
            features.extend([
                np.sum(prop_values),
                np.mean(prop_values),
                np.std(prop_values) if len(prop_values) > 1 else 0
            ])
        
        return np.array(features)
    
    def extract_structure_features(
        self,
        structure: Dict[str, Any]
    ) -> np.ndarray:
        """
        提取结构特征
        
        Args:
            structure: 结构信息字典
        
        Returns:
            特征向量
        """
        if self.matminer_available:
            return self._extract_with_matminer_structure(structure)
        else:
            return self._extract_structure_fallback(structure)
    
    def _extract_with_matminer_structure(
        self,
        structure: Dict[str, Any]
    ) -> np.ndarray:
        """使用Matminer提取结构特征"""
        try:
            from pymatgen.core import Structure, Lattice
            
            # 创建pymatgen Structure
            if "lattice" in structure and "species" in structure:
                lattice = Lattice(structure["lattice"])
                struct = Structure(
                    lattice,
                    structure["species"],
                    structure["coords"]
                )
            else:
                return self._extract_structure_fallback(structure)
            
            features = []
            for featurizer in self.featurizers.get("structure", []):
                try:
                    feat = featurizer.featurize(struct)
                    features.extend(feat)
                except Exception as e:
                    warnings.warn(f"Structure featurizer error: {e}")
                    n_features = len(featurizer.feature_labels())
                    features.extend([0.0] * n_features)
            
            return np.array(features)
            
        except Exception as e:
            warnings.warn(f"Matminer structure extraction failed: {e}")
            return self._extract_structure_fallback(structure)
    
    def _extract_structure_fallback(
        self,
        structure: Dict[str, Any]
    ) -> np.ndarray:
        """备用的结构特征提取"""
        features = []
        
        # 晶格参数
        if "lattice" in structure:
            lattice = structure["lattice"]
            if isinstance(lattice, (list, np.ndarray)):
                if len(lattice) == 6:  # a, b, c, alpha, beta, gamma
                    features.extend(lattice)
                elif len(lattice) == 3:  # a, b, c (立方)
                    features.extend(lattice)
                    features.extend([90.0, 90.0, 90.0])
        
        # 体积和密度
        if "volume" in structure:
            features.append(structure["volume"])
        if "density" in structure:
            features.append(structure["density"])
        
        # 原子数
        if "num_sites" in structure:
            features.append(structure["num_sites"])
        
        # 空间群
        if "space_group" in structure:
            features.append(structure["space_group"])
        
        # 补充默认值
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20])
    
    def _parse_formula(self, formula: str) -> Dict[str, float]:
        """解析化学式"""
        import re
        
        pattern = r'([A-Z][a-z]*)(\d*\.?\d*)'
        matches = re.findall(pattern, formula)
        
        composition = {}
        for element, count in matches:
            composition[element] = float(count) if count else 1.0
        
        return composition
    
    def _get_element_properties(self) -> Dict[str, Dict[str, float]]:
        """获取元素属性"""
        # 简化的元素属性表
        return {
            "H": {"atomic_number": 1, "atomic_mass": 1.008, "electronegativity": 2.20},
            "Li": {"atomic_number": 3, "atomic_mass": 6.94, "electronegativity": 0.98},
            "Be": {"atomic_number": 4, "atomic_mass": 9.01, "electronegativity": 1.57},
            "B": {"atomic_number": 5, "atomic_mass": 10.81, "electronegativity": 2.04},
            "C": {"atomic_number": 6, "atomic_mass": 12.01, "electronegativity": 2.55},
            "N": {"atomic_number": 7, "atomic_mass": 14.01, "electronegativity": 3.04},
            "O": {"atomic_number": 8, "atomic_mass": 16.00, "electronegativity": 3.44},
            "F": {"atomic_number": 9, "atomic_mass": 19.00, "electronegativity": 3.98},
            "Na": {"atomic_number": 11, "atomic_mass": 22.99, "electronegativity": 0.93},
            "Mg": {"atomic_number": 12, "atomic_mass": 24.31, "electronegativity": 1.31},
            "Al": {"atomic_number": 13, "atomic_mass": 26.98, "electronegativity": 1.61},
            "Si": {"atomic_number": 14, "atomic_mass": 28.09, "electronegativity": 1.90},
            "P": {"atomic_number": 15, "atomic_mass": 30.97, "electronegativity": 2.19},
            "S": {"atomic_number": 16, "atomic_mass": 32.06, "electronegativity": 2.58},
            "Cl": {"atomic_number": 17, "atomic_mass": 35.45, "electronegativity": 3.16},
            "K": {"atomic_number": 19, "atomic_mass": 39.10, "electronegativity": 0.82},
            "Ca": {"atomic_number": 20, "atomic_mass": 40.08, "electronegativity": 1.00},
            "Sc": {"atomic_number": 21, "atomic_mass": 44.96, "electronegativity": 1.36},
            "Ti": {"atomic_number": 22, "atomic_mass": 47.87, "electronegativity": 1.54},
            "V": {"atomic_number": 23, "atomic_mass": 50.94, "electronegativity": 1.63},
            "Cr": {"atomic_number": 24, "atomic_mass": 52.00, "electronegativity": 1.66},
            "Mn": {"atomic_number": 25, "atomic_mass": 54.94, "electronegativity": 1.55},
            "Fe": {"atomic_number": 26, "atomic_mass": 55.85, "electronegativity": 1.83},
            "Co": {"atomic_number": 27, "atomic_mass": 58.93, "electronegativity": 1.88},
            "Ni": {"atomic_number": 28, "atomic_mass": 58.69, "electronegativity": 1.91},
            "Cu": {"atomic_number": 29, "atomic_mass": 63.55, "electronegativity": 1.90},
            "Zn": {"atomic_number": 30, "atomic_mass": 65.38, "electronegativity": 1.65},
            "Ga": {"atomic_number": 31, "atomic_mass": 69.72, "electronegativity": 1.81},
            "Ge": {"atomic_number": 32, "atomic_mass": 72.63, "electronegativity": 2.01},
            "As": {"atomic_number": 33, "atomic_mass": 74.92, "electronegativity": 2.18},
            "Se": {"atomic_number": 34, "atomic_mass": 78.96, "electronegativity": 2.55},
            "Br": {"atomic_number": 35, "atomic_mass": 79.90, "electronegativity": 2.96},
            "Rb": {"atomic_number": 37, "atomic_mass": 85.47, "electronegativity": 0.82},
            "Sr": {"atomic_number": 38, "atomic_mass": 87.62, "electronegativity": 0.95},
            "Y": {"atomic_number": 39, "atomic_mass": 88.91, "electronegativity": 1.22},
            "Zr": {"atomic_number": 40, "atomic_mass": 91.22, "electronegativity": 1.33},
            "Nb": {"atomic_number": 41, "atomic_mass": 92.91, "electronegativity": 1.6},
            "Mo": {"atomic_number": 42, "atomic_mass": 95.95, "electronegativity": 2.16},
            "Tc": {"atomic_number": 43, "atomic_mass": 98.0, "electronegativity": 1.9},
            "Ru": {"atomic_number": 44, "atomic_mass": 101.07, "electronegativity": 2.2},
            "Rh": {"atomic_number": 45, "atomic_mass": 102.91, "electronegativity": 2.28},
            "Pd": {"atomic_number": 46, "atomic_mass": 106.42, "electronegativity": 2.20},
            "Ag": {"atomic_number": 47, "atomic_mass": 107.87, "electronegativity": 1.93},
            "Cd": {"atomic_number": 48, "atomic_mass": 112.41, "electronegativity": 1.69},
            "In": {"atomic_number": 49, "atomic_mass": 114.82, "electronegativity": 1.78},
            "Sn": {"atomic_number": 50, "atomic_mass": 118.71, "electronegativity": 1.96},
            "Sb": {"atomic_number": 51, "atomic_mass": 121.76, "electronegativity": 2.05},
            "Te": {"atomic_number": 52, "atomic_mass": 127.60, "electronegativity": 2.1},
            "I": {"atomic_number": 53, "atomic_mass": 126.90, "electronegativity": 2.66},
            "Cs": {"atomic_number": 55, "atomic_mass": 132.91, "electronegativity": 0.79},
            "Ba": {"atomic_number": 56, "atomic_mass": 137.33, "electronegativity": 0.89},
            "La": {"atomic_number": 57, "atomic_mass": 138.91, "electronegativity": 1.1},
            "Ce": {"atomic_number": 58, "atomic_mass": 140.12, "electronegativity": 1.12},
            "Hf": {"atomic_number": 72, "atomic_mass": 178.49, "electronegativity": 1.3},
            "Ta": {"atomic_number": 73, "atomic_mass": 180.95, "electronegativity": 1.5},
            "W": {"atomic_number": 74, "atomic_mass": 183.84, "electronegativity": 2.36},
            "Re": {"atomic_number": 75, "atomic_mass": 186.21, "electronegativity": 1.9},
            "Os": {"atomic_number": 76, "atomic_mass": 190.23, "electronegativity": 2.2},
            "Ir": {"atomic_number": 77, "atomic_mass": 192.22, "electronegativity": 2.20},
            "Pt": {"atomic_number": 78, "atomic_mass": 195.08, "electronegativity": 2.28},
            "Au": {"atomic_number": 79, "atomic_mass": 196.97, "electronegativity": 2.54},
            "Hg": {"atomic_number": 80, "atomic_mass": 200.59, "electronegativity": 2.00},
            "Tl": {"atomic_number": 81, "atomic_mass": 204.38, "electronegativity": 1.62},
            "Pb": {"atomic_number": 82, "atomic_mass": 207.2, "electronegativity": 2.33},
            "Bi": {"atomic_number": 83, "atomic_mass": 208.98, "electronegativity": 2.02},
        }
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称"""
        return self.feature_names


class CompositionFeatureLibrary:
    """成分特征库"""
    
    def __init__(self):
        self.extractor = MatminerFeatureExtractor()
        self.feature_cache = {}
    
    def featurize_batch(
        self,
        compositions: List[Union[str, Dict]]
    ) -> np.ndarray:
        """
        批量提取成分特征
        """
        features = []
        
        for comp in compositions:
            # 检查缓存
            cache_key = str(comp)
            if cache_key in self.feature_cache:
                features.append(self.feature_cache[cache_key])
            else:
                feat = self.extractor.extract_composition_features(comp)
                self.feature_cache[cache_key] = feat
                features.append(feat)
        
        # 对齐特征维度
        max_len = max(len(f) for f in features)
        aligned_features = []
        
        for feat in features:
            if len(feat) < max_len:
                feat = np.pad(feat, (0, max_len - len(feat)), mode='constant')
            aligned_features.append(feat[:max_len])
        
        return np.array(aligned_features)


def demonstrate_matminer_features():
    """演示Matminer特征提取"""
    print("Matminer Feature Extraction Demo")
    print("=" * 60)
    
    # 创建提取器
    config = MatminerConfig()
    extractor = MatminerFeatureExtractor(config)
    
    # 测试成分
    test_compositions = [
        "Fe2O3",
        "LiFePO4",
        "SiO2",
        "TiO2",
        "Al2O3"
    ]
    
    print("\nExtracting composition features...")
    for comp in test_compositions:
        features = extractor.extract_composition_features(comp)
        print(f"  {comp}: {len(features)} features")
    
    # 批量提取
    library = CompositionFeatureLibrary()
    batch_features = library.featurize_batch(test_compositions)
    print(f"\nBatch feature matrix shape: {batch_features.shape}")


if __name__ == "__main__":
    demonstrate_matminer_features()
