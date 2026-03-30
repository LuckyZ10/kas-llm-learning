"""
Database Connector
==================
实验数据库连接器

支持的数据库:
- ICSD (Inorganic Crystal Structure Database)
- Materials Project
- COD (Crystallography Open Database)
- AFLOWLIB
- OQMD
"""

import os
import json
import time
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import logging
from urllib.parse import urlencode

from .base_connector import BaseConnector, ExperimentalData, DataMetadata

logger = logging.getLogger(__name__)


@dataclass
class DatabaseQuery:
    """数据库查询条件"""
    formula: Optional[str] = None
    elements: Optional[List[str]] = None
    space_group: Optional[str] = None
    min_volume: Optional[float] = None
    max_volume: Optional[float] = None
    min_density: Optional[float] = None
    max_density: Optional[float] = None
    has_xrd: bool = False
    has_properties: bool = False
    
    def to_dict(self) -> Dict:
        result = {}
        if self.formula:
            result['formula'] = self.formula
        if self.elements:
            result['elements'] = self.elements
        if self.space_group:
            result['space_group'] = self.space_group
        if self.min_volume is not None:
            result['min_volume'] = self.min_volume
        if self.max_volume is not None:
            result['max_volume'] = self.max_volume
        return result


class DatabaseConnector(BaseConnector):
    """
    数据库连接器基类
    
    为各种材料数据库提供统一接口
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.api_key = self.config.get('api_key')
        self.base_url = self.config.get('base_url', '')
        self.cache_dir = Path(self.config.get('cache_dir', './database_cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = self.config.get('use_cache', True)
        self.rate_limit = self.config.get('rate_limit', 0.1)  # 请求间隔（秒）
        self.last_request_time = 0
    
    def _rate_limited_request(self):
        """确保请求速率限制"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _get_cache_path(self, query_hash: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{query_hash}.json"
    
    def _load_from_cache(self, query_hash: str) -> Optional[Dict]:
        """从缓存加载数据"""
        if not self.use_cache:
            return None
        
        cache_path = self._get_cache_path(query_hash)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def _save_to_cache(self, query_hash: str, data: Dict):
        """保存数据到缓存"""
        if not self.use_cache:
            return
        
        cache_path = self._get_cache_path(query_hash)
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @abstractmethod
    def search(self, query: DatabaseQuery) -> List[Dict]:
        """搜索数据库"""
        pass
    
    @abstractmethod
    def get_structure(self, material_id: str) -> Optional[Dict]:
        """获取晶体结构"""
        pass
    
    @abstractmethod
    def get_xrd_pattern(self, material_id: str, wavelength: float = 1.5406) -> Optional[ExperimentalData]:
        """获取XRD图谱"""
        pass


class MaterialsProjectConnector(DatabaseConnector):
    """
    Materials Project数据库连接器
    
    Materials Project是一个开放的材料计算数据库
    
    需要API密钥，可从 https://materialsproject.org 获取
    """
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        config.setdefault('base_url', 'https://api.materialsproject.org')
        super().__init__(config)
        
        if not self.api_key:
            # 尝试从环境变量获取
            self.api_key = os.environ.get('MP_API_KEY')
        
        self.headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
    
    def validate_format(self, filepath: str) -> bool:
        """验证文件格式（数据库连接器不使用文件）"""
        return False
    
    def read(self, filepath: str, **kwargs) -> ExperimentalData:
        """数据库连接器不支持直接文件读取"""
        raise NotImplementedError("Use search() or get_structure() methods instead")
    
    def search(self, query: DatabaseQuery) -> List[Dict]:
        """
        搜索Materials Project数据库
        
        Args:
            query: 搜索条件
            
        Returns:
            材料列表
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests is required for database access")
        
        self._rate_limited_request()
        
        # 构建查询参数
        params = query.to_dict()
        
        # 构建URL
        url = f"{self.base_url}/materials/core/summary/"
        
        # 添加API密钥到参数
        params['api_key'] = self.api_key
        
        # 检查缓存
        query_hash = str(hash(json.dumps(params, sort_keys=True)))
        cached = self._load_from_cache(query_hash)
        if cached:
            return cached.get('data', [])
        
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # 保存到缓存
            self._save_to_cache(query_hash, data)
            
            return data.get('data', [])
            
        except Exception as e:
            logger.error(f"Failed to query Materials Project: {e}")
            return []
    
    def get_structure(self, material_id: str) -> Optional[Dict]:
        """
        获取晶体结构
        
        Args:
            material_id: Materials Project材料ID
            
        Returns:
            结构数据
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests is required for database access")
        
        self._rate_limited_request()
        
        # 检查缓存
        cache_path = self._get_cache_path(f"structure_{material_id}")
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        url = f"{self.base_url}/materials/core/summary/{material_id}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # 保存到缓存
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return data.get('data', {})
            
        except Exception as e:
            logger.error(f"Failed to get structure {material_id}: {e}")
            return None
    
    def get_xrd_pattern(self, material_id: str, wavelength: float = 1.5406) -> Optional[ExperimentalData]:
        """
        获取XRD图谱
        
        使用pymatgen计算理论XRD图谱
        
        Args:
            material_id: 材料ID
            wavelength: X射线波长
            
        Returns:
            XRD数据
        """
        structure_data = self.get_structure(material_id)
        
        if not structure_data:
            return None
        
        try:
            from pymatgen.core import Structure, Lattice
            from pymatgen.analysis.diffraction.xrd import XRDCalculator
            
            # 从MP数据构建结构
            lattice = Lattice.from_parameters(
                a=structure_data['structure']['lattice']['a'],
                b=structure_data['structure']['lattice']['b'],
                c=structure_data['structure']['lattice']['c'],
                alpha=structure_data['structure']['lattice']['alpha'],
                beta=structure_data['structure']['lattice']['beta'],
                gamma=structure_data['structure']['lattice']['gamma']
            )
            
            structure = Structure(
                lattice=lattice,
                species=[s['label'] for s in structure_data['structure']['sites']],
                coords=[s['xyz'] for s in structure_data['structure']['sites']]
            )
            
            # 计算XRD
            calculator = XRDCalculator(wavelength=wavelength)
            pattern = calculator.get_pattern(structure)
            
            data = np.column_stack([pattern.x, pattern.y])
            
            return ExperimentalData(
                data_type='xrd_theoretical',
                raw_data=data,
                column_names=['two_theta', 'intensity'],
                units={'two_theta': 'deg', 'intensity': 'a.u.'},
                metadata=DataMetadata(
                    source=f"Materials Project: {material_id}",
                    data_type='xrd_theoretical',
                    conditions={
                        'material_id': material_id,
                        'formula': structure_data.get('formula_pretty'),
                        'wavelength': wavelength,
                        'space_group': structure_data.get('symmetry', {}).get('symbol')
                    }
                )
            )
            
        except ImportError:
            logger.error("pymatgen is required for XRD calculation")
            return None
    
    def get_band_gap(self, material_id: str) -> Optional[float]:
        """获取带隙"""
        data = self.get_structure(material_id)
        if data and 'band_gap' in data:
            return data['band_gap']
        return None
    
    def get_formation_energy(self, material_id: str) -> Optional[float]:
        """获取形成能"""
        data = self.get_structure(material_id)
        if data and 'formation_energy_per_atom' in data:
            return data['formation_energy_per_atom']
        return None
    
    def get_dielectric_constant(self, material_id: str) -> Optional[Dict]:
        """获取介电常数"""
        data = self.get_structure(material_id)
        if data and 'dielectric' in data:
            return data['dielectric']
        return None
    
    def get_elastic_properties(self, material_id: str) -> Optional[Dict]:
        """获取弹性性质"""
        data = self.get_structure(material_id)
        if data and 'elasticity' in data:
            return data['elasticity']
        return None
    
    def query_by_elements(self, elements: List[str], 
                         max_results: int = 100) -> List[Dict]:
        """
        按元素查询材料
        
        Args:
            elements: 元素列表
            max_results: 最大结果数
            
        Returns:
            材料列表
        """
        query = DatabaseQuery(elements=elements)
        results = self.search(query)
        return results[:max_results]
    
    def query_by_formula(self, formula: str) -> List[Dict]:
        """
        按化学式查询材料
        
        Args:
            formula: 化学式，如 "LiFePO4"
            
        Returns:
            材料列表
        """
        query = DatabaseQuery(formula=formula)
        return self.search(query)


class ICSDConnector(DatabaseConnector):
    """
    ICSD (Inorganic Crystal Structure Database) 连接器
    
    ICSD是实验晶体结构数据库，需要订阅访问
    
    参考: https://icsd.fiz-karlsruhe.de/
    """
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        config.setdefault('base_url', 'https://icsd.fiz-karlsruhe.de/ws')
        super().__init__(config)
        
        self.username = self.config.get('username')
        self.password = self.config.get('password')
        
        if not self.username or not self.password:
            self.username = os.environ.get('ICSD_USERNAME')
            self.password = os.environ.get('ICSD_PASSWORD')
        
        self.auth_token = None
    
    def _authenticate(self):
        """获取认证令牌"""
        if self.auth_token:
            return
        
        try:
            import requests
            
            url = f"{self.base_url}/auth/token"
            response = requests.post(url, auth=(self.username, self.password))
            response.raise_for_status()
            
            self.auth_token = response.json().get('token')
            
        except Exception as e:
            logger.error(f"ICSD authentication failed: {e}")
            raise
    
    def validate_format(self, filepath: str) -> bool:
        return False
    
    def read(self, filepath: str, **kwargs) -> ExperimentalData:
        raise NotImplementedError("Use search() or get_structure() methods instead")
    
    def search(self, query: DatabaseQuery) -> List[Dict]:
        """搜索ICSD数据库"""
        self._authenticate()
        
        try:
            import requests
            
            self._rate_limited_request()
            
            url = f"{self.base_url}/search/experimental"
            headers = {'Authorization': f'Bearer {self.auth_token}'}
            
            # 构建查询
            search_params = {}
            if query.formula:
                search_params['formula'] = query.formula
            if query.elements:
                search_params['elements'] = ','.join(query.elements)
            if query.space_group:
                search_params['spacegroup'] = query.space_group
            
            response = requests.get(url, params=search_params, headers=headers, timeout=30)
            response.raise_for_status()
            
            return response.json().get('structures', [])
            
        except Exception as e:
            logger.error(f"ICSD search failed: {e}")
            return []
    
    def get_structure(self, icsd_id: str) -> Optional[Dict]:
        """获取晶体结构"""
        self._authenticate()
        
        try:
            import requests
            
            self._rate_limited_request()
            
            url = f"{self.base_url}/structures/{icsd_id}"
            headers = {'Authorization': f'Bearer {self.auth_token}'}
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get ICSD structure {icsd_id}: {e}")
            return None
    
    def get_xrd_pattern(self, icsd_id: str, wavelength: float = 1.5406) -> Optional[ExperimentalData]:
        """获取XRD图谱"""
        structure_data = self.get_structure(icsd_id)
        
        if not structure_data:
            return None
        
        try:
            from pymatgen.core import Structure, Lattice
            from pymatgen.analysis.diffraction.xrd import XRDCalculator
            
            lattice_params = structure_data.get('cell', {})
            lattice = Lattice.from_parameters(
                a=lattice_params.get('a', 1),
                b=lattice_params.get('b', 1),
                c=lattice_params.get('c', 1),
                alpha=lattice_params.get('alpha', 90),
                beta=lattice_params.get('beta', 90),
                gamma=lattice_params.get('gamma', 90)
            )
            
            sites = structure_data.get('sites', [])
            structure = Structure(
                lattice=lattice,
                species=[s['element'] for s in sites],
                coords=[s['frac_coords'] for s in sites]
            )
            
            calculator = XRDCalculator(wavelength=wavelength)
            pattern = calculator.get_pattern(structure)
            
            data = np.column_stack([pattern.x, pattern.y])
            
            return ExperimentalData(
                data_type='xrd_experimental',
                raw_data=data,
                column_names=['two_theta', 'intensity'],
                units={'two_theta': 'deg', 'intensity': 'a.u.'},
                metadata=DataMetadata(
                    source=f"ICSD: {icsd_id}",
                    data_type='xrd_experimental',
                    conditions={
                        'icsd_id': icsd_id,
                        'formula': structure_data.get('formula'),
                        'wavelength': wavelength,
                        'space_group': structure_data.get('spacegroup')
                    }
                )
            )
            
        except ImportError:
            logger.error("pymatgen is required for XRD calculation")
            return None


class CODConnector(DatabaseConnector):
    """
    COD (Crystallography Open Database) 连接器
    
    COD是一个开放获取的晶体学数据库
    
    参考: http://www.crystallography.net/cod/
    """
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        config.setdefault('base_url', 'http://www.crystallography.net/cod/result')
        super().__init__(config)
    
    def validate_format(self, filepath: str) -> bool:
        return False
    
    def read(self, filepath: str, **kwargs) -> ExperimentalData:
        raise NotImplementedError("Use search() or get_structure() methods instead")
    
    def search(self, query: DatabaseQuery) -> List[Dict]:
        """搜索COD数据库"""
        try:
            import requests
            
            self._rate_limited_request()
            
            params = {}
            if query.formula:
                params['formula'] = query.formula
            if query.elements:
                params['el1'] = query.elements[0] if len(query.elements) > 0 else ''
                params['el2'] = query.elements[1] if len(query.elements) > 1 else ''
            if query.space_group:
                params['sg'] = query.space_group
            
            params['format'] = 'json'
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"COD search failed: {e}")
            return []
    
    def get_structure(self, cod_id: str) -> Optional[Dict]:
        """获取CIF格式的晶体结构"""
        try:
            import requests
            
            self._rate_limited_request()
            
            url = f"http://www.crystallography.net/cod/{cod_id}.cif"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            return {'cif_data': response.text, 'cod_id': cod_id}
            
        except Exception as e:
            logger.error(f"Failed to get COD structure {cod_id}: {e}")
            return None
    
    def get_xrd_pattern(self, cod_id: str, wavelength: float = 1.5406) -> Optional[ExperimentalData]:
        """获取XRD图谱"""
        structure_data = self.get_structure(cod_id)
        
        if not structure_data:
            return None
        
        try:
            from pymatgen.core import Structure
            from pymatgen.analysis.diffraction.xrd import XRDCalculator
            
            # 从CIF解析结构
            from io import StringIO
            structure = Structure.from_str(structure_data['cif_data'], fmt='cif')
            
            calculator = XRDCalculator(wavelength=wavelength)
            pattern = calculator.get_pattern(structure)
            
            data = np.column_stack([pattern.x, pattern.y])
            
            return ExperimentalData(
                data_type='xrd_experimental',
                raw_data=data,
                column_names=['two_theta', 'intensity'],
                units={'two_theta': 'deg', 'intensity': 'a.u.'},
                metadata=DataMetadata(
                    source=f"COD: {cod_id}",
                    data_type='xrd_experimental',
                    conditions={
                        'cod_id': cod_id,
                        'formula': structure.formula,
                        'wavelength': wavelength
                    }
                )
            )
            
        except ImportError:
            logger.error("pymatgen is required for CIF parsing")
            return None


# 便捷函数
def query_materials_project(elements: List[str] = None, 
                            formula: str = None,
                            api_key: str = None) -> List[Dict]:
    """便捷函数：查询Materials Project"""
    config = {'api_key': api_key} if api_key else {}
    connector = MaterialsProjectConnector(config)
    query = DatabaseQuery(elements=elements, formula=formula)
    return connector.search(query)
