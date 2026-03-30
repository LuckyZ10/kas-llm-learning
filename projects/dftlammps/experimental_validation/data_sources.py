"""
实验数据源连接器
支持ICSD、Materials Project、AFLOWLib、COD等数据库
"""
import os
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
import requests
from urllib.parse import urlencode, urljoin
import numpy as np

from .data_formats import (
    CrystalStructure, ExperimentalProperty, ExperimentalDataset,
    AtomSite, Lattice
)


class DatabaseType(Enum):
    """支持的实验数据库类型"""
    MATERIALS_PROJECT = "materials_project"
    ICSD = "icsd"
    AFLOW = "aflow"
    COD = "cod"  # Crystallography Open Database
    OQMD = "oqmd"  # Open Quantum Materials Database
    NOMAD = "nomad"
    LOCAL = "local"


@dataclass
class DatabaseConfig:
    """数据库连接配置"""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3
    rate_limit: float = 0.5  # 请求间隔（秒）
    cache_dir: Optional[str] = None
    use_cache: bool = True
    cache_ttl: int = 86400  # 缓存有效期（秒）


@dataclass
class SearchQuery:
    """数据库搜索查询"""
    elements: Optional[List[str]] = None
    formula: Optional[str] = None
    space_group: Optional[str] = None
    property_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    exclude_elements: List[str] = field(default_factory=list)
    min_atoms: Optional[int] = None
    max_atoms: Optional[int] = None
    has_structure: bool = True
    has_properties: bool = True


@dataclass
class SearchResult:
    """搜索结果"""
    entry_id: str
    formula: str
    database: DatabaseType
    structure: Optional[CrystalStructure] = None
    properties: List[ExperimentalProperty] = field(default_factory=list)
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatabaseConnector(ABC):
    """数据库连接器抽象基类"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._session = requests.Session()
        self._last_request_time = 0
        self._cache: Dict[str, Any] = {}
        
        if config.cache_dir and not os.path.exists(config.cache_dir):
            os.makedirs(config.cache_dir)
    
    @property
    @abstractmethod
    def database_type(self) -> DatabaseType:
        """返回数据库类型"""
        pass
    
    @abstractmethod
    def search(self, query: SearchQuery, limit: int = 100) -> List[SearchResult]:
        """搜索数据库"""
        pass
    
    @abstractmethod
    def get_structure(self, entry_id: str) -> Optional[CrystalStructure]:
        """获取晶体结构"""
        pass
    
    @abstractmethod
    def get_properties(self, entry_id: str) -> List[ExperimentalProperty]:
        """获取实验属性"""
        pass
    
    def _rate_limited_request(self, url: str, **kwargs) -> requests.Response:
        """带速率限制的请求"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit:
            time.sleep(self.config.rate_limit - elapsed)
        
        for attempt in range(self.config.retry_count):
            try:
                response = self._session.get(url, timeout=self.config.timeout, **kwargs)
                self._last_request_time = time.time()
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt == self.config.retry_count - 1:
                    raise
                time.sleep(2 ** attempt)  # 指数退避
        
        raise RuntimeError("Request failed after retries")
    
    def _get_cache_key(self, *args) -> str:
        """生成缓存键"""
        return f"{self.database_type.value}_" + "_".join(str(a) for a in args)
    
    def _load_from_cache(self, key: str) -> Optional[Any]:
        """从缓存加载"""
        if not self.config.use_cache:
            return None
        
        # 内存缓存
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self.config.cache_ttl:
                return data
        
        # 文件缓存
        if self.config.cache_dir:
            cache_file = os.path.join(self.config.cache_dir, f"{key}.json")
            if os.path.exists(cache_file):
                mtime = os.path.getmtime(cache_file)
                if time.time() - mtime < self.config.cache_ttl:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
        
        return None
    
    def _save_to_cache(self, key: str, data: Any) -> None:
        """保存到缓存"""
        self._cache[key] = (data, time.time())
        
        if self.config.cache_dir:
            cache_file = os.path.join(self.config.cache_dir, f"{key}.json")
            with open(cache_file, 'w') as f:
                json.dump(data, f, default=str)


class MaterialsProjectConnector(DatabaseConnector):
    """Materials Project数据库连接器"""
    
    BASE_URL = "https://api.materialsproject.org"
    
    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.MATERIALS_PROJECT
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        if config is None:
            config = DatabaseConfig()
        if config.api_key is None:
            config.api_key = os.environ.get('MP_API_KEY')
        super().__init__(config)
    
    def search(self, query: SearchQuery, limit: int = 100) -> List[SearchResult]:
        """搜索Materials Project"""
        cache_key = self._get_cache_key("search", hash(str(query)))
        cached = self._load_from_cache(cache_key)
        if cached:
            return [SearchResult(**r) for r in cached]
        
        # 构建查询参数
        params = {"_limit": limit}
        
        if query.elements:
            params["formula"] = "-".join(sorted(query.elements))
        if query.formula:
            params["formula"] = query.formula
        
        headers = {"X-API-KEY": self.config.api_key} if self.config.api_key else {}
        
        url = f"{self.BASE_URL}/materials/core"
        
        try:
            response = self._rate_limited_request(url, params=params, headers=headers)
            data = response.json()
            
            results = []
            for item in data.get('data', []):
                result = SearchResult(
                    entry_id=item.get('material_id', ''),
                    formula=item.get('formula_pretty', 'Unknown'),
                    database=self.database_type,
                    url=f"https://materialsproject.org/materials/{item.get('material_id')}",
                    metadata=item
                )
                results.append(result)
            
            self._save_to_cache(cache_key, [r.__dict__ for r in results])
            return results
            
        except Exception as e:
            print(f"Materials Project API error: {e}")
            return []
    
    def get_structure(self, material_id: str) -> Optional[CrystalStructure]:
        """获取晶体结构"""
        cache_key = self._get_cache_key("structure", material_id)
        cached = self._load_from_cache(cache_key)
        if cached:
            return self._dict_to_structure(cached)
        
        headers = {"X-API-KEY": self.config.api_key} if self.config.api_key else {}
        url = f"{self.BASE_URL}/materials/core/{material_id}"
        
        try:
            response = self._rate_limited_request(url, headers=headers)
            data = response.json()
            
            if 'data' in data:
                structure_data = data['data']
                structure = self._parse_mp_structure(structure_data)
                self._save_to_cache(cache_key, structure.__dict__)
                return structure
            
        except Exception as e:
            print(f"Error fetching structure {material_id}: {e}")
        
        return None
    
    def _parse_mp_structure(self, data: Dict) -> CrystalStructure:
        """解析MP结构数据"""
        # 解析晶格
        lattice_data = data.get('structure', {}).get('lattice', {})
        lattice = Lattice(
            a=lattice_data.get('a', 1.0),
            b=lattice_data.get('b', 1.0),
            c=lattice_data.get('c', 1.0),
            alpha=lattice_data.get('alpha', 90.0),
            beta=lattice_data.get('beta', 90.0),
            gamma=lattice_data.get('gamma', 90.0)
        )
        
        # 解析位点
        sites = []
        for site in data.get('structure', {}).get('sites', []):
            sites.append(AtomSite(
                element=site['species'][0]['element'] if site['species'] else 'X',
                x=site['abc'][0],
                y=site['abc'][1],
                z=site['abc'][2]
            ))
        
        return CrystalStructure(
            formula=data.get('formula_pretty', 'Unknown'),
            lattice=lattice,
            sites=sites,
            space_group=str(data.get('symmetry', {}).get('symbol', '')),
            space_group_number=data.get('symmetry', {}).get('number'),
            source='Materials Project',
            source_id=data.get('material_id'),
            metadata=data
        )
    
    def get_properties(self, material_id: str) -> List[ExperimentalProperty]:
        """获取计算/实验属性"""
        cache_key = self._get_cache_key("properties", material_id)
        cached = self._load_from_cache(cache_key)
        if cached:
            return [ExperimentalProperty(**p) for p in cached]
        
        headers = {"X-API-KEY": self.config.api_key} if self.config.api_key else {}
        
        properties = []
        
        # 获取电子结构数据
        try:
            url = f"{self.BASE_URL}/materials/electronic_structure/{material_id}"
            response = self._rate_limited_request(url, headers=headers)
            data = response.json()
            
            if 'data' in data:
                es_data = data['data']
                if 'band_gap' in es_data:
                    properties.append(ExperimentalProperty(
                        name='band_gap',
                        value=es_data['band_gap'],
                        unit='eV',
                        method='DFT (PBE)'
                    ))
        except:
            pass
        
        # 获取热力学数据
        try:
            url = f"{self.BASE_URL}/materials/thermo/{material_id}"
            response = self._rate_limited_request(url, headers=headers)
            data = response.json()
            
            if 'data' in data:
                thermo_data = data['data']
                if 'energy_per_atom' in thermo_data:
                    properties.append(ExperimentalProperty(
                        name='energy_per_atom',
                        value=thermo_data['energy_per_atom'],
                        unit='eV/atom',
                        method='DFT'
                    ))
                if 'formation_energy_per_atom' in thermo_data:
                    properties.append(ExperimentalProperty(
                        name='formation_energy',
                        value=thermo_data['formation_energy_per_atom'],
                        unit='eV/atom',
                        method='DFT'
                    ))
        except:
            pass
        
        self._save_to_cache(cache_key, [p.__dict__ for p in properties])
        return properties
    
    def _dict_to_structure(self, data: Dict) -> CrystalStructure:
        """从字典重建结构"""
        lattice_data = data.get('lattice', {})
        lattice = Lattice(**lattice_data)
        
        sites = [AtomSite(**s) for s in data.get('sites', [])]
        
        return CrystalStructure(
            formula=data.get('formula', 'Unknown'),
            lattice=lattice,
            sites=sites,
            space_group=data.get('space_group'),
            space_group_number=data.get('space_group_number'),
            source=data.get('source'),
            source_id=data.get('source_id'),
            metadata=data.get('metadata', {})
        )


class AFLOWConnector(DatabaseConnector):
    """AFLOWLib数据库连接器"""
    
    BASE_URL = "https://aflowlib.duke.edu/search/API"
    
    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.AFLOW
    
    def search(self, query: SearchQuery, limit: int = 100) -> List[SearchResult]:
        """搜索AFLOW数据库"""
        cache_key = self._get_cache_key("search", hash(str(query)))
        cached = self._load_from_cache(cache_key)
        if cached:
            return [SearchResult(**r) for r in cached]
        
        # 构建匹配条件
        match_conditions = []
        
        if query.elements:
            match_conditions.append(f"(species:{'%2C'.join(query.elements)})")
        if query.formula:
            match_conditions.append(f"(compound:{query.formula})")
        
        match_str = ",".join(match_conditions) if match_conditions else ""
        
        # 选择返回的属性
        properties = [
            "auid", "aurl", "compound", "species", "nspecies",
            "spacegroup_relax", "geometry", "ldau_TLUJ"
        ]
        
        params = {
            "catalog": "PAULING_FILE,ICSD,LIB1,LIB2,LIB3",
            "format": "json"
        }
        
        url = f"{self.BASE_URL}/{match_str}?{urlencode(params)}" if match_str else f"{self.BASE_URL}?{urlencode(params)}"
        
        try:
            response = self._rate_limited_request(url)
            data = response.json()
            
            results = []
            for item in data:
                result = SearchResult(
                    entry_id=item.get('auid', ''),
                    formula=item.get('compound', 'Unknown'),
                    database=self.database_type,
                    url=item.get('aurl', ''),
                    metadata=item
                )
                results.append(result)
            
            self._save_to_cache(cache_key, [r.__dict__ for r in results[:limit]])
            return results[:limit]
            
        except Exception as e:
            print(f"AFLOW API error: {e}")
            return []
    
    def get_structure(self, entry_id: str) -> Optional[CrystalStructure]:
        """获取晶体结构"""
        cache_key = self._get_cache_key("structure", entry_id)
        cached = self._load_from_cache(cache_key)
        if cached:
            return self._dict_to_structure_aflow(cached)
        
        url = f"{self.BASE_URL}/?auid={entry_id},format=json"
        
        try:
            response = self._rate_limited_request(url)
            data = response.json()
            
            if data:
                structure = self._parse_aflow_structure(data[0])
                self._save_to_cache(cache_key, structure.__dict__)
                return structure
            
        except Exception as e:
            print(f"Error fetching AFLOW structure {entry_id}: {e}")
        
        return None
    
    def _parse_aflow_structure(self, data: Dict) -> CrystalStructure:
        """解析AFLOW结构数据"""
        geometry = data.get('geometry', [])
        
        if len(geometry) >= 6:
            lattice = Lattice(
                a=geometry[0],
                b=geometry[1],
                c=geometry[2],
                alpha=geometry[3],
                beta=geometry[4],
                gamma=geometry[5]
            )
        else:
            lattice = Lattice(1, 1, 1, 90, 90, 90)
        
        # 获取位点数据
        sites = []
        # AFLOW格式需要解析geometry_orig或geometry_relax
        # 这里简化处理
        
        return CrystalStructure(
            formula=data.get('compound', 'Unknown'),
            lattice=lattice,
            sites=sites,
            space_group=str(data.get('spacegroup_relax', '')),
            source='AFLOW',
            source_id=data.get('auid'),
            metadata=data
        )
    
    def _dict_to_structure_aflow(self, data: Dict) -> CrystalStructure:
        """从字典重建AFLOW结构"""
        return CrystalStructure(
            formula=data.get('formula', 'Unknown'),
            lattice=Lattice(**data.get('lattice', {})),
            sites=[AtomSite(**s) for s in data.get('sites', [])],
            space_group=data.get('space_group'),
            source=data.get('source'),
            source_id=data.get('source_id')
        )
    
    def get_properties(self, entry_id: str) -> List[ExperimentalProperty]:
        """获取属性数据"""
        properties = []
        # AFLOW属性获取实现
        return properties


class CODConnector(DatabaseConnector):
    """Crystallography Open Database连接器"""
    
    BASE_URL = "http://www.crystallography.net/cod/result"
    
    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.COD
    
    def search(self, query: SearchQuery, limit: int = 100) -> List[SearchResult]:
        """搜索COD数据库"""
        params = {
            "format": "json",
            "limit": limit
        }
        
        if query.formula:
            params["formula"] = query.formula
        if query.elements:
            params["el1"] = query.elements[0] if len(query.elements) > 0 else None
            params["el2"] = query.elements[1] if len(query.elements) > 1 else None
        if query.space_group:
            params["spacegroup"] = query.space_group
        
        url = f"{self.BASE_URL}?{urlencode({k: v for k, v in params.items() if v is not None})}"
        
        try:
            response = self._rate_limited_request(url)
            data = response.json()
            
            results = []
            for item in data:
                result = SearchResult(
                    entry_id=str(item.get('file', '')),
                    formula=item.get('formula', 'Unknown'),
                    database=self.database_type,
                    url=f"http://www.crystallography.net/cod/{item.get('file')}.html",
                    metadata=item
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"COD API error: {e}")
            return []
    
    def get_structure(self, entry_id: str) -> Optional[CrystalStructure]:
        """获取CIF结构"""
        url = f"http://www.crystallography.net/cod/{entry_id}.cif"
        
        try:
            response = self._rate_limited_request(url)
            # 使用CIF解析器处理
            cif_content = response.text
            
            # 临时保存并解析
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as f:
                f.write(cif_content)
                temp_path = f.name
            
            from .data_formats import CIFHandler
            handler = CIFHandler()
            structure = handler.read_structure(temp_path)
            os.unlink(temp_path)
            
            structure.source = 'COD'
            structure.source_id = entry_id
            return structure
            
        except Exception as e:
            print(f"Error fetching COD structure {entry_id}: {e}")
            return None
    
    def get_properties(self, entry_id: str) -> List[ExperimentalProperty]:
        """获取属性"""
        return []


class DatabaseManager:
    """数据库管理器 - 统一管理多个数据源"""
    
    def __init__(self):
        self._connectors: Dict[DatabaseType, DatabaseConnector] = {}
        self._default_connector: Optional[DatabaseType] = None
    
    def register_connector(self, connector: DatabaseConnector, set_default: bool = False):
        """注册连接器"""
        self._connectors[connector.database_type] = connector
        if set_default or self._default_connector is None:
            self._default_connector = connector.database_type
    
    def get_connector(self, db_type: Optional[DatabaseType] = None) -> Optional[DatabaseConnector]:
        """获取连接器"""
        if db_type is None:
            db_type = self._default_connector
        return self._connectors.get(db_type)
    
    def search_all(self, query: SearchQuery, limit_per_db: int = 50) -> Dict[DatabaseType, List[SearchResult]]:
        """在所有数据库中搜索"""
        results = {}
        for db_type, connector in self._connectors.items():
            try:
                results[db_type] = connector.search(query, limit_per_db)
            except Exception as e:
                print(f"Error searching {db_type}: {e}")
                results[db_type] = []
        return results
    
    def get_structure_any(self, entry_id: str, preferred_db: Optional[DatabaseType] = None) -> Optional[CrystalStructure]:
        """从任意可用数据库获取结构"""
        if preferred_db:
            connector = self._connectors.get(preferred_db)
            if connector:
                return connector.get_structure(entry_id)
        
        for connector in self._connectors.values():
            try:
                structure = connector.get_structure(entry_id)
                if structure:
                    return structure
            except:
                continue
        
        return None


# =============================================================================
# 便捷函数
# =============================================================================

def create_mp_connector(api_key: Optional[str] = None) -> MaterialsProjectConnector:
    """创建Materials Project连接器"""
    config = DatabaseConfig(api_key=api_key)
    return MaterialsProjectConnector(config)


def create_aflow_connector() -> AFLOWConnector:
    """创建AFLOW连接器"""
    return AFLOWConnector(DatabaseConfig())


def create_cod_connector() -> CODConnector:
    """创建COD连接器"""
    return CODConnector(DatabaseConfig())


def create_default_manager() -> DatabaseManager:
    """创建默认数据库管理器"""
    manager = DatabaseManager()
    
    # 尝试注册可用的连接器
    try:
        manager.register_connector(create_mp_connector(), set_default=True)
    except:
        pass
    
    try:
        manager.register_connector(create_aflow_connector())
    except:
        pass
    
    try:
        manager.register_connector(create_cod_connector())
    except:
        pass
    
    return manager


# =============================================================================
# 演示
# =============================================================================

def demo():
    """演示数据源连接器"""
    print("=" * 80)
    print("🌐 实验数据源连接器演示")
    print("=" * 80)
    
    # 创建数据库管理器
    manager = DatabaseManager()
    
    print("\n🔹 注册数据源:")
    
    # Materials Project
    mp_connector = create_mp_connector()
    manager.register_connector(mp_connector, set_default=True)
    print("   ✓ Materials Project")
    
    # AFLOW
    aflow_connector = create_aflow_connector()
    manager.register_connector(aflow_connector)
    print("   ✓ AFLOW")
    
    # COD
    cod_connector = create_cod_connector()
    manager.register_connector(cod_connector)
    print("   ✓ Crystallography Open Database (COD)")
    
    print("\n🔹 示例搜索查询:")
    query = SearchQuery(
        elements=['Li', 'Fe', 'P', 'O'],
        has_structure=True,
        has_properties=True
    )
    print(f"   元素: {query.elements}")
    print(f"   要求有结构数据: {query.has_structure}")
    print(f"   要求有属性数据: {query.has_properties}")
    
    print("\n🔹 数据库元数据:")
    print(f"   已注册连接器: {list(manager._connectors.keys())}")
    print(f"   默认连接器: {manager._default_connector}")
    
    # 演示属性对象
    print("\n🔹 属性数据示例:")
    prop = ExperimentalProperty(
        name='band_gap',
        value=3.4,
        unit='eV',
        uncertainty=0.15,
        temperature=300,
        method='UV-Vis spectroscopy',
        instrument='Cary 5000',
        reference='DOI:10.1000/example'
    )
    print(f"   {prop}")
    print(f"   相对不确定度: {prop.relative_uncertainty*100:.2f}%")
    
    print("\n" + "=" * 80)
    print("✅ 数据源连接器演示完成!")
    print("=" * 80)
    
    return manager


if __name__ == '__main__':
    demo()
