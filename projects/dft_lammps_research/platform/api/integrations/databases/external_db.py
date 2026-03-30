"""
External Database Integrations

Connectors for:
- Materials Project
- AFLOW
- OQMD
- NOMAD
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Iterator
from dataclasses import dataclass
import aiohttp
import structlog

logger = structlog.get_logger()


@dataclass
class ExternalStructure:
    """Structure from external database"""
    source: str
    external_id: str
    formula: str
    elements: List[str]
    structure: Dict[str, Any]
    properties: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class SearchQuery:
    """Search query for external databases"""
    elements: Optional[List[str]] = None
    formula: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    band_gap_range: Optional[tuple] = None
    e_above_hull_max: Optional[float] = None
    limit: int = 100


class ExternalDatabase(ABC):
    """Abstract base class for external database connectors"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def search(self, query: SearchQuery) -> List[ExternalStructure]:
        """Search for structures"""
        pass
    
    @abstractmethod
    async def get_structure(self, structure_id: str) -> Optional[ExternalStructure]:
        """Get structure by ID"""
        pass
    
    @abstractmethod
    async def get_properties(self, structure_id: str) -> Dict[str, Any]:
        """Get calculated properties"""
        pass


class MaterialsProjectConnector(ExternalDatabase):
    """
    Materials Project (MP) Database Connector
    
    API: https://api.materialsproject.org
    Documentation: https://docs.materialsproject.org/
    """
    
    BASE_URL = "https://api.materialsproject.org"
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        if not self.api_key:
            import os
            self.api_key = os.getenv("MP_API_KEY")
    
    async def search(self, query: SearchQuery) -> List[ExternalStructure]:
        """Search Materials Project database"""
        if not self.session:
            raise RuntimeError("Use async context manager")
        
        params = {
            "_limit": query.limit,
            "fields": "material_id,formula_pretty,structure,band_gap,energy_above_hull",
        }
        
        if query.elements:
            params["elements"] = ",".join(query.elements)
        if query.formula:
            params["formula"] = query.formula
        if query.band_gap_range:
            params["band_gap_min"] = query.band_gap_range[0]
            params["band_gap_max"] = query.band_gap_range[1]
        if query.e_above_hull_max is not None:
            params["energy_above_hull_max"] = query.e_above_hull_max
        
        headers = {"X-API-KEY": self.api_key} if self.api_key else {}
        
        async with self.session.get(
            f"{self.BASE_URL}/materials/core",
            params=params,
            headers=headers
        ) as resp:
            if resp.status != 200:
                logger.error("mp_search_failed", status=resp.status)
                return []
            
            data = await resp.json()
            results = []
            
            for item in data.get("data", []):
                structure = ExternalStructure(
                    source="materials_project",
                    external_id=item.get("material_id"),
                    formula=item.get("formula_pretty", ""),
                    elements=list(set(item.get("elements", []))),
                    structure=item.get("structure", {}),
                    properties={
                        "band_gap": item.get("band_gap"),
                        "energy_above_hull": item.get("energy_above_hull"),
                        "formation_energy": item.get("formation_energy_per_atom"),
                    },
                    metadata={
                        "symmetry": item.get("symmetry"),
                        "nsites": item.get("nsites"),
                    }
                )
                results.append(structure)
            
            logger.info("mp_search_complete", count=len(results))
            return results
    
    async def get_structure(self, material_id: str) -> Optional[ExternalStructure]:
        """Get structure by Materials Project ID"""
        if not self.session:
            raise RuntimeError("Use async context manager")
        
        headers = {"X-API-KEY": self.api_key} if self.api_key else {}
        
        async with self.session.get(
            f"{self.BASE_URL}/materials/core/{material_id}",
            headers=headers
        ) as resp:
            if resp.status != 200:
                return None
            
            data = await resp.json()
            item = data.get("data", {})
            
            return ExternalStructure(
                source="materials_project",
                external_id=item.get("material_id"),
                formula=item.get("formula_pretty", ""),
                elements=list(set(item.get("elements", []))),
                structure=item.get("structure", {}),
                properties={
                    "band_gap": item.get("band_gap"),
                    "energy_above_hull": item.get("energy_above_hull"),
                },
                metadata={
                    "symmetry": item.get("symmetry"),
                    "nsites": item.get("nsites"),
                }
            )
    
    async def get_properties(self, material_id: str) -> Dict[str, Any]:
        """Get all properties for a material"""
        if not self.session:
            raise RuntimeError("Use async context manager")
        
        headers = {"X-API-KEY": self.api_key} if self.api_key else {}
        
        async with self.session.get(
            f"{self.BASE_URL}/materials/core/{material_id}",
            headers=headers
        ) as resp:
            if resp.status != 200:
                return {}
            
            data = await resp.json()
            return data.get("data", {})
    
    async def get_phonon_data(self, material_id: str) -> Optional[Dict[str, Any]]:
        """Get phonon data for a material"""
        if not self.session:
            raise RuntimeError("Use async context manager")
        
        headers = {"X-API-KEY": self.api_key} if self.api_key else {}
        
        async with self.session.get(
            f"{self.BASE_URL}/materials/phonon/{material_id}",
            headers=headers
        ) as resp:
            if resp.status != 200:
                return None
            return await resp.json()
    
    async def get_electronic_structure(self, material_id: str) -> Optional[Dict[str, Any]]:
        """Get electronic structure data (DOS, band structure)"""
        if not self.session:
            raise RuntimeError("Use async context manager")
        
        headers = {"X-API-KEY": self.api_key} if self.api_key else {}
        
        async with self.session.get(
            f"{self.BASE_URL}/materials/electronic_structure/{material_id}",
            headers=headers
        ) as resp:
            if resp.status != 200:
                return None
            return await resp.json()


class AFLOWConnector(ExternalDatabase):
    """
    AFLOW (Automatic Flow) Database Connector
    
    API: http://aflowlib.org
    """
    
    BASE_URL = "http://aflowlib.org/API/aflux"
    
    async def search(self, query: SearchQuery) -> List[ExternalStructure]:
        """Search AFLOW database"""
        if not self.session:
            raise RuntimeError("Use async context manager")
        
        # Build AFLOW query
        filters = []
        
        if query.elements:
            filters.append(f"species({'|'.join(query.elements)})")
        if query.formula:
            filters.append(f"compound({query.formula})")
        if query.band_gap_range:
            filters.append(f"Egap({query.band_gap_range[0]},{query.band_gap_range[1]})")
        
        aflow_query = ",".join(filters) if filters else ""
        
        params = {
            "schema": "aflowlib",
            "catalog": "icsd",  # or "lib1", "lib2", "lib3"
        }
        
        if aflow_query:
            params["query"] = aflow_query
        
        async with self.session.get(
            self.BASE_URL,
            params=params
        ) as resp:
            if resp.status != 200:
                logger.error("aflow_search_failed", status=resp.status)
                return []
            
            data = await resp.json()
            results = []
            
            for item in data:
                structure = ExternalStructure(
                    source="aflow",
                    external_id=item.get("auid", ""),
                    formula=item.get("compound", ""),
                    elements=item.get("species", []),
                    structure={
                        "lattice": item.get("geometry", []),
                        "positions": item.get("positions_cartesian", []),
                    },
                    properties={
                        "band_gap": item.get("Egap", 0),
                        "energy_per_atom": item.get("energy_atom", 0),
                        "enthalpy_formation_atom": item.get("enthalpy_formation_atom", 0),
                    },
                    metadata={
                        "catalog": item.get("catalog", ""),
                        "spacegroup": item.get("spacegroup_relax", ""),
                    }
                )
                results.append(structure)
            
            return results
    
    async def get_structure(self, auid: str) -> Optional[ExternalStructure]:
        """Get structure by AFLOW UID"""
        # Implementation similar to search
        pass
    
    async def get_properties(self, auid: str) -> Dict[str, Any]:
        """Get properties by AFLOW UID"""
        pass


class OQMDConnector(ExternalDatabase):
    """
    OQMD (Open Quantum Materials Database) Connector
    
    API: http://oqmd.org
    """
    
    BASE_URL = "http://oqmd.org/oqmdapi"
    
    async def search(self, query: SearchQuery) -> List[ExternalStructure]:
        """Search OQMD database"""
        if not self.session:
            raise RuntimeError("Use async context manager")
        
        params = {
            "format": "json",
            "limit": query.limit,
        }
        
        if query.elements:
            params["element_set"] = ",".join(query.elements)
        if query.band_gap_range:
            params["band_gap"] = f"{query.band_gap_range[0]},{query.band_gap_range[1]}"
        if query.e_above_hull_max is not None:
            params["stability"] = f"0,{query.e_above_hull_max}"
        
        async with self.session.get(
            f"{self.BASE_URL}/formationenergy",
            params=params
        ) as resp:
            if resp.status != 200:
                return []
            
            data = await resp.json()
            results = []
            
            for item in data.get("data", []):
                entry = item.get("entry", {})
                calculation = item.get("calculation", {})
                
                structure = ExternalStructure(
                    source="oqmd",
                    external_id=str(entry.get("id", "")),
                    formula=entry.get("composition", ""),
                    elements=list(set(entry.get("elements", []))),
                    structure=calculation.get("output", {}).get("structure", {}),
                    properties={
                        "band_gap": calculation.get("band_gap", 0),
                        "energy_per_atom": item.get("energy_per_atom", 0),
                        "formation_energy": item.get("formationenergy", 0),
                    },
                    metadata={
                        "spacegroup": calculation.get("output", {}).get("spacegroup", ""),
                    }
                )
                results.append(structure)
            
            return results
    
    async def get_structure(self, entry_id: str) -> Optional[ExternalStructure]:
        """Get structure by entry ID"""
        pass
    
    async def get_properties(self, entry_id: str) -> Dict[str, Any]:
        """Get properties by entry ID"""
        pass


class DatabaseManager:
    """Manager for all external database connections"""
    
    def __init__(self):
        self.connectors: Dict[str, ExternalDatabase] = {}
    
    def register(self, name: str, connector: ExternalDatabase):
        """Register a database connector"""
        self.connectors[name] = connector
    
    async def search_all(
        self,
        query: SearchQuery,
        sources: Optional[List[str]] = None
    ) -> Dict[str, List[ExternalStructure]]:
        """Search across multiple databases"""
        sources = sources or list(self.connectors.keys())
        results = {}
        
        for name in sources:
            if name in self.connectors:
                connector = self.connectors[name]
                async with connector:
                    try:
                        results[name] = await connector.search(query)
                    except Exception as e:
                        logger.error(f"{name}_search_failed", error=str(e))
                        results[name] = []
        
        return results
    
    async def import_to_project(
        self,
        source: str,
        external_id: str,
        project_id: str,
        api_client  # DFT-LAMMPS API client
    ) -> Optional[Dict[str, Any]]:
        """Import structure from external database to project"""
        if source not in self.connectors:
            raise ValueError(f"Unknown source: {source}")
        
        connector = self.connectors[source]
        
        async with connector:
            structure = await connector.get_structure(external_id)
            
            if not structure:
                return None
            
            # Upload structure to DFT-LAMMPS project
            # This would use the SDK client
            uploaded = await api_client.structures.upload(
                project_id=project_id,
                name=f"{structure.formula}_{external_id}",
                format="json",
                data=structure.structure
            )
            
            return {
                "uploaded_id": uploaded.id,
                "source": source,
                "external_id": external_id,
                "properties": structure.properties,
            }


# Create global manager
db_manager = DatabaseManager()
