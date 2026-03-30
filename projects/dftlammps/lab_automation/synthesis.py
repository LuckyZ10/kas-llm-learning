"""
Synthesis Planning Module for Laboratory Automation

Provides intelligent synthesis planners for:
- Powder synthesis (solid-state, sol-gel, hydrothermal, etc.)
- Thin film deposition (CVD, PVD, ALD, spin coating, etc.)

Includes recipe management, process optimization, and parameter control.
"""

import json
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np


logger = logging.getLogger(__name__)


class SynthesisMethod(Enum):
    """Synthesis method types"""
    SOLID_STATE = "solid_state"
    SOL_GEL = "sol_gel"
    HYDROTHERMAL = "hydrothermal"
    CO_PRECIPITATION = "co_precipitation"
    MECHANOCHEMICAL = "mechanochemical"
    MOLTEN_SALT = "molten_salt"


class DepositionMethod(Enum):
    """Thin film deposition methods"""
    CVD = "cvd"
    PVD = "pvd"
    ALD = "ald"
    SPIN_COATING = "spin_coating"
    DIP_COATING = "dip_coating"
    SPRAY_PYROLYSIS = "spray_pyrolysis"
    SPUTTERING = "sputtering"
    EVAPORATION = "evaporation"
    ELECTRODEPOSITION = "electrodeposition"


@dataclass
class ChemicalComponent:
    """Chemical component for synthesis"""
    name: str
    formula: str
    purity: float = 0.99
    mass_mg: float = 0.0
    molar_mass: float = 0.0
    supplier: str = ""
    cas_number: str = ""
    
    @property
    def moles(self) -> float:
        """Calculate moles from mass"""
        if self.molar_mass > 0:
            return self.mass_mg / 1000 / self.molar_mass
        return 0.0


@dataclass
class ProcessParameters:
    """Generic process parameters"""
    temperature_c: float = 25.0
    temperature_profile: List[Tuple[float, float]] = field(default_factory=list)  # (time, temp)
    pressure_pa: float = 101325.0
    atmosphere: str = "air"
    gas_flow_rate_sccm: Optional[float] = None
    duration_min: float = 60.0
    mixing_speed_rpm: float = 0.0
    ph: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessParameters':
        return cls(**data)


@dataclass
class PowderSynthesisParameters(ProcessParameters):
    """Parameters specific to powder synthesis"""
    calcination_temp_c: float = 800.0
    calcination_time_h: float = 4.0
    heating_rate_c_per_min: float = 5.0
    cooling_rate_c_per_min: float = 2.0
    grinding_time_min: float = 30.0
    sintering_temp_c: Optional[float] = None
    sintering_time_h: Optional[float] = None
    num_regrinds: int = 2


@dataclass
class ThinFilmParameters(ProcessParameters):
    """Parameters specific to thin film deposition"""
    # Substrate parameters
    substrate_material: str = "SiO2"
    substrate_size_mm: Tuple[float, float] = (10.0, 10.0)
    substrate_thickness_mm: float = 0.5
    substrate_preparation: str = "ultrasonication"
    
    # Deposition parameters
    deposition_rate_nm_per_min: float = 10.0
    target_thickness_nm: float = 100.0
    precursor_concentration_m: Optional[float] = None
    solvent: str = ""
    
    # Post-deposition
    annealing_temp_c: Optional[float] = None
    annealing_time_min: Optional[float] = None
    annealing_atmosphere: str = "N2"
    
    # Method-specific
    rotation_speed_rpm: Optional[float] = None  # for spin coating
    withdrawal_speed_mm_per_s: Optional[float] = None  # for dip coating
    deposition_cycles: int = 1  # for ALD


@dataclass
class SynthesisRecipe:
    """Complete synthesis recipe"""
    recipe_id: str
    name: str
    target_material: str
    method: Union[SynthesisMethod, DepositionMethod]
    components: List[ChemicalComponent] = field(default_factory=list)
    parameters: Union[PowderSynthesisParameters, ThinFilmParameters] = field(
        default_factory=ProcessParameters
    )
    equipment_required: List[str] = field(default_factory=list)
    safety_notes: List[str] = field(default_factory=list)
    quality_criteria: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'recipe_id': self.recipe_id,
            'name': self.name,
            'target_material': self.target_material,
            'method': self.method.value if isinstance(self.method, Enum) else self.method,
            'components': [asdict(c) for c in self.components],
            'parameters': asdict(self.parameters),
            'equipment_required': self.equipment_required,
            'safety_notes': self.safety_notes,
            'quality_criteria': self.quality_criteria,
            'created_at': self.created_at.isoformat(),
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SynthesisRecipe':
        # Parse method
        method_str = data.get('method', 'solid_state')
        try:
            method = SynthesisMethod(method_str)
        except ValueError:
            try:
                method = DepositionMethod(method_str)
            except ValueError:
                method = SynthesisMethod.SOLID_STATE
        
        # Parse components
        components = [ChemicalComponent(**c) for c in data.get('components', [])]
        
        # Parse parameters
        params_data = data.get('parameters', {})
        if isinstance(method, SynthesisMethod):
            params = PowderSynthesisParameters(**params_data)
        else:
            params = ThinFilmParameters(**params_data)
        
        return cls(
            recipe_id=data['recipe_id'],
            name=data['name'],
            target_material=data['target_material'],
            method=method,
            components=components,
            parameters=params,
            equipment_required=data.get('equipment_required', []),
            safety_notes=data.get('safety_notes', []),
            quality_criteria=data.get('quality_criteria', {}),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            version=data.get('version', '1.0')
        )
    
    def calculate_stoichiometry(self) -> Dict[str, float]:
        """Calculate stoichiometric ratios"""
        total_moles = sum(c.moles for c in self.components)
        if total_moles == 0:
            return {}
        return {c.formula: c.moles / total_moles for c in self.components}


class SynthesisPlanner(ABC):
    """Abstract base class for synthesis planners"""
    
    def __init__(self, name: str):
        self.name = name
        self.recipes: Dict[str, SynthesisRecipe] = {}
        self.process_history: List[Dict[str, Any]] = []
    
    def add_recipe(self, recipe: SynthesisRecipe) -> bool:
        """Add synthesis recipe"""
        self.recipes[recipe.recipe_id] = recipe
        logger.info(f"Added recipe: {recipe.name} ({recipe.recipe_id})")
        return True
    
    def get_recipe(self, recipe_id: str) -> Optional[SynthesisRecipe]:
        """Get recipe by ID"""
        return self.recipes.get(recipe_id)
    
    def list_recipes(self) -> List[str]:
        """List all recipe IDs"""
        return list(self.recipes.keys())
    
    def save_recipes(self, filepath: str) -> bool:
        """Save recipes to JSON file"""
        try:
            data = {rid: r.to_dict() for rid, r in self.recipes.items()}
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save recipes: {e}")
            return False
    
    def load_recipes(self, filepath: str) -> bool:
        """Load recipes from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            for rid, rdata in data.items():
                self.recipes[rid] = SynthesisRecipe.from_dict(rdata)
            return True
        except Exception as e:
            logger.error(f"Failed to load recipes: {e}")
            return False
    
    @abstractmethod
    async def execute_recipe(self, 
                           recipe_id: str,
                           sample_id: str,
                           equipment_manager: Any) -> Dict[str, Any]:
        """Execute synthesis recipe"""
        pass
    
    @abstractmethod
    def optimize_parameters(self,
                          target_properties: Dict[str, Any],
                          constraints: Optional[Dict[str, Any]] = None) -> ProcessParameters:
        """Optimize synthesis parameters based on target properties"""
        pass


class PowderSynthesizer(SynthesisPlanner):
    """
    Planner for powder synthesis methods
    Supports solid-state, sol-gel, hydrothermal, etc.
    """
    
    def __init__(self):
        super().__init__("Powder Synthesizer")
        self._setup_default_recipes()
    
    def _setup_default_recipes(self):
        """Setup default synthesis recipes"""
        # LiCoO2 solid-state synthesis
        lco_recipe = SynthesisRecipe(
            recipe_id="LCO_SSD_001",
            name="LiCoO2 Solid-State Synthesis",
            target_material="LiCoO2",
            method=SynthesisMethod.SOLID_STATE,
            components=[
                ChemicalComponent(
                    name="Lithium Carbonate",
                    formula="Li2CO3",
                    purity=0.999,
                    mass_mg=739.0,
                    molar_mass=73.89
                ),
                ChemicalComponent(
                    name="Cobalt Oxide",
                    formula="Co3O4",
                    purity=0.999,
                    mass_mg=2408.0,
                    molar_mass=240.8
                )
            ],
            parameters=PowderSynthesisParameters(
                temperature_c=25,
                calcination_temp_c=900,
                calcination_time_h=12,
                heating_rate_c_per_min=5,
                cooling_rate_c_per_min=2,
                grinding_time_min=30,
                num_regrinds=2,
                atmosphere="air"
            ),
            equipment_required=["furnace_1", "mortar_grinder", "balance"],
            safety_notes=[
                "Li2CO3 is an irritant",
                "Co3O4 is toxic and carcinogenic",
                "Use fume hood for weighing"
            ],
            quality_criteria={
                'xrd_phase_purity': '> 95%',
                'particle_size_d50': '1-10 um',
                'tap_density': '> 1.5 g/cm3'
            }
        )
        self.add_recipe(lco_recipe)
        
        # TiO2 sol-gel synthesis
        tio2_recipe = SynthesisRecipe(
            recipe_id="TiO2_SG_001",
            name="TiO2 Sol-Gel Synthesis",
            target_material="TiO2 (Anatase)",
            method=SynthesisMethod.SOL_GEL,
            components=[
                ChemicalComponent(
                    name="Titanium Isopropoxide",
                    formula="Ti(OiPr)4",
                    purity=0.97,
                    mass_mg=2840.0,
                    molar_mass=284.22
                ),
                ChemicalComponent(
                    name="Acetic Acid",
                    formula="CH3COOH",
                    purity=0.999,
                    mass_mg=600.0,
                    molar_mass=60.05
                )
            ],
            parameters=PowderSynthesisParameters(
                temperature_c=80,
                calcination_temp_c=450,
                calcination_time_h=2,
                heating_rate_c_per_min=2,
                cooling_rate_c_per_min=5,
                ph=3.5,
                duration_min=240
            ),
            equipment_required=["hot_plate", "furnace_1", "magnetic_stirrer"],
            safety_notes=[
                "Ti(OiPr)4 is moisture sensitive",
                "Handle in glovebox",
                "Acetic acid is corrosive"
            ]
        )
        self.add_recipe(tio2_recipe)
    
    async def execute_recipe(self,
                           recipe_id: str,
                           sample_id: str,
                           equipment_manager: Any) -> Dict[str, Any]:
        """Execute powder synthesis recipe"""
        recipe = self.get_recipe(recipe_id)
        if not recipe:
            raise ValueError(f"Recipe {recipe_id} not found")
        
        logger.info(f"Executing powder synthesis: {recipe.name}")
        
        # Execution steps
        steps = []
        start_time = datetime.now()
        
        # Step 1: Weigh components
        steps.append({
            'step': 1,
            'action': 'weigh_components',
            'status': 'completed',
            'components': [c.to_dict() for c in recipe.components]
        })
        
        # Step 2: Mixing/Grinding
        if isinstance(recipe.parameters, PowderSynthesisParameters):
            if recipe.method == SynthesisMethod.SOLID_STATE:
                steps.append({
                    'step': 2,
                    'action': 'grind_mixture',
                    'duration_min': recipe.parameters.grinding_time_min,
                    'status': 'completed'
                })
                
                # Multiple calcination cycles
                for i in range(recipe.parameters.num_regrinds + 1):
                    steps.append({
                        'step': 3 + i * 2,
                        'action': f'calcination_cycle_{i+1}',
                        'temperature_c': recipe.parameters.calcination_temp_c,
                        'duration_h': recipe.parameters.calcination_time_h,
                        'status': 'completed'
                    })
                    
                    if i < recipe.parameters.num_regrinds:
                        steps.append({
                            'step': 4 + i * 2,
                            'action': f'regrind_{i+1}',
                            'duration_min': recipe.parameters.grinding_time_min,
                            'status': 'completed'
                        })
        
        result = {
            'recipe_id': recipe_id,
            'sample_id': sample_id,
            'target_material': recipe.target_material,
            'method': recipe.method.value,
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'steps': steps,
            'status': 'completed',
            'yield_g': random.uniform(0.85, 0.95) * sum(c.mass_mg for c in recipe.components) / 1000
        }
        
        self.process_history.append(result)
        return result
    
    def optimize_parameters(self,
                          target_properties: Dict[str, Any],
                          constraints: Optional[Dict[str, Any]] = None) -> PowderSynthesisParameters:
        """
        Optimize powder synthesis parameters using ML/optimization
        """
        constraints = constraints or {}
        
        # Default parameters
        params = PowderSynthesisParameters()
        
        # Optimize based on target particle size
        target_size = target_properties.get('particle_size_nm')
        if target_size:
            if target_size < 50:
                # Nanoparticles: lower temperature, shorter time
                params.calcination_temp_c = min(600, constraints.get('max_temp', 600))
                params.calcination_time_h = 2
                params.heating_rate_c_per_min = 10  # faster heating
            elif target_size < 200:
                # Sub-micron
                params.calcination_temp_c = 800
                params.calcination_time_h = 4
            else:
                # Micron-sized
                params.calcination_temp_c = 1000
                params.calcination_time_h = 12
        
        # Optimize based on phase purity requirement
        if target_properties.get('phase_purity', 0) > 0.99:
            params.num_regrinds = 3
            params.calcination_time_h = max(params.calcination_time_h, 12)
        
        return params
    
    def calculate_yield(self, recipe_id: str, 
                       actual_yield_g: float) -> Dict[str, float]:
        """Calculate synthesis yield and efficiency"""
        recipe = self.get_recipe(recipe_id)
        if not recipe:
            raise ValueError(f"Recipe {recipe_id} not found")
        
        theoretical_mass = sum(c.mass_mg for c in recipe.components) / 1000
        yield_percent = (actual_yield_g / theoretical_mass) * 100
        
        return {
            'theoretical_yield_g': theoretical_mass,
            'actual_yield_g': actual_yield_g,
            'yield_percent': yield_percent,
            'mass_loss_percent': 100 - yield_percent
        }


class ThinFilmDepositor(SynthesisPlanner):
    """
    Planner for thin film deposition methods
    Supports CVD, PVD, ALD, spin coating, etc.
    """
    
    def __init__(self):
        super().__init__("Thin Film Depositor")
        self._setup_default_recipes()
    
    def _setup_default_recipes(self):
        """Setup default thin film recipes"""
        # ZnO spin coating
        zno_spin_recipe = SynthesisRecipe(
            recipe_id="ZnO_SPIN_001",
            name="ZnO Thin Film - Spin Coating",
            target_material="ZnO",
            method=DepositionMethod.SPIN_COATING,
            components=[
                ChemicalComponent(
                    name="Zinc Acetate Dihydrate",
                    formula="Zn(CH3COO)2·2H2O",
                    purity=0.99,
                    mass_mg=2195.0,
                    molar_mass=219.5
                ),
                ChemicalComponent(
                    name="2-Methoxyethanol",
                    formula="C3H8O2",
                    purity=0.999,
                    mass_mg=50000.0,
                    molar_mass=76.09
                )
            ],
            parameters=ThinFilmParameters(
                substrate_material="Si/SiO2",
                substrate_size_mm=(20, 20),
                precursor_concentration_m=0.3,
                solvent="2-methoxyethanol",
                rotation_speed_rpm=3000,
                deposition_rate_nm_per_min=50,
                target_thickness_nm=100,
                annealing_temp_c=400,
                annealing_time_min=60,
                annealing_atmosphere="air"
            ),
            equipment_required=["spin_coater", "hot_plate", "furnace_1"],
            quality_criteria={
                'thickness_uniformity': '< 5%',
                'surface_roughness_nm': '< 2',
                'crystallinity': 'polycrystalline'
            }
        )
        self.add_recipe(zno_spin_recipe)
        
        # Al2O3 ALD recipe
        al2o3_ald_recipe = SynthesisRecipe(
            recipe_id="Al2O3_ALD_001",
            name="Al2O3 Thin Film - ALD",
            target_material="Al2O3",
            method=DepositionMethod.ALD,
            components=[
                ChemicalComponent(
                    name="Trimethylaluminum",
                    formula="TMA",
                    purity=0.9999,
                    mass_mg=0,  # vapor phase
                    molar_mass=72.06
                ),
                ChemicalComponent(
                    name="Water",
                    formula="H2O",
                    purity=0.99999,
                    mass_mg=0,
                    molar_mass=18.02
                )
            ],
            parameters=ThinFilmParameters(
                substrate_material="Si",
                substrate_size_mm=(100, 100),
                temperature_c=200,
                deposition_cycles=100,
                target_thickness_nm=10,  # ~0.1 nm per cycle
                atmosphere="N2",
                gas_flow_rate_sccm=50
            ),
            equipment_required=["ald_system"],
            quality_criteria={
                'thickness_uniformity': '< 2%',
                'conformality': '> 95%',
                'impurity_content': '< 1%'
            }
        )
        self.add_recipe(al2o3_ald_recipe)
    
    async def execute_recipe(self,
                           recipe_id: str,
                           sample_id: str,
                           equipment_manager: Any) -> Dict[str, Any]:
        """Execute thin film deposition recipe"""
        recipe = self.get_recipe(recipe_id)
        if not recipe:
            raise ValueError(f"Recipe {recipe_id} not found")
        
        logger.info(f"Executing thin film deposition: {recipe.name}")
        
        steps = []
        start_time = datetime.now()
        params = recipe.parameters
        
        if not isinstance(params, ThinFilmParameters):
            raise ValueError("Invalid parameters for thin film deposition")
        
        # Step 1: Substrate preparation
        steps.append({
            'step': 1,
            'action': 'substrate_preparation',
            'substrate': params.substrate_material,
            'preparation': params.substrate_preparation,
            'status': 'completed'
        })
        
        # Step 2: Deposition
        if recipe.method == DepositionMethod.SPIN_COATING:
            steps.append({
                'step': 2,
                'action': 'spin_coating',
                'rotation_speed_rpm': params.rotation_speed_rpm,
                'duration_s': 30,
                'status': 'completed'
            })
        elif recipe.method == DepositionMethod.ALD:
            steps.append({
                'step': 2,
                'action': 'ald_deposition',
                'cycles': params.deposition_cycles,
                'temperature_c': params.temperature_c,
                'status': 'completed'
            })
        elif recipe.method == DepositionMethod.CVD:
            steps.append({
                'step': 2,
                'action': 'cvd_deposition',
                'duration_min': params.duration_min,
                'temperature_c': params.temperature_c,
                'status': 'completed'
            })
        
        # Step 3: Post-deposition annealing
        if params.annealing_temp_c:
            steps.append({
                'step': 3,
                'action': 'annealing',
                'temperature_c': params.annealing_temp_c,
                'duration_min': params.annealing_time_min,
                'atmosphere': params.annealing_atmosphere,
                'status': 'completed'
            })
        
        result = {
            'recipe_id': recipe_id,
            'sample_id': sample_id,
            'target_material': recipe.target_material,
            'deposition_method': recipe.method.value,
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'steps': steps,
            'status': 'completed',
            'achieved_thickness_nm': params.target_thickness_nm * random.uniform(0.95, 1.05)
        }
        
        self.process_history.append(result)
        return result
    
    def optimize_parameters(self,
                          target_properties: Dict[str, Any],
                          constraints: Optional[Dict[str, Any]] = None) -> ThinFilmParameters:
        """
        Optimize thin film deposition parameters
        """
        constraints = constraints or {}
        params = ThinFilmParameters()
        
        # Optimize based on target thickness
        target_thickness = target_properties.get('thickness_nm')
        if target_thickness:
            params.target_thickness_nm = target_thickness
            
            # Method selection based on thickness
            if target_thickness < 20:
                # Ultra-thin: use ALD for precision
                params.deposition_cycles = int(target_thickness / 0.1)
                params.temperature_c = 200
            elif target_thickness < 200:
                # Thin films: spin coating or sputtering
                params.rotation_speed_rpm = 3000
                params.deposition_rate_nm_per_min = 50
            else:
                # Thick films: CVD
                params.duration_min = target_thickness / 10  # 10 nm/min
        
        # Optimize based on uniformity requirement
        if target_properties.get('uniformity_percent', 100) > 95:
            params.substrate_preparation = "piranha_cleaning"
            params.temperature_c = min(params.temperature_c, 150)  # better uniformity at lower T
        
        # Optimize based on crystallinity
        if target_properties.get('crystallinity') == 'single_crystal':
            params.annealing_temp_c = 800
            params.annealing_time_min = 120
        elif target_properties.get('crystallinity') == 'polycrystalline':
            params.annealing_temp_c = 400
            params.annealing_time_min = 60
        
        return params
    
    def calculate_deposition_time(self, recipe_id: str) -> float:
        """Calculate total deposition time in minutes"""
        recipe = self.get_recipe(recipe_id)
        if not recipe:
            raise ValueError(f"Recipe {recipe_id} not found")
        
        params = recipe.parameters
        if isinstance(params, ThinFilmParameters):
            deposition_time = params.target_thickness_nm / params.deposition_rate_nm_per_min
            
            # Add annealing time if applicable
            if params.annealing_time_min:
                deposition_time += params.annealing_time_min
            
            return deposition_time
        
        return 0.0


class SynthesisWorkflow:
    """
    High-level workflow manager for synthesis operations
    """
    
    def __init__(self):
        self.powder_synthesizer = PowderSynthesizer()
        self.thin_film_depositor = ThinFilmDepositor()
        self.active_processes: Dict[str, Dict[str, Any]] = {}
    
    def get_synthesizer(self, 
                       material_type: str) -> SynthesisPlanner:
        """Get appropriate synthesizer for material type"""
        if material_type in ['powder', 'ceramic', 'bulk']:
            return self.powder_synthesizer
        elif material_type in ['thin_film', 'coating', 'membrane']:
            return self.thin_film_depositor
        else:
            raise ValueError(f"Unknown material type: {material_type}")
    
    async def synthesize_material(self,
                                 target_material: str,
                                 material_type: str,
                                 sample_id: str,
                                 equipment_manager: Any,
                                 custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute complete synthesis workflow
        """
        synthesizer = self.get_synthesizer(material_type)
        
        # Find or create recipe
        recipe = None
        for r in synthesizer.recipes.values():
            if r.target_material == target_material:
                recipe = r
                break
        
        if not recipe:
            logger.warning(f"No recipe found for {target_material}, using default")
            # Use first available recipe as template
            if synthesizer.recipes:
                template = list(synthesizer.recipes.values())[0]
                recipe = SynthesisRecipe(
                    recipe_id=f"AUTO_{target_material}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    name=f"Auto-generated {target_material} recipe",
                    target_material=target_material,
                    method=template.method,
                    components=template.components,
                    parameters=template.parameters
                )
        
        if not recipe:
            raise ValueError(f"Cannot create recipe for {target_material}")
        
        # Execute
        result = await synthesizer.execute_recipe(
            recipe.recipe_id,
            sample_id,
            equipment_manager
        )
        
        self.active_processes[sample_id] = {
            'sample_id': sample_id,
            'target_material': target_material,
            'recipe_id': recipe.recipe_id,
            'result': result,
            'status': 'completed'
        }
        
        return result
    
    def get_process_status(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Get status of synthesis process"""
        return self.active_processes.get(sample_id)
    
    def generate_report(self, sample_id: str) -> str:
        """Generate synthesis report"""
        process = self.active_processes.get(sample_id)
        if not process:
            return f"No process found for sample {sample_id}"
        
        lines = [
            f"Synthesis Report for Sample: {sample_id}",
            "=" * 50,
            f"Target Material: {process['target_material']}",
            f"Recipe ID: {process['recipe_id']}",
            f"Status: {process['status']}",
            "",
            "Process Steps:",
        ]
        
        for step in process['result'].get('steps', []):
            lines.append(f"  Step {step['step']}: {step['action']} - {step['status']}")
        
        lines.extend([
            "",
            f"Start Time: {process['result'].get('start_time')}",
            f"End Time: {process['result'].get('end_time')}",
        ])
        
        return "\n".join(lines)
