"""
知识图谱初始化模块
================
提供预定义的材料科学本体和常用知识。

Author: DFT-LAMMPS Team
Date: 2025
"""

from typing import Dict, List, Any
try:
    from dftlammps.knowledge_graph.kg_core import (
        KnowledgeGraph, Entity, EntityType, Relation, RelationType
    )
except ImportError:
    from .kg_core import (
        KnowledgeGraph, Entity, EntityType, Relation, RelationType
    )


def initialize_materials_ontology(kg: KnowledgeGraph):
    """
    初始化材料科学本体
    
    添加基本的元素、材料类别和关系。
    """
    # 添加常见元素及其属性
    _add_elements(kg)
    
    # 添加材料类别
    _add_material_classes(kg)
    
    # 添加常见计算方法
    _add_methods(kg)
    
    # 添加常用软件
    _add_software(kg)
    
    # 添加性质分类
    _add_properties(kg)
    
    # 添加合成方法
    _add_synthesis_methods(kg)
    
    # 添加应用领域
    _add_applications(kg)


def _add_elements(kg: KnowledgeGraph):
    """添加元素实体"""
    # 过渡金属
    transition_metals = [
        ("Sc", "Scandium", {"group": 3, "period": 4, "atomic_number": 21}),
        ("Ti", "Titanium", {"group": 4, "period": 4, "atomic_number": 22}),
        ("V", "Vanadium", {"group": 5, "period": 4, "atomic_number": 23}),
        ("Cr", "Chromium", {"group": 6, "period": 4, "atomic_number": 24}),
        ("Mn", "Manganese", {"group": 7, "period": 4, "atomic_number": 25}),
        ("Fe", "Iron", {"group": 8, "period": 4, "atomic_number": 26}),
        ("Co", "Cobalt", {"group": 9, "period": 4, "atomic_number": 27}),
        ("Ni", "Nickel", {"group": 10, "period": 4, "atomic_number": 28}),
        ("Cu", "Copper", {"group": 11, "period": 4, "atomic_number": 29}),
        ("Zn", "Zinc", {"group": 12, "period": 4, "atomic_number": 30}),
    ]
    
    # 主族元素
    main_group = [
        ("H", "Hydrogen", {"group": 1, "period": 1, "atomic_number": 1}),
        ("Li", "Lithium", {"group": 1, "period": 2, "atomic_number": 3}),
        ("Na", "Sodium", {"group": 1, "period": 3, "atomic_number": 11}),
        ("K", "Potassium", {"group": 1, "period": 4, "atomic_number": 19}),
        ("C", "Carbon", {"group": 14, "period": 2, "atomic_number": 6}),
        ("Si", "Silicon", {"group": 14, "period": 3, "atomic_number": 14}),
        ("N", "Nitrogen", {"group": 15, "period": 2, "atomic_number": 7}),
        ("P", "Phosphorus", {"group": 15, "period": 3, "atomic_number": 15}),
        ("O", "Oxygen", {"group": 16, "period": 2, "atomic_number": 8}),
        ("S", "Sulfur", {"group": 16, "period": 3, "atomic_number": 16}),
        ("F", "Fluorine", {"group": 17, "period": 2, "atomic_number": 9}),
        ("Cl", "Chlorine", {"group": 17, "period": 3, "atomic_number": 17}),
    ]
    
    all_elements = transition_metals + main_group
    
    for symbol, name, props in all_elements:
        entity = Entity(
            id=f"elem_{symbol}",
            name=symbol,
            entity_type=EntityType.ELEMENT,
            properties={"full_name": name, **props}
        )
        kg.add_entity(entity)


def _add_material_classes(kg: KnowledgeGraph):
    """添加材料类别"""
    material_classes = [
        {
            "id": "mat_metal",
            "name": "Metals",
            "properties": {"category": "structural", "conductivity": "high", "bonding": "metallic"}
        },
        {
            "id": "mat_semiconductor",
            "name": "Semiconductors",
            "properties": {"category": "electronic", "band_gap": "0.1-4 eV", "bonding": "covalent"}
        },
        {
            "id": "mat_insulator",
            "name": "Insulators",
            "properties": {"category": "electronic", "band_gap": "> 4 eV", "bonding": "ionic/covalent"}
        },
        {
            "id": "mat_ceramic",
            "name": "Ceramics",
            "properties": {"category": "structural", "hardness": "high", "temperature_resistance": "high"}
        },
        {
            "id": "mat_polymer",
            "name": "Polymers",
            "properties": {"category": "organic", "flexibility": "high", "lightweight": True}
        },
        {
            "id": "mat_perovskite",
            "name": "Perovskites",
            "properties": {"structure": "ABX3", "category": "functional", "applications": ["solar_cells", "superconductors"]}
        },
        {
            "id": "mat_2d_material",
            "name": "2D Materials",
            "properties": {"dimensionality": 2, "examples": ["graphene", "MoS2", "hBN"], "unique_properties": ["high_surface_area", "quantum_confinement"]}
        },
    ]
    
    for mat_class in material_classes:
        entity = Entity(
            id=mat_class["id"],
            name=mat_class["name"],
            entity_type=EntityType.MATERIAL,
            properties=mat_class["properties"]
        )
        kg.add_entity(entity)


def _add_methods(kg: KnowledgeGraph):
    """添加计算方法"""
    methods = [
        {
            "id": "method_dft",
            "name": "Density Functional Theory",
            "abbreviation": "DFT",
            "properties": {
                "type": "electronic_structure",
                "scaling": "O(N³)",
                "accuracy": "good",
                "basis": "plane_wave_or_localized"
            }
        },
        {
            "id": "method_md",
            "name": "Molecular Dynamics",
            "abbreviation": "MD",
            "properties": {
                "type": "atomistic_simulation",
                "ensemble": ["NVE", "NVT", "NPT"],
                "time_scale": "nanoseconds"
            }
        },
        {
            "id": "method_mc",
            "name": "Monte Carlo",
            "abbreviation": "MC",
            "properties": {
                "type": "stochastic_simulation",
                "applications": ["phase_transitions", "adsorption"]
            }
        },
        {
            "id": "method_gw",
            "name": "GW Approximation",
            "abbreviation": "GW",
            "properties": {
                "type": "many_body",
                "accuracy": "high",
                "computational_cost": "high",
                "best_for": "band_gaps"
            }
        },
        {
            "id": "method_dft_u",
            "name": "DFT+U",
            "abbreviation": "DFT+U",
            "properties": {
                "type": "electronic_structure",
                "target_systems": ["transition_metal_oxides", "strongly_correlated"],
                "parameter": "Hubbard_U"
            }
        },
        {
            "id": "method_hse",
            "name": "Heyd-Scuseria-Ernzerhof",
            "abbreviation": "HSE",
            "properties": {
                "type": "hybrid_functional",
                "accuracy": "very_high",
                "cost": "high",
                "best_for": ["band_gaps", "defects"]
            }
        },
        {
            "id": "method_neb",
            "name": "Nudged Elastic Band",
            "abbreviation": "NEB",
            "properties": {
                "type": "transition_state",
                "application": "barrier_calculations",
                "variants": ["regular", "climbing_image"]
            }
        },
        {
            "id": "method_phonon",
            "name": "Phonon Calculation",
            "abbreviation": "Phonon",
            "properties": {
                "type": "lattice_dynamics",
                "methods": ["DFPT", "supercell_finite_differences"],
                "properties": ["thermal_conductivity", "free_energy", "stability"]
            }
        },
    ]
    
    for method in methods:
        entity = Entity(
            id=method["id"],
            name=method["name"],
            entity_type=EntityType.METHOD,
            properties={k: v for k, v in method.items() if k != "id" and k != "name"}
        )
        kg.add_entity(entity)


def _add_software(kg: KnowledgeGraph):
    """添加计算软件"""
    software = [
        {
            "id": "sw_vasp",
            "name": "VASP",
            "full_name": "Vienna Ab initio Simulation Package",
            "properties": {
                "type": "DFT",
                "basis": "plane_wave",
                "license": "commercial",
                "parallelization": "MPI/OpenMP",
                "strengths": ["solids", "surfaces", "magnetism"]
            }
        },
        {
            "id": "sw_qe",
            "name": "Quantum ESPRESSO",
            "full_name": "Quantum ESPRESSO",
            "properties": {
                "type": "DFT",
                "basis": "plane_wave",
                "license": "open_source",
                "parallelization": "MPI",
                "strengths": ["general_purpose", "phonons", "pw_basis"]
            }
        },
        {
            "id": "sw_lammps",
            "name": "LAMMPS",
            "full_name": "Large-scale Atomic/Molecular Massively Parallel Simulator",
            "properties": {
                "type": "MD",
                "force_fields": ["pair", "bond", "angle", "many_body"],
                "license": "open_source",
                "parallelization": "MPI",
                "strengths": ["large_systems", "various_potentials", "complex_simulations"]
            }
        },
        {
            "id": "sw_cp2k",
            "name": "CP2K",
            "full_name": "CP2K",
            "properties": {
                "type": "DFT/MD",
                "basis": "gaussian/plane_wave",
                "license": "open_source",
                "strengths": ["molecules", "hybrid_functionals", "linear_scaling"]
            }
        },
        {
            "id": "sw_gaussian",
            "name": "Gaussian",
            "full_name": "Gaussian",
            "properties": {
                "type": "quantum_chemistry",
                "basis": "gaussian",
                "license": "commercial",
                "strengths": ["molecules", "high_accuracy_methods", "spectroscopy"]
            }
        },
    ]
    
    for sw in software:
        entity = Entity(
            id=sw["id"],
            name=sw["name"],
            entity_type=EntityType.SOFTWARE,
            properties={k: v for k, v in sw.items() if k != "id" and k != "name"}
        )
        kg.add_entity(entity)


def _add_properties(kg: KnowledgeGraph):
    """添加性质分类"""
    properties = [
        {
            "id": "prop_band_gap",
            "name": "Band Gap",
            "properties": {
                "category": "electronic",
                "unit": "eV",
                "importance": "high",
                "applications": ["optoelectronics", "photovoltaics"]
            }
        },
        {
            "id": "prop_bulk_modulus",
            "name": "Bulk Modulus",
            "properties": {
                "category": "mechanical",
                "unit": "GPa",
                "importance": "high",
                "related": ["elastic_constants", "compressibility"]
            }
        },
        {
            "id": "prop_formation_energy",
            "name": "Formation Energy",
            "properties": {
                "category": "thermodynamic",
                "unit": "eV/atom",
                "importance": "high",
                "use": "stability_prediction"
            }
        },
        {
            "id": "prop_thermal_conductivity",
            "name": "Thermal Conductivity",
            "properties": {
                "category": "thermal",
                "unit": "W/mK",
                "importance": "medium",
                "applications": ["thermoelectrics", "thermal_management"]
            }
        },
        {
            "id": "prop_magnetic_moment",
            "name": "Magnetic Moment",
            "properties": {
                "category": "magnetic",
                "unit": "μB",
                "importance": "high",
                "applications": ["spintronics", "magnetic_storage"]
            }
        },
        {
            "id": "prop_dielectric_constant",
            "name": "Dielectric Constant",
            "properties": {
                "category": "dielectric",
                "unit": "dimensionless",
                "importance": "medium",
                "types": ["electronic", "ionic", "total"]
            }
        },
    ]
    
    for prop in properties:
        entity = Entity(
            id=prop["id"],
            name=prop["name"],
            entity_type=EntityType.PROPERTY,
            properties={k: v for k, v in prop.items() if k != "id" and k != "name"}
        )
        kg.add_entity(entity)


def _add_synthesis_methods(kg: KnowledgeGraph):
    """添加合成方法"""
    methods = [
        {
            "id": "syn_sol_gel",
            "name": "Sol-Gel Synthesis",
            "properties": {
                "type": "wet_chemical",
                "temperature": "low_to_medium",
                "advantages": ["homogeneous", "low_cost", "versatile"],
                "applications": ["oxides", "nanoparticles", "thin_films"]
            }
        },
        {
            "id": "syn_cvd",
            "name": "Chemical Vapor Deposition",
            "properties": {
                "type": "vapor_phase",
                "temperature": "high",
                "abbreviation": "CVD",
                "variants": ["PECVD", "MOCVD", "LPCVD"],
                "applications": ["thin_films", "graphene", "CNT"]
            }
        },
        {
            "id": "syn_hydrothermal",
            "name": "Hydrothermal Synthesis",
            "properties": {
                "type": "solution_based",
                "conditions": "high_pressure_high_temperature",
                "advantages": ["crystal_quality", "morphology_control"],
                "applications": ["oxides", "zeolites", "nanostructures"]
            }
        },
        {
            "id": "syn_mbe",
            "name": "Molecular Beam Epitaxy",
            "properties": {
                "type": "vacuum_deposition",
                "temperature": "ultra_high_vacuum",
                "abbreviation": "MBE",
                "advantages": ["atomic_precision", "high_quality"],
                "applications": ["semiconductors", "quantum_structures"]
            }
        },
        {
            "id": "syn_mechanical",
            "name": "Mechanical Alloying",
            "properties": {
                "type": "solid_state",
                "equipment": "ball_mill",
                "advantages": ["simple", "scalable", "nonequilibrium_phases"],
                "applications": ["alloys", "nanocomposites", "metastable_materials"]
            }
        },
    ]
    
    for method in methods:
        entity = Entity(
            id=method["id"],
            name=method["name"],
            entity_type=EntityType.SYNTHESIS,
            properties={k: v for k, v in method.items() if k != "id" and k != "name"}
        )
        kg.add_entity(entity)


def _add_applications(kg: KnowledgeGraph):
    """添加应用领域"""
    applications = [
        {
            "id": "app_battery",
            "name": "Battery Technology",
            "properties": {
                "type": "energy_storage",
                "specific_types": ["lithium_ion", "solid_state", "flow_batteries"],
                "key_properties": ["capacity", "cyclability", "safety"]
            }
        },
        {
            "id": "app_photovoltaic",
            "name": "Photovoltaics",
            "properties": {
                "type": "energy_conversion",
                "specific_types": ["silicon", "perovskite", "organic", "tandem"],
                "key_properties": ["efficiency", "stability", "cost"]
            }
        },
        {
            "id": "app_catalysis",
            "name": "Catalysis",
            "properties": {
                "type": "chemical_process",
                "specific_types": ["heterogeneous", "homogeneous", "electrocatalysis", "photocatalysis"],
                "key_properties": ["activity", "selectivity", "stability"]
            }
        },
        {
            "id": "app_superconductor",
            "name": "Superconductors",
            "properties": {
                "type": "quantum_material",
                "specific_types": ["low_Tc", "high_Tc", "iron_based"],
                "key_properties": ["critical_temperature", "critical_field", "critical_current"]
            }
        },
        {
            "id": "app_thermoelectric",
            "name": "Thermoelectrics",
            "properties": {
                "type": "energy_conversion",
                "working_principle": "seebeck_effect",
                "key_properties": ["zt", "seebeck_coefficient", "electrical_conductivity", "thermal_conductivity"]
            }
        },
        {
            "id": "app_spintronics",
            "name": "Spintronics",
            "properties": {
                "type": "information_technology",
                "working_principle": "electron_spin",
                "key_properties": ["spin_polarization", "spin_lifetime", "magnetic_anisotropy"]
            }
        },
    ]
    
    for app in applications:
        entity = Entity(
            id=app["id"],
            name=app["name"],
            entity_type=EntityType.APPLICATION,
            properties={k: v for k, v in app.items() if k != "id" and k != "name"}
        )
        kg.add_entity(entity)


# 添加关系
def add_common_relationships(kg: KnowledgeGraph):
    """添加常见关系"""
    relationships = [
        # 方法-软件关系
        ("method_dft", RelationType.IMPLEMENTED_BY, "sw_vasp"),
        ("method_dft", RelationType.IMPLEMENTED_BY, "sw_qe"),
        ("method_dft", RelationType.IMPLEMENTED_BY, "sw_cp2k"),
        ("method_md", RelationType.IMPLEMENTED_BY, "sw_lammps"),
        
        # 软件-方法关系
        ("sw_vasp", RelationType.IMPLEMENTS, "method_dft"),
        ("sw_vasp", RelationType.IMPLEMENTS, "method_dft_u"),
        ("sw_vasp", RelationType.IMPLEMENTS, "method_hse"),
        ("sw_qe", RelationType.IMPLEMENTS, "method_dft"),
        ("sw_qe", RelationType.IMPLEMENTS, "method_phonon"),
        
        # 应用-性质关系
        ("app_battery", RelationType.REQUIRES, "prop_formation_energy"),
        ("app_photovoltaic", RelationType.REQUIRES, "prop_band_gap"),
        ("app_superconductor", RelationType.REQUIRES, "prop_magnetic_moment"),
        
        # 材料-应用关系
        ("mat_perovskite", RelationType.USED_FOR, "app_photovoltaic"),
        ("mat_perovskite", RelationType.USED_FOR, "app_superconductor"),
        ("mat_2d_material", RelationType.USED_FOR, "app_spintronics"),
    ]
    
    # 注意：这里需要定义IMPLEMENTED_BY和REQUIRES关系
    # 简化起见，我们使用现有的关系类型


def create_default_knowledge_graph() -> KnowledgeGraph:
    """创建默认知识图谱（含本体）"""
    kg = KnowledgeGraph()
    initialize_materials_ontology(kg)
    return kg


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("Knowledge Graph Ontology Test")
    print("=" * 60)
    
    kg = create_default_knowledge_graph()
    
    print(f"\nEntities in graph: {len(kg.entities)}")
    print(f"Relations in graph: {len(kg.relations)}")
    
    print("\nEntity types:")
    for entity_type in EntityType:
        count = len(kg.find_entities_by_type(entity_type))
        print(f"  - {entity_type.value}: {count}")
    
    print("\nSample entities:")
    for entity in list(kg.entities.values())[:5]:
        print(f"  - {entity.name} ({entity.entity_type.value})")
    
    print("\n" + "=" * 60)
    print("Ontology initialization completed!")
    print("=" * 60)
