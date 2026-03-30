"""
DFTLammps Many-Body Module
============================
Advanced many-body perturbation theory calculations including:
- GW quasiparticle corrections (Yambo, BerkeleyGW)
- Bethe-Salpeter Equation (BSE) for excitons
- Exciton properties analysis

Submodules:
- yambo_interface: Interface to Yambo GW-BSE code
- berkeleygw_interface: Interface to BerkeleyGW
- exciton_properties: Comprehensive exciton analysis tools

Example usage:
    from dftlammps.many_body import YamboGWBSE, GWParameters
    from dftlammps.many_body import ExcitonBindingEnergy, get_material
    
    # GW calculation
    yambo = YamboGWBSE(work_dir='./gw_calc')
    gw_input = yambo.run_gw(GWParameters(gw_approximation='G0W0'))
    
    # BSE calculation
    bse_input = yambo.run_bse(BSEParameters(n_excitons=10))
    
    # Exciton analysis
    mat = get_material('MoS2')
    analyzer = ExcitonBindingEnergy(mat)
    binding_energy = analyzer.hydrogenic_binding_energy(n=1)
"""

from .yambo_interface import (
    YamboGWBSE,
    SelfConsistentGW,
    BSEWithGW,
    GWParameters,
    BSEParameters,
    QPEigenvalue,
    ExcitonState,
)

from .berkeleygw_interface import (
    BerkeleyGW,
    BGWBandStructureCalculator,
    BGW2DMaterials,
    BGWEpsilonParameters,
    BGWSigmaParameters,
    BGWBSEParameters,
    BGWBandStructure,
)

from .exciton_properties import (
    ExcitonBindingEnergy,
    ExcitonPhononInteraction,
    ExcitonDynamics,
    ExcitonVisualizer,
    MaterialParameters,
    ExcitonState as ExcitonPropertiesState,
    ExcitonPhononCoupling,
    get_material,
    MATERIAL_DATABASE,
)

__all__ = [
    # Yambo interface
    'YamboGWBSE',
    'SelfConsistentGW',
    'BSEWithGW',
    'GWParameters',
    'BSEParameters',
    'QPEigenvalue',
    'ExcitonState',
    
    # BerkeleyGW interface
    'BerkeleyGW',
    'BGWBandStructureCalculator',
    'BGW2DMaterials',
    'BGWEpsilonParameters',
    'BGWSigmaParameters',
    'BGWBSEParameters',
    'BGWBandStructure',
    
    # Exciton properties
    'ExcitonBindingEnergy',
    'ExcitonPhononInteraction',
    'ExcitonDynamics',
    'ExcitonVisualizer',
    'MaterialParameters',
    'ExcitonPropertiesState',
    'ExcitonPhononCoupling',
    'get_material',
    'MATERIAL_DATABASE',
]

__version__ = '1.0.0'
