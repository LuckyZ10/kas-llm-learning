"""
dftlammps.dft_advanced
======================
Advanced DFT calculation modules for VASP/QE/CP2K

Modules:
- optical_properties: Dielectric functions, absorption, excitons, ellipsometry
- magnetic_properties: Spin-polarized DFT, MAE, exchange coupling, Curie T
- defect_calculations: Formation energies, transition levels, NEB diffusion
- nonlinear_response: Elastic, piezoelectric, and SHG coefficients
"""

from .optical_properties import (
    OpticalPropertyWorkflow,
    VASPOpticalCalculator,
    QEOpticalCalculator,
    DielectricFunction,
    OpticalSpectrum,
    ExcitonPeak,
    EllipsometryParams,
    VASPOpticalConfig,
    QEOpticalConfig,
)

from .magnetic_properties import (
    MagneticPropertyWorkflow,
    VASPMagneticCalculator,
    QEMagneticCalculator,
    MagneticState,
    MagneticAnisotropy,
    ExchangeCoupling,
    CurieTemperature,
    SpinConfiguration,
    VASPMagneticConfig,
    QEMagneticConfig,
    SpinConfigurationGenerator,
)

from .defect_calculations import (
    DefectCalculationWorkflow,
    DefectStructureGenerator,
    FormationEnergyCalculator,
    FiniteSizeCorrectionCalculator,
    NEBDiffusionCalculator,
    DefectSpec,
    FormationEnergy,
    TransitionLevel,
    DefectConfig,
    DefectType,
)

from .nonlinear_response import (
    NonlinearResponseWorkflow,
    ElasticConstantsCalculator,
    PiezoelectricCalculator,
    SHGCalculator,
    ElasticTensor,
    PiezoelectricTensor,
    SHGTensor,
)

__all__ = [
    # Optical
    'OpticalPropertyWorkflow',
    'VASPOpticalCalculator',
    'QEOpticalCalculator',
    'DielectricFunction',
    'OpticalSpectrum',
    'ExcitonPeak',
    'EllipsometryParams',
    'VASPOpticalConfig',
    'QEOpticalConfig',
    
    # Magnetic
    'MagneticPropertyWorkflow',
    'VASPMagneticCalculator',
    'QEMagneticCalculator',
    'MagneticState',
    'MagneticAnisotropy',
    'ExchangeCoupling',
    'CurieTemperature',
    'SpinConfiguration',
    'VASPMagneticConfig',
    'QEMagneticConfig',
    'SpinConfigurationGenerator',
    
    # Defect
    'DefectCalculationWorkflow',
    'DefectStructureGenerator',
    'FormationEnergyCalculator',
    'FiniteSizeCorrectionCalculator',
    'NEBDiffusionCalculator',
    'DefectSpec',
    'FormationEnergy',
    'TransitionLevel',
    'DefectConfig',
    'DefectType',
    
    # Nonlinear
    'NonlinearResponseWorkflow',
    'ElasticConstantsCalculator',
    'PiezoelectricCalculator',
    'SHGCalculator',
    'ElasticTensor',
    'PiezoelectricTensor',
    'SHGTensor',
]
