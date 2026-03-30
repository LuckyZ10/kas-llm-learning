"""
Mott Insulator Analysis: NiO and CoO Case Studies

This module provides detailed workflows for studying classic Mott insulators:
- NiO: 3d⁸ system with charge-transfer character
- CoO: 3d⁷ system with orbital ordering

Author: DFT-LAMMPS Team
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MottInsulatorConfig:
    """Configuration for Mott insulator calculations"""
    material: str = "NiO"
    
    # Crystal structure (rocksalt)
    lattice_constant: float = 4.17  # Å for NiO
    
    # Electronic configuration
    n_d_electrons: int = 8  # 8 for Ni²⁺, 7 for Co²⁺
    oxidation_state: int = 2
    
    # Magnetic ordering
    magnetic_structure: str = "AFII"  # Type-II antiferromagnetic
    
    # DFT+U parameters
    U_metal: float = 6.3  # eV for Ni
    J_metal: float = 1.0  # eV
    
    # Hybridization
    Delta_ct: float = 4.0  # Charge transfer energy
    pd_sigma: float = 2.0  # p-d hybridization


class NiOAnalyzer:
    """
    Detailed analyzer for NiO (classic charge-transfer insulator)
    
    Key features:
    - Type-II antiferromagnetic ordering
    - Charge-transfer gap (not Mott-Hubbard)
    - Strong superexchange interactions
    - Excitonic effects
    """
    
    def __init__(self, config: MottInsulatorConfig = None):
        self.config = config or MottInsulatorConfig("NiO")
        self._setup_crystal_structure()
        self._setup_electronic_structure()
    
    def _setup_crystal_structure(self):
        """Setup NiO crystal structure (rocksalt with AFM ordering)"""
        a = self.config.lattice_constant
        
        # Rocksalt structure: Ni at (0,0,0), O at (0.5, 0.5, 0.5)
        self.lattice = np.array([
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a]
        ])
        
        # Type-II AFM: ferromagnetic (111) planes, AFM stacking
        # Ni positions with spin
        self.ni_positions = {
            'up': np.array([
                [0, 0, 0],
                [0, 0.5, 0.5],
                [0.5, 0, 0.5],
                [0.5, 0.5, 0]
            ]),
            'down': np.array([
                [0.5, 0.5, 0.5],
                [0.5, 0, 0],
                [0, 0.5, 0],
                [0, 0, 0.5]
            ])
        }
        
        self.o_positions = np.array([
            [0.5, 0.5, 0.5],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0, 0.5]
        ])
    
    def _setup_electronic_structure(self):
        """Setup Ni²⁺ 3d⁸ electronic configuration"""
        # Ni²⁺: 3d⁸ configuration
        # Crystal field splits into t2g⁶ eg² (octahedral)
        
        self.d_config = {
            't2g': 6,  # Fully occupied
            'eg': 2,   # 2 electrons (S=1)
        }
        
        # Spin moment (Hund's rule: S=1 for d⁸)
        self.spin_moment = 1.0  # μB per Ni
        
        # Orbital moment (quenched in octahedral field)
        self.orbital_moment = 0.0
    
    def calculate_electronic_structure(self) -> Dict:
        """
        Calculate electronic structure parameters
        
        NiO has a charge-transfer gap:
        Δ = E(d⁹L) - E(d⁸) where L is ligand hole
        """
        U = self.config.U_metal
        Delta = self.config.Delta_ct
        
        # Classification: Zaanen-Sawatzky-Allen
        if U < Delta:
            gap_type = "Mott-Hubbard"
            gap_size = U
        elif U > Delta:
            gap_type = "charge-transfer"
            gap_size = Delta
        else:
            gap_type = "intermediate"
            gap_size = (U + Delta) / 2
        
        # NiO is charge-transfer (Δ < U)
        results = {
            'gap_type': gap_type,
            'gap_size': gap_size,
            'U': U,
            'Delta_ct': Delta,
            'classification': 'charge_transfer_insulator',
            'valence_band_character': 'O_2p',
            'conduction_band_character': 'Ni_3d'
        }
        
        return results
    
    def calculate_superexchange(self) -> Dict:
        """
        Calculate superexchange interactions
        
        For NiO (180° Ni-O-Ni):
        J_super = 4t_pd⁴ / (Δ² U_eff)
        """
        t_pd = self.config.pd_sigma
        Delta = self.config.Delta_ct
        U_eff = self.config.U_metal - self.config.J_metal
        
        # Anderson superexchange
        J_180 = 4 * t_pd**4 / (Delta**2 * U_eff)
        
        # Neel temperature estimate (mean field)
        # T_N = (2/3) z S(S+1) J / kB
        z = 6  # coordination number
        S = 1
        kB = 8.617e-5  # eV/K
        
        T_N = (2.0/3.0) * z * S * (S + 1) * J_180 / kB
        
        return {
            'J_superexchange_180': J_180,
            'T_Neel_estimated': T_N,
            'T_Neel_experimental': 523.0,  # K
            'exchange_path': 'Ni-O-Ni_180deg'
        }
    
    def analyze_magnetic_structure(self) -> Dict:
        """
        Analyze magnetic structure
        
        NiO has Type-II AFM structure:
        - Ferromagnetic (111) planes
        - Antiferromagnetic stacking along [111]
        """
        # Magnetic unit cell is doubled along [111]
        
        # Exchange constants
        J1 = -19.0  # meV, nearest neighbor (AFM)
        J2 = 76.0   # meV, next-nearest neighbor (FM)
        
        # Spin wave dispersion
        # ω(q) = 2S √[(J1(0) - J1(q))(J1(0) - J1(q) + 2J2(0) - 2J2(q))]
        
        return {
            'magnetic_structure': 'Type_II_AFM',
            'propagation_vector': [0.5, 0.5, 0.5],
            'J1_meV': J1,
            'J2_meV': J2,
            'spin_wave_gap': 0,  # Goldstone mode
            'magnetic_moment': 1.77  # μB (experimental)
        }
    
    def calculate_optical_properties(self) -> Dict:
        """
        Calculate optical properties
        
        Key features:
        - d-d transitions (crystal field)
        - Charge-transfer excitations
        - Excitons
        """
        # Crystal field transitions
        # Ni²⁺ in octahedral field: ³A2g → ³T2g, ³T1g
        
        d_d_transitions = {
            '³A2g_→_³T2g': 1.1,   # eV
            '³A2g_→_³T1g': 1.8,   # eV
            '³A2g_→_³T1g(P)': 3.5  # eV
        }
        
        # Charge-transfer excitations
        # O 2p → Ni 3d (ligand-to-metal)
        ct_gap = self.config.Delta_ct
        
        # Exciton binding energy
        # In NiO, excitons are strongly bound due to localized nature
        exciton_binding = 0.5  # eV
        
        return {
            'd_d_transitions': d_d_transitions,
            'charge_transfer_gap': ct_gap,
            'exciton_binding_energy': exciton_binding,
            'optical_gap': ct_gap - exciton_binding
        }


class CoOAnalyzer:
    """
    Detailed analyzer for CoO (orbital-ordered Mott insulator)
    
    Key features:
    - Type-II antiferromagnetic ordering
    - Orbital ordering below T_N (spin-orbit coupling effects)
    - Anisotropic magnetic properties
    - Unquenched orbital moment
    """
    
    def __init__(self, config: MottInsulatorConfig = None):
        self.config = config or MottInsulatorConfig("CoO")
        self.config.n_d_electrons = 7  # Co²⁺ is 3d⁷
        self.config.lattice_constant = 4.26  # Å
        self._setup_crystal_structure()
        self._setup_electronic_structure()
    
    def _setup_crystal_structure(self):
        """Setup CoO crystal structure (distorted rocksalt)"""
        # CoO has tetragonal distortion below T_N
        a = self.config.lattice_constant
        c = a * 1.02  # Small tetragonal distortion
        
        self.lattice = np.array([
            [a, 0, 0],
            [0, a, 0],
            [0, 0, c]
        ])
        
        # Type-II AFM ordering similar to NiO
        self.co_positions = {
            'up': np.array([[0, 0, 0]]),
            'down': np.array([[0.5, 0.5, 0.5]])
        }
    
    def _setup_electronic_structure(self):
        """Setup Co²⁺ 3d⁷ electronic configuration"""
        # Co²⁺: 3d⁷ configuration
        # High-spin: t2g⁵ eg² → S=3/2
        
        self.d_config = {
            't2g': 5,  # 5 electrons (one hole)
            'eg': 2,   # Fully occupied
        }
        
        # Spin moment
        self.spin_moment = 1.5  # μB per Co (S=3/2)
        
        # Orbital moment (partially unquenched due to t2g hole)
        # L = 1 for t2g hole (effective p-like)
        self.orbital_moment = 1.0  # μB
    
    def calculate_orbital_ordering(self) -> Dict:
        """
        Analyze orbital ordering in CoO
        
        Below T_N = 293 K, CoO develops orbital ordering
        due to cooperative Jahn-Teller effect
        """
        # Orbital order parameter
        # In CoO, the t2g hole orders in the xy orbital
        
        # Exchange striction mechanism
        # Orbital ordering is driven by exchange interactions
        
        T_N = 293.0  # K
        
        # Order parameter temperature dependence
        # m_orb ∝ (T_N - T)^β with β ≈ 0.3 (critical exponent)
        
        return {
            'orbital_ordered': True,
            'T_ordering': T_N,
            'ordered_orbital': 't2g_xy',
            'distortion_type': 'tetragonal',
            'c_over_a': 1.02,
            'orbital_moment': 1.0,
            'spin_orbit_coupling': True
        }
    
    def analyze_magnetic_anisotropy(self) -> Dict:
        """
        Analyze magnetic anisotropy in CoO
        
        Due to unquenched orbital moment, CoO has strong
        magnetocrystalline anisotropy
        """
        # Anisotropy constant
        # Easy axis is along [117] direction (tilted from [001])
        
        # Single-ion anisotropy from spin-orbit coupling
        # D S_z² term
        D = -2.0  # meV, negative means easy axis perpendicular to z
        
        # Anisotropy field
        g = 2.0
        mu_B = 5.788e-2  # meV/T
        H_anisotropy = 2 * abs(D) / (g * mu_B)
        
        return {
            'easy_axis': '[117]',
            'anisotropy_constant_D_meV': D,
            'anisotropy_field_T': H_anisotropy,
            'origin': 'spin_orbit_coupling',
            'magnetocrystalline': 'strong'
        }
    
    def calculate_spin_orbit_coupling(self) -> Dict:
        """
        Calculate spin-orbit coupling effects
        
        For Co²⁺ (3d⁷):
        - Ground state: ⁴T1 (from ⁴F free ion)
        - Spin-orbit splits into Γ6, Γ7, Γ8 levels
        """
        # Spin-orbit coupling constant
        lambda_soc = -25.0  # meV (negative for less than half-filled)
        
        # Effective moment (including orbital contribution)
        # μ_eff = g_J √[J(J+1)] μB
        
        # For ⁴T1: J_eff = 1/2 ground state (Kramers doublet)
        g_J = 2.0  # Effective g-factor
        J_eff = 0.5
        
        mu_eff = g_J * np.sqrt(J_eff * (J_eff + 1))
        
        return {
            'spin_orbit_coupling_meV': lambda_soc,
            'ground_state': 'Gamma6_doublet',
            'effective_g_factor': g_J,
            'effective_moment': mu_eff,
            'experimental_moment': 3.8  # μB
        }


class MottInsulatorWorkflow:
    """
    Complete workflow for Mott insulator calculations
    
    Combines DFT+U, DMFT, and analysis tools for comprehensive
    study of Mott insulators.
    """
    
    def __init__(self, material: str):
        self.material = material
        
        if material == "NiO":
            self.analyzer = NiOAnalyzer()
        elif material == "CoO":
            self.analyzer = CoOAnalyzer()
        else:
            raise ValueError(f"Material {material} not supported")
    
    def run_complete_analysis(self) -> Dict:
        """Run complete analysis workflow"""
        results = {}
        
        # Electronic structure
        results['electronic'] = self.analyzer.calculate_electronic_structure()
        
        # Magnetic properties
        if hasattr(self.analyzer, 'analyze_magnetic_structure'):
            results['magnetic'] = self.analyzer.analyze_magnetic_structure()
        
        if hasattr(self.analyzer, 'calculate_superexchange'):
            results['exchange'] = self.analyzer.calculate_superexchange()
        
        # Orbital properties (CoO specific)
        if hasattr(self.analyzer, 'calculate_orbital_ordering'):
            results['orbital'] = self.analyzer.calculate_orbital_ordering()
        
        if hasattr(self.analyzer, 'analyze_magnetic_anisotropy'):
            results['anisotropy'] = self.analyzer.analyze_magnetic_anisotropy()
        
        # Optical properties (NiO specific)
        if hasattr(self.analyzer, 'calculate_optical_properties'):
            results['optical'] = self.analyzer.calculate_optical_properties()
        
        return results
    
    def generate_dft_input(self, output_dir: str = "vasp_mott"):
        """Generate VASP input files"""
        os.makedirs(output_dir, exist_ok=True)
        
        material = self.material
        config = self.analyzer.config
        
        # POSCAR
        a = config.lattice_constant
        with open(os.path.join(output_dir, "POSCAR"), "w") as f:
            f.write(f"{material}\n")
            f.write("1.0\n")
            f.write(f"{a:.6f} 0.0 0.0\n")
            f.write(f"0.0 {a:.6f} 0.0\n")
            f.write(f"0.0 0.0 {a:.6f}\n")
            
            if material == "NiO":
                f.write("Ni O\n")
                f.write("2 2\n")
                f.write("Direct\n")
                f.write("0.0 0.0 0.0\n")
                f.write("0.5 0.5 0.0\n")
                f.write("0.5 0.0 0.5\n")
                f.write("0.0 0.5 0.5\n")
            elif material == "CoO":
                f.write("Co O\n")
                f.write("2 2\n")
                f.write("Direct\n")
                f.write("0.0 0.0 0.0\n")
                f.write("0.5 0.5 0.0\n")
                f.write("0.5 0.0 0.5\n")
                f.write("0.0 0.5 0.5\n")
        
        # INCAR for AFM calculation
        with open(os.path.join(output_dir, "INCAR"), "w") as f:
            f.write(f"""# {material} Mott Insulator Calculation
SYSTEM = {material}

# Electronic structure
PREC = Accurate
ENCUT = 600
ISMEAR = 0
SIGMA = 0.05
EDIFF = 1E-8

# DFT+U
LDAU = .TRUE.
LDAUTYPE = 2
LDAUL = 2 -1
LDAUU = {config.U_metal:.1f} 0.0
LDAUJ = {config.J_metal:.1f} 0.0

# AFM ordering
ISPIN = 2
MAGMOM = 2 -2 0 0

# Spin-orbit coupling (important for CoO)
LSORBIT = .{'TRUE' if material == 'CoO' else 'FALSE.'}

# Output
LDAUPRINT = 2
LORBIT = 11
""")
        
        logger.info(f"VASP input files for {material} written to {output_dir}")


# Utility functions

def classify_insulator_type(U: float, Delta: float) -> str:
    """
    Classify insulator type using Zaanen-Sawatzky-Allen scheme
    
    Parameters:
    -----------
    U : float
        d-d Coulomb repulsion
    Delta : float
        Charge transfer energy
        
    Returns:
    --------
    insulator_type : str
        Mott-Hubbard, charge-transfer, or intermediate
    """
    ratio = U / Delta
    
    if ratio < 0.8:
        return "charge-transfer"
    elif ratio > 1.2:
        return "Mott-Hubbard"
    else:
        return "intermediate"


def estimate_neel_temperature(exchange_J: float, 
                              spin: float,
                              coordination: int = 6) -> float:
    """
    Estimate Neel temperature from superexchange
    
    Mean-field: k_B T_N = (2/3) z S(S+1) J
    """
    kB = 8.617e-5  # eV/K
    T_N = (2.0/3.0) * coordination * spin * (spin + 1) * exchange_J / kB
    return T_N


def calculate_spin_wave_spectrum(q_points: np.ndarray,
                                 J1: float, J2: float,
                                 S: float = 1) -> np.ndarray:
    """
    Calculate spin wave dispersion for Type-II AFM
    
    ω(q) = 2S z J1 |sin(q·a/2)|
    """
    omega = np.zeros(len(q_points))
    
    for i, q in enumerate(q_points):
        # Simplified dispersion
        omega[i] = 2 * S * abs(J1) * np.sqrt(
            1 - np.cos(q[0]) * np.cos(q[1]) * np.cos(q[2])
        )
    
    return omega


__all__ = [
    'MottInsulatorConfig',
    'NiOAnalyzer',
    'CoOAnalyzer',
    'MottInsulatorWorkflow',
    'classify_insulator_type',
    'estimate_neel_temperature',
    'calculate_spin_wave_spectrum'
]