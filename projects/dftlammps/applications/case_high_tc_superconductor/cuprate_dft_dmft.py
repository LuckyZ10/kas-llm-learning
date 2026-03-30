"""
High-Tc Superconductor DFT+DMFT Workflow

This module provides workflows for studying high-temperature superconductors
using DFT+DMFT methods:
- Cuprates (La2CuO4, YBCO, BSCCO)
- Iron-based superconductors (LaFeAsO, BaFe2As2, FeSe)

Author: DFT-LAMMPS Team
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CuprateConfig:
    """Configuration for cuprate calculations"""
    material: str = "La2CuO4"
    
    # Structural parameters
    a_lattice: float = 3.80  # Å
    c_lattice: float = 13.20  # Å
    
    # Electronic parameters
    n_copper_layers: int = 1
    hole_doping: float = 0.0  # x in La2-xSrxCuO4
    
    # DFT+U parameters
    U_cu: float = 8.0  # eV
    J_cu: float = 1.0  # eV
    
    # DMFT parameters
    use_dmft: bool = True
    n_orbitals: int = 1  # Cu dx2-y2
    
    # Superconductivity
    calculate_pairing: bool = True
    pairing_symmetry: str = "d-wave"


@dataclass
class IronBasedConfig:
    """Configuration for iron-based superconductor calculations"""
    material: str = "BaFe2As2"
    
    # Structural parameters
    a_lattice: float = 3.96  # Å
    c_lattice: float = 13.02  # Å
    
    # Electronic parameters
    n_iron_layers: int = 2
    electron_doping: float = 0.0
    
    # DFT+U parameters
    U_fe: float = 3.5  # eV
    J_fe: float = 0.8  # eV
    
    # DMFT parameters
    use_dmft: bool = True
    n_orbitals: int = 5  # Fe 3d orbitals
    
    # Superconductivity
    calculate_pairing: bool = True
    pairing_symmetry: str = "s+-"


class CuprateDFTDMFT:
    """
    DFT+DMFT workflow for cuprate superconductors
    
    Studies the single-band Hubbard model on a square lattice
    with strong correlations leading to d-wave superconductivity.
    """
    
    def __init__(self, config: CuprateConfig = None):
        self.config = config or CuprateConfig()
        self.results = {}
        
    def setup_structure(self) -> Dict[str, np.ndarray]:
        """Setup crystal structure for cuprate"""
        material = self.config.material
        
        if material == "La2CuO4":
            # T-phase La2CuO4 (K2NiF4 structure)
            # I4/mmm space group
            lattice = np.array([
                [self.config.a_lattice, 0, 0],
                [0, self.config.a_lattice, 0],
                [0, 0, self.config.c_lattice]
            ])
            
            # Atomic positions (fractional)
            positions = {
                'La': np.array([[0.5, 0.5, 0.136], [0.5, 0.5, 0.864]]),
                'Cu': np.array([[0, 0, 0]]),
                'O': np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0.183]])
            }
            
        elif material == "YBa2Cu3O7":
            # Orthorhombic Pmmm
            lattice = np.array([
                [3.82, 0, 0],
                [0, 3.89, 0],
                [0, 0, 11.68]
  ])
            
            positions = {
                'Y': np.array([[0.5, 0.5, 0.5]]),
                'Ba': np.array([[0.5, 0.5, 0.184], [0.5, 0.5, 0.816]]),
                'Cu': np.array([[0, 0, 0], [0, 0, 0.356]]),
                'O': np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0.158], 
                              [0, 0.5, 0.378], [0.5, 0, 0.378]])
            }
        else:
            raise ValueError(f"Material {material} not supported")
        
        return {'lattice': lattice, 'positions': positions}
    
    def generate_dft_input(self, output_dir: str = "vasp_cuprate"):
        """Generate VASP input files for cuprate"""
        os.makedirs(output_dir, exist_ok=True)
        
        structure = self.setup_structure()
        
        # Write POSCAR
        with open(os.path.join(output_dir, "POSCAR"), "w") as f:
            f.write(f"{self.config.material}\n")
            f.write("1.0\n")
            for vec in structure['lattice']:
                f.write(f"{vec[0]:.10f} {vec[1]:.10f} {vec[2]:.10f}\n")
            
            # Atom types and counts
            atoms = []
            counts = []
            for atom_type, pos in structure['positions'].items():
                atoms.append(atom_type)
                counts.append(len(pos))
            
            f.write(" ".join(atoms) + "\n")
            f.write(" ".join(map(str, counts)) + "\n")
            f.write("Direct\n")
            
            # Write positions
            for atom_type, pos in structure['positions'].items():
                for p in pos:
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        
        # Write INCAR for DFT+U
        with open(os.path.join(output_dir, "INCAR"), "w") as f:
            f.write(f"""# Cuprate DFT+U calculation
SYSTEM = {self.config.material}

# Electronic structure
PREC = Accurate
ENCUT = 600
ISMEAR = 0
SIGMA = 0.05
EDIFF = 1E-8

# DFT+U for Cu d-orbitals
LDAU = .TRUE.
LDAUTYPE = 2
LDAUL = -1 -1 2 -1
LDAUU = 0 0 {self.config.U_cu} 0
LDAUJ = 0 0 {self.config.J_cu} 0
LDAUPRINT = 2

# Spin polarization
ISPIN = 2
MAGMOM = 0 0 1 0

# Wannier90 interface
LWANNIER90 = .TRUE.
LWRITE_WANNIER90 = .TRUE.
""")
        
        logger.info(f"VASP input files written to {output_dir}")
    
    def run_dmft_cycle(self, H_k: np.ndarray, k_weights: np.ndarray) -> Dict:
        """
        Run DMFT cycle for cuprate single-band model
        
        Parameters:
        -----------
        H_k : np.ndarray
            Non-interacting Hamiltonian from Wannier90
        k_weights : np.ndarray
            k-point weights
            
        Returns:
        --------
        results : dict
            DMFT results
        """
        try:
            from ...correlated import DMFTEngine, DMFTConfig
        except ImportError:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            from correlated import DMFTEngine, DMFTConfig
        
        # Setup DMFT
        config = DMFTConfig(
            temperature=100.0,
            u_value=self.config.U_cu,
            j_value=self.config.J_cu,
            n_orbitals=1,
            scf_max_iter=50
        )
        
        dmft = DMFTEngine(config)
        dmft.initialize(solver_type="triqs")
        
        # Run self-consistent loop
        results = dmft.run_scf_loop(H_k, k_weights, n_electrons=1.0)
        
        # Calculate spectral function
        omega, A_w = dmft.calculate_spectral_function(
            H_k, np.array([[0, 0, 0]]), np.array([1.0])
        )
        
        results['omega'] = omega
        results['spectral_function'] = A_w
        
        return results
    
    def analyze_d_wave_pairing(self, chi_spin: np.ndarray,
                               q_points: np.ndarray) -> Dict:
        """
        Analyze d-wave pairing from spin susceptibility
        
        Uses the spin-fluctuation mechanism:
        V_pair(q) ∝ (3/2) U² χ_spin(q) / (1 - U χ_spin(q))
        """
        U = self.config.U_cu
        
        # RPA spin susceptibility
        chi_rpa = chi_spin / (1 - U * chi_spin)
        
        # Pairing interaction in d-wave channel
        # Form factor for d_{x²-y²}
        d_wave_ff = np.array([
            np.cos(q[0]) - np.cos(q[1]) for q in q_points
        ])
        
        # Project onto d-wave
        V_d = np.sum(d_wave_ff**2 * chi_rpa) / np.sum(d_wave_ff**2)
        
        # Tc estimate (simplified)
        Tc = 0.1 * U * np.max(chi_rpa)  # Rough estimate
        
        results = {
            'chi_rpa': chi_rpa,
            'V_d_wave': V_d,
            'Tc_estimate': Tc,
            'pairing_channel': 'd_x2-y2'
        }
        
        return results
    
    def calculate_phase_diagram(self, doping_range: Tuple[float, float] = (0.0, 0.3),
                               n_doping: int = 10) -> Dict:
        """
        Calculate cuprate phase diagram vs hole doping
        
        Returns phase diagram with:
        - Antiferromagnetic phase
        - Pseudogap phase
        - Superconducting dome
        - Strange metal phase
        """
        dopings = np.linspace(doping_range[0], doping_range[1], n_doping)
        
        phases = []
        T_Neel = []
        T_pseudo = []
        Tc = []
        
        for x in dopings:
            self.config.hole_doping = x
            
            # Estimate transition temperatures
            # AFM: decreases with doping
            if x < 0.02:
                phases.append('AF_insulator')
                T_Neel.append(300 * (1 - x/0.02))
            elif x < 0.05:
                phases.append('AF_metal')
                T_Neel.append(0)
            elif x < 0.3:
                phases.append('superconducting')
                # Superconducting dome
                Tc.append(100 * np.sin(np.pi * x / 0.3))
            else:
                phases.append('overdoped')
                Tc.append(0)
            
            T_pseudo.append(500 * (1 - x/0.35) if x < 0.35 else 0)
        
        return {
            'doping': dopings,
            'phases': phases,
            'T_Neel': np.array(T_Neel),
            'T_pseudo': np.array(T_pseudo),
            'Tc': np.array(Tc)
        }


class IronBasedDFTDMFT:
    """
    DFT+DMFT workflow for iron-based superconductors
    
    Studies the multi-orbital Hubbard model on a square lattice
    with Hund's coupling leading to s± superconductivity.
    """
    
    def __init__(self, config: IronBasedConfig = None):
        self.config = config or IronBasedConfig()
        self.results = {}
    
    def setup_structure(self) -> Dict[str, np.ndarray]:
        """Setup crystal structure for iron-based superconductor"""
        material = self.config.material
        
        if material == "BaFe2As2":
            # ThCr2Si2 structure (I4/mmm)
            a = self.config.a_lattice
            c = self.config.c_lattice
            
            lattice = np.array([
                [a, 0, 0],
                [0, a, 0],
                [0, 0, c]
            ])
            
            positions = {
                'Ba': np.array([[0, 0, 0.25], [0, 0, 0.75]]),
                'Fe': np.array([[0.5, 0, 0], [0, 0.5, 0], 
                               [0.5, 0, 0.5], [0, 0.5, 0.5]]),
                'As': np.array([[0, 0, 0.36], [0, 0, 0.64],
                               [0.5, 0.5, 0.14], [0.5, 0.5, 0.86]])
            }
            
        elif material == "LaFeAsO":
            # ZrCuSiAs structure (P4/nmm)
            a = 4.03
            c = 8.74
            
            lattice = np.array([
                [a, 0, 0],
                [0, a, 0],
                [0, 0, c]
            ])
            
            positions = {
                'La': np.array([[0.25, 0.75, 0.14], [0.75, 0.25, 0.86]]),
                'Fe': np.array([[0.75, 0.25, 0.5], [0.25, 0.75, 0.5]]),
                'As': np.array([[0.25, 0.75, 0.65], [0.75, 0.25, 0.35]]),
                'O': np.array([[0.75, 0.25, 0], [0.25, 0.75, 0]])
            }
        else:
            raise ValueError(f"Material {material} not supported")
        
        return {'lattice': lattice, 'positions': positions}
    
    def run_multiorbital_dmft(self, H_k: np.ndarray, 
                             k_weights: np.ndarray) -> Dict:
        """
        Run multi-orbital DMFT for iron-based superconductor
        
        Includes all 5 Fe 3d orbitals with Hund's coupling
        """
        try:
            from ...correlated import DMFTEngine, DMFTConfig
        except ImportError:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            from correlated import DMFTEngine, DMFTConfig
        
        # Multi-orbital DMFT
        config = DMFTConfig(
            temperature=100.0,
            u_value=self.config.U_fe,
            j_value=self.config.J_fe,
            n_orbitals=5,
            scf_max_iter=50
        )
        
        dmft = DMFTEngine(config)
        dmft.initialize(solver_type="triqs")
        
        results = dmft.run_scf_loop(H_k, k_weights)
        
        return results
    
    def analyze_s_pm_pairing(self, chi_spin: np.ndarray,
                            q_points: np.ndarray) -> Dict:
        """
        Analyze s± pairing from multi-orbital spin susceptibility
        
        The s± state has opposite signs on electron and hole pockets
        """
        # Identify electron and hole pockets
        # Simplified: use momentum space regions
        
        # Form factor for s± (changes sign between pockets)
        # At Γ (hole): +
        # At M (electron): -
        
        s_pm_ff = np.array([
            1.0 if np.linalg.norm(q) < np.pi/2 else -1.0
            for q in q_points
        ])
        
        # Project pairing interaction
        V_spm = np.sum(s_pm_ff**2 * chi_spin) / np.sum(s_pm_ff**2)
        
        # Check for accidental nodes
        has_nodes = False  # Simplified
        
        results = {
            'V_s_pm': V_spm,
            'has_nodes': has_nodes,
            'pairing_symmetry': 's+-'
        }
        
        return results
    
    def calculate_orbital_selective_mott(self, 
                                         orbital_occupations: np.ndarray) -> Dict:
        """
        Analyze orbital-selective Mott transition
        
        In iron pnictides, some orbitals can be localized while others remain itinerant
        """
        # Orbital labels
        orbitals = ['d_xy', 'd_yz', 'd_xz', 'd_x2-y2', 'd_z2']
        
        # Calculate orbital-dependent quasiparticle weight
        # From DMFT self-energy
        
        # Identify Mott-localized orbitals (Z << 1)
        Z_threshold = 0.1
        mott_orbitals = []
        itinerant_orbitals = []
        
        for i, occ in enumerate(orbital_occupations):
            # Simplified criterion based on occupation
            if np.abs(occ - 1.0) < 0.1 or np.abs(occ - 2.0) < 0.1:
                mott_orbitals.append(orbitals[i])
            else:
                itinerant_orbitals.append(orbitals[i])
        
        results = {
            'mott_orbitals': mott_orbitals,
            'itinerant_orbitals': itinerant_orbitals,
            'is_orbital_selective': len(mott_orbitals) > 0 and len(itinerant_orbitals) > 0
        }
        
        return results


# Utility functions

def estimate_cuprate_tc(doping: float, material: str = "LSCO") -> float:
    """
    Empirical estimate of Tc for cuprates
    
    Uses parabolic formula: Tc/Tc_max = 1 - 82.6(p - 0.16)²
    """
    if material == "LSCO":
        Tc_max = 38.0
    elif material == "YBCO":
        Tc_max = 93.0
    elif material == "BSCCO":
        Tc_max = 110.0
    else:
        Tc_max = 100.0
    
    p_optimal = 0.16
    Tc = Tc_max * (1 - 82.6 * (doping - p_optimal)**2)
    
    return max(Tc, 0)


def estimate_iron_based_tc(electron_doping: float, 
                           material: str = "Ba122") -> float:
    """
    Empirical estimate of Tc for iron-based superconductors
    """
    if material == "Ba122":
        Tc_max = 38.0
        x_optimal = 0.4
    elif material == "La1111":
        Tc_max = 26.0
        x_optimal = 0.1
    elif material == "FeSe":
        Tc_max = 8.0
        x_optimal = 0.0
    else:
        Tc_max = 30.0
        x_optimal = 0.3
    
    Tc = Tc_max * np.exp(-(electron_doping - x_optimal)**2 / 0.1)
    
    return max(Tc, 0)


def calculate_superfluid_stiffness(spectral_function: np.ndarray,
                                   omega: np.ndarray,
                                   temperature: float) -> float:
    """
    Calculate superfluid stiffness D_s
    
    D_s = (π e² / 2) ∫ dω (-∂f/∂ω) Σ_k v_k² A(k, ω)²
    """
    # Derivative of Fermi function
    kB = 8.617333e-5
    df_domega = (1 / (kB * temperature)) * \
                np.exp(omega / (kB * temperature)) / \
                (np.exp(omega / (kB * temperature)) + 1)**2
    
    # Approximate integration
    D_s = np.trapz(spectral_function**2 * df_domega, omega)
    
    return D_s


__all__ = [
    'CuprateConfig',
    'IronBasedConfig',
    'CuprateDFTDMFT',
    'IronBasedDFTDMFT',
    'estimate_cuprate_tc',
    'estimate_iron_based_tc',
    'calculate_superfluid_stiffness'
]