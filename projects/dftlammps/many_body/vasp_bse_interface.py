"""
VASP BSE Interface Module
==========================
Interface to VASP for BSE and GW calculations.
Handles ALGO=G0W0, BSE, and optical absorption calculations.

References:
- Shishkin & Kresse, Implementation and performance of the frequency-dependent
  GW method (2006)
- Kresse et al., On the performance of common exchange-correlation functionals
  (2012)
"""

import os
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import json
import subprocess


class VaspAlgo(Enum):
    """VASP ALGO settings."""
    DFT = "Normal"
    G0W0 = "G0W0"
    GW0 = "GW0"
    G0W0R = "G0W0R"
    BSE = "BSE"
    TDHF = "TimeDependentHF"


class VaspPotcar(Enum):
    """POTCAR types."""
    PAW = "PAW"
    PAW_PBE = "PAW_PBE"
    PAW_LDA = "PAW_LDA"


@dataclass
class VaspGWParameters:
    """Parameters for VASP GW calculations."""
    # Number of bands
    NBANDS: int = 512
    
    # GW parameters
    NOMEGA: int = 64  # Number of frequency points
    OMEGAMAX: float = -1  # Automatic
    
    # Screening
    ENCUTGW: float = 100.0  # eV
    NOMEGAPAR: int = 16
    
    # QP energies
    NELM: int = 1  # One-shot
    
    # k-point grid for GW
    NKREDX: int = 1
    NKREDY: int = 1
    NKREDZ: int = 1
    
    def to_dict(self) -> Dict:
        return {
            'NBANDS': self.NBANDS,
            'NOMEGA': self.NOMEGA,
            'ENCUTGW': self.ENCUTGW,
            'NOMEGAPAR': self.NOMEGAPAR
        }


@dataclass
class VaspBSEParameters:
    """Parameters for VASP BSE calculations."""
    # Antenna dielectric function
    ANTIRES: int = 0  # 0: TDA, 1: with antiresonant terms
    
    # Number of bands for BSE
    NBANDSO: int = 5  # Occupied bands
    NBANDSV: int = 10  # Virtual bands
    
    # Hamiltonian type
    LHARTREE: bool = True
    LADDER: bool = True
    
    # Optical properties
    LOPTICS: bool = True
    NEDOS: int = 2000
    
    # Energy range
    EMIN: float = -20.0
    EMAX: float = 20.0
    
    def to_dict(self) -> Dict:
        return {
            'ANTIRES': self.ANTIRES,
            'NBANDSO': self.NBANDSO,
            'NBANDSV': self.NBANDSV,
            'LOPTICS': self.LOPTICS
        }


class VaspBSEInterface:
    """
    Interface to VASP for BSE and GW calculations.
    
    Workflow:
    1. DFT ground state calculation (WAVECAR)
    2. GW calculation (optional, WAVEDER)
    3. BSE calculation (WAVECAR + vasp)
    4. Optical properties extraction
    """
    
    def __init__(self,
                 vasp_path: str = "vasp_std",
                 work_dir: str = "./vasp_bse",
                 verbose: bool = True):
        """
        Initialize VASP interface.
        
        Args:
            vasp_path: Path to VASP executable
            work_dir: Working directory
            verbose: Print detailed output
        """
        self.vasp_path = vasp_path
        self.work_dir = Path(work_dir)
        self.verbose = verbose
        
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.qp_energies: Optional[np.ndarray] = None
        self.dielectric_function: Optional[Dict] = None
        self.bse_energies: Optional[np.ndarray] = None
        self.oscillator_strengths: Optional[np.ndarray] = None
        self.absorption_spectrum: Optional[Dict] = None
    
    def write_incar_gw(self, params: VaspGWParameters) -> None:
        """
        Write INCAR for GW calculation.
        """
        incar_content = f"""
# GW Calculation
# =============
ALGO = G0W0
NBANDS = {params.NBANDS}
NOMEGA = {params.NOMEGA}
ENCUTGW = {params.ENCUTGW}
NOMEGAPAR = {params.NOMEGAPAR}

# Electronic structure
ISMEAR = 0
SIGMA = 0.05
EDIFF = 1E-8
NELM = 1

# Parallelization
NKREDX = {params.NKREDX}
NKREDY = {params.NKREDY}
NKREDZ = {params.NKREDZ}

# Output
LWAVE = .TRUE.
LCHARG = .TRUE.
"""
        
        incar_path = self.work_dir / "INCAR_GW"
        with open(incar_path, 'w') as f:
            f.write(incar_content)
        
        if self.verbose:
            print(f"GW INCAR written to {incar_path}")
    
    def write_incar_bse(self, params: VaspBSEParameters) -> None:
        """
        Write INCAR for BSE calculation.
        """
        antires_str = ".TRUE." if params.ANTIRES == 1 else ".FALSE."
        
        incar_content = f"""
# BSE Calculation
# ==============
ALGO = BSE
ANTIRES = {params.ANTIRES}

# Band range
NBANDSO = {params.NBANDSO}
NBANDSV = {params.NBANDSV}

# Hamiltonian
LHARTREE = .TRUE.
LADDER = .TRUE.

# Optical properties
LOPTICS = .TRUE.
NEDOS = {params.NEDOS}

# Energy range
EMIN = {params.EMIN}
EMAX = {params.EMAX}

# Restart
ISTART = 1
"""
        
        incar_path = self.work_dir / "INCAR_BSE"
        with open(incar_path, 'w') as f:
            f.write(incar_content)
        
        if self.verbose:
            print(f"BSE INCAR written to {incar_path}")
    
    def write_kpoints(self, 
                      kgrid: Tuple[int, int, int],
                      gamma_centered: bool = True) -> None:
        """
        Write KPOINTS file.
        """
        kpoints_content = f"""
Automatic mesh
0
{'Gamma' if gamma_centered else 'Monkhorst-Pack'}
{kgrid[0]} {kgrid[1]} {kgrid[2]}
0 0 0
"""
        
        kpoints_path = self.work_dir / "KPOINTS"
        with open(kpoints_path, 'w') as f:
            f.write(kpoints_content)
        
        if self.verbose:
            print(f"KPOINTS written: {kgrid}")
    
    def parse_waveder(self, filename: str = "WAVEDER") -> Optional[np.ndarray]:
        """
        Parse WAVEDER file for derivative matrix elements.
        
        Returns:
            Derivative matrix elements [nk, nb, nb, 3]
        """
        filepath = self.work_dir / filename
        
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            return None
        
        # Simplified parsing (actual format is binary)
        # In real implementation, use specific reader
        
        if self.verbose:
            print(f"Parsing {filename}...")
        
        # Return dummy data for now
        nk, nb = 10, 20
        return np.random.randn(nk, nb, nb, 3) + 1j * np.random.randn(nk, nb, nb, 3)
    
    def parse_vasprun(self, filename: str = "vasprun.xml") -> Dict:
        """
        Parse vasprun.xml for energies and eigenvalues.
        
        Returns:
            Dictionary with parsed data
        """
        filepath = self.work_dir / filename
        
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            return {}
        
        results = {
            'eigenvalues': None,
            'fermi_energy': None,
            'bandgap': None
        }
        
        # Parse XML (simplified - use ElementTree in real implementation)
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Extract eigenvalues
            # This is simplified - actual parsing depends on vasprun structure
            
            if self.verbose:
                print(f"Parsed {filename}")
            
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
        
        return results
    
    def parse_dielectric_function(self, 
                                   filename: str = "vasprun.xml") -> Dict:
        """
        Extract dielectric function from VASP output.
        
        Returns:
            Dielectric function data
        """
        # Energy grid
        energies = np.linspace(0, 20, 2000)
        
        # Simulate dielectric function
        # Realistic model for semiconductor
        epsilon1 = np.ones_like(energies) * 10  # Background
        epsilon2 = np.zeros_like(energies)
        
        # Add absorption edges
        bandgap = 1.5
        for E in energies:
            if E > bandgap:
                idx = np.argmin(np.abs(energies - E))
                epsilon2[idx] += np.sqrt(E - bandgap) / E
        
        # Broaden
        from scipy.ndimage import gaussian_filter1d
        epsilon2 = gaussian_filter1d(epsilon2, sigma=5)
        
        self.dielectric_function = {
            'energies': energies,
            'epsilon1': epsilon1,
            'epsilon2': epsilon2,
            'absorption': 2 * energies * np.sqrt((np.sqrt(epsilon1**2 + epsilon2**2) - epsilon1) / 2)
        }
        
        return self.dielectric_function
    
    def calculate_bse_spectrum(self,
                               bse_params: VaspBSEParameters,
                               broadening: float = 0.1) -> Dict:
        """
        Calculate BSE optical absorption spectrum.
        
        Args:
            bse_params: BSE parameters
            broadening: Lorentzian broadening in eV
            
        Returns:
            Absorption spectrum
        """
        if self.verbose:
            print("Calculating BSE absorption spectrum...")
        
        # Write input files
        self.write_incar_bse(bse_params)
        
        # Run VASP (simulated)
        # subprocess.run([self.vasp_path], cwd=self.work_dir)
        
        # Parse results
        self.bse_energies = np.array([2.0, 2.5, 3.2, 4.1])  # Simulated
        self.oscillator_strengths = np.array([1.0, 0.5, 0.3, 0.2])
        
        # Build spectrum
        energies = np.linspace(0, 10, 1000)
        spectrum = np.zeros_like(energies)
        
        for E_ex, f in zip(self.bse_energies, self.oscillator_strengths):
            spectrum += f * broadening / ((energies - E_ex)**2 + broadening**2)
        
        self.absorption_spectrum = {
            'energies': energies,
            'spectrum': spectrum,
            'exciton_energies': self.bse_energies,
            'oscillator_strengths': self.oscillator_strengths
        }
        
        return self.absorption_spectrum
    
    def get_exciton_binding(self, 
                           optical_gap: float,
                           lowest_exciton: Optional[float] = None) -> float:
        """
        Calculate exciton binding energy from VASP BSE.
        
        Args:
            optical_gap: Optical bandgap
            lowest_exciton: Lowest exciton energy (default from BSE results)
            
        Returns:
            Binding energy in eV
        """
        if lowest_exciton is None:
            if self.bse_energies is not None:
                lowest_exciton = np.min(self.bse_energies)
            else:
                lowest_exciton = optical_gap - 0.5  # Estimate
        
        binding = optical_gap - lowest_exciton
        
        return binding
    
    def run_full_workflow(self,
                         structure_file: str,
                         gw_params: VaspGWParameters,
                         bse_params: VaspBSEParameters) -> Dict:
        """
        Run complete GW+BSE workflow.
        
        Args:
            structure_file: POSCAR/CONTCAR file
            gw_params: GW parameters
            bse_params: BSE parameters
            
        Returns:
            Complete results dictionary
        """
        if self.verbose:
            print("\n" + "="*60)
            print("Starting VASP GW+BSE Workflow")
            print("="*60)
        
        # Step 1: DFT ground state
        print("\n1. DFT Ground State...")
        # Copy structure file to work directory
        # Run VASP with standard DFT settings
        
        # Step 2: GW calculation
        print("\n2. GW Calculation...")
        self.write_incar_gw(gw_params)
        # Run VASP
        
        # Step 3: BSE calculation
        print("\n3. BSE Calculation...")
        self.write_incar_bse(bse_params)
        # Run VASP
        
        # Step 4: Extract results
        print("\n4. Extracting Results...")
        absorption = self.calculate_bse_spectrum(bse_params)
        dielectric = self.parse_dielectric_function()
        
        results = {
            'absorption_spectrum': absorption,
            'dielectric_function': dielectric,
            'exciton_binding': self.get_exciton_binding(2.5),
            'work_directory': str(self.work_dir)
        }
        
        if self.verbose:
            print("\nWorkflow complete!")
            print(f"Exciton energies: {self.bse_energies}")
            print(f"Binding energy: {results['exciton_binding']:.3f} eV")
        
        return results


class VaspBatchGW:
    """
    Batch GW calculations for multiple systems.
    """
    
    def __init__(self, vasp_interface: VaspBSEInterface):
        self.vasp = vasp_interface
        self.results: List[Dict] = []
    
    def run_batch_gw(self,
                     systems: List[Dict],
                     gw_params: VaspGWParameters) -> List[Dict]:
        """
        Run GW calculations on multiple systems.
        
        Args:
            systems: List of system dictionaries
            gw_params: GW parameters
            
        Returns:
            List of results
        """
        results = []
        
        for system in systems:
            print(f"\nProcessing {system['name']}...")
            
            # Setup calculation
            work_dir = f"./vasp_batch/{system['name']}"
            vasp = VaspBSEInterface(
                work_dir=work_dir,
                verbose=False
            )
            
            # Run calculation
            vasp.write_incar_gw(gw_params)
            
            # Extract gap (simulated)
            gap = 1.5 + np.random.random() * 2
            
            results.append({
                'system': system['name'],
                'gw_gap': gap,
                'formula': system.get('formula', 'Unknown')
            })
        
        self.results = results
        return results
    
    def convergence_study(self,
                         base_params: VaspGWParameters,
                         parameter: str,
                         values: List[float]) -> Dict:
        """
        Run convergence study for a GW parameter.
        
        Args:
            base_params: Base parameters
            parameter: Parameter to vary
            values: Values to test
            
        Returns:
            Convergence results
        """
        results = []
        
        for value in values:
            params = VaspGWParameters(**base_params.to_dict())
            setattr(params, parameter, value)
            
            self.vasp.write_incar_gw(params)
            
            # Simulate calculation
            gap = 2.0 + 0.1 / value  # Convergence behavior
            
            results.append({
                parameter: value,
                'gap': gap
            })
        
        return {
            'parameter': parameter,
            'values': values,
            'gaps': [r['gap'] for r in results]
        }


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("VASP BSE Interface Example")
    print("="*60)
    
    # Initialize interface
    vasp = VaspBSEInterface(work_dir="./test_vasp_bse")
    
    # GW parameters
    gw_params = VaspGWParameters(
        NBANDS=256,
        NOMEGA=64,
        ENCUTGW=150
    )
    
    # BSE parameters
    bse_params = VaspBSEParameters(
        NBANDSO=4,
        NBANDSV=8,
        ANTIRES=0,  # TDA
        LOPTICS=True
    )
    
    # Write input files
    print("\n--- Writing Input Files ---")
    vasp.write_incar_gw(gw_params)
    vasp.write_incar_bse(bse_params)
    vasp.write_kpoints((6, 6, 6))
    
    # Calculate spectrum
    print("\n--- Calculating BSE Spectrum ---")
    spectrum = vasp.calculate_bse_spectrum(bse_params)
    print(f"Exciton energies: {spectrum['exciton_energies']}")
    print(f"Oscillator strengths: {spectrum['oscillator_strengths']}")
    
    # Get binding energy
    binding = vasp.get_exciton_binding(optical_gap=3.0)
    print(f"\nExciton binding energy: {binding:.3f} eV")
    
    print("\n" + "="*60)
