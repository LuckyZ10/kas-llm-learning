"""
EPW (Electron-Phonon Wannier) Interface
========================================

Interface to Quantum ESPRESSO EPW for electron-phonon coupling calculations.

Features:
- Wannier function generation (pw.x + pw2wannier90 + wannier90)
- EPW calculation setup and execution
- Electron-phonon matrix element extraction
- Fine q-point interpolation
- Temperature-dependent properties

Author: DFTLammps Electron-Phonon Team
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from subprocess import run, PIPE
import shutil

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EPWConfig:
    """Configuration for EPW calculations."""
    
    # Paths
    espresso_path: str = "/usr/bin"  # Path to QE binaries
    pseudo_dir: str = "./pseudo"
    outdir: str = "./tmp"
    
    # Parallelization
    nproc: int = 4
    npool: int = 4
    
    # Calculation parameters
    prefix: str = "epw"
    calculation: str = "epw"  # 'epw', 'eliashberg', 'supercond'
    
    # k/q grids
    k_mesh_coarse: Tuple[int, int, int] = (6, 6, 6)
    q_mesh_coarse: Tuple[int, int, int] = (6, 6, 6)
    k_mesh_fine: Tuple[int, int, int] = (20, 20, 20)
    q_mesh_fine: Tuple[int, int, int] = (20, 20, 20)
    
    # Wannier parameters
    n_wannier: Optional[int] = None
    dis_froz_min: Optional[float] = None
    dis_froz_max: Optional[float] = None
    
    # EPW parameters
    fsthick: float = 1.0  # eV - Fermi surface thickness
    degaussw: float = 0.05  # eV - smearing for Fermi window
    ephwrite: bool = True  # Write e-ph matrix elements
    
    # Temperature range
    temps: Optional[List[float]] = None  # K
    
    # Output
    output_dir: str = "./epw_output"
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.outdir, exist_ok=True)


@dataclass
class EPWResults:
    """Results from EPW calculation."""
    
    # Convergence info
    converged: bool = False
    
    # Electron-phonon coupling
    lambda_eph: Optional[float] = None  # Total coupling constant
    lambda_k: Optional[np.ndarray] = None  # k-dependent λ
    lambda_nuq: Optional[np.ndarray] = None  # Mode-resolved λ
    
    # Phonon frequencies at fine grid
    omega_nuq: Optional[np.ndarray] = None  # meV
    
    # Spectral function
    a2f: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (ω, α²F)
    
    # Superconducting properties
    omega_log: Optional[float] = None  # Logarithmic average frequency
    mu_star: Optional[float] = 0.1  # Coulomb pseudopotential
    tc_mcmillan: Optional[float] = None  # McMillan Tc
    tc_allen_dynes: Optional[float] = None  # Allen-Dynes Tc
    
    # Transport
    eph_transport: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'converged': self.converged,
            'lambda_eph': self.lambda_eph,
            'omega_log_meV': self.omega_log,
            'omega_log_K': self.omega_log * 11.605 if self.omega_log else None,
            'mu_star': self.mu_star,
            'tc_mcmillan_K': self.tc_mcmillan,
            'tc_allen_dynes_K': self.tc_allen_dynes
        }
    
    def save(self, filepath: str):
        """Save results to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class EPWInterface:
    """
    Interface to Quantum ESPRESSO EPW package.
    
    Manages the full workflow:
    1. SCF calculation (pw.x)
    2. Phonon calculation (ph.x)
    3. Wannier function generation (wannier90.x)
    4. EPW calculation (epw.x)
    """
    
    def __init__(self, config: Optional[EPWConfig] = None):
        """
        Initialize EPW interface.
        
        Args:
            config: EPW configuration
        """
        self.config = config or EPWConfig()
        self.results: Optional[EPWResults] = None
        
        # Check executables
        self._check_executables()
        
        logger.info("Initialized EPWInterface")
    
    def _check_executables(self):
        """Check if required executables are available."""
        required = ['pw.x', 'ph.x', 'wannier90.x', 'epw.x']
        missing = []
        
        for exe in required:
            path = Path(self.config.espresso_path) / exe
            if not shutil.which(str(path)) and not shutil.which(exe):
                missing.append(exe)
        
        if missing:
            logger.warning(f"Missing executables: {missing}. "
                          "EPW workflow may fail.")
    
    def generate_scf_input(
        self,
        structure: Any,
        pseudopotentials: Dict[str, str],
        ecutwfc: float = 60.0,
        ecutrho: float = 480.0,
        k_mesh: Optional[Tuple[int, int, int]] = None,
        occupations: str = 'smearing',
        smearing: str = 'mp',
        degauss: float = 0.02
    ) -> str:
        """
        Generate SCF input file for pw.x.
        
        Args:
            structure: Pymatgen Structure or ASE Atoms
            pseudopotentials: Dict of {element: pseudo_file}
            ecutwfc: Wavefunction cutoff (Ry)
            ecutrho: Density cutoff (Ry)
            k_mesh: k-point mesh
            occupations: Occupation type
            smearing: Smearing type
            degauss: Smearing width (Ry)
            
        Returns:
            Input file content as string
        """
        # Extract structure data
        if hasattr(structure, 'lattice'):
            # Pymatgen
            cell = structure.lattice.matrix
            positions = structure.cart_coords
            symbols = [str(s) for s in structure.species]
        else:
            # ASE
            cell = structure.get_cell()
            positions = structure.get_positions()
            symbols = structure.get_chemical_symbols()
        
        k_mesh = k_mesh or self.config.k_mesh_coarse
        
        # Generate input
        input_lines = []
        input_lines.append("&CONTROL")
        input_lines.append(f"  calculation = 'scf'")
        input_lines.append(f"  prefix = '{self.config.prefix}'")
        input_lines.append(f"  outdir = '{self.config.outdir}'")
        input_lines.append(f"  pseudo_dir = '{self.config.pseudo_dir}'")
        input_lines.append(f"  tprnfor = .true.")
        input_lines.append(f"  tstress = .true.")
        input_lines.append("/")
        
        input_lines.append("&SYSTEM")
        input_lines.append(f"  ibrav = 0")
        input_lines.append(f"  nat = {len(symbols)}")
        input_lines.append(f"  ntyp = {len(set(symbols))}")
        input_lines.append(f"  ecutwfc = {ecutwfc}")
        input_lines.append(f"  ecutrho = {ecutrho}")
        input_lines.append(f"  occupations = '{occupations}'")
        input_lines.append(f"  smearing = '{smearing}'")
        input_lines.append(f"  degauss = {degauss}")
        input_lines.append("/")
        
        input_lines.append("&ELECTRONS")
        input_lines.append(f"  conv_thr = 1.0d-10")
        input_lines.append("/")
        
        input_lines.append("ATOMIC_SPECIES")
        for elem in set(symbols):
            # Get mass (simplified)
            mass = self._get_atomic_mass(elem)
            input_lines.append(f"{elem} {mass:.2f} {pseudopotentials[elem]}")
        
        input_lines.append("ATOMIC_POSITIONS (angstrom)")
        for symbol, pos in zip(symbols, positions):
            input_lines.append(f"{symbol} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}")
        
        input_lines.append("K_POINTS automatic")
        input_lines.append(f"{k_mesh[0]} {k_mesh[1]} {k_mesh[2]} 0 0 0")
        
        input_lines.append("CELL_PARAMETERS (angstrom)")
        for row in cell:
            input_lines.append(f"  {row[0]:.10f} {row[1]:.10f} {row[2]:.10f}")
        
        return "\n".join(input_lines)
    
    def generate_ph_input(
        self,
        q_mesh: Optional[Tuple[int, int, int]] = None,
        ldisp: bool = True
    ) -> str:
        """
        Generate phonon input file for ph.x.
        
        Args:
            q_mesh: q-point mesh for phonon calculation
            ldisp: Calculate on uniform grid
            
        Returns:
            Input file content as string
        """
        q_mesh = q_mesh or self.config.q_mesh_coarse
        
        input_lines = []
        input_lines.append("Phonon calculation")
        input_lines.append("&INPUTPH")
        input_lines.append(f"  prefix = '{self.config.prefix}'")
        input_lines.append(f"  outdir = '{self.config.outdir}'")
        input_lines.append(f"  ldisp = .{str(ldisp).lower()}.")
        input_lines.append(f"  nq1 = {q_mesh[0]}")
        input_lines.append(f"  nq2 = {q_mesh[1]}")
        input_lines.append(f"  nq3 = {q_mesh[2]}")
        input_lines.append(f"  tr2_ph = 1.0d-16")
        input_lines.append(f"  fildyn = '{self.config.prefix}.dyn'")
        input_lines.append(f"  fildvscf = 'dvscf'")
        input_lines.append("/")
        
        return "\n".join(input_lines)
    
    def generate_epw_input(
        self,
        wannierize: bool = True,
        elph: bool = True,
        epbwrite: bool = True,
        epbread: bool = False
    ) -> str:
        """
        Generate EPW input file.
        
        Args:
            wannierize: Perform Wannierization
            elph: Calculate electron-phonon matrix elements
            epbwrite: Write electron-phonon matrix elements
            epbread: Read pre-computed matrix elements
            
        Returns:
            Input file content as string
        """
        input_lines = []
        input_lines.append("EPW calculation")
        input_lines.append("&INPUTEPW")
        input_lines.append(f"  prefix = '{self.config.prefix}'")
        input_lines.append(f"  outdir = '{self.config.outdir}'")
        input_lines.append(f"  dvscf_dir = '{self.config.outdir}/save'")
        input_lines.append(f"")
        input_lines.append(f"  elph = .{str(elph).lower()}.")
        input_lines.append(f"  epbwrite = .{str(epbwrite).lower()}.")
        input_lines.append(f"  epbread = .{str(epbread).lower()}.")
        input_lines.append(f"  epwwrite = .{str(self.config.ephwrite).lower()}.")
        input_lines.append(f"")
        input_lines.append(f"  wannierize = .{str(wannierize).lower()}.")
        input_lines.append(f"  nbndsub = {self.config.n_wannier or 8}")
        
        if self.config.dis_froz_min is not None:
            input_lines.append(f"  dis_froz_min = {self.config.dis_froz_min}")
        if self.config.dis_froz_max is not None:
            input_lines.append(f"  dis_froz_max = {self.config.dis_froz_max}")
        
        input_lines.append(f"")
        input_lines.append(f"  fsthick = {self.config.fsthick}")
        input_lines.append(f"  degaussw = {self.config.degaussw}")
        input_lines.append(f"")
        input_lines.append(f"  nkf1 = {self.config.k_mesh_fine[0]}")
        input_lines.append(f"  nkf2 = {self.config.k_mesh_fine[1]}")
        input_lines.append(f"  nkf3 = {self.config.k_mesh_fine[2]}")
        input_lines.append(f"  nqf1 = {self.config.q_mesh_fine[0]}")
        input_lines.append(f"  nqf2 = {self.config.q_mesh_fine[1]}")
        input_lines.append(f"  nqf3 = {self.config.q_mesh_fine[2]}")
        input_lines.append(f"")
        
        if self.config.temps:
            input_lines.append(f"  temps = {','.join(map(str, self.config.temps))}")
        
        input_lines.append("/")
        
        return "\n".join(input_lines)
    
    def _get_atomic_mass(self, symbol: str) -> float:
        """Get atomic mass for element."""
        masses = {
            'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811,
            'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
            'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
            'S': 32.065, 'Cl': 35.453, 'Na': 22.99, 'Mg': 24.305, 'Al': 26.982,
            'Si': 28.086, 'P': 30.974, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948,
            'K': 39.098, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867, 'V': 50.942,
            'Cr': 51.996, 'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693,
            'Cu': 63.546, 'Zn': 65.38, 'Ga': 69.723, 'Ge': 72.64, 'As': 74.922,
            'Se': 78.96, 'Br': 79.904, 'Kr': 83.798, 'Rb': 85.468, 'Sr': 87.62,
            'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.96, 'Tc': 98.0,
            'Ru': 101.07, 'Rh': 102.906, 'Pd': 106.42, 'Ag': 107.868, 'Cd': 112.411
        }
        return masses.get(symbol, 50.0)
    
    def read_epw_output(self, filepath: str) -> EPWResults:
        """
        Read EPW output file and extract results.
        
        Args:
            filepath: Path to EPW output file
            
        Returns:
            EPWResults object
        """
        results = EPWResults()
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            # Parse λ
            if 'lambda:' in line.lower() or 'lambda =' in line.lower():
                try:
                    results.lambda_eph = float(line.split('=')[-1].split()[0])
                except (ValueError, IndexError):
                    pass
            
            # Parse ω_log
            if 'omega_log' in line.lower() or 'logarithmic' in line.lower():
                try:
                    results.omega_log = float(
                        [s for s in line.split() if s.replace('.','').isdigit()][0]
                    )
                except (ValueError, IndexError):
                    pass
            
            # Parse Tc
            if 't_c' in line.lower() or 'critical' in line.lower():
                try:
                    tc = float([s for s in line.split() if s.replace('.','').isdigit()][0])
                    if 'mcmillan' in line.lower():
                        results.tc_mcmillan = tc
                    elif 'allen' in line.lower():
                        results.tc_allen_dynes = tc
                except (ValueError, IndexError):
                    pass
        
        results.converged = results.lambda_eph is not None
        
        self.results = results
        return results
    
    def read_a2f(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read Eliashberg spectral function α²F(ω).
        
        Args:
            filepath: Path to a2F file
            
        Returns:
            Tuple of (frequencies, a2f_values)
        """
        data = np.loadtxt(filepath)
        omega = data[:, 0]  # meV or cm^-1
        a2f = data[:, 1]
        
        if self.results:
            self.results.a2f = (omega, a2f)
        
        return omega, a2f
    
    def run_full_workflow(
        self,
        structure: Any,
        pseudopotentials: Dict[str, str],
        run_scf: bool = True,
        run_ph: bool = True,
        run_epw: bool = True
    ) -> EPWResults:
        """
        Run complete EPW workflow.
        
        Args:
            structure: Input structure
            pseudopotentials: Dict of pseudopotential files
            run_scf: Run SCF calculation
            run_ph: Run phonon calculation
            run_epw: Run EPW calculation
            
        Returns:
            EPWResults object
        """
        output_dir = Path(self.config.output_dir)
        
        # Generate input files
        scf_input = self.generate_scf_input(structure, pseudopotentials)
        ph_input = self.generate_ph_input()
        epw_input = self.generate_epw_input()
        
        # Write inputs
        with open(output_dir / 'scf.in', 'w') as f:
            f.write(scf_input)
        with open(output_dir / 'ph.in', 'w') as f:
            f.write(ph_input)
        with open(output_dir / 'epw.in', 'w') as f:
            f.write(epw_input)
        
        logger.info(f"Written input files to {output_dir}")
        
        # Note: Actual execution would require QE installation
        # This is a template workflow
        
        if run_scf:
            logger.info("Run: mpirun -np {nproc} pw.x -in scf.in > scf.out")
        
        if run_ph:
            logger.info("Run: mpirun -np {nproc} ph.x -in ph.in > ph.out")
        
        if run_epw:
            logger.info("Run: mpirun -np {nproc} epw.x -in epw.in > epw.out")
        
        return EPWResults()


def calculate_lambda_from_a2f(
    omega: np.ndarray,
    a2f: np.ndarray,
    mu_star: float = 0.1
) -> Tuple[float, float, float]:
    """
    Calculate electron-phonon coupling constant from α²F(ω).
    
    λ = 2∫(α²F(ω)/ω)dω
    
    Args:
        omega: Frequencies (meV)
        a2f: Spectral function values
        mu_star: Coulomb pseudopotential
        
    Returns:
        Tuple of (lambda, omega_log, Tc_mcmillan)
    """
    # Calculate λ
    integrand = 2 * a2f / (omega + 1e-10)
    lambda_eph = integrate.simpson(integrand, omega)
    
    # Calculate ω_log
    log_integrand = (2 / lambda_eph) * a2f * np.log(omega + 1e-10) / (omega + 1e-10)
    omega_log = np.exp(integrate.simpson(log_integrand, omega))
    
    # McMillan Tc formula
    # Tc = (ω_log / 1.2) * exp(-1.04(1 + λ) / (λ - μ*(1 + 0.62λ)))
    if lambda_eph > mu_star:
        numerator = -1.04 * (1 + lambda_eph)
        denominator = lambda_eph - mu_star * (1 + 0.62 * lambda_eph)
        tc_mcmillan = (omega_log / 1.2) * np.exp(numerator / denominator)
    else:
        tc_mcmillan = 0.0
    
    return lambda_eph, omega_log, tc_mcmillan


# Helper function for integration
from scipy import integrate


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='EPW Interface')
    parser.add_argument('--structure', type=str, help='Structure file')
    parser.add_argument('--pseudo-dir', type=str, default='./pseudo')
    parser.add_argument('--outdir', type=str, default='./epw_output')
    
    args = parser.parse_args()
    
    print("EPW Interface - use within Python for full functionality")
