"""
PySCF-QMC Interface Module
============================

Provides interface between PySCF (Python-based Simulations of Chemistry Framework)
and Quantum Monte Carlo (QMC) calculations.

Features:
- Hartree-Fock/DFT wave function preparation
- Gaussian/plane-wave basis set conversion
- Wave function export for QMC codes (CASINO, QWalk, etc.)
- Molecular and periodic systems support

Author: QMC Expert Module
Date: 2026-03-09
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings

try:
    import pyscf
    from pyscf import gto, scf, dft, ao2mo, mcscf, cc, mp
    from pyscf.pbc import gto as pbc_gto
    from pyscf.pbc import scf as pbc_scf
    from pyscf.pbc import dft as pbc_dft
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False
    warnings.warn("PySCF not installed. Some features will be unavailable.")

try:
    from pyscf import lo
    HAS_LO = True
except ImportError:
    HAS_LO = False


@dataclass
class WaveFunctionData:
    """Container for wave function data."""
    mo_coeff: np.ndarray
    mo_energy: np.ndarray
    mo_occ: np.ndarray
    e_tot: float
    nuclear_repulsion: float
    basis_set: str
    n_electrons: int
    n_basis: int
    spin: int
    charge: int
    coordinates: np.ndarray = field(default_factory=lambda: np.array([]))
    atom_symbols: List[str] = field(default_factory=list)
    basis_data: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'mo_coeff': self.mo_coeff.tolist(),
            'mo_energy': self.mo_energy.tolist(),
            'mo_occ': self.mo_occ.tolist(),
            'e_tot': self.e_tot,
            'nuclear_repulsion': self.nuclear_repulsion,
            'basis_set': self.basis_set,
            'n_electrons': self.n_electrons,
            'n_basis': self.n_basis,
            'spin': self.spin,
            'charge': self.charge,
            'coordinates': self.coordinates.tolist(),
            'atom_symbols': self.atom_symbols,
            'basis_data': self.basis_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WaveFunctionData':
        """Create from dictionary."""
        return cls(
            mo_coeff=np.array(data['mo_coeff']),
            mo_energy=np.array(data['mo_energy']),
            mo_occ=np.array(data['mo_occ']),
            e_tot=data['e_tot'],
            nuclear_repulsion=data['nuclear_repulsion'],
            basis_set=data['basis_set'],
            n_electrons=data['n_electrons'],
            n_basis=data['n_basis'],
            spin=data['spin'],
            charge=data['charge'],
            coordinates=np.array(data['coordinates']),
            atom_symbols=data['atom_symbols'],
            basis_data=data.get('basis_data', {})
        )


class PySCFQMCInterface:
    """
    Interface between PySCF and QMC calculations.
    
    This class handles:
    1. Building molecular/periodic systems in PySCF
    2. Running HF/DFT calculations
    3. Preparing wave functions for QMC
    4. Exporting to various QMC formats
    """
    
    def __init__(self, 
                 atom_symbols: List[str],
                 coordinates: np.ndarray,
                 basis: str = 'cc-pVTZ',
                 charge: int = 0,
                 spin: int = 0,
                 unit: str = 'Angstrom',
                 periodic: bool = False,
                 cell: Optional[np.ndarray] = None):
        """
        Initialize PySCF-QMC interface.
        
        Parameters:
        -----------
        atom_symbols : List[str]
            List of atomic symbols
        coordinates : np.ndarray
            Atomic coordinates (N, 3)
        basis : str
            Basis set name (default: 'cc-pVTZ')
        charge : int
            Total charge (default: 0)
        spin : int
            Spin multiplicity (2S+1) or number of unpaired electrons
        unit : str
            Unit for coordinates ('Angstrom' or 'Bohr')
        periodic : bool
            Whether to use periodic boundary conditions
        cell : Optional[np.ndarray]
            Unit cell vectors for periodic systems (3, 3)
        """
        if not HAS_PYSCF:
            raise ImportError("PySCF is required for this interface.")
        
        self.atom_symbols = atom_symbols
        self.coordinates = np.array(coordinates)
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.unit = unit
        self.periodic = periodic
        self.cell = cell
        
        self.mol = None
        self.mf = None
        self.wf_data = None
        
        self._build_molecule()
    
    def _build_molecule(self):
        """Build PySCF molecule or cell object."""
        if self.periodic and self.cell is not None:
            # Periodic system
            self.mol = pbc_gto.Cell()
            self.mol.atom = [[sym, coord] for sym, coord in 
                            zip(self.atom_symbols, self.coordinates)]
            self.mol.a = self.cell
            self.mol.basis = self.basis
            self.mol.charge = self.charge
            self.mol.spin = self.spin
            self.mol.unit = self.unit
            self.mol.verbose = 3
            self.mol.build()
        else:
            # Molecular system
            self.mol = gto.Mole()
            self.mol.atom = [[sym, coord] for sym, coord in 
                            zip(self.atom_symbols, self.coordinates)]
            self.mol.basis = self.basis
            self.mol.charge = self.charge
            self.mol.spin = self.spin
            self.mol.unit = self.unit
            self.mol.verbose = 3
            self.mol.build()
    
    def run_hf(self, 
               conv_tol: float = 1e-10,
               max_cycle: int = 100) -> Dict:
        """
        Run Hartree-Fock calculation.
        
        Parameters:
        -----------
        conv_tol : float
            Convergence tolerance
        max_cycle : int
            Maximum number of SCF cycles
            
        Returns:
        --------
        Dict with calculation results
        """
        if self.periodic:
            if self.spin == 0:
                self.mf = pbc_scf.KRHF(self.mol, kpts=self.mol.make_kpts([1, 1, 1]))
            else:
                self.mf = pbc_scf.KUHF(self.mol, kpts=self.mol.make_kpts([1, 1, 1]))
        else:
            if self.spin == 0:
                self.mf = scf.RHF(self.mol)
            else:
                self.mf = scf.UHF(self.mol)
        
        self.mf.conv_tol = conv_tol
        self.mf.max_cycle = max_cycle
        
        e_tot = self.mf.kernel()
        
        self._extract_wavefunction()
        
        return {
            'energy': e_tot,
            'converged': self.mf.converged,
            'mo_energy': self.mf.mo_energy,
            'mo_occ': self.mf.mo_occ
        }
    
    def run_dft(self,
                xc: str = 'PBE',
                conv_tol: float = 1e-10,
                max_cycle: int = 100,
                grids_level: int = 3) -> Dict:
        """
        Run DFT calculation.
        
        Parameters:
        -----------
        xc : str
            Exchange-correlation functional
        conv_tol : float
            Convergence tolerance
        max_cycle : int
            Maximum SCF cycles
        grids_level : int
            Grid level for numerical integration
            
        Returns:
        --------
        Dict with calculation results
        """
        if self.periodic:
            if self.spin == 0:
                self.mf = pbc_dft.KRKS(self.mol, kpts=self.mol.make_kpts([1, 1, 1]))
            else:
                self.mf = pbc_dft.KUKS(self.mol, kpts=self.mol.make_kpts([1, 1, 1]))
        else:
            if self.spin == 0:
                self.mf = dft.RKS(self.mol)
            else:
                self.mf = dft.UKS(self.mol)
        
        self.mf.xc = xc
        self.mf.conv_tol = conv_tol
        self.mf.max_cycle = max_cycle
        self.mf.grids.level = grids_level
        
        e_tot = self.mf.kernel()
        
        self._extract_wavefunction()
        
        return {
            'energy': e_tot,
            'converged': self.mf.converged,
            'mo_energy': self.mf.mo_energy,
            'mo_occ': self.mf.mo_occ,
            'xc': xc
        }
    
    def run_casscf(self,
                   ncas: int,
                   nelecas: int,
                   conv_tol: float = 1e-8) -> Dict:
        """
        Run CASSCF for multireference systems.
        
        Parameters:
        -----------
        ncas : int
            Number of active orbitals
        nelecas : int
            Number of active electrons
        conv_tol : float
            Convergence tolerance
            
        Returns:
        --------
        Dict with calculation results
        """
        if self.mf is None:
            self.run_hf()
        
        mc = mcscf.CASSCF(self.mf, ncas, nelecas)
        mc.conv_tol = conv_tol
        e_tot = mc.kernel()
        
        # Update wave function with CASSCF orbitals
        self.mf.mo_coeff = mc.mo_coeff
        self._extract_wavefunction()
        
        return {
            'energy': e_tot,
            'converged': mc.converged,
            'ci_coeffs': mc.ci
        }
    
    def run_mp2(self) -> Dict:
        """Run MP2 calculation for correlation energy."""
        if self.mf is None:
            self.run_hf()
        
        mp2 = mp.MP2(self.mf)
        e_corr = mp2.kernel()
        
        return {
            'e_hf': self.mf.e_tot,
            'e_corr': e_corr,
            'e_tot': self.mf.e_tot + e_corr
        }
    
    def run_ccsd(self, 
                 with_t: bool = True) -> Dict:
        """
        Run CCSD(T) calculation.
        
        Parameters:
        -----------
        with_t : bool
            Include perturbative triples (T)
        """
        if self.mf is None:
            self.run_hf()
        
        mycc = cc.CCSD(self.mf)
        mycc.kernel()
        
        result = {
            'e_hf': self.mf.e_tot,
            'e_ccsd': mycc.e_tot,
            'converged': mycc.converged
        }
        
        if with_t:
            e_t = mycc.ccsd_t()
            result['e_(t)'] = e_t
            result['e_tot'] = mycc.e_tot + e_t
        
        return result
    
    def _extract_wavefunction(self):
        """Extract wave function data from SCF calculation."""
        if self.mf is None:
            raise ValueError("No SCF calculation has been run.")
        
        self.wf_data = WaveFunctionData(
            mo_coeff=self.mf.mo_coeff,
            mo_energy=self.mf.mo_energy if hasattr(self.mf, 'mo_energy') else np.array([]),
            mo_occ=self.mf.mo_occ if hasattr(self.mf, 'mo_occ') else np.array([]),
            e_tot=self.mf.e_tot,
            nuclear_repulsion=self.mol.energy_nuc(),
            basis_set=self.basis,
            n_electrons=self.mol.nelectron,
            n_basis=self.mol.nao,
            spin=self.spin,
            charge=self.charge,
            coordinates=self.coordinates,
            atom_symbols=self.atom_symbols,
            basis_data=self._get_basis_info()
        )
    
    def _get_basis_info(self) -> Dict:
        """Extract basis set information."""
        basis_info = {}
        
        for i, symbol in enumerate(self.atom_symbols):
            if hasattr(self.mol, '_basis'):
                basis_info[symbol] = self.mol._basis.get(symbol, {})
        
        return basis_info
    
    def get_jastrow_factor(self, 
                          order: int = 2,
                          params: Optional[Dict] = None) -> Callable:
        """
        Generate Jastrow factor for QMC.
        
        Parameters:
        -----------
        order : int
            Order of Jastrow factor (1, 2, or 3)
        params : Optional[Dict]
            Jastrow parameters
            
        Returns:
        --------
        Callable that computes Jastrow factor
        """
        if params is None:
            # Default parameters for simple Jastrow
            params = {
                'u': [0.5, 1.0, 0.1],  # Electron-electron
                'chi': [0.1, 0.5],      # Electron-nucleus
                'f': [0.01]             # Electron-electron-nucleus
            }
        
        def jastrow_1b(coords: np.ndarray) -> float:
            """One-body Jastrow (electron-nucleus)."""
            u = 0.0
            for i, (sym, coord) in enumerate(zip(self.atom_symbols, self.coordinates)):
                r = np.linalg.norm(coords - coord, axis=1)
                for k, coeff in enumerate(params['chi'][:order]):
                    u += coeff * r ** (k + 1)
            return np.exp(-u)
        
        def jastrow_2b(coords: np.ndarray) -> float:
            """Two-body Jastrow (electron-electron)."""
            n_elec = len(coords)
            u = 0.0
            for i in range(n_elec):
                for j in range(i + 1, n_elec):
                    r_ij = np.linalg.norm(coords[i] - coords[j])
                    for k, coeff in enumerate(params['u'][:order]):
                        u += coeff * r_ij ** (k + 1)
            return np.exp(-u)
        
        def jastrow(coords: np.ndarray) -> float:
            """Full Jastrow factor."""
            if order >= 1:
                u = jastrow_1b(coords)
            else:
                u = 1.0
            
            if order >= 2:
                u *= jastrow_2b(coords)
            
            return u
        
        return jastrow
    
    def export_to_casino(self, 
                        output_dir: str,
                        jastrow_order: int = 2) -> str:
        """
        Export wave function to CASINO format.
        
        Parameters:
        -----------
        output_dir : str
            Output directory for CASINO files
        jastrow_order : int
            Order of Jastrow factor
            
        Returns:
        --------
        Path to output directory
        """
        if self.wf_data is None:
            raise ValueError("No wave function data available. Run HF or DFT first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Write input file
        input_file = output_path / 'input'
        with open(input_file, 'w') as f:
            f.write("# CASINO input file generated by PySCF-QMC Interface\n")
            f.write(f"title: PySCF {self.basis} calculation\n")
            f.write(f"basis: {self.basis}\n")
            f.write(f"neu: {(self.wf_data.n_electrons + self.spin) // 2}\n")
            f.write(f"ned: {(self.wf_data.n_electrons - self.spin) // 2}\n")
            f.write("runtype: vmc\n")
            f.write("vmc_method: 1\n")
        
        # Write correlation.data (Jastrow parameters)
        corr_file = output_path / 'correlation.data'
        with open(corr_file, 'w') as f:
            f.write("# Jastrow factor parameters\n")
            f.write(f"START JASTROW\n")
            f.write(f"Title: Simple Jastrow\n")
            f.write(f"Truncation order: {jastrow_order}\n")
            f.write(f"END JASTROW\n")
        
        # Write gwfn.data (Gaussian wave function)
        gwfn_file = output_path / 'gwfn.data'
        with open(gwfn_file, 'w') as f:
            f.write("# Gaussian wave function data\n")
            f.write(f"Number of atoms: {len(self.atom_symbols)}\n")
            f.write(f"Number of basis functions: {self.wf_data.n_basis}\n")
            f.write(f"Number of MOs: {self.wf_data.mo_coeff.shape[-1]}\n")
            
            # Write geometry
            f.write("\nGeometry (Bohr):\n")
            for sym, coord in zip(self.atom_symbols, self.coordinates):
                f.write(f"{sym:2s} {coord[0]:15.8f} {coord[1]:15.8f} {coord[2]:15.8f}\n")
            
            # Write MO coefficients
            f.write("\nMO Coefficients:\n")
            np.savetxt(f, self.wf_data.mo_coeff, fmt='%15.8e')
        
        return str(output_path)
    
    def export_to_qwalk(self,
                       output_dir: str,
                       method: str = 'slater_jastrow') -> str:
        """
        Export wave function to QWalk format.
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        method : str
            QMC method ('slater_jastrow', 'multidet', etc.)
        """
        if self.wf_data is None:
            raise ValueError("No wave function data available.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Write system file
        sys_file = output_path / 'system.json'
        system_data = {
            'atoms': [
                {'symbol': sym, 'position': pos.tolist()}
                for sym, pos in zip(self.atom_symbols, self.coordinates)
            ],
            'basis': self.basis,
            'charge': self.charge,
            'spin': self.spin
        }
        with open(sys_file, 'w') as f:
            json.dump(system_data, f, indent=2)
        
        # Write wave function file
        wf_file = output_path / 'wavefunction.json'
        wf_export = self.wf_data.to_dict()
        wf_export['method'] = method
        with open(wf_file, 'w') as f:
            json.dump(wf_export, f, indent=2)
        
        return str(output_path)
    
    def get_slater_determinant(self) -> Callable:
        """
        Get Slater determinant wave function.
        
        Returns:
        --------
        Callable that evaluates Slater determinant
        """
        if self.wf_data is None:
            raise ValueError("No wave function data available.")
        
        mo_coeff = self.wf_data.mo_coeff
        n_up = (self.wf_data.n_electrons + self.spin) // 2
        n_dn = (self.wf_data.n_electrons - self.spin) // 2
        
        # Get occupied MOs
        if mo_coeff.ndim == 3:  # UHF case
            mo_up = mo_coeff[0][:, :n_up]
            mo_dn = mo_coeff[1][:, :n_dn]
        else:  # RHF case
            mo_up = mo_coeff[:, :n_up]
            mo_dn = mo_coeff[:, :n_dn]
        
        def eval_basis(coords: np.ndarray) -> np.ndarray:
            """Evaluate basis functions at given coordinates."""
            if not HAS_PYSCF:
                raise ImportError("PySCF required for basis evaluation.")
            
            # Build ao_value matrix
            if coords.ndim == 1:
                coords = coords.reshape(1, -1)
            
            ao_value = self.mol.eval_gto('GTOval', coords)
            return ao_value
        
        def slater_det(coords_up: np.ndarray, coords_dn: np.ndarray) -> float:
            """
            Evaluate Slater determinant.
            
            Parameters:
            -----------
            coords_up : np.ndarray
                Spin-up electron coordinates (n_up, 3)
            coords_dn : np.ndarray
                Spin-down electron coordinates (n_dn, 3)
            """
            # Evaluate basis functions
            ao_up = eval_basis(coords_up)
            ao_dn = eval_basis(coords_dn)
            
            # Build Slater matrices
            D_up = ao_up @ mo_up
            D_dn = ao_dn @ mo_dn
            
            # Compute determinants
            det_up = np.linalg.det(D_up)
            det_dn = np.linalg.det(D_dn)
            
            return det_up * det_dn
        
        return slater_det
    
    def localize_orbitals(self, 
                         method: str = 'boys') -> np.ndarray:
        """
        Localize molecular orbitals for better QMC convergence.
        
        Parameters:
        -----------
        method : str
            Localization method ('boys', 'pipek-mezey', 'edminston-ruedenberg')
            
        Returns:
        --------
        Localized MO coefficients
        """
        if not HAS_LO:
            raise ImportError("PySCF localization module not available.")
        
        if self.mf is None:
            raise ValueError("Run HF/DFT first.")
        
        if method.lower() == 'boys':
            loc = lo.Boys(self.mol, self.mf.mo_coeff[:, self.mf.mo_occ > 0])
        elif method.lower() == 'pipek-mezey':
            loc = lo.PipekMezey(self.mol, self.mf.mo_coeff[:, self.mf.mo_occ > 0])
        elif method.lower() == 'ibo':
            loc = lo.ibo.ibo(self.mol, self.mf.mo_coeff[:, self.mf.mo_occ > 0])
        else:
            raise ValueError(f"Unknown localization method: {method}")
        
        mo_loc = loc.kernel()
        
        # Update wave function data
        n_occ = np.sum(self.mf.mo_occ > 0)
        self.mf.mo_coeff[:, :n_occ] = mo_loc
        self._extract_wavefunction()
        
        return mo_loc
    
    def compute_trial_energy(self) -> Dict:
        """
        Compute trial energy components for DMC.
        
        Returns:
        --------
        Dict with energy components
        """
        if self.wf_data is None:
            raise ValueError("No wave function available.")
        
        return {
            'e_total': self.wf_data.e_tot,
            'e_nuclear': self.wf_data.nuclear_repulsion,
            'e_electronic': self.wf_data.e_tot - self.wf_data.nuclear_repulsion,
            'kinetic': None,  # Would need explicit calculation
            'potential': None
        }
    
    def get_basis_transformation(self, 
                                 target_basis: str) -> np.ndarray:
        """
        Get transformation matrix to convert between basis sets.
        
        Parameters:
        -----------
        target_basis : str
            Target basis set name
            
        Returns:
        --------
        Transformation matrix
        """
        # This would require overlap calculation between basis sets
        # For now, return identity
        warnings.warn("Basis transformation not fully implemented.")
        return np.eye(self.wf_data.n_basis if self.wf_data else 1)


def create_molecule_from_xyz(xyz_file: str,
                            basis: str = 'cc-pVTZ',
                            charge: int = 0,
                            spin: int = 0) -> PySCFQMCInterface:
    """
    Create PySCF-QMC interface from XYZ file.
    
    Parameters:
    -----------
    xyz_file : str
        Path to XYZ file
    basis : str
        Basis set name
    charge : int
        Total charge
    spin : int
        Spin multiplicity
    """
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    n_atoms = int(lines[0].strip())
    # Skip comment line and read coordinates
    atom_symbols = []
    coordinates = []
    
    for line in lines[2:2+n_atoms]:
        parts = line.split()
        atom_symbols.append(parts[0])
        coordinates.append([float(x) for x in parts[1:4]])
    
    return PySCFQMCInterface(
        atom_symbols=atom_symbols,
        coordinates=np.array(coordinates),
        basis=basis,
        charge=charge,
        spin=spin
    )


# Example usage
if __name__ == "__main__":
    # Example: H2O molecule
    atom_symbols = ['O', 'H', 'H']
    coordinates = np.array([
        [0.0, 0.0, 0.0],
        [0.757, 0.586, 0.0],
        [-0.757, 0.586, 0.0]
    ])
    
    # Create interface
    qmc = PySCFQMCInterface(
        atom_symbols=atom_symbols,
        coordinates=coordinates,
        basis='cc-pVTZ',
        charge=0,
        spin=0
    )
    
    # Run HF
    hf_result = qmc.run_hf()
    print(f"HF Energy: {hf_result['energy']:.6f} Ha")
    
    # Run DFT
    dft_result = qmc.run_dft(xc='PBE')
    print(f"PBE Energy: {dft_result['energy']:.6f} Ha")
    
    # Export to CASINO
    casino_dir = qmc.export_to_casino('h2o_casino')
    print(f"Exported to: {casino_dir}")
    
    # Export to QWalk
    qwalk_dir = qmc.export_to_qwalk('h2o_qwalk')
    print(f"Exported to: {qwalk_dir}")
