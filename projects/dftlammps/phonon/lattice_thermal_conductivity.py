"""
Lattice Thermal Conductivity Module
====================================

Calculation of lattice thermal conductivity using Phono3py.

Features:
- Third-order force constants calculation (Phono3py interface)
- Relaxation Time Approximation (RTA) for thermal conductivity
- Lattice thermal conductivity κ_L(T)
- Phonon lifetime and mean free path calculations
- Isotope scattering effects
- Boundary scattering

Author: DFTLammps Phonon Team
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Phono3py imports
try:
    from phono3py import Phono3py
    from phono3py.file_IO import (
        write_fc2_to_hdf5,
        write_fc3_to_hdf5,
        read_fc2_from_hdf5,
        read_fc3_from_hdf5
    )
    from phono3py.phonon3.conductivity_RTA import Conductivity_RTA
    from phono3py.phonon3.conductivity_LBTE import Conductivity_LBTE
    PHONO3PY_AVAILABLE = True
except ImportError:
    PHONO3PY_AVAILABLE = False
    logging.warning("Phono3py not available - thermal conductivity features disabled")

# Phonopy imports
try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    PHONOPY_AVAILABLE = True
except ImportError:
    PHONOPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ConductivityMethod(Enum):
    """Method for calculating thermal conductivity."""
    RTA = auto()
    LBTE = auto()


@dataclass
class ThermalConductivityConfig:
    """Configuration for thermal conductivity calculations."""
    
    mesh: Tuple[int, int, int] = (11, 11, 11)
    method: ConductivityMethod = ConductivityMethod.RTA
    temperature_range: Tuple[float, float, float] = (300.0, 1000.0, 100.0)
    temperatures: Optional[np.ndarray] = None
    sigmas: Optional[List[float]] = None
    sigma_cutoff: float = 3.0
    include_isotope: bool = True
    include_boundary: bool = False
    boundary_size: float = 1e6
    mass_variances: Optional[Dict[str, float]] = None
    output_dir: str = "./thermal_conductivity_output"
    write_fc3: bool = True
    write_kappa: bool = True
    write_lifetime: bool = True
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.temperatures is None:
            t_min, t_max, t_step = self.temperature_range
            self.temperatures = np.arange(t_min, t_max + t_step, t_step)
        if self.sigmas is None:
            self.sigmas = [0.1]


@dataclass
class ThermalConductivityResults:
    """Container for thermal conductivity calculation results."""
    
    temperatures: np.ndarray = field(default_factory=lambda: np.array([]))
    kappa_tensor: np.ndarray = field(default_factory=lambda: np.array([]))
    kappa_scalar: Optional[np.ndarray] = None
    kappa_mode: Optional[np.ndarray] = None
    frequencies: Optional[np.ndarray] = None
    group_velocities: Optional[np.ndarray] = None
    lifetimes: Optional[np.ndarray] = None
    mean_free_paths: Optional[np.ndarray] = None
    mesh: Optional[Tuple[int, int, int]] = None
    n_qpoints: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'temperatures': self.temperatures.tolist(),
            'kappa_tensor_W_m_K': self.kappa_tensor.tolist(),
            'kappa_scalar_W_m_K': self.kappa_scalar.tolist() if self.kappa_scalar is not None else None,
            'mesh': self.mesh,
            'n_qpoints': self.n_qpoints
        }
    
    def save(self, filepath: str):
        ext = Path(filepath).suffix
        if ext == '.json':
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        elif ext == '.npz':
            data = {
                'temperatures': self.temperatures,
                'kappa_tensor': self.kappa_tensor,
            }
            if self.frequencies is not None:
                data['frequencies'] = self.frequencies
            if self.kappa_scalar is not None:
                data['kappa_scalar'] = self.kappa_scalar
            np.savez_compressed(filepath, **data)


class LatticeThermalConductivity:
    """Calculator for lattice thermal conductivity using Phono3py."""
    
    def __init__(self, config: Optional[ThermalConductivityConfig] = None):
        if not PHONO3PY_AVAILABLE:
            raise ImportError("Phono3py is required for thermal conductivity calculations")
        
        self.config = config or ThermalConductivityConfig()
        self.phono3py: Optional[Phono3py] = None
        self.results: Optional[ThermalConductivityResults] = None
        
        logger.info(f"Initialized LatticeThermalConductivity with {self.config.method.name}")
    
    def create_displacements(
        self,
        structure: Union[str, PhonopyAtoms],
        supercell_matrix: Optional[np.ndarray] = None,
        displacement_distance: float = 0.03
    ) -> List[PhonopyAtoms]:
        """Generate displaced supercells for third-order force constants."""
        if isinstance(structure, str):
            from phonopy.file_IO import read_crystal_structure
            cell = read_crystal_structure(structure)[0]
        else:
            cell = structure
        
        supercell_matrix = supercell_matrix or np.diag([2, 2, 2])
        
        self.phono3py = Phono3py(
            cell,
            supercell_matrix=supercell_matrix,
            primitive_matrix='auto',
            mesh=self.config.mesh
        )
        
        self.phono3py.generate_displacements(
            distance=displacement_distance,
            cutoff_pair_distance=None
        )
        
        displacements = self.phono3py.supercells_with_displacements
        
        output_dir = Path(self.config.output_dir)
        self.phono3py.save_displacements(output_dir / 'disp_fc3.yaml')
        
        logger.info(f"Generated {len(displacements)} displacement structures for fc3")
        return displacements
    
    def set_force_constants(
        self,
        fc2: Union[str, np.ndarray],
        fc3: Union[str, np.ndarray],
        structure: Optional[Union[str, PhonopyAtoms]] = None
    ):
        """Set second and third order force constants."""
        if isinstance(fc2, str):
            fc2 = read_fc2_from_hdf5(fc2)
        
        if isinstance(fc3, str):
            fc3 = read_fc3_from_hdf5(fc3)
        
        if self.phono3py is None:
            if structure is None:
                raise ValueError("Structure required if phono3py not initialized")
            self.create_displacements(structure)
        
        self.phono3py.fc2 = fc2
        self.phono3py.fc3 = fc3
        
        if self.config.write_fc3:
            output_dir = Path(self.config.output_dir)
            write_fc2_to_hdf5(fc2, output_dir / 'fc2.hdf5')
            write_fc3_to_hdf5(fc3, output_dir / 'fc3.hdf5')
        
        logger.info(f"Set force constants: fc2 shape {fc2.shape}, fc3 shape {fc3.shape}")
    
    def run_thermal_conductivity_rta(
        self,
        temperatures: Optional[np.ndarray] = None,
        mesh: Optional[Tuple[int, int, int]] = None,
        write_results: bool = True
    ) -> ThermalConductivityResults:
        """Calculate thermal conductivity using RTA method."""
        if self.phono3py is None:
            raise RuntimeError("Phono3py not initialized")
        
        if self.phono3py.fc2 is None or self.phono3py.fc3 is None:
            raise RuntimeError("Force constants not set")
        
        temps = temperatures or self.config.temperatures
        mesh = mesh or self.config.mesh
        
        if self.config.include_isotope:
            self.phono3py.init_phph_interaction(mesh=mesh, is_nosym=False)
            if self.config.mass_variances:
                for symbol, variance in self.config.mass_variances.items():
                    self.phono3py.set_mass_variances(symbol, variance)
        
        self.phono3py.run_thermal_conductivity(
            temperatures=temps,
            write_kappa=write_results,
            output_dir=self.config.output_dir,
            is_LBTE=False
        )
        
        kappa = self.phono3py.thermal_conductivity.kappa
        
        if len(kappa.shape) == 3:
            kappa_tensor = kappa
        else:
            kappa_tensor = kappa.reshape(-1, 3, 3)
        
        kappa_scalar = np.trace(kappa_tensor, axis1=1, axis2=2) / 3.0
        
        mode_kappa = getattr(self.phono3py.thermal_conductivity, 'mode_kappa', None)
        
        self.results = ThermalConductivityResults(
            temperatures=temps,
            kappa_tensor=kappa_tensor,
            kappa_scalar=kappa_scalar,
            kappa_mode=mode_kappa,
            mesh=mesh,
            n_qpoints=np.prod(mesh)
        )
        
        if write_results:
            self._write_results()
        
        logger.info(f"Completed RTA: κ = {kappa_scalar[0]:.2f} W/m/K at {temps[0]:.0f}K")
        return self.results
    
    def run_thermal_conductivity_lbte(
        self,
        temperatures: Optional[np.ndarray] = None,
        mesh: Optional[Tuple[int, int, int]] = None,
        write_results: bool = True
    ) -> ThermalConductivityResults:
        """Calculate thermal conductivity using LBTE method."""
        if self.phono3py is None:
            raise RuntimeError("Phono3py not initialized")
        
        temps = temperatures or self.config.temperatures
        mesh = mesh or self.config.mesh
        
        self.phono3py.run_thermal_conductivity(
            temperatures=temps,
            write_kappa=write_results,
            output_dir=self.config.output_dir,
            is_LBTE=True
        )
        
        kappa = self.phono3py.thermal_conductivity.kappa
        kappa_tensor = kappa if len(kappa.shape) == 3 else kappa.reshape(-1, 3, 3)
        kappa_scalar = np.trace(kappa_tensor, axis1=1, axis2=2) / 3.0
        
        self.results = ThermalConductivityResults(
            temperatures=temps,
            kappa_tensor=kappa_tensor,
            kappa_scalar=kappa_scalar,
            mesh=mesh,
            n_qpoints=np.prod(mesh)
        )
        
        if write_results:
            self._write_results()
        
        logger.info(f"Completed LBTE: κ = {kappa_scalar[0]:.2f} W/m/K at {temps[0]:.0f}K")
        return self.results
    
    def _write_results(self):
        """Write calculation results to files."""
        output_dir = Path(self.config.output_dir)
        
        self.results.save(output_dir / 'kappa_results.npz')
        self.results.save(output_dir / 'kappa_results.json')
        
        if self.config.write_kappa:
            with open(output_dir / 'kappa.dat', 'w') as f:
                f.write("# Temperature (K)  κ_xx  κ_yy  κ_zz  κ_avg (W/m/K)\n")
                for i, T in enumerate(self.results.temperatures):
                    kappa = self.results.kappa_tensor[i]
                    f.write(f"{T:8.2f}  {kappa[0,0]:12.4f}  {kappa[1,1]:12.4f} "
                           f" {kappa[2,2]:12.4f}  {self.results.kappa_scalar[i]:12.4f}\n")
        
        logger.info(f"Wrote results to {output_dir}")
    
    def plot_kappa_vs_temperature(
        self,
        results: Optional[ThermalConductivityResults] = None,
        plot_components: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> Figure:
        """Plot thermal conductivity vs temperature."""
        results = results or self.results
        if results is None:
            raise ValueError("No results available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        temps = results.temperatures
        
        ax.plot(temps, results.kappa_scalar, 'k-', lw=2.5, label='κ_avg')
        
        if plot_components and len(results.kappa_tensor.shape) == 3:
            ax.plot(temps, results.kappa_tensor[:, 0, 0], 'r--', lw=1.5, 
                   label='κ_xx', alpha=0.7)
            ax.plot(temps, results.kappa_tensor[:, 1, 1], 'g--', lw=1.5,
                   label='κ_yy', alpha=0.7)
            ax.plot(temps, results.kappa_tensor[:, 2, 2], 'b--', lw=1.5,
                   label='κ_zz', alpha=0.7)
        
        ax.set_xlabel('Temperature (K)', fontsize=12)
        ax.set_ylabel('Thermal Conductivity (W/m/K)', fontsize=12)
        ax.set_title('Lattice Thermal Conductivity', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved kappa vs T plot to {save_path}")
        
        return fig


def calculate_thermal_conductivity_workflow(
    structure: Union[str, Any],
    fc2_path: str,
    fc3_path: str,
    temperatures: Optional[np.ndarray] = None,
    mesh: Tuple[int, int, int] = (11, 11, 11),
    method: str = 'RTA',
    output_dir: str = './kappa_output'
) -> ThermalConductivityResults:
    """
    Complete workflow for thermal conductivity calculation.
    
    Args:
        structure: Input structure file or object
        fc2_path: Path to second-order force constants (fc2.hdf5)
        fc3_path: Path to third-order force constants (fc3.hdf5)
        temperatures: Temperature array (K)
        mesh: q-point mesh
        method: 'RTA' or 'LBTE'
        output_dir: Output directory
        
    Returns:
        ThermalConductivityResults object
    """
    config = ThermalConductivityConfig(
        mesh=mesh,
        method=ConductivityMethod.RTA if method == 'RTA' else ConductivityMethod.LBTE,
        temperatures=temperatures,
        output_dir=output_dir
    )
    
    calc = LatticeThermalConductivity(config)
    calc.set_force_constants(fc2_path, fc3_path, structure)
    
    if method == 'RTA':
        return calc.run_thermal_conductivity_rta()
    else:
        return calc.run_thermal_conductivity_lbte()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Lattice Thermal Conductivity Calculator')
    parser.add_argument('--fc2', type=str, required=True, help='Path to fc2.hdf5')
    parser.add_argument('--fc3', type=str, required=True, help='Path to fc3.hdf5')
    parser.add_argument('--structure', type=str, required=True, help='Structure file')
    parser.add_argument('--mesh', type=int, nargs=3, default=[11, 11, 11], help='Mesh')
    parser.add_argument('--method', type=str, default='RTA', choices=['RTA', 'LBTE'])
    parser.add_argument('--tmin', type=float, default=300.0)
    parser.add_argument('--tmax', type=float, default=1000.0)
    parser.add_argument('--tstep', type=float, default=100.0)
    parser.add_argument('--outdir', type=str, default='./kappa_output')
    
    args = parser.parse_args()
    
    temps = np.arange(args.tmin, args.tmax + args.tstep, args.tstep)
    
    results = calculate_thermal_conductivity_workflow(
        args.structure,
        args.fc2,
        args.fc3,
        temperatures=temps,
        mesh=tuple(args.mesh),
        method=args.method,
        output_dir=args.outdir
    )
    
    print(f"Thermal conductivity at {temps[0]}K: {results.kappa_scalar[0]:.2f} W/m/K")
