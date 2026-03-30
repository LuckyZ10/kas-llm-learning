"""
Proton Transfer with Nuclear Quantum Effects
=============================================

This example demonstrates the use of PIMD and RPMD to study
proton transfer reactions with nuclear quantum effects.

Proton transfer is a classic example where quantum effects are crucial:
- Zero-point energy affects proton transfer barriers
- Tunneling can dominate at low temperatures
- Isotope effects (H vs D) are significant

Systems:
--------
- Proton transfer in water (Grotthuss mechanism)
- Proton conduction in solid acids
- Proton transfer in enzymes
- Hydrogen-bonded systems

Key observables:
----------------
- Proton transfer rates
- Kinetic isotope effects (KIE)
- Reaction mechanism (concerted vs stepwise)
- Quantum free energy profiles

Example:
--------
    from dftlammps.pimd import PIMDSimulation, IPIConfig
    from dftlammps.examples.proton_transfer import setup_proton_transfer_system
    
    # Setup system
    atoms = setup_proton_transfer_system('water_cluster')
    
    # Run PIMD
    config = IPIConfig(n_beads=32, temperature=300)
    sim = PIMDSimulation(config)
    results = sim.run(atoms)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def setup_water_cluster(n_molecules: int = 3, 
                       with_excess_proton: bool = True) -> Any:
    """Setup a water cluster with optional excess proton.
    
    This creates a protonated water cluster (H+(H2O)n) which is
    a model system for studying proton transfer.
    
    Args:
        n_molecules: Number of water molecules
        with_excess_proton: Whether to add excess proton
        
    Returns:
        ASE Atoms object
    """
    try:
        from ase import Atoms
        from ase.build import molecule
        from ase.geometry import get_distances
    except ImportError:
        raise ImportError("ASE required for setup_water_cluster")
    
    # Build water cluster
    # Start with one water molecule
    water = molecule('H2O')
    
    positions = []
    symbols = []
    
    # Create cluster by replicating and positioning
    for i in range(n_molecules):
        offset = np.array([i * 3.0, 0, 0])
        for pos, sym in zip(water.get_positions(), water.get_chemical_symbols()):
            positions.append(pos + offset)
            symbols.append(sym)
    
    # Add excess proton if requested
    if with_excess_proton:
        # Place proton near first oxygen
        proton_pos = np.array([0.5, 0.5, 0.5])
        positions.append(proton_pos)
        symbols.append('H')
    
    atoms = Atoms(symbols=symbols, positions=positions)
    
    # Center and add cell
    atoms.center(vacuum=5.0)
    
    logger.info(f"Created water cluster: {len(atoms)} atoms")
    
    return atoms


def setup_proton_conduction_system(system_type: str = 'nafion') -> Any:
    """Setup a proton conduction system.
    
    Args:
        system_type: Type of system ('nafion', 'csHSO4', 'water')
        
    Returns:
        ASE Atoms object
    """
    try:
        from ase import Atoms
    except ImportError:
        raise ImportError("ASE required")
    
    if system_type == 'water':
        return setup_water_cluster(n_molecules=8)
    
    elif system_type == 'csHSO4':
        # Cesium hydrogen sulfate - solid acid proton conductor
        # Simplified model with just the key HSO4 units
        symbols = ['Cs', 'H', 'S', 'O', 'O', 'O', 'O']
        positions = [
            [0.0, 0.0, 0.0],
            [2.5, 0.5, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.4, 0.0],
            [2.0, -1.4, 0.0],
            [3.4, 0.0, 0.0],
            [1.0, 0.0, 1.0]
        ]
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.center(vacuum=5.0)
        return atoms
    
    else:
        raise ValueError(f"Unknown system type: {system_type}")


class ProtonTransferAnalyzer:
    """Analyzer for proton transfer reactions.
    
    Provides methods to:
    - Identify proton transfer events
    - Calculate transfer rates
    - Analyze Grotthuss mechanism
    - Compute kinetic isotope effects
    
    Attributes:
        trajectory: Proton trajectory
        oxygen_positions: Oxygen atom positions over time
        proton_indices: Indices of mobile protons
    """
    
    def __init__(self, positions: np.ndarray, 
                 symbols: List[str],
                 cell: Optional[np.ndarray] = None):
        """Initialize analyzer.
        
        Args:
            positions: Trajectory [n_frames, n_atoms, 3]
            symbols: Atom symbols
            cell: Simulation cell
        """
        self.positions = positions
        self.symbols = symbols
        self.cell = cell
        
        # Identify atoms
        self.oxygen_indices = [i for i, s in enumerate(symbols) if s == 'O']
        self.hydrogen_indices = [i for i, s in enumerate(symbols) if s == 'H']
        
        logger.info(f"Found {len(self.oxygen_indices)} oxygens, "
                   f"{len(self.hydrogen_indices)} hydrogens")
    
    def identify_proton_transfer(self, 
                                  donor_threshold: float = 1.2,
                                  acceptor_threshold: float = 1.2) -> List[Dict]:
        """Identify proton transfer events.
        
        A proton transfer occurs when a hydrogen changes its closest
        oxygen from one frame to the next.
        
        Args:
            donor_threshold: Maximum O-H distance for donor
            acceptor_threshold: Maximum O-H distance for acceptor
            
        Returns:
            List of transfer events
        """
        transfers = []
        n_frames = len(self.positions)
        
        for h_idx in self.hydrogen_indices:
            closest_o = []
            
            for frame in range(n_frames):
                h_pos = self.positions[frame, h_idx]
                
                # Find closest oxygen
                min_dist = float('inf')
                closest_oxygen = -1
                
                for o_idx in self.oxygen_indices:
                    o_pos = self.positions[frame, o_idx]
                    dist = np.linalg.norm(h_pos - o_pos)
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_oxygen = o_idx
                
                closest_o.append(closest_oxygen)
            
            # Detect transfers (change in closest oxygen)
            for i in range(len(closest_o) - 1):
                if closest_o[i] != closest_o[i + 1] and closest_o[i] != -1:
                    transfers.append({
                        'proton': h_idx,
                        'frame': i,
                        'donor': closest_o[i],
                        'acceptor': closest_o[i + 1]
                    })
        
        logger.info(f"Identified {len(transfers)} proton transfers")
        return transfers
    
    def calculate_transfer_rate(self, timestep: float,
                                transfers: Optional[List[Dict]] = None) -> float:
        """Calculate proton transfer rate.
        
        Args:
            timestep: Time step in femtoseconds
            transfers: Pre-computed transfer events (optional)
            
        Returns:
            Transfer rate in Hz
        """
        if transfers is None:
            transfers = self.identify_proton_transfer()
        
        if len(transfers) == 0:
            return 0.0
        
        total_time = len(self.positions) * timestep * 1e-15  # to seconds
        rate = len(transfers) / total_time
        
        return rate
    
    def analyze_grotthuss_mechanism(self) -> Dict:
        """Analyze Grotthuss proton hopping mechanism.
        
        The Grotthuss mechanism involves:
        1. Proton transfer along hydrogen bond
        2. Reorientation of water molecules
        3. Continued hopping
        
        Returns:
            Analysis dictionary
        """
        transfers = self.identify_proton_transfer()
        
        # Analyze transfer patterns
        consecutive_transfers = 0
        for i in range(len(transfers) - 1):
            # Check if transfers are consecutive (same proton or correlated)
            if transfers[i]['frame'] + 1 == transfers[i + 1]['frame']:
                consecutive_transfers += 1
        
        return {
            'total_transfers': len(transfers),
            'consecutive_transfers': consecutive_transfers,
            'grotthuss_efficiency': consecutive_transfers / len(transfers) if transfers else 0
        }
    
    def calculate_quantum_kie(self,
                             positions_h: np.ndarray,
                             positions_d: np.ndarray,
                             temperature: float) -> float:
        """Calculate kinetic isotope effect (H vs D).
        
        KIE = k_H / k_D
        
        For proton transfer, KIE is typically 2-10 at room temperature,
        indicating significant quantum tunneling.
        
        Args:
            positions_h: Positions for H system
            positions_d: Positions for D system
            temperature: Temperature in K
            
        Returns:
            Kinetic isotope effect
        """
        # Analyze both systems
        analyzer_h = ProtonTransferAnalyzer(positions_h, ['H'] * positions_h.shape[1])
        analyzer_d = ProtonTransferAnalyzer(positions_d, ['D'] * positions_d.shape[1])
        
        transfers_h = analyzer_h.identify_proton_transfer()
        transfers_d = analyzer_d.identify_proton_transfer()
        
        rate_h = len(transfers_h)
        rate_d = len(transfers_d)
        
        if rate_d == 0:
            return float('inf')
        
        kie = rate_h / rate_d
        
        return kie


def run_proton_transfer_pimd(atoms: Any,
                             n_beads: int = 32,
                             temperature: float = 300.0,
                             n_steps: int = 10000,
                             timestep: float = 0.5) -> Dict:
    """Run PIMD simulation for proton transfer.
    
    Args:
        atoms: Initial atomic configuration
        n_beads: Number of path integral beads
        temperature: Temperature in K
        n_steps: Number of MD steps
        timestep: Time step in fs
        
    Returns:
        Results dictionary
    """
    from dftlammps.pimd import IPIConfig, PIMDSimulation
    
    config = IPIConfig(
        n_beads=n_beads,
        temperature=temperature,
        timestep=timestep,
        n_steps=n_steps,
        trajectory_freq=10
    )
    
    sim = PIMDSimulation(config)
    
    # In real implementation, would need driver for forces
    # For now, return configuration
    results = {
        'config': config,
        'atoms': atoms,
        'simulation': sim
    }
    
    logger.info(f"Setup PIMD for proton transfer: P={n_beads}, T={temperature}K")
    
    return results


def analyze_proton_transfer_results(positions: np.ndarray,
                                    symbols: List[str],
                                    timestep: float = 0.5) -> Dict:
    """Analyze proton transfer from PIMD/RPMD results.
    
    Args:
        positions: Trajectory positions
        symbols: Atom symbols
        timestep: Time step in fs
        
    Returns:
        Analysis results
    """
    analyzer = ProtonTransferAnalyzer(positions, symbols)
    
    transfers = analyzer.identify_proton_transfer()
    rate = analyzer.calculate_transfer_rate(timestep, transfers)
    grotthuss = analyzer.analyze_grotthuss_mechanism()
    
    return {
        'n_transfers': len(transfers),
        'transfer_rate': rate,
        'grotthuss_analysis': grotthuss
    }


# Example usage demonstration
def example_proton_transfer():
    """Example workflow for proton transfer simulation."""
    
    print("=" * 60)
    print("PROTON TRANSFER WITH NUCLEAR QUANTUM EFFECTS")
    print("=" * 60)
    
    # Setup system
    atoms = setup_water_cluster(n_molecules=3, with_excess_proton=True)
    print(f"System: {len(atoms)} atoms")
    
    # Classical vs Quantum comparison
    print("\n1. Classical MD (P=1 bead)")
    print("   - No quantum effects")
    print("   - Faster but misses ZPE and tunneling")
    
    print("\n2. PIMD (P=32 beads)")
    print("   - Includes quantum statistics")
    print("   - Correct ZPE and thermodynamics")
    
    print("\n3. RPMD (P=32 beads)")
    print("   - Approximate quantum dynamics")
    print("   - Can compute rates with tunneling")
    
    print("\nKey observables:")
    print("   - Proton transfer rate")
    print("   - Kinetic isotope effect (KIE)")
    print("   - Grotthuss mechanism efficiency")
    
    print("=" * 60)
    
    return atoms


if __name__ == "__main__":
    example_proton_transfer()
