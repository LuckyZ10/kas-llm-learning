"""
case_molecular_junction.py

Molecular Electronic Devices - Application Case Study

This module demonstrates quantum transport calculations for molecular
junctions, including:
- Benzene dithiol (BDT) molecular junctions
- Oligoacene molecular wires
- Redox-active molecular switches
- Single-molecule conductance

References:
- Venkataraman et al., Nature 442, 904 (2006) - BDT junctions
- Lindsay & Ratner, Adv. Mater. 19, 23 (2007) - Molecular electronics review
- Su et al., Nature Rev. Mater. 1, 16002 (2016) - Single-molecule switches
"""

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# Import from quantum_transport module
import sys
sys.path.insert(0, '/root/.openclaw/workspace/dftlammps')

from quantum_transport.transiesta_interface import (
    ElectrodeConfig, ElectrodeType, TransportStructure,
    NEGFSystem, IVCalculator, TransmissionAnalyzer
)
from quantum_transport.negf_formalism import (
    SelfEnergy, NEGFSystemAdvanced, LandauerButtiker
)


class MolecularHamiltonian:
    """
    Construct tight-binding Hamiltonian for molecular systems.
    
    Uses Hückel approximation for π-conjugated systems.
    """
    
    def __init__(self, molecule_type: str = "BDT"):
        self.molecule_type = molecule_type
        
        # Hückel parameters (eV)
        self.alpha_C = 0.0  # On-site for carbon
        self.beta_CC = -2.5  # C-C hopping
        self.beta_CS = -1.5  # C-S hopping
        self.alpha_S = -0.5  # On-site for sulfur
    
    def build_benzene_dithiol(self, with_anchors: bool = True) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Build Hückel Hamiltonian for benzene dithiol (BDT).
        
        Structure: S-C6H4-S with thiol anchors
        """
        # 6 carbon ring + 2 sulfur anchors
        num_atoms = 8 if with_anchors else 6
        
        # Connectivity (atom indices)
        # Ring: 0-1-2-3-4-5-0
        # Anchors: 6-0, 7-3
        bonds = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),  # Ring
        ]
        
        if with_anchors:
            bonds.extend([(6, 0), (7, 3)])  # Anchors
        
        # Build Hamiltonian
        H = np.zeros((num_atoms, num_atoms))
        
        # On-site energies
        for i in range(6):
            H[i, i] = self.alpha_C  # Carbon
        
        if with_anchors:
            H[6, 6] = self.alpha_S  # Sulfur
            H[7, 7] = self.alpha_S
        
        # Hopping
        for i, j in bonds:
            if i < 6 and j < 6:
                H[i, j] = self.beta_CC
                H[j, i] = self.beta_CC
            else:
                H[i, j] = self.beta_CS
                H[j, i] = self.beta_CS
        
        return H, bonds
    
    def build_oligoacene(self, n_rings: int = 3) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Build Hückel Hamiltonian for oligoacene molecular wire.
        
        n_rings: Number of benzene rings (1 = naphthalene, 2 = anthracene, etc.)
        """
        num_carbons = 4 * n_rings + 2
        
        # Linear acene structure
        bonds = []
        
        # Build connectivity for linear acenes
        for ring in range(n_rings):
            base = ring * 4
            # Each ring connects to the next
            bonds.extend([
                (base, base + 1),
                (base + 1, base + 2),
                (base + 2, base + 3),
                (base + 3, base),
            ])
            
            # Connect to next ring
            if ring < n_rings - 1:
                bonds.append((base + 2, base + 4))
        
        # Hamiltonian
        H = np.zeros((num_carbons, num_carbons))
        
        for i in range(num_carbons):
            H[i, i] = self.alpha_C
        
        for i, j in bonds:
            H[i, j] = self.beta_CC
            H[j, i] = self.beta_CC
        
        return H, bonds
    
    def add_electrode_coupling(self, H_mol: np.ndarray,
                               coupling_strength: float = -0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add coupling to gold electrodes via sulfur anchors.
        
        Returns extended Hamiltonian including electrode surface atoms.
        """
        n_mol = H_mol.shape[0]
        n_elec = 2  # One surface atom per electrode
        
        n_total = n_mol + n_elec
        H_total = np.zeros((n_total, n_total))
        
        # Copy molecular Hamiltonian
        H_total[:n_mol, :n_mol] = H_mol
        
        # Electrode surface atoms (simplified)
        H_total[n_mol, n_mol] = -2.0  # Left electrode
        H_total[n_mol + 1, n_mol + 1] = -2.0  # Right electrode
        
        # Coupling (assuming S atoms are at indices 6 and 7)
        H_total[6, n_mol] = coupling_strength
        H_total[n_mol, 6] = coupling_strength
        
        H_total[7, n_mol + 1] = coupling_strength
        H_total[n_mol + 1, 7] = coupling_strength
        
        return H_total, n_mol


class GoldElectrode:
    """
    Simplified model for gold electrodes using wide-band approximation.
    """
    
    def __init__(self, fermi_level: float = -5.0,  # eV
                 coupling_gamma: float = 0.5):  # eV
        self.fermi_level = fermi_level
        self.gamma = coupling_gamma
    
    def get_self_energy(self, energy: complex, 
                       num_orbitals: int) -> np.ndarray:
        """
        Get electrode self-energy (wide-band approximation).
        
        Σ = -iΓ/2
        """
        sigma = np.zeros((num_orbitals, num_orbitals), dtype=complex)
        
        # Only couple to the surface orbital
        sigma[0, 0] = -1j * self.gamma / 2
        
        return sigma


class MolecularJunctionSimulator:
    """
    Complete molecular junction simulator.
    """
    
    def __init__(self, molecule_hamiltonian: np.ndarray,
                 electrode: GoldElectrode):
        self.H_mol = molecule_hamiltonian
        self.electrode = electrode
        
        # Extended Hamiltonian with electrode coupling
        self.H_total, self.n_mol = self._build_extended_hamiltonian()
    
    def _build_extended_hamiltonian(self) -> Tuple[np.ndarray, int]:
        """Build Hamiltonian including electrode coupling."""
        n_mol = self.H_mol.shape[0]
        
        # Simple model: 2 electrode orbitals (one per side)
        H_total = np.zeros((n_mol + 2, n_mol + 2))
        H_total[:n_mol, :n_mol] = self.H_mol
        
        # Electrode on-site energies
        H_total[n_mol, n_mol] = self.electrode.fermi_level
        H_total[n_mol + 1, n_mol + 1] = self.electrode.fermi_level
        
        # Coupling (assuming molecule couples at first and last sites)
        coupling = -0.5
        H_total[0, n_mol] = coupling
        H_total[n_mol, 0] = coupling
        
        H_total[n_mol - 1, n_mol + 1] = coupling
        H_total[n_mol + 1, n_mol - 1] = coupling
        
        return H_total, n_mol
    
    def calculate_transmission(self, energies: np.ndarray,
                              eta: float = 1e-8) -> np.ndarray:
        """
        Calculate transmission function T(E).
        """
        transmissions = []
        
        n_total = self.H_total.shape[0]
        
        for E in energies:
            z = E + 1j * eta
            
            # Build self-energies
            sigma_L = np.zeros((n_total, n_total), dtype=complex)
            sigma_L[n_mol, n_mol] = -1j * self.electrode.gamma / 2
            
            sigma_R = np.zeros((n_total, n_total), dtype=complex)
            sigma_R[n_mol + 1, n_mol + 1] = -1j * self.electrode.gamma / 2
            
            # Green's function
            G = np.linalg.inv(z * np.eye(n_total) - self.H_total - sigma_L - sigma_R)
            
            # Broadening functions
            Gamma_L = 1j * (sigma_L - sigma_L.T.conj())
            Gamma_R = 1j * (sigma_R - sigma_R.T.conj())
            
            # Transmission
            T = np.real(np.trace(Gamma_L @ G @ Gamma_R @ G.T.conj()))
            transmissions.append(T)
        
        return np.array(transmissions)
    
    def calculate_iv_curve(self, bias_range: np.ndarray,
                          temperature: float = 300.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate I-V characteristics.
        """
        currents = []
        
        kB = 8.617e-5  # eV/K
        
        for V in bias_range:
            # Chemical potentials
            mu_L = self.electrode.fermi_level + V / 2
            mu_R = self.electrode.fermi_level - V / 2
            
            # Integration range
            energies = np.linspace(mu_R - 5*kB*temperature, 
                                  mu_L + 5*kB*temperature, 500)
            
            # Calculate transmission
            transmissions = self.calculate_transmission(energies)
            
            # Current integral
            integral = 0.0
            dE = energies[1] - energies[0]
            
            for i, E in enumerate(energies):
                # Fermi functions
                f_L = 1.0 / (np.exp((E - mu_L) / (kB * temperature)) + 1)
                f_R = 1.0 / (np.exp((E - mu_R) / (kB * temperature)) + 1)
                
                integral += transmissions[i] * (f_L - f_R) * dE
            
            # Current in units of e²/h × V
            current = integral
            currents.append(current)
        
        return bias_range, np.array(currents)


class MolecularSwitch:
    """
    Molecular switch with redox-active center.
    
    Models conductance switching via redox state changes.
    """
    
    def __init__(self, base_hamiltonian: np.ndarray,
                 switch_site: int = 3):
        self.H_base = base_hamiltonian
        self.switch_site = switch_site
        
        # Redox states: 0 = oxidized (low conductance), 1 = reduced (high conductance)
        self.state = 0
    
    def set_state(self, state: int):
        """Set switch state (0 or 1)."""
        self.state = state
    
    def get_hamiltonian(self) -> np.ndarray:
        """
        Get Hamiltonian for current switch state.
        
        State 0: Site energy shifted up (reduced coupling)
        State 1: Site energy at normal position
        """
        H = self.H_base.copy()
        
        if self.state == 0:
            # Oxidized state - higher site energy
            H[self.switch_site, self.switch_site] += 1.0
        
        return H
    
    def calculate_conductance_ratio(self, electrode: GoldElectrode) -> float:
        """
        Calculate ON/OFF conductance ratio.
        """
        # ON state
        self.set_state(1)
        H_on = self.get_hamiltonian()
        sim_on = MolecularJunctionSimulator(H_on, electrode)
        
        Ef = electrode.fermi_level
        energies = np.linspace(Ef - 0.1, Ef + 0.1, 50)
        T_on = sim_on.calculate_transmission(energies)
        G_on = np.mean(T_on)
        
        # OFF state
        self.set_state(0)
        H_off = self.get_hamiltonian()
        sim_off = MolecularJunctionSimulator(H_off, electrode)
        
        T_off = sim_off.calculate_transmission(energies)
        G_off = np.mean(T_off)
        
        if G_off > 1e-10:
            ratio = G_on / G_off
        else:
            ratio = float('inf')
        
        return ratio, G_on, G_off


def example_bdt_junction():
    """
    Example: Benzene dithiol molecular junction.
    """
    print("=" * 70)
    print("Example: Benzene Dithiol (BDT) Molecular Junction")
    print("=" * 70)
    
    # Build molecule
    hamiltonian_builder = MolecularHamiltonian("BDT")
    H_BDT, bonds = hamiltonian_builder.build_benzene_dithiol(with_anchors=True)
    
    print(f"\nBDT molecule:")
    print(f"  Number of atoms: {H_BDT.shape[0]}")
    print(f"  Number of bonds: {len(bonds)}")
    
    # Create electrodes
    electrode = GoldElectrode(fermi_level=0.0, coupling_gamma=0.5)
    
    # Create junction simulator
    simulator = MolecularJunctionSimulator(H_BDT, electrode)
    
    # Calculate transmission
    print("\nCalculating transmission...")
    energies = np.linspace(-3, 3, 200)
    transmissions = simulator.calculate_transmission(energies)
    
    # Find conductance at Fermi level
    G_0 = np.interp(0.0, energies, transmissions)
    
    print(f"\nResults:")
    print(f"  Zero-bias conductance: {G_0:.4f} G₀")
    print(f"  Peak transmission: {np.max(transmissions):.4f}")
    
    # Find resonances
    analyzer = TransmissionAnalyzer(energies, transmissions)
    resonances = analyzer.find_resonances(threshold=0.1)
    
    print(f"  Number of transmission resonances: {len(resonances)}")
    
    if resonances:
        print(f"  Resonant states:")
        for res in resonances[:3]:
            print(f"    E = {res['energy']:.2f} eV, T = {res['height']:.3f}")
    
    # Calculate I-V curve
    print("\nCalculating I-V curve...")
    bias_range = np.linspace(0, 2.0, 21)
    biases, currents = simulator.calculate_iv_curve(bias_range)
    
    print(f"\nI-V characteristics:")
    for i in [0, 5, 10, 15, 20]:
        print(f"  V = {biases[i]:.2f} V, I = {currents[i]*1e6:.3f} μA")
    
    return simulator, energies, transmissions


def example_molecular_wire():
    """
    Example: Oligoacene molecular wire - length dependence.
    """
    print("\n" + "=" * 70)
    print("Example: Oligoacene Molecular Wire - Length Dependence")
    print("=" * 70)
    
    hamiltonian_builder = MolecularHamiltonian()
    electrode = GoldElectrode(fermi_level=0.0, coupling_gamma=0.5)
    
    lengths = [1, 2, 3, 4, 5]  # Number of rings
    conductances = []
    
    print(f"\nLength dependence of conductance:")
    print(f"{'Rings':>10} {'Carbons':>10} {'G/G₀':>12} {'Decay':>12}")
    print("-" * 50)
    
    for n in lengths:
        H, _ = hamiltonian_builder.build_oligoacene(n_rings=n)
        
        # Add electrode coupling
        H_ext, n_mol = hamiltonian_builder.add_electrode_coupling(H)
        
        simulator = MolecularJunctionSimulator(H_ext, electrode)
        
        energies = np.linspace(-0.5, 0.5, 100)
        transmissions = simulator.calculate_transmission(energies)
        
        G = np.interp(0.0, energies, transmissions)
        conductances.append(G)
        
        decay = conductances[-2] / G if len(conductances) > 1 else 0
        
        print(f"{n:>10} {H.shape[0]:>10} {G:>12.4f} {decay:>12.2f}")
    
    # Calculate decay constant
    if len(conductances) >= 2:
        # Fit to exponential decay: G = G₀ exp(-βL)
        log_G = np.log(conductances)
        lengths_angstrom = np.array(lengths) * 2.5  # Approximate conversion
        
        # Linear fit
        beta = -(log_G[-1] - log_G[0]) / (lengths_angstrom[-1] - lengths_angstrom[0])
        
        print(f"\nConductance decay constant: β = {beta:.3f} Å⁻¹")
        print(f"  (Typical for molecular wires: 0.2-0.8 Å⁻¹)")
    
    return lengths, conductances


def example_molecular_switch():
    """
    Example: Redox-active molecular switch.
    """
    print("\n" + "=" * 70)
    print("Example: Redox-Active Molecular Switch")
    print("=" * 70)
    
    # Build base molecule (BDT with central modification)
    hamiltonian_builder = MolecularHamiltonian("BDT")
    H_base, _ = hamiltonian_builder.build_benzene_dithiol(with_anchors=True)
    
    electrode = GoldElectrode(fermi_level=0.0, coupling_gamma=0.5)
    
    # Create switch
    switch = MolecularSwitch(H_base, switch_site=2)
    
    print(f"\nMolecular switch:")
    print(f"  Switch site: {switch.switch_site}")
    
    # Calculate ON/OFF ratio
    ratio, G_on, G_off = switch.calculate_conductance_ratio(electrode)
    
    print(f"\nConductance states:")
    print(f"  ON state (reduced):  G = {G_on:.4f} G₀")
    print(f"  OFF state (oxidized): G = {G_off:.4f} G₀")
    print(f"  ON/OFF ratio: {ratio:.1f}")
    
    # State-dependent transmission
    print("\nTransmission spectra:")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    energies = np.linspace(-3, 3, 200)
    
    for state, label in [(1, 'ON'), (0, 'OFF')]:
        switch.set_state(state)
        H = switch.get_hamiltonian()
        sim = MolecularJunctionSimulator(H, electrode)
        T = sim.calculate_transmission(energies)
        
        print(f"  {label} state: peak T = {np.max(T):.3f}")
    
    return switch


def example_conductance_histogram():
    """
    Example: Simulate conductance histogram for single-molecule measurements.
    """
    print("\n" + "=" * 70)
    print("Example: Single-Molecule Conductance Histogram")
    print("=" * 70)
    
    # Simulate multiple junction configurations
    np.random.seed(42)
    
    n_measurements = 1000
    conductances = []
    
    # BDT molecule
    hamiltonian_builder = MolecularHamiltonian("BDT")
    H_BDT, _ = hamiltonian_builder.build_benzene_dithiol(with_anchors=True)
    
    electrode = GoldElectrode(fermi_level=0.0, coupling_gamma=0.5)
    
    print(f"\nSimulating {n_measurements} junction configurations...")
    
    for i in range(n_measurements):
        # Randomize coupling strength (mimics different contact geometries)
        coupling_variation = np.random.normal(1.0, 0.2)
        
        # Modify Hamiltonian coupling
        H = H_BDT.copy()
        H[6, 0] *= coupling_variation  # Left anchor
        H[0, 6] *= coupling_variation
        H[7, 3] *= coupling_variation  # Right anchor
        H[3, 7] *= coupling_variation
        
        simulator = MolecularJunctionSimulator(H, electrode)
        
        energies = np.linspace(-0.2, 0.2, 50)
        transmissions = simulator.calculate_transmission(energies)
        G = np.interp(0.0, energies, transmissions)
        
        conductances.append(G)
    
    conductances = np.array(conductances)
    
    # Analyze histogram
    log_conductances = np.log10(conductances[conductances > 1e-6])
    
    print(f"\nConductance statistics:")
    print(f"  Mean log₁₀(G/G₀): {np.mean(log_conductances):.2f}")
    print(f"  Std log₁₀(G/G₀): {np.std(log_conductances):.2f}")
    print(f"  Most probable G: {10**np.mean(log_conductances):.4f} G₀")
    
    # Typical experimental values for BDT
    print(f"\n  (Experimental BDT: log₁₀(G/G₀) ≈ -4.5 ± 0.5)")
    
    return conductances


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Molecular Electronics - Application Cases")
    print("=" * 70)
    
    # Run examples
    simulator_bdt, energies_bdt, trans_bdt = example_bdt_junction()
    lengths, conductances = example_molecular_wire()
    switch = example_molecular_switch()
    conductance_hist = example_conductance_histogram()
    
    print("\n" + "=" * 70)
    print("Molecular Electronics Cases - Complete")
    print("=" * 70)
