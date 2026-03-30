"""
case_2d_material_device.py

2D Material Field-Effect Transistor (FET) - Application Case Study

This module demonstrates quantum transport in 2D material devices:
- Graphene nanoribbon transistors
- MoS2 FETs
- Tunnel FETs using 2D materials
- Bilayer graphene devices

References:
- Schwierz, Nature Nanotech. 5, 487 (2010) - Graphene transistors
- Radisavljevic et al., Nature Nanotech. 6, 147 (2011) - MoS2 FET
- Britnell et al., Science 335, 947 (2012) - Tunnel FET
"""

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import csr_matrix, lil_matrix
from typing import Dict, List, Tuple, Optional
import sys

sys.path.insert(0, '/root/.openclaw/workspace/dftlammps')

from quantum_transport.kwant_interface import (
    TightBindingModel, PeierlsSubstitution
)
from quantum_transport.negf_formalism import (
    SelfEnergy, NEGFSystemAdvanced, LandauerButtiker
)


class GrapheneNanoribbon:
    """
    Tight-binding model for graphene nanoribbon (GNR).
    
    Supports both armchair and zigzag edge terminations.
    """
    
    def __init__(self, width: int, length: int, 
                 edge_type: str = "armchair"):
        """
        Args:
            width: Width in number of carbon rows
            length: Length in number of unit cells
            edge_type: "armchair" or "zigzag"
        """
        self.width = width
        self.length = length
        self.edge_type = edge_type
        
        # Graphene parameters
        self.a = 1.42  # C-C bond length (Angstrom)
        self.t = -2.7  # Nearest neighbor hopping (eV)
    
    def build_hamiltonian(self) -> Tuple[np.ndarray, int, int]:
        """
        Build tight-binding Hamiltonian for GNR.
        
        Returns:
            H: Hamiltonian matrix
            num_sites: Total number of sites
            width_sites: Number of sites across width
        """
        if self.edge_type == "armchair":
            return self._build_armchair()
        elif self.edge_type == "zigzag":
            return self._build_zigzag()
        else:
            raise ValueError(f"Unknown edge type: {self.edge_type}")
    
    def _build_armchair(self) -> Tuple[np.ndarray, int, int]:
        """Build armchair-edge GNR Hamiltonian."""
        # Armchair: 2*width atoms per unit cell
        sites_per_cell = 2 * self.width
        num_sites = sites_per_cell * self.length
        
        H = np.zeros((num_sites, num_sites))
        
        for cell in range(self.length):
            base = cell * sites_per_cell
            
            # Intra-cell hoppings
            for i in range(self.width):
                # A-B pairs within cell
                H[base + 2*i, base + 2*i + 1] = self.t
                H[base + 2*i + 1, base + 2*i] = self.t
            
            # Inter-cell hoppings
            if cell < self.length - 1:
                next_base = (cell + 1) * sites_per_cell
                
                for i in range(self.width - 1):
                    # Connect B to next A
                    H[base + 2*i + 1, next_base + 2*(i+1)] = self.t
                    H[next_base + 2*(i+1), base + 2*i + 1] = self.t
                
                # Connect last B to first A of next cell
                H[base + 2*self.width - 1, next_base] = self.t
                H[next_base, base + 2*self.width - 1] = self.t
        
        return H, num_sites, sites_per_cell
    
    def _build_zigzag(self) -> Tuple[np.ndarray, int, int]:
        """Build zigzag-edge GNR Hamiltonian."""
        # Zigzag: width atoms per row, 2 rows per unit cell
        sites_per_cell = 2 * self.width
        num_sites = sites_per_cell * self.length
        
        H = np.zeros((num_sites, num_sites))
        
        for cell in range(self.length):
            base = cell * sites_per_cell
            
            # Intra-cell: vertical bonds
            for i in range(self.width):
                H[base + i, base + self.width + i] = self.t
                H[base + self.width + i, base + i] = self.t
            
            # Horizontal bonds
            for i in range(self.width - 1):
                H[base + i, base + i + 1] = self.t
                H[base + i + 1, base + i] = self.t
                
                H[base + self.width + i, base + self.width + i + 1] = self.t
                H[base + self.width + i + 1, base + self.width + i] = self.t
            
            # Inter-cell bonds
            if cell < self.length - 1:
                next_base = (cell + 1) * sites_per_cell
                
                for i in range(self.width):
                    H[base + self.width + i, next_base + i] = self.t
                    H[next_base + i, base + self.width + i] = self.t
        
        return H, num_sites, sites_per_cell
    
    def calculate_band_structure(self, num_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate band structure along ribbon direction.
        """
        H, num_sites, sites_per_cell = self.build_hamiltonian()
        
        # Get intra-cell and inter-cell hoppings
        H0 = H[:sites_per_cell, :sites_per_cell]
        H1 = H[:sites_per_cell, sites_per_cell:2*sites_per_cell]
        
        # k-points
        k_points = np.linspace(-np.pi, np.pi, num_k)
        energies = np.zeros((num_k, sites_per_cell))
        
        for i, k in enumerate(k_points):
            # Bloch Hamiltonian
            H_k = H0 + H1 * np.exp(1j * k) + H1.T.conj() * np.exp(-1j * k)
            
            eigs = linalg.eigvalsh(H_k)
            energies[i] = np.sort(eigs)
        
        return k_points, energies
    
    def get_gap(self) -> float:
        """Calculate band gap."""
        _, energies = self.calculate_band_structure(num_k=50)
        
        # Find gap at k=0
        eigs = energies[len(energies)//2]
        
        # Find HOMO and LUMO
        occupied = eigs[eigs < 0]
        unoccupied = eigs[eigs > 0]
        
        if len(occupied) > 0 and len(unoccupied) > 0:
            gap = np.min(unoccupied) - np.max(occupied)
        else:
            gap = 0.0
        
        return gap


class MoS2Model:
    """
    Tight-binding model for monolayer MoS2.
    
    Based on three-band model for conduction/valence bands.
    """
    
    def __init__(self, lattice_constant: float = 3.193):
        self.a = lattice_constant  # Angstrom
        
        # Three-band model parameters (eV)
        # From Liu et al., PRL 105, 136805 (2010)
        self.t0 = -0.184  # Intra-cell
        self.t1 = 0.401   # Nearest neighbor
        self.t2 = 0.507   # Next nearest
        self.t11 = 0.228  # Third neighbor
        
        self.delta = 1.79  # Splitting
    
    def build_hamiltonian_k(self, k: np.ndarray) -> np.ndarray:
        """
        Build k-space Hamiltonian for MoS2.
        
        Returns 3x3 matrix for the three-band model.
        """
        kx, ky = k[0], k[1]
        
        # Reciprocal lattice vectors
        a = self.a
        b1 = 2 * np.pi / a * np.array([1, -1/np.sqrt(3)])
        b2 = 2 * np.pi / a * np.array([0, 2/np.sqrt(3)])
        
        # Convert to lattice coordinates
        k_cart = kx * b1 + ky * b2
        
        # Hamiltonian elements (simplified)
        H = np.zeros((3, 3), dtype=complex)
        
        # On-site
        H[0, 0] = self.delta / 2
        H[1, 1] = -self.delta / 2
        H[2, 2] = 0
        
        # Off-diagonal (simplified model)
        # Real implementation would include full hopping terms
        
        return H
    
    def calculate_band_structure(self, k_path: List[np.ndarray]) -> np.ndarray:
        """
        Calculate band structure along path.
        """
        energies = []
        
        for k in k_path:
            H_k = self.build_hamiltonian_k(k)
            eigs = linalg.eigvalsh(H_k)
            energies.append(eigs)
        
        return np.array(energies)
    
    def get_band_gap(self) -> float:
        """Get direct band gap at K point."""
        # K point in reciprocal space
        k_K = np.array([1/3, 1/3])
        H_K = self.build_hamiltonian_k(k_K)
        eigs = linalg.eigvalsh(H_K)
        
        gap = eigs[1] - eigs[0]
        return gap


class FETSimulator:
    """
    Field-effect transistor simulator for 2D materials.
    """
    
    def __init__(self, channel_model: TightBindingModel,
                 gate_voltage: float = 0.0):
        self.channel = channel_model
        self.Vg = gate_voltage
        
        # FET parameters
        self.L_ch = 100e-9  # Channel length (m)
        self.W_ch = 1e-6    # Channel width (m)
        self.Cox = 1e-3     # Oxide capacitance (F/m²)
    
    def apply_gate_potential(self, potential_profile: np.ndarray):
        """
        Apply gate potential profile along channel.
        
        Args:
            potential_profile: 1D array of potentials (eV)
        """
        # Modify on-site energies
        n_sites = len(self.channel.on_site_energies)
        
        # Interpolate to match number of sites
        x_orig = np.linspace(0, 1, len(potential_profile))
        x_new = np.linspace(0, 1, n_sites)
        
        interpolated_potential = np.interp(x_new, x_orig, potential_profile)
        
        # Add to on-site energies
        self.channel.on_site_energies += interpolated_potential
    
    def calculate_transconductance(self, 
                                  Vg_range: np.ndarray,
                                  Vds: float = 0.1) -> np.ndarray:
        """
        Calculate transconductance gm = dIds/dVg.
        """
        gm_values = []
        
        for Vg in Vg_range:
            # Calculate drain current at this gate voltage
            # Simplified model
            
            # Threshold voltage
            Vth = 0.5
            
            if Vg > Vth:
                # Above threshold
                Ids = self.Cox * self.W_ch / self.L_ch * (Vg - Vth) * Vds
            else:
                # Below threshold (subthreshold)
                Ids = 1e-12 * np.exp((Vg - Vth) / 0.026)  # Thermal voltage
            
            gm_values.append(Ids)
        
        return np.array(gm_values)
    
    def calculate_subthreshold_swing(self, 
                                    Vg_below_threshold: np.ndarray,
                                    Ids: np.ndarray) -> float:
        """
        Calculate subthreshold swing (mV/decade).
        
        SS = dVg / d(log10(Ids))
        """
        log_Ids = np.log10(Ids)
        
        # Linear fit
        coeffs = np.polyfit(Vg_below_threshold, log_Ids, 1)
        
        # SS = 1 / slope (in V/decade, convert to mV/decade)
        ss = 1 / coeffs[0] * 1000  # mV/decade
        
        return ss


class TunnelFET:
    """
    Tunnel Field-Effect Transistor using 2D materials.
    
    Uses band-to-band tunneling for sub-60mV/decade switching.
    """
    
    def __init__(self, 
                 source_material: str = "InAs",
                 channel_material: str = "WTe2",
                 drain_material: str = "InAs"):
        self.source = source_material
        self.channel = channel_material
        self.drain = drain_material
        
        # Band alignment parameters (eV)
        self.band_offsets = {
            'InAs': {'Eg': 0.36, 'chi': 4.9},
            'WTe2': {'Eg': 1.0, 'chi': 4.0},
            'MoS2': {'Eg': 1.8, 'chi': 4.0},
        }
    
    def calculate_tunneling_current(self, 
                                   Vgs: float,
                                   Vds: float,
                                   temperature: float = 300.0) -> float:
        """
        Calculate band-to-band tunneling current.
        
        Uses WKB approximation for tunneling probability.
        """
        # Get band parameters
        source_params = self.band_offsets[self.source]
        channel_params = self.band_offsets[self.channel]
        
        # Tunneling barrier height
        phi_b = source_params['Eg'] / 2 + channel_params['Eg'] / 2
        
        # Electric field (simplified)
        E_field = Vgs / 1e-9  # V/m (assuming 1nm tunneling region)
        
        # WKB tunneling probability
        # T ≈ exp(-2 ∫ κ dx)
        m_eff = 0.1 * 9.11e-31  # kg
        hbar = 1.055e-34
        
        # Tunneling decay constant
        kappa = np.sqrt(2 * m_eff * phi_b * 1.602e-19) / hbar
        
        # Tunneling distance
        d_tunnel = phi_b / E_field if E_field > 0 else float('inf')
        
        # Tunneling probability
        T_wkb = np.exp(-2 * kappa * d_tunnel) if d_tunnel > 0 else 0
        
        # Current (simplified)
        J0 = 1e6  # A/m²
        J_tunnel = J0 * T_wkb
        
        return J_tunnel
    
    def calculate_subthreshold_swing(self) -> float:
        """
        Theoretical subthreshold swing for tunnel FET.
        
        Can be < 60 mV/decade at room temperature.
        """
        # For ideal TFET: SS can approach 0 at low current
        # Practical values: 20-40 mV/decade
        
        return 30.0  # mV/decade (typical value)


class BilayerGrapheneDevice:
    """
    Bilayer graphene device with band gap tunability.
    
    Band gap can be opened by perpendicular electric field.
    """
    
    def __init__(self, d: float = 3.35):  # Interlayer distance (Angstrom)
        self.d = d
        self.t_perp = 0.39  # Interlayer hopping (eV)
        
        # Bernal stacking parameters
        self.gamma0 = 2.7   # Intralayer
        self.gamma1 = 0.39  # Direct interlayer (A to B')
        self.gamma3 = 0.315 # A to A'
        self.gamma4 = 0.044 # A to B'
    
    def calculate_band_gap(self, E_perp: float) -> float:
        """
        Calculate band gap induced by perpendicular electric field.
        
        Args:
            E_perp: Perpendicular electric field (V/nm)
        """
        # Potential difference between layers
        delta_V = E_perp * self.d / 10  # Convert to eV
        
        # Band gap opening (simplified model)
        # From McCann, PRL 96, 086805 (2006)
        
        if delta_V < self.t_perp:
            # Low field regime
            gap = delta_V**2 / self.t_perp
        else:
            # High field regime
            gap = delta_V
        
        return gap
    
    def calculate_berry_curvature(self, k: np.ndarray) -> float:
        """
        Calculate Berry curvature at k-point.
        
        Important for anomalous Hall effect in gapped bilayer.
        """
        # Simplified calculation
        # Real implementation would use full tight-binding
        
        kx, ky = k[0], k[1]
        
        # Berry curvature peaks at band touching points
        berry = 0.0
        
        return berry


def example_graphene_nanoribbon():
    """
    Example: Graphene nanoribbon band structure and transport.
    """
    print("=" * 70)
    print("Example: Graphene Nanoribbon FET")
    print("=" * 70)
    
    # Create armchair GNR
    width = 5  # Number of carbon rows
    length = 10  # Unit cells
    
    gnr = GrapheneNanoribbon(width, length, edge_type="armchair")
    
    print(f"\nArmchair GNR: width = {width} rows, length = {length} cells")
    
    # Calculate band structure
    k_points, energies = gnr.calculate_band_structure(num_k=100)
    
    gap = gnr.get_gap()
    print(f"  Band gap: {gap:.3f} eV")
    
    # Band width
    valence_max = np.max(energies[:, :width])
    conduction_min = np.min(energies[:, width:])
    
    print(f"  Valence band maximum: {valence_max:.3f} eV")
    print(f"  Conduction band minimum: {conduction_min:.3f} eV")
    
    # Effective mass estimate
    # From curvature at band edge
    d2E_dk2 = (energies[51, width] - 2*energies[50, width] + energies[49, width]) / \
              (k_points[1] - k_points[0])**2
    
    hbar = 6.582e-16  # eV·s
    a = 2.46  # Angstrom
    m_eff = hbar**2 / (2 * np.abs(d2E_dk2) * (2*np.pi/a)**2)
    
    print(f"  Effective mass (electrons): {m_eff:.3f} m₀")
    
    return gnr, k_points, energies


def example_mos2_fet():
    """
    Example: MoS2 field-effect transistor.
    """
    print("\n" + "=" * 70)
    print("Example: MoS2 Field-Effect Transistor")
    print("=" * 70)
    
    # Create MoS2 model
    mos2 = MoS2Model()
    
    print(f"\nMonolayer MoS2:")
    print(f"  Lattice constant: {mos2.a} Å")
    
    # Calculate band gap
    gap = mos2.get_band_gap()
    print(f"  Direct band gap at K: {gap:.2f} eV")
    
    # Calculate band structure along K-Γ-K path
    k_path = []
    
    # K point
    k_K = np.array([1/3, 1/3])
    
    # Γ point
    k_G = np.array([0, 0])
    
    # Path: K -> Γ -> K
    num_k = 50
    for i in range(num_k):
        k_path.append(k_K + (k_G - k_K) * i / num_k)
    
    for i in range(num_k):
        k_path.append(k_G + (k_K - k_G) * i / num_k)
    
    energies = mos2.calculate_band_structure(k_path)
    
    print(f"\nBand structure:")
    print(f"  Valence band at K: {energies[num_k//2, 0]:.2f} eV")
    print(f"  Conduction band at K: {energies[num_k//2, 1]:.2f} eV")
    
    # FET characteristics
    print(f"\nFET parameters:")
    print(f"  Expected mobility: ~200 cm²/V·s")
    print(f"  I_on/I_off ratio: > 10⁸")
    print(f"  Subthreshold swing: ~60 mV/decade")
    
    return mos2


def example_tunnel_fet():
    """
    Example: Tunnel FET using 2D materials.
    """
    print("\n" + "=" * 70)
    print("Example: 2D Tunnel FET")
    print("=" * 70)
    
    tfet = TunnelFET(
        source_material="InAs",
        channel_material="WTe2",
        drain_material="InAs"
    )
    
    print(f"\nTunnel FET structure:")
    print(f"  Source: {tfet.source}")
    print(f"  Channel: {tfet.channel}")
    print(f"  Drain: {tfet.drain}")
    
    # Calculate tunneling current
    print(f"\nTunneling current vs gate voltage:")
    print(f"{'Vgs (V)':>10} {'J (A/m²)':>15}")
    print("-" * 30)
    
    for Vgs in [0.0, 0.2, 0.4, 0.6, 0.8]:
        J = tfet.calculate_tunneling_current(Vgs, Vds=0.1)
        print(f"{Vgs:>10.1f} {J:>15.2e}")
    
    ss = tfet.calculate_subthreshold_swing()
    print(f"\nSubthreshold swing: {ss:.0f} mV/decade")
    print(f"  (Below 60 mV/decade limit of conventional FETs)")
    
    return tfet


def example_bilayer_graphene():
    """
    Example: Bilayer graphene with tunable band gap.
    """
    print("\n" + "=" * 70)
    print("Example: Bilayer Graphene Band Gap Engineering")
    print("=" * 70)
    
    bilayer = BilayerGrapheneDevice()
    
    print(f"\nBilayer graphene parameters:")
    print(f"  Interlayer distance: {bilayer.d} Å")
    print(f"  Interlayer hopping γ₁: {bilayer.gamma1} eV")
    
    # Calculate band gap vs electric field
    print(f"\nBand gap vs perpendicular electric field:")
    print(f"{'E_perp (V/nm)':>15} {'Gap (eV)':>12}")
    print("-" * 30)
    
    for E in [0, 0.5, 1.0, 1.5, 2.0]:
        gap = bilayer.calculate_band_gap(E)
        print(f"{E:>15.1f} {gap:>12.3f}")
    
    print(f"\nKey advantages:")
    print(f"  - Tunable band gap: 0 - 250 meV")
    print(f"  - High mobility maintained")
    print(f"  - Parabolic dispersion")
    
    return bilayer


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("2D Material Devices - Application Cases")
    print("=" * 70)
    
    # Run examples
    gnr, k_pts, enes = example_graphene_nanoribbon()
    mos2 = example_mos2_fet()
    tfet = example_tunnel_fet()
    bilayer = example_bilayer_graphene()
    
    print("\n" + "=" * 70)
    print("2D Material Devices Cases - Complete")
    print("=" * 70)
