"""
Photovoltaic Carrier Dynamics Case Study
========================================

Simulation of carrier dynamics in photovoltaic materials including:
- Hot carrier cooling
- Exciton dissociation
- Charge separation and transport
- Recombination processes

This case study demonstrates the use of dftlammps non-adiabatic dynamics
for understanding photophysical processes in solar cell materials.

Author: dftlammps development team
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nonadiabatic.excited_state_dynamics import (
    ExcitedStateDynamicsWorkflow,
    ExcitonState,
    CarrierState,
    ChargeSeparationState,
    ExcitedProcess
)
from nonadiabatic.pyxaid_interface import PYXAIDConfig, PYXAIDWorkflow


class PerovskiteSolarCell:
    """
    Model for carrier dynamics in perovskite solar cells.
    
    Material parameters typical of MAPbI3 perovskites.
    """
    
    def __init__(self):
        # Material parameters for MAPbI3
        self.band_gap = 1.6  # eV
        self.electron_mass = 0.23  # m0
        self.hole_mass = 0.29  # m0
        self.electron_mobility = 10.0  # cm^2/(V·s)
        self.hole_mobility = 1.0  # cm^2/(V·s)
        self.dielectric_constant = 35.0
        self.exciton_binding = 0.02  # eV (weakly bound)
        self.exciton_radius = 6.0  # nm (large Wannier exciton)
        
        # Carrier cooling parameters
        self.cooling_rate = 1e13  # s^-1 (hot phonon bottleneck)
        
        self.workflow = None
        
    def setup_simulation(self, temperature: float = 300.0):
        """Setup excited state dynamics simulation."""
        
        self.workflow = ExcitedStateDynamicsWorkflow()
        
        # Exciton states (near band edge)
        exciton_energies = [self.band_gap - self.exciton_binding]
        binding_energies = [self.exciton_binding]
        radii = [self.exciton_radius]
        
        self.workflow.setup_exciton_system(
            exciton_energies=exciton_energies,
            binding_energies=binding_energies,
            radii=radii
        )
        
        # Carrier states (band-like)
        # Multiple energies representing hot carriers
        electron_energies = np.linspace(0, 1.0, 10)  # Above CBM
        hole_energies = np.linspace(0, 0.5, 5)  # Below VBM
        
        self.workflow.setup_carrier_system(
            electron_masses=[self.electron_mass] * len(electron_energies),
            hole_masses=[self.hole_mass] * len(hole_energies),
            mobilities={
                'electron': self.electron_mobility,
                'hole': self.hole_mobility
            }
        )
        
        print(f"Perovskite simulation setup at T={temperature}K")
        print(f"  Band gap: {self.band_gap} eV")
        print(f"  Exciton binding: {self.exciton_binding} eV")
        
    def simulate_photogeneration(self, 
                                 photon_energy: float = 3.0,
                                 temperature: float = 300.0) -> Dict:
        """
        Simulate photogeneration and initial carrier cooling.
        
        Parameters
        ----------
        photon_energy : float
            Incident photon energy in eV
        temperature : float
            Temperature in K
            
        Returns
        -------
        Dict with photogeneration analysis
        """
        print(f"\n{'='*60}")
        print("PHOTOGENERATION AND HOT CARRIER COOLING")
        print(f"{'='*60}")
        
        # Initial carrier energies (above band gap)
        excess_energy = photon_energy - self.band_gap
        
        # Electron gets more energy (effective mass ratio)
        me_ratio = self.hole_mass / (self.electron_mass + self.hole_mass)
        electron_energy = excess_energy * me_ratio
        hole_energy = excess_energy * (1 - me_ratio)
        
        print(f"Photon energy: {photon_energy:.2f} eV")
        print(f"Excess energy: {excess_energy:.2f} eV")
        print(f"Initial electron energy: {electron_energy:.2f} eV")
        print(f"Initial hole energy: {hole_energy:.2f} eV")
        
        # Simulate hot carrier relaxation
        results = self.workflow.run_carrier_relaxation(
            initial_electron_energy=electron_energy,
            initial_hole_energy=hole_energy
        )
        
        print(f"\nHot Carrier Cooling Results:")
        print(f"  Electron cooling time: {results['electron_relaxation']['cooling_time']:.1f} fs")
        print(f"  Hole cooling time: {results['hole_relaxation']['cooling_time']:.1f} fs")
        
        return results
    
    def simulate_exciton_dissociation(self,
                                      electric_field: float = 0.05) -> Dict:
        """
        Simulate exciton dissociation under electric field.
        
        Parameters
        ----------
        electric_field : float
            Built-in electric field in V/nm
            
        Returns
        -------
        Dict with dissociation analysis
        """
        print(f"\n{'='*60}")
        print("EXCITON DISSOCIATION")
        print(f"{'='*60}")
        
        print(f"Electric field: {electric_field*100:.1f} V/μm")
        
        results = self.workflow.run_exciton_dissociation(
            initial_exciton_idx=0,
            electric_field=electric_field
        )
        
        print(f"\nDissociation Results:")
        print(f"  Dissociation yield: {results['dissociation_yield']*100:.1f}%")
        print(f"  Diffusion length: {results['diffusion_length']:.1f} nm")
        
        return results
    
    def simulate_charge_transport(self,
                                   electric_field: np.ndarray = None,
                                   simulation_time: float = 1000.0) -> Dict:
        """
        Simulate charge carrier transport.
        
        Parameters
        ----------
        electric_field : np.ndarray
            Electric field vector in V/nm
        simulation_time : float
            Simulation time in fs
            
        Returns
        -------
        Dict with transport analysis
        """
        print(f"\n{'='*60}")
        print("CHARGE TRANSPORT")
        print(f"{'='*60}")
        
        if electric_field is None:
            electric_field = np.array([0.01, 0, 0])  # V/nm
        
        # Create carrier at band edge
        electron = CarrierState(
            charge=-1,
            energy=0.0,
            effective_mass=self.electron_mass,
            mobility=self.electron_mobility,
            state_index=0
        )
        
        results = self.workflow.carrier_dynamics.simulate_transport(
            carrier=electron,
            electric_field=electric_field,
            simulation_time=simulation_time
        )
        
        displacement = results['displacement']
        drift_velocity = results['drift_velocity']
        
        print(f"Transport Results ({simulation_time:.0f} fs):")
        print(f"  Displacement: ({displacement[0]:.2f}, {displacement[1]:.2f}, {displacement[2]:.2f}) nm")
        print(f"  Drift velocity: {np.linalg.norm(drift_velocity):.4f} nm/fs")
        print(f"  Mobility check: {self.electron_mobility:.1f} cm²/(V·s)")
        
        return results
    
    def calculate_pce(self,
                     jsc: float = 25.0,
                     voc: float = 1.1,
                     ff: float = 0.8) -> Dict:
        """
        Estimate power conversion efficiency (simplified).
        
        Parameters
        ----------
        jsc : float
            Short-circuit current in mA/cm²
        voc : float
            Open-circuit voltage in V
        ff : float
            Fill factor
            
        Returns
        -------
        Dict with PCE analysis
        """
        print(f"\n{'='*60}")
        print("POWER CONVERSION EFFICIENCY")
        print(f"{'='*60}")
        
        # PCE = Jsc * Voc * FF / P_in
        P_in = 100.0  # mW/cm² (AM1.5G)
        pce = jsc * voc * ff / P_in * 100  # %
        
        print(f"Device Parameters:")
        print(f"  Jsc: {jsc:.1f} mA/cm²")
        print(f"  Voc: {voc:.2f} V")
        print(f"  FF: {ff*100:.1f}%")
        print(f"\nPower Conversion Efficiency: {pce:.1f}%")
        
        # SQ limit for this band gap
        sq_limit = self._shockley_queisser_limit(self.band_gap)
        print(f"Shockley-Queisser limit: {sq_limit:.1f}%")
        
        return {
            'pce': pce,
            'jsc': jsc,
            'voc': voc,
            'ff': ff,
            'sq_limit': sq_limit
        }
    
    def _shockley_queisser_limit(self, band_gap: float) -> float:
        """Estimate Shockley-Queisser limit for given band gap."""
        # Simplified analytical approximation
        # Maximum at ~1.34 eV with ~33%
        x = band_gap
        sq = 33.7 * np.exp(-(x - 1.34)**2 / 0.5)
        return sq
    
    def visualize_results(self, output_dir: str = "./carrier_dynamics_results"):
        """Generate comprehensive visualizations."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Energy level diagram
        ax = plt.subplot(3, 3, 1)
        levels = [0, self.band_gap - self.exciton_binding, self.band_gap, 
                 self.band_gap + 1.0]
        labels = ['VB', 'Exciton', 'CB', 'Hot carrier']
        colors = ['blue', 'green', 'red', 'orange']
        
        for i, (E, label, color) in enumerate(zip(levels, labels, colors)):
            ax.axhline(E, color=color, linestyle='-', linewidth=2)
            ax.text(0.5, E, label, fontsize=10, color=color)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, self.band_gap + 1.5)
        ax.set_ylabel('Energy (eV)')
        ax.set_title('Energy Level Diagram')
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Hot carrier cooling
        if 'carrier_relaxation' in self.workflow.analyses:
            ax = plt.subplot(3, 3, 2)
            data = self.workflow.analyses['carrier_relaxation']
            
            ax.plot(data['electron_relaxation']['times'], 
                   data['electron_relaxation']['energies'], 
                   'b-', label='Electron')
            ax.plot(data['hole_relaxation']['times'],
                   data['hole_relaxation']['energies'],
                   'r-', label='Hole')
            ax.set_xlabel('Time (fs)')
            ax.set_ylabel('Energy (eV)')
            ax.set_title('Hot Carrier Cooling')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Exciton diffusion
        if 'exciton_dissociation' in self.workflow.analyses:
            ax = plt.subplot(3, 3, 3)
            data = self.workflow.analyses['exciton_dissociation']
            traj = data['diffusion_trajectory']
            
            ax.plot(traj['times'], np.linalg.norm(traj['positions'], axis=1), 'g-')
            ax.set_xlabel('Time (fs)')
            ax.set_ylabel('Displacement (nm)')
            ax.set_title('Exciton Diffusion')
            ax.grid(True, alpha=0.3)
        
        # 4. Dissociation yield vs field
        ax = plt.subplot(3, 3, 4)
        fields = np.linspace(0, 0.1, 50)  # V/nm
        yields = []
        
        for F in fields:
            diss_results = self.workflow.exciton_dynamics.calculate_dissociation_yield(F)
            yields.append(list(diss_results.values())[0]['dissociation_yield'])
        
        ax.plot(fields * 100, np.array(yields) * 100, 'b-')
        ax.set_xlabel('Electric Field (V/μm)')
        ax.set_ylabel('Dissociation Yield (%)')
        ax.set_title('Field-Dependent Dissociation')
        ax.grid(True, alpha=0.3)
        
        # 5. Mobility comparison
        ax = plt.subplot(3, 3, 5)
        materials = ['MAPI', 'FAPbI3', 'CsPbI3', 'GaAs', 'Si']
        mobilities_e = [10, 20, 5, 8500, 1400]
        mobilities_h = [1, 2, 0.5, 400, 450]
        
        x = np.arange(len(materials))
        width = 0.35
        
        ax.bar(x - width/2, mobilities_e, width, label='Electrons', color='blue')
        ax.bar(x + width/2, mobilities_h, width, label='Holes', color='red')
        ax.set_ylabel('Mobility (cm²/V·s)')
        ax.set_title('Carrier Mobilities Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(materials)
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 6. PCE summary
        ax = plt.subplot(3, 3, 6)
        pce_data = self.calculate_pce()
        
        metrics = ['PCE', 'SQ Limit']
        values = [pce_data['pce'], pce_data['sq_limit']]
        colors = ['green', 'gray']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        ax.set_ylabel('Efficiency (%)')
        ax.set_title('Device Performance')
        ax.set_ylim(0, 35)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', va='bottom')
        
        # 7. Band diagram sketch
        ax = plt.subplot(3, 3, 7)
        x = np.linspace(0, 100, 100)  # Device length in nm
        
        # Simplified band diagram
        cb = self.band_gap + 0.5 * x / 100  # Slight slope
        vb = 0.5 * x / 100
        
        ax.fill_between(x, vb, cb, alpha=0.3, color='gray', label='Band gap')
        ax.plot(x, cb, 'r-', linewidth=2, label='CB')
        ax.plot(x, vb, 'b-', linewidth=2, label='VB')
        
        # Electron and hole trajectories (schematic)
        ax.annotate('', xy=(80, cb[-1]-0.1), xytext=(20, cb[20]+0.3),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.annotate('', xy=(80, vb[-1]+0.1), xytext=(20, vb[20]-0.3),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        
        ax.set_xlabel('Position (nm)')
        ax.set_ylabel('Energy (eV)')
        ax.set_title('Device Band Diagram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 8. Temperature dependence
        ax = plt.subplot(3, 3, 8)
        temps = np.linspace(100, 400, 10)
        
        # Simplified temperature dependence
        # Voc decreases with temperature
        voc_t = 1.2 - 0.002 * (temps - 300)
        pce_t = pce_data['jsc'] * voc_t * pce_data['ff'] / 100 * 100
        
        ax.plot(temps, pce_t, 'g-', linewidth=2)
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('PCE (%)')
        ax.set_title('Temperature Dependence')
        ax.grid(True, alpha=0.3)
        
        # 9. Summary statistics
        ax = plt.subplot(3, 3, 9)
        ax.axis('off')
        
        summary_text = f"""
Photovoltaic Material: MAPbI3 Perovskite
{'='*40}
Band Gap: {self.band_gap:.2f} eV
Dielectric Constant: {self.dielectric_constant:.0f}
Exciton Binding: {self.exciton_binding:.3f} eV
Exciton Radius: {self.exciton_radius:.1f} nm

Electron Mobility: {self.electron_mobility:.1f} cm²/V·s
Hole Mobility: {self.hole_mobility:.1f} cm²/V·s

Power Conversion Efficiency: {pce_data['pce']:.1f}%
Shockley-Queisser Limit: {pce_data['sq_limit']:.1f}%
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/perovskite_analysis.png", dpi=150)
        plt.close()
        
        print(f"\nVisualizations saved to {output_dir}")


def run_perovskite_case_study():
    """Run complete perovskite solar cell case study."""
    
    print("="*70)
    print("PHOTOVOLTAIC CARRIER DYNAMICS CASE STUDY")
    print("Material: MAPbI3 Perovskite Solar Cell")
    print("="*70)
    
    # Create perovskite model
    cell = PerovskiteSolarCell()
    
    # Setup simulation
    cell.setup_simulation(temperature=300.0)
    
    # Run simulations
    print("\n" + "="*70)
    print("STEP 1: PHOTOGENERATION AND HOT CARRIER COOLING")
    print("="*70)
    photogen_results = cell.simulate_photogeneration(
        photon_energy=3.0,  # Visible light
        temperature=300.0
    )
    
    print("\n" + "="*70)
    print("STEP 2: EXCITON DISSOCIATION")
    print("="*70)
    dissociation_results = cell.simulate_exciton_dissociation(
        electric_field=0.05  # V/nm
    )
    
    print("\n" + "="*70)
    print("STEP 3: CHARGE TRANSPORT")
    print("="*70)
    transport_results = cell.simulate_charge_transport(
        electric_field=np.array([0.01, 0, 0]),
        simulation_time=1000.0
    )
    
    print("\n" + "="*70)
    print("STEP 4: DEVICE PERFORMANCE")
    print("="*70)
    pce_results = cell.calculate_pce()
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    cell.visualize_results()
    
    # Final summary
    print("\n" + "="*70)
    print("CASE STUDY COMPLETE")
    print("="*70)
    print(f"""
Summary of Results:
  - Hot carrier cooling completed in {photogen_results['total_cooling_time']:.0f} fs
  - Exciton dissociation yield: {dissociation_results['dissociation_yield']*100:.1f}%
  - Exciton diffusion length: {dissociation_results['diffusion_length']:.1f} nm
  - Estimated PCE: {pce_results['pce']:.1f}% (SQ limit: {pce_results['sq_limit']:.1f}%)
    """)
    
    return cell


if __name__ == "__main__":
    cell = run_perovskite_case_study()
