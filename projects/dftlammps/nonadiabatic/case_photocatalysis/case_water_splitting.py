"""
Photocatalytic Reaction Mechanism Case Study
============================================

Simulation of photocatalytic processes including:
- Light harvesting and excited state formation
- Electron transfer to catalytic sites
- Proton-coupled electron transfer (PCET)
- Reaction kinetics and quantum yields

Example: Water splitting on semiconductor photocatalysts

Author: dftlammps development team
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nonadiabatic.excited_state_dynamics import (
    ExcitedStateDynamicsWorkflow,
    ExcitonState,
    CarrierState,
    ExcitedProcess
)
from nonadiabatic.sharc_interface import SHARCConfig, MultiReferenceMethod


class PhotocatalystSystem:
    """
    Model for semiconductor photocatalyst (e.g., TiO2, CdS).
    
    Simulates complete photocatalytic cycle from light absorption
to chemical reaction.
    """
    
    def __init__(self):
        # Semiconductor parameters (TiO2 anatase)
        self.band_gap = 3.2  # eV
        self.cbm_potential = -0.1  # V vs NHE (conduction band)
        self.vbm_potential = 3.1  # V vs NHE (valence band)
        self.dielectric_constant = 80.0
        
        # Surface catalytic sites
        self.cocatalyst = "Pt"  # Hydrogen evolution cocatalyst
        self.cocatalyst_workfunction = 5.6  # eV
        
        # Redox potentials
        self.h2o_h2_potential = 0.0  # V vs NHE (pH=0)
        self.o2_h2o_potential = 1.23  # V vs NHE
        
        # Reaction kinetics
        self.electron_transfer_rate = 1e12  # s^-1
        self.hole_transfer_rate = 1e11  # s^-1
        self.recombination_rate = 1e9  # s^-1
        
        self.workflow = None
        
    def setup_simulation(self):
        """Setup photocatalysis simulation."""
        
        self.workflow = ExcitedStateDynamicsWorkflow()
        
        # Carrier states in semiconductor
        electron_energies = np.linspace(0, 1.0, 5)  # Above CBM
        hole_energies = np.linspace(0, 0.5, 3)  # Below VBM
        
        self.workflow.setup_carrier_system(
            electron_masses=[0.5] * len(electron_energies),
            hole_masses=[1.0] * len(hole_energies),
            mobilities={'electron': 0.1, 'hole': 0.01}
        )
        
        print(f"Photocatalyst System Setup:")
        print(f"  Material: TiO2 (anatase)")
        print(f"  Band gap: {self.band_gap:.2f} eV")
        print(f"  CBM: {self.cbm_potential:.2f} V vs NHE")
        print(f"  VBM: {self.vbm_potential:.2f} V vs NHE")
        print(f"  Cocatalyst: {self.cocatalyst}")
        
    def calculate_thermodynamic_feasibility(self) -> Dict:
        """
        Check thermodynamic feasibility of water splitting.
        
        Returns
        -------
        Dict with thermodynamic analysis
        """
        print(f"\n{'='*60}")
        print("THERMODYNAMIC FEASIBILITY ANALYSIS")
        print(f"{'='*60}")
        
        # Water splitting requirements
        water_splitting_potential = self.o2_h2o_potential - self.h2o_h2_potential
        
        print(f"\nBand Edge Alignment:")
        print(f"  CBM: {self.cbm_potential:.2f} V vs NHE")
        print(f"  H+/H2: {self.h2o_h2_potential:.2f} V vs NHE")
        print(f"  → Proton reduction: {'FEASIBLE' if self.cbm_potential < self.h2o_h2_potential else 'NOT FEASIBLE'}")
        
        print(f"\n  VBM: {self.vbm_potential:.2f} V vs NHE")
        print(f"  O2/H2O: {self.o2_h2o_potential:.2f} V vs NHE")
        print(f"  → Water oxidation: {'FEASIBLE' if self.vbm_potential > self.o2_h2o_potential else 'NOT FEASIBLE'}")
        
        # Overpotentials
        h2_overpotential = self.h2o_h2_potential - self.cbm_potential
        o2_overpotential = self.vbm_potential - self.o2_h2o_potential
        
        print(f"\nOverpotentials:")
        print(f"  H2 evolution: {h2_overpotential:.2f} V")
        print(f"  O2 evolution: {o2_overpotential:.2f} V")
        
        # Light requirement
        min_photon_energy = max(water_splitting_potential, self.band_gap)
        max_wavelength = 1240 / min_photon_energy
        
        print(f"\nLight Requirements:")
        print(f"  Minimum photon energy: {min_photon_energy:.2f} eV")
        print(f"  Maximum wavelength: {max_wavelength:.0f} nm")
        print(f"  Visible light active: {'YES' if max_wavelength > 400 else 'NO'}")
        
        return {
            'proton_reduction_feasible': self.cbm_potential < self.h2o_h2_potential,
            'water_oxidation_feasible': self.vbm_potential > self.o2_h2o_potential,
            'h2_overpotential': h2_overpotential,
            'o2_overpotential': o2_overpotential,
            'min_photon_energy': min_photon_energy,
            'max_wavelength': max_wavelength
        }
    
    def simulate_charge_separation(self,
                                   photon_energy: float = 3.5) -> Dict:
        """
        Simulate charge separation after photoexcitation.
        
        Parameters
        ----------
        photon_energy : float
            Incident photon energy in eV
            
        Returns
        -------
        Dict with charge separation analysis
        """
        print(f"\n{'='*60}")
        print("CHARGE SEPARATION DYNAMICS")
        print(f"{'='*60}")
        
        excess_energy = photon_energy - self.band_gap
        print(f"\nExcitation:")
        print(f"  Photon energy: {photon_energy:.2f} eV")
        print(f"  Excess energy: {excess_energy:.2f} eV")
        
        # Hot carrier cooling
        cooling_results = self.workflow.run_carrier_relaxation(
            initial_electron_energy=excess_energy * 0.6,
            initial_hole_energy=excess_energy * 0.4
        )
        
        print(f"\nCooling Dynamics:")
        print(f"  Electron cooling: {cooling_results['electron_relaxation']['cooling_time']:.0f} fs")
        print(f"  Hole cooling: {cooling_results['hole_relaxation']['cooling_time']:.0f} fs")
        
        # Charge separation efficiency
        # Competition between separation and recombination
        k_sep = self.electron_transfer_rate
        k_rec = self.recombination_rate
        
        separation_efficiency = k_sep / (k_sep + k_rec)
        
        print(f"\nSepetition Competition:")
        print(f"  Separation rate: {k_sep:.2e} s^-1")
        print(f"  Recombination rate: {k_rec:.2e} s^-1")
        print(f"  Separation efficiency: {separation_efficiency*100:.1f}%")
        
        return {
            'cooling_results': cooling_results,
            'separation_efficiency': separation_efficiency,
            'recombination_yield': 1 - separation_efficiency
        }
    
    def simulate_catalytic_cycle(self,
                                 temperature: float = 300.0) -> Dict:
        """
        Simulate catalytic reaction cycle.
        
        Parameters
        ----------
        temperature : float
            Reaction temperature in K
            
        Returns
        -------
        Dict with catalytic cycle analysis
        """
        print(f"\n{'='*60}")
        print("CATALYTIC CYCLE SIMULATION")
        print(f"{'='*60}")
        
        # Simplified kinetic model
        # States: S0 (ground), S1 (excited), CS (charge separated), P (products)
        
        times = np.linspace(0, 1000, 1000)  # ps
        
        # Rate constants
        k_exc = 1e12  # Excitation rate (under illumination)
        k_relax = 1e10  # Relaxation to ground state
        k_et = self.electron_transfer_rate  # Electron transfer to catalyst
        k_ht = self.hole_transfer_rate  # Hole transfer to solution
        k_rec = self.recombination_rate  # Recombination
        k_cat = 1e3  # Catalytic turnover
        
        # Simplified kinetic equations
        populations = np.zeros((len(times), 4))
        populations[0, 0] = 1.0  # Start in ground state
        
        dt = times[1] - times[0]
        
        for i in range(1, len(times)):
            P_gs = populations[i-1, 0]
            P_ex = populations[i-1, 1]
            P_cs = populations[i-1, 2]
            P_prod = populations[i-1, 3]
            
            # Rate equations
            dP_gs = -k_exc * P_gs * 1e-12 * dt + k_relax * P_ex * 1e-12 * dt
            dP_ex = k_exc * P_gs * 1e-12 * dt - (k_relax + k_et) * P_ex * 1e-12 * dt
            dP_cs = k_et * P_ex * 1e-12 * dt - (k_ht + k_rec) * P_cs * 1e-12 * dt
            dP_prod = k_cat * P_cs * 1e-12 * dt
            
            populations[i, 0] = max(0, P_gs + dP_gs)
            populations[i, 1] = max(0, P_ex + dP_ex)
            populations[i, 2] = max(0, P_cs + dP_cs)
            populations[i, 3] = P_prod + dP_prod
        
        # Quantum yield
        qy = populations[-1, 3] / (np.sum(populations[0]) - populations[-1, 0])
        
        print(f"\nCatalytic Cycle Results:")
        print(f"  Final product yield: {populations[-1, 3]*100:.2f}%")
        print(f"  Quantum yield: {qy*100:.3f}%")
        print(f"  Turnover frequency: {k_cat:.2e} s^-1")
        
        return {
            'times': times,
            'populations': populations,
            'quantum_yield': qy,
            'turnover_frequency': k_cat
        }
    
    def analyze_sacrificial_reagents(self,
                                     donor_potential: float = 0.5) -> Dict:
        """
        Analyze sacrificial electron donor system.
        
        Parameters
        ----------
        donor_potential : float
            Redox potential of sacrificial donor in V vs NHE
            
        Returns
        -------
        Dict with sacrificial system analysis
        """
        print(f"\n{'='*60}")
        print("SACRIFICIAL REAGENT SYSTEM")
        print(f"{'='*60}")
        
        # Common sacrificial donors
        donors = {
            'TEOA': 0.8,
            'MeOH': 0.9,
            'EDTA': 0.4,
            'Ascorbic Acid': 0.3
        }
        
        print(f"\nSacrificial Donors:")
        for name, potential in donors.items():
            can_oxidize = self.vbm_potential > potential
            print(f"  {name}: E = {potential:.2f} V, "
                  f"Oxidizable: {'YES' if can_oxidize else 'NO'}")
        
        # Select donor
        print(f"\nSelected Donor:")
        print(f"  Potential: {donor_potential:.2f} V vs NHE")
        print(f"  Hole scavenging: {'FEASIBLE' if self.vbm_potential > donor_potential else 'NOT FEASIBLE'}")
        
        # With sacrificial donor, only H2 evolution needs to be considered
        h2_efficiency = 0.95  # High efficiency with good hole scavenger
        
        print(f"\nExpected H2 Evolution Efficiency: {h2_efficiency*100:.1f}%")
        
        return {
            'donors': donors,
            'selected_donor_potential': donor_potential,
            'expected_efficiency': h2_efficiency
        }
    
    def calculate_solar_to_hydrogen_efficiency(self,
                                               absorption_efficiency: float = 0.8,
                                               charge_separation_efficiency: float = 0.6,
                                               catalytic_efficiency: float = 0.5) -> Dict:
        """
        Calculate solar-to-hydrogen (STH) conversion efficiency.
        
        Parameters
        ----------
        absorption_efficiency : float
            Light absorption efficiency
        charge_separation_efficiency : float
            Charge separation quantum yield
        catalytic_efficiency : float
            Catalytic reaction efficiency
            
        Returns
        -------
        Dict with STH efficiency
        """
        print(f"\n{'='*60}")
        print("SOLAR-TO-HYDROGEN EFFICIENCY")
        print(f"{'='*60}")
        
        # STH efficiency calculation
        # η_STH = η_abs × η_sep × η_cat × η_thermo
        
        thermodynamic_efficiency = 1.23 / self.band_gap  # Maximum theoretical
        
        sth_efficiency = (absorption_efficiency * 
                         charge_separation_efficiency * 
                         catalytic_efficiency * 
                         thermodynamic_efficiency)
        
        print(f"Efficiency Analysis:")
        print(f"  Light absorption: {absorption_efficiency*100:.1f}%")
        print(f"  Charge separation: {charge_separation_efficiency*100:.1f}%")
        print(f"  Catalytic efficiency: {catalytic_efficiency*100:.1f}%")
        print(f"  Thermodynamic factor: {thermodynamic_efficiency*100:.1f}%")
        print(f"\n  STH Efficiency: {sth_efficiency*100:.2f}%")
        
        # Benchmark comparison
        print(f"\nBenchmark Comparison:")
        print(f"  Current record (particulate): ~2%")
        print(f"  This system: {sth_efficiency*100:.2f}%")
        print(f"  Target (DOE): 10%")
        
        return {
            'sth_efficiency': sth_efficiency,
            'absorption_efficiency': absorption_efficiency,
            'separation_efficiency': charge_separation_efficiency,
            'catalytic_efficiency': catalytic_efficiency,
            'thermodynamic_efficiency': thermodynamic_efficiency
        }
    
    def visualize_results(self, output_dir: str = "./photocatalysis_results"):
        """Generate visualizations."""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Band diagram and redox levels
        ax = plt.subplot(2, 3, 1)
        
        # Semiconductor bands
        ax.barh(self.vbm_potential, 2, height=0.1, color='blue', alpha=0.5)
        ax.barh(self.cbm_potential, 2, height=0.1, color='red', alpha=0.5)
        ax.fill_betweenx([self.cbm_potential, self.vbm_potential], 0, 2, alpha=0.1, color='gray')
        
        # Redox levels
        ax.axhline(self.h2o_h2_potential, color='green', linestyle='--', label='H+/H2')
        ax.axhline(self.o2_h2o_potential, color='orange', linestyle='--', label='O2/H2O')
        
        # Band gap arrow
        ax.annotate('', xy=(1, self.vbm_potential), xytext=(1, self.cbm_potential),
                   arrowprops=dict(arrowstyle='<->', color='black', lw=2))
        ax.text(1.5, (self.vbm_potential + self.cbm_potential)/2, 
               f'{self.band_gap:.1f} eV', fontsize=10)
        
        ax.set_xlim(0, 2.5)
        ax.set_ylim(-0.5, 3.5)
        ax.set_ylabel('Potential (V vs NHE)')
        ax.set_title('Band Edge Alignment')
        ax.legend()
        ax.set_xticks([])
        
        # 2. Kinetic scheme
        ax = plt.subplot(2, 3, 2)
        ax.axis('off')
        
        scheme = """
Photocatalytic Cycle:

  hν
S0 →→ S1 (Excitation)
       ↓
       CS (Charge Separation)
      / \\
     /   \\
    ↓     ↓
   e-     h+
   ↓       ↓
  H2O    H2O
   ↓       ↓
  ½H2   ½O2
        
Quantum Yield: η = k_cat/(k_cat + k_rec)
        """
        ax.text(0.1, 0.9, scheme, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        # 3. Charge separation dynamics
        ax = plt.subplot(2, 3, 3)
        
        times = np.linspace(0, 100, 100)  # ps
        P_sep = 1 - np.exp(-self.electron_transfer_rate * 1e-12 * times)
        P_rec = 1 - np.exp(-self.recombination_rate * 1e-12 * times)
        
        ax.plot(times, P_sep, 'g-', label='Separation', linewidth=2)
        ax.plot(times, P_rec, 'r--', label='Recombination', linewidth=2)
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Probability')
        ax.set_title('Competing Processes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Catalytic cycle populations
        ax = plt.subplot(2, 3, 4)
        
        # Simplified population evolution
        t = np.linspace(0, 1000, 1000)  # ps
        P_ground = np.exp(-1e-3 * t)
        P_excited = (1 - np.exp(-1e-3 * t)) * np.exp(-1e-1 * t)
        P_separated = (1 - np.exp(-1e-1 * t)) * (1 - np.exp(-1e-3 * t))
        P_product = 1 - P_ground - P_excited - P_separated
        
        ax.plot(t, P_ground, label='Ground', color='blue')
        ax.plot(t, P_excited, label='Excited', color='red')
        ax.plot(t, P_separated, label='Charge Sep.', color='green')
        ax.plot(t, P_product, label='Products', color='purple')
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Population')
        ax.set_title('Catalytic Cycle Populations')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 5. Efficiency breakdown
        ax = plt.subplot(2, 3, 5)
        
        processes = ['Absorption', 'Separation', 'Catalysis', 'STH']
        efficiencies = [80, 60, 50, 10]  # Example values
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        
        bars = ax.bar(processes, efficiencies, color=colors, alpha=0.7)
        ax.set_ylabel('Efficiency (%)')
        ax.set_title('Efficiency Analysis')
        ax.set_ylim(0, 100)
        
        for bar, eff in zip(bars, efficiencies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{eff}%', ha='center')
        
        # 6. Summary table
        ax = plt.subplot(2, 3, 6)
        ax.axis('off')
        
        summary = f"""
Photocatalysis Summary
{'='*30}
Material: TiO₂ (anatase)
Band Gap: {self.band_gap:.1f} eV
CBM: {self.cbm_potential:.2f} V vs NHE
VBM: {self.vbm_potential:.2f} V vs NHE

Feasibility:
  H₂ evolution: YES
  O₂ evolution: YES

Key Rates:
  k_et = {self.electron_transfer_rate:.0e} s⁻¹
  k_rec = {self.recombination_rate:.0e} s⁻¹

Target: STH > 10%
        """
        
        ax.text(0.1, 0.9, summary, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/photocatalysis_analysis.png", dpi=150)
        plt.close()
        
        print(f"Visualizations saved to {output_dir}")


def run_photocatalysis_case_study():
    """Run complete photocatalysis case study."""
    
    print("="*70)
    print("PHOTOCATALYTIC REACTION MECHANISM CASE STUDY")
    print("System: TiO2 Water Splitting")
    print("="*70)
    
    # Create photocatalyst system
    catalyst = PhotocatalystSystem()
    catalyst.setup_simulation()
    
    # Run simulations
    print("\n" + "="*70)
    print("STEP 1: THERMODYNAMIC FEASIBILITY")
    print("="*70)
    thermo_results = catalyst.calculate_thermodynamic_feasibility()
    
    print("\n" + "="*70)
    print("STEP 2: CHARGE SEPARATION")
    print("="*70)
    separation_results = catalyst.simulate_charge_separation(
        photon_energy=3.5
    )
    
    print("\n" + "="*70)
    print("STEP 3: CATALYTIC CYCLE")
    print("="*70)
    catalytic_results = catalyst.simulate_catalytic_cycle()
    
    print("\n" + "="*70)
    print("STEP 4: SACRIFICIAL SYSTEM")
    print("="*70)
    sacrificial_results = catalyst.analyze_sacrificial_reagents()
    
    print("\n" + "="*70)
    print("STEP 5: STH EFFICIENCY")
    print("="*70)
    sth_results = catalyst.calculate_solar_to_hydrogen_efficiency()
    
    # Visualize
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    catalyst.visualize_results()
    
    # Summary
    print("\n" + "="*70)
    print("CASE STUDY COMPLETE")
    print("="*70)
    print(f"""
Summary of Results:
  - Water splitting: Thermodynamically {'FEASIBLE' if thermo_results['proton_reduction_feasible'] and thermo_results['water_oxidation_feasible'] else 'NOT FEASIBLE'}
  - Charge separation efficiency: {separation_results['separation_efficiency']*100:.1f}%
  - Catalytic quantum yield: {catalytic_results['quantum_yield']*100:.3f}%
  - STH efficiency: {sth_results['sth_efficiency']*100:.2f}%
    """)
    
    return catalyst


if __name__ == "__main__":
    catalyst = run_photocatalysis_case_study()
