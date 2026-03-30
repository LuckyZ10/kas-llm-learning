"""
Exciton Dissociation in Organic Solar Cells
============================================

Case study: Exciton dissociation at donor-acceptor interfaces in 
organic photovoltaic materials (e.g., P3HT:PCBM).

Key processes:
- Frenkel exciton diffusion in donor
- Charge transfer state formation
- Charge separation at interface
- Geminate recombination

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
    ChargeSeparationState,
    EnergyTransferNetwork
)


class DonorAcceptorInterface:
    """
    Model for donor-acceptor interface in organic solar cells.
    
    Typical P3HT:PCBM parameters.
    """
    
    def __init__(self):
        # Donor (P3HT) parameters
        self.donor_gap = 2.0  # eV
        self.donor_exciton_binding = 0.5  # eV (strongly bound)
        self.donor_exciton_radius = 0.5  # nm (Frenkel exciton)
        self.donor_hole_mobility = 0.01  # cm^2/(V·s)
        
        # Acceptor (PCBM) parameters
        self.acceptor_gap = 2.3  # eV
        self.acceptor_electron_mobility = 0.01  # cm^2/(V·s)
        
        # Interface parameters
        self.lumo_offset = 1.0  # eV (driving force for separation)
        self.ct_state_energy = 1.6  # eV (charge transfer state)
        self.interface_width = 1.0  # nm
        
        # Exciton diffusion
        self.exciton_diffusion_length = 8.0  # nm
        self.exciton_lifetime = 300e-12  # s (300 ps)
        
        self.workflow = None
        
    def setup_simulation(self):
        """Setup excited state dynamics simulation."""
        
        self.workflow = ExcitedStateDynamicsWorkflow()
        
        # Donor exciton states
        exciton_energies = [self.donor_gap - self.donor_exciton_binding]
        binding_energies = [self.donor_exciton_binding]
        radii = [self.donor_exciton_radius]
        
        self.workflow.setup_exciton_system(
            exciton_energies=exciton_energies,
            binding_energies=binding_energies,
            radii=radii
        )
        
        print(f"Donor-Acceptor Interface Setup:")
        print(f"  Donor band gap: {self.donor_gap:.2f} eV")
        print(f"  Donor exciton binding: {self.donor_exciton_binding:.2f} eV")
        print(f"  LUMO offset: {self.lumo_offset:.2f} eV")
        print(f"  CT state energy: {self.ct_state_energy:.2f} eV")
        
    def simulate_exciton_diffusion(self,
                                   interface_distance: float = 8.0) -> Dict:
        """
        Simulate exciton diffusion to interface.
        
        Parameters
        ----------
        interface_distance : float
            Distance from generation to interface in nm
            
        Returns
        -------
        Dict with diffusion analysis
        """
        print(f"\n{'='*60}")
        print("EXCITON DIFFUSION TO INTERFACE")
        print(f"{'='*60}")
        print(f"Generation distance: {interface_distance:.1f} nm")
        print(f"Diffusion length: {self.exciton_diffusion_length:.1f} nm")
        
        # Calculate probability to reach interface
        # P = exp(-d/L_D)
        P_reach = np.exp(-interface_distance / self.exciton_diffusion_length)
        
        print(f"\nDiffusion Results:")
        print(f"  Probability to reach interface: {P_reach*100:.1f}%")
        
        # Simulate diffusion
        exciton = self.workflow.exciton_dynamics.exciton_states[0]
        times, positions = self.workflow.exciton_dynamics.simulate_diffusion(
            exciton, 1000.0
        )
        
        # Check if reaches interface
        final_distance = np.linalg.norm(positions[-1])
        reaches_interface = final_distance >= interface_distance
        
        print(f"  Final distance: {final_distance:.1f} nm")
        print(f"  Reaches interface: {reaches_interface}")
        
        return {
            'probability_to_reach': P_reach,
            'diffusion_trajectory': {'times': times, 'positions': positions},
            'reaches_interface': reaches_interface
        }
    
    def simulate_charge_transfer(self,
                                 electric_field: float = 0.0) -> Dict:
        """
        Simulate charge transfer state formation and separation.
        
        Parameters
        ----------
        electric_field : float
            Applied electric field in V/nm
            
        Returns
        -------
        Dict with CT analysis
        """
        print(f"\n{'='*60}")
        print("CHARGE TRANSFER STATE DYNAMICS")
        print(f"{'='*60}")
        
        # Energy levels at interface
        print(f"\nEnergy Level Diagram:")
        print(f"  Donor HOMO: 0.0 eV")
        print(f"  Donor LUMO: {self.donor_gap:.2f} eV")
        print(f"  Acceptor LUMO: {self.donor_gap - self.lumo_offset:.2f} eV")
        print(f"  CT state: {self.ct_state_energy:.2f} eV")
        
        # Driving force for charge separation
        driving_force = self.donor_gap - self.lumo_offset - self.ct_state_energy
        print(f"  Driving force for separation: {driving_force:.2f} eV")
        
        # Simulate charge separation
        # Create initial charge-separated state
        separation_distances = np.linspace(0.5, 5.0, 20)
        
        separation_probabilities = []
        for r in separation_distances:
            cs_state = ChargeSeparationState(
                electron_position=np.array([r, 0, 0]),
                hole_position=np.array([0, 0, 0]),
                electron_energy=self.donor_gap - self.lumo_offset,
                hole_energy=0.0,
                geminate=True
            )
            
            # Calculate escape probability
            P_escape = 1 - cs_state.get_recombination_probability()
            separation_probabilities.append(P_escape)
        
        separation_probabilities = np.array(separation_probabilities)
        
        # Optimal separation distance
        optimal_idx = np.argmax(separation_probabilities)
        optimal_distance = separation_distances[optimal_idx]
        max_separation_prob = separation_probabilities[optimal_idx]
        
        print(f"\nCharge Separation Results:")
        print(f"  Optimal separation: {optimal_distance:.1f} nm")
        print(f"  Maximum separation probability: {max_separation_prob*100:.1f}%")
        
        # Field-dependent separation
        if electric_field > 0:
            print(f"  Electric field: {electric_field*100:.1f} V/μm")
            # Field assists separation
            P_field = min(1.0, max_separation_prob * (1 + electric_field * 10))
            print(f"  Enhanced separation probability: {P_field*100:.1f}%")
        
        return {
            'driving_force': driving_force,
            'separation_distances': separation_distances,
            'separation_probabilities': separation_probabilities,
            'optimal_distance': optimal_distance,
            'max_separation_probability': max_separation_prob
        }
    
    def simulate_energy_transfer_cascade(self,
                                        n_sites: int = 20) -> Dict:
        """
        Simulate energy transfer cascade in donor aggregate.
        
        Parameters
        ----------
        n_sites : int
            Number of donor chromophores
            
        Returns
        -------
        Dict with energy transfer analysis
        """
        print(f"\n{'='*60}")
        print("ENERGY TRANSFER CASCADE")
        print(f"{'='*60}")
        
        # Setup energy funnel
        # Sites have progressively lower energy toward interface
        positions = np.zeros((n_sites, 3))
        for i in range(n_sites):
            positions[i] = [i * 0.5, 0, 0]  # Linear chain, 0.5 nm spacing
        
        # Energy gradient (funnel)
        energies = np.linspace(2.0, 1.8, n_sites)
        
        self.workflow.setup_energy_network(positions, energies)
        
        # Build rate matrix with FRET
        rates = self.workflow.energy_network.build_rate_matrix(
            mechanism="FRET",
            fret={'R0': 3.0}  # Förster radius 3 nm
        )
        
        # Simulate energy transfer
        P0 = np.zeros(n_sites)
        P0[0] = 1.0  # Excitation at one end
        
        times = np.linspace(0, 500, 500)  # fs
        populations = self.workflow.energy_network.simulate_energy_transfer(P0, times)
        
        # Transfer time to interface (last site)
        interface_pop = populations[:, -1]
        transfer_time = self.workflow.analyses.get('energy_transfer', {}).get(
            'transfer_time', times[np.argmax(interface_pop > 0.5 * interface_pop[-1])]
        )
        
        print(f"Energy Transfer Results:")
        print(f"  Number of sites: {n_sites}")
        print(f"  Energy gradient: {energies[0] - energies[-1]:.2f} eV")
        print(f"  Transfer time to interface: {transfer_time:.0f} fs")
        
        return {
            'times': times,
            'populations': populations,
            'transfer_time': transfer_time,
            'final_interface_population': interface_pop[-1]
        }
    
    def calculate_device_efficiency(self,
                                    jsc: float = 12.0,
                                    voc: float = 0.6,
                                    ff: float = 0.65) -> Dict:
        """
        Estimate device efficiency.
        
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
        Dict with efficiency data
        """
        print(f"\n{'='*60}")
        print("DEVICE EFFICIENCY")
        print(f"{'='*60}")
        
        P_in = 100.0  # mW/cm²
        pce = jsc * voc * ff / P_in * 100
        
        print(f"Device Parameters:")
        print(f"  Jsc: {jsc:.1f} mA/cm²")
        print(f"  Voc: {voc:.2f} V")
        print(f"  FF: {ff*100:.1f}%")
        print(f"\nPower Conversion Efficiency: {pce:.2f}%")
        
        return {
            'pce': pce,
            'jsc': jsc,
            'voc': voc,
            'ff': ff
        }
    
    def visualize_results(self, output_dir: str = "./exciton_dissociation_results"):
        """Generate visualizations."""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Energy level diagram
        ax = plt.subplot(2, 3, 1)
        
        # Donor
        ax.barh(0, 1, height=0.1, color='blue', alpha=0.5, label='Donor HOMO')
        ax.barh(self.donor_gap, 1, height=0.1, color='lightblue', alpha=0.5, label='Donor LUMO')
        
        # Acceptor
        ax.barh(self.donor_gap - self.lumo_offset, 1, height=0.1, 
               left=1.5, color='red', alpha=0.5, label='Acceptor LUMO')
        ax.barh(-0.3, 1, height=0.1, left=1.5, color='orange', alpha=0.5, label='Acceptor HOMO')
        
        # Exciton and CT
        ax.axhline(self.donor_gap - self.donor_exciton_binding, 
                  color='green', linestyle='--', label='Exciton')
        ax.axhline(self.ct_state_energy, color='purple', linestyle=':', label='CT State')
        
        ax.set_xlim(-0.5, 3)
        ax.set_ylim(-0.5, 2.5)
        ax.set_ylabel('Energy (eV)')
        ax.set_title('Energy Level Diagram')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xticks([])
        
        # 2. Exciton diffusion
        ax = plt.subplot(2, 3, 2)
        distances = np.linspace(0, 20, 100)
        P_survive = np.exp(-distances / self.exciton_diffusion_length)
        
        ax.plot(distances, P_survive * 100, 'b-', linewidth=2)
        ax.axvline(self.exciton_diffusion_length, color='r', linestyle='--',
                  label=f'L_D = {self.exciton_diffusion_length} nm')
        ax.set_xlabel('Distance to Interface (nm)')
        ax.set_ylabel('Survival Probability (%)')
        ax.set_title('Exciton Diffusion Limited Collection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Charge separation probability
        ax = plt.subplot(2, 3, 3)
        
        if 'ct_analysis' in dir(self) and self.ct_analysis:
            data = self.ct_analysis
            ax.plot(data['separation_distances'], 
                   data['separation_probabilities'] * 100, 'g-', linewidth=2)
            ax.axvline(data['optimal_distance'], color='r', linestyle='--',
                      label=f"Optimal: {data['optimal_distance']:.1f} nm")
        else:
            # Simplified calculation
            r = np.linspace(0.5, 5, 50)
            r_c = 1.44 / 3.0  # Coulomb radius
            P = 1 - np.exp(-r_c / r)
            ax.plot(r, P * 100, 'g-', linewidth=2)
        
        ax.set_xlabel('e-h Separation (nm)')
        ax.set_ylabel('Separation Probability (%)')
        ax.set_title('Charge Separation vs Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Energy transfer cascade
        ax = plt.subplot(2, 3, 4)
        
        if hasattr(self.workflow, 'analyses') and 'energy_transfer' in self.workflow.analyses:
            data = self.workflow.analyses['energy_transfer']
            times = data['times']
            pops = data['populations']
            
            for i in [0, 5, 10, 15, 19]:
                ax.plot(times, pops[:, i], label=f'Site {i}')
            
            ax.set_xlabel('Time (fs)')
            ax.set_ylabel('Population')
            ax.set_title('Energy Transfer Cascade')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # 5. Process efficiencies
        ax = plt.subplot(2, 3, 5)
        
        processes = ['Absorption', 'Diffusion', 'CT Formation', 'Separation', 'Collection']
        efficiencies = [100, 90, 85, 70, 65]  # Example values
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(processes)))
        
        bars = ax.barh(processes, efficiencies, color=colors)
        ax.set_xlabel('Efficiency (%)')
        ax.set_title('Process Efficiencies')
        ax.set_xlim(0, 100)
        
        for bar, eff in zip(bars, efficiencies):
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                   f'{eff}%', va='center')
        
        # 6. Summary
        ax = plt.subplot(2, 3, 6)
        ax.axis('off')
        
        summary = f"""
Organic Solar Cell Interface
{'='*30}
Donor: P3HT
Acceptor: PCBM

Exciton Binding: {self.donor_exciton_binding:.2f} eV
Diffusion Length: {self.exciton_diffusion_length:.1f} nm
LUMO Offset: {self.lumo_offset:.2f} eV

Key Challenges:
• Short exciton diffusion
• Geminate recombination
• Energetic disorder
        """
        
        ax.text(0.1, 0.9, summary, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/exciton_dissociation_analysis.png", dpi=150)
        plt.close()
        
        print(f"Visualizations saved to {output_dir}")


def run_exciton_dissociation_case_study():
    """Run complete exciton dissociation case study."""
    
    print("="*70)
    print("EXCITON DISSOCIATION IN ORGANIC SOLAR CELLS")
    print("Interface: P3HT:PCBM Donor-Acceptor")
    print("="*70)
    
    # Create interface model
    interface = DonorAcceptorInterface()
    interface.setup_simulation()
    
    # Run simulations
    print("\n" + "="*70)
    print("STEP 1: EXCITON DIFFUSION TO INTERFACE")
    print("="*70)
    diffusion_results = interface.simulate_exciton_diffusion(
        interface_distance=8.0
    )
    
    print("\n" + "="*70)
    print("STEP 2: CHARGE TRANSFER DYNAMICS")
    print("="*70)
    interface.ct_analysis = interface.simulate_charge_transfer(
        electric_field=0.0
    )
    
    print("\n" + "="*70)
    print("STEP 3: ENERGY TRANSFER CASCADE")
    print("="*70)
    cascade_results = interface.simulate_energy_transfer_cascade(n_sites=20)
    
    print("\n" + "="*70)
    print("STEP 4: DEVICE EFFICIENCY")
    print("="*70)
    efficiency_results = interface.calculate_device_efficiency()
    
    # Visualize
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    interface.visualize_results()
    
    # Summary
    print("\n" + "="*70)
    print("CASE STUDY COMPLETE")
    print("="*70)
    print(f"""
Summary of Results:
  - Diffusion to interface: {diffusion_results['probability_to_reach']*100:.1f}% yield
  - Charge separation: {interface.ct_analysis['max_separation_probability']*100:.1f}% yield
  - Energy transfer time: {cascade_results['transfer_time']:.0f} fs
  - Estimated PCE: {efficiency_results['pce']:.1f}%
    """)
    
    return interface


if __name__ == "__main__":
    interface = run_exciton_dissociation_case_study()
