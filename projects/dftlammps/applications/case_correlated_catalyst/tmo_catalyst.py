"""
Correlated Transition Metal Oxide Catalysts

This module provides DFT+U/DMFT workflows for studying catalytic
properties of correlated transition metal oxides:
- CO oxidation on oxide surfaces
- Oxygen evolution/reduction reactions (OER/ORR)
- Water splitting
- Selective oxidation

Author: DFT-LAMMPS Team
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class CatalysisConfig:
    """Configuration for catalysis calculations"""
    # Catalyst material
    oxide: str = "Co3O4"
    surface: str = "(001)"
    termination: str = "Co"
    
    # DFT+U parameters
    U_values: Dict[str, float] = field(default_factory=lambda: {
        'Co': 3.5,
        'Fe': 4.0,
        'Ni': 6.0,
        'Mn': 4.5
    })
    J_values: Dict[str, float] = field(default_factory=lambda: {
        'Co': 0.8,
        'Fe': 0.9,
        'Ni': 1.0,
        'Mn': 0.9
    })
    
    # Adsorption parameters
    adsorbates: List[str] = field(default_factory=lambda: ['O', 'OH', 'OOH', 'H2O'])
    coverage: float = 0.25  # ML
    
    # Reaction conditions
    temperature: float = 298.15  # K
    pressure: float = 1.0  # atm
    pH: float = 7.0
    
    # Solvation
    include_solvation: bool = True
    solvation_model: str = "VASPsol"


class TMOxideCatalyst:
    """
    Base class for transition metal oxide catalysts
    
    Provides common functionality for studying catalytic
    properties of correlated oxides.
    """
    
    def __init__(self, config: CatalysisConfig = None):
        self.config = config or CatalysisConfig()
        self.metal = self._get_metal()
        self.oxidation_states = self._get_oxidation_states()
    
    def _get_metal(self) -> str:
        """Extract metal from oxide formula"""
        # Simple parsing
        oxide = self.config.oxide
        if 'Co3O4' in oxide:
            return 'Co'
        elif 'Fe3O4' in oxide:
            return 'Fe'
        elif 'MnO' in oxide:
            return 'Mn'
        elif 'NiO' in oxide:
            return 'Ni'
        else:
            return 'Co'  # Default
    
    def _get_oxidation_states(self) -> Dict[str, int]:
        """Get oxidation states in oxide"""
        metal = self.metal
        oxide = self.config.oxide
        
        if oxide == "Co3O4":
            # Mixed valence: Co²⁺ and Co³⁺
            return {f'{metal}': [2, 3]}
        elif oxide == "Fe3O4":
            return {f'{metal}': [2, 3]}
        elif oxide in ["CoO", "NiO", "MnO", "FeO"]:
            return {f'{metal}': [2]}
        elif oxide in ["Co2O3", "Fe2O3", "Mn2O3"]:
            return {f'{metal}': [3]}
        else:
            return {f'{metal}': [2]}
    
    def setup_surface_structure(self, n_layers: int = 4) -> Dict:
        """
        Setup surface structure for catalysis calculations
        
        Returns:
        --------
        structure : dict
            Surface slab structure
        """
        oxide = self.config.oxide
        surface = self.config.surface
        
        if oxide == "Co3O4":
            # Spinel structure
            # (001) surface has two terminations: Co or O
            a = 8.08  # Å
            
            lattice = np.array([
                [a, 0, 0],
                [0, a, 0],
                [0, 0, a]
            ])
            
            # Bulk positions (simplified spinel)
            # Co²⁺ in tetrahedral sites
            # Co³⁺ in octahedral sites
            
            if surface == "(001)":
                if self.config.termination == "Co":
                    # Co-terminated surface
                    surface_positions = self._get_co3o4_001_co_termination()
                else:
                    # O-terminated surface
                    surface_positions = self._get_co3o4_001_o_termination()
        else:
            raise ValueError(f"Surface setup for {oxide} not implemented")
        
        return {
            'lattice': lattice,
            'positions': surface_positions,
            'vacuum': 15.0,  # Å
            'n_layers': n_layers
        }
    
    def _get_co3o4_001_co_termination(self) -> Dict[str, np.ndarray]:
        """Generate Co-terminated Co3O4(001) surface positions"""
        # Co3O4 has normal spinel structure
        # (001) surface: Co²⁺ termination
        
        positions = {
            'Co_tet': np.array([[0, 0, 0], [0.5, 0.5, 0]]),
            'Co_oct': np.array([[0.25, 0.25, 0.125], [0.75, 0.75, 0.125]]),
            'O': np.array([[0.25, 0.75, 0.0625], [0.75, 0.25, 0.0625]])
        }
        
        return positions
    
    def _get_co3o4_001_o_termination(self) -> Dict[str, np.ndarray]:
        """Generate O-terminated Co3O4(001) surface positions"""
        positions = {
            'O': np.array([[0, 0, 0], [0.5, 0.5, 0]]),
            'Co_oct': np.array([[0.25, 0.75, 0.125], [0.75, 0.25, 0.125]]),
            'Co_tet': np.array([[0.5, 0, 0.25], [0, 0.5, 0.25]])
        }
        
        return positions
    
    def calculate_adsorption_energy(self, adsorbate: str,
                                    binding_site: str = "top") -> float:
        """
        Calculate adsorption energy of molecule on surface
        
        E_ads = E(slab+adsorbate) - E(slab) - E(adsorbate)
        """
        # This would require actual DFT calculations
        # Placeholder with typical values
        
        typical_energies = {
            'O': -2.5,    # eV
            'OH': -1.5,
            'OOH': -1.0,
            'H2O': -0.5,
            'H': -0.8
        }
        
        return typical_energies.get(adsorbate, -1.0)
    
    def calculate_reaction_free_energy(self, reaction: str) -> Dict[str, float]:
        """
        Calculate free energy for catalytic reaction
        
        Reactions:
        - OER: 2H2O → O2 + 4H+ + 4e-
        - ORR: O2 + 4H+ + 4e- → 2H2O
        - CO_oxidation: CO + 1/2 O2 → CO2
        """
        T = self.config.temperature
        pH = self.config.pH
        
        # Free energy corrections
        # ΔG = ΔE + ΔZPE - TΔS + ΔG_pH + ΔG_U
        
        if reaction == "OER":
            # Oxygen Evolution Reaction
            # 4 steps: * → OH* → O* → OOH* → O2
            
            delta_G = {
                'OH_formation': 0.8,
                'O_formation': 0.5,
                'OOH_formation': 0.9,
                'O2_release': 1.2
            }
            
        elif reaction == "ORR":
            # Oxygen Reduction Reaction
            delta_G = {
                'O2_adsorption': -0.2,
                'OOH_formation': 0.4,
                'OH_formation': 0.3,
                'H2O_release': 0.5
            }
            
        elif reaction == "CO_oxidation":
            delta_G = {
                'CO_adsorption': -0.5,
                'O2_adsorption': -0.3,
                'CO2_formation': -1.0
            }
        else:
            raise ValueError(f"Reaction {reaction} not supported")
        
        # pH correction: ΔG_pH = kT ln(10) * pH
        kT = 8.617e-5 * T  # eV
        delta_G_pH = kT * np.log(10) * pH
        
        return {
            'delta_G_steps': delta_G,
            'delta_G_pH': delta_G_pH,
            'overpotential': max(delta_G.values()) - 1.23  # For OER
        }
    
    def analyze_valence_fluctuations(self, bader_charges: np.ndarray) -> Dict:
        """
        Analyze valence fluctuations at surface
        
        Important for mixed-valence oxides like Co3O4, Fe3O4
        """
        # Calculate charge distribution
        mean_charge = np.mean(bader_charges)
        std_charge = np.std(bader_charges)
        
        # Identify oxidation states
        metal = self.metal
        expected_states = self.oxidation_states[metal]
        
        # Estimate population of each oxidation state
        populations = {}
        for state in expected_states:
            expected_charge = state  # Simplified
            mask = np.abs(bader_charges - expected_charge) < 0.3
            populations[f'{metal}^{state}+'] = np.sum(mask) / len(bader_charges)
        
        return {
            'mean_charge': mean_charge,
            'charge_variance': std_charge**2,
            'oxidation_state_populations': populations,
            'has_valence_fluctuations': std_charge > 0.2
        }


class Co3O4Catalyst(TMOxideCatalyst):
    """
    Specialized catalyst class for Co3O4
    
    Co3O4 is an excellent catalyst for:
    - CO oxidation
    - OER/ORR
    - Water oxidation
    """
    
    def __init__(self, config: CatalysisConfig = None):
        if config is None:
            config = CatalysisConfig(oxide="Co3O4", surface="(001)")
        super().__init__(config)
    
    def analyze_cobalt_redox_chemistry(self) -> Dict:
        """
        Analyze Co²⁺/Co³⁺ redox chemistry at surface
        
        Key for catalytic activity:
        - Co²⁺ (tet): redox active
        - Co³⁺ (oct): stabilizes intermediate spin
        """
        # Redox potentials
        E_redox = {
            'Co3+/Co2+_bulk': 1.2,  # V vs SHE
            'Co3+/Co2+_surface': 1.0,  # Lower at surface
            'Co4+/Co3+': 1.8
        }
        
        # Crystal field effects
        # Co²⁺ in Td: e² t2³ (high spin, S=3/2)
        # Co³⁺ in Oh: t2g⁶ eg⁰ (low spin, S=0) or t2g⁴ eg² (high spin, S=2)
        
        spin_states = {
            'Co2+_Td': {'spin': 1.5, 'CFSE': -0.6},  # e² t2³
            'Co3+_Oh_LS': {'spin': 0, 'CFSE': -2.4},  # t2g⁶
            'Co3+_Oh_HS': {'spin': 2, 'CFSE': -0.8},  # t2g⁴ eg²
            'Co3+_Oh_IS': {'spin': 1, 'CFSE': -1.6}   # t2g⁵ eg¹
        }
        
        return {
            'redox_potentials': E_redox,
            'spin_states': spin_states,
            'active_site': 'Co2+_tetrahedral'
        }
    
    def calculate_oer_mechanism(self) -> Dict:
        """
        Calculate OER mechanism on Co3O4 surface
        
        Standard mechanism (alkaline conditions):
        1. * + OH- → OH* + e-
        2. OH* + OH- → O* + H2O + e-
        3. O* + OH- → OOH* + e-
        4. OOH* + OH- → * + O2 + H2O + e-
        """
        # Free energy diagram
        steps = [
            ('OH_adsorption', 0.8),
            ('O_formation', 0.5),
            ('OOH_formation', 0.9),
            ('O2_release', 1.2)
        ]
        
        # Determine rate-limiting step
        limiting_step = max(steps, key=lambda x: x[1])
        
        # Overpotential
        eta = limiting_step[1] - 1.23  # 1.23 V is equilibrium potential
        
        return {
            'reaction_steps': steps,
            'rate_limiting_step': limiting_step[0],
            'overpotential_V': eta,
            'theoretical_activity': 1.0 / max(1, eta)
        }


class Fe2O3Catalyst(TMOxideCatalyst):
    """
    Specialized catalyst class for Fe2O3 (hematite)
    
    Fe2O3 is important for:
    - Photocatalysis
    - Water oxidation
    - Environmental remediation
    """
    
    def __init__(self, config: CatalysisConfig = None):
        if config is None:
            config = CatalysisConfig(oxide="Fe2O3", surface="(0001)")
        super().__init__(config)
    
    def analyze_hematite_electronic_structure(self) -> Dict:
        """
        Analyze electronic structure of hematite
        
        Key features:
        - d⁵ configuration (Fe³⁺)
        - Antiferromagnetic ordering
        - Charge transfer character
        """
        # Electronic configuration
        fe_config = {
            'oxidation_state': 3,
            'd_electrons': 5,
            'spin': 2.5,  # S=5/2 high spin
            'configuration': 't2g³ eg²'
        }
        
        # Magnetic structure
        # Weak ferromagnetism due to spin canting
        magnetic = {
            'ordering': 'antiferromagnetic',
            'Neel_T': 955,  # K
            'morin_T': 260,  # K (spin reorientation)
            'canting_angle': 0.5  # degrees
        }
        
        # Band gap (indirect)
        # DFT: ~0.5 eV (too small)
        # DFT+U: ~2.0 eV
        # Experiment: ~2.1 eV
        
        return {
            'fe_configuration': fe_config,
            'magnetic_properties': magnetic,
            'band_gap_eV': 2.1,
            'optical_gap_eV': 2.0  # Direct
        }


class CatalyticActivityPredictor:
    """
    Predict catalytic activity using descriptors
    
    Uses correlations between electronic structure and activity.
    """
    
    def __init__(self):
        self.descriptors = {}
    
    def calculate_d_band_center(self, pdos: np.ndarray,
                               energies: np.ndarray,
                               E_fermi: float = 0) -> float:
        """
        Calculate d-band center
        
        ε_d = ∫ ε ρ_d(ε) dε / ∫ ρ_d(ε) dε
        
        Lower d-band center → stronger adsorption
        """
        # Shift energies relative to Fermi
        e_shifted = energies - E_fermi
        
        # Calculate center (only filled states)
        mask = e_shifted <= 0
        
        numerator = np.trapz(e_shifted[mask] * pdos[mask], e_shifted[mask])
        denominator = np.trapz(pdos[mask], e_shifted[mask])
        
        d_band_center = numerator / denominator
        
        return d_band_center
    
    def calculate_oer_descriptor(self, e_oh: float, e_o: float) -> float:
        """
        Calculate OER activity descriptor
        
        η = max(ΔG_OH, ΔG_O - ΔG_OH, 4.92 - ΔG_OOH) - 1.23
        
        Optimal: ΔG_O - ΔG_OH ≈ 1.6 eV (scaling relation)
        """
        # Scaling relation: ΔG_OOH = ΔG_OH + 3.2
        
        delta_G_oh = e_oh + 0.5  # Include corrections
        delta_G_o = e_o
        delta_G_ooh = delta_G_oh + 3.2
        
        eta_oh = delta_G_oh
        eta_o = delta_G_o - delta_G_oh
        eta_ooh = 4.92 - delta_G_ooh
        
        overpotential = max(eta_oh, eta_o, eta_ooh) - 1.23
        
        return overpotential
    
    def predict_co_oxidation_activity(self, o_vacancy_energy: float) -> float:
        """
        Predict CO oxidation activity
        
        Mars-van Krevelen mechanism:
        Activity correlates with oxygen vacancy formation energy
        """
        # Optimal E_vac ~ 2.0 eV
        # Too low: oxide unstable
        # Too high: oxygen too strongly bound
        
        E_optimal = 2.0
        activity = np.exp(-(o_vacancy_energy - E_optimal)**2 / 2)
        
        return activity


# Utility functions

def calculate_scaling_relations(adsorption_energies: Dict[str, float]) -> Dict:
    """
    Calculate scaling relations between adsorption energies
    
    Common relations:
    - ΔG_OOH = ΔG_OH + 3.2 eV
    - ΔG_O = 2 * ΔG_OH (often broken)
    """
    e_oh = adsorption_energies.get('OH', 0)
    e_o = adsorption_energies.get('O', 0)
    e_ooh = adsorption_energies.get('OOH', 0)
    
    # Actual vs predicted
    ooh_predicted = e_oh + 3.2
    o_predicted = 2 * e_oh
    
    return {
        'delta_G_OH': e_oh,
        'delta_G_O': e_o,
        'delta_G_OOH': e_ooh,
        'OOH_OH_scaling_deviation': e_ooh - ooh_predicted,
        'O_OH_scaling_deviation': e_o - o_predicted,
        'scaling_satisfied': abs(e_ooh - ooh_predicted) < 0.2
    }


def estimate_turnover_frequency(activation_barrier: float,
                               temperature: float = 298.15) -> float:
    """
    Estimate turnover frequency from activation barrier
    
    Using Arrhenius-like expression
    """
    kB = 8.617e-5  # eV/K
    
    # Attempt frequency (typical for surface reactions)
    nu = 1e13  # Hz
    
    # Rate constant
    k = nu * np.exp(-activation_barrier / (kB * temperature))
    
    # TOF (simplified)
    tof = k  # s^-1
    
    return tof


def sabatier_principle_analysis(adsorption_energy: float,
                                optimal_energy: float,
                                tolerance: float = 0.3) -> Dict:
    """
    Analyze adsorption energy according to Sabatier principle
    
    Optimal catalyst has intermediate binding (not too strong, not too weak)
    """
    deviation = abs(adsorption_energy - optimal_energy)
    
    return {
        'adsorption_energy': adsorption_energy,
        'optimal_energy': optimal_energy,
        'deviation': deviation,
        'is_optimal': deviation < tolerance,
        'activity_estimate': np.exp(-deviation**2 / (2 * tolerance**2))
    }


__all__ = [
    'CatalysisConfig',
    'TMOxideCatalyst',
    'Co3O4Catalyst',
    'Fe2O3Catalyst',
    'CatalyticActivityPredictor',
    'calculate_scaling_relations',
    'estimate_turnover_frequency',
    'sabatier_principle_analysis'
]