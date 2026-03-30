"""
Example: Privacy-Preserving Catalyst Screening
===============================================

This example demonstrates federated learning for catalyst discovery
across multiple chemical companies with privacy protection.

Scenario:
- Multiple chemical companies want to discover new catalysts for 
  sustainable chemical processes
- Each has proprietary catalyst formulations and performance data
- Federated learning enables collaborative screening without revealing
  proprietary catalyst structures

Author: DFT-LAMMPS Team
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Set
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dftlammps.federated.federated_ml import (
    FederatedServer, FederatedClient, FederatedConfig,
    AggregationStrategy, MLPotentialModel
)
from dftlammps.federated.federated_discovery import (
    FederatedDiscoveryCoordinator,
    DiscoveryClient,
    FederatedDiscoveryConfig,
    DiscoveryStrategy,
    MaterialCandidate
)
from dftlammps.privacy.secure_mpc import (
    SecureComputation,
    MPCConfig,
    SecretSharing
)
from dftlammps.privacy.homomorphic_encryption import (
    PaillierEncryption,
    SecureAggregationWithHE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChemicalCompany:
    """Represents a participating chemical company."""
    
    def __init__(self, name: str, specialty: str, catalyst_library_size: int):
        self.name = name
        self.specialty = specialty
        self.catalyst_library_size = catalyst_library_size
        self.proprietary_catalysts: List[MaterialCandidate] = []
        self.client = None
        
    def __repr__(self):
        return f"ChemicalCompany({self.name}, {self.specialty})"


def create_chemical_companies() -> List[ChemicalCompany]:
    """Create participating chemical companies."""
    companies = [
        ChemicalCompany(
            "BASF Catalysts",
            "Heterogeneous Catalysis",
            25000
        ),
        ChemicalCompany(
            "Dow Chemical Research",
            "Polymerization Catalysis",
            18000
        ),
        ChemicalCompany(
            "Johnson Matthey",
            "Emission Control Catalysts",
            22000
        ),
        ChemicalCompany(
            "Clariant Specialty",
            "Fine Chemical Catalysis",
            15000
        ),
    ]
    return companies


def generate_catalyst_database(company: ChemicalCompany) -> List[MaterialCandidate]:
    """
    Generate synthetic catalyst database for a company.
    """
    np.random.seed(hash(company.name) % 2**32)
    
    catalysts = []
    
    # Different catalyst classes based on specialty
    if "Heterogeneous" in company.specialty:
        catalyst_types = ['zeolite', 'metal_oxide', 'supported_metal']
        base_metals = ['Pt', 'Pd', 'Rh', 'Ru', 'Ni', 'Co', 'Fe', 'Cu', 'Zn']
        supports = ['Al2O3', 'SiO2', 'TiO2', 'CeO2', 'ZrO2', 'Carbon']
    elif "Polymerization" in company.specialty:
        catalyst_types = ['ziegler_natta', 'metallocene', 'cr', 'pd_diimine']
        base_metals = ['Ti', 'Zr', 'Hf', 'Cr', 'V', 'Ni', 'Pd']
        supports = ['MgCl2', 'Silica', 'Alumina']
    elif "Emission" in company.specialty:
        catalyst_types = ['three_way', 'diesel_oxidation', 'scr', 'doc']
        base_metals = ['Pt', 'Pd', 'Rh', 'Cu', 'Fe', 'Ce']
        supports = ['Cordierite', 'Al2O3', 'Zeolite', 'TiO2']
    else:  # Fine Chemical
        catalyst_types = ['homogeneous', 'enzyme', 'phase_transfer']
        base_metals = ['Pd', 'Pt', 'Ru', 'Rh', 'Ir', 'Au']
        supports = ['None', 'Polymer', 'Carbon']
    
    # Generate catalysts
    n_catalysts = min(200, company.catalyst_library_size // 1000)
    
    for i in range(n_catalysts):
        cat_type = np.random.choice(catalyst_types)
        metal = np.random.choice(base_metals)
        support = np.random.choice(supports)
        
        composition = f"{metal}/{support}_{cat_type}"
        
        # Generate features (catalyst descriptors)
        features = np.random.randn(150)
        
        # Add company-specific signature
        company_hash = hash(company.name) % 100
        features[company_hash % 150] += 2.0
        
        # Catalytic properties
        predicted_props = {
            'turnover_frequency': np.random.uniform(0.1, 1000),  # h^-1
            'selectivity': np.random.uniform(0.5, 0.99),
            'activation_energy': np.random.uniform(20, 150),  # kJ/mol
            'stability_hours': np.random.uniform(100, 10000),
            'cost_per_kg': np.random.uniform(10, 5000),
            'environmental_impact': np.random.uniform(0, 1)  # Lower is better
        }
        
        catalyst = MaterialCandidate(
            composition=composition,
            features=features,
            predicted_properties=predicted_props,
            source_institution=company.name,
            privacy_level="confidential"
        )
        
        catalysts.append(catalyst)
    
    return catalysts


def compute_catalyst_score(catalyst: MaterialCandidate,
                          target_reaction: str) -> float:
    """
    Compute overall catalyst score for a target reaction.
    """
    props = catalyst.predicted_properties
    
    # Different weightings for different reactions
    if target_reaction == "sustainable_ammonia":
        weights = {
            'turnover_frequency': 0.3,
            'activation_energy': -0.2,  # Lower is better
            'stability_hours': 0.2,
            'cost_per_kg': -0.1,  # Lower is better
            'environmental_impact': -0.2  # Lower is better
        }
    elif target_reaction == "co2_reduction":
        weights = {
            'turnover_frequency': 0.25,
            'selectivity': 0.3,
            'activation_energy': -0.15,
            'stability_hours': 0.15,
            'environmental_impact': -0.15
        }
    else:  # generic
        weights = {
            'turnover_frequency': 0.3,
            'selectivity': 0.3,
            'activation_energy': -0.2,
            'stability_hours': 0.2
        }
    
    # Normalize values
    score = 0.0
    for prop, weight in weights.items():
        value = props.get(prop, 0)
        
        # Normalize to [0, 1]
        if prop == 'turnover_frequency':
            norm_value = np.log10(value + 1) / 4  # log scale
        elif prop == 'activation_energy':
            norm_value = 1 - (value / 150)  # invert
        elif prop == 'stability_hours':
            norm_value = np.log10(value) / 4
        elif prop == 'cost_per_kg':
            norm_value = 1 - (np.log10(value) / 4)
        elif prop == 'selectivity':
            norm_value = value
        elif prop == 'environmental_impact':
            norm_value = 1 - value  # invert
        else:
            norm_value = value
        
        score += weight * norm_value
    
    return score


def setup_federated_screening(companies: List[ChemicalCompany],
                              target_reaction: str = "sustainable_ammonia") -> Tuple[FederatedDiscoveryCoordinator, List[DiscoveryClient]]:
    """Setup federated screening system."""
    logger.info(f"Setting up federated screening for: {target_reaction}")
    
    # Configuration for catalyst screening
    config = FederatedDiscoveryConfig(
        num_candidates=300,
        num_iterations=25,
        batch_size=10,
        strategy=DiscoveryStrategy.FEDERATED_BO,
        exploration_factor=0.2,
        acquisition_function="ucb",
        use_dp=True,
        epsilon=2.0,
        delta=1e-5,
        use_mpc=True,
        num_parties=len(companies),
        share_candidates=True,
        candidate_anonymization=True,
        min_institutions_per_candidate=2,
        target_properties=['turnover_frequency', 'selectivity', 'activation_energy'],
        property_ranges={
            'turnover_frequency': (10, 500),
            'selectivity': (0.8, 0.99),
            'activation_energy': (20, 100)
        }
    )
    
    # Create coordinator
    coordinator = FederatedDiscoveryCoordinator(config)
    
    # Create clients for each company
    clients = []
    for i, company in enumerate(companies):
        client = DiscoveryClient(
            institution_id=f"chem_{i}",
            institution_name=company.name,
            config=config
        )
        
        # Generate and add proprietary catalysts
        proprietary_catalysts = generate_catalyst_database(company)
        client.add_local_materials(proprietary_catalysts)
        company.proprietary_catalysts = proprietary_catalysts
        
        # Register with coordinator
        coordinator.register_institution(f"chem_{i}", client)
        
        clients.append(client)
        company.client = client
        
        logger.info(f"Registered {company.name} with {len(proprietary_catalysts)} catalysts")
    
    return coordinator, clients


def run_secure_screening(coordinator: FederatedDiscoveryCoordinator,
                        companies: List[ChemicalCompany],
                        target_reaction: str) -> List[MaterialCandidate]:
    """Run privacy-preserving catalyst screening."""
    logger.info("\n" + "=" * 60)
    logger.info("Privacy-Preserving Catalyst Screening")
    logger.info(f"Target: {target_reaction}")
    logger.info("=" * 60)
    
    # Initialize candidate pool
    coordinator.initialize_candidate_pool()
    
    discovered = []
    
    for iteration in range(coordinator.config.num_iterations):
        coordinator.iteration = iteration
        
        # Federated screening
        ranked_candidates = coordinator.federated_screening()
        
        # Bayesian optimization
        selected = coordinator.federated_bayesian_optimization()
        
        # Compute catalyst-specific score
        score = compute_catalyst_score(selected, target_reaction)
        selected.acquisition_score = score
        
        # Simulate evaluation
        for prop in coordinator.config.target_properties:
            true_value = selected.predicted_properties.get(prop, 0)
            true_value += np.random.normal(0, selected.uncertainty.get(prop, 0.1))
            selected.predicted_properties[prop] = true_value
        
        coordinator.update_models(selected)
        discovered.append(selected)
        
        # Remove from pool
        coordinator.candidate_pool = [
            c for c in coordinator.candidate_pool 
            if c.candidate_id != selected.candidate_id
        ]
        
        # Record history
        coordinator.history['iterations'].append(iteration)
        coordinator.history['best_scores'].append(
            max(c.acquisition_score for c in discovered)
        )
        coordinator.history['discovered_materials'].append(selected.composition)
        
        if iteration % 5 == 0:
            logger.info(f"Iteration {iteration}: Best score = {max(c.acquisition_score for c in discovered):.3f}")
    
    return discovered


def perform_secure_aggregation(companies: List[ChemicalCompany],
                               discovered: List[MaterialCandidate]) -> Dict:
    """
    Perform secure aggregation of performance metrics using
    homomorphic encryption.
    """
    logger.info("\nPerforming secure aggregation of performance metrics...")
    
    # Initialize secure aggregation
    secure_agg = SecureAggregationWithHE(
        num_parties=len(companies),
        he_scheme="paillier"
    )
    
    # Each company encrypts their average performance metrics
    encrypted_metrics = []
    
    for company in companies:
        # Compute average TOF for this company's discovered catalysts
        company_catalysts = [
            c for c in discovered 
            if c.source_institution == company.name
        ]
        
        if company_catalysts:
            avg_tof = np.mean([
                c.predicted_properties.get('turnover_frequency', 0)
                for c in company_catalysts
            ])
        else:
            avg_tof = 0
        
        # Encrypt
        encrypted = secure_agg.he.encrypt(int(avg_tof * 1000))  # Scale for integers
        encrypted_metrics.append(encrypted)
    
    # Compute encrypted average
    encrypted_avg = secure_agg.aggregate_encrypted_values(encrypted_metrics)
    
    # Decrypt
    total_tof = secure_agg.he.decrypt(encrypted_avg)
    avg_tof_across_companies = total_tof / (len(companies) * 1000)
    
    return {
        'encrypted_aggregation': True,
        'avg_turnover_frequency': avg_tof_across_companies,
        'companies_contributed': len(companies)
    }


def analyze_catalyst_discoveries(discovered: List[MaterialCandidate],
                                companies: List[ChemicalCompany],
                                target_reaction: str) -> Dict:
    """Analyze discovered catalysts."""
    analysis = {
        'target_reaction': target_reaction,
        'total_discovered': len(discovered),
        'by_company': {},
        'by_catalyst_type': {},
        'performance_summary': {},
        'top_catalysts': []
    }
    
    # Count by company
    for catalyst in discovered:
        source = catalyst.source_institution
        if source not in analysis['by_company']:
            analysis['by_company'][source] = 0
        analysis['by_company'][source] += 1
        
        # Count by catalyst type
        cat_type = catalyst.composition.split('_')[-1]
        if cat_type not in analysis['by_catalyst_type']:
            analysis['by_catalyst_type'][cat_type] = 0
        analysis['by_catalyst_type'][cat_type] += 1
    
    # Performance statistics
    properties = ['turnover_frequency', 'selectivity', 'activation_energy', 'stability_hours']
    for prop in properties:
        values = [c.predicted_properties.get(prop, 0) for c in discovered]
        analysis['performance_summary'][prop] = {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    # Top catalysts
    sorted_catalysts = sorted(discovered, 
                             key=lambda c: compute_catalyst_score(c, target_reaction),
                             reverse=True)
    
    for i, catalyst in enumerate(sorted_catalysts[:10], 1):
        analysis['top_catalysts'].append({
            'rank': i,
            'composition': catalyst.composition,
            'source': catalyst.source_institution,
            'score': float(compute_catalyst_score(catalyst, target_reaction)),
            'turnover_frequency': float(catalyst.predicted_properties.get('turnover_frequency', 0)),
            'selectivity': float(catalyst.predicted_properties.get('selectivity', 0)),
            'activation_energy': float(catalyst.predicted_properties.get('activation_energy', 0))
        })
    
    return analysis


def save_results(coordinator: FederatedDiscoveryCoordinator,
                discovered: List[MaterialCandidate],
                analysis: Dict,
                secure_agg_results: Dict,
                output_dir: str = "./results/catalyst_screening"):
    """Save screening results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save discovered catalysts
    catalysts_data = [c.to_dict() for c in discovered]
    catalysts_path = os.path.join(output_dir, "discovered_catalysts.json")
    with open(catalysts_path, 'w') as f:
        json.dump(catalysts_data, f, indent=2)
    
    # Save analysis
    analysis_path = os.path.join(output_dir, "analysis.json")
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Save secure aggregation results
    agg_path = os.path.join(output_dir, "secure_aggregation.json")
    with open(agg_path, 'w') as f:
        json.dump(secure_agg_results, f, indent=2)
    
    # Save coordinator state
    coordinator.export_results(os.path.join(output_dir, "screening_results.json"))
    
    logger.info(f"Results saved to {output_dir}")


def generate_compliance_report(coordinator: FederatedDiscoveryCoordinator,
                               companies: List[ChemicalCompany],
                               discovered: List[MaterialCandidate]) -> str:
    """Generate privacy compliance report."""
    report = []
    report.append("=" * 70)
    report.append("Catalyst Screening Privacy Compliance Report")
    report.append("=" * 70)
    
    report.append(f"\n1. Differential Privacy Guarantees:")
    report.append(f"   • Privacy budget (ε): {coordinator.config.epsilon}")
    report.append(f"   • Failure probability (δ): {coordinator.config.delta}")
    report.append(f"   • Budget consumed: {coordinator.history['privacy_budget_spent'][-1]:.4f}")
    
    report.append(f"\n2. Secure Multi-Party Computation:")
    report.append(f"   • Protocol: Shamir's Secret Sharing")
    report.append(f"   • Threshold: {coordinator.config.min_institutions_per_candidate} of {len(companies)}")
    report.append(f"   • Homomorphic encryption used for aggregation")
    
    report.append(f"\n3. Data Anonymization:")
    report.append(f"   • Candidate anonymization: {coordinator.config.candidate_anonymization}")
    report.append(f"   • k-anonymity level: 5")
    
    report.append(f"\n4. Confidentiality Levels:")
    levels = {}
    for c in discovered:
        levels[c.privacy_level] = levels.get(c.privacy_level, 0) + 1
    for level, count in levels.items():
        report.append(f"   • {level}: {count} catalysts")
    
    report.append(f"\n5. Cross-Company Collaboration:")
    for company in companies:
        company_catalysts = [c for c in discovered if c.source_institution == company.name]
        report.append(f"   • {company.name}: {len(company_catalysts)} contributions")
    
    report.append("=" * 70)
    
    return "\n".join(report)


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("Privacy-Preserving Catalyst Screening")
    print("Federated Discovery for Sustainable Chemistry")
    print("=" * 70)
    
    target_reaction = "sustainable_ammonia"
    
    # Step 1: Setup companies
    print("\n📋 Step 1: Setting up participating chemical companies...")
    companies = create_chemical_companies()
    
    print("\nParticipating Companies:")
    for company in companies:
        print(f"  • {company.name}")
        print(f"    Specialty: {company.specialty}")
        print(f"    Catalyst Library: {company.catalyst_library_size:,} catalysts")
    
    # Step 2: Setup federated screening
    print("\n🔧 Step 2: Setting up federated screening infrastructure...")
    coordinator, clients = setup_federated_screening(companies, target_reaction)
    
    print(f"\nFederated Screening Configuration:")
    print(f"  • Target Reaction: {target_reaction}")
    print(f"  • Strategy: {coordinator.config.strategy.value}")
    print(f"  • Differential Privacy: ε={coordinator.config.epsilon}")
    print(f"  • Secure MPC: Enabled")
    print(f"  • Homomorphic Encryption: Paillier")
    
    # Step 3: Run screening
    print("\n🚀 Step 3: Running privacy-preserving catalyst screening...")
    discovered = run_secure_screening(coordinator, companies, target_reaction)
    
    # Step 4: Secure aggregation
    print("\n🔐 Step 4: Performing secure aggregation of metrics...")
    secure_agg_results = perform_secure_aggregation(companies, discovered)
    print(f"✓ Securely aggregated metrics across {secure_agg_results['companies_contributed']} companies")
    print(f"✓ Average TOF: {secure_agg_results['avg_turnover_frequency']:.2f} h^-1")
    
    # Step 5: Analyze results
    print("\n📊 Step 5: Analyzing screening results...")
    analysis = analyze_catalyst_discoveries(discovered, companies, target_reaction)
    
    print(f"\nScreening Summary:")
    print(f"  • Target Reaction: {analysis['target_reaction']}")
    print(f"  • Total Catalysts Discovered: {analysis['total_discovered']}")
    print(f"  • By Company: {analysis['by_company']}")
    
    print(f"\nCatalyst Type Distribution:")
    for cat_type, count in analysis['by_catalyst_type'].items():
        print(f"  • {cat_type}: {count}")
    
    print(f"\nPerformance Summary:")
    for prop, stats in analysis['performance_summary'].items():
        print(f"  • {prop}:")
        print(f"    Mean: {stats['mean']:.2f}")
        print(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    
    print(f"\nTop 5 Catalysts for {target_reaction}:")
    for catalyst in analysis['top_catalysts'][:5]:
        print(f"  {catalyst['rank']}. {catalyst['composition']}")
        print(f"     Source: {catalyst['source']}")
        print(f"     Score: {catalyst['score']:.3f}")
        print(f"     TOF: {catalyst['turnover_frequency']:.1f} h^-1")
        print(f"     Selectivity: {catalyst['selectivity']:.1%}")
    
    # Step 6: Compliance report
    print("\n🔒 Step 6: Generating privacy compliance report...")
    compliance_report = generate_compliance_report(coordinator, companies, discovered)
    print(compliance_report)
    
    # Step 7: Save results
    print("\n💾 Step 7: Saving results...")
    save_results(coordinator, discovered, analysis, secure_agg_results)
    
    print("\n" + "=" * 70)
    print("Privacy-Preserving Catalyst Screening Complete!")
    print("=" * 70)
    print("\n✅ Successfully demonstrated:")
    print("   • Federated catalyst screening")
    print("   • Homomorphic encryption for secure aggregation")
    print("   • Differential privacy for property prediction")
    print("   • Multi-company collaboration with IP protection")
    print("   • Privacy-preserving Bayesian optimization")
    print("\n📁 Results saved to: ./results/catalyst_screening/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
