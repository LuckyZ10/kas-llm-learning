"""
Example: Cross-Company Drug Discovery
======================================

This example demonstrates privacy-preserving collaborative drug discovery
between multiple pharmaceutical companies using federated learning and
secure multi-party computation.

Scenario:
- Three pharmaceutical companies want to discover new drug candidates
- Each has proprietary molecular databases and screening data
- IP and competitive concerns prevent direct data sharing
- Federated discovery enables collaboration while preserving privacy

Author: DFT-LAMMPS Team
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dftlammps.federated.federated_discovery import (
    FederatedDiscoveryCoordinator,
    DiscoveryClient,
    FederatedDiscoveryConfig,
    DiscoveryStrategy,
    MaterialCandidate,
    CrossInstitutionalCollaboration
)
from dftlammps.privacy.data_anonymization import (
    AnonymizationConfig,
    AnonymizationLevel,
    PrivacyAuditor
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PharmaceuticalCompany:
    """Represents a participating pharmaceutical company."""
    
    def __init__(self, name: str, therapeutic_area: str, 
                 proprietary_compounds: int):
        self.name = name
        self.therapeutic_area = therapeutic_area
        self.proprietary_compounds = proprietary_compounds
        self.discovery_client = None
        self.private_compounds: List[MaterialCandidate] = []
        
    def __repr__(self):
        return f"PharmaCompany({self.name}, {self.therapeutic_area})"


def create_pharmaceutical_companies() -> List[PharmaceuticalCompany]:
    """Create participating pharmaceutical companies."""
    companies = [
        PharmaceuticalCompany(
            "Novartis Bio",
            "Oncology & Immunology",
            50000
        ),
        PharmaceuticalCompany(
            "PfizerMed Research",
            "Cardiovascular & Infectious Disease",
            75000
        ),
        PharmaceuticalCompany(
            "RocheGen",
            "Neuroscience & Rare Diseases",
            45000
        ),
    ]
    return companies


def generate_molecular_database(company: PharmaceuticalCompany) -> List[MaterialCandidate]:
    """
    Generate synthetic molecular database for a company.
    
    In practice, this would be the company's actual compound library.
    """
    np.random.seed(hash(company.name) % 2**32)
    
    compounds = []
    
    # Generate different compound classes based on therapeutic area
    if "Oncology" in company.therapeutic_area:
        scaffolds = [
            "imatinib", "erlotinib", "sorafenib", "dasatinib",
            "nilotinib", "lapatinib", "pazopanib", "sunitinib"
        ]
        target_properties = {'ic50': (0.1, 100), 'solubility': (10, 100)}
    elif "Cardiovascular" in company.therapeutic_area:
        scaffolds = [
            "atorvastatin", "lisinopril", "metoprolol", "amlodipine",
            "losartan", "simvastatin", "valsartan", "carvedilol"
        ]
        target_properties = {'ic50': (1, 500), 'solubility': (20, 200)}
    else:  # Neuroscience
        scaffolds = [
            "donepezil", "memantine", "rivastigmine", "galantamine",
            "sertraline", "fluoxetine", "aripiprazole", "risperidone"
        ]
        target_properties = {'ic50': (0.5, 200), 'solubility': (5, 150)}
    
    # Generate compounds
    n_compounds = min(100, company.proprietary_compounds // 1000)
    
    for i in range(n_compounds):
        scaffold = np.random.choice(scaffolds)
        
        # Generate features (molecular descriptors)
        features = np.random.randn(200)
        
        # Add therapeutic area signal
        if "Oncology" in company.therapeutic_area:
            features[0] += 1.5  # Distinct signature
        elif "Cardiovascular" in company.therapeutic_area:
            features[1] += 1.2
        else:
            features[2] += 1.0
        
        # Predicted properties
        predicted_props = {
            'binding_affinity': np.random.uniform(-12, -6),
            'bioavailability': np.random.uniform(0.1, 0.9),
            'toxicity_score': np.random.uniform(0, 1),
            'synthetic_accessibility': np.random.uniform(1, 10)
        }
        
        compound = MaterialCandidate(
            composition=f"{scaffold}_analog_{i:04d}",
            features=features,
            predicted_properties=predicted_props,
            source_institution=company.name,
            privacy_level="confidential"
        )
        
        compounds.append(compound)
    
    return compounds


def setup_federated_discovery(companies: List[PharmaceuticalCompany]) -> Tuple[FederatedDiscoveryCoordinator, List[DiscoveryClient]]:
    """Setup federated discovery system for drug discovery."""
    logger.info("Setting up federated drug discovery system...")
    
    # Configuration for drug discovery
    config = FederatedDiscoveryConfig(
        num_candidates=200,
        num_iterations=20,
        batch_size=5,
        strategy=DiscoveryStrategy.FEDERATED_BO,
        exploration_factor=0.15,
        acquisition_function="ei",
        use_dp=True,
        epsilon=1.5,
        delta=1e-5,
        use_mpc=True,
        num_parties=len(companies),
        share_candidates=True,
        candidate_anonymization=True,
        min_institutions_per_candidate=2,
        target_properties=['binding_affinity', 'bioavailability', 'toxicity_score'],
        property_ranges={
            'binding_affinity': (-15, -8),
            'bioavailability': (0.3, 0.9),
            'toxicity_score': (0, 0.3)
        }
    )
    
    # Create coordinator
    coordinator = FederatedDiscoveryCoordinator(config)
    
    # Create clients for each company
    clients = []
    for i, company in enumerate(companies):
        client = DiscoveryClient(
            institution_id=f"pharma_{i}",
            institution_name=company.name,
            config=config
        )
        
        # Generate and add proprietary compounds
        proprietary_compounds = generate_molecular_database(company)
        client.add_local_materials(proprietary_compounds)
        
        # Register with coordinator
        coordinator.register_institution(f"pharma_{i}", client)
        
        clients.append(client)
        company.discovery_client = client
        
        logger.info(f"Registered {company.name} with {len(proprietary_compounds)} compounds")
    
    return coordinator, clients


def run_collaborative_discovery(coordinator: FederatedDiscoveryCoordinator) -> List[MaterialCandidate]:
    """Run the federated drug discovery process."""
    logger.info("\n" + "=" * 60)
    logger.info("Starting Cross-Company Drug Discovery")
    logger.info("=" * 60)
    
    # Run discovery
    discovered = coordinator.run_discovery()
    
    logger.info("\n" + "=" * 60)
    logger.info("Drug Discovery Complete")
    logger.info("=" * 60)
    
    return discovered


def analyze_discovered_candidates(discovered: List[MaterialCandidate],
                                  companies: List[PharmaceuticalCompany]) -> Dict:
    """Analyze discovered drug candidates."""
    analysis = {
        'total_discovered': len(discovered),
        'by_company': {},
        'property_stats': {}
    }
    
    # Count by source company
    for candidate in discovered:
        source = candidate.source_institution
        if source not in analysis['by_company']:
            analysis['by_company'][source] = 0
        analysis['by_company'][source] += 1
    
    # Property statistics
    properties = ['binding_affinity', 'bioavailability', 'toxicity_score']
    for prop in properties:
        values = [c.predicted_properties.get(prop, 0) for c in discovered]
        analysis['property_stats'][prop] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    # Score candidates
    scored_candidates = []
    for candidate in discovered:
        # Multi-objective score
        ba = candidate.predicted_properties.get('binding_affinity', -10)
        bio = candidate.predicted_properties.get('bioavailability', 0.5)
        tox = candidate.predicted_properties.get('toxicity_score', 0.5)
        
        # Higher is better for binding affinity and bioavailability
        # Lower is better for toxicity
        score = (-ba / 15) + bio - tox
        scored_candidates.append((candidate, score))
    
    # Top candidates
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    analysis['top_candidates'] = [
        {
            'composition': c.composition,
            'source': c.source_institution,
            'score': float(score),
            'properties': c.predicted_properties
        }
        for c, score in scored_candidates[:5]
    ]
    
    return analysis


def establish_collaboration_agreements(companies: List[PharmaceuticalCompany]) -> CrossInstitutionalCollaboration:
    """Establish legal and technical collaboration frameworks."""
    collaboration = CrossInstitutionalCollaboration(
        institutions=[c.name for c in companies]
    )
    
    # Establish bilateral agreements
    for i, company1 in enumerate(companies):
        for company2 in companies[i+1:]:
            agreement = {
                'data_sharing': 'anonymized_only',
                'ip_ownership': 'shared_discovery',
                'commercial_rights': 'negotiated',
                'audit_rights': 'mutual'
            }
            collaboration.establish_collaboration(
                company1.name, company2.name, agreement
            )
    
    return collaboration


def generate_privacy_report(discovered: List[MaterialCandidate],
                           coordinator: FederatedDiscoveryCoordinator) -> str:
    """Generate privacy compliance report."""
    report = []
    report.append("=" * 60)
    report.append("Privacy Compliance Report")
    report.append("=" * 60)
    
    # Differential privacy
    report.append(f"\nDifferential Privacy:")
    report.append(f"  Privacy budget (ε): {coordinator.config.epsilon}")
    report.append(f"  Failure probability (δ): {coordinator.config.delta}")
    report.append(f"  Budget consumed: {coordinator.history['privacy_budget_spent'][-1]:.4f}")
    
    # Anonymization
    report.append(f"\nAnonymization:")
    report.append(f"  Candidate anonymization: {coordinator.config.candidate_anonymization}")
    
    # Secure computation
    report.append(f"\nSecure Computation:")
    report.append(f"  MPC enabled: {coordinator.config.use_mpc}")
    report.append(f"  Minimum institutions per candidate: {coordinator.config.min_institutions_per_candidate}")
    
    # Data exposure
    report.append(f"\nData Exposure Summary:")
    public_count = sum(1 for c in discovered if c.privacy_level == 'public')
    internal_count = sum(1 for c in discovered if c.privacy_level == 'internal')
    confidential_count = sum(1 for c in discovered if c.privacy_level == 'confidential')
    
    report.append(f"  Public candidates: {public_count}")
    report.append(f"  Internal candidates: {internal_count}")
    report.append(f"  Confidential candidates: {confidential_count}")
    
    report.append("=" * 60)
    
    return "\n".join(report)


def save_results(coordinator: FederatedDiscoveryCoordinator,
                discovered: List[MaterialCandidate],
                analysis: Dict,
                output_dir: str = "./results/drug_discovery"):
    """Save discovery results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save discovered candidates
    candidates_data = [c.to_dict() for c in discovered]
    candidates_path = os.path.join(output_dir, "discovered_candidates.json")
    with open(candidates_path, 'w') as f:
        json.dump(candidates_data, f, indent=2)
    logger.info(f"Discovered candidates saved to {candidates_path}")
    
    # Save analysis
    analysis_path = os.path.join(output_dir, "analysis.json")
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Analysis saved to {analysis_path}")
    
    # Save privacy report
    privacy_report = generate_privacy_report(discovered, coordinator)
    privacy_path = os.path.join(output_dir, "privacy_report.txt")
    with open(privacy_path, 'w') as f:
        f.write(privacy_report)
    logger.info(f"Privacy report saved to {privacy_path}")
    
    # Save coordinator state
    coordinator.export_results(os.path.join(output_dir, "discovery_results.json"))


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("Cross-Company Drug Discovery")
    print("Privacy-Preserving Collaborative Discovery Demonstration")
    print("=" * 70)
    
    # Step 1: Setup companies
    print("\n📋 Step 1: Setting up participating pharmaceutical companies...")
    companies = create_pharmaceutical_companies()
    
    print("\nParticipating Companies:")
    for company in companies:
        print(f"  • {company.name}")
        print(f"    Focus: {company.therapeutic_area}")
        print(f"    Library Size: {company.proprietary_compounds:,} compounds")
    
    # Step 2: Establish collaboration framework
    print("\n📄 Step 2: Establishing collaboration agreements...")
    collaboration = establish_collaboration_agreements(companies)
    print("✓ Bilateral collaboration agreements established")
    print("✓ Data sharing protocols configured")
    print("✓ IP ownership frameworks defined")
    
    # Step 3: Setup federated discovery
    print("\n🔧 Step 3: Setting up federated discovery infrastructure...")
    coordinator, clients = setup_federated_discovery(companies)
    
    print(f"\nFederated Discovery Configuration:")
    print(f"  • Strategy: {coordinator.config.strategy.value}")
    print(f"  • Differential Privacy: {coordinator.config.use_dp}")
    print(f"  • Secure MPC: {coordinator.config.use_mpc}")
    print(f"  • Candidate Anonymization: {coordinator.config.candidate_anonymization}")
    print(f"  • Target Properties: {', '.join(coordinator.config.target_properties)}")
    
    # Step 4: Run discovery
    print("\n🚀 Step 4: Starting collaborative drug discovery...")
    discovered = run_collaborative_discovery(coordinator)
    
    # Step 5: Analyze results
    print("\n📊 Step 5: Analyzing discovery results...")
    analysis = analyze_discovered_candidates(discovered, companies)
    
    print(f"\nDiscovery Summary:")
    print(f"  • Total candidates discovered: {analysis['total_discovered']}")
    print(f"  • By company: {analysis['by_company']}")
    
    print(f"\nProperty Statistics:")
    for prop, stats in analysis['property_stats'].items():
        print(f"  • {prop}:")
        print(f"    Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
        print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    print(f"\nTop 5 Drug Candidates:")
    for i, candidate in enumerate(analysis['top_candidates'], 1):
        print(f"  {i}. {candidate['composition']}")
        print(f"     Source: {candidate['source']}")
        print(f"     Score: {candidate['score']:.3f}")
        print(f"     Binding Affinity: {candidate['properties']['binding_affinity']:.2f}")
        print(f"     Bioavailability: {candidate['properties']['bioavailability']:.2f}")
    
    # Step 6: Privacy report
    print("\n🔒 Step 6: Generating privacy compliance report...")
    privacy_report = generate_privacy_report(discovered, coordinator)
    print(privacy_report)
    
    # Step 7: Save results
    print("\n💾 Step 7: Saving results...")
    save_results(coordinator, discovered, analysis)
    
    print("\n" + "=" * 70)
    print("Cross-Company Drug Discovery Complete!")
    print("=" * 70)
    print("\n✅ Successfully demonstrated:")
    print("   • Privacy-preserving collaborative discovery")
    print("   • Secure multi-party computation for screening")
    print("   • Differential privacy for property prediction")
    print("   • Anonymized candidate sharing")
    print("   • Cross-company IP protection")
    print("\n📁 Results saved to: ./results/drug_discovery/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
