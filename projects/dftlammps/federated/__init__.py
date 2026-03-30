"""
Federated Learning Module for DFT-LAMMPS
=========================================

This module provides federated learning capabilities for privacy-preserving
collaborative machine learning in materials science.

Submodules:
- federated_ml: Federated machine learning for ML potential training
- federated_discovery: Federated materials discovery

Author: DFT-LAMMPS Team
"""

from .federated_ml import (
    FederatedServer,
    FederatedClient,
    FederatedConfig,
    AggregationStrategy,
    DifferentialPrivacyMechanism,
    SecureAggregationProtocol,
    MLPotentialModel,
    create_federated_ml_system
)

from .federated_discovery import (
    FederatedDiscoveryCoordinator,
    DiscoveryClient,
    FederatedDiscoveryConfig,
    DiscoveryStrategy,
    MaterialCandidate,
    SecureGaussianProcess,
    PrivacyPreservingSampler,
    CrossInstitutionalCollaboration,
    create_federated_discovery_demo
)

__all__ = [
    # Federated ML
    'FederatedServer',
    'FederatedClient',
    'FederatedConfig',
    'AggregationStrategy',
    'DifferentialPrivacyMechanism',
    'SecureAggregationProtocol',
    'MLPotentialModel',
    'create_federated_ml_system',
    
    # Federated Discovery
    'FederatedDiscoveryCoordinator',
    'DiscoveryClient',
    'FederatedDiscoveryConfig',
    'DiscoveryStrategy',
    'MaterialCandidate',
    'SecureGaussianProcess',
    'PrivacyPreservingSampler',
    'CrossInstitutionalCollaboration',
    'create_federated_discovery_demo'
]
