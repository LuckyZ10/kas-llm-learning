"""
Privacy Protection Module for DFT-LAMMPS
=========================================

This module provides privacy-preserving computation capabilities for
multi-institutional collaboration in materials science.

Submodules:
- homomorphic_encryption: Homomorphic encryption schemes (Paillier, CKKS, BFV)
- secure_mpc: Secure multi-party computation protocols
- data_anonymization: Data anonymization and de-identification

Author: DFT-LAMMPS Team
"""

from .homomorphic_encryption import (
    PaillierEncryption,
    CKKSEncryption,
    BFVEncryption,
    HomomorphicEncryptionScheme,
    HEConfig,
    EncryptedNeuralNetwork,
    SecureAggregationWithHE
)

from .secure_mpc import (
    SecretSharing,
    SecureComputation,
    MPCParty,
    SecureML,
    GarbledCircuit,
    MPCConfig,
    MPCProtocol,
    BeaverTriples,
    PrivacyPreservingAggregation
)

from .data_anonymization import (
    KAnonymityAnonymizer,
    LDiversityAnonymizer,
    DifferentialPrivacyAnonymizer,
    SyntheticDataGenerator,
    PrivacyAuditor,
    AnonymizationConfig,
    AnonymizationLevel,
    create_anonymization_pipeline,
    DataAnonymizer
)

__all__ = [
    # Homomorphic Encryption
    'PaillierEncryption',
    'CKKSEncryption',
    'BFVEncryption',
    'HomomorphicEncryptionScheme',
    'HEConfig',
    'EncryptedNeuralNetwork',
    'SecureAggregationWithHE',
    
    # Secure MPC
    'SecretSharing',
    'SecureComputation',
    'MPCParty',
    'SecureML',
    'GarbledCircuit',
    'MPCConfig',
    'MPCProtocol',
    'BeaverTriples',
    'PrivacyPreservingAggregation',
    
    # Data Anonymization
    'KAnonymityAnonymizer',
    'LDiversityAnonymizer',
    'DifferentialPrivacyAnonymizer',
    'SyntheticDataGenerator',
    'PrivacyAuditor',
    'AnonymizationConfig',
    'AnonymizationLevel',
    'create_anonymization_pipeline',
    'DataAnonymizer'
]
