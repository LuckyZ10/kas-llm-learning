# Federated Learning and Privacy Protection Module

This module provides comprehensive privacy-preserving federated learning capabilities for multi-institutional collaboration in materials science research.

## Overview

The federated learning module enables multiple institutions to collaboratively train machine learning models and discover new materials without sharing their proprietary data. It combines state-of-the-art privacy protection techniques including differential privacy, secure multi-party computation, and homomorphic encryption.

## Directory Structure

```
dftlammps/federated/
├── __init__.py
├── federated_ml.py          # Federated ML for potential training
├── federated_discovery.py   # Federated materials discovery
└── examples/
    ├── multi_lab_training.py
    ├── cross_company_drug_discovery.py
    └── privacy_catalyst_screening.py

dftlammps/privacy/
├── __init__.py
├── homomorphic_encryption.py  # HE schemes (Paillier, CKKS, BFV)
├── secure_mpc.py              # Secure multi-party computation
└── data_anonymization.py      # Data anonymization techniques
```

## Features

### Federated Machine Learning (`federated_ml.py`)

- **FederatedServer**: Central coordinator for federated training
- **FederatedClient**: Client-side training interface
- **Aggregation Strategies**:
  - FedAvg: Standard federated averaging
  - FedProx: With proximal regularization for heterogeneous data
  - SCAFFOLD: With control variates for variance reduction
  - FedOpt: Server-side optimization
  
- **Differential Privacy**: Built-in DP-SGD with privacy accounting
- **Secure Aggregation**: Multi-party secure aggregation protocol

### Federated Materials Discovery (`federated_discovery.py`)

- **FederatedDiscoveryCoordinator**: Central discovery coordinator
- **DiscoveryClient**: Institution-side discovery client
- **Privacy-Preserving Screening**: Secure high-throughput screening
- **Bayesian Optimization**: Federated BO with privacy protection
- **Cross-Institutional Collaboration**: Secure collaboration protocols

### Privacy Protection (`dftlammps/privacy/`)

#### Homomorphic Encryption (`homomorphic_encryption.py`)

- **PaillierEncryption**: Additively homomorphic encryption
- **CKKSEncryption**: Approximate arithmetic for ML (simplified)
- **BFVEncryption**: Exact arithmetic for integer operations
- **EncryptedNeuralNetwork**: Neural network inference on encrypted data

#### Secure Multi-Party Computation (`secure_mpc.py`)

- **SecretSharing**: Shamir's secret sharing
- **SecureComputation**: General MPC computation engine
- **BeaverTriples**: Multiplication triples for secure computation
- **GarbledCircuits**: Yao's garbled circuits for boolean operations
- **SecureML**: Privacy-preserving machine learning protocols

#### Data Anonymization (`data_anonymization.py`)

- **KAnonymityAnonymizer**: k-anonymity with generalization
- **LDiversityAnonymizer**: l-diversity for sensitive attributes
- **DifferentialPrivacyAnonymizer**: DP-based data release
- **SyntheticDataGenerator**: Statistical synthetic data generation
- **PrivacyAuditor**: Privacy risk assessment

## Quick Start

### Example 1: Multi-Laboratory Joint Training

```python
from dftlammps.federated import (
    FederatedServer, FederatedClient, FederatedConfig,
    AggregationStrategy, create_federated_ml_system
)

# Create federated system
server, clients = create_federated_ml_system(num_institutions=5)

# Run training
history = server.train()

# Access results
print(f"Final loss: {history['train_loss'][-1]}")
```

### Example 2: Privacy-Preserving Drug Discovery

```python
from dftlammps.federated.federated_discovery import (
    FederatedDiscoveryCoordinator,
    FederatedDiscoveryConfig,
    DiscoveryStrategy
)

# Configure discovery
config = FederatedDiscoveryConfig(
    num_iterations=20,
    strategy=DiscoveryStrategy.FEDERATED_BO,
    use_dp=True,
    epsilon=1.0
)

# Create coordinator
coordinator = FederatedDiscoveryCoordinator(config)

# Run discovery
discovered = coordinator.run_discovery()
```

### Example 3: Homomorphic Encryption

```python
from dftlammps.privacy import PaillierEncryption

# Initialize encryption
he = PaillierEncryption(key_size=2048)
pk, sk = he.generate_keys()

# Encrypt and compute
m1 = 42
m2 = 23
c1 = he.encrypt(m1, pk)
c2 = he.encrypt(m2, pk)

# Homomorphic addition
c_sum = he.add(c1, c2)
m_sum = he.decrypt(c_sum, sk)

print(f"{m1} + {m2} = {m_sum}")  # Output: 65
```

### Example 4: Secure Multi-Party Computation

```python
from dftlammps.privacy import SecureComputation, MPCConfig

# Initialize MPC
config = MPCConfig(num_parties=3, threshold=2)
mpc = SecureComputation(config)

# Split secret
from dftlammps.privacy import SecretSharing
ss = SecretSharing()
shares = ss.split_secret(secret=12345, n=5, t=3)

# Reconstruct
reconstructed = ss.reconstruct_secret(shares[:3])
```

## Application Examples

### 1. Multi-Laboratory Joint Training of ML Potentials

Run the example:
```bash
python dftlammps/federated/examples/multi_lab_training.py
```

This example demonstrates:
- 5 research institutions collaborating on ML potential training
- Each institution keeps its training data private
- Secure aggregation of model updates
- Differential privacy protection with configurable epsilon
- Early stopping and convergence monitoring

### 2. Cross-Company Drug Discovery

Run the example:
```bash
python dftlammps/federated/examples/cross_company_drug_discovery.py
```

This example demonstrates:
- 3 pharmaceutical companies discovering drug candidates
- Privacy-preserving molecular screening
- Secure Bayesian optimization
- Cross-company IP protection
- Anonymized candidate sharing

### 3. Privacy-Preserving Catalyst Screening

Run the example:
```bash
python dftlammps/federated/examples/privacy_catalyst_screening.py
```

This example demonstrates:
- 4 chemical companies screening catalysts
- Homomorphic encryption for secure aggregation
- Federated high-throughput screening
- Multi-objective catalyst optimization
- Privacy compliance reporting

## Configuration Options

### FederatedConfig

```python
FederatedConfig(
    num_rounds=100,              # Number of training rounds
    num_clients=5,               # Number of participating clients
    clients_per_round=3,         # Clients per round
    local_epochs=5,              # Local training epochs
    batch_size=32,               # Batch size
    global_lr=1.0,               # Global learning rate
    local_lr=0.001,              # Local learning rate
    aggregation=AggregationStrategy.FEDAVG,
    use_secure_aggregation=True,
    use_differential_privacy=True,
    epsilon=1.0,                 # Privacy budget
    delta=1e-5,                  # Privacy failure probability
    max_grad_norm=1.0            # Gradient clipping norm
)
```

### FederatedDiscoveryConfig

```python
FederatedDiscoveryConfig(
    num_candidates=1000,
    num_iterations=50,
    strategy=DiscoveryStrategy.FEDERATED_BO,
    exploration_factor=0.1,
    acquisition_function="ei",  # "ei", "ucb", or "pi"
    use_dp=True,
    epsilon=1.0,
    use_mpc=True,
    num_parties=3,
    candidate_anonymization=True
)
```

## Privacy Guarantees

### Differential Privacy

The module implements (ε, δ)-differential privacy:
- **ε (epsilon)**: Privacy budget (smaller = more private)
- **δ (delta)**: Probability of privacy failure
- Uses moments accountant for tight privacy bounds
- Supports gradient clipping and calibrated noise

### Secure Aggregation

Secure aggregation ensures:
- Server sees only aggregated updates
- Individual client updates remain secret
- Tolerates dropout of up to n-t clients
- Uses secret sharing for robustness

### Homomorphic Encryption

Homomorphic encryption enables:
- Computation on encrypted data
- No decryption needed during aggregation
- Paillier for additive operations
- CKKS for approximate ML inference

## Performance Considerations

### Communication Costs

- **Federated Learning**: O(d × rounds) where d is model size
- **Secure Aggregation**: Additional overhead for secret sharing
- **Homomorphic Encryption**: Larger ciphertext sizes

### Computational Costs

- **Differential Privacy**: Minimal overhead (~5%)
- **Secure MPC**: Moderate overhead (~20-50%)
- **Homomorphic Encryption**: Significant overhead (~10-100x)

### Privacy-Utility Tradeoff

- Smaller ε → Better privacy, worse utility
- Larger noise → Better privacy, slower convergence
- More rounds → Better model, higher privacy cost

## Best Practices

1. **Set appropriate privacy budget**: ε=1-10 for moderate privacy, ε<1 for strong privacy
2. **Use gradient clipping**: Prevents privacy leakage from outliers
3. **Monitor privacy spent**: Track cumulative privacy budget
4. **Combine techniques**: Use DP + secure aggregation for strongest protection
5. **Validate anonymization**: Use privacy auditors to check guarantees

## Testing

Run demos:
```python
# Homomorphic encryption demo
from dftlammps.privacy.homomorphic_encryption import demo_homomorphic_encryption
demo_homomorphic_encryption()

# MPC demo
from dftlammps.privacy.secure_mpc import demo_mpc
demo_mpc()

# Anonymization demo
from dftlammps.privacy.data_anonymization import demo_anonymization
demo_anonymization()
```

## References

1. McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (AISTATS 2017)
2. Abadi et al. "Deep Learning with Differential Privacy" (CCS 2016)
3. Bonawitz et al. "Practical Secure Aggregation for Privacy-Preserving Machine Learning" (CCS 2017)
4. Shamir, "How to Share a Secret" (CACM 1979)
5. Paillier, "Public-Key Cryptosystems Based on Composite Degree Residue Classes" (EUROCRYPT 1999)

## License

This module is part of the DFT-LAMMPS project.
