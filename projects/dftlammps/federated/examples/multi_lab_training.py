"""
Example: Multi-Laboratory Joint Training of ML Potentials
==========================================================

This example demonstrates how multiple laboratories can collaboratively
train a machine learning interatomic potential without sharing their
proprietary training data.

Scenario:
- 5 major research institutions want to create a universal ML potential
- Each has proprietary datasets of different material classes
- Data cannot leave institutional firewalls due to IP concerns
- The collaboration uses federated learning with differential privacy

Author: DFT-LAMMPS Team
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import json
import logging
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dftlammps.federated.federated_ml import (
    FederatedServer, FederatedClient, FederatedConfig,
    AggregationStrategy, MLPotentialModel, create_federated_ml_system
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Laboratory:
    """Represents a participating research laboratory."""
    
    def __init__(self, name: str, specialty: str, dataset_size: int):
        self.name = name
        self.specialty = specialty
        self.dataset_size = dataset_size
        self.client = None
        
    def __repr__(self):
        return f"Laboratory({self.name}, {self.specialty}, {self.dataset_size} samples)"


def create_laboratories() -> List[Laboratory]:
    """Create participating laboratories with their specialties."""
    laboratories = [
        Laboratory("MIT Materials Lab", "Battery Materials", 5000),
        Laboratory("Stanford Chemistry", "Catalysis", 4500),
        Laboratory("Berkeley Physics", "Semiconductors", 6000),
        Laboratory("Caltech Nanoscience", "2D Materials", 3500),
        Laboratory("Argonne National Lab", "Energy Storage", 8000),
    ]
    return laboratories


def generate_synthetic_training_data(lab: Laboratory, 
                                     feature_dim: int = 100,
                                     batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Generate synthetic training data for a laboratory.
    
    In practice, this would load actual DFT/Experimental data.
    """
    np.random.seed(hash(lab.name) % 2**32)
    
    # Generate features (e.g., SOAP descriptors)
    n_samples = lab.dataset_size
    
    # Different data distributions for different specialties
    if lab.specialty == "Battery Materials":
        # Li-based materials have distinct features
        features = np.random.randn(n_samples, feature_dim) + np.array([0.5] + [0]*99)
        targets = np.random.randn(n_samples, 1) - 3.5  # Lower formation energies
    elif lab.specialty == "Catalysis":
        # Transition metal features
        features = np.random.randn(n_samples, feature_dim) + np.array([0, 0.3] + [0]*98)
        targets = np.random.randn(n_samples, 1) - 2.0
    elif lab.specialty == "Semiconductors":
        features = np.random.randn(n_samples, feature_dim)
        targets = np.random.randn(n_samples, 1) - 1.5
    elif lab.specialty == "2D Materials":
        features = np.random.randn(n_samples, feature_dim) * 0.8
        targets = np.random.randn(n_samples, 1) - 1.0
    else:  # Energy Storage
        features = np.random.randn(n_samples, feature_dim)
        targets = np.random.randn(n_samples, 1) - 4.0
    
    # Split into train/val
    n_train = int(0.8 * n_samples)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(features[:n_train]),
        torch.FloatTensor(targets[:n_train])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(features[n_train:]),
        torch.FloatTensor(targets[n_train:])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


def setup_federated_system(laboratories: List[Laboratory]) -> Tuple[FederatedServer, List[FederatedClient]]:
    """
    Setup the federated learning system for all laboratories.
    """
    logger.info("Setting up federated learning system...")
    
    # Model configuration for ML potential
    model_config = {
        'input_dim': 100,
        'hidden_dims': [256, 256, 128],
        'output_dim': 1,
        'activation': 'silu'
    }
    
    # Create global model
    global_model = MLPotentialModel(**model_config)
    
    # Configuration with privacy settings
    config = FederatedConfig(
        num_rounds=20,
        num_clients=len(laboratories),
        clients_per_round=min(3, len(laboratories)),
        local_epochs=3,
        batch_size=32,
        global_lr=0.5,
        local_lr=0.001,
        aggregation=AggregationStrategy.FEDAVG,
        use_secure_aggregation=True,
        use_differential_privacy=True,
        epsilon=2.0,  # Privacy budget
        delta=1e-5,
        max_grad_norm=1.0,
        target_loss=0.01,
        patience=5
    )
    
    # Create server
    server = FederatedServer(global_model, config)
    
    # Create clients for each laboratory
    clients = []
    for i, lab in enumerate(laboratories):
        client_model = MLPotentialModel(**model_config)
        client = FederatedClient(
            client_id=f"lab_{i}",
            institution=lab.name,
            model=client_model,
            config=config
        )
        
        # Generate and set local data
        train_loader, val_loader = generate_synthetic_training_data(lab)
        client.set_data_loaders(train_loader, val_loader)
        
        # Register with server
        server.register_client(
            client_id=f"lab_{i}",
            institution=lab.name,
            data_size=lab.dataset_size
        )
        
        clients.append(client)
        lab.client = client
        
        logger.info(f"Registered {lab.name} with {lab.dataset_size} samples")
    
    return server, clients


def run_collaborative_training(server: FederatedServer, 
                               clients: List[FederatedClient]) -> Dict:
    """
    Run the federated training process.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Starting Multi-Laboratory Federated Training")
    logger.info("=" * 60)
    
    # Run training
    history = server.train()
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete")
    logger.info("=" * 60)
    
    return history


def analyze_privacy_guarantees(clients: List[FederatedClient]) -> Dict:
    """
    Analyze privacy guarantees for each laboratory.
    """
    privacy_analysis = {
        'total_epsilon': 0.0,
        'clients': []
    }
    
    for client in clients:
        epsilon_spent = client.get_privacy_spent()
        privacy_analysis['clients'].append({
            'institution': client.institution,
            'epsilon_spent': epsilon_spent,
            'local_steps': client.local_steps
        })
        privacy_analysis['total_epsilon'] += epsilon_spent
    
    return privacy_analysis


def save_results(server: FederatedServer, 
                clients: List[FederatedClient],
                history: Dict,
                output_dir: str = "./results/multi_lab_training"):
    """Save training results and analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save global model
    model_path = os.path.join(output_dir, "global_model.pt")
    torch.save(server.global_model.state_dict(), model_path)
    logger.info(f"Global model saved to {model_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    
    # Save privacy analysis
    privacy_analysis = analyze_privacy_guarantees(clients)
    privacy_path = os.path.join(output_dir, "privacy_analysis.json")
    with open(privacy_path, 'w') as f:
        json.dump(privacy_analysis, f, indent=2)
    logger.info(f"Privacy analysis saved to {privacy_path}")
    
    # Save collaboration report
    report = {
        'participating_institutions': [
            {
                'name': client.institution,
                'data_size': server.client_states[client.client_id].data_size,
                'rounds_participated': server.client_states[client.client_id].round_participated
            }
            for client in clients
        ],
        'final_loss': history['train_loss'][-1] if history['train_loss'] else None,
        'total_rounds': len(history['train_loss'])
    }
    
    report_path = os.path.join(output_dir, "collaboration_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Collaboration report saved to {report_path}")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("Multi-Laboratory Joint Training of ML Potentials")
    print("Privacy-Preserving Federated Learning Demonstration")
    print("=" * 70)
    
    # Step 1: Setup laboratories
    print("\n📋 Step 1: Setting up participating laboratories...")
    laboratories = create_laboratories()
    
    print("\nParticipating Laboratories:")
    for lab in laboratories:
        print(f"  • {lab.name}")
        print(f"    Specialty: {lab.specialty}")
        print(f"    Dataset: {lab.dataset_size:,} samples")
    
    # Step 2: Setup federated system
    print("\n🔧 Step 2: Setting up federated learning infrastructure...")
    server, clients = setup_federated_system(laboratories)
    
    print(f"\nFederated System Configuration:")
    print(f"  • Aggregation Strategy: {server.config.aggregation.value}")
    print(f"  • Secure Aggregation: {server.config.use_secure_aggregation}")
    print(f"  • Differential Privacy: {server.config.use_differential_privacy}")
    print(f"  • Privacy Budget (ε): {server.config.epsilon}")
    print(f"  • Training Rounds: {server.config.num_rounds}")
    
    # Step 3: Run training
    print("\n🚀 Step 3: Starting collaborative training...")
    history = run_collaborative_training(server, clients)
    
    # Step 4: Analyze results
    print("\n📊 Step 4: Analyzing results...")
    
    print(f"\nTraining Summary:")
    print(f"  • Total rounds completed: {len(history['train_loss'])}")
    print(f"  • Initial loss: {history['train_loss'][0]:.6f}")
    print(f"  • Final loss: {history['train_loss'][-1]:.6f}")
    print(f"  • Best loss: {min(history['train_loss']):.6f}")
    
    # Privacy analysis
    privacy_analysis = analyze_privacy_guarantees(clients)
    print(f"\nPrivacy Guarantees:")
    print(f"  • Total privacy budget spent: {privacy_analysis['total_epsilon']:.4f}")
    for client_info in privacy_analysis['clients']:
        print(f"  • {client_info['institution']}: ε = {client_info['epsilon_spent']:.4f}")
    
    # Step 5: Save results
    print("\n💾 Step 5: Saving results...")
    save_results(server, clients, history)
    
    print("\n" + "=" * 70)
    print("Multi-Laboratory Training Complete!")
    print("=" * 70)
    print("\n✅ Successfully demonstrated:")
    print("   • Collaborative training without data sharing")
    print("   • Secure aggregation of model updates")
    print("   • Differential privacy protection")
    print("   • Multi-institutional coordination")
    print("\n📁 Results saved to: ./results/multi_lab_training/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
