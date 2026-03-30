"""
MD Potential Energy Surface Fitting

Integration of physics-constrained AI models for fitting molecular
dynamics potential energy surfaces.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Union
import os
import json


class MDPotentialFitter:
    """
    Fitter for molecular dynamics potential energy surfaces.
    
    Combines DFT data with physics constraints to train accurate
    and transferable interatomic potentials.
    
    Features:
    - Multi-fidelity training (DFT + experimental data)
    - Physics constraint enforcement
    - Active learning for data selection
    - Uncertainty quantification
    - Export to LAMMPS format
    """
    
    def __init__(
        self,
        model_type: str = 'egnn',
        model_config: Optional[Dict] = None,
        physics_constraints: Optional[List[str]] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize MD potential fitter.
        
        Args:
            model_type: Type of model ('egnn', 'painn', 'schnet', 'deeponet', 'fno')
            model_config: Model configuration
            physics_constraints: List of physics constraints to enforce
            device: Computing device
        """
        self.model_type = model_type
        self.model_config = model_config or {}
        self.physics_constraints = physics_constraints or ['energy', 'force']
        self.device = device
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'energy_mae': [],
            'force_mae': [],
            'physics_loss': []
        }
        
        # Normalization parameters
        self.energy_mean = 0.0
        self.energy_std = 1.0
        self.force_mean = 0.0
        self.force_std = 1.0
        
    def create_model(
        self,
        n_atom_types: int = 10,
        n_features: int = 128,
        **kwargs
    ):
        """
        Create the neural network model.
        
        Args:
            n_atom_types: Number of atom types
            n_features: Feature dimension
            **kwargs: Additional model arguments
        """
        if self.model_type == 'egnn':
            from ..models.physics_gnn import PhysicsInformedGNN
            self.model = PhysicsInformedGNN(
                node_dim=n_atom_types,
                hidden_dim=n_features,
                output_type='both',
                **{**self.model_config, **kwargs}
            )
            
        elif self.model_type == 'painn':
            # PaiNN model (would import from external library)
            raise NotImplementedError("PaiNN model not yet implemented")
            
        elif self.model_type == 'schnet':
            # SchNet model
            raise NotImplementedError("SchNet model not yet implemented")
            
        elif self.model_type == 'deeponet':
            from ..models.deeponet import DeepONet
            self.model = DeepONet(
                branch_input_dim=self.model_config.get('n_sensors', 100),
                trunk_input_dim=3,
                output_dim=1,
                **{**self.model_config, **kwargs}
            )
            
        elif self.model_type == 'fno':
            from ..models.fno import FourierNeuralOperator
            self.model = PhysicsInformedFNO(
                modes=self.model_config.get('modes', 16),
                width=n_features,
                dim=3,
                **{**self.model_config, **kwargs}
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model = self.model.to(self.device)
        
    def preprocess_data(
        self,
        positions: np.ndarray,
        energies: np.ndarray,
        forces: Optional[np.ndarray] = None,
        atom_types: Optional[np.ndarray] = None,
        normalize: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess training data.
        
        Args:
            positions: Atomic positions [n_configs, n_atoms, 3]
            energies: Total energies [n_configs]
            forces: Atomic forces [n_configs, n_atoms, 3]
            atom_types: Atom type indices [n_configs, n_atoms]
            normalize: Whether to normalize energies/forces
            
        Returns:
            Preprocessed data dictionary
        """
        # Convert to tensors
        data = {
            'positions': torch.tensor(positions, dtype=torch.float32, device=self.device),
            'energies': torch.tensor(energies, dtype=torch.float32, device=self.device),
        }
        
        if forces is not None:
            data['forces'] = torch.tensor(forces, dtype=torch.float32, device=self.device)
        
        if atom_types is not None:
            data['atom_types'] = torch.tensor(atom_types, dtype=torch.long, device=self.device)
        else:
            # Assume single atom type
            data['atom_types'] = torch.zeros(
                positions.shape[0], positions.shape[1],
                dtype=torch.long, device=self.device
            )
        
        # Normalize
        if normalize:
            self.energy_mean = data['energies'].mean().item()
            self.energy_std = data['energies'].std().item()
            data['energies'] = (data['energies'] - self.energy_mean) / (self.energy_std + 1e-8)
            
            if 'forces' in data:
                self.force_mean = data['forces'].mean().item()
                self.force_std = data['forces'].std().item()
                data['forces'] = (data['forces'] - self.force_mean) / (self.force_std + 1e-8)
        
        return data
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        lambda_energy: float = 1.0,
        lambda_force: float = 10.0,
        lambda_physics: float = 0.1
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute training loss with physics constraints.
        
        Args:
            predictions: Model predictions
            targets: Target values
            lambda_energy: Weight for energy loss
            lambda_force: Weight for force loss
            lambda_physics: Weight for physics constraint loss
            
        Returns:
            Total loss and individual loss components
        """
        losses = {}
        
        # Energy loss
        if 'energy' in predictions and 'energies' in targets:
            losses['energy'] = torch.nn.functional.mse_loss(
                predictions['energy'], targets['energies']
            )
        
        # Force loss
        if 'forces' in predictions and 'forces' in targets:
            losses['force'] = torch.nn.functional.mse_loss(
                predictions['forces'], targets['forces']
            )
        
        # Physics constraints
        physics_loss = torch.tensor(0.0, device=self.device)
        
        if 'energy_conservation' in self.physics_constraints:
            # Check energy consistency across batch
            if 'energy' in predictions:
                physics_loss = physics_loss + torch.var(predictions['energy'])
        
        if 'force_conservation' in self.physics_constraints:
            # Newton's 3rd law: sum of forces should be zero
            if 'forces' in predictions:
                total_force = torch.sum(predictions['forces'], dim=1)
                physics_loss = physics_loss + torch.mean(total_force ** 2)
        
        losses['physics'] = physics_loss
        
        # Total loss
        total_loss = (
            lambda_energy * losses.get('energy', 0) +
            lambda_force * losses.get('force', 0) +
            lambda_physics * physics_loss
        )
        
        return total_loss, losses
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            
        Returns:
            Average losses for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        total_energy_loss = 0.0
        total_force_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(
                node_attr=batch['atom_types'],
                pos=batch['positions']
            )
            
            # Compute loss
            loss, losses = self.compute_loss(predictions, batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_energy_loss += losses.get('energy', 0).item()
            total_force_loss += losses.get('force', 0).item()
            n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'energy_loss': total_energy_loss / n_batches,
            'force_loss': total_force_loss / n_batches
        }
    
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        energy_mae = 0.0
        force_mae = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                predictions = self.model(
                    node_attr=batch['atom_types'],
                    pos=batch['positions']
                )
                
                loss, _ = self.compute_loss(predictions, batch)
                
                # Compute MAE
                if 'energy' in predictions:
                    energy_mae += torch.mean(
                        torch.abs(predictions['energy'] - batch['energies'])
                    ).item()
                
                if 'forces' in predictions:
                    force_mae += torch.mean(
                        torch.abs(predictions['forces'] - batch['forces'])
                    ).item()
                
                total_loss += loss.item()
                n_batches += 1
        
        return {
            'val_loss': total_loss / n_batches,
            'energy_mae': energy_mae / n_batches * self.energy_std,
            'force_mae': force_mae / n_batches * self.force_std
        }
    
    def fit(
        self,
        train_data: Dict[str, torch.Tensor],
        val_data: Optional[Dict[str, torch.Tensor]] = None,
        n_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        patience: int = 20
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            n_epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        # Create data loaders
        from torch.utils.data import TensorDataset, DataLoader
        
        train_dataset = TensorDataset(
            train_data['positions'],
            train_data['energies'],
            train_data.get('forces', torch.zeros_like(train_data['positions'])),
            train_data['atom_types']
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        if val_data is not None:
            val_dataset = TensorDataset(
                val_data['positions'],
                val_data['energies'],
                val_data.get('forces', torch.zeros_like(val_data['positions'])),
                val_data['atom_types']
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=patience // 2
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, self.optimizer)
            
            # Validate
            if val_data is not None:
                val_metrics = self.validate(val_loader)
                self.scheduler.step(val_metrics['val_loss'])
                
                # Early stopping
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    patience_counter = 0
                    # Save best model
                    self.save_checkpoint('best_model.pt')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            if val_data is not None:
                self.history['val_loss'].append(val_metrics['val_loss'])
                self.history['energy_mae'].append(val_metrics['energy_mae'])
                self.history['force_mae'].append(val_metrics['force_mae'])
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.6f}")
                if val_data is not None:
                    print(f"  val_loss={val_metrics['val_loss']:.6f}, "
                          f"energy_mae={val_metrics['energy_mae']:.6f}, "
                          f"force_mae={val_metrics['force_mae']:.6f}")
        
        return self.history
    
    def predict(
        self,
        positions: np.ndarray,
        atom_types: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Predict energies and forces.
        
        Args:
            positions: Atomic positions [n_configs, n_atoms, 3]
            atom_types: Atom type indices [n_configs, n_atoms]
            
        Returns:
            Dictionary with 'energy' and 'forces' predictions
        """
        self.model.eval()
        
        if atom_types is None:
            atom_types = np.zeros(
                (positions.shape[0], positions.shape[1]), dtype=np.int64
            )
        
        pos_tensor = torch.tensor(
            positions, dtype=torch.float32, device=self.device
        )
        type_tensor = torch.tensor(
            atom_types, dtype=torch.long, device=self.device
        )
        
        with torch.no_grad():
            predictions = self.model(
                node_attr=type_tensor,
                pos=pos_tensor
            )
        
        # Denormalize and convert to numpy
        results = {}
        if 'energy' in predictions:
            results['energy'] = (
                predictions['energy'].cpu().numpy() * self.energy_std + 
                self.energy_mean
            )
        
        if 'forces' in predictions:
            results['forces'] = (
                predictions['forces'].cpu().numpy() * self.force_std + 
                self.force_mean
            )
        
        return results
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_type': self.model_type,
            'model_state_dict': self.model.state_dict(),
            'energy_mean': self.energy_mean,
            'energy_std': self.energy_std,
            'force_mean': self.force_mean,
            'force_std': self.force_std,
            'history': self.history
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model_type = checkpoint['model_type']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.energy_mean = checkpoint['energy_mean']
        self.energy_std = checkpoint['energy_std']
        self.force_mean = checkpoint['force_mean']
        self.force_std = checkpoint['force_std']
        self.history = checkpoint['history']
    
    def export_to_lammps(self, path: str):
        """
        Export model to LAMMPS MLIP format.
        
        Note: This is a placeholder. Full implementation would
        require custom LAMMPS pair style.
        """
        # Save model in TorchScript format
        self.model.eval()
        
        # Create a wrapper for LAMMPS
        class LAMMPSWrapper(torch.nn.Module):
            def __init__(self, model, energy_mean, energy_std):
                super().__init__()
                self.model = model
                self.energy_mean = energy_mean
                self.energy_std = energy_std
            
            def forward(self, positions, atom_types):
                output = self.model(atom_types, positions)
                energy = output['energy'] * self.energy_std + self.energy_mean
                forces = output['forces'] * self.energy_std  # Force scaling
                return energy, forces
        
        wrapper = LAMMPSWrapper(
            self.model, self.energy_mean, self.energy_std
        )
        
        # Script and save
        scripted = torch.jit.script(wrapper)
        scripted.save(path)
        
        print(f"Model exported to {path}")
        print("Note: Custom LAMMPS pair style required for loading")
    
    def active_learning_selection(
        self,
        pool_data: Dict[str, np.ndarray],
        n_select: int = 100,
        method: str = 'uncertainty'
    ) -> np.ndarray:
        """
        Select configurations for active learning.
        
        Args:
            pool_data: Pool of unlabeled configurations
            n_select: Number of configurations to select
            method: Selection method ('uncertainty', 'diversity', 'forces')
            
        Returns:
            Indices of selected configurations
        """
        self.model.eval()
        
        if method == 'uncertainty':
            # Use ensemble or dropout for uncertainty estimation
            # Placeholder: random selection
            indices = np.random.choice(
                len(pool_data['positions']),
                size=n_select,
                replace=False
            )
            
        elif method == 'diversity':
            # Select diverse configurations using clustering
            from sklearn.cluster import KMeans
            
            # Flatten positions for clustering
            features = pool_data['positions'].reshape(
                len(pool_data['positions']), -1
            )
            
            kmeans = KMeans(n_clusters=n_select, random_state=42)
            labels = kmeans.fit_predict(features)
            
            # Select one from each cluster
            indices = []
            for i in range(n_select):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) > 0:
                    indices.append(cluster_indices[0])
            indices = np.array(indices)
            
        elif method == 'forces':
            # Select configurations with high predicted forces
            predictions = self.predict(
                pool_data['positions'],
                pool_data.get('atom_types')
            )
            
            force_magnitudes = np.linalg.norm(predictions['forces'], axis=(1, 2))
            indices = np.argsort(force_magnitudes)[-n_select:]
            
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        return indices
