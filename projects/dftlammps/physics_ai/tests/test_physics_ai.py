"""
Tests for physics_ai module.
"""

import unittest
import torch
import numpy as np
from dftlammps.physics_ai.core import PhysicsConstraintLayer, EnergyConservation, MomentumConservation
from dftlammps.physics_ai.models import PhysicsInformedNN, DeepONet, FourierNeuralOperator, PhysicsInformedGNN
from dftlammps.physics_ai.validators import PhysicsLawValidator


class TestPhysicsConstraintLayer(unittest.TestCase):
    """Test physics constraint layer."""
    
    def test_initialization(self):
        layer = PhysicsConstraintLayer(
            constraints=['energy', 'momentum'],
            enforcement='soft',
            weight=0.1
        )
        self.assertEqual(layer.constraint_types, ['energy', 'momentum'])
        self.assertEqual(layer.enforcement, 'soft')
    
    def test_energy_conservation_loss(self):
        layer = PhysicsConstraintLayer(['energy'])
        
        # Create synthetic data with constant energy
        positions = torch.randn(10, 5, 3)
        velocities = torch.randn(10, 5, 3)
        masses = torch.ones(10, 5)
        
        # Potential function
        def potential_fn(pos):
            return torch.zeros(pos.shape[0])
        
        loss = layer.energy_conservation_loss(
            positions, velocities, masses, potential_fn
        )
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)


class TestConservationLaws(unittest.TestCase):
    """Test conservation law implementations."""
    
    def test_energy_conservation(self):
        law = EnergyConservation()
        
        state = {
            'positions': torch.randn(10, 5, 3),
            'velocities': torch.randn(10, 5, 3),
            'masses': torch.ones(10, 5)
        }
        
        # Compute kinetic energy
        ke = law.compute_kinetic_energy(state['velocities'], state['masses'])
        self.assertEqual(ke.shape, (10,))
        self.assertTrue((ke >= 0).all())
    
    def test_momentum_conservation(self):
        law = MomentumConservation()
        
        state = {
            'positions': torch.randn(10, 5, 3),
            'velocities': torch.randn(10, 5, 3),
            'masses': torch.ones(10, 5)
        }
        
        # Compute linear momentum
        p = law.compute_linear_momentum(state['velocities'], state['masses'])
        self.assertEqual(p.shape, (10, 3))


class TestPINN(unittest.TestCase):
    """Test Physics-Informed Neural Network."""
    
    def test_pinn_forward(self):
        def simple_pde(x, u, derivatives):
            return u[:, 0] - x[:, 0]
        
        model = PhysicsInformedNN(
            input_dim=1,
            output_dim=1,
            hidden_dims=[32, 32],
            pde_fn=simple_pde
        )
        
        x = torch.randn(10, 1)
        output = model(x)
        
        self.assertEqual(output.shape, (10, 1))
    
    def test_pinn_loss(self):
        def simple_pde(x, u, derivatives):
            return u[:, 0]
        
        model = PhysicsInformedNN(
            input_dim=1,
            output_dim=1,
            hidden_dims=[32, 32],
            pde_fn=simple_pde
        )
        
        x_data = torch.randn(5, 1)
        u_data = torch.randn(5, 1)
        x_collocation = torch.randn(10, 1)
        
        losses = model.compute_loss(
            x_data=x_data,
            u_data=u_data,
            x_collocation=x_collocation
        )
        
        self.assertIn('data', losses)
        self.assertIn('pde', losses)
        self.assertIn('total', losses)


class TestDeepONet(unittest.TestCase):
    """Test Deep Operator Network."""
    
    def test_deeponet_forward(self):
        model = DeepONet(
            branch_input_dim=100,
            trunk_input_dim=1,
            output_dim=1,
            branch_hidden_dims=[32, 32],
            trunk_hidden_dims=[32, 32]
        )
        
        u = torch.randn(5, 100)
        y = torch.randn(20, 1)
        
        output = model(u, y)
        
        self.assertEqual(output.shape, (5, 20, 1))
    
    def test_deeponet_same_batch(self):
        model = DeepONet(
            branch_input_dim=50,
            trunk_input_dim=1,
            output_dim=1,
            branch_hidden_dims=[32, 32],
            trunk_hidden_dims=[32, 32]
        )
        
        u = torch.randn(5, 50)
        y = torch.randn(5, 1)
        
        output = model(u, y)
        
        self.assertEqual(output.shape, (5, 1))


class TestFNO(unittest.TestCase):
    """Test Fourier Neural Operator."""
    
    def test_fno_1d(self):
        model = FourierNeuralOperator(
            modes=16,
            width=32,
            in_channels=1,
            out_channels=1,
            n_layers=3,
            dim=1
        )
        
        x = torch.randn(2, 1, 64)
        output = model(x)
        
        self.assertEqual(output.shape, (2, 64, 1))
    
    def test_fno_2d(self):
        model = FourierNeuralOperator(
            modes=(12, 12),
            width=32,
            in_channels=1,
            out_channels=1,
            n_layers=3,
            dim=2
        )
        
        x = torch.randn(2, 1, 32, 32)
        output = model(x)
        
        self.assertEqual(output.shape, (2, 32, 32, 1))


class TestPhysicsGNN(unittest.TestCase):
    """Test Physics-Informed GNN."""
    
    def test_egnn_layer(self):
        from dftlammps.physics_ai.models.physics_gnn import EGNNLayer
        
        layer = EGNNLayer(node_dim=64, edge_dim=0, hidden_dim=64)
        
        h = torch.randn(10, 64)
        x = torch.randn(10, 3)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        
        h_out, x_out = layer(h, x, edge_index)
        
        self.assertEqual(h_out.shape, (10, 64))
        self.assertEqual(x_out.shape, (10, 3))
    
    def test_physics_gnn_forward(self):
        model = PhysicsInformedGNN(
            node_dim=5,
            hidden_dim=32,
            n_layers=2,
            output_type='both'
        )
        
        node_attr = torch.randint(0, 5, (10,))
        pos = torch.randn(10, 3)
        
        output = model(node_attr, pos)
        
        self.assertIn('energy', output)
        self.assertIn('forces', output)


class TestPhysicsValidator(unittest.TestCase):
    """Test physics law validator."""
    
    def test_validator_initialization(self):
        validator = PhysicsLawValidator(tolerance=1e-4)
        self.assertEqual(validator.tolerance, 1e-4)
        self.assertTrue(len(validator.tests) > 0)
    
    def test_trajectory_validation(self):
        validator = PhysicsLawValidator(tolerance=1e-3)
        
        # Create simple trajectory
        n_steps = 100
        trajectory = np.random.randn(n_steps, 5, 6)
        
        results = validator.validate_trajectory(trajectory, dt=0.001)
        
        self.assertIn('energy_conservation', results)
        self.assertIn('momentum_conservation', results)
    
    def test_model_validation(self):
        validator = PhysicsLawValidator(tolerance=1e-3)
        
        model = PhysicsInformedGNN(
            node_dim=5,
            hidden_dim=32,
            output_type='both'
        )
        
        test_data = {
            'positions': torch.randn(4, 10, 3),
            'velocities': torch.randn(4, 10, 3),
            'masses': torch.ones(4, 10)
        }
        
        results = validator.validate_model(model, test_data)
        
        self.assertTrue(len(results) > 0)


class TestSymbolicRegression(unittest.TestCase):
    """Test symbolic regression engine."""
    
    def test_symbolic_expression(self):
        from dftlammps.physics_ai.symbolic import SymbolicExpression
        
        expr = SymbolicExpression(
            expression="x**2 + y",
            variables=['x', 'y']
        )
        
        result = expr.evaluate(x=2, y=3)
        self.assertEqual(result, 7)
    
    def test_symbolic_engine_creation(self):
        from dftlammps.physics_ai.symbolic import SymbolicRegressionEngine
        
        engine = SymbolicRegressionEngine(backend='gplearn')
        self.assertEqual(engine.backend_name, 'gplearn')


class TestMDPotentialFitter(unittest.TestCase):
    """Test MD potential fitter."""
    
    def test_fitter_creation(self):
        from dftlammps.physics_ai.integration import MDPotentialFitter
        
        fitter = MDPotentialFitter(
            model_type='egnn',
            physics_constraints=['energy', 'force']
        )
        
        self.assertEqual(fitter.model_type, 'egnn')
        self.assertEqual(fitter.physics_constraints, ['energy', 'force'])
    
    def test_data_preprocessing(self):
        from dftlammps.physics_ai.integration import MDPotentialFitter
        
        fitter = MDPotentialFitter()
        fitter.create_model(n_atom_types=2)
        
        positions = np.random.randn(10, 5, 3)
        energies = np.random.randn(10)
        forces = np.random.randn(10, 5, 3)
        atom_types = np.random.randint(0, 2, (10, 5))
        
        data = fitter.preprocess_data(
            positions, energies, forces, atom_types
        )
        
        self.assertIn('positions', data)
        self.assertIn('energies', data)
        self.assertIn('forces', data)
        self.assertIn('atom_types', data)


if __name__ == '__main__':
    unittest.main()
