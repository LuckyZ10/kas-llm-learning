#!/usr/bin/env python3
"""
Test suite for multiscale coupling module.
"""
import unittest
import numpy as np
import sys
import os

# Add workspace to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from dftlammps.multiscale_coupling import (
    QMMMInterface, CoarseGrainer, CGGNN, 
    CrossScaleValidator, MultiscaleGNN
)
from dftlammps.multiscale_coupling.utils import (
    UnitConverter, AtomSelection, BoundaryHandler
)
from dftlammps.multiscale_coupling.ml_cg import CGMapping, ForceMatcher
from dftlammps.multiscale_coupling.gnn_models import build_graph, Graph
from dftlammps.multiscale_coupling.validation import ValidationResult


class TestUnitConverter(unittest.TestCase):
    """Test unit conversion utilities."""
    
    def test_energy_conversion(self):
        """Test energy unit conversions."""
        ev_value = 1.0
        kcal = UnitConverter.energy_eV_to_kcal(ev_value)
        self.assertAlmostEqual(kcal, 23.0605, places=4)
        
        back_to_ev = UnitConverter.energy_kcal_to_eV(kcal)
        self.assertAlmostEqual(back_to_ev, ev_value, places=6)
    
    def test_length_conversion(self):
        """Test length unit conversions."""
        ang_value = 1.0
        bohr = UnitConverter.length_ang_to_bohr(ang_value)
        self.assertAlmostEqual(bohr, 1.88973, places=5)
        
        back_to_ang = UnitConverter.length_bohr_to_ang(bohr)
        self.assertAlmostEqual(back_to_ang, ang_value, places=4)


class TestAtomSelection(unittest.TestCase):
    """Test atom selection tools."""
    
    def test_select_sphere(self):
        """Test spherical selection."""
        positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [5, 0, 0],
            [10, 0, 0]
        ])
        center = np.array([0, 0, 0])
        radius = 3.0
        
        selected = AtomSelection.select_sphere(positions, center, radius)
        
        self.assertTrue(selected[0])
        self.assertTrue(selected[1])
        self.assertFalse(selected[2])
        self.assertFalse(selected[3])
    
    def test_select_index_range(self):
        """Test index range selection."""
        mask = AtomSelection.select_index_range(2, 5, 10)
        
        self.assertFalse(mask[0])
        self.assertFalse(mask[1])
        self.assertTrue(mask[2])
        self.assertTrue(mask[3])
        self.assertTrue(mask[4])
        self.assertFalse(mask[5])


class TestBoundaryHandler(unittest.TestCase):
    """Test boundary handling."""
    
    def test_link_atoms(self):
        """Test link atom placement."""
        positions = np.array([
            [0, 0, 0],    # QM
            [1.5, 0, 0],  # MM
            [3, 0, 0]     # MM
        ])
        qm_mask = np.array([True, False, False])
        mm_mask = np.array([False, True, True])
        
        handler = BoundaryHandler(qm_mask, mm_mask)
        
        # Mock boundary bond
        handler.boundary_pairs = [(0, 1)]
        
        new_positions, link_indices = handler.add_link_atoms(positions)
        
        self.assertEqual(len(new_positions), 4)  # Added 1 link atom
        self.assertEqual(len(link_indices), 1)
        
        # Link atom should be placed along the bond from QM atom
        link_pos = new_positions[link_indices[0]]
        # Link atom should be at distance 1.09 from QM atom
        qm_pos = positions[0]  # [0, 0, 0]
        dist = np.linalg.norm(link_pos - qm_pos)
        self.assertAlmostEqual(dist, 1.09, places=2)


class TestCGMapping(unittest.TestCase):
    """Test coarse-grained mapping."""
    
    def test_mapping_creation(self):
        """Test CG mapping creation."""
        atom_to_bead = np.array([0, 0, 1, 1, 1])
        bead_positions = np.array([[0, 0, 0], [1, 1, 1]])
        bead_types = ['A', 'B']
        
        mapping = CGMapping(
            atom_to_bead=atom_to_bead,
            bead_positions=bead_positions,
            bead_types=bead_types,
            n_beads=2,
            n_atoms=5
        )
        
        self.assertEqual(mapping.n_beads, 2)
        self.assertEqual(mapping.n_atoms, 5)
    
    def test_force_matching(self):
        """Test force matching."""
        atom_to_bead = np.array([0, 0, 1, 1])
        mapping = CGMapping(
            atom_to_bead=atom_to_bead,
            bead_positions=np.array([[0, 0, 0], [1, 1, 1]]),
            bead_types=['A', 'B'],
            n_beads=2,
            n_atoms=4
        )
        
        fm = ForceMatcher(mapping)
        
        atom_positions = np.array([
            [[0, 0, 0], [0.1, 0, 0], [1, 0, 0], [1.1, 0, 0]]
        ])
        atom_forces = np.array([
            [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]
        ])
        
        cg_forces = fm.compute_reference_forces(atom_positions, atom_forces)
        
        self.assertEqual(cg_forces.shape, (1, 2, 3))
        # Bead 0 should have force = 1 + 2 = 3
        self.assertAlmostEqual(cg_forces[0, 0, 0], 3.0)
        # Bead 1 should have force = 3 + 4 = 7
        self.assertAlmostEqual(cg_forces[0, 1, 0], 7.0)


class TestGraphBuilding(unittest.TestCase):
    """Test graph construction."""
    
    def test_build_graph(self):
        """Test graph building."""
        positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [5, 0, 0]
        ])
        elements = ['O', 'H', 'H']
        
        graph = build_graph(positions, elements, cutoff=2.0)
        
        self.assertEqual(graph.n_nodes, 3)
        # First two atoms should be connected, third shouldn't
        self.assertGreater(graph.n_edges, 0)
    
    def test_graph_properties(self):
        """Test graph properties."""
        nodes = np.array([[1, 0], [0, 1]])
        edges = np.array([[0, 1], [1, 0]])
        edge_features = np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        
        graph = Graph(
            nodes=nodes,
            edges=edges,
            edge_features=edge_features,
            positions=positions,
            node_types=['A', 'B']
        )
        
        self.assertEqual(graph.n_nodes, 2)
        self.assertEqual(graph.n_edges, 2)


class TestCrossScaleValidator(unittest.TestCase):
    """Test validation tools."""
    
    def test_energy_conservation(self):
        """Test energy conservation check."""
        validator = CrossScaleValidator()
        
        # Conserved energies
        energies = np.array([100.0, 100.001, 99.999, 100.002])
        result = validator.validate_energy_conservation(energies)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertIn('drift', result.details)
    
    def test_force_consistency(self):
        """Test force consistency check."""
        validator = CrossScaleValidator()
        
        atom_forces = np.array([
            [1, 0, 0], [2, 0, 0],  # Bead 0
            [3, 0, 0], [4, 0, 0]   # Bead 1
        ])
        cg_forces = np.array([
            [3, 0, 0],  # 1 + 2
            [7, 0, 0]   # 3 + 4
        ])
        atom_to_cg = np.array([0, 0, 1, 1])
        
        result = validator.validate_force_consistency(
            atom_forces, cg_forces, atom_to_cg
        )
        
        self.assertIsInstance(result, ValidationResult)
        # Should have very small error
        self.assertLess(result.score, 0.01)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_end_to_end_cg(self):
        """Test end-to-end coarse-graining workflow."""
        # Create simple system
        n_atoms = 8
        n_frames = 10
        positions = np.random.randn(n_frames, n_atoms, 3)
        atom_types = ['C'] * 4 + ['H'] * 4
        
        # Create mapping
        from dftlammps.multiscale_coupling.ml_cg import CentroidCoarseGrainer
        
        cg = CentroidCoarseGrainer(n_beads=2)
        mapping = cg.fit([positions], atom_types)
        
        self.assertEqual(mapping.n_beads, 2)
        
        # Transform
        cg_traj = cg.transform(positions)
        self.assertEqual(cg_traj.shape, (n_frames, 2, 3))


class TestGNNModels(unittest.TestCase):
    """Test GNN models."""
    
    def test_cggnn_initialization(self):
        """Test CG-GNN initialization."""
        model = CGGNN(
            n_node_features=5,
            hidden_dim=32,
            n_layers=3
        )
        
        self.assertEqual(model.n_layers, 3)
        self.assertEqual(model.hidden_dim, 32)
    
    def test_multiscale_gnn(self):
        """Test multiscale GNN."""
        model = MultiscaleGNN(
            atom_features=2,
            cg_features=1,
            hidden_dim=16,
            n_atom_layers=2,
            n_cg_layers=2
        )
        
        self.assertIsNotNone(model.atom_gnn)
        self.assertIsNotNone(model.cg_gnn)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestUnitConverter))
    suite.addTests(loader.loadTestsFromTestCase(TestAtomSelection))
    suite.addTests(loader.loadTestsFromTestCase(TestBoundaryHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestCGMapping))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphBuilding))
    suite.addTests(loader.loadTestsFromTestCase(TestCrossScaleValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestGNNModels))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
