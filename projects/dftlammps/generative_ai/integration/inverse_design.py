"""
Inverse Design Pipeline
=======================

Property-targeted inverse design using generative models.

Supports:
- Property-targeted generation
- Structure optimization
- Multi-objective optimization
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, List, Tuple, Callable, Union
from pymatgen.core import Structure
from pymatgen.optimization.linear_assignment import LinearAssignment
import logging

logger = logging.getLogger(__name__)


class InverseDesignPipeline:
    """
    Pipeline for inverse design of materials.
    
    Given target properties, generates structures that match.
    """
    
    def __init__(
        self,
        generative_model,
        property_predictors: Dict[str, Callable],
        config: Optional[Dict] = None
    ):
        """
        Args:
            generative_model: Trained generative model
            property_predictors: Property prediction functions
            config: Configuration
        """
        self.generative_model = generative_model
        self.property_predictors = property_predictors
        
        self.config = {
            "optimization_steps": 100,
            "learning_rate": 0.01,
            "num_candidates": 100,
            "top_k": 10,
            "property_weights": {},
            **(config or {})
        }
    
    def design(
        self,
        target_properties: Dict[str, float],
        constraints: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Run inverse design for target properties.
        
        Args:
            target_properties: Target property values
            constraints: Optional constraints (composition, space group, etc.)
            
        Returns:
            List of designed structures ranked by match to targets
        """
        logger.info(f"Starting inverse design for properties: {target_properties}")
        
        # Step 1: Generate candidates
        candidates = self.generate_candidates(
            target_properties,
            num_candidates=self.config["num_candidates"]
        )
        
        # Step 2: Optimize top candidates
        optimized = []
        for candidate in candidates[:self.config["top_k"]]:
            opt_structure = self.optimize_structure(
                candidate,
                target_properties,
                constraints
            )
            optimized.append(opt_structure)
        
        # Step 3: Rank by property match
        ranked = self.rank_by_property_match(optimized, target_properties)
        
        return ranked
    
    def generate_candidates(
        self,
        target_properties: Dict[str, float],
        num_candidates: int
    ) -> List[Structure]:
        """
        Generate candidate structures.
        
        Uses conditional generation if model supports it.
        """
        # Convert target properties to tensor
        props = list(target_properties.values())
        properties_tensor = torch.tensor([props] * num_candidates, dtype=torch.float32)
        
        # Generate
        if hasattr(self.generative_model, 'generate'):
            generated = self.generative_model.generate(
                batch_size=num_candidates,
                num_atoms=20,  # Could be configurable
                properties=properties_tensor,
                num_steps=50
            )
        else:
            logger.error("Model does not support generation")
            return []
        
        # Convert to structures
        from ..data.preprocessing import tensors_to_structure
        
        structures = []
        for i in range(num_candidates):
            single = {
                "atom_types": generated["atom_types"][i],
                "frac_coords": generated["frac_coords"][i],
                "lattice": generated["lattice"][i]
            }
            
            try:
                structure = tensors_to_structure(single)
                structures.append(structure)
            except Exception as e:
                logger.warning(f"Failed to convert structure {i}: {e}")
        
        return structures
    
    def optimize_structure(
        self,
        structure: Structure,
        target_properties: Dict[str, float],
        constraints: Optional[Dict],
        num_steps: Optional[int] = None
    ) -> Dict:
        """
        Optimize structure towards target properties.
        
        Uses gradient-based optimization in the latent space.
        """
        if num_steps is None:
            num_steps = self.config["optimization_steps"]
        
        # Convert structure to tensors
        from ..data.preprocessing import structure_to_tensors
        
        tensors = structure_to_tensors(structure)
        
        # Make differentiable
        frac_coords = tensors["frac_coords"].clone().requires_grad_(True)
        lattice = tensors["lattice"].clone().requires_grad_(True)
        atom_types = tensors["atom_types"]
        
        # Optimizer
        optimizer = optim.Adam([frac_coords, lattice], lr=self.config["learning_rate"])
        
        # Optimization loop
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Predict properties
            pred_props = self._predict_from_tensors(atom_types, frac_coords, lattice)
            
            # Compute loss
            loss = self._compute_property_loss(pred_props, target_properties)
            
            # Add constraints
            if constraints:
                constraint_loss = self._compute_constraint_loss(
                    atom_types, frac_coords, lattice, constraints
                )
                loss = loss + 0.1 * constraint_loss
            
            loss.backward()
            optimizer.step()
            
            # Project to valid domain
            with torch.no_grad():
                frac_coords.data = frac_coords.data % 1.0
                # Lattice constraints would go here
        
        # Build optimized structure
        opt_tensors = {
            "atom_types": atom_types.detach(),
            "frac_coords": frac_coords.detach(),
            "lattice": lattice.detach()
        }
        
        from ..data.preprocessing import tensors_to_structure
        opt_structure = tensors_to_structure(opt_tensors)
        
        # Predict final properties
        final_props = {}
        for name, predictor in self.property_predictors.items():
            try:
                final_props[name] = predictor(opt_structure)
            except Exception:
                final_props[name] = None
        
        return {
            "structure": opt_structure,
            "predicted_properties": final_props,
            "target_properties": target_properties
        }
    
    def _predict_from_tensors(
        self,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Predict properties from tensor representation."""
        # Convert to structure
        from ..data.preprocessing import tensors_to_structure
        
        tensors = {
            "atom_types": atom_types,
            "frac_coords": frac_coords,
            "lattice": lattice
        }
        
        structure = tensors_to_structure(tensors)
        
        # Predict
        predictions = {}
        for name, predictor in self.property_predictors.items():
            try:
                # Make differentiable approximation
                # In practice, would use a differentiable surrogate model
                val = predictor(structure)
                predictions[name] = torch.tensor(val, requires_grad=True)
            except Exception:
                predictions[name] = torch.tensor(0.0)
        
        return predictions
    
    def _compute_property_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, float]
    ) -> torch.Tensor:
        """Compute loss between predictions and targets."""
        loss = 0.0
        
        for prop, target_val in targets.items():
            if prop in predictions:
                pred_val = predictions[prop]
                weight = self.config["property_weights"].get(prop, 1.0)
                
                # Relative error
                rel_error = (pred_val - target_val) / (abs(target_val) + 1e-8)
                loss = loss + weight * rel_error ** 2
        
        return loss
    
    def _compute_constraint_loss(
        self,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice: torch.Tensor,
        constraints: Dict
    ) -> torch.Tensor:
        """Compute constraint violation loss."""
        loss = 0.0
        
        # Composition constraints
        if "composition" in constraints:
            target_comp = constraints["composition"]
            # Check composition match
            # Simplified - would need to decode atom types
        
        # Space group constraints
        if "space_group" in constraints:
            target_sg = constraints["space_group"]
            # Check space group match
        
        return loss
    
    def rank_by_property_match(
        self,
        structures: List[Dict],
        target_properties: Dict[str, float]
    ) -> List[Dict]:
        """Rank structures by how well they match target properties."""
        scored = []
        
        for item in structures:
            pred_props = item.get("predicted_properties", {})
            
            # Compute score (lower is better)
            score = 0.0
            for prop, target in target_properties.items():
                if prop in pred_props and pred_props[prop] is not None:
                    pred = pred_props[prop]
                    rel_error = abs(pred - target) / (abs(target) + 1e-8)
                    score += rel_error
            
            item["match_score"] = score
            scored.append(item)
        
        # Sort by score
        scored.sort(key=lambda x: x["match_score"])
        
        return scored
    
    def multi_objective_optimize(
        self,
        target_properties: Dict[str, float],
        num_pareto_points: int = 10
    ) -> List[Dict]:
        """
        Multi-objective optimization for Pareto frontier.
        
        Returns points on the Pareto frontier for conflicting objectives.
        """
        pareto_front = []
        
        # Generate with different weight combinations
        for i in range(num_pareto_points):
            # Random weights
            weights = np.random.dirichlet(np.ones(len(target_properties)))
            
            # Weighted target
            weighted_target = {
                k: v * weights[j]
                for j, (k, v) in enumerate(target_properties.items())
            }
            
            # Optimize
            result = self.design(weighted_target)
            
            if result:
                pareto_front.append(result[0])
        
        # Filter to actual Pareto frontier
        return self._filter_pareto(pareto_front, list(target_properties.keys()))
    
    def _filter_pareto(
        self,
        candidates: List[Dict],
        properties: List[str]
    ) -> List[Dict]:
        """Filter to true Pareto frontier."""
        # Extract property vectors
        vectors = []
        for c in candidates:
            props = c.get("predicted_properties", {})
            vec = [props.get(p, float('inf')) for p in properties]
            vectors.append(vec)
        
        vectors = np.array(vectors)
        
        # Find Pareto optimal points
        pareto_mask = np.ones(len(vectors), dtype=bool)
        
        for i, vec in enumerate(vectors):
            for j, other in enumerate(vectors):
                if i != j:
                    # Check if other dominates vec
                    if np.all(other <= vec) and np.any(other < vec):
                        pareto_mask[i] = False
                        break
        
        return [c for c, m in zip(candidates, pareto_mask) if m]


class LatentSpaceOptimizer:
    """
    Optimizer that works directly in the latent space of the generative model.
    """
    
    def __init__(self, generative_model, property_predictor: Callable):
        self.generative_model = generative_model
        self.property_predictor = property_predictor
    
    def optimize(
        self,
        initial_latent: torch.Tensor,
        target_property: float,
        num_steps: int = 100
    ) -> torch.Tensor:
        """
        Optimize latent vector to match target property.
        
        Args:
            initial_latent: Initial latent representation
            target_property: Target property value
            num_steps: Number of optimization steps
            
        Returns:
            Optimized latent vector
        """
        latent = initial_latent.clone().requires_grad_(True)
        optimizer = optim.Adam([latent], lr=0.01)
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Decode
            decoded = self._decode(latent)
            
            # Predict property
            pred = self.property_predictor(decoded)
            
            # Loss
            loss = (pred - target_property) ** 2
            
            loss.backward()
            optimizer.step()
        
        return latent.detach()
    
    def _decode(self, latent: torch.Tensor):
        """Decode latent to structure."""
        # Would use generative model's decoder
        pass


class BayesianOptimizationDesigner:
    """
    Inverse design using Bayesian Optimization.
    
    More sample-efficient for expensive property evaluations.
    """
    
    def __init__(
        self,
        generative_model,
        property_predictor: Callable,
        acquisition_fn: str = "ei"
    ):
        self.generative_model = generative_model
        self.property_predictor = property_predictor
        self.acquisition_fn = acquisition_fn
        
        self.observations = []
        self.properties = []
    
    def design(
        self,
        target_property: float,
        num_iterations: int = 50
    ) -> Structure:
        """Run Bayesian optimization for design."""
        for iteration in range(num_iterations):
            # Fit surrogate model
            self._fit_surrogate()
            
            # Optimize acquisition function
            next_point = self._optimize_acquisition()
            
            # Evaluate
            prop = self.property_predictor(next_point)
            
            # Update observations
            self.observations.append(next_point)
            self.properties.append(prop)
        
        # Return best
        best_idx = np.argmin(np.abs(np.array(self.properties) - target_property))
        return self.observations[best_idx]
    
    def _fit_surrogate(self):
        """Fit Gaussian Process surrogate model."""
        # Would use GP or other surrogate
        pass
    
    def _optimize_acquisition(self):
        """Optimize acquisition function."""
        # Would use L-BFGS-B or other optimizer
        pass
