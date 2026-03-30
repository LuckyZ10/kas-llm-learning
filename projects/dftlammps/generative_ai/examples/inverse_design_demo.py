"""
Example: Inverse Design Pipeline
=================================

Demonstrates property-targeted inverse design.
"""

import torch
import numpy as np
from dftlammps.generative_ai import (
    CrystalDiT,
    CrystalDiTConfig,
    ConditionalDiffusion,
    ConditionalConfig
)
from dftlammps.generative_ai.integration import InverseDesignPipeline


def dummy_property_predictor(structure):
    """Dummy property predictor for demo."""
    # In real use, would use M3GNet, CHGNet, etc.
    return np.random.uniform(0, 5)


def main():
    print("=" * 60)
    print("Inverse Design Pipeline Example")
    print("=" * 60)
    
    # Create model
    print("\n1. Setting up generative model...")
    config = CrystalDiTConfig(
        latent_dim=256,
        num_layers=6,
        num_heads=4,
        use_conditioning=True,
        num_properties=2
    )
    
    base_model = CrystalDiT(config)
    cond_config = ConditionalConfig(
        num_properties=2,
        property_names=["band_gap", "formation_energy"],
        use_cfg=True,
        guidance_scale=2.0
    )
    model = ConditionalDiffusion(base_model, cond_config)
    
    # Property predictors
    predictors = {
        "band_gap": dummy_property_predictor,
        "formation_energy": dummy_property_predictor
    }
    
    # Create pipeline
    print("\n2. Creating inverse design pipeline...")
    pipeline = InverseDesignPipeline(
        generative_model=model,
        property_predictors=predictors,
        config={
            "num_candidates": 20,
            "top_k": 5,
            "optimization_steps": 20
        }
    )
    
    # Target properties
    print("\n3. Running inverse design...")
    print("   Target: band_gap = 2.5 eV, formation_energy = -4.0 eV/atom")
    
    target_properties = {
        "band_gap": 2.5,
        "formation_energy": -4.0
    }
    
    # Note: This would take time with a real model
    print("   (Skipping actual optimization for demo)")
    
    # Generate candidates
    print("\n4. Generating candidate structures...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    target_props_tensor = torch.tensor([[2.5, -4.0]], device=device)
    
    with torch.no_grad():
        candidates = model.generate(
            batch_size=5,
            num_atoms=20,
            properties=target_props_tensor.expand(5, -1),
            num_steps=50
        )
    
    print(f"   Generated {len(candidates['atom_types'])} candidates")
    
    # Multi-objective optimization demo
    print("\n5. Multi-objective optimization demo...")
    print("   Finding Pareto frontier for conflicting objectives...")
    
    # In real use, would call:
    # pareto_frontier = pipeline.multi_objective_optimize(
    #     target_properties={"band_gap": 2.0, "ionic_conductivity": 1.0},
    #     num_pareto_points=10
    # )
    
    print("   (Demo: Pareto frontier would have multiple points)")
    
    # Save results
    print("\n6. Saving results...")
    from dftlammps.generative_ai.data import tensors_to_structure
    
    for i in range(len(candidates['atom_types'])):
        single = {
            "atom_types": candidates["atom_types"][i],
            "frac_coords": candidates["frac_coords"][i],
            "lattice": candidates["lattice"][i]
        }
        
        try:
            struct = tensors_to_structure(single)
            struct.to(filename=f"inverse_design_candidate_{i}.cif")
            print(f"   Saved: inverse_design_candidate_{i}.cif")
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    print("Inverse design example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
