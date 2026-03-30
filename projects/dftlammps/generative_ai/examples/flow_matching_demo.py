"""
Example: Flow Matching for Crystal Generation
==============================================

Demonstrates flow matching for fast crystal generation.
"""

import torch
from dftlammps.generative_ai import (
    CrystalFlow,
    FlowMatchingConfig,
    FlowMatchingTrainer
)
from dftlammps.generative_ai.data import SyntheticCrystalDataset, collate_crystal_batch
from dftlammps.generative_ai.utils import FlowSampler, CrystalMetrics
from torch.utils.data import DataLoader
from pymatgen.core import Structure


def main():
    print("=" * 60)
    print("Flow Matching Crystal Generation Example")
    print("=" * 60)
    
    # Configuration
    config = FlowMatchingConfig(
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        num_steps=50,  # Fewer steps than diffusion
        ode_method="euler"
    )
    
    # Create model
    print("\n1. Creating CrystalFlow model...")
    model = CrystalFlow(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    
    # Dataset
    print("\n2. Creating dataset...")
    dataset = SyntheticCrystalDataset(num_samples=500)
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_crystal_batch
    )
    
    # Train briefly (demo only)
    print("\n3. Training for a few steps (demo)...")
    trainer = FlowMatchingTrainer(
        model=model,
        train_loader=train_loader,
        config={
            "max_epochs": 2,
            "log_every": 10
        }
    )
    
    # Just demo - don't actually train for long
    print("   (Skipping full training for demo)")
    
    # Generate
    print("\n4. Generating structures...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    sampler = FlowSampler(model, num_steps=50, method="euler")
    
    with torch.no_grad():
        generated = sampler.sample(
            batch_size=10,
            num_atoms=20,
            device=device
        )
    
    print(f"   Generated {len(generated['atom_types'])} structures")
    
    # Evaluate
    print("\n5. Evaluating generated structures...")
    from dftlammps.generative_ai.data import tensors_to_structure
    
    structures = []
    for i in range(len(generated['atom_types'])):
        single = {
            "atom_types": generated["atom_types"][i],
            "frac_coords": generated["frac_coords"][i],
            "lattice": generated["lattice"][i]
        }
        
        try:
            struct = tensors_to_structure(single)
            structures.append(struct)
        except Exception:
            pass
    
    print(f"   Successfully converted {len(structures)} structures")
    
    if structures:
        metrics = CrystalMetrics()
        validity = metrics.compute_validity(structures)
        print(f"   Validity: {validity}")
    
    print("\n" + "=" * 60)
    print("Flow matching example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
