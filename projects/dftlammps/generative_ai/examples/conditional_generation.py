"""
Example: Conditional Crystal Generation
========================================

Demonstrates conditional generation of crystals with target properties.
"""

import torch
from dftlammps.generative_ai import (
    CrystalDiT,
    CrystalDiTConfig,
    ConditionalDiffusion,
    ConditionalConfig,
    DiffusionTrainer
)
from dftlammps.generative_ai.data import (
    SyntheticCrystalDataset,
    CrystalPreprocessor,
    collate_crystal_batch
)
from torch.utils.data import DataLoader


def main():
    print("=" * 60)
    print("Conditional Crystal Generation Example")
    print("=" * 60)
    
    # Configuration
    config = CrystalDiTConfig(
        latent_dim=256,
        num_layers=6,
        num_heads=4,
        max_atoms=50,
        num_timesteps=1000,
        use_conditioning=True,
        num_properties=3
    )
    
    # Create base model
    print("\n1. Creating CrystalDiT model...")
    base_model = CrystalDiT(config)
    num_params = sum(p.numel() for p in base_model.parameters())
    print(f"   Model parameters: {num_params:,}")
    
    # Wrap with conditional generation
    print("\n2. Adding conditional generation wrapper...")
    cond_config = ConditionalConfig(
        num_properties=3,
        property_names=["band_gap", "formation_energy", "bulk_modulus"],
        use_cfg=True,
        guidance_scale=2.0
    )
    model = ConditionalDiffusion(base_model, cond_config)
    
    # Create synthetic dataset
    print("\n3. Creating synthetic dataset...")
    dataset = SyntheticCrystalDataset(
        num_samples=1000,
        min_atoms=5,
        max_atoms=50
    )
    
    # Preprocessor
    preprocessor = CrystalPreprocessor()
    
    # DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_crystal_batch
    )
    
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Batches: {len(train_loader)}")
    
    # Generate with target properties
    print("\n4. Generating structures with target properties...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Target: band_gap = 2.0 eV, formation_energy = -3.0 eV/atom
    target_properties = torch.tensor([[2.0, -3.0, 50.0]], device=device)
    
    with torch.no_grad():
        generated = model.generate(
            batch_size=5,
            num_atoms=20,
            properties=target_properties.expand(5, -1),
            num_steps=50
        )
    
    print(f"   Generated {len(generated['atom_types'])} structures")
    print(f"   Atom types shape: {generated['atom_types'].shape}")
    print(f"   Fractional coords shape: {generated['frac_coords'].shape}")
    
    # Save example
    print("\n5. Saving example structure...")
    from dftlammps.generative_ai.data import tensors_to_structure
    
    for i in range(min(3, len(generated['atom_types']))):
        single = {
            "atom_types": generated["atom_types"][i],
            "frac_coords": generated["frac_coords"][i],
            "lattice": generated["lattice"][i]
        }
        
        try:
            structure = tensors_to_structure(single)
            structure.to(filename=f"generated_structure_{i}.cif")
            print(f"   Saved: generated_structure_{i}.cif")
        except Exception as e:
            print(f"   Error saving structure {i}: {e}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
