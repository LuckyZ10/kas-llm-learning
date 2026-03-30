"""
Example: Fourier Neural Operator for PDEs

This example demonstrates FNO for solving the 2D Darcy flow equation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dftlammps.physics_ai.models import FourierNeuralOperator, PhysicsInformedFNO


def generate_darcy_data(n_samples=1000, resolution=64):
    """
    Generate data for 2D Darcy flow.
    
    PDE: -∇·(a(x)∇u(x)) = f(x) in [0,1]^2
    where a(x) is the permeability field.
    
    Returns:
        a: Permeability fields [n_samples, 1, res, res]
        u: Solution fields [n_samples, 1, res, res]
    """
    res = resolution
    
    # Generate permeability fields (log-normal)
    a = np.zeros((n_samples, 1, res, res))
    u = np.zeros((n_samples, 1, res, res))
    
    # Grid
    x = np.linspace(0, 1, res)
    y = np.linspace(0, 1, res)
    X, Y = np.meshgrid(x, y)
    
    for i in range(n_samples):
        # Random permeability field (smooth)
        n_modes = 10
        a_field = np.zeros((res, res))
        
        for _ in range(n_modes):
            kx = np.random.randint(1, 5)
            ky = np.random.randint(1, 5)
            amp = np.random.randn() / (kx**2 + ky**2)
            phase = np.random.rand() * 2 * np.pi
            a_field += amp * np.sin(2 * np.pi * (kx * X + ky * Y) + phase)
        
        # Make positive (log-normal)
        a_field = np.exp(a_field)
        a[i, 0] = a_field
        
        # Simplified solution (would solve actual PDE in practice)
        # For demo purposes, use a simple transformation
        u[i, 0] = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y) / (a_field + 1)
    
    return (
        torch.tensor(a, dtype=torch.float32),
        torch.tensor(u, dtype=torch.float32)
    )


def train_fno():
    """Train FNO for Darcy flow."""
    
    print("Generating data...")
    a_train, u_train = generate_darcy_data(n_samples=1000, resolution=64)
    a_test, u_test = generate_darcy_data(n_samples=200, resolution=64)
    
    print(f"Data shape: {a_train.shape}")
    
    print("Creating FNO model...")
    model = FourierNeuralOperator(
        modes=(12, 12),
        width=32,
        in_channels=1,
        out_channels=1,
        n_layers=4,
        dim=2
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    # Training
    n_epochs = 500
    batch_size = 20
    
    print("Training...")
    for epoch in range(n_epochs):
        # Sample batch
        idx = np.random.choice(len(a_train), batch_size, replace=False)
        a_batch = a_train[idx]
        u_batch = u_train[idx]
        
        optimizer.zero_grad()
        
        # Forward pass
        u_pred = model(a_batch)
        
        # Loss
        loss = torch.mean((u_pred - u_batch) ** 2)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 50 == 0:
            # Validation
            with torch.no_grad():
                u_val_pred = model(a_test)
                val_loss = torch.mean((u_val_pred - u_test) ** 2)
            
            print(f"Epoch {epoch}: Train Loss={loss.item():.6f}, "
                  f"Val Loss={val_loss.item():.6f}")
    
    return model, a_test, u_test


def evaluate_fno(model, a_test, u_test):
    """Evaluate FNO and visualize results."""
    
    model.eval()
    with torch.no_grad():
        u_pred = model(a_test)
    
    # Convert to numpy
    a_np = a_test.numpy()
    u_true_np = u_test.numpy()
    u_pred_np = u_pred.numpy()
    
    # Compute errors
    mse = np.mean((u_pred_np - u_true_np) ** 2)
    mae = np.mean(np.abs(u_pred_np - u_true_np))
    relative_l2 = np.mean(
        np.linalg.norm(u_pred_np - u_true_np, axis=(2, 3)) /
        np.linalg.norm(u_true_np, axis=(2, 3))
    )
    
    print(f"\nTest Results:")
    print(f"  MSE: {mse:.6e}")
    print(f"  MAE: {mae:.6e}")
    print(f"  Relative L2: {relative_l2:.6e}")
    
    # Visualize
    n_examples = 3
    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 4 * n_examples))
    
    for i in range(n_examples):
        # Input permeability
        im1 = axes[i, 0].imshow(a_np[i, 0], cmap='viridis')
        axes[i, 0].set_title('Permeability a(x)')
        axes[i, 0].set_xlabel('x')
        axes[i, 0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # True solution
        im2 = axes[i, 1].imshow(u_true_np[i, 0], cmap='viridis')
        axes[i, 1].set_title('True Solution u(x)')
        axes[i, 1].set_xlabel('x')
        axes[i, 1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[i, 1])
        
        # Predicted solution
        im3 = axes[i, 2].imshow(u_pred_np[i, 0], cmap='viridis')
        axes[i, 2].set_title('FNO Prediction')
        axes[i, 2].set_xlabel('x')
        axes[i, 2].set_ylabel('y')
        plt.colorbar(im3, ax=axes[i, 2])
    
    plt.tight_layout()
    plt.savefig('fno_darcy_flow.png', dpi=150)
    plt.close()
    
    print("\nVisualization saved to fno_darcy_flow.png")
    
    # Test resolution invariance
    test_resolution_invariance(model)


def test_resolution_invariance(model):
    """Test if FNO is resolution invariant."""
    print("\n" + "=" * 60)
    print("Testing Resolution Invariance")
    print("=" * 60)
    
    # Test on different resolutions
    resolutions = [32, 64, 128]
    
    for res in resolutions:
        a_res, u_res = generate_darcy_data(n_samples=50, resolution=res)
        
        with torch.no_grad():
            u_pred = model(a_res)
        
        mse = torch.mean((u_pred - u_res) ** 2).item()
        print(f"Resolution {res}x{res}: MSE = {mse:.6e}")


def example_physics_informed_fno():
    """Example of physics-informed FNO."""
    print("\n" + "=" * 60)
    print("Physics-Informed FNO Example")
    print("=" * 60)
    
    # Create physics-informed FNO
    model = PhysicsInformedFNO(
        modes=(8, 8),
        width=32,
        in_channels=1,
        out_channels=1,
        n_layers=3,
        dim=2
    )
    
    # Define PDE residual (simplified Laplace equation)
    def laplace_residual(u, x):
        """Laplace equation residual: ∇²u = 0."""
        # Compute Laplacian using finite differences
        # This is a simplified version
        laplacian = torch.zeros_like(u)
        
        # Interior points
        laplacian[:, :, 1:-1, 1:-1] = (
            u[:, :, 2:, 1:-1] + u[:, :, :-2, 1:-1] +
            u[:, :, 1:-1, 2:] + u[:, :, 1:-1, :-2] -
            4 * u[:, :, 1:-1, 1:-1]
        )
        
        return laplacian
    
    model.set_physics_loss(laplace_residual, weight=0.1)
    
    print("Created Physics-Informed FNO")
    print("PDE: Laplace equation (∇²u = 0)")
    
    return model


if __name__ == '__main__':
    print("Fourier Neural Operator Examples")
    print("=" * 60)
    
    # Main FNO example
    model, a_test, u_test = train_fno()
    evaluate_fno(model, a_test, u_test)
    
    # Physics-informed example
    example_physics_informed_fno()
    
    print("\nAll examples completed!")
