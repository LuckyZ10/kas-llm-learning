"""
Example: Physics-Informed Neural Network for Harmonic Oscillator

This example demonstrates how to use PINN to solve the 1D harmonic oscillator.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dftlammps.physics_ai.models import PhysicsInformedNN


def harmonic_oscillator_pde(x, u, derivatives, omega=1.0):
    """
    Harmonic oscillator: d²u/dt² + ω²u = 0
    
    Args:
        x: [t] - time coordinate
        u: [u] - displacement
        derivatives: dict with 'first' and 'second' derivatives
        omega: natural frequency
    """
    u_t = derivatives['first'][:, 0, 0]  # du/dt
    u_tt = derivatives['second'][:, 0, 0, 0]  # d²u/dt²
    
    # PDE residual
    residual = u_tt + omega**2 * u[:, 0]
    return residual


def generate_training_data(n_points=100, omega=1.0):
    """Generate training data for harmonic oscillator."""
    # Initial conditions: u(0) = 1, u'(0) = 0
    # Analytical solution: u(t) = cos(ωt)
    
    # Collocation points
    t_collocation = np.linspace(0, 2*np.pi, n_points).reshape(-1, 1)
    
    # Initial condition points
    t_ic = np.array([[0.0]])
    u_ic = np.array([[1.0]])  # u(0) = 1
    
    # Initial velocity condition
    t_ic2 = np.array([[0.0]])
    u_t_ic = np.array([[0.0]])  # u'(0) = 0
    
    return (
        torch.tensor(t_collocation, dtype=torch.float32),
        torch.tensor(t_ic, dtype=torch.float32),
        torch.tensor(u_ic, dtype=torch.float32),
        torch.tensor(t_ic2, dtype=torch.float32),
        torch.tensor(u_t_ic, dtype=torch.float32)
    )


def train_harmonic_oscillator():
    """Train PINN for harmonic oscillator."""
    
    # Create model
    model = PhysicsInformedNN(
        input_dim=1,
        output_dim=1,
        hidden_dims=[64, 64, 64],
        pde_fn=harmonic_oscillator_pde,
        activation='tanh'
    )
    
    # Generate data
    t_collocation, t_ic, u_ic, t_ic2, u_t_ic = generate_training_data()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    n_epochs = 5000
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Compute loss
        losses = model.compute_loss(
            x_ic=t_ic,
            u_ic=u_ic,
            x_collocation=t_collocation,
            lambda_pde=1.0,
            lambda_ic=100.0,  # Strong IC enforcement
            omega=1.0
        )
        
        loss = losses['total']
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.6f}, "
                  f"PDE={losses['pde'].item():.6f}, "
                  f"IC={losses['ic'].item():.6f}")
    
    return model


def plot_results(model):
    """Plot PINN solution vs analytical."""
    t_test = np.linspace(0, 2*np.pi, 200).reshape(-1, 1)
    t_test_tensor = torch.tensor(t_test, dtype=torch.float32)
    
    with torch.no_grad():
        u_pred = model(t_test_tensor).numpy()
    
    u_analytical = np.cos(t_test)  # Analytical solution
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_test, u_analytical, 'b-', label='Analytical: cos(t)', linewidth=2)
    plt.plot(t_test, u_pred, 'r--', label='PINN', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Displacement')
    plt.title('Harmonic Oscillator: PINN vs Analytical')
    plt.legend()
    plt.grid(True)
    plt.savefig('harmonic_oscillator_pinn.png', dpi=150)
    plt.close()
    
    # Compute error
    error = np.mean((u_pred - u_analytical)**2)
    print(f"\nMean Squared Error: {error:.6e}")


if __name__ == '__main__':
    print("Training Physics-Informed Neural Network for Harmonic Oscillator...")
    model = train_harmonic_oscillator()
    plot_results(model)
    print("\nDone! Check harmonic_oscillator_pinn.png")
