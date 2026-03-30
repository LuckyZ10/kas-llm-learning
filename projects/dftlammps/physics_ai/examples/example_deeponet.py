"""
Example: DeepONet for Learning PDE Solutions

This example demonstrates how to use DeepONet to learn the solution
operator for a parametric ODE/PDE.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dftlammps.physics_ai.models import DeepONet, MultiOutputDeepONet


def generate_training_data(n_sensors=100, n_functions=1000, n_points=50):
    """
    Generate training data for learning ODE solution operator.
    
    ODE: du/dt = f(t), u(0) = 0
    Solution: u(t) = integral_0^t f(s) ds
    
    Returns:
        u: Input function values at sensor points [n_functions, n_sensors]
        y: Evaluation coordinates [n_points, 1]
        G_u: Solution values [n_functions, n_points]
    """
    # Sensor locations
    t_sensors = np.linspace(0, 1, n_sensors)
    
    # Generate random functions (using Gaussian processes)
    def generate_function(t, n_samples):
        """Generate smooth random functions."""
        # Covariance matrix (RBF kernel)
        length_scale = 0.2
        K = np.exp(-0.5 * ((t[:, None] - t[None, :]) / length_scale) ** 2)
        
        # Sample from GP
        L = np.linalg.cholesky(K + 1e-10 * np.eye(len(t)))
        samples = L @ np.random.randn(len(t), n_samples)
        return samples.T
    
    # Generate input functions
    u_samples = generate_function(t_sensors, n_functions)
    
    # Evaluation points
    y = np.linspace(0, 1, n_points).reshape(-1, 1)
    
    # Compute solutions (cumulative integral)
    G_u = np.zeros((n_functions, n_points))
    for i in range(n_functions):
        # Numerical integration (trapezoidal rule)
        for j in range(n_points):
            t_eval = y[j, 0]
            # Interpolate function
            f_interp = np.interp(t_sensors * t_eval, t_sensors, u_samples[i])
            # Integrate
            G_u[i, j] = np.trapz(f_interp, t_sensors * t_eval)
    
    return (
        torch.tensor(u_samples, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(G_u, dtype=torch.float32)
    )


def train_deeponet():
    """Train DeepONet for ODE solution operator."""
    
    print("Generating training data...")
    u_train, y_train, G_u_train = generate_training_data(
        n_sensors=100, n_functions=2000, n_points=50
    )
    
    u_test, y_test, G_u_test = generate_training_data(
        n_sensors=100, n_functions=200, n_points=50
    )
    
    print("Creating DeepONet model...")
    model = DeepONet(
        branch_input_dim=100,
        trunk_input_dim=1,
        output_dim=1,
        branch_hidden_dims=[128, 128, 128],
        trunk_hidden_dims=[128, 128, 128],
        activation='relu'
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    
    # Training loop
    n_epochs = 5000
    batch_size = 100
    
    print("Training...")
    for epoch in range(n_epochs):
        # Sample batch
        idx = np.random.choice(len(u_train), batch_size, replace=False)
        u_batch = u_train[idx]
        G_u_batch = G_u_train[idx]
        
        optimizer.zero_grad()
        
        # Forward pass
        # For each function in batch, evaluate at all points
        G_u_pred = model(u_batch, y_train)  # [batch_size, n_points, 1]
        G_u_pred = G_u_pred.squeeze(-1)  # [batch_size, n_points]
        
        # Loss
        loss = torch.mean((G_u_pred - G_u_batch) ** 2)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 500 == 0:
            # Validation
            with torch.no_grad():
                G_u_val_pred = model(u_test, y_test).squeeze(-1)
                val_loss = torch.mean((G_u_val_pred - G_u_test) ** 2)
            
            print(f"Epoch {epoch}: Train Loss={loss.item():.6f}, "
                  f"Val Loss={val_loss.item():.6f}")
    
    return model, u_test, y_test, G_u_test


def evaluate_deeponet(model, u_test, y_test, G_u_test):
    """Evaluate DeepONet performance."""
    
    with torch.no_grad():
        G_u_pred = model(u_test, y_test).squeeze(-1).numpy()
    
    G_u_true = G_u_test.numpy()
    
    # Compute errors
    mse = np.mean((G_u_pred - G_u_true) ** 2)
    mae = np.mean(np.abs(G_u_pred - G_u_true))
    
    print(f"\nTest Results:")
    print(f"  MSE: {mse:.6e}")
    print(f"  MAE: {mae:.6e}")
    
    # Plot a few examples
    n_examples = 3
    fig, axes = plt.subplots(1, n_examples, figsize=(15, 4))
    
    y_plot = y_test.numpy().flatten()
    
    for i in range(n_examples):
        ax = axes[i]
        ax.plot(y_plot, G_u_true[i], 'b-', label='True', linewidth=2)
        ax.plot(y_plot, G_u_pred[i], 'r--', label='Predicted', linewidth=2)
        ax.set_xlabel('t')
        ax.set_ylabel('u(t)')
        ax.set_title(f'Example {i+1}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('deeponet_integration.png', dpi=150)
    plt.close()
    
    print("\nPlot saved to deeponet_integration.png")


def example_multoutput_deeponet():
    """Example with multi-output DeepONet."""
    print("\n" + "=" * 60)
    print("Multi-Output DeepONet Example")
    print("=" * 60)
    
    # Generate data for 2D output
    n_sensors = 50
    n_functions = 500
    
    # Input functions
    t = np.linspace(0, 1, n_sensors)
    u = np.array([np.sin(2 * np.pi * t * (i + 1)) for i in range(n_functions)])
    u = torch.tensor(u, dtype=torch.float32)
    
    # Evaluation points (2D)
    y = torch.rand(100, 2)  # Random points in [0,1]x[0,1]
    
    # Output: vector field [f1, f2]
    G_u = torch.zeros(n_functions, 100, 2)
    for i in range(n_functions):
        G_u[i, :, 0] = torch.sin(2 * np.pi * y[:, 0]) * torch.cos(2 * np.pi * y[:, 1])
        G_u[i, :, 1] = torch.cos(2 * np.pi * y[:, 0]) * torch.sin(2 * np.pi * y[:, 1])
    
    # Create model
    model = MultiOutputDeepONet(
        branch_input_dim=n_sensors,
        trunk_input_dim=2,
        output_dim=2,
        branch_hidden_dims=[64, 64],
        trunk_hidden_dims=[64, 64]
    )
    
    print(f"Model created:")
    print(f"  Branch input: {n_sensors}")
    print(f"  Trunk input: 2")
    print(f"  Output: 2 (vector field)")
    
    return model


if __name__ == '__main__':
    print("DeepONet Examples")
    print("=" * 60)
    
    # Main example
    model, u_test, y_test, G_u_test = train_deeponet()
    evaluate_deeponet(model, u_test, y_test, G_u_test)
    
    # Multi-output example
    example_multoutput_deeponet()
    
    print("\nAll examples completed!")
