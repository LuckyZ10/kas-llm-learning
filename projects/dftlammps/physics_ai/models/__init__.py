"""
Model implementations for Physics-AI module.
"""

from .pinns import (
    PhysicsInformedNN,
    SIREN,
    AdaptiveWeightPINN,
    burgers_pde,
    navier_stokes_pde,
    heat_equation_pde,
    schrodinger_pde,
    poisson_pde
)
from .deeponet import (
    DeepONet,
    SeparablePhysicsInformedDeepONet,
    MultiOutputDeepONet,
    AttentionDeepONet
)
from .fno import (
    FourierNeuralOperator,
    PhysicsInformedFNO,
    MultiScaleFNO,
    AdaptiveFNO
)
from .physics_gnn import (
    PhysicsInformedGNN,
    MomentumConservingGNN,
    HamiltonianGNN,
    EquivariantTransformer
)

__all__ = [
    # PINNs
    'PhysicsInformedNN',
    'SIREN',
    'AdaptiveWeightPINN',
    'burgers_pde',
    'navier_stokes_pde',
    'heat_equation_pde',
    'schrodinger_pde',
    'poisson_pde',
    
    # DeepONet
    'DeepONet',
    'SeparablePhysicsInformedDeepONet',
    'MultiOutputDeepONet',
    'AttentionDeepONet',
    
    # FNO
    'FourierNeuralOperator',
    'PhysicsInformedFNO',
    'MultiScaleFNO',
    'AdaptiveFNO',
    
    # GNN
    'PhysicsInformedGNN',
    'MomentumConservingGNN',
    'HamiltonianGNN',
    'EquivariantTransformer',
]
