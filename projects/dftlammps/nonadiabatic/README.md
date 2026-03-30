# Non-Adiabatic Dynamics Module

This module provides comprehensive tools for simulating excited state processes using non-adiabatic molecular dynamics (NAMD) methods.

## Overview

Non-adiabatic dynamics describes the coupled evolution of electronic and nuclear degrees of freedom in excited states. This module integrates multiple NAMD approaches for complete excited state treatment.

## Modules

### 1. PYXAID Interface (`pyxaid_interface.py`)

Interface to PYXAID for TD-DFT based non-adiabatic dynamics.

**Features:**
- VASP TD-DFT wave function extraction
- Non-adiabatic coupling (NAC) calculations
- Fewest Switches Surface Hopping (FSSH)
- Decoherence corrections (EDC, ID-A, AFSSH)
- Carrier lifetime analysis

**Key Classes:**
- `PYXAIDConfig`: Configuration for PYXAID simulations
- `PYXAIDWorkflow`: Complete simulation workflow
- `SurfaceHoppingDynamics`: Surface hopping engine
- `CarrierLifetimeAnalyzer`: Lifetime and recombination analysis
- `NonAdiabaticCouplingCalculator`: NAC calculation methods

### 2. SHARC Interface (`sharc_interface.py`)

Interface to SHARC for multi-reference non-adiabatic dynamics.

**Features:**
- CASSCF/MRCI/NEVPT2/CASPT2 support
- Spin-orbit coupling (SOC) calculations
- Intersystem crossing dynamics
- Multiplicity changing surface hops
- Laser-driven dynamics

**Key Classes:**
- `SHARCConfig`: Configuration for SHARC simulations
- `SHARCSurfaceHopping`: SHARC dynamics engine
- `MultiReferenceInterface`: Electronic structure interface
- `SpinOrbitMatrix`: SOC calculations
- `SHARCAnalyzer`: Trajectory analysis

### 3. Excited State Dynamics (`excited_state_dynamics.py`)

High-level excited state dynamics workflows.

**Features:**
- Exciton formation and dissociation
- Hot carrier relaxation
- Energy transfer networks
- Charge separation dynamics
- Comprehensive workflow orchestration

**Key Classes:**
- `ExcitedStateDynamicsWorkflow`: Master workflow
- `ExcitonDynamics`: Exciton simulation
- `CarrierDynamics`: Free carrier dynamics
- `EnergyTransferNetwork`: FRET/Dexter transfer
- `ExcitonState`, `CarrierState`: State representations

## Case Studies

### Photovoltaic Carrier Dynamics (`case_carrier_dynamics/`)

Perovskite solar cell simulation:
- Hot carrier cooling
- Exciton dissociation
- Charge transport
- Device efficiency analysis

### Exciton Dissociation (`case_exciton_dissociation/`)

Organic solar cell interface:
- Exciton diffusion
- Charge transfer state formation
- Geminate recombination
- Energy transfer cascades

### Photocatalysis (`case_photocatalysis/`)

Water splitting on TiO2:
- Band edge alignment
- Charge separation efficiency
- Catalytic reaction cycle
- Solar-to-hydrogen efficiency

## Usage Examples

### Basic Surface Hopping

```python
from dftlammps.nonadiabatic import PYXAIDConfig, PYXAIDWorkflow

# Configure simulation
config = PYXAIDConfig(
    nstates=5,
    dt=0.5,  # fs
    nsteps=1000,
    hopping_method='fssh',
    decoherence_method='edc'
)

# Create workflow
workflow = PYXAIDWorkflow(config)

# Run dynamics
trajectory = workflow.run_dynamics_simulation(
    structure='POSCAR',
    initial_state=1,
    temperature=300.0
)

# Analyze results
results = workflow.analyze_results()
```

### Exciton Dynamics

```python
from dftlammps.nonadiabatic import ExcitedStateDynamicsWorkflow

# Create workflow
workflow = ExcitedStateDynamicsWorkflow()

# Setup exciton system
workflow.setup_exciton_system(
    exciton_energies=[2.0, 2.2],
    binding_energies=[0.4, 0.3],
    radii=[1.5, 1.2]
)

# Run dissociation simulation
results = workflow.run_exciton_dissociation(
    initial_exciton_idx=0,
    electric_field=0.1  # V/nm
)
```

### SHARC Multi-Reference Dynamics

```python
from dftlammps.nonadiabatic import SHARCConfig, SHARCSurfaceHopping, MultiReferenceMethod

# Configure for CASSCF
config = SHARCConfig(
    method=MultiReferenceMethod.CASSCF,
    nstates=6,
    spin_multiplicities=[1, 3],  # Singlets and triplets
    include_soc=True
)

# Initialize dynamics
sh = SHARCSurfaceHopping(config)
sh.initialize(nstates=6, initial_state=1)
```

## Theory Background

### Surface Hopping

Surface hopping methods solve the time-dependent Schrödinger equation for electrons simultaneously with classical equations of motion for nuclei. The fewest switches surface hopping (FSSH) algorithm minimizes the number of hops while maintaining correct state populations.

The hopping probability from state $j$ to $k$ is:

$$g_{jk} = -2 \frac{\text{Re}(c_k^* c_j d_{jk})}{|c_j|^2} \Delta t$$

where $c_j$ are electronic amplitudes and $d_{jk}$ is the non-adiabatic coupling.

### Non-Adiabatic Coupling

NAC represents the coupling between electronic states due to nuclear motion:

$$\mathbf{d}_{jk} = \langle \psi_j | \nabla_R | \psi_k \rangle \cdot \dot{\mathbf{R}}$$

Methods for NAC calculation:
- Finite difference approximation
- Wave function overlap method
- Analytic derivative methods

### Spin-Orbit Coupling

SOC enables transitions between states of different spin multiplicity. The Breit-Pauli Hamiltonian:

$$\hat{H}_{SO} = \frac{e^2}{2m_e^2c^2} \sum_i \sum_\alpha \frac{Z_\alpha}{r_{i\alpha}^3} \mathbf{L}_i \cdot \mathbf{S}_i$$

## References

1. Tully, J. C. "Molecular Dynamics with Electronic Transitions" J. Chem. Phys. 93, 1061 (1990)
2. Akimov, A. V.; Prezhdo, O. V. "The PYXAID Program" J. Chem. Theory Comput. 9, 4959 (2013)
3. Mai, S. et al. "SHARC" Chem. Sci. 8, 6819 (2018)
4. Nelson, T. et al. "Nonadiabatic Excited-State Molecular Dynamics" J. Chem. Phys. 152, 204105 (2020)

## Installation Requirements

Required:
- Python >= 3.7
- NumPy

Optional:
- SciPy (for advanced analysis)
- Matplotlib (for visualization)
- ASE (for structure handling)

## License

MIT License
