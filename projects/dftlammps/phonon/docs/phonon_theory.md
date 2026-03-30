# Phonon Theory and Computational Methods

## Table of Contents
1. [Introduction to Lattice Dynamics](#introduction)
2. [Harmonic Approximation](#harmonic-approximation)
3. [Force Constants and Dynamical Matrix](#force-constants)
4. [Phonon Dispersion Relations](#dispersion-relations)
5. [Density of States](#density-of-states)
6. [Thermodynamic Properties](#thermodynamic-properties)
7. [Anharmonic Effects](#anharmonic-effects)
8. [Quasi-Harmonic Approximation](#quasi-harmonic)
9. [Electron-Phonon Coupling](#electron-phonon)
10. [Computational Workflow](#workflow)
11. [Convergence Tests](#convergence)

---

## 1. Introduction to Lattice Dynamics <a name="introduction"></a>

Phonons are the quantized modes of vibration occurring in a rigid crystal lattice. They play a crucial role in determining many physical properties of materials, including thermal conductivity, specific heat, electrical resistivity, and superconducting properties.

### Key Concepts

- **Harmonic Oscillator Model**: Atoms vibrate around their equilibrium positions
- **Collective Excitations**: Phonons are collective motions of the crystal lattice
- **Bosonic Quasiparticles**: Phonons obey Bose-Einstein statistics
- **Dispersion Relations**: ω(q) describes how phonon frequencies depend on wave vector

### Importance in Materials Science

1. **Thermal Properties**: Heat capacity, thermal expansion, thermal conductivity
2. **Electronic Properties**: Electron-phonon scattering, superconductivity
3. **Mechanical Properties**: Elastic constants, stability
4. **Phase Transitions**: Soft modes, structural transitions

---

## 2. Harmonic Approximation <a name="harmonic-approximation"></a>

The harmonic approximation treats atomic vibrations as a collection of independent harmonic oscillators. This is the foundation of lattice dynamics.

### Potential Energy Expansion

The potential energy of the crystal is expanded to second order in atomic displacements:

```
V = V₀ + ½ Σ_{iα,jβ} Φ_{iα,jβ} u_{iα} u_{jβ}
```

where:
- V₀ is the equilibrium potential energy
- u_{iα} is the displacement of atom i in direction α
- Φ_{iα,jβ} = ∂²V/∂u_{iα}∂u_{jβ}|₀ are the force constants

### Equations of Motion

The classical equations of motion are:

```
M_i ü_{iα} = -∂V/∂u_{iα} = -Σ_{jβ} Φ_{iα,jβ} u_{jβ}
```

### Periodic Solutions (Bloch's Theorem)

Due to translational symmetry, solutions have the form:

```
u_{iα}(t) = (1/√M_i) ε_{α}(q) exp[i(q·R_i - ωt)]
```

where:
- q is the wave vector in the first Brillouin zone
- ε(q) is the polarization vector
- ω is the angular frequency

---

## 3. Force Constants and Dynamical Matrix <a name="force-constants"></a>

### Force Constant Matrix

The force constant matrix Φ describes how the force on atom i depends on the displacement of atom j:

```
Φ_{iα,jβ} = ∂²V/∂u_{iα}∂u_{jβ}|₀
```

Properties:
- **Translation Invariance**: Σ_j Φ_{iα,jβ} = 0
- **Rotation Invariance**: Specific constraints on elements
- **Crystal Symmetry**: Reduces number of independent elements

### Dynamical Matrix

The dynamical matrix D(q) is the Fourier transform of the force constant matrix:

```
D_{αβ}(q) = Σ_j Φ_{0α,jβ} exp[iq·(R_j - R_0)] / √(M_0 M_j)
```

### Eigenvalue Problem

The phonon frequencies and eigenvectors are obtained by diagonalizing D(q):

```
Σ_β D_{αβ}(q) ε_β(q) = ω²(q) ε_α(q)
```

This yields 3N solutions for each q-point (N = number of atoms in primitive cell):
- 3 acoustic modes (ω → 0 as q → 0)
- 3N-3 optical modes

---

## 4. Phonon Dispersion Relations <a name="dispersion-relations"></a>

### Acoustic Modes

In the long-wavelength limit (q → 0):
- Frequencies: ω_a(q) = v_s |q| (linear dispersion)
- v_s is the sound velocity
- 3 branches: one longitudinal (LA), two transverse (TA)

### Optical Modes

At q = 0 (Γ point):
- Finite frequencies for optical modes
- Atoms vibrate out of phase
- Can be Raman or IR active

### Special Points

Common high-symmetry points in reciprocal space:
- Γ = (0, 0, 0)
- X = (0.5, 0, 0.5) for FCC
- L = (0.5, 0.5, 0.5) for FCC
- K = (0.375, 0.375, 0.75) for hexagonal

---

## 5. Density of States <a name="density-of-states"></a>

### Definition

The phonon density of states (DOS) counts the number of modes per unit frequency:

```
g(ω) = Σ_{q,s} δ(ω - ω_s(q))
```

### Debye Model

Simple model assuming linear dispersion up to cutoff frequency ω_D:

```
g_D(ω) = 3ω²/ω_D³ for ω ≤ ω_D
```

### Van Hove Singularities

Critical points where ∇_q ω(q) = 0 lead to singularities in g(ω).

### Projected DOS (PDOS)

Contribution of specific atoms or directions to the total DOS:

```
g_i(ω) = Σ_{q,s} |ε_{i,s}(q)|² δ(ω - ω_s(q))
```

---

## 6. Thermodynamic Properties <a name="thermodynamic-properties"></a>

### Partition Function

For a system of independent phonon modes:

```
Z = Π_{q,s} [2 sinh(ℏω_s(q)/2k_BT)]⁻¹
```

### Internal Energy

```
U = Σ_{q,s} ℏω_s(q) [n_B(ω_s(q)) + ½]
```

where n_B(ω) = 1/[exp(ℏω/k_BT) - 1] is the Bose-Einstein distribution.

### Free Energy

```
F = k_BT Σ_{q,s} ln[2 sinh(ℏω_s(q)/2k_BT)]
```

### Entropy

```
S = k_B Σ_{q,s} [(n_B + 1)ln(n_B + 1) - n_B ln(n_B)]
```

### Heat Capacity

At constant volume:

```
C_V = (∂U/∂T)_V = k_B Σ_{q,s} (ℏω_s/k_BT)² exp(ℏω_s/k_BT) / [exp(ℏω_s/k_BT) - 1]²
```

Low-temperature limit (T → 0): C_V ∝ T³ (Debye law)
High-temperature limit (T → ∞): C_V → 3Nk_B (Dulong-Petit law)

---

## 7. Anharmonic Effects <a name="anharmonic-effects"></a>

### Beyond Harmonic Approximation

The harmonic approximation neglects phonon-phonon interactions. Higher-order terms in the potential expansion lead to:

- Thermal expansion
- Finite phonon lifetimes
- Thermal resistivity
- Frequency shifts with temperature

### Third-Order Force Constants

The cubic anharmonic term:

```
V^(3) = (1/6) Σ_{iα,jβ,kγ} Φ_{iα,jβ,kγ} u_{iα} u_{jβ} u_{kγ}
```

### Phonon-Phonon Scattering

Three-phonon processes:
- **Type I**: ω₁ + ω₂ = ω₃ (decay)
- **Type II**: ω₁ = ω₂ + ω₃ (coalescence)

Selection rules from energy and momentum conservation.

### Phonon Lifetime

The inverse lifetime (scattering rate) from three-phonon processes:

```
Γ_s(q) ∝ Σ |V^(3)|² δ(ω_s(q) ± ω_s'(q') - ω_s''(q±q')) [n_B(ω') - n_B(ω'')]
```

---

## 8. Quasi-Harmonic Approximation <a name="quasi-harmonic"></a>

### Volume Dependence

The quasi-harmonic approximation (QHA) accounts for thermal expansion by calculating phonon frequencies at different volumes and minimizing the free energy:

```
F(V,T) = E₀(V) + F_vib(V,T)
```

### Thermal Expansion

The equilibrium volume at temperature T minimizes F(V,T):

```
∂F/∂V|_{V=V_eq(T)} = 0
```

The thermal expansion coefficient:

```
α_V = (1/V)(∂V/∂T)_P
```

### Grüneisen Parameter

Mode-dependent Grüneisen parameter:

```
γ_s(q) = -V/ω_s(q) · ∂ω_s(q)/∂V = -∂lnω_s(q)/∂lnV
```

Average Grüneisen parameter:

```
γ = Σ_{q,s} γ_s(q) C_{V,s}(q) / C_V
```

---

## 9. Electron-Phonon Coupling <a name="electron-phonon"></a>

### Interaction Hamiltonian

The electron-phonon interaction:

```
H_ep = Σ_{kq,s} g_{kq,s} c†_{k+q} c_k (a_{q,s} + a†_{-q,s})
```

where g_{kq,s} is the electron-phonon matrix element.

### Eliashberg Function

```
α²F(ω) = N(0)⁻¹ Σ_{k,q,s} |g_{kq,s}|² δ(ε_k) δ(ε_{k+q}) δ(ω - ω_{q,s})
```

### Coupling Constant

```
λ = 2∫₀^∞ dω α²F(ω)/ω
```

### Critical Temperature (McMillan)

```
T_c = (ℏω_log / 1.2k_B) exp[-1.04(1 + λ) / (λ - μ*(1 + 0.62λ))]
```

where:
- ω_log is the logarithmic average phonon frequency
- μ* is the Coulomb pseudopotential

---

## 10. Computational Workflow <a name="workflow"></a>

### Step 1: Structure Optimization

1. Optimize unit cell geometry
2. Ensure forces < threshold (e.g., 10⁻⁵ eV/Å)
3. Converge total energy

### Step 2: Supercell Setup

1. Choose supercell size (typically 2×2×2 or larger)
2. Check convergence with respect to supercell size

### Step 3: Displacement Generation

1. Generate atomic displacements (typically ±0.01 Å)
2. Use symmetry to reduce number of displacements
3. Create displaced supercell structures

### Step 4: Force Calculations

1. Run DFT calculations for each displacement
2. Extract forces on all atoms
3. Check force convergence and numerical accuracy

### Step 5: Force Constants

1. Calculate force constant matrix from forces
2. Check acoustic sum rules
3. Apply symmetrization if needed

### Step 6: Phonon Calculation

1. Calculate phonon frequencies on q-point mesh
2. Compute DOS and PDOS
3. Generate band structure along high-symmetry paths
4. Check for imaginary frequencies (indicate instability)

### Step 7: Thermodynamic Properties

1. Calculate free energy, entropy, heat capacity
2. Run QHA for thermal expansion if needed
3. Compare with experimental data

### Step 8: Transport Properties (Optional)

1. Calculate third-order force constants (Phono3py)
2. Compute lattice thermal conductivity
3. Analyze phonon lifetimes and mean free paths

---

## 11. Convergence Tests <a name="convergence"></a>

### Critical Parameters

1. **Plane Wave Cutoff**: Converge total energy and forces
2. **k-point Grid**: Converge electronic structure
3. **Supercell Size**: Converge force constants
4. **q-point Mesh**: Converge DOS and thermodynamic properties
5. **Displacement Size**: Balance accuracy and numerical stability

### Typical Convergence Criteria

| Parameter | Criterion |
|-----------|-----------|
| Total Energy | < 1 meV/atom |
| Forces | < 10⁻⁵ eV/Å |
| Phonon Frequencies | < 0.1 THz |
| Heat Capacity | < 1% |
| Free Energy | < 1 meV/atom |

### Mesh Convergence

For phonon DOS calculations:
- Start with coarse mesh (e.g., 10×10×10)
- Increase until frequencies converge
- Typical converged meshes: 20×20×20 to 40×40×40

For thermal conductivity:
- Require finer meshes (typically odd numbers: 11×11×11, 15×15×15)
- Check convergence with respect to mesh density

---

## References

1. Born, M., & Huang, K. (1954). Dynamical Theory of Crystal Lattices.
2. Wallace, D. C. (1998). Thermodynamics of Crystals.
3. Grimvall, G. (1981). The Electron-Phonon Interaction in Metals.
4. Togo, A., & Tanaka, I. (2015). Phonopy documentation.
5. Allen, P. B., & Mitrović, B. (1982). Theory of Superconducting Tc.

---

## Appendix: Units and Conversions

| Quantity | Common Units | Conversion |
|----------|--------------|------------|
| Frequency | THz | 1 THz = 33.36 cm⁻¹ |
| Energy | meV | 1 meV = 0.2418 THz |
| Temperature | K | k_BT (300K) = 25.85 meV |
| Force Constant | eV/Å² | |
| Thermal Conductivity | W/m/K | |

---

*Document version: 1.0*
*Last updated: 2026-03-09*
