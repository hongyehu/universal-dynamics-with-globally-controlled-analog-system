# Direct Quantum Optimal Control for ZXZ Hamiltonian Engineering

This directory implements the **direct trajectory optimization** approach for engineering effective three-body ZXZ Hamiltonian dynamics on Rydberg atom chains, as described in [arXiv:2508.19075](https://arxiv.org/abs/2508.19075).

## Overview

The goal is to find globally-controlled Rydberg pulse sequences $\{\Omega(t), \Delta(t)\}$ that realize the target unitary evolution

$$U_{\text{goal}} = \exp(-i\theta\, H_{\text{ZXZ}}), \qquad H_{\text{ZXZ}} = \sum_j Z_{j-1}\, X_j\, Z_{j+1}$$

using only the native two-body Rydberg Hamiltonian (Equation 19):

$$\frac{H(t)}{\hbar} = \frac{\Omega(t)}{2} \sum_i \sigma_i^x - \Delta(t) \sum_i n_i + \sum_{i<j} \frac{C_6}{|r_i - r_j|^6}\, n_i n_j$$

This is a fundamentally challenging problem: the target contains **three-body** interactions that do not appear in the native Hamiltonian. The optimizer must discover pulse sequences that effectively generate these interactions through the interplay of drives and native two-body physics.

## Method: Direct Trajectory Optimization

Unlike GRAPE (gradient ascent on piecewise-constant controls), the direct method treats the full quantum trajectory — unitaries **and** controls at every time step — as optimization variables, subject to the Schrödinger equation enforced as a constraint via direct collocation. This approach:

- **Explores unphysical trajectories** during optimization, finding better solutions
- **Handles hardware constraints** (amplitude bounds, smoothness) as first-class inequality constraints
- **Converges with fewer iterations** by leveraging exact constraint Jacobians and (optionally) Hessians

The implementation uses [Piccolo.jl](https://github.com/harmoniqs/Piccolo.jl), a quantum optimal control framework built on direct trajectory optimization.

## Quick Start

### Prerequisites

- **Julia ≥ 1.10** ([julialang.org](https://julialang.org/downloads/))

### Installation

```bash
cd direct_optimization
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Run

```bash
julia --project=. ZXZ.jl
```

This will:
1. Set up the 3-atom Rydberg system with parameters matching the paper
2. Run a two-phase optimization (L-BFGS → exact Hessian, 1000 total iterations)
3. Validate the result with an independent ODE forward simulation
4. Save control pulse and population plots to PNG files

**Expected runtime**: ~2–5 minutes on a modern laptop.

### Extending to More Atoms

Change `N_atoms` in `ZXZ.jl`:

```julia
N_atoms = 5  # 5-atom chain with 3 ZXZ terms
```

The Hilbert space dimension scales as $2^N$, so $N = 5$ ($\dim = 32$) is still tractable; $N \geq 7$ requires more time and memory.

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_atoms` | 3 | Number of atoms in the chain |
| `dist` | 8.9 μm | Interatomic spacing (outside blockade radius) |
| `θ` | 0.8 | Evolution parameter for $U = \exp(-i\theta H_{\text{ZXZ}})$ |
| `Δt` | 0.05 μs | Time step |
| `T_samples` | 26 | Number of time samples (total time = 1.25 μs) |
| `Ω_max` | 15.7 MHz | Maximum Rabi frequency |
| `Δ_max` | 100.0 MHz | Maximum detuning |
| `Q` | 10⁴ | Fidelity objective weight |
| `R_u` | 1.0 | Control regularization weight |

## File Structure

```
direct_optimization/
├── ZXZ.jl              # Main optimization script
├── src/
│   ├── ZXZAtoms.jl     # Module interface
│   ├── trajectories.jl # ODE rollout and trajectory construction
│   └── plotting.jl     # Control pulse visualization
├── Project.toml        # Julia dependencies
└── README.md           # This file
```

## Dependencies

- [Piccolo.jl](https://github.com/harmoniqs/Piccolo.jl) — quantum optimal control via direct trajectory optimization
- [PiccoloQuantumObjects.jl](https://github.com/harmoniqs/PiccoloQuantumObjects.jl) — quantum state and operator representations
- [NamedTrajectories.jl](https://github.com/harmoniqs/NamedTrajectories.jl) — structured trajectory data
- [CairoMakie.jl](https://github.com/MakieOrg/Makie.jl) — publication-quality plotting
- [DataInterpolations.jl](https://github.com/SciML/DataInterpolations.jl) — control function interpolation

## Citation

```bibtex
@article{hu2025universal,
  title   = {Universal Dynamics with Globally Controlled Analog Quantum Simulators},
  author  = {Hu, Hong-Ye and McClain Gomez, Abigail and Chen, Liyuan and Trowbridge, Aaron and Goldschmidt, Andy J. and Manchester, Zachary and Chong, Frederic T. and Jaffe, Arthur and Yelin, Susanne F.},
  journal = {arXiv preprint arXiv:2508.19075},
  year    = {2025}
}
```
