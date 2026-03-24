# Universal Dynamics with Globally Controlled Analog Quantum Simulators

Code and data repository for the paper:

> **Universal Dynamics with Globally Controlled Analog Quantum Simulators**
> Hong-Ye Hu, Abigail McClain Gomez, Liyuan Chen, Aaron Trowbridge, Andy J. Goldschmidt, and collaborators
> [arXiv:2508.19075](https://arxiv.org/abs/2508.19075)

## Overview

This repository contains the quantum optimal control code and experimental data accompanying the paper. The work establishes a necessary and sufficient condition for universal quantum computation using only global pulse control on analog quantum simulators, and demonstrates the engineering of effective multi-body interactions (including three-body interactions) on a Rydberg-atom array via **direct quantum optimal control**.

Key results reproduced by this codebase:

- **GRAPE-based pulse optimization** for synthesizing target Hamiltonians (e.g., the ZXZ cluster-Ising model) from globally controlled Rydberg interactions.
- **Experimental pulse sequences** used to drive topological dynamics and observe symmetry-protected-topological (SPT) edge modes on a 5-atom Rydberg array.
- **Error analysis** of longer-duration pulse protocols under realistic hardware constraints.

## Repository Structure

```
.
├── GRAPE_code/                      # Quantum optimal control via GRAPE
│   ├── GRAPE_PiecewiseConstant_Script.py   # Main optimization script
│   └── GRAPE_HelperFunctions.py            # Helper utilities for GRAPE
│
├── Experimental_Pulses/             # Optimized pulse sequences for experiment
│   └── zero_state_pulses/           # Pulses starting from |00...0⟩
│       └── Dict_N5_theta{θ}_dist8.9_N8.json   # θ ∈ {0.1, 0.2, ..., 0.8}
│
├── Experimental_Data/               # Raw and processed experimental results
│   ├── N8_Rabi_{0,1}_N5_theta{θ}_dist8.9.json  # Shot data (Rabi index 0 or 1)
│   ├── All_Results_N5_dist8.9_N8.json           # Aggregated results
│   ├── All_Results_N5_dist8.9_N8_with_error.json
│   ├── All_Results_Triangle_N5_dist8.9_N8.json
│   └── correlations/               # Processed correlation measurements
│       ├── Processed_Results_N5_theta{θ}_dist8.9_N8.json
│       └── Processed_Results_Triangle_N5_theta{θ}_dist8.9_N8.json
│
├── Longer-duration-error-scan/      # Error analysis for extended pulse durations
│   └── long_time_pulse_data/
│       ├── Dict_N5_T73_theta0.8_dist8.9_N3_T{t}.json  # t ∈ {0.2, 0.4, ..., 3.0}
│       └── All_Results_N5_T73_theta0.8_dist8.9_N3.json
│
└── README.md
```

### `GRAPE_code/`

Implementation of the **GRadient Ascent Pulse Engineering (GRAPE)** algorithm for quantum optimal control. `GRAPE_PiecewiseConstant_Script.py` is the main entry point that optimizes piecewise-constant pulse sequences to realize target unitary dynamics under global control constraints. `GRAPE_HelperFunctions.py` provides supporting functions (Hamiltonian construction, fidelity computation, gradient evaluation, etc.).

### `Experimental_Pulses/`

Optimized pulse parameter files (JSON) used in the Rydberg-atom array experiment. Each file corresponds to a different mixing angle `θ` for the target ZXZ Hamiltonian evolution, with a system of `N = 5` atoms at an interatomic distance of `8.9 μm`, discretized into `N_steps = 8` piecewise-constant segments. The `zero_state_pulses/` subdirectory contains pulses designed for initial state preparation from the computational zero state.

### `Experimental_Data/`

Raw measurement outcomes from the Rydberg-atom experiment. Files are labeled by Rabi drive index (`Rabi_0` or `Rabi_1`) and mixing angle `θ`. The `correlations/` subdirectory contains post-processed correlation function data, including both standard and triangle (three-point) correlation measurements that reveal signatures of SPT edge modes.

### `Longer-duration-error-scan/`

Pulse optimization results exploring the effect of total evolution time on control fidelity. Data files sweep the evolution time parameter `T` from 0.2 to 3.0 μs for a fixed configuration (`N = 5` atoms, `θ = 0.8`, distance `8.9 μm`), enabling analysis of the trade-off between pulse duration and achievable fidelity under decoherence.

## File Naming Convention

The JSON data files follow a consistent naming scheme:

| Parameter | Meaning |
|-----------|---------|
| `N5` | Number of atoms (5) |
| `theta{θ}` | Mixing angle θ for the target Hamiltonian |
| `dist8.9` | Interatomic distance in μm |
| `N8` or `N3` | Number of piecewise-constant pulse segments |
| `T{t}` | Total evolution time in μs |
| `T73` | Truncation parameter for interaction range |
| `Rabi_{0,1}` | Rabi drive configuration index |

## Citation

If you use this code or data, please cite:

```
@ARTICLE{Universal_Dynamics,
       author = {{Hu}, Hong-Ye and {McClain Gomez}, Abigail and {Chen}, Liyuan and {Trowbridge}, Aaron and {Goldschmidt}, Andy J. and {Manchester}, Zachary and {Chong}, Frederic T. and {Jaffe}, Arthur and {Yelin}, Susanne F.},
        title = "{Universal Dynamics with Globally Controlled Analog Quantum Simulators}",
      journal = {arXiv e-prints},
     keywords = {Quantum Physics, Quantum Gases, Strongly Correlated Electrons, Machine Learning, Systems and Control},
         year = 2025,
        month = aug,
          eid = {arXiv:2508.19075},
        pages = {arXiv:2508.19075},
          doi = {10.48550/arXiv.2508.19075},
archivePrefix = {arXiv},
       eprint = {2508.19075},
 primaryClass = {quant-ph},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250819075H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Contact

For questions, please contact Hong-Ye Hu (hongyehu.physics@gmail.com).

# Direct Optimization
The direct method is partially open sourced in [https://github.com/harmoniqs/ZXZ-atoms/blob/main/ZXZ.ipynb]

