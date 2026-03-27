"""
    ZXZAtoms

Utilities for engineering effective ZXZ Hamiltonian dynamics on Rydberg atom chains
via direct trajectory optimization.

Provides trajectory construction, forward simulation (ODE rollout), and visualization
for quantum optimal control with Piccolo.jl.
"""
module ZXZAtoms

export unitary_rollout_trajectory
export unitary_trajectory
export plot_controls

include("trajectories.jl")
include("plotting.jl")

end # module
