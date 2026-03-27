# # Engineering Three-Body ZXZ Interactions via Direct Quantum Optimal Control
#
# This script demonstrates the synthesis of effective three-body ZXZ Hamiltonian
# dynamics on a globally-controlled Rydberg atom chain using **direct trajectory
# optimization** with [Piccolo.jl](https://github.com/harmoniqs/Piccolo.jl).
#
# ## Physical Setup
#
# **Target Hamiltonian** — the cluster-Ising model with three-body interactions:
#
# ```math
# H_{\text{ZXZ}} = J_{\text{eff}} \sum_j Z_{j-1}\, X_j\, Z_{j+1}
# ```
#
# **Native Rydberg Hamiltonian** (Equation 19 of the paper):
#
# ```math
# \frac{H(t)}{\hbar} = \frac{\Omega(t)}{2} \sum_i \sigma_i^x
#   - \Delta(t) \sum_i n_i
#   + \sum_{i < j} \frac{C_6}{|r_i - r_j|^6}\, n_i n_j
# ```
#
# where $\Omega(t)$ is the global Rabi frequency, $\Delta(t)$ is the global
# detuning, $n_i = |r\rangle\langle r|_i$ is the Rydberg number operator, and
# $C_6/r^6$ encodes the van der Waals interaction.
#
# **Goal**: Find control pulses $\{\Omega(t), \Delta(t)\}$ such that the
# time-ordered evolution under the native Hamiltonian approximates the target
# unitary $U_{\text{goal}} = \exp(-i\theta\, H_{\text{ZXZ}})$.
#
# ## Running this script
#
# ```bash
# cd direct_optimization
# julia --project=. -e 'using Pkg; Pkg.instantiate()'
# julia --project=. ZXZ.jl
# ```

# ## 1. Setup

using Piccolo
using DataInterpolations
using LinearAlgebra
using SparseArrays
using CairoMakie

include("src/ZXZAtoms.jl")
using .ZXZAtoms

# ## 2. Physical Parameters
#
# These match the experimental configuration in the paper:
# - $N$ atoms in a 1D chain with spacing $d = 8.9\;\mu\text{m}$
#   (outside the blockade radius $r_b \approx 8.37\;\mu\text{m}$)
# - $C_6 = 862{,}690 \times 2\pi$ (van der Waals coefficient for $^{87}$Rb)
# - Global controls: Rabi frequency $\Omega(t) \in [0, 15.7]$ MHz,
#   detuning $\Delta(t) \in [-100, 100]$ MHz

N_atoms = 3            # Number of atoms (extensible: set to 4 or 5)
dist    = 8.9          # Interatomic spacing [μm]
θ       = 0.8          # Evolution parameter (mixing angle)
Δt      = 0.05         # Time step [μs]
T_samples = 26         # Number of time samples (total time = (T-1)·Δt = 1.25 μs)

Ω_max = 15.7           # Maximum Rabi frequency [MHz]
Δ_max = 100.0          # Maximum detuning [MHz]

println("=" ^ 60)
println("ZXZ Hamiltonian Engineering — Direct Quantum Optimal Control")
println("=" ^ 60)
println("  Atoms:    N = $N_atoms")
println("  Spacing:  d = $dist μm")
println("  Angle:    θ = $θ")
println("  Duration: T = $((T_samples - 1) * Δt) μs  ($T_samples samples, Δt = $Δt μs)")
println("  Ω_max:    $Ω_max MHz")
println("  Δ_max:    $Δ_max MHz")
println("=" ^ 60)

# ## 3. Target Unitary
#
# Construct the ZXZ Hamiltonian using Pauli string operators. For $N$ atoms,
# the nearest-neighbour three-body terms are
# $\sum_{j=1}^{N-2} Z_j X_{j+1} Z_{j+2}$.

if N_atoms == 3
    H_eff = operator_from_string("ZXZ")
elseif N_atoms == 4
    H_eff = operator_from_string("ZXZI") + operator_from_string("IZXZ")
elseif N_atoms == 5
    H_eff = (
        operator_from_string("ZXZII") +
        operator_from_string("IZXZI") +
        operator_from_string("IIZXZ")
    )
else
    error("Set N_atoms ∈ {3, 4, 5}. Larger systems require more time and memory.")
end

U_goal = sparse(exp(-im * θ * H_eff))

println("\nTarget: U_goal = exp(-iθ H_ZXZ),  θ = $θ")
println("Hilbert space dimension: $(2^N_atoms)")

# ## 4. Rydberg System
#
# Build the native Rydberg Hamiltonian using Piccolo's `RydbergChainSystem`.
# With `ignore_Y_drive=true`, the two control channels are $\Omega(t)$ and
# $\Delta(t)$ — matching the experimental setup where only a single laser
# beam with tunable intensity and frequency is available.

sys = RydbergChainSystem(N=N_atoms, distance=dist, ignore_Y_drive=true)

a_bounds = (
    [0.0,    -Δ_max],   # Lower bounds: Ω ≥ 0, Δ ≥ -Δ_max
    [Ω_max,   Δ_max],   # Upper bounds: Ω ≤ Ω_max, Δ ≤ Δ_max
)

# ## 5. Initial Trajectory
#
# Create a random initial trajectory by:
# 1. Sampling random control values at each time point
# 2. Forward-integrating the Schrödinger equation via ODE (Tsit5)
# 3. Packaging into a `NamedTrajectory` for optimization

u_init = vcat(rand(1, T_samples), 2.0 .* rand(1, T_samples) .- 1.0)
times = [(k - 1) * Δt for k in 1:T_samples]

Id = sparse(I(size(U_goal, 1)))
G(a, t) = sys.G(a, t)
Ĝ(a, t) = kron(Id, sys.G(a, t))

u_fn = t -> begin
    [LinearInterpolation(u_init[j, :], times)(t) for j in 1:2]
end

traj = unitary_rollout_trajectory(
    u_fn, G, times[end];
    samples=T_samples,
    control_bounds=a_bounds,
    Δt_min=Δt,
    Δt_max=2.0Δt,
)
update_bound!(traj, :u, a_bounds)

println("\nInitial fidelity: ",
    unitary_fidelity(iso_vec_to_operator(traj[end].Ũ⃗), U_goal))

# ## 6. Optimization Problem
#
# The direct trajectory optimization minimizes:
#
# ```math
# \mathcal{L} = Q \cdot \left(1 - \frac{|\mathrm{tr}(U_{\text{goal}}^\dagger\, U_{\text{final}})|^2}{d^2}\right)
#   + R_u \sum_t \|u(t)\|^2
# ```
#
# subject to the dynamics constraint $\dot{\tilde{U}}(t) = (I \otimes G(u(t)))\, \tilde{U}(t)$,
# enforced via direct collocation with linear-spline interpolation of controls.

Q   = 1.0e4    # Fidelity weight
R_u = 1.0      # Control regularization

J = UnitaryInfidelityObjective(U_goal, :Ũ⃗, traj; Q=Q)
J += QuadraticRegularizer(:u, traj, R_u)

integrators = [
    TimeDependentBilinearIntegrator(Ĝ, :Ũ⃗, :u, :t, traj; spline_order=1)
]

prob = DirectTrajOptProblem(traj, J, integrators)
println("\n", prob)

# ## 7. Solve — Phase 1: L-BFGS (no Hessian)
#
# Fast basin-finding with L-BFGS approximation to the Hessian.

println("\n--- Phase 1: L-BFGS (500 iterations) ---")
DirectTrajOpt.solve!(prob;
    max_iter=500,
    options=IpoptOptions(
        recalc_y="yes",
        recalc_y_feas_tol=1e8,
        eval_hessian=false,
    )
)

fid_phase1 = unitary_fidelity(
    iso_vec_to_operator(prob.trajectory[end].Ũ⃗), U_goal
)
println("Phase 1 fidelity: $fid_phase1")

# ## 8. Solve — Phase 2: Exact Hessian
#
# Polish the solution using IPOPT with exact second-order information.

println("\n--- Phase 2: Exact Hessian (500 iterations) ---")
DirectTrajOpt.solve!(prob;
    max_iter=500,
    options=IpoptOptions(
        recalc_y="yes",
        recalc_y_feas_tol=1e8,
        eval_hessian=true,
    )
)

fid_phase2 = unitary_fidelity(
    iso_vec_to_operator(prob.trajectory[end].Ũ⃗), U_goal
)
println("Phase 2 fidelity: $fid_phase2")

# ## 9. Validation — Independent ODE Rollout
#
# Extract the optimized controls and forward-integrate with a fresh ODE solver
# to verify the result independently of the collocation discretization.

u_opt = prob.trajectory.u
t_opt = get_times(prob.trajectory)

u_opt_fn = t -> begin
    [LinearInterpolation(u_opt[j, :], t_opt)(t) for j in 1:2]
end

rollout_traj = unitary_rollout_trajectory(
    u_opt_fn, G, t_opt[end]; samples=T_samples
)

fid_rollout = unitary_fidelity(
    iso_vec_to_operator(rollout_traj[end].Ũ⃗), U_goal
)

println("\n" * "=" ^ 60)
println("RESULTS")
println("=" ^ 60)
println("  Collocation fidelity:  $fid_phase2")
println("  ODE rollout fidelity:  $fid_rollout")
println("  Gate dimension:        $(2^N_atoms) × $(2^N_atoms)")
println("  Total pulse duration:  $(t_opt[end]) μs")
println("  Number of time steps:  $T_samples")
println("=" ^ 60)

# ## 10. Visualization
#
# Plot the optimized control pulses Ω(t) and Δ(t).

fig = plot_controls(prob.trajectory;
    title = "Optimized Pulses: exp(-i·$(θ)·H_ZXZ), N=$N_atoms atoms",
    save_path = "ZXZ_controls_N$(N_atoms)_theta$(θ).png",
)
display(fig)

# Plot unitary populations during the pulse
fig_pop = plot_unitary_populations(prob.trajectory; control_name=:u)
save("ZXZ_populations_N$(N_atoms)_theta$(θ).png", fig_pop)
display(fig_pop)

println("\nFigures saved:")
println("  ZXZ_controls_N$(N_atoms)_theta$(θ).png")
println("  ZXZ_populations_N$(N_atoms)_theta$(θ).png")
