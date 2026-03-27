"""
    trajectories.jl

Core trajectory utilities: ODE-based forward simulation (rollout) and
NamedTrajectory construction for direct trajectory optimization.
"""

using OrdinaryDiffEqTsit5
using LinearAlgebra
using NamedTrajectories
import PiccoloQuantumObjects as PQO
import PiccoloQuantumObjects: Isomorphisms

"""
    unitary_rollout_trajectory(u_fn, G, T; samples=100, kwargs...)

Forward-integrate the Schrödinger equation dŨ⃗/dt = (I ⊗ G(u(t), t)) Ũ⃗ from the identity
using an ODE solver (Tsit5), then package the result as a `NamedTrajectory`.

# Arguments
- `u_fn`: Control function `t -> u(t)` returning a vector of control amplitudes.
- `G`: Generator function `(u, t) -> G` returning the (non-isomorphic) Hamiltonian generator.
- `T`: Final time.
- `samples`: Number of evenly-spaced time samples (including t=0 and t=T).
- `kwargs...`: Forwarded to `unitary_trajectory` (e.g., `control_bounds`, `U_goal`).
"""
function unitary_rollout_trajectory(
    u_fn::Function,
    G::Function,
    T::Float64;
    samples::Int=100,
    kwargs...
)
    ketdim = size(G(u_fn(0.0), 0.0), 1) ÷ 2
    Id = I(ketdim)
    Ũ⃗_init = PQO.operator_to_iso_vec(1.0I(ketdim))

    f! = (dx, x, p, t) -> mul!(dx, kron(Id, G(u_fn(t), t)), x)
    prob = ODEProblem(f!, Ũ⃗_init, (0.0, T))
    times = collect(range(0.0, T, samples))

    Ũ⃗_traj = stack(solve(prob, Tsit5();
        abstol=1e-12,
        reltol=1e-12,
        saveat=times
    ).u)

    return unitary_trajectory(
        Ũ⃗_traj,
        stack([u_fn(t) for t in times]),
        times;
        kwargs...
    )
end

"""
    unitary_trajectory(Ũ⃗_traj, controls, times; U_goal=nothing, control_bounds=nothing, ...)

Construct a `NamedTrajectory` from pre-computed unitary propagation data and controls.

# Arguments
- `Ũ⃗_traj`: Isomorphic unitary vectors at each time step (matrix: dim × T).
- `controls`: Control amplitudes at each time step (matrix: n_controls × T).
- `times`: Time grid.
- `U_goal`: Target unitary (optional, sets the trajectory goal).
- `control_bounds`: Tuple of (lower, upper) bound vectors for controls.
"""
function unitary_trajectory(
    Ũ⃗_traj::AbstractMatrix,
    controls::AbstractMatrix,
    times::AbstractVector;
    U_goal=nothing,
    control_bounds=nothing,
    Δt_min=1e-3minimum(diff(times)),
    Δt_max=2maximum(diff(times))
)
    u_dim = size(controls, 1)
    ketdim = Int(sqrt(size(Ũ⃗_traj, 1) ÷ 2))

    Δt = diff(times)
    Δt = [Δt; Δt[end]]

    data = (
        Ũ⃗ = Ũ⃗_traj,
        u = controls,
        Δt = Δt,
        t = times,
    )

    initial = (;
        Ũ⃗ = Isomorphisms.operator_to_iso_vec(1.0I(ketdim)),
        u = zeros(u_dim),
    )

    final = (; u = zeros(u_dim))

    goal = (;)
    if !isnothing(U_goal)
        goal = merge(goal, (; Ũ⃗ = Isomorphisms.operator_to_iso_vec(U_goal)))
    end

    bounds = (;
        Ũ⃗ = (-ones(size(Ũ⃗_traj, 1)), ones(size(Ũ⃗_traj, 1))),
        Δt = (Δt_min, Δt_max),
    )

    if !isnothing(control_bounds)
        bounds = merge(bounds, (; u = control_bounds))
    end

    return NamedTrajectory(
        data,
        controls=(:u,),
        timestep=:Δt,
        bounds=bounds,
        initial=initial,
        final=final,
        goal=goal,
    )
end
