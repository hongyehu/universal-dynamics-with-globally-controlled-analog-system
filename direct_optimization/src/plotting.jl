"""
    plotting.jl

Visualization utilities for quantum control pulses on Rydberg atom chains.
"""

using CairoMakie

_to_vector(x::AbstractVector) = collect(x)
_to_vector(x::AbstractMatrix) = collect(x)
_extract_control(controls::AbstractMatrix, i::Int) = collect(controls[i, :])
_extract_times(times::AbstractVector) = collect(times)

"""
    plot_controls(traj::NamedTrajectory; kwargs...)
    plot_controls(times, controls; kwargs...)

Plot quantum control pulses with a separate panel for each control channel.

For Rydberg systems with `ignore_Y_drive=true`:
- Control 1: Rabi frequency Ω(t) [MHz]
- Control 2: Detuning Δ(t) [MHz]
"""
function plot_controls(
    traj::NamedTrajectory;
    kwargs...
)
    times = get_times(traj)
    controls = traj.u
    return plot_controls(times, controls; kwargs...)
end

function plot_controls(
    times::AbstractVector{<:Real},
    controls::AbstractMatrix{<:Real};
    control_labels::Union{Vector{String}, Nothing} = nothing,
    time_units::String = "μs",
    control_units::String = "MHz",
    figsize::Tuple = (700, 450),
    linewidth::Real = 2.5,
    colors::Vector = [:royalblue, :firebrick, :forestgreen, :darkorange],
    title::String = "Control Pulses",
    save_path::Union{String, Nothing} = nothing
)
    n_controls, n_times = size(controls)

    if isnothing(control_labels)
        if n_controls == 2
            control_labels = ["Ω(t)", "Δ(t)"]
        elseif n_controls == 3
            control_labels = ["Ωₓ(t)", "Ωᵧ(t)", "Δ(t)"]
        else
            control_labels = ["Control $i" for i in 1:n_controls]
        end
    end

    fig = Figure(size=figsize, fontsize=14)
    Label(fig[0, :], title, fontsize=16, font=:bold)

    axes = []
    for i in 1:n_controls
        ax = Axis(fig[i, 1],
            xlabel = i == n_controls ? "Time [$time_units]" : "",
            ylabel = "$(control_labels[i]) [$control_units]",
        )

        lines!(ax, _extract_times(times), _extract_control(controls, i),
            color=colors[mod1(i, length(colors))],
            linewidth=linewidth,
        )

        ax.xgridvisible = true
        ax.ygridvisible = true
        push!(axes, ax)
    end

    for i in 2:n_controls
        linkxaxes!(axes[1], axes[i])
    end
    rowgap!(fig.layout, 5)

    if !isnothing(save_path)
        save(save_path, fig)
    end

    return fig
end
