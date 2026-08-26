# Generate feasible initial states for the linear MPC example.
using LinearAlgebra
using OSQP
using Printf

include(joinpath(@__DIR__, "..", "mpc", "solver.jl"))

function arg_value(name::String, default::String)
    prefix = "--$name="
    for arg in ARGS
        startswith(arg, prefix) && return arg[length(prefix)+1:end]
    end
    return get(ENV, uppercase(name), default)
end

function parse_range_arg(value::String)
    parts = split(value, ":")
    length(parts) == 3 || throw(ArgumentError("range must have format start:step:stop"))
    start, step, stop = parse.(Float64, parts)
    step > 0.0 || throw(ArgumentError("range step must be positive"))
    return start:step:stop
end

function feasible_initial_states(problem::LinearMPC; solver_name::String, tol::Float64, x1_grid, x2_grid)
    is_feasible = initialization(solver_name, problem, tol)
    feasible = Vector{Vector{Float64}}()
    infeasible = Vector{Vector{Float64}}()

    for x1 in x1_grid, x2 in x2_grid
        x0 = [Float64(x1), Float64(x2)]
        if is_feasible(x0)
            push!(feasible, x0)
        else
            push!(infeasible, x0)
        end
    end

    return feasible, infeasible
end

function main()
    solver_name = arg_value("solver", "OSQP")
    tol = parse(Float64, arg_value("tol", "1e-6"))
    x1_grid = parse_range_arg(arg_value("x1", "-5.0:0.1:5.0"))
    x2_grid = parse_range_arg(arg_value("x2", "-5.0:0.1:5.0"))

    problem = mpc_problem()
    feasible, infeasible = feasible_initial_states(
        problem;
        solver_name = solver_name,
        tol = tol,
        x1_grid = x1_grid,
        x2_grid = x2_grid,
    )

    total = length(feasible) + length(infeasible)
    feasible_ratio = total == 0 ? 0.0 : 100 * length(feasible) / total

    println("================ Feasible x0 generation ================")
    println("solver = $solver_name")
    println("horizon N = $(problem.N)")
    println("x1 grid = $(first(x1_grid)):$(step(x1_grid)):$(last(x1_grid))")
    println("x2 grid = $(first(x2_grid)):$(step(x2_grid)):$(last(x2_grid))")
    @printf("feasible points = %d / %d (%6.2f%%)\n", length(feasible), total, feasible_ratio)
    println()
    println("feasible x0:")
    for x0 in feasible
        @printf("[% .6f, % .6f]\n", x0[1], x0[2])
    end
end

main()
