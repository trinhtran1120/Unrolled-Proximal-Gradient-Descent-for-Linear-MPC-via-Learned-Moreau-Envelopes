# Generate feasible initial states for the linear MPC example.
using LinearAlgebra
using OSQP
using Printf
using NPZ

include(joinpath(@__DIR__, "..", "mpc", "solver.jl"))


function feasible_initial(problem::LinearMPC; solver_name::String, tol::Float64, x1_grid, x2_grid)
    is_feasible = initialization(solver_name, problem, tol)
    feasible = Vector{Vector{Float64}}()

    for x1 in x1_grid, x2 in x2_grid
        x0 = [Float64(x1), Float64(x2)]
        if is_feasible(x0)
            push!(feasible, x0)
        end
    end

    return feasible
end


function main()
    solver_name = "OSQP"
    tol = 1e-6
    x1_grid = -5.0:0.1:5.0
    x2_grid = -5.0:0.1:5.0

    problem = mpc_problem()
    feasible = feasible_initial(problem; solver_name = solver_name, tol = tol, x1_grid = x1_grid, x2_grid = x2_grid)
    feasible_x0 = reduce(hcat, feasible)

    println("================ Feasible x0 generation ================")
    println("solver = $solver_name")
    println("horizon N = $(problem.N)")
    println("feasible x0:")
    for x0 in feasible
        @printf("[% .2f, % .2f]\n", x0[1], x0[2])
    end

    output_path = joinpath(@__DIR__, "..","..","data","feasible_x0_nx=$(problem.nx)_N=$(problem.N).npz")
    npzwrite(output_path, Dict("feasible_x0" => feasible_x0, "nx" => problem.nx, "N" => problem.N))
    println("Saved x0 data to $output_path")
end

main()
