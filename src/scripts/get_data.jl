# Generate training and testing datasets for the linear MPC example
using JuMP
import MathOptInterface as MOI
using LinearAlgebra
using NPZ
using OSQP, Ipopt, MosekTools
using Printf

include(joinpath(@__DIR__, "..", "mpc", "problem.jl"))
include(joinpath(@__DIR__, "..", "utils", "preprocess.jl"))
include(joinpath(@__DIR__, "..", "utils", "utils.jl"))
include(joinpath(@__DIR__, "..", "mpc", "solver.jl"))
include(joinpath(@__DIR__, "..", "mpc", "pgm.jl"))

#Set path
const DATASET_DIR = joinpath(@__DIR__, "..", "..", "data")
mkpath(DATASET_DIR)

#solver settings
solver_name = "Ipopt"
feasibility_solver_name = "OSQP"
const solver_tol = 1e-6

# PGM algorithm settings
const pgm_tol = 1e-3
const pgm_mu = 1.0
const pgm_gamma = 0.1
const pgm_relaxation = 1.0
const pgm_adaptive = true
const pgm_increase_gamma = 1.05
const pgm_max_iter = 1000

function stack_samples(data, key)
    isempty(data[key]) && error("No samples collected for `$key`; check the feasible initial-state pool and PGM settings.")
    return reduce(hcat, data[key])
end

function dataset_payload(data, mpc_data)
    return Dict{String,Any}(
        "x0" => stack_samples(data, "x0"),
        "U_query" => stack_samples(data, "U_query"),
        "V_star" => stack_samples(data, "V_star"),
        "N" => mpc_data.N,
        "nx" => mpc_data.nx,
        "nu" => mpc_data.nu,
    )
end

function main()
    mode_tag = pgm_adaptive ? "adaptive" : "fixed"

    mpc_data = mpc_problem()
    solve_mpc = mpc_solver(solver_name, mpc_data, solver_tol)
    solve_pgm = PGM_solver(
        mpc_data;
        mu = pgm_mu,
        gamma = pgm_gamma,
        relaxation = pgm_relaxation,
        adaptive = pgm_adaptive,
        increase_gamma = pgm_increase_gamma,
        max_iter = pgm_max_iter,
        tol = pgm_tol,
    )
    cost_func = mpc_data.cost_func

    data_train = Dict{String,Any}()
    data_test = Dict{String,Any}()

    is_feasible = initialization(feasibility_solver_name, mpc_data, solver_tol)

    x1_grid = -3.0:0.5:3.0
    x2_grid = -3.0:0.5:4.0
    initial_pool = [
        [Float64(x1), Float64(x2)]
        for x1 in x1_grid
        for x2 in x2_grid
        if !isapprox([x1, x2], [0.0, 0.0]; atol = 1e-12) && is_feasible([Float64(x1), Float64(x2)])
    ]
    split_idx = round(Int, 0.8 * length(initial_pool))
    isempty(initial_pool) && error("No feasible initial states found; try a different feasibility solver or a wider grid.")

    train_pool = initial_pool[1:split_idx]
    test_pool = initial_pool[split_idx+1:end]
    isempty(train_pool) && error("Training pool is empty; increase the initial-state grid size.")
    isempty(test_pool) && error("Testing pool is empty; increase the initial-state grid size.")

    for x0 in train_pool
        println("================ Collecting training data with initial state = $x0 ================")
        println("---------------- $solver_name ----------------")

        opt_X, opt_U, _solver_time, J_opt = solve_mpc(x0; verbose=false)
        @printf("J_opt = %8.4f\n", J_opt)

        println("---------------- PGM ----------------")
        pgm_U, pgm_X, J_pgm_smooth = solve_pgm(x0; data = data_train, verbose = true)
        J_PGM = sum(cost_func(pgm_X[:, k], pgm_U[:, k]) for k in 1:mpc_data.N)
        delta_J = abs(J_opt - J_PGM)

        optimality_gap = if abs(J_opt) <= eps(Float64)
            delta_J <= eps(Float64) ? 0.0 : Inf
        else
            delta_J / abs(J_opt) * 100
        end

        @printf("J_PGM = %8.4f\n", J_PGM)
        @printf("J_PGM_smooth = %8.4f\n", J_pgm_smooth)
        @printf("Delta J = %8.4e\n", delta_J)
        @printf("Delta J/J = %8.4f%%\n", optimality_gap)
        @printf("max|opt_U - PGM_U| = %8.4f\n\n", maximum(abs.(opt_U - pgm_U)))
    end

    @printf("Collected %4d training data points\n\n", length(data_train["U_query"]))
    train_data = dataset_payload(data_train, mpc_data)
    npzwrite(
        joinpath(DATASET_DIR, "PGM-mu=$(pgm_mu)_gamma=$(pgm_gamma)_nx=$(mpc_data.nx)_N=$(mpc_data.N)-train_$(mode_tag).npz"),
        train_data,
    )

    for x0 in test_pool
        println("================ Collecting testing data with initial state = $x0 ================")
        pgm_U, pgm_X, J_pgm_smooth = solve_pgm(x0; data = data_test, verbose = true)
        J_PGM = sum(cost_func(pgm_X[:, k], pgm_U[:, k]) for k in 1:mpc_data.N)
        @printf("J_PGM = %8.4f\n", J_PGM)
        @printf("J_PGM_smooth = %8.4f\n", J_pgm_smooth)
    end

    @printf("Collected %4d testing data points\n\n", length(data_test["U_query"]))
    test_data = dataset_payload(data_test, mpc_data)
    npzwrite(
        joinpath(DATASET_DIR, "PGM-mu=$(pgm_mu)_gamma=$(pgm_gamma)_nx=$(mpc_data.nx)_N=$(mpc_data.N)-test_$(mode_tag).npz"),
        test_data,
    )
end

main()
