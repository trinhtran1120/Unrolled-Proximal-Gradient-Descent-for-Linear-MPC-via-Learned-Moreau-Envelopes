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

#PGM algorithm settings
const pgm_tol = 1e-2
const pgm_rho = 0.1
const pgm_gamma = 0.001
const pgm_adaptive = true
const pgm_max_iter = 1000

# Tolerances for down-sampling near-zero Moreau envelopes and gradients
const near_zero_env_tol = 1e-12
const near_zero_grad_tol = 1e-10
const zero_to_nonzero_ratio = 0.1

function stack_samples(data, key)
    isempty(data[key]) && error("No samples collected for `$key`; check the feasible initial-state pool and PGM settings.")
    return reduce(hcat, data[key])
end

function main()
    mode_tag = pgm_adaptive ? "adaptive" : "fixed"

    mpc_data = mpc_problem()
    solve_mpc = mpc_solver(solver_name, mpc_data, solver_tol)
    solve_pgm = PGM_solver(
        mpc_data;
        rho = pgm_rho,
        gamma = pgm_gamma,
        adaptive = pgm_adaptive,
        max_iter = pgm_max_iter,
        tol = pgm_tol,
    )
    cost_func = mpc_data.cost_func

    data_train = Dict("input" => Vector{Float64}[], "parameter" => Vector{Float64}[], "proj" => Vector{Float64}[], "env" => Float64[], "grad" => Vector{Float64}[])
    data_test = Dict("input" => Vector{Float64}[], "parameter" => Vector{Float64}[], "proj" => Vector{Float64}[], "env" => Float64[], "grad" => Vector{Float64}[])

    is_feasible = initialization(feasibility_solver_name, mpc_data, solver_tol)

    x1_grid = -2.0:0.5:3.0
    x2_grid = -2.0:0.5:4.0
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
        pgm_U, pgm_X, _ = solve_pgm(x0; data = data_train, verbose = true)
        J_PGM = sum(cost_func(pgm_X[:, k], pgm_U[:, k]) for k in 1:mpc_data.N)
        delta_J = abs(J_opt - J_PGM)

        optimality_gap = if abs(J_opt) <= eps(Float64)
            delta_J <= eps(Float64) ? 0.0 : Inf
        else
            delta_J / abs(J_opt) * 100
        end

        @printf("J_PGM = %8.4f\n", J_PGM)
        @printf("Delta J = %8.4e\n", delta_J)
        @printf("Delta J/J = %8.4f%%\n", optimality_gap)
        @printf("max|opt_U - PGM_U| = %8.4f\n\n", maximum(abs.(opt_U - pgm_U)))
    end

    filter_train = filter_zero!(data_train; env_tol = near_zero_env_tol, grad_tol = near_zero_grad_tol, zero_to_nonzero_ratio = zero_to_nonzero_ratio)
    @printf(
        "Training samples: %d generated, %d informative, %d near-zero retained, %d near-zero removed, %d total retained\n\n",
        filter_train.generated,
        filter_train.informative,
        filter_train.retained_near_zero,
        filter_train.removed_near_zero,
        filter_train.retained,
    )
    train_data = Dict{String,Any}(
        "input" => stack_samples(data_train, "input"),
        "parameter" => stack_samples(data_train, "parameter"),
        "proj" => stack_samples(data_train, "proj"),
        "grad" => stack_samples(data_train, "grad"),
        "adaptive" => pgm_adaptive,
        "env" => data_train["env"],
        "N" => mpc_data.N,
        "nx" => mpc_data.nx,
        "nu" => mpc_data.nu,
        "rho_initial" => pgm_rho,
        "gamma_initial" => pgm_gamma,
    )
    npzwrite(
        joinpath(DATASET_DIR, "PGM-rho=$(pgm_rho)_nx=$(mpc_data.nx)_N=$(mpc_data.N)-train_$(mode_tag).npz"),
        train_data,
    )

    for x0 in test_pool
        println("================ Collecting testing data with initial state = $x0 ================")
        pgm_U, pgm_X, _ = solve_pgm(x0; data = data_test, verbose = true)
        J_PGM = sum(cost_func(pgm_X[:, k], pgm_U[:, k]) for k in 1:mpc_data.N)
        @printf("J_PGM = %8.4f\n", J_PGM)
    end

    @printf("Collected %4d testing data points\n\n", length(data_test["input"]))
    test_data = Dict{String,Any}(
        "input" => stack_samples(data_test, "input"),
        "parameter" => stack_samples(data_test, "parameter"),
        "proj" => stack_samples(data_test, "proj"),
        "grad" => stack_samples(data_test, "grad"),
        "adaptive" => pgm_adaptive,
        "env" => data_test["env"],
        "N" => mpc_data.N,
        "nx" => mpc_data.nx,
        "nu" => mpc_data.nu,
        "rho_initial" => pgm_rho,
        "gamma_initial" => pgm_gamma,
    )
    npzwrite(
        joinpath(DATASET_DIR, "PGM-rho=$(pgm_rho)_nx=$(mpc_data.nx)_N=$(mpc_data.N)-test_$(mode_tag).npz"),
        test_data,
    )
end

main()
