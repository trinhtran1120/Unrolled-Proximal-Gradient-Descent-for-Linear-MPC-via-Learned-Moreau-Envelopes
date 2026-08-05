# Generate training and testing datasets for the linear MPC example.
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using JuMP
using LinearAlgebra
using NPZ
using OSQP, Gurobi, Ipopt, MosekTools
using Printf

include(joinpath(@__DIR__, "..", "mpc", "problem.jl"))
include(joinpath(@__DIR__, "..", "utils", "preprocess.jl"))
include(joinpath(@__DIR__, "..", "mpc", "solver.jl"))
include(joinpath(@__DIR__, "..", "mpc", "pgm.jl"))

const DATASET_DIR = joinpath(@__DIR__, "..", "..", "data")
mkpath(DATASET_DIR)

solver_name = "Gurobi"
tol = 1e-2
pgm_rho = 0.1
pgm_max_iter = 1000

mpc_data = mpc_problem()
solve_mpc = mpc_solver(solver_name, mpc_data, tol)
solve_pgm = PGM_solver(mpc_data; rho = pgm_rho, max_iter = pgm_max_iter, tol = tol)
cost_func = mpc_data.cost_func

data_train = Dict("input" => Vector{Float64}[], "env" => Float64[], "grad" => Vector{Float64}[], "gamma" => Float64[])
data_test = Dict("input" => Vector{Float64}[], "env" => Float64[], "grad" => Vector{Float64}[], "gamma" => Float64[])

train_pool = [
    [3.0, 1.0],
    [2.5, 0.5],
    [1.5, -0.5],
    [2.5, 1.0],
    [2.0, 1.0],
    [2.0, 0.5],
    [2.0, 0.0],
    [1.5, 1.0],
    [1.5, 0.5],
    [1.5, 0.0],
    [1.0, 1.0],
    [1.0, 0.5],
    [1.0, -0.5],
]

test_pool = [
    [2.0, 0.0],
]

## train data generation
for x0 in train_pool
    println("================ Collecting training data with initial state = $x0 ================")
    println("---------------- $solver_name ----------------")

    opt_X, opt_U, _solver_time, J_opt = solve_mpc(x0; verbose=false)
    @printf("J_opt = %8.4f\n", J_opt)

    println("---------------- PGM ----------------")
    pgm_U, pgm_X, _ = solve_pgm(
        x0;
        data = data_train,
        verbose = true,
    )
    J_PGM = sum(cost_func(pgm_X[:, k], pgm_U[:, k]) for k in 1:mpc_data.N)

    @printf("J_PGM = %8.4f\n", J_PGM)
    @printf("Delta J/J = %8.4f%%\n", abs(J_opt - J_PGM) / abs(J_opt) * 100)
    @printf("max|opt_U - PGM_U| = %8.4f\n\n", maximum(abs.(opt_U - pgm_U)))
end

@printf("Collected %4d training data points\n\n", length(data_train["input"]))
train_data = Dict(
    "input" => reduce(hcat, data_train["input"]),
    "grad" => reduce(hcat, data_train["grad"]),
    "rho_initial" => pgm_rho,
    "gamma" => data_train["gamma"],
    "env" => data_train["env"],
    "N" => mpc_data.N,
    "nx" => mpc_data.nx,
    "nu" => mpc_data.nu,
)
npzwrite(
    joinpath(DATASET_DIR, "PGM-rho=$(pgm_rho)_nx=$(mpc_data.nx)_N=$(mpc_data.N)-train.npz"),
    train_data,
)


## test data generation
for x0 in test_pool
    println("================ Collecting testing data with initial state = $x0 ================")

    pgm_U, pgm_X, _ = solve_pgm(
        x0;
        data = data_test,
        verbose = true,
    )
    J_PGM = sum(cost_func(pgm_X[:, k], pgm_U[:, k]) for k in 1:mpc_data.N)
    @printf("J_PGM = %8.4f\n", J_PGM)
end

@printf("Collected %4d testing data points\n\n", length(data_test["input"]))
test_data = Dict(
    "input" => reduce(hcat, data_test["input"]),
    "grad" => reduce(hcat, data_test["grad"]),
    "rho_initial" => pgm_rho,
    "gamma" => data_test["gamma"],
    "env" => data_test["env"],
    "N" => mpc_data.N,
    "nx" => mpc_data.nx,
    "nu" => mpc_data.nu,
)
npzwrite(
    joinpath(DATASET_DIR, "PGM-rho=$(pgm_rho)_nx=$(mpc_data.nx)_N=$(mpc_data.N)-test.npz"),
    test_data,
)
