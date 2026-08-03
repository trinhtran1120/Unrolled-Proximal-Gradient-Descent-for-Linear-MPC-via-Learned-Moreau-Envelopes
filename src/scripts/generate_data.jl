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

solver_name = "Ipopt"
tol = 5e-4
pgm_rho = 0.001
pgm_max_iter = 1000

mpc_data = mpc_problem()
solve_mpc = mpc_solver(solver_name, mpc_data, tol)
cost_func = mpc_data.cost_func


data_train = Dict("input" => Vector{Float64}[], "env" => Float64[], "grad" => Vector{Float64}[])
data_test = Dict("input" => Vector{Float64}[], "env" => Float64[], "grad" => Vector{Float64}[])

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

for x0 in train_pool
    println("================ Collecting training data with initial state = $x0 ================")
    println("---------------- $solver_name ----------------")

    opt_X, opt_U, J_opt = solve_mpc(x0)
    @printf("J_opt = %8.4f\n", J_opt)

    println("---------------- PGM ----------------")
    pgm_sol = PGM(
        mpc_data,
        x0;
        rho = pgm_rho,
        max_iter = pgm_max_iter,
        tol = tol,
        verbose = true,
    )
    J_PGM, _ = grad_cost(mpc_data, x0, pgm_sol.U)

    @printf("J_PGM = %8.4f\n", J_PGM)
    @printf("Delta J/J = %8.4f%%\n", abs(J_opt - J_PGM) / abs(J_opt) * 100)
    @printf("max|opt_U - PGM_U| = %8.4f\n\n", maximum(abs.(opt_U - pgm_sol.U)))

    for i in eachindex(pgm_sol.W_data)
        push!(data_train["input"], vcat(pgm_sol.W_data[i], x0))
        push!(data_train["env"], pgm_sol.ME_data[i])
        push!(data_train["grad"], pgm_sol.ME_grad_data[i])
    end
end

@printf("Collected %4d training data points\n\n", length(data_train["input"]))
train_data = Dict(
    "input" => reduce(hcat, data_train["input"]),
    "grad" => reduce(hcat, data_train["grad"]),
    "rho" => pgm_rho,
    "env" => data_train["env"],
    "N" => mpc_data.N,
    "nx" => mpc_data.nx,
    "nu" => mpc_data.nu,
)
npzwrite(
    joinpath(DATASET_DIR, "PGM-rho=$(pgm_rho)-train.npz"),
    train_data,
)

for x0 in test_pool
    println("================ Collecting testing data with initial state = $x0 ================")

    pgm_sol = PGM(
        mpc_data,
        x0;
        rho = pgm_rho,
        max_iter = pgm_max_iter,
        tol = tol,
        verbose = true,
    )
    J_PGM, _ = grad_cost(mpc_data, x0, pgm_sol.U)
    @printf("J_PGM = %8.4f\n", J_PGM)

    for i in eachindex(pgm_sol.W_data)
        push!(data_test["input"], vcat(pgm_sol.W_data[i], x0))
        push!(data_test["env"], pgm_sol.ME_data[i])
        push!(data_test["grad"], pgm_sol.ME_grad_data[i])
    end
end

@printf("Collected %4d testing data points\n\n", length(data_test["input"]))
test_data = Dict(
    "input" => reduce(hcat, data_test["input"]),
    "grad" => reduce(hcat, data_test["grad"]),
    "rho" => pgm_rho,
    "env" => data_test["env"],
    "N" => mpc_data.N,
    "nx" => mpc_data.nx,
    "nu" => mpc_data.nu,
)
npzwrite(
    joinpath(DATASET_DIR, "PGM-rho=$(pgm_rho)-test.npz"),
    test_data,
)
