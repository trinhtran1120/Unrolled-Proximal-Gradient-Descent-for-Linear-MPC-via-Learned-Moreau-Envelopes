# Generate training and testing datasets for the linear MPC example.
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using JuMP
import MathOptInterface as MOI
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

solver_name = "OSQP"
tol = 1e-2
pgm_rho = 0.1
pgm_adaptive = true
pgm_max_iter = 1000

mpc_data = mpc_problem()
solve_mpc = mpc_solver(solver_name, mpc_data, 1e-6)
solve_pgm = PGM_solver(
    mpc_data;
    rho = pgm_rho,
    adaptive = pgm_adaptive,
    max_iter = pgm_max_iter,
    tol = tol,
)
cost_func = mpc_data.cost_func

data_train = Dict(
    "input" => Vector{Float64}[],
    "env" => Float64[],
    "grad" => Vector{Float64}[],
    "q" => Float64[],
    "r" => Vector{Float64}[],
    "gamma" => Float64[],
    "iter" => Int[],
)
data_test = Dict(
    "input" => Vector{Float64}[],
    "env" => Float64[],
    "grad" => Vector{Float64}[],
    "q" => Float64[],
    "r" => Vector{Float64}[],
    "gamma" => Float64[],
    "iter" => Int[],
)

function initial_state_feasibility_checker(problem, solver_name)
    model = pick_solver(solver_name)
    nx, nu, N = problem.nx, problem.nu, problem.N

    @variable(model, problem.xmin <= x[1:nx, 1:N+1] <= problem.xmax)
    @variable(model, problem.umin <= u[1:nu, 1:N] <= problem.umax)
    @variable(model, x0[i in 1:nx] in MOI.Parameter(problem.x0[i]))

    @constraint(model, x[:, 1] .== x0)
    for k in 1:N
        @constraint(model, x[:, k+1] .== problem.A * x[:, k] + problem.B * u[:, k])
    end
    @objective(model, Min, 0.0)

    return function (init)
        set_parameter_value.(x0, init)
        optimize!(model)
        return termination_status(model) == MOI.OPTIMAL &&
               primal_status(model) == MOI.FEASIBLE_POINT
    end
end

is_feasible = initial_state_feasibility_checker(mpc_data, solver_name)

# Split feasible initial states spatially. Nearby grid points generally land in
# different subsets, while every PGM trajectory remains entirely in one subset.
x1_grid = -1.0:0.5:2.0
x2_grid = -1.0:0.5:1.0
candidate_points = [
    (ix = ix, iy = iy, x0 = [Float64(x1), Float64(x2)])
    for (ix, x1) in enumerate(x1_grid)
    for (iy, x2) in enumerate(x2_grid)
]
feasible_points = filter(point -> is_feasible(point.x0), candidate_points)

holdout_states = [[2.0, 0.0], [3.0, 4.0]]
is_holdout(point) = any(
    isapprox(point.x0, holdout; atol = 1e-12) for holdout in holdout_states
)
is_nan_prone_train_point(point) = isapprox(point.x0, [0.0, 0.0]; atol = 1e-12)
is_spatial_test(point) = mod(point.ix + 2 * point.iy, 7) == 0

test_pool = [
    point.x0 for point in feasible_points
    if is_holdout(point) || is_spatial_test(point)
]
train_pool = [
    point.x0 for point in feasible_points
    if !is_holdout(point) && !is_spatial_test(point) && !is_nan_prone_train_point(point)
]

@printf(
    "Initial-state grid: %d candidates, %d feasible, %d training, %d testing\n\n",
    length(candidate_points),
    length(feasible_points),
    length(train_pool),
    length(test_pool),
)


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

@printf(
    "Collected %4d training data points; saving all samples without downsampling\n\n",
    length(data_train["input"]),
)
train_data = Dict(
    "input" => reduce(hcat, data_train["input"]),
    "q" => data_train["q"],
    "r" => reduce(hcat, data_train["r"]),
    "rho_initial" => pgm_rho,
    "adaptive" => pgm_adaptive,
    "gamma" => data_train["gamma"],
    "env" => data_train["env"],
    "grad" => reduce(hcat, data_train["grad"]),
    "iter" => data_train["iter"],
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
    "q" => data_test["q"],
    "r" => reduce(hcat, data_test["r"]),
    "rho_initial" => pgm_rho,
    "adaptive" => pgm_adaptive,
    "gamma" => data_test["gamma"],
    "env" => data_test["env"],
    "grad" => reduce(hcat, data_test["grad"]),
    "iter" => data_test["iter"],
    "N" => mpc_data.N,
    "nx" => mpc_data.nx,
    "nu" => mpc_data.nu,
)
npzwrite(
    joinpath(DATASET_DIR, "PGM-rho=$(pgm_rho)_nx=$(mpc_data.nx)_N=$(mpc_data.N)-test.npz"),
    test_data,
)
