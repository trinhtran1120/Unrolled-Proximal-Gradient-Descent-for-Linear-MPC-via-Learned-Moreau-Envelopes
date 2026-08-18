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
using Random

include(joinpath(@__DIR__, "..", "mpc", "problem.jl"))
include(joinpath(@__DIR__, "..", "utils", "preprocess.jl"))
include(joinpath(@__DIR__, "..", "mpc", "solver.jl"))
include(joinpath(@__DIR__, "..", "mpc", "pgm.jl"))

const DATASET_DIR = joinpath(@__DIR__, "..", "..", "data")
mkpath(DATASET_DIR)

solver_name = "OSQP"
tol = 1e-2
pgm_rho = 0.001
pgm_adaptive = true
pgm_max_iter = 1000
near_zero_env_tol = 1e-12
near_zero_grad_tol = 1e-10
zero_to_nonzero_ratio = 0.1

# `pgm_rho` is only in force when adaptive = false; with backtracking enabled the
# step size is chosen by the algorithm, so it is neither recorded nor named.
mode_tag = pgm_adaptive ? "adaptive" : "fixed"

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

# Store the projection itself. For an indicator, prox(g, q, gamma) = Pi_F(q)
# regardless of gamma; training derives psi = 0.5*||q-proj||^2 and grad psi =
# q-proj, then Moreau quantities are recovered by scaling with 1/gamma.
data_train = Dict(
    "q" => Vector{Float64}[],
    "x0" => Vector{Float64}[],
    "proj" => Vector{Float64}[],
    "env" => Float64[],
    "grad" => Vector{Float64}[],
)
data_test = Dict(
    "q" => Vector{Float64}[],
    "x0" => Vector{Float64}[],
    "proj" => Vector{Float64}[],
    "env" => Float64[],
    "grad" => Vector{Float64}[],
)

function downsample_near_zero!(
    data;
    env_tol,
    grad_tol,
    zero_to_nonzero_ratio,
    seed = 1234,
)
    sample_count = length(data["env"])
    informative = [
        data["env"][i] > env_tol || norm(data["grad"][i], Inf) > grad_tol
        for i in 1:sample_count
    ]
    informative_indices = findall(informative)
    near_zero_indices = findall(.!informative)

    isempty(informative_indices) && error("No informative training samples were generated")

    max_near_zero = min(
        length(near_zero_indices),
        round(Int, zero_to_nonzero_ratio * length(informative_indices)),
    )
    rng = MersenneTwister(seed)
    retained_near_zero = if max_near_zero == 0
        Int[]
    else
        shuffle(rng, near_zero_indices)[1:max_near_zero]
    end
    retained_indices = sort!(vcat(informative_indices, retained_near_zero))

    for key in ("q", "x0", "proj", "env", "grad")
        data[key] = data[key][retained_indices]
    end

    return (
        generated = sample_count,
        informative = length(informative_indices),
        retained_near_zero = length(retained_near_zero),
        removed_near_zero = length(near_zero_indices) - length(retained_near_zero),
        retained = length(retained_indices),
    )
end

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
x1_grid = -2.0:0.5:3.0
x2_grid = -2.0:0.5:4.0
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
is_zero_state(point) = isapprox(point.x0, [0.0, 0.0]; atol = 1e-12)
is_spatial_test(point) = mod(point.ix + 2 * point.iy, 7) == 0

test_pool = [
    point.x0 for point in feasible_points
    if !is_zero_state(point) && (is_holdout(point) || is_spatial_test(point))
]
train_pool = [
    point.x0 for point in feasible_points
    if !is_zero_state(point) && !is_holdout(point) && !is_spatial_test(point)
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
    delta_J = abs(J_opt - J_PGM)
    relative_delta_J = if abs(J_opt) <= eps(Float64)
        delta_J <= eps(Float64) ? 0.0 : Inf
    else
        delta_J / abs(J_opt) * 100
    end

    @printf("J_PGM = %8.4f\n", J_PGM)
    @printf("Delta J = %8.4e\n", delta_J)
    @printf("Delta J/J = %8.4f%%\n", relative_delta_J)
    @printf("max|opt_U - PGM_U| = %8.4f\n\n", maximum(abs.(opt_U - pgm_U)))
end

filter_stats = downsample_near_zero!(
    data_train;
    env_tol = near_zero_env_tol,
    grad_tol = near_zero_grad_tol,
    zero_to_nonzero_ratio = zero_to_nonzero_ratio,
)
@printf(
    "Training samples: %d generated, %d informative, %d near-zero retained, %d near-zero removed, %d total retained\n\n",
    filter_stats.generated,
    filter_stats.informative,
    filter_stats.retained_near_zero,
    filter_stats.removed_near_zero,
    filter_stats.retained,
)
train_data = Dict{String,Any}(
    "q" => reduce(hcat, data_train["q"]),
    "x0" => reduce(hcat, data_train["x0"]),
    "proj" => reduce(hcat, data_train["proj"]),
    "input" => reduce(hcat, [vcat(q, x0) for (q, x0) in zip(data_train["q"], data_train["x0"])]),
    "grad" => reduce(hcat, data_train["grad"]),
    "adaptive" => pgm_adaptive,
    "env" => data_train["env"],
    "N" => mpc_data.N,
    "nx" => mpc_data.nx,
    "nu" => mpc_data.nu,
)
pgm_adaptive || (train_data["rho_initial"] = pgm_rho)
npzwrite(
    joinpath(DATASET_DIR, "PGM_nx=$(mpc_data.nx)_N=$(mpc_data.N)-train_$(mode_tag).npz"),
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

@printf("Collected %4d testing data points\n\n", length(data_test["q"]))
test_data = Dict{String,Any}(
    "q" => reduce(hcat, data_test["q"]),
    "x0" => reduce(hcat, data_test["x0"]),
    "proj" => reduce(hcat, data_test["proj"]),
    "input" => reduce(hcat, [vcat(q, x0) for (q, x0) in zip(data_test["q"], data_test["x0"])]),
    "grad" => reduce(hcat, data_test["grad"]),
    "adaptive" => pgm_adaptive,
    "env" => data_test["env"],
    "N" => mpc_data.N,
    "nx" => mpc_data.nx,
    "nu" => mpc_data.nu,
)
pgm_adaptive || (test_data["rho_initial"] = pgm_rho)
npzwrite(
    joinpath(DATASET_DIR, "PGM_nx=$(mpc_data.nx)_N=$(mpc_data.N)-test_$(mode_tag).npz"),
    test_data,
)
