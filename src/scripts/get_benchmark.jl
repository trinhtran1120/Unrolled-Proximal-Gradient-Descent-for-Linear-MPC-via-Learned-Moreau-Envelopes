# Benchmark open-loop linear MPC methods.
using LinearAlgebra
using Printf
using Statistics

using OSQP, Ipopt, Gurobi

include(joinpath(@__DIR__, "..", "mpc", "learned_pgm.jl"))
include(joinpath(@__DIR__, "..", "mpc", "solver.jl"))

function arg_value(name::String, default::String)
    prefix = "--$name="
    for arg in ARGS
        startswith(arg, prefix) && return arg[length(prefix)+1:end]
    end
    return get(ENV, uppercase(name), default)
end

solver_name = "Gurobi"
tol = 1e-3
pgm_max_iter = 1000
learned_pgm_max_iter = 1000
benchmark_samples = 10
exact_pgm_rho = 0.1
exact_pgm_gamma = 0.1
exact_pgm_adaptive = true
learned_pgm_gamma = 0.01
learned_pgm_minimum_gamma = 1e-6
learned_pgm_reduce_gamma = 0.5
learned_pgm_increase_gamma = 1.05
model_path = arg_value(
    "model_path",
    joinpath(@__DIR__, "..", "..", "model", "linear-mpc-projection-mlp.json"),
)

mpc_data = mpc_problem()
x0 = [3.5, 1.5]
n_batch = 64
println(n_batch)
BLAS.set_num_threads(12)

solve_mpc = mpc_solver(solver_name, mpc_data, 1e-6)
solve_pgm = PGM_solver(mpc_data; gamma = exact_pgm_gamma, adaptive = exact_pgm_adaptive, max_iter = pgm_max_iter, tol = 1e-2)

learned_model = load_projection_mlp(model_path)
solve_learned = learned_PGM(
    learned_model,
    mpc_data;
    gamma = learned_pgm_gamma,
    minimum_gamma = learned_pgm_minimum_gamma,
    reduce_gamma = learned_pgm_reduce_gamma,
    increase_gamma = learned_pgm_increase_gamma,
    max_iter = learned_pgm_max_iter,
    gradient_batch_size = n_batch,
    tol = 1e-2,
)

function max_constraint_violation(problem, U, X)
    return max(
        0.0,
        maximum(problem.umin .- U),
        maximum(U .- problem.umax),
        maximum(problem.xmin .- X),
        maximum(X .- problem.xmax),
    )
end

function relative_objective_gap(J_ref, J)
    delta = abs(J_ref - J)
    if abs(J_ref) <= eps(Float64)
        return delta <= eps(Float64) ? 0.0 : Inf
    end
    return delta / abs(J_ref) * 100
end

println("================ Open-loop linear MPC benchmark ================")
println("initial state = $x0")
println("horizon N = $(mpc_data.N)")
println("solver = $solver_name")
println("exact PGM rho = $exact_pgm_rho")
println("exact PGM gamma = $exact_pgm_gamma")
println("exact PGM mode = $(exact_pgm_adaptive ? "adaptive" : "fixed")")
println("learned projection-MLP rho = $(learned_model.rho)")
println("learned projection-MLP PGM initial gamma = $learned_pgm_gamma")
println("learned projection-MLP PGM minimum gamma = $learned_pgm_minimum_gamma")
println("learned projection-MLP PGM reduce gamma = $learned_pgm_reduce_gamma")
println("learned projection-MLP PGM increase gamma = $learned_pgm_increase_gamma")
println("learned projection-MLP PGM depth = $learned_pgm_max_iter")
println("learned gradient batch size = $n_batch")
println("active Julia threads = $(Threads.nthreads())")
println("BLAS threads = $(BLAS.get_num_threads())")
println("learned model = $model_path")
println("learned model input_dim = $(learned_model.input_dim)")
println("learned model parameter_dim = $(learned_model.parameter_dim)")
println("learned model represents neural projection Pi_F(x0)(q)")
println()

println("---------------- $solver_name ----------------")
opt_X, opt_U, solver_time, J_opt = solve_mpc(x0; verbose = false)
@printf("objective = %10.6f\n", J_opt)
@printf("solve time = %8.3f ms\n\n", solver_time * 1000)

println("---------------- exact PGM ----------------")
pgm_time = @elapsed pgm_U, pgm_X, _ = solve_pgm(x0; verbose = true)
J_pgm, _ = evaluate_cost_gradient(mpc_data, x0, pgm_U)
@printf("objective = %10.6f\n", J_pgm)
@printf("solve time = %8.3f ms\n", pgm_time * 1000)
@printf("relative objective gap = %8.4f%%\n", relative_objective_gap(J_opt, J_pgm))
@printf("max constraint violation = %8.4e\n", max_constraint_violation(mpc_data, pgm_U, pgm_X))
@printf("max |solver U - PGM U| = %8.4e\n", maximum(abs.(opt_U - pgm_U)))
@printf("max |solver X - PGM X| = %8.4e\n\n", maximum(abs.(opt_X - pgm_X)))

println("---------------- learned projection-MLP PGM ----------------")
learned_sol = solve_learned(x0; verbose = false)
J_learned, _ = evaluate_cost_gradient(mpc_data, x0, learned_sol.U)
@printf("objective = %10.6f\n", J_learned)
@printf("solve time = %8.3f ms\n", learned_sol.solve_time * 1000)
@printf("relative objective gap = %8.4f%%\n", relative_objective_gap(J_opt, J_learned))
@printf("max constraint violation = %8.4e\n", max_constraint_violation(mpc_data, learned_sol.U, learned_sol.X))
@printf("max |solver U - learned PGM U| = %8.4e\n", maximum(abs.(opt_U - learned_sol.U)))
@printf("max |solver X - learned PGM X| = %8.4e\n\n", maximum(abs.(opt_X - learned_sol.X)))

# println("---------------- repeated timing ----------------")
# solver_times = zeros(Float64, benchmark_samples)
# pgm_times = zeros(Float64, benchmark_samples)
# learned_pgm_times = zeros(Float64, benchmark_samples)

# for sample in 1:benchmark_samples
#     _, _, solver_times[sample], _ = solve_mpc(x0)

#     pgm_times[sample] = @elapsed solve_pgm(x0)

#     learned_result = solve_learned(x0)
#     learned_pgm_times[sample] = learned_result.solve_time
# end

# @printf("%s mean time = %8.3f ms\n", solver_name, mean(solver_times[2:end]) * 1000)
# @printf("exact PGM mean time = %8.3f ms\n", mean(pgm_times[2:end]) * 1000)
# @printf("learned projection-MLP PGM mean time = %8.3f ms\n", mean(learned_pgm_times[2:end]) * 1000)
