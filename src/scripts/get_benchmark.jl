# Benchmark open-loop linear MPC methods.
using LinearAlgebra
using Printf
using Statistics

using OSQP, Ipopt, Clarabel
include(joinpath(@__DIR__, "..", "mpc", "learned_pgm.jl"))
include(joinpath(@__DIR__, "..", "mpc", "solver.jl"))


solver_name = "OSQP"
tol = 1e-3
pgm_max_iter = 1000
learned_pgm_max_iter = 1000
benchmark_samples = 10
exact_pgm_mu = 1.0
exact_pgm_gamma = 1.0
exact_pgm_adaptive = true
exact_pgm_increase_gamma = 1.0
learned_pgm_gamma = 1.0
learned_pgm_minimum_gamma = 1e-6
learned_pgm_reduce_gamma = 0.5
learned_pgm_increase_gamma = 1.0
learned_pgm_relaxation = 1.0
model_path = joinpath(@__DIR__, "..", "..", "model", "linear-mpc-projection-mlp_tanh.json")

mpc_data = mpc_problem()
x0 = [1.4, 2.6]
BLAS.set_num_threads(12)

solve_mpc = mpc_solver(solver_name, mpc_data, 1e-6)
solve_pgm = PGM_solver(
    mpc_data;
    mu = exact_pgm_mu,
    gamma = exact_pgm_gamma,
    adaptive = exact_pgm_adaptive,
    increase_gamma = exact_pgm_increase_gamma,
    max_iter = pgm_max_iter,
    tol = 1e-2,
)

learned_model = load_projection_mlp(model_path)
solve_learned = learned_PGM(
    learned_model,
    mpc_data;
    mu = learned_model.mu,
    gamma = learned_pgm_gamma,
    relaxation = learned_pgm_relaxation,
    minimum_gamma = learned_pgm_minimum_gamma,
    reduce_gamma = learned_pgm_reduce_gamma,
    increase_gamma = learned_pgm_increase_gamma,
    max_iter = learned_pgm_max_iter,
    tol = 0.01,
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
println("exact TOS Moreau mu = $exact_pgm_mu")
println("exact TOS initial splitting gamma = $exact_pgm_gamma")
println("exact TOS mode = $(exact_pgm_adaptive ? "adaptive" : "fixed")")
println("exact TOS gamma increase factor = $exact_pgm_increase_gamma")
println("Learned TOS Moreau mu = $(learned_model.mu)")
println("Learned TOS initial gamma = $learned_pgm_gamma")
println("Learned TOS minimum gamma = $learned_pgm_minimum_gamma")
println("Learned TOS reduce gamma = $learned_pgm_reduce_gamma")
println("Learned TOS increase gamma = $learned_pgm_increase_gamma")
println("Learned TOS relaxation = $learned_pgm_relaxation")
println("Learned TOS depth = $learned_pgm_max_iter")
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

println("---------------- exact TOS ----------------")
pgm_time = @elapsed pgm_U, pgm_X, _ = solve_pgm(x0; verbose = true)
J_pgm, _ = evaluate_cost_gradient(mpc_data, x0, pgm_U)
@printf("objective = %10.6f\n", J_pgm)
@printf("solve time = %8.3f ms\n", pgm_time * 1000)
@printf("relative objective gap = %8.4f%%\n", relative_objective_gap(J_opt, J_pgm))
@printf("max constraint violation = %8.4e\n", max_constraint_violation(mpc_data, pgm_U, pgm_X))
@printf("max |solver U - PGM U| = %8.4e\n", maximum(abs.(opt_U - pgm_U)))
@printf("max |solver X - PGM X| = %8.4e\n\n", maximum(abs.(opt_X - pgm_X)))

println("---------------- Learned TOS ----------------")
learned_sol = solve_learned(x0; verbose = true)
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
# @printf("exact TOS mean time = %8.3f ms\n", mean(pgm_times[2:end]) * 1000)
# @printf("Learned TOS mean time = %8.3f ms\n", mean(learned_pgm_times[2:end]) * 1000)
