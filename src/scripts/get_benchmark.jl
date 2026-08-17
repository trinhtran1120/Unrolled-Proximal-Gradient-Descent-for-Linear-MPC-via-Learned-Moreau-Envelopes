# Benchmark open-loop linear MPC methods.
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using LinearAlgebra
using Printf
using Statistics

using OSQP

include(joinpath(@__DIR__, "..", "mpc", "learned_pgm_adaptive.jl"))
include(joinpath(@__DIR__, "..", "mpc", "solver.jl"))

const SOLVER_NAME = "OSQP"
const TOL = 1e-3
const PGM_MAX_ITER = 1000
const LEARNED_PGM_MAX_ITER = 1000
const BENCHMARK_SAMPLES = 10
const EXACT_PGM_RHO = 0.1
const EXACT_PGM_ADAPTIVE = true
const LEARNED_PGM_RHO = 0.01
const LEARNED_PGM_MINIMUM_RHO = 1e-6
const LEARNED_PGM_REDUCE_RHO = 0.5
const LEARNED_PGM_INCREASE_RHO = 1.05
const LEARNED_PGM_POLISH_ITER = 10

model_path = joinpath(
    @__DIR__,
    "..",
    "..",
    "model",
    "linear-mpc-lpcf001-adaptive.json",
)

mpc_data = mpc_problem()
x0 = mpc_data.x0

solve_mpc = mpc_solver(SOLVER_NAME, mpc_data, TOL)
solve_pgm = PGM_solver(
    mpc_data;
    rho = EXACT_PGM_RHO,
    adaptive = EXACT_PGM_ADAPTIVE,
    max_iter = PGM_MAX_ITER,
    tol = TOL,
)
learned_model = load_learned_pcf(model_path)

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
println("solver = $SOLVER_NAME")
println("exact PGM rho = $EXACT_PGM_RHO")
println("exact PGM mode = $(EXACT_PGM_ADAPTIVE ? "adaptive" : "fixed")")
println("learned adaptive PCF-PGM initial rho = $LEARNED_PGM_RHO")
println("learned adaptive PCF-PGM minimum rho = $LEARNED_PGM_MINIMUM_RHO")
println("learned adaptive PCF-PGM reduce rho = $LEARNED_PGM_REDUCE_RHO")
println("learned adaptive PCF-PGM increase rho = $LEARNED_PGM_INCREASE_RHO")
println("learned PCF-PGM depth = $LEARNED_PGM_MAX_ITER")
println("exact PGM polish depth = $LEARNED_PGM_POLISH_ITER")
println("learned model = $model_path")
println()

println("---------------- $SOLVER_NAME ----------------")
opt_X, opt_U, solver_time, J_opt = solve_mpc(x0; verbose = false)
@printf("objective = %10.6f\n", J_opt)
@printf("solve time = %8.3f ms\n\n", solver_time * 1000)

println("---------------- exact PGM ----------------")
pgm_time = @elapsed pgm_U, pgm_X, _ = solve_pgm(x0; verbose = true)
J_pgm, _ = grad_cost(mpc_data, x0, pgm_U)
@printf("objective = %10.6f\n", J_pgm)
@printf("solve time = %8.3f ms\n", pgm_time * 1000)
@printf("relative objective gap = %8.4f%%\n", relative_objective_gap(J_opt, J_pgm))
@printf("max constraint violation = %8.4e\n", max_constraint_violation(mpc_data, pgm_U, pgm_X))
@printf("max |solver U - PGM U| = %8.4e\n", maximum(abs.(opt_U - pgm_U)))
@printf("max |solver X - PGM X| = %8.4e\n\n", maximum(abs.(opt_X - pgm_X)))

println("---------------- learned adaptive PCF-PGM ----------------")
learned_sol = learned_PGM(
    learned_model,
    mpc_data;
    x0 = x0,
    rho = LEARNED_PGM_RHO,
    minimum_rho = LEARNED_PGM_MINIMUM_RHO,
    reduce_rho = LEARNED_PGM_REDUCE_RHO,
    increase_rho = LEARNED_PGM_INCREASE_RHO,
    max_iter = LEARNED_PGM_MAX_ITER,
    polish_iter = LEARNED_PGM_POLISH_ITER,
    tol = TOL,
    verbose = true,
)
J_learned, _ = grad_cost(mpc_data, x0, learned_sol.U)
@printf("objective = %10.6f\n", J_learned)
@printf("solve time = %8.3f ms\n", learned_sol.solve_time * 1000)
@printf("learned iterations = %d\n", length(learned_sol.objective_history))
@printf("learned final residual = %8.4e\n", learned_sol.residual_history[end])
@printf("polish iterations = %d\n", length(learned_sol.polish_residual_history))
if !isempty(learned_sol.polish_residual_history)
    @printf("polish final residual = %8.4e\n", learned_sol.polish_residual_history[end])
end
@printf("initial rho = %8.4e\n", LEARNED_PGM_RHO)
@printf("final rho = %8.4e\n", learned_sol.rho)
@printf("min rho = %8.4e\n", minimum(learned_sol.rho_history))
@printf("max rho = %8.4e\n", maximum(learned_sol.rho_history))
@printf("relative objective gap = %8.4f%%\n", relative_objective_gap(J_opt, J_learned))
@printf("max constraint violation = %8.4e\n", max_constraint_violation(mpc_data, learned_sol.U, learned_sol.X))
@printf("max |solver U - learned PGM U| = %8.4e\n", maximum(abs.(opt_U - learned_sol.U)))
@printf("max |solver X - learned PGM X| = %8.4e\n\n", maximum(abs.(opt_X - learned_sol.X)))

# println("---------------- repeated timing ----------------")
# solver_times = zeros(Float64, BENCHMARK_SAMPLES)
# pgm_times = zeros(Float64, BENCHMARK_SAMPLES)
# learned_pgm_times = zeros(Float64, BENCHMARK_SAMPLES)

# for sample in 1:BENCHMARK_SAMPLES
#     _, _, solver_times[sample], _ = solve_mpc(x0)

#     pgm_times[sample] = @elapsed solve_pgm(x0)

#     learned_result = learned_PGM(
#         learned_model,
#         mpc_data;
#         x0 = x0,
#         rho = LEARNED_PGM_RHO,
#         minimum_rho = LEARNED_PGM_MINIMUM_RHO,
#         reduce_rho = LEARNED_PGM_REDUCE_RHO,
#         increase_rho = LEARNED_PGM_INCREASE_RHO,
#         max_iter = LEARNED_PGM_MAX_ITER,
#         tol = TOL,
#     )
#     learned_pgm_times[sample] = learned_result.solve_time
# end

# @printf("%s mean time = %8.3f ms\n", SOLVER_NAME, mean(solver_times[2:end]) * 1000)
# @printf("exact PGM mean time = %8.3f ms\n", mean(pgm_times[2:end]) * 1000)
# @printf("learned adaptive PCF-PGM mean time = %8.3f ms\n", mean(learned_pgm_times[2:end]) * 1000)
