# Benchmark open-loop linear MPC methods.
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using JuMP
using LinearAlgebra
using Printf
using Statistics

using Ipopt
using Gurobi
using OSQP
using ProximalOperators

import MathOptInterface as MOI

include(joinpath(@__DIR__, "..", "mpc", "learned_pgm.jl"))
include(joinpath(@__DIR__, "..", "mpc", "solver.jl"))

const SOLVER_NAME = "Gurobi"
const TOL = 1e-3
const PGM_MAX_ITER = 1000
const LEARNED_PGM_MAX_ITER = 1000
const BENCHMARK_SAMPLES = 10
const RHO_TAG = "0.1"
const LEARNED_DIAGNOSTICS = false
const LEARNED_PGM_ADAPTIVE = true
const LEARNED_PGM_ACCELERATION = true
const LEARNED_PGM_INCREASE_GAMMA = 1.0
const LEARNED_PGM_REDUCE_GAMMA = 0.5
const LEARNED_PGM_MINIMUM_GAMMA = 1e-6
const LEARNED_PGM_MAX_BACKTRACKS = 50
const LEARNED_PGM_SYNC_EVERY = 5
const LEARNED_EXACT_PROX_MAX_SWEEPS = 100

model_path = joinpath(
    @__DIR__,
    "..",
    "..",
    "model",
    "linear-mpc-icnn-rho=$(RHO_TAG)-distance.json",
)

mpc_data = mpc_problem()
x0 = mpc_data.x0

solve_mpc = mpc_solver(SOLVER_NAME, mpc_data, TOL)
solve_pgm = PGM_solver(mpc_data; max_iter = PGM_MAX_ITER, tol = TOL)
learned_model = load_learned_icnn(model_path)
learned_pgm_rho = learned_model.rho_initial

function max_constraint_violation(problem, U, X)
    return max(
        0.0,
        maximum(problem.umin .- U),
        maximum(U .- problem.umax),
        maximum(problem.xmin .- X),
        maximum(X .- problem.xmax),
    )
end

println("================ Open-loop linear MPC benchmark ================")
println("initial state = $x0")
println("horizon N = $(mpc_data.N)")
println("solver = $SOLVER_NAME")
println("exact PGM step = adaptive")
println("learned PGM rho = $learned_pgm_rho")
println("learned PGM trained median rho = $(learned_model.rho)")
println("learned PGM rho mode = $(LEARNED_PGM_ADAPTIVE ? "adaptive backtracking" : "fixed")")
println("learned PGM acceleration = $LEARNED_PGM_ACCELERATION")
println("learned PGM increase gamma = $LEARNED_PGM_INCREASE_GAMMA")
println("learned PGM reduce gamma = $LEARNED_PGM_REDUCE_GAMMA")
println("learned PGM minimum gamma = $LEARNED_PGM_MINIMUM_GAMMA")
println("learned PGM max backtracks = $LEARNED_PGM_MAX_BACKTRACKS")
println("learned PGM correction = FBE-gated learned proposal")
println("learned PGM exact prox = active-set, OSQP-free")
println("learned proposal prox max sweeps = $LEARNED_EXACT_PROX_MAX_SWEEPS")
println("learned PGM envelope = normalized distance")
println("learned PGM depth = $LEARNED_PGM_MAX_ITER")
println("learned PGM diagnostics = $LEARNED_DIAGNOSTICS")
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
@printf("relative objective gap = %8.4f%%\n", abs(J_opt - J_pgm) / abs(J_opt) * 100)
@printf("max constraint violation = %8.4e\n", max_constraint_violation(mpc_data, pgm_U, pgm_X))
@printf("max |solver U - PGM U| = %8.4e\n", maximum(abs.(opt_U - pgm_U)))
@printf("max |solver X - PGM X| = %8.4e\n\n", maximum(abs.(opt_X - pgm_X)))

println("---------------- learned PGM ----------------")
learned_sol = learned_PGM(
    learned_model,
    mpc_data;
    x0 = x0,
    rho = learned_pgm_rho,
    adaptive = LEARNED_PGM_ADAPTIVE,
    acceleration = LEARNED_PGM_ACCELERATION,
    minimum_gamma = LEARNED_PGM_MINIMUM_GAMMA,
    reduce_gamma = LEARNED_PGM_REDUCE_GAMMA,
    increase_gamma = LEARNED_PGM_INCREASE_GAMMA,
    max_backtracks = LEARNED_PGM_MAX_BACKTRACKS,
    sync_every = LEARNED_PGM_SYNC_EVERY,
    learned_exact_prox_max_sweeps = LEARNED_EXACT_PROX_MAX_SWEEPS,
    max_iter = LEARNED_PGM_MAX_ITER,
    tol = TOL,
    diagnostics = LEARNED_DIAGNOSTICS,
    verbose = true,
)
J_learned, _ = grad_cost(mpc_data, x0, learned_sol.U)
@printf("objective = %10.6f\n", J_learned)
@printf("solve time = %8.3f ms\n", learned_sol.solve_time * 1000)
@printf("iterations = %d\n", length(learned_sol.objective_history))
@printf("final residual = %8.4e\n", learned_sol.residual_history[end])
@printf("final rho = %8.4e\n", learned_sol.gamma_history[end])
@printf("total backtracks = %d\n", sum(learned_sol.backtrack_history))
@printf("final projection norm = %8.4e\n", learned_sol.final_projection_norm)
@printf("final soft violation = %8.4e\n", learned_sol.final_soft_violation)
@printf("relative objective gap = %8.4f%%\n", abs(J_opt - J_learned) / abs(J_opt) * 100)
@printf("max constraint violation = %8.4e\n", max_constraint_violation(mpc_data, learned_sol.U, learned_sol.X))
@printf("max |solver U - learned PGM U| = %8.4e\n", maximum(abs.(opt_U - learned_sol.U)))
@printf("max |solver X - learned PGM X| = %8.4e\n\n", maximum(abs.(opt_X - learned_sol.X)))

println("---------------- repeated timing ----------------")
solver_times = zeros(Float64, BENCHMARK_SAMPLES)
pgm_times = zeros(Float64, BENCHMARK_SAMPLES)
learned_pgm_times = zeros(Float64, BENCHMARK_SAMPLES)

for sample in 1:BENCHMARK_SAMPLES
    _, _, solver_times[sample], _ = solve_mpc(x0)

    pgm_times[sample] = @elapsed solve_pgm(x0)

    learned_result = learned_PGM(
        learned_model,
        mpc_data;
        x0 = x0,
        rho = learned_pgm_rho,
        adaptive = LEARNED_PGM_ADAPTIVE,
        acceleration = LEARNED_PGM_ACCELERATION,
        minimum_gamma = LEARNED_PGM_MINIMUM_GAMMA,
        reduce_gamma = LEARNED_PGM_REDUCE_GAMMA,
        increase_gamma = LEARNED_PGM_INCREASE_GAMMA,
        max_backtracks = LEARNED_PGM_MAX_BACKTRACKS,
        sync_every = LEARNED_PGM_SYNC_EVERY,
        learned_exact_prox_max_sweeps = LEARNED_EXACT_PROX_MAX_SWEEPS,
        max_iter = LEARNED_PGM_MAX_ITER,
        tol = TOL,
    )
    learned_pgm_times[sample] = learned_result.solve_time
end

@printf("%s mean time = %8.3f ms\n", SOLVER_NAME, mean(solver_times[2:end]) * 1000)
@printf("exact PGM mean time = %8.3f ms\n", mean(pgm_times[2:end]) * 1000)
@printf("learned PGM mean time = %8.3f ms\n", mean(learned_pgm_times[2:end]) * 1000)
