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

const SOLVER_NAME = "OSQP"
const TOL = 1e-3
const PGM_RHO = 0.001
const PGM_MAX_ITER = 1000
const LEARNED_PGM_MAX_ITER = 1000
const BENCHMARK_SAMPLES = 10

model_path = joinpath(@__DIR__, "..", "..", "model", "linear-mpc-icnn-rho=0.json")

mpc_data = mpc_problem()
x0 = mpc_data.x0

solve_mpc = mpc_solver(SOLVER_NAME, mpc_data, TOL)
learned_model = load_learned_icnn(model_path)

println("================ Open-loop linear MPC benchmark ================")
println("initial state = $x0")
println("horizon N = $(mpc_data.N)")
println("solver = $SOLVER_NAME")
println("PGM rho = $PGM_RHO")
println("learned model = $model_path")
println()

println("---------------- $SOLVER_NAME ----------------")
opt_X, opt_U, solver_time, J_opt = solve_mpc(x0; verbose = true)
@printf("objective = %10.6f\n", J_opt)
@printf("solve time = %8.3f ms\n\n", solver_time * 1000)

println("---------------- exact PGM ----------------")
pgm_start = time()
pgm_sol = PGM(
    mpc_data,
    x0;
    rho = PGM_RHO,
    max_iter = PGM_MAX_ITER,
    tol = TOL,
    verbose = true,
)
pgm_time = time() - pgm_start
J_pgm, _ = grad_cost(mpc_data, x0, pgm_sol.U)
@printf("objective = %10.6f\n", J_pgm)
# @printf("solve time = %8.3f ms\n", pgm_time * 1000)
@printf("iterations = %d\n", length(pgm_sol.objective_history))
@printf("relative objective gap = %8.4f%%\n", abs(J_opt - J_pgm) / abs(J_opt) * 100)
@printf("max |solver U - PGM U| = %8.4e\n", maximum(abs.(opt_U - pgm_sol.U)))
@printf("max |solver X - PGM X| = %8.4e\n\n", maximum(abs.(opt_X - pgm_sol.X)))

println("---------------- learned PGM ----------------")
learned_sol = learned_PGM(
    learned_model,
    mpc_data;
    x0 = x0,
    rho = learned_model.rho,
    max_iter = LEARNED_PGM_MAX_ITER,
    tol = TOL,
    verbose = true,
)
J_learned, _ = grad_cost(mpc_data, x0, learned_sol.U)
@printf("objective = %10.6f\n", J_learned)
# @printf("solve time = %8.3f ms\n", learned_sol.solve_time * 1000)
@printf("iterations = %d\n", length(learned_sol.objective_history))
@printf("relative objective gap = %8.4f%%\n", abs(J_opt - J_learned) / abs(J_opt) * 100)
@printf("max |solver U - learned PGM U| = %8.4e\n", maximum(abs.(opt_U - learned_sol.U)))
@printf("max |solver X - learned PGM X| = %8.4e\n\n", maximum(abs.(opt_X - learned_sol.X)))

println("---------------- repeated timing ----------------")
solver_times = zeros(Float64, BENCHMARK_SAMPLES)
pgm_times = zeros(Float64, BENCHMARK_SAMPLES)
learned_pgm_times = zeros(Float64, BENCHMARK_SAMPLES)

for sample in 1:BENCHMARK_SAMPLES
    _, _, solver_times[sample], _ = solve_mpc(x0)

    start_time = time()
    PGM(
        mpc_data,
        x0;
        rho = PGM_RHO,
        max_iter = PGM_MAX_ITER,
        tol = TOL,
    )
    pgm_times[sample] = time() - start_time

    learned_result = learned_PGM(
        learned_model,
        mpc_data;
        x0 = x0,
        rho = learned_model.rho,
        max_iter = LEARNED_PGM_MAX_ITER,
        tol = TOL,
    )
    learned_pgm_times[sample] = learned_result.solve_time
end

@printf("%s mean time = %8.3f ms\n", SOLVER_NAME, mean(solver_times[2:end]) * 1000)
@printf("exact PGM mean time = %8.3f ms\n", mean(pgm_times[2:end]) * 1000)
@printf("learned PGM mean time = %8.3f ms\n", mean(learned_pgm_times[2:end]) * 1000)
