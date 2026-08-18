# Benchmark the fixed-gamma learned PCF-PGM controller only.
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using Printf

using OSQP

include(joinpath(@__DIR__, "..", "mpc", "learned_pgm_fix.jl"))
include(joinpath(@__DIR__, "..", "mpc", "solver.jl"))

const SOLVER_NAME = "OSQP"
const TOL = 1e-3
const LEARNED_PGM_MAX_ITER = 2000

model_path = joinpath(
    @__DIR__,
    "..",
    "..",
    "model",
    "linear-mpc-lpcf001-fix.json",
)

mpc_data = mpc_problem()
x0 = mpc_data.x0
solve_mpc = mpc_solver(SOLVER_NAME, mpc_data, TOL)
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

println("================ Fixed learned PCF-PGM benchmark ================")
println("initial state = $x0")
println("horizon N = $(mpc_data.N)")
println("solver = $SOLVER_NAME")
println("learned PCF-PGM depth = $LEARNED_PGM_MAX_ITER")
println("rho forward = $(learned_model.rho)")
println("rho backward = $(learned_model.rho)")
println("learned model = $model_path")
println()

println("---------------- $SOLVER_NAME ----------------")
opt_X, opt_U, solver_time, J_opt = solve_mpc(x0; verbose = false)
@printf("objective = %10.6f\n", J_opt)
@printf("solve time = %8.3f ms\n\n", solver_time * 1000)

learned_sol = learned_PGM(
    learned_model,
    mpc_data;
    x0 = x0,
    rho_forward = learned_model.rho,
    rho_backward = learned_model.rho,
    max_iter = LEARNED_PGM_MAX_ITER,
    tol = TOL,
    verbose = true,
)

J_learned, _ = grad_cost(mpc_data, x0, learned_sol.U)

println("---------------- learned fixed PCF-PGM ----------------")
@printf("objective = %10.6f\n", J_learned)
@printf("solve time = %8.3f ms\n", learned_sol.solve_time * 1000)
@printf("iterations = %d\n", length(learned_sol.objective_history))
@printf("final residual = %8.4e\n", learned_sol.residual_history[end])
@printf("max constraint violation = %8.4e\n", max_constraint_violation(mpc_data, learned_sol.U, learned_sol.X))
@printf("relative objective gap = %8.4f%%\n", relative_objective_gap(J_opt, J_learned))
@printf("max |solver U - learned PGM U| = %8.4e\n", maximum(abs.(opt_U - learned_sol.U)))
@printf("max |solver X - learned PGM X| = %8.4e\n", maximum(abs.(opt_X - learned_sol.X)))
