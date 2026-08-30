using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "pgm.jl"))
include(joinpath(@__DIR__, "solver.jl"))

function max_constraint_violation(problem::LinearMPC, U::Matrix{Float64}, X::Matrix{Float64})
    return max(
        0.0,
        maximum(problem.umin .- U),
        maximum(U .- problem.umax),
        maximum(problem.xmin .- X),
        maximum(X .- problem.xmax),
    )
end

problem = mpc_problem()
x0 = problem.x0

solver_name = "OSQP"
tol = 1e-6
mu = 1.
gamma = 0.01
relaxation = 1.0
line_search = true
minimum_gamma = 1e-6
reduce_gamma = 0.5
increase_gamma = 1.0
max_iter = 1000
pgm_tol = 1e-4

solve_mpc = mpc_solver(solver_name, problem, tol)
solve_pgm = PGM_solver(
    problem;
    mu = mu,
    gamma = gamma,
    relaxation = relaxation,
    adaptive = line_search,
    minimum_gamma = minimum_gamma,
    reduce_gamma = reduce_gamma,
    increase_gamma = increase_gamma,
    max_iter = max_iter,
    tol = pgm_tol,
)


println("---------------- $solver_name ----------------")
opt_X, opt_U, opt_time, J_opt = solve_mpc(x0; verbose = false)

@printf("objective = %10.6f\n", J_opt)
@printf("solve time = %8.3f ms\n", opt_time * 1000)
@printf("max constraint violation = %8.4e\n\n", max_constraint_violation(problem, opt_U, opt_X))

println("---------------- exact TOS PGM ----------------")
@printf("Moreau mu = %.4e\n", mu)
@printf("initial splitting gamma = %.4e\n", gamma)
@printf("relaxation = %.4e\n", relaxation)
@printf("Line search = %s\n", line_search ? "on" : "off")
if line_search
    @printf("minimum step gamma = %.4e\n", minimum_gamma)
    @printf("step gamma reduce factor = %.4e\n", reduce_gamma)
    @printf("step gamma increase factor = %.4e\n", increase_gamma)
end
@printf("max iterations = %d\n", max_iter)
@printf("tolerance = %.4e\n", pgm_tol)

pgm_data = Dict{String,Any}()
pgm_time = @elapsed pgm_U, pgm_X, J_pgm_smooth = solve_pgm(x0; data = pgm_data, verbose = true)
J_pgm, _ = evaluate_cost_gradient(problem, x0, pgm_U)
projection_samples = length(get(pgm_data, "U_query", Vector{Vector{Float64}}()))
objective_gap_percent = abs(J_opt - J_pgm) / max(abs(J_opt), eps(Float64)) * 100

@printf("trajectory objective = %10.6f\n", J_pgm)
@printf("smoothed objective = %10.6f\n", J_pgm_smooth)
@printf("solve time = %8.3f ms\n", pgm_time * 1000)
@printf("recorded projection samples = %d\n", projection_samples)
@printf("max constraint violation = %8.4e\n", max_constraint_violation(problem, pgm_U, pgm_X))
@printf("max |solver U - PGM U| = %8.4e\n", maximum(abs.(opt_U - pgm_U)))
@printf("max |solver X - PGM X| = %8.4e\n", maximum(abs.(opt_X - pgm_X)))
@printf("Objective gap = %.5f %%\n", objective_gap_percent)
