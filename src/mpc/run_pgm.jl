using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "pgm.jl"))
include(joinpath(@__DIR__, "solver.jl"))

function max_constraint_violation(problem::LinearMPC, U::Matrix{Float64}, X::Matrix{Float64})
    return max(
        0.0,
        maximum(as_column(problem.umin) .- U),
        maximum(U .- as_column(problem.umax)),
        maximum(as_column(problem.xmin) .- X),
        maximum(X .- as_column(problem.xmax)),
    )
end



problem = mpc_problem()
x0 = problem.x0

solver_name = "OSQP"
tol = 1e-6
rho = 0.1
gamma = 0.01
max_iter = 100

solve_mpc = mpc_solver(solver_name, problem, tol)
solve_pgm = PGM_solver(
    problem;
    rho = rho,
    gamma = gamma,
    adaptive = true,
    max_iter = max_iter,
    tol = 1e-2,
)


println("---------------- $solver_name ----------------")
opt_X, opt_U, opt_time, J_opt = solve_mpc(x0; verbose = false)

@printf("objective = %10.6f\n", J_opt)
@printf("solve time = %8.3f ms\n", opt_time * 1000)
@printf("max constraint violation = %8.4e\n\n", max_constraint_violation(problem, opt_U, opt_X))

println("---------------- PGM ----------------")
pgm_U, pgm_X, _ = solve_pgm(x0; data = nothing, verbose = true)
J_pgm, _ = evaluate_cost_gradient(problem, x0, pgm_U)

@printf("objective = %10.6f\n", J_pgm)
@printf("max constraint violation = %8.4e\n", max_constraint_violation(problem, pgm_U, pgm_X))
@printf("max |solver U - PGM U| = %8.4e\n", maximum(abs.(opt_U - pgm_U)))
@printf("max |solver X - PGM X| = %8.4e\n", maximum(abs.(opt_X - pgm_X)))

@printf("Optimality gap = %.5e %%\n", abs(J_opt - J_pgm) / abs(J_opt) * 100)
