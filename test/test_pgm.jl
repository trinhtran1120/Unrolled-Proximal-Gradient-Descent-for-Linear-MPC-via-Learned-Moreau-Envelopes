using Test
using LinearAlgebra
using Ipopt

include(joinpath(@__DIR__, "..", "src", "mpc", "pgm_std.jl"))
include(joinpath(@__DIR__, "..", "src", "mpc", "solver.jl"))

function trajectory_cost(problem::LinearMPC, X::Matrix{Float64}, U::Matrix{Float64})
    return sum(problem.cost_func(X[:, k], U[:, k]) for k in 1:problem.N)
end

@testset "Test proximal gradient method" begin
    problem = mpc_problem()
    A_ss, B_ss, _ = converted_matrices(problem)
    
    @test size(A_ss) == (problem.N * problem.nx, problem.nx)
    @test size(B_ss) == (problem.N * problem.nx, problem.N * problem.nu)

    U = zeros(problem.nu, problem.N)
    X = rollout(problem, problem.x0, U)

    @test size(X) == (problem.nx, problem.N + 1)
    @test X[:, 1] ≈ problem.x0

    for k in 1:problem.N
        @test X[:, k+1] ≈ problem.A * X[:, k] + problem.B * U[:, k]
    end

    f = single_shooting_cost(problem, problem.x0)
    zero_input_cost = trajectory_cost(problem, X, U)
    U_trial = fill(0.2, problem.nu, problem.N)
    X_trial = rollout(problem, problem.x0, U_trial)
    @test f(vec(U_trial)) + zero_input_cost ≈ trajectory_cost(problem, X_trial, U_trial)

    W = fill(10.0, problem.nu, problem.N)
    feasible_set = condensed_constraints(problem, problem.x0; solver = :osqp)
    u_proj, _ = prox(feasible_set, vec(W))
    U_proj = reshape(u_proj, problem.nu, problem.N)
    X_proj = rollout(problem, problem.x0, U_proj)

    @test all(problem.umin .- 1e-6 .<= U_proj)
    @test all(U_proj .<= problem.umax .+ 1e-6)
    @test all(problem.xmin .- 1e-6 .<= X_proj)
    @test all(X_proj .<= problem.xmax .+ 1e-6)

    result = PGM(problem, problem.x0; rho=0.001, max_iter=1000, tol=5e-4)

    @test size(result.U) == (problem.nu, problem.N)
    @test size(result.X) == (problem.nx, problem.N + 1)
    @test isfinite(result.objective_value)
    @test all(isfinite, result.U)
    @test all(isfinite, result.X)
    @test result.solve_time >= 0.0


    solve = mpc_solver("Ipopt", problem, 1e-6)
    X_opt, U_opt, _solve_time, J_opt = solve(problem.x0)

    result = PGM(problem, problem.x0; rho=0.001, max_iter=1000, tol=5e-4)
    J_pgm = trajectory_cost(problem, result.X, result.U)
    @test abs(J_pgm - J_opt) / abs(J_opt) <= 1e-2
    
end
