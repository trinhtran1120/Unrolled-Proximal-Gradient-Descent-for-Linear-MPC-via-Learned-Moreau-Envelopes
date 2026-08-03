using Test
using LinearAlgebra
using Ipopt

include(joinpath(@__DIR__, "..", "src", "mpc", "pgm.jl"))
include(joinpath(@__DIR__, "..", "src", "mpc", "solver.jl"))

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

    cost, grad = grad_cost(problem, problem.x0, U)

    @test cost isa Real
    @test isfinite(cost)
    @test size(grad) == size(U)
    @test all(isfinite, grad)

    W = fill(10.0, problem.nu, problem.N)
    project = projection(problem, problem.x0, W)
    U_proj = project(problem.x0, W)
    X_proj = rollout(problem, problem.x0, U_proj)

    @test all(problem.umin .- 1e-6 .<= U_proj)
    @test all(U_proj .<= problem.umax .+ 1e-6)
    @test all(problem.xmin .- 1e-6 .<= X_proj)
    @test all(X_proj .<= problem.xmax .+ 1e-6)

    result = PGM(problem, problem.x0; rho=0.001, max_iter=1000, tol=5e-4)

    @test size(result.U) == (problem.nu, problem.N)
    @test size(result.X) == (problem.nx, problem.N + 1)
    @test !isempty(result.objective_history)
    @test all(isfinite, result.U)
    @test all(isfinite, result.X)


    solve = mpc_solver("Ipopt", problem, 1e-6)
    X_opt, U_opt, _solve_time, J_opt = solve(problem.x0)

    result = PGM(problem, problem.x0; rho=0.001, max_iter=1000, tol=5e-4)
    J_pgm, _ = grad_cost(problem, problem.x0, result.U)
    @test abs(J_pgm - J_opt) / abs(J_opt) <= 1e-2
    
end
