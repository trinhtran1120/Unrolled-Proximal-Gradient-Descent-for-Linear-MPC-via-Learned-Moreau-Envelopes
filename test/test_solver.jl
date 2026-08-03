using Test
using LinearAlgebra
using OSQP

include("../src/mpc/solver.jl")

@testset "Baseline solver for linear MPC" begin
    problem = mpc_problem()
    solve = mpc_solver("OSQP", problem, 1e-6)
    X, U, solve_time, J = solve(problem.x0)

    @test size(X) == (problem.nx, problem.N + 1)
    @test size(U) == (problem.nu, problem.N)
    @test J isa Real
    @test isfinite(J)
    @test solve_time >= 0.0

    for k in 1:problem.N
        @test X[:, k+1] ≈ problem.A * X[:, k] + problem.B * U[:, k] atol=1e-5
    end

    for k in 1:(problem.N + 1)
        @test all(problem.xmin .- 1e-5 .<= X[:, k])
        @test all(X[:, k] .<= problem.xmax .+ 1e-5)
    end

    for k in 1:problem.N
        @test all(problem.umin .- 1e-5 .<= U[:, k])
        @test all(U[:, k] .<= problem.umax .+ 1e-5)
    end
end
