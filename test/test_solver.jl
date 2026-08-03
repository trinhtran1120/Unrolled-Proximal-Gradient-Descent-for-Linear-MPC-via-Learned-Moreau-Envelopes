using Test
using LinearAlgebra
using OSQP

include("../src/mpc/solver.jl")

@testset "Baseline solver for linear MPC" begin
    problem = mpc_problem()
    solve = mpc_solver("OSQP", problem, 1e-6)
    X, U, J = solve(problem.x0)

    @test size(X) == (problem.nx, problem.N + 1)
    @test size(U) == (problem.nu, problem.N)
    @test J isa Real
    @test isfinite(J)

    for k in 1:problem.N
        @test X[:, k+1] ≈ problem.A * X[:, k] + problem.B * U[:, k] atol=1e-5
    end

    @test all(problem.xmin .<= X)
    @test all(X .<= problem.xmax)
    @test all(problem.umin .- 1e-5 .<= U)
    @test all(U .<= problem.umax .+ 1e-5)
end