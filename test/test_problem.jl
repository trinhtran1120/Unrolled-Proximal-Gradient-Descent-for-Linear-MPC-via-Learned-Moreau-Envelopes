using Test
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "mpc", "problem.jl"))

@testset "linear MPC problem" begin
    problem = mpc_problem()
    @test size(problem.A) == (problem.nx, problem.nx)
    @test size(problem.B) == (problem.nx, problem.nu)
    @test size(problem.Q) == (problem.nx, problem.nx)
    @test size(problem.R) == (problem.nu, problem.nu)

    @test problem.xmin < problem.xmax
    @test problem.umin < problem.umax

    u0 = zeros(problem.nu)
    cost_value = problem.cost_func(problem.x0, u0)

    @test cost_value isa Real
end