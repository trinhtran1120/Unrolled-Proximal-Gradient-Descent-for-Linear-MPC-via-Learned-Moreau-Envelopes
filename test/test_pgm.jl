using Test
using LinearAlgebra
using Ipopt

include(joinpath(@__DIR__, "..", "src", "mpc", "pgm.jl"))
include(joinpath(@__DIR__, "..", "src", "mpc", "solver.jl"))

function trajectory_cost(problem::LinearMPC, X::Matrix{Float64}, U::Matrix{Float64})
    return sum(problem.cost_func(X[:, k], U[:, k]) for k in 1:problem.N)
end

@testset "Test proximal gradient method" begin
    problem = mpc_problem()
    
    @test size(problem.A_ro) == (problem.N * problem.nx, problem.nx)
    @test size(problem.B_ro) == (problem.N * problem.nx, problem.N * problem.nu)

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
    project_state = constraint(problem; solver = :osqp)
    u_proj, env = project_state(problem.x0, vec(W))
    U_proj = reshape(u_proj, problem.nu, problem.N)
    X_proj = rollout(problem, problem.x0, U_proj)

    @test env ≈ 0.5 * norm(vec(W) - u_proj)^2
    @test all(problem.xmin .- 1e-6 .<= X_proj)
    @test all(X_proj .<= problem.xmax .+ 1e-6)
    @test any(U_proj .> problem.umax .+ 1e-6)

    solve_pgm = PGM_solver(problem; rho=0.001, max_iter=1000, tol=5e-4)
    U_pgm, X_pgm, J_pgm = solve_pgm(problem.x0)

    @test size(U_pgm) == (problem.nu, problem.N)
    @test size(X_pgm) == (problem.nx, problem.N + 1)
    @test isfinite(J_pgm)
    @test all(isfinite, U_pgm)
    @test all(isfinite, X_pgm)
    @test all(problem.umin .- 1e-6 .<= U_pgm)
    @test all(U_pgm .<= problem.umax .+ 1e-6)

    fixed_data = Dict("input" => Vector{Float64}[], "parameter" => Vector{Float64}[], "proj" => Vector{Float64}[], "env" => Float64[], "grad" => Vector{Float64}[], "gamma" => Float64[])
    fixed_rho = 0.001
    fixed_gamma = 0.001
    solve_fixed_pgm = PGM_solver(problem; rho=fixed_rho, gamma=fixed_gamma, adaptive=false, max_iter=3, tol=0.0)
    solve_fixed_pgm(problem.x0; data=fixed_data)
    @test !isempty(fixed_data["gamma"])
    @test all(fixed_data["gamma"] .≈ fixed_gamma)
    @test length(fixed_data["input"]) == length(fixed_data["proj"]) == length(fixed_data["env"]) == length(fixed_data["grad"])

    adaptive_data = Dict("input" => Vector{Float64}[], "parameter" => Vector{Float64}[], "proj" => Vector{Float64}[], "env" => Float64[], "grad" => Vector{Float64}[], "gamma" => Float64[])
    solve_adaptive_pgm = PGM_solver(problem; rho=fixed_rho, adaptive=true, max_iter=3, tol=0.0)
    solve_adaptive_pgm(problem.x0; data=adaptive_data)
    @test !isempty(adaptive_data["gamma"])
    @test all(adaptive_data["gamma"] .> 0)
    @test length(adaptive_data["input"]) >= length(adaptive_data["gamma"])

    solve = mpc_solver("Ipopt", problem, 1e-6)
    X_opt, U_opt, _solve_time, J_opt = solve(problem.x0)

    U_pgm, X_pgm, _ = solve_pgm(problem.x0)
    J_pgm = trajectory_cost(problem, X_pgm, U_pgm)
    @test abs(J_pgm - J_opt) / abs(J_opt) <= 1e-2
    
end
