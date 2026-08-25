using Test
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "mpc", "learned_pgm.jl"))

@testset "Learned proximal gradient method" begin
    problem = mpc_problem()
    input_dim = problem.nu * problem.N
    slack_dim = 2 * problem.nx * problem.N
    learned_model = ProjectionMLP(
        input_dim,
        problem.nx,
        input_dim + slack_dim,
        [zeros(input_dim + slack_dim, input_dim + problem.nx)],
        [zeros(input_dim + slack_dim)],
        zeros(input_dim),
        ones(input_dim),
        zeros(problem.nx),
        ones(problem.nx),
        vcat(problem.B_ro, -problem.B_ro),
        vcat(fill(problem.xmax, problem.nx * problem.N), fill(-problem.xmin, problem.nx * problem.N)),
        vcat(-problem.A_ro, problem.A_ro),
        0.1,
    )

    @test learned_model.input_dim == problem.nu * problem.N
    @test learned_model.parameter_dim == problem.nx

    U = zeros(problem.nu, problem.N)
    V_tilde, s_tilde = projection_mlp_corrected(learned_model, vec(U), problem.x0)
    moreau_value = learned_moreau_value(learned_model, U, problem.x0)
    moreau_grad = learned_moreau_gradient(learned_model, problem, U, problem.x0)
    learned_value, learned_grad = learned_objective_gradient(learned_model, problem, problem.x0, U)
    cost_value, cost_grad = evaluate_cost_gradient(problem, problem.x0, U)

    @test length(V_tilde) == input_dim
    @test length(s_tilde) == slack_dim
    @test learned_model.Gx * V_tilde + s_tilde ≈ learned_model.b_offset + learned_model.b_theta * problem.x0
    @test isfinite(moreau_value)
    @test size(moreau_grad) == size(U)
    @test all(isfinite, moreau_grad)
    @test learned_value ≈ cost_value + moreau_value
    @test learned_grad ≈ cost_grad .+ moreau_grad
    @test_throws DimensionMismatch learned_moreau_gradient(
        learned_model,
        problem,
        zeros(problem.nu, problem.N + 1),
        problem.x0,
    )
    @test_throws DimensionMismatch learned_moreau_gradient(
        learned_model,
        problem,
        U,
        [problem.x0; 0.0],
    )

    solve = learned_PGM(
        learned_model,
        problem;
        gamma = 0.01,
        max_iter = 3,
        tol = 0.0,
    )
    result = solve(problem.x0)

    @test size(result.U) == (problem.nu, problem.N)
    @test size(result.X) == (problem.nx, problem.N + 1)
    @test length(result.objective_history) == 3
    @test length(result.gamma_history) == length(result.objective_history)
    @test length(result.backtrack_history) == length(result.objective_history)
    @test length(result.residual_history) == length(result.objective_history)
    @test all(result.gamma_history .> 0)
    @test all(result.backtrack_history .>= 0)
    @test result.solve_time >= 0.0
    @test all(isfinite, result.U)
    @test all(isfinite, result.X)
    @test all(problem.umin .- 1e-6 .<= result.U)
    @test all(result.U .<= problem.umax .+ 1e-6)

    for k in 1:problem.N
        @test result.X[:, k + 1] ≈ problem.A * result.X[:, k] + problem.B * result.U[:, k] atol = 1e-5
    end
end
