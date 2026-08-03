using Test
using LinearAlgebra
using OSQP

include(joinpath(@__DIR__, "..", "src", "mpc", "learned_pgm.jl"))

@testset "Learned proximal gradient method" begin
    problem = mpc_problem()
    model_path = joinpath(
        @__DIR__,
        "..",
        "model",
        "linear-mpc-icnn-rho=0.json",
    )
    learned_model = load_learned_icnn(model_path)

    expected_input_dim = problem.nu * problem.N + problem.nx

    if length(learned_model.a) != expected_input_dim
        U = zeros(problem.nu, problem.N)
        local_gradients = (
            gradient_struct(icnn_from_learned(learned_model), 1, length(learned_model.a)),
        )

        @test_throws DimensionMismatch learned_moreau_gradient(
            learned_model,
            local_gradients,
            problem,
            U,
            problem.x0,
        )
        @test_throws DimensionMismatch learned_PGM(
            learned_model,
            problem;
            x0 = problem.x0,
        )
    else
        @test length(learned_model.a) == expected_input_dim

        U = zeros(problem.nu, problem.N)
        W = U
        local_gradients = (
            gradient_struct(icnn_from_learned(learned_model), 1, length(learned_model.a)),
        )
        learned_grad = learned_moreau_gradient(
            learned_model,
            local_gradients,
            problem,
            W,
            problem.x0,
        )

        @test size(learned_grad) == size(U)
        @test all(isfinite, learned_grad)

        result = learned_PGM(
            learned_model,
            problem;
            x0 = problem.x0,
            rho = learned_model.rho,
            max_iter = 1000,
            tol = 1e-3,
        )

        @test size(result.U) == (problem.nu, problem.N)
        @test size(result.X) == (problem.nx, problem.N + 1)
        @test !isempty(result.objective_history)
        @test length(result.objective_history) <= 1000
        @test result.solve_time >= 0.0
        @test all(isfinite, result.U)
        @test all(isfinite, result.X)

        for k in 1:problem.N
            @test result.X[:, k + 1] ≈ problem.A * result.X[:, k] + problem.B * result.U[:, k] atol = 1e-5
        end

        for k in 1:problem.N
            @test all(problem.umin .- 1e-5 .<= result.U[:, k])
            @test all(result.U[:, k] .<= problem.umax .+ 1e-5)
            @test all(problem.xmin .- 1e-5 .<= result.X[:, k + 1])
            @test all(result.X[:, k + 1] .<= problem.xmax .+ 1e-5)
        end

        final_cost, _ = grad_cost(problem, problem.x0, result.U)
        @test isfinite(final_cost)
    end
end
