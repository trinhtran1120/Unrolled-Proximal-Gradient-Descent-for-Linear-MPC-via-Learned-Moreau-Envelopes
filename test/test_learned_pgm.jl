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
        "linear-mpc-icnn-rho=0.1-distance.json",
    )
    learned_model = load_learned_icnn(model_path)

    expected_input_dim = problem.nu * problem.N + problem.nx
    @test length(learned_model.a) == expected_input_dim

    U = zeros(problem.nu, problem.N)
    W = U
    local_gradients = (
        gradient_struct(icnn_from_learned(learned_model), 1, length(learned_model.a)),
    )
    distance_grad = learned_squared_distance_gradient(
        learned_model,
        local_gradients,
        problem,
        W,
        problem.x0,
    )

    @test size(distance_grad) == size(U)
    @test all(isfinite, distance_grad)
    @test_throws ArgumentError learned_moreau_gradient(
        learned_model,
        local_gradients,
        problem,
        W,
        problem.x0,
        0.0,
    )

    grad_at_rho = learned_moreau_gradient(
        learned_model,
        local_gradients,
        problem,
        W,
        problem.x0,
        learned_model.rho,
    )
    grad_at_twice_rho = learned_moreau_gradient(
        learned_model,
        local_gradients,
        problem,
        W,
        problem.x0,
        2 * learned_model.rho,
    )
    @test grad_at_rho ≈ 2 * grad_at_twice_rho

    @test distance_grad ≈ learned_model.rho * grad_at_rho

    projection_correction = learned_projection_correction(
        learned_model,
        local_gradients,
        problem,
        W,
        problem.x0,
        2 * learned_model.rho,
    )
    @test projection_correction ≈ distance_grad

    step_size = inv(opnorm(single_shooting_cost(problem, problem.x0).H, 2))
    _, initial_grad = grad_cost(problem, problem.x0, U)
    first_W = U - step_size * initial_grad
    expected_first_U = first_W - learned_projection_correction(
        learned_model,
        local_gradients,
        problem,
        first_W,
        problem.x0,
        step_size,
    )
    expected_first_vec = similar(vec(expected_first_U))
    corrected_prox!(
        expected_first_vec,
        constraint(problem, problem.x0; solver = :osqp),
        vec(expected_first_U),
        step_size;
        warm_start = vec(expected_first_U),
    )
    expected_first_U = reshape(expected_first_vec, problem.nu, problem.N)
    first_layer = learned_PGM(
        learned_model,
        problem;
        x0 = problem.x0,
        rho = step_size,
        adaptive = false,
        max_iter = 1,
        tol = 0.0,
    )
    @test first_layer.U ≈ expected_first_U
    @test first_layer.gamma_history == [step_size]
    @test first_layer.backtrack_history == [0]

    result = learned_PGM(
        learned_model,
        problem;
        x0 = problem.x0,
        max_iter = 20,
        tol = 1e-3,
    )

    @test size(result.U) == (problem.nu, problem.N)
    @test size(result.X) == (problem.nx, problem.N + 1)
    @test !isempty(result.objective_history)
    @test length(result.objective_history) <= 20
    @test length(result.gamma_history) == length(result.objective_history)
    @test length(result.backtrack_history) == length(result.objective_history)
    @test length(result.residual_history) == length(result.objective_history)
    @test all(result.gamma_history .>= 1e-6)
    @test result.gamma_history[1] <= learned_model.rho
    @test all(result.backtrack_history .>= 0)
    @test all(
        result.gamma_history[2:end] .<=
        result.gamma_history[1:end-1] .* 1.05 .+ sqrt(eps(Float64))
    )
    @test result.solve_time >= 0.0
    @test all(isfinite, result.U)
    @test all(isfinite, result.X)

    for k in 1:problem.N
        @test result.X[:, k + 1] ≈ problem.A * result.X[:, k] + problem.B * result.U[:, k] atol = 1e-5
    end

    max_violation = max(
        0.0,
        maximum(problem.umin .- result.U),
        maximum(result.U .- problem.umax),
        maximum(problem.xmin .- result.X),
        maximum(result.X .- problem.xmax),
    )
    @test isfinite(max_violation)

    final_cost, _ = grad_cost(problem, problem.x0, result.U)
    @test isfinite(final_cost)
end
