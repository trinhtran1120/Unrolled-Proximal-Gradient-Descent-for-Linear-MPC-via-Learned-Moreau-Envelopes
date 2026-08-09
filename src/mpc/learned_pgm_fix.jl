using LinearAlgebra

include("pgm.jl")
include(joinpath(@__DIR__, "..", "utils", "utils.jl"))

function learned_moreau_gradient(
    model::LearnedICNN,
    local_gradients::NTuple,
    system::LinearMPC,
    W::Matrix{Float64},
    x0::Vector{Float64},
    gamma::Real,
)
    expected_u_size = (system.nu, system.N)
    if size(W) != expected_u_size
        throw(DimensionMismatch("W must have size $expected_u_size, got $(size(W))"))
    end

    if length(x0) != system.nx
        throw(DimensionMismatch("x0 must have length $(system.nx), got $(length(x0))"))
    end

    base_input_dim = system.nu * system.N + system.nx
    expected_input_dim = model.target == "moreau_envelope" ? base_input_dim + 1 : base_input_dim
    if length(model.a) != expected_input_dim
        throw(DimensionMismatch(
            "learned ICNN input dimension must be $expected_input_dim " *
            "(base nu*N + nx = $(system.nu)*$(system.N) + $(system.nx)), got $(length(model.a))",
        ))
    end

    control_dim = system.nu * system.N
    input = vcat(vec(W), x0)
    full_gradient = learned_moreau_full_gradient(model, local_gradients, input, gamma)
    return reshape(full_gradient[1:control_dim], system.nu, system.N)
end

function learned_squared_distance_gradient(
    model::LearnedICNN,
    local_gradients::NTuple,
    system::LinearMPC,
    W::Matrix{Float64},
    x0::Vector{Float64},
)
    expected_u_size = (system.nu, system.N)
    if size(W) != expected_u_size
        throw(DimensionMismatch("W must have size $expected_u_size, got $(size(W))"))
    end

    if length(x0) != system.nx
        throw(DimensionMismatch("x0 must have length $(system.nx), got $(length(x0))"))
    end

    base_input_dim = system.nu * system.N + system.nx
    expected_input_dim = model.target == "moreau_envelope" ? base_input_dim + 1 : base_input_dim
    if length(model.a) != expected_input_dim
        throw(DimensionMismatch(
            "learned ICNN input dimension must be $expected_input_dim " *
            "(base nu*N + nx = $(system.nu)*$(system.N) + $(system.nx)), got $(length(model.a))",
        ))
    end

    control_dim = system.nu * system.N
    input = vcat(vec(W), x0)
    full_gradient = learned_squared_distance_full_gradient(model, local_gradients, input)
    return reshape(full_gradient[1:control_dim], system.nu, system.N)
end

function learned_projection_correction(
    model::LearnedICNN,
    local_gradients::NTuple,
    system::LinearMPC,
    W::Matrix{Float64},
    x0::Vector{Float64},
    rho::Real,
)
    if model.target == "half_squared_distance"
        return learned_squared_distance_gradient(
            model,
            local_gradients,
            system,
            W,
            x0,
        )
    end

    moreau_grad = learned_moreau_gradient(
        model,
        local_gradients,
        system,
        W,
        x0,
        rho,
    )
    return rho .* moreau_grad
end

function learned_forward_backward_candidate!(
    W::Matrix{Float64},
    U_next::Matrix{Float64},
    model::LearnedICNN,
    local_gradients::NTuple,
    system::LinearMPC,
    U::Matrix{Float64},
    grad::Matrix{Float64},
    x0::Vector{Float64},
    rho::Real,
)
    @. W = U - rho * grad
    learned_grad = learned_moreau_gradient(
        model,
        local_gradients,
        system,
        W,
        x0,
        rho,
    )
    @. U_next = W - rho * learned_grad
    return U_next
end

function learned_backtracking_accepts(
    f::DifferentiableQuadratic,
    f_u::Real,
    grad_vec::AbstractVector,
    U::Matrix{Float64},
    U_next::Matrix{Float64},
    rho::Real,
)
    step = vec(U) - vec(U_next)
    upper = f_u - dot(grad_vec, step) + norm(step)^2 / (2 * rho)
    trial = f(vec(U_next))
    tol = 10 * eps(Float64) * (1 + abs(trial))
    return trial <= upper + tol
end

function learned_PGM(
    model::LearnedICNN,
    system::LinearMPC;
    x0::Vector{Float64} = system.x0,
    rho::Float64 = model.rho,
    max_iter::Int = 100,
    tol::Float64 = 1e-6,
    verbose = false,
)
    if !isfinite(rho) || rho <= 0
        throw(ArgumentError("rho must be finite and strictly positive, got $rho"))
    end

    U = zeros(Float64, system.nu, system.N)
    if length(x0) != system.nx
        throw(DimensionMismatch("x0 must have length $(system.nx), got $(length(x0))"))
    end

    base_input_dim = system.nu * system.N + system.nx
    expected_input_dim = model.target == "moreau_envelope" ? base_input_dim + 1 : base_input_dim
    if length(model.a) != expected_input_dim
        throw(DimensionMismatch(
            "learned ICNN input dimension must be $expected_input_dim " *
            "(base nu*N + nx = $(system.nu)*$(system.N) + $(system.nx)), got $(length(model.a))",
        ))
    end

    W = similar(U)
    U_next = similar(U)
    objective_history = Float64[]
    gamma_history = Float64[]
    backtrack_history = Int[]
    residual_history = Float64[]
    gradient_model = icnn_from_learned(model)
    local_gradients = (gradient_struct(gradient_model, 1, length(model.a)),)
    current_rho = rho
    start_time = time()
    solve_time = 0.0

    @inbounds for iter in 1:max_iter
        cost, grad = grad_cost(system, x0, U)
        push!(objective_history, cost)

        learned_forward_backward_candidate!(
            W,
            U_next,
            model,
            local_gradients,
            system,
            U,
            grad,
            x0,
            current_rho,
        )

        residual = norm(U_next - U, Inf) / current_rho
        push!(gamma_history, current_rho)
        push!(backtrack_history, 0)
        push!(residual_history, residual)
        copyto!(U, U_next)
        solve_time = time() - start_time

        if residual <= tol
            if verbose
                println("Learned PGM converged at iteration $iter")
            end
            break
        end
    end

    solve_time = time() - start_time

    return (
        U = U,
        X = rollout(system, x0, U),
        objective_history = objective_history,
        gamma_history = gamma_history,
        backtrack_history = backtrack_history,
        residual_history = residual_history,
        solve_time = solve_time,
    )
end
