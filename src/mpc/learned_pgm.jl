using LinearAlgebra
using Base.Threads

include("pgm.jl")
include(joinpath(@__DIR__, "..", "utils", "utils.jl"))


function mini_batch_grad!(
    grad::AbstractVector{Float64},
    input::AbstractVector{Float64},
    projection::AbstractVector{Float64},
    mu::Float64;
    batch_size::Int = length(input),
)
    ranges = [first:min(first + batch_size - 1, length(input)) for first in 1:min(batch_size, length(input)):length(input)]
    Threads.@threads for batch in eachindex(ranges)
        range = ranges[batch]
        @inbounds @views grad[range] .= (input[range] .- projection[range]) ./ mu
    end
    return grad
end


function learned_moreau(
    model::ProjectionMLP,
    system::LinearMPC,
    U::Matrix{Float64},
    x0::Vector{Float64};
    mu::Float64 = model.mu,
    batch_size::Int = length(U),
)
    input = vec(U)
    V_tilde, _ = projection_mlp_forward(model, input, x0)
    d = input .- V_tilde
    value = dot(d, d) / (2 * mu)
    grad = similar(input)
    mini_batch_grad!(grad, input, V_tilde, mu; batch_size = batch_size)
    return value, reshape(grad, system.nu, system.N)
end


function learned_objective_gradient(
    model::ProjectionMLP,
    system::LinearMPC,
    x0::Vector{Float64},
    U::Matrix{Float64};
    mu::Float64 = model.mu,
    batch_size::Int = length(U),
)
    cost, cost_grad = evaluate_cost_gradient(system, x0, U)
    moreau_value, moreau_grad = learned_moreau(model, system, U, x0; mu = mu, batch_size = batch_size)
    return cost + moreau_value, cost_grad .+ moreau_grad
end

## Learned three-operator splitting step: exact prox_g (clip) -> learned Moreau
## gradient -> exact prox_f (quadratic-cost solve). Only the state-projection
## gradient is learned; the input box and cost prox stay exact, per the guide.
function learned_tos_step!(
    X_iter::Vector{Float64},
    Y::Vector{Float64},
    residual_vector::Vector{Float64},
    cost::DifferentiableQuadratic,
    model::ProjectionMLP,
    x0::Vector{Float64},
    Z::Vector{Float64},
    lower_U::Vector{Float64},
    upper_U::Vector{Float64},
    mu::Float64,
    gamma::Float64,
)
    X_iter .= clamp.(Z, lower_U, upper_U)
    V_tilde, _ = projection_mlp_forward(model, X_iter, x0)

    D = (X_iter .- V_tilde) ./ mu
    R = 2.0 .* X_iter .- Z .- gamma .* D
    cost_factor = cholesky(Symmetric(Matrix{Float64}(I, length(Z), length(Z)) + gamma * cost.H))
    Y .= cost_factor \ (R .- gamma .* cost.h)
    residual_vector .= Y .- X_iter

    return V_tilde
end

function backtrack_stepsize!(
    mu::Float64,
    gamma::Float64,
    X_iter::Vector{Float64},
    Y::Vector{Float64},
    residual_vector::Vector{Float64},
    cost::DifferentiableQuadratic,
    model::ProjectionMLP,
    x0::Vector{Float64},
    Z::Vector{Float64},
    lower_U::Vector{Float64},
    upper_U::Vector{Float64},
    ;
    minimum_gamma::Float64 = 1e-6,
    reduce_gamma::Float64 = 0.5,
)
    candidate_gamma = gamma

    while true
        V_tilde_x = learned_tos_step!(
            X_iter,
            Y,
            residual_vector,
            cost,
            model,
            x0,
            Z,
            lower_U,
            upper_U,
            mu,
            candidate_gamma,
        )
        V_tilde_y, _ = projection_mlp_forward(model, Y, x0)

        h_x = dot(X_iter .- V_tilde_x, X_iter .- V_tilde_x) / (2 * mu)
        h_y = dot(Y .- V_tilde_y, Y .- V_tilde_y) / (2 * mu)
        D = (X_iter .- V_tilde_x) ./ mu
        linear_model = dot(residual_vector, D)
        quadratic_model = dot(residual_vector, residual_vector) / (2.0 * candidate_gamma)
        line_search_margin = 1e-12 * max(1.0, abs(h_x), abs(h_y))

        if h_y <= h_x + linear_model + quadratic_model + line_search_margin || candidate_gamma <= minimum_gamma
            residual = norm(residual_vector) / max(1.0, norm(X_iter))
            return candidate_gamma, residual, V_tilde_x, h_x
        end

        candidate_gamma = max(minimum_gamma, candidate_gamma * reduce_gamma)
    end
end


function learned_PGM(
    model::ProjectionMLP,
    system::LinearMPC;
    mu::Float64 = model.mu,
    gamma::Float64 = 1.0,
    relaxation::Float64 = 1.0,
    minimum_gamma::Float64 = 1e-6,
    reduce_gamma::Float64 = 0.5,
    increase_gamma::Float64 = 1.05,
    max_iter::Int = 100,
    tol::Float64 = 1e-6,
)
    nu = system.nu
    N = system.N
    nU = nu * N
    Z0 = zeros(Float64, nU)
    lower_U = fill(system.umin, nU)
    upper_U = fill(system.umax, nU)

    return @inbounds function solver(x0::Vector{Float64}; verbose = false)
        cost, constant_cost = cost_cache(system, x0)
        current_gamma = gamma

        Z = copy(Z0)
        X_iter = similar(Z)
        Y = similar(Z)
        residual_vector = similar(Z)
        residual = Inf

        objective_history = Float64[]
        residual_history = Float64[]
        gamma_history = Float64[]

        start_time = time()
        for iter in 1:max_iter
            current_gamma *= increase_gamma

            current_gamma, residual, _V_tilde, h_x = backtrack_stepsize!(
                mu,
                current_gamma,
                X_iter,
                Y,
                residual_vector,
                cost,
                model,
                x0,
                Z,
                lower_U,
                upper_U;
                minimum_gamma = minimum_gamma,
                reduce_gamma = reduce_gamma,
            )

            push!(objective_history, constant_cost + cost(X_iter) + h_x)
            push!(gamma_history, current_gamma)
            push!(residual_history, residual)

            if residual <= tol
                if verbose
                    println("Learned TOS converged at iteration $iter")
                end
                break
            end

            Z .+= relaxation .* residual_vector
        end

        wall_time = time() - start_time
        solve_time = wall_time

        U = reshape(copy(X_iter), nu, N)
        X = rollout(system, x0, U)

        return (
            U = U,
            X = X,
            objective_history = objective_history,
            residual_history = residual_history,
            gamma_history = gamma_history,
            gamma = current_gamma,
            wall_time = wall_time,
            solve_time = solve_time,
        )
    end
end
