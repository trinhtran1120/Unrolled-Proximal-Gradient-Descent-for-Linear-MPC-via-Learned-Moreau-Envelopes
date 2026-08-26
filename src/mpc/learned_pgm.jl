using LinearAlgebra
using Base.Threads

include("pgm.jl")
include(joinpath(@__DIR__, "..", "utils", "utils.jl"))


function batch_ranges(n::Int, batch_size::Int)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    return [first:min(first + batch_size - 1, n) for first in 1:batch_size:n]
end


function moreau_gradient_from_projection!(
    grad::AbstractVector{Float64},
    input::AbstractVector{Float64},
    projection::AbstractVector{Float64},
    rho::Float64;
    batch_size::Int = length(input),
    threaded::Bool = Threads.nthreads() > 1,
)
    rho > 0.0 || throw(ArgumentError("rho must be positive"))
    length(grad) == length(input) == length(projection) ||
        throw(DimensionMismatch("grad, input, and projection must have the same length"))

    ranges = batch_ranges(length(input), min(batch_size, length(input)))
    if threaded && length(ranges) > 1
        Threads.@threads for batch in eachindex(ranges)
            range = ranges[batch]
            @inbounds @views grad[range] .= (input[range] .- projection[range]) ./ rho
        end
    else
        for range in ranges
            @inbounds @views grad[range] .= (input[range] .- projection[range]) ./ rho
        end
    end
    return grad
end


function learned_moreau_value(model::ProjectionMLP, U::Matrix{Float64}, x0::Vector{Float64})
    V_tilde, _ = projection_mlp_corrected(model, vec(U), x0)
    d = vec(U) .- V_tilde
    return dot(d, d) / (2 * model.rho)
end


function learned_moreau_value_gradient(
    model::ProjectionMLP,
    system::LinearMPC,
    U::Matrix{Float64},
    x0::Vector{Float64};
    batch_size::Int = length(U),
    threaded::Bool = Threads.nthreads() > 1,
)
    size(U) == (system.nu, system.N) ||
        throw(DimensionMismatch("U has size $(size(U)); expected $((system.nu, system.N))"))
    length(x0) == system.nx ||
        throw(DimensionMismatch("x0 has length $(length(x0)); expected $(system.nx)"))

    input = vec(U)
    V_tilde, _ = projection_mlp_corrected(model, input, x0)
    d = input .- V_tilde
    value = dot(d, d) / (2 * model.rho)
    grad = similar(input)
    moreau_gradient_from_projection!(grad, input, V_tilde, model.rho; batch_size = batch_size, threaded = threaded)
    return value, reshape(grad, system.nu, system.N)
end


function learned_moreau_gradient(
    model::ProjectionMLP,
    system::LinearMPC,
    U::Matrix{Float64},
    x0::Vector{Float64};
    batch_size::Int = length(U),
    threaded::Bool = Threads.nthreads() > 1,
)
    _, grad = learned_moreau_value_gradient(
        model,
        system,
        U,
        x0;
        batch_size = batch_size,
        threaded = threaded,
    )
    return grad
end


function learned_objective_gradient(
    model::ProjectionMLP,
    system::LinearMPC,
    x0::Vector{Float64},
    U::Matrix{Float64};
    batch_size::Int = length(U),
    threaded::Bool = Threads.nthreads() > 1,
)
    cost, cost_grad = evaluate_cost_gradient(system, x0, U)
    moreau_value, moreau_grad = learned_moreau_value_gradient(
        model,
        system,
        U,
        x0;
        batch_size = batch_size,
        threaded = threaded,
    )
    return cost + moreau_value, cost_grad .+ moreau_grad
end


function learned_objective_gradient(model::ProjectionMLP, system::LinearMPC, x0::Vector{Float64}, U::Matrix{Float64})
    cost, cost_grad = evaluate_cost_gradient(system, x0, U)
    moreau_value = learned_moreau_value(model, U, x0)
    moreau_grad = learned_moreau_gradient(model, system, U, x0)
    return cost + moreau_value, cost_grad .+ moreau_grad
end


function f_model(f_x, grad_f_x, res, gamma)
    return f_x - real(dot(vec(grad_f_x), vec(res))) + norm(res)^2 / (2 * gamma)
end


function backtrack_stepsize!(
    gamma::R,
    model::ProjectionMLP,
    system::LinearMPC,
    x0::Vector{Float64},
    U::Matrix{Float64},
    f_U::R,
    grad_U::Matrix{Float64},
    q::Matrix{Float64},
    U_next::Matrix{Float64},
    res::Matrix{Float64};
    minimum_gamma::R = R(1e-6),
    reduce_gamma::R = R(0.5),
    gradient_batch_size::Int = length(U),
    threaded_gradient::Bool = Threads.nthreads() > 1,
) where {R}
    @. q = U - gamma * grad_U
    @. U_next = clamp(q, system.umin, system.umax)
    @. res = U - U_next

    f_next, _ = learned_objective_gradient(
        model,
        system,
        x0,
        U_next;
        batch_size = gradient_batch_size,
        threaded = threaded_gradient,
    )
    f_next_upper = f_model(f_U, grad_U, res, gamma)
    tol = 10 * eps(R) * (1 + abs(f_next))
    backtracks = 0

    while f_next > f_next_upper + tol && gamma >= minimum_gamma
        gamma *= reduce_gamma
        backtracks += 1

        @. q = U - gamma * grad_U
        @. U_next = clamp(q, system.umin, system.umax)
        @. res = U - U_next

        f_next, _ = learned_objective_gradient(
            model,
            system,
            x0,
            U_next;
            batch_size = gradient_batch_size,
            threaded = threaded_gradient,
        )
        f_next_upper = f_model(f_U, grad_U, res, gamma)
        tol = 10 * eps(R) * (1 + abs(f_next))
    end

    if gamma < minimum_gamma
        @warn "stepsize `gamma` became too small ($(gamma))"
    end

    return gamma, f_next, backtracks
end


function learned_PGM(
    model::ProjectionMLP,
    system::LinearMPC;
    gamma::Float64 = model.rho,
    minimum_gamma::Float64 = 1e-6,
    reduce_gamma::Float64 = 0.5,
    increase_gamma::Float64 = 1.05,
    max_iter::Int = 100,
    tol::Float64 = 1e-6,
    gradient_batch_size::Int = system.nu * system.N,
    threaded_gradient::Bool = Threads.nthreads() > 1,
)
    U = zeros(Float64, system.nu, system.N)
    q = similar(U)
    U_next = similar(U)
    res = similar(U)

    return @inbounds function solver(x0::Vector{Float64}; verbose = false)
        fill!(U, 0.0)
        current_gamma = gamma

        objective_history = Float64[]
        residual_history = Float64[]
        gamma_history = Float64[]
        backtrack_history = Int[]

        start_time = time()
        for iter in 1:max_iter
            cost, grad = learned_objective_gradient(
                model,
                system,
                x0,
                U;
                batch_size = gradient_batch_size,
                threaded = threaded_gradient,
            )
            current_gamma *= increase_gamma

            current_gamma, trial_cost, backtracks = backtrack_stepsize!(
                current_gamma,
                model,
                system,
                x0,
                U,
                cost,
                grad,
                q,
                U_next,
                res;
                minimum_gamma = minimum_gamma,
                reduce_gamma = reduce_gamma,
                gradient_batch_size = gradient_batch_size,
                threaded_gradient = threaded_gradient,
            )

            push!(objective_history, trial_cost)
            push!(gamma_history, current_gamma)
            push!(backtrack_history, backtracks)

            @. res = U - U_next
            residual = norm(res, Inf)
            push!(residual_history, residual)
            copyto!(U, U_next)

            if verbose
                println("Learned PGM iteration $iter: residual = $residual, gamma = $current_gamma")
            end

            if residual <= tol
                # if verbose
                    println("Learned PGM converged at iteration $iter")
                # end
                break
            end
        end

        wall_time = time() - start_time

        return (
            U = copy(U),
            X = rollout(system, x0, U),
            objective_history = objective_history,
            residual_history = residual_history,
            gamma_history = gamma_history,
            backtrack_history = backtrack_history,
            gamma = current_gamma,
            wall_time = wall_time,
            solve_time = wall_time,
        )
    end
end
