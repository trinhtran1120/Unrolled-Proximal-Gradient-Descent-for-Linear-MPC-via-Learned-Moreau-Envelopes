using LinearAlgebra
using Base.Threads

include("pgm.jl")
include(joinpath(@__DIR__, "..", "utils", "utils.jl"))


function mini_batch_grad!(
    grad::AbstractVector{Float64},
    input::AbstractVector{Float64},
    projection::AbstractVector{Float64},
    gamma::Float64;
    batch_size::Int = length(input),
)
    ranges = [first:min(first + batch_size - 1, length(input)) for first in 1:min(batch_size, length(input)):length(input)]
    Threads.@threads for batch in eachindex(ranges)
        range = ranges[batch]
        @inbounds @views grad[range] .= (input[range] .- projection[range]) ./ gamma
    end
    return grad
end


function learned_moreau(
    model::ProjectionMLP,
    system::LinearMPC,
    U::Matrix{Float64},
    x0::Vector{Float64};
    gamma::Float64 = 1.0,
    batch_size::Int = length(U),
)
    input = vec(U)
    V_tilde, _ = projection_mlp_forward(model, input, x0)
    d = input .- V_tilde
    value = dot(d, d) / (2 * gamma)
    grad = similar(input)
    mini_batch_grad!(grad, input, V_tilde, gamma; batch_size = batch_size)
    return value, reshape(grad, system.nu, system.N)
end


function learned_objective_gradient(
    model::ProjectionMLP,
    system::LinearMPC,
    x0::Vector{Float64},
    U::Matrix{Float64};
    gamma::Float64 = 1.0,
    batch_size::Int = length(U),
)
    cost, cost_grad = evaluate_cost_gradient(system, x0, U)
    moreau_value, moreau_grad = learned_moreau(model, system, U, x0; gamma = gamma, batch_size = batch_size)
    return cost + moreau_value, cost_grad .+ moreau_grad
end

function f_model(f_x, grad_f_x, res, gamma)
    return f_x - real(dot(vec(grad_f_x), vec(res))) + norm(res)^2 / (2 * gamma)
end


struct FixedNesterovSequence{R} end

FixedNesterovSequence(R) = FixedNesterovSequence{R}()

function Base.iterate(::FixedNesterovSequence{R}, t = R(1)) where {R}
    t_next = (1 + sqrt(1 + 4 * t^2)) / 2
    return (t - 1) / t_next, t_next
end

Base.IteratorSize(::Type{<:FixedNesterovSequence}) = Base.IsInfinite()
Base.eltype(::Type{FixedNesterovSequence{R}}) where {R} = R


function backtrack_stepsize!(
    gamma::R,
    model,
    system::LinearMPC,
    x0::Vector{Float64},
    U::Matrix{Float64},
    q::Matrix{Float64},
    U_next::Matrix{Float64},
    res::Matrix{Float64};
    moreau_gamma::R = R(1.0),
    cost_gradient = (U_eval -> evaluate_cost_gradient(system, x0, U_eval)),
    minimum_gamma::R = R(1e-6),
    reduce_gamma::R = R(0.5),
) where {R}
    f_U, grad_U = cost_gradient(U)
    learned_grad = reshape(projection_mlp_gradient(model, vec(U), x0, moreau_gamma), system.nu, system.N)
    @. q = U - gamma * (grad_U + learned_grad)
    @. U_next = clamp(q, system.umin, system.umax)
    @. res = U - U_next
    f_next, _ = cost_gradient(U_next)

    return backtrack_stepsize!(
        gamma,
        model,
        system,
        x0,
        U,
        f_U,
        grad_U,
        q,
        U_next,
        res,
        f_next;
        learned_grad = learned_grad,
        cost_gradient = cost_gradient,
        minimum_gamma = minimum_gamma,
        reduce_gamma = reduce_gamma,
    )
end

function backtrack_stepsize!(
    gamma::R,
    model,
    system::LinearMPC,
    x0::Vector{Float64},
    U::Matrix{Float64},
    f_U::R,
    grad_U::Matrix{Float64},
    q::Matrix{Float64},
    U_next::Matrix{Float64},
    res::Matrix{Float64},
    f_next::R;
    learned_grad::Matrix{Float64},
    cost_gradient = (U_eval -> evaluate_cost_gradient(system, x0, U_eval)),
    minimum_gamma::R = R(1e-6),
    reduce_gamma::R = R(0.5),
) where {R}
    f_next_upper = f_model(f_U, grad_U, res, gamma)
    tol = 10 * eps(R) * (1 + abs(f_next))

    while f_next > f_next_upper + tol && gamma >= minimum_gamma
        gamma *= reduce_gamma

        @. q = U - gamma * (grad_U + learned_grad)
        @. U_next = clamp(q, system.umin, system.umax)
        @. res = U - U_next
        f_next, _ = cost_gradient(U_next)
        f_next_upper = f_model(f_U, grad_U, res, gamma)
        tol = 10 * eps(R) * (1 + abs(f_next))
    end

    if gamma < minimum_gamma
        @warn "stepsize `gamma` became too small ($(gamma))"
    end

    return gamma, f_next
end


function learned_PGM(
    model::ProjectionMLP,
    system::LinearMPC;
    gamma::Float64 = 1.0,
    minimum_gamma::Float64 = 1e-6,
    reduce_gamma::Float64 = 0.5,
    increase_gamma::Float64 = 1.05,
    max_iter::Int = 100,
    tol::Float64 = 1e-6,
    gradient_batch_size::Int = system.nu * system.N,
    moreau_gamma::Float64 = 1.0,
    accelerated::Bool = true,
)
    U = zeros(Float64, system.nu, system.N)
    x = similar(U)
    z_prev = similar(U)
    q = similar(U)
    U_next = similar(U)
    res = similar(U)

    return @inbounds function solver(x0::Vector{Float64}; verbose = false)
        fill!(U, 0.0)
        fill!(x, 0.0)
        fill!(z_prev, 0.0)
        current_gamma = gamma
        nesterov = Iterators.Stateful(FixedNesterovSequence(Float64))

        objective_history = Float64[]
        residual_history = Float64[]
        gamma_history = Float64[]
        clock_time= 0.0
        cost_model, constant_cost = cost_cache(system, x0)
        cost_gradient = U_eval -> cached_cost_gradient(cost_model, constant_cost, system, U_eval)

        start_time = time()
        for iter in 1:max_iter
            current_gamma *= increase_gamma

            current_gamma, trial_cost = backtrack_stepsize!(
                current_gamma,
                model,
                system,
                x0,
                x,
                q,
                U_next,
                res;
                moreau_gamma = moreau_gamma,
                cost_gradient = cost_gradient,
                minimum_gamma = minimum_gamma,
                reduce_gamma = reduce_gamma,
            )

            @. res = x - U_next
            residual = norm(res, Inf) / current_gamma

            clock_start = time()
            push!(objective_history, trial_cost)
            push!(gamma_history, current_gamma)
            push!(residual_history, residual)
            
            clock_time+= time() - clock_start

            copyto!(U, U_next)

            if residual <= tol
                if verbose
                    println("Learned PGM converged at iteration $iter")
                end
                break
            end

            if accelerated
                beta = popfirst!(nesterov)
                @. x = U_next + beta * (U_next - z_prev)
                copyto!(z_prev, U_next)
            else
                copyto!(x, U_next)
            end
        end

        wall_time = time() - start_time
        solve_time = wall_time - clock_time

        return (
            U = copy(U),
            X = rollout(system, x0, U),
            objective_history = objective_history,
            residual_history = residual_history,
            gamma_history = gamma_history,
            gamma = current_gamma,
            wall_time = wall_time,
            solve_time = solve_time,
        )
    end
end
