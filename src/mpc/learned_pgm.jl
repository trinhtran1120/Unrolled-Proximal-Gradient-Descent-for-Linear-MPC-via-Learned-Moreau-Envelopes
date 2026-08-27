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
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
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
    size(U) == (system.nu, system.N) ||
        throw(DimensionMismatch("U has size $(size(U)); expected $((system.nu, system.N))"))
    length(x0) == system.nx ||
        throw(DimensionMismatch("x0 has length $(length(x0)); expected $(system.nx)"))

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


function learned_grad(model::ProjectionMLP, system::LinearMPC, q::Matrix{Float64}, x0::Vector{Float64})
    input = vec(q)
    V_tilde, _ = projection_mlp_forward(model, input, x0)
    return reshape(input .- V_tilde, system.nu, system.N)
end


function learned_correction_scale(::ProjectionMLP, gamma::Float64, projection_relaxation)
    return isnothing(projection_relaxation) ? gamma : projection_relaxation
end


function learned_cost_cache(system::LinearMPC, x0::Vector{Float64})
    cost = single_shooting_cost(system, x0)
    zero_U = zeros(Float64, system.nu, system.N)
    zero_X = rollout(system, x0, zero_U)
    terminal_dx = zero_X[:, system.N+1] - system.xr
    constant_cost = sum(system.cost_func(zero_X[:, k], zero_U[:, k]) for k in 1:system.N) +
                    dot(terminal_dx, system.QN * terminal_dx)
    return cost, constant_cost
end


function learned_cached_cost_gradient(
    cost::DifferentiableQuadratic,
    constant_cost::Float64,
    system::LinearMPC,
    U::Matrix{Float64},
)
    input = vec(U)
    Hu = cost.H * input
    grad = Hu .+ cost.h
    value = 0.5 * dot(input, Hu) + dot(cost.h, input) + constant_cost
    return value, reshape(grad, system.nu, system.N)
end


function f_model(f_x, grad_f_x, res, gamma)
    return f_x - real(dot(vec(grad_f_x), vec(res))) + norm(res)^2 / (2 * gamma)
end


function nesterov_step(theta::Float64)
    theta_next = (1 + sqrt(1 + 4 * theta^2)) / 2
    beta = (theta - 1) / theta_next
    return theta_next, beta
end


function learned_trial_step!(
    model,
    system::LinearMPC,
    x0::Vector{Float64},
    U::Matrix{Float64},
    grad_U::Matrix{Float64},
    gamma::Float64,
    q::Matrix{Float64},
    U_next::Matrix{Float64},
    projection_relaxation,
)
    @. q = U - gamma * grad_U
    grad_g = learned_grad(model, system, q, x0)
    scale = learned_correction_scale(model, gamma, projection_relaxation)
    umin = as_column(system.umin)
    umax = as_column(system.umax)
    @. U_next = clamp(q - scale * grad_g, umin, umax)
    return U_next
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
    res::Matrix{Float64};
    cost_gradient = (U_eval -> evaluate_cost_gradient(system, x0, U_eval)),
    minimum_gamma::R = R(1e-6),
    reduce_gamma::R = R(0.5),
    projection_relaxation = nothing,
) where {R}
    learned_trial_step!(model, system, x0, U, grad_U, gamma, q, U_next, projection_relaxation)

    @. res = U - U_next
    f_next, _ = cost_gradient(U_next)
    f_next_upper = f_model(f_U, grad_U, res, gamma)
    tol = 10 * eps(R) * (1 + abs(f_next))
    backtracks = 0

    while f_next > f_next_upper + tol && gamma >= minimum_gamma
        gamma *= reduce_gamma
        backtracks += 1

        learned_trial_step!(model, system, x0, U, grad_U, gamma, q, U_next, projection_relaxation)

        @. res = U - U_next
        f_next, _ = cost_gradient(U_next)
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
    gamma::Float64 = 1.0,
    minimum_gamma::Float64 = 1e-6,
    reduce_gamma::Float64 = 0.5,
    increase_gamma::Float64 = 1.05,
    max_iter::Int = 100,
    tol::Float64 = 1e-6,
    gradient_batch_size::Int = system.nu * system.N,
    projection_relaxation = nothing,
    accelerated::Bool = true,
    adaptive_restart::Bool = false,
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
        theta = 1.0

        objective_history = Float64[]
        residual_history = Float64[]
        gamma_history = Float64[]
        backtrack_history = Int[]
        clock_time= 0.0
        cost_model, constant_cost = learned_cost_cache(system, x0)
        cost_gradient =
            U_eval -> learned_cached_cost_gradient(cost_model, constant_cost, system, U_eval)

        start_time = time()
        for iter in 1:max_iter
            cost, grad = cost_gradient(x)
            current_gamma *= increase_gamma

            current_gamma, trial_cost, backtracks = backtrack_stepsize!(
                current_gamma,
                model,
                system,
                x0,
                x,
                cost,
                grad,
                q,
                U_next,
                res;
                cost_gradient = cost_gradient,
                minimum_gamma = minimum_gamma,
                reduce_gamma = reduce_gamma,
                projection_relaxation = projection_relaxation,
            )

            @. res = x - U_next
            residual = norm(res, Inf) / current_gamma

            clock_start = time()
            push!(objective_history, trial_cost)
            push!(gamma_history, current_gamma)
            push!(backtrack_history, backtracks)
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
                if adaptive_restart && real(dot(vec(U_next .- z_prev), vec(x .- U_next))) > 0
                    theta = 1.0
                    copyto!(x, U_next)
                else
                    theta, beta = nesterov_step(theta)
                    @. x = U_next + beta * (U_next - z_prev)
                end
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
            backtrack_history = backtrack_history,
            gamma = current_gamma,
            wall_time = wall_time,
            solve_time = solve_time,
        )
    end
end
