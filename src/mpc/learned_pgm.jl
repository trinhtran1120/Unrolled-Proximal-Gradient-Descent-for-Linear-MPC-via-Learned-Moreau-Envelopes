using LinearAlgebra
using Base.Threads

include("pgm.jl")
include(joinpath(@__DIR__, "..", "utils", "utils.jl"))


function mini_batch_grad!(
    grad::AbstractVector{Float64},
    input::AbstractVector{Float64},
    projection::AbstractVector{Float64},
    rho::Float64;
    batch_size::Int = length(input),
)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    ranges = [first:min(first + batch_size - 1, length(input)) for first in 1:min(batch_size, length(input)):length(input)]
    Threads.@threads for batch in eachindex(ranges)
        range = ranges[batch]
        @inbounds @views grad[range] .= (input[range] .- projection[range]) ./ rho
    end
    return grad
end


function learned_moreau(
    model::ProjectionMLP,
    system::LinearMPC,
    U::Matrix{Float64},
    x0::Vector{Float64};
    batch_size::Int = length(U),
)
    size(U) == (system.nu, system.N) ||
        throw(DimensionMismatch("U has size $(size(U)); expected $((system.nu, system.N))"))
    length(x0) == system.nx ||
        throw(DimensionMismatch("x0 has length $(length(x0)); expected $(system.nx)"))

    input = vec(U)
    V_tilde, _ = projection_mlp_forward(model, input, x0)
    d = input .- V_tilde
    value = dot(d, d) / (2 * model.rho)
    grad = similar(input)
    mini_batch_grad!(grad, input, V_tilde, model.rho; batch_size = batch_size)
    return value, reshape(grad, system.nu, system.N)
end


function learned_objective_gradient(
    model::ProjectionMLP,
    system::LinearMPC,
    x0::Vector{Float64},
    U::Matrix{Float64};
    batch_size::Int = length(U),
)
    cost, cost_grad = evaluate_cost_gradient(system, x0, U)
    moreau_value, moreau_grad = learned_moreau(model, system, U, x0; batch_size = batch_size)
    return cost + moreau_value, cost_grad .+ moreau_grad
end


function learned_grad(model::ProjectionMLP, system::LinearMPC, q::Matrix{Float64}, x0::Vector{Float64})
    grad = projection_mlp_gradient(model, vec(q), x0)
    return reshape(grad, system.nu, system.N)
end


learned_correction_scale(::ProjectionMLP, gamma::Float64) = gamma


function f_model(f_x, grad_f_x, res, gamma)
    return f_x - real(dot(vec(grad_f_x), vec(res))) + norm(res)^2 / (2 * gamma)
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
)
    @. q = U - gamma * grad_U
    grad_g = learned_grad(model, system, q, x0)
    scale = learned_correction_scale(model, gamma)
    @. U_next = clamp(q - scale * grad_g, system.umin, system.umax)
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
    minimum_gamma::R = R(1e-6),
    reduce_gamma::R = R(0.5),
) where {R}
    learned_trial_step!(model, system, x0, U, grad_U, gamma, q, U_next)

    @. res = U - U_next
    f_next, _ = evaluate_cost_gradient(system, x0, U_next)
    f_next_upper = f_model(f_U, grad_U, res, gamma)
    tol = 10 * eps(R) * (1 + abs(f_next))
    backtracks = 0

    while f_next > f_next_upper + tol && gamma >= minimum_gamma
        gamma *= reduce_gamma
        backtracks += 1

        learned_trial_step!(model, system, x0, U, grad_U, gamma, q, U_next)

        @. res = U - U_next
        f_next, _ = evaluate_cost_gradient(system, x0, U_next)
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
        clock_time= 0.0

        start_time = time()
        for iter in 1:max_iter
            cost, grad = evaluate_cost_gradient(system, x0, U)
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
            )

            @. res = U - U_next
            residual = norm(res, Inf)

            clock_start = time()
            push!(objective_history, trial_cost)
            push!(gamma_history, current_gamma)
            push!(backtrack_history, backtracks)
            push!(residual_history, residual)
            clock_time+= time() - clock_start

            copyto!(U, U_next)

            if verbose
                println("Learned PGM iteration $iter: residual = $residual, gamma = $current_gamma")
            end

            if residual <= tol
                if verbose
                    println("Learned PGM converged at iteration $iter")
                end
                break
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
