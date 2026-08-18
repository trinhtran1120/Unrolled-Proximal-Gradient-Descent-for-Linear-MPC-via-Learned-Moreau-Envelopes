using LinearAlgebra
using JuMP
using Gurobi
import MathOptInterface as MOI

include("pgm.jl")
include(joinpath(@__DIR__, "..", "utils", "utils.jl"))


function learned_grad(model::LearnedPCF, system::LinearMPC, q::Matrix{Float64}, x0::Vector{Float64})
    input = vec(q)
    grad = pcf_grad(model, input, x0)
    return reshape(grad, system.nu, system.N)
end


function f_model(f_x, grad_f_x, res, gamma)
    return f_x - real(dot(vec(grad_f_x), vec(res))) + norm(res)^2 / (2 * gamma)
end


function backtrack_stepsize!(
    gamma::R,
    model::LearnedPCF,
    system::LinearMPC,
    project,
    refine::Bool,
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
    @. q = U - gamma * grad_U
    grad_g = learned_grad(model, system, q, x0)
    @. U_next = q - grad_g
    refine_residual = 0.0
    projection_solve_time = 0.0
    if refine
        refine_input = copy(U_next)
        projected_U, project_time = project(x0, refine_input)
        copyto!(U_next, projected_U)
        projection_solve_time += project_time
        refine_residual = norm(refine_input - U_next, Inf)
    end
    @. res = U - U_next

    f_next, _ = evaluate_cost_gradient(system, x0, U_next)
    f_next_upper = f_model(f_U, grad_U, res, gamma)
    tol = 10 * eps(R) * (1 + abs(f_next))

    while f_next > f_next_upper + tol && gamma >= minimum_gamma
        gamma *= reduce_gamma
        @. q = U - gamma * grad_U
        grad_g = learned_grad(model, system, q, x0)
        @. U_next = q - grad_g
        refine_residual = 0.0
        if refine
            refine_input = copy(U_next)
            projected_U, project_time = project(x0, refine_input)
            copyto!(U_next, projected_U)
            projection_solve_time += project_time
            refine_residual = norm(refine_input - U_next, Inf)
        end
        @. res = U - U_next

        f_next, _ = evaluate_cost_gradient(system, x0, U_next)
        f_next_upper = f_model(f_U, grad_U, res, gamma)
        tol = 10 * eps(R) * (1 + abs(f_next))
    end

    if gamma < minimum_gamma
        @warn "stepsize `gamma` became too small ($(gamma))"
    end

    return gamma, f_next, refine_residual, projection_solve_time
end


function projection(system::LinearMPC)
    model = Model(Gurobi.Optimizer)
    set_silent(model)

    nx, nu, N = system.nx, system.nu, system.N
    @variable(model, system.xmin <= x[1:nx, 1:N+1] <= system.xmax)
    @variable(model, system.umin <= u[1:nu, 1:N] <= system.umax)
    @variable(model, x0[i in 1:nx] in MOI.Parameter(system.x0[i]))
    @variable(model, q_ref[i in 1:nu, k in 1:N] in MOI.Parameter(0.0))

    @constraint(model, x[:, 1] .== x0)
    for k in 1:N
        @constraint(model, x[:, k+1] .== system.A * x[:, k] + system.B * u[:, k])
    end
    @objective(model, Min, 0.5 * sum((u[i, k] - q_ref[i, k])^2 for i in 1:nu, k in 1:N))

    return function solver(init::Vector{Float64}, q::Matrix{Float64}; verbose = false)
        set_parameter_value.(x0, init)
        set_parameter_value.(q_ref, q)
        optimize!(model)

        if verbose
            println(solution_summary(model))
        end

        return value.(u), solve_time(model)
    end
end


function learned_PGM(
    model::LearnedPCF,
    system::LinearMPC;
    gamma::Float64 = model.rho,
    minimum_gamma::Float64 = 1e-6,
    reduce_gamma::Float64 = 0.5,
    increase_gamma::Float64 = 1.05,
    max_iter::Int = 100,
    refine::Bool = false,
    tol::Float64 = 1e-6,
)
    U = zeros(Float64, system.nu, system.N)
    q = similar(U)
    U_next = similar(U)
    res = similar(U)
    project = projection(system)

    return @inbounds function solver(x0::Vector{Float64}; verbose = false)
        fill!(U, 0.0)
        current_gamma = gamma
        
        objective_history = Float64[]
        residual_history = Float64[]
        gamma_history = Float64[]
        refine_residual_history = Float64[]
        projection_solve_time = 0.0

        start_time = time()
        for iter in 1:max_iter
            cost, grad = evaluate_cost_gradient(system, x0, U)
            current_gamma *= increase_gamma

            current_gamma, trial_cost, refine_residual, project_time = backtrack_stepsize!(
                current_gamma,
                model,
                system,
                project,
                refine,
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
            projection_solve_time += project_time

            push!(gamma_history, current_gamma)
            if refine
                push!(refine_residual_history, refine_residual)
            end

            push!(objective_history, trial_cost)

            @. res = U - U_next
            residual = norm(res, Inf)
            push!(residual_history, residual)
            copyto!(U, U_next)

            if residual <= tol
                if verbose
                    println("Learned adaptive PGM converged at iteration $iter")
                end
                break
            end
        end

        wall_time = time() - start_time
        solve_time = wall_time + projection_solve_time

        return (
            U = copy(U),
            X = rollout(system, x0, U),
            objective_history = objective_history,
            residual_history = residual_history,
            gamma_history = gamma_history,
            gamma = current_gamma,
            refine_residual_history = refine_residual_history,
            projection_solve_time = projection_solve_time,
            wall_time = wall_time,
            solve_time = solve_time,
        )
    end
end
