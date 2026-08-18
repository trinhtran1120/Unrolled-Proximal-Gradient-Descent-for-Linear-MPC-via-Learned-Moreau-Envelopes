using LinearAlgebra
using ProximalOperators

include("pgm.jl")
include(joinpath(@__DIR__, "..", "utils", "utils.jl"))


function learned_grad_ME(
    model::LearnedPCF,
    system::LinearMPC,
    W::Matrix{Float64},
    x0::Vector{Float64},
    rho::Real,
)
    # The learned PCF represents psi(q; x0) = 0.5 * dist_F(x0)(q)^2.
    # For the indicator constraint, grad phi^rho = grad psi / rho.
    input = vcat(vec(W), x0)
    full_gradient = learned_moreau_full_gradient(
        model,
        input,
        rho,
    )
    control_dim = system.nu * system.N
    return reshape(full_gradient[1:control_dim], system.nu, system.N) ./ rho
end


function f_model(f_x, grad_f_x, res, rho)
    return f_x - real(dot(vec(grad_f_x), vec(res))) + norm(res)^2 / (2 * rho)
end


function backtrack_stepsize!(
    rho::R,
    model::LearnedPCF,
    system::LinearMPC,
    x0::Vector{Float64},
    U::Matrix{Float64},
    f_U::R,
    grad_U::Matrix{Float64},
    W::Matrix{Float64},
    U_next::Matrix{Float64},
    res::Matrix{Float64};
    minimum_rho::R = R(1e-6),
    reduce_rho::R = R(0.5),
) where {R}
    @. W = U - rho * grad_U
    moreau_grad = learned_grad_ME(model, system, W, x0, rho)
    @. U_next = W - rho * moreau_grad
    @. res = U - U_next

    f_next, _ = grad_cost(system, x0, U_next)
    f_next_upper = f_model(f_U, grad_U, res, rho)
    tol = 10 * eps(R) * (1 + abs(f_next))

    while f_next > f_next_upper + tol && rho >= minimum_rho
        rho *= reduce_rho
        @. W = U - rho * grad_U
        moreau_grad = learned_grad_ME(model, system, W, x0, rho)
        @. U_next = W - rho * moreau_grad
        @. res = U - U_next

        f_next, _ = grad_cost(system, x0, U_next)
        f_next_upper = f_model(f_U, grad_U, res, rho)
        tol = 10 * eps(R) * (1 + abs(f_next))
    end

    if rho < minimum_rho
        @warn "stepsize `rho` became too small ($(rho))"
    end

    return rho, f_next, f_next_upper
end


function exact_PGM_polish(
    system::LinearMPC,
    x0::Vector{Float64},
    U0::Matrix{Float64};
    rho::Float64,
    max_iter::Int = 100,
    tol::Float64 = 1e-6,
)
    U = copy(U0)
    W = similar(U)
    U_next = similar(U)
    feasible_set = constraint(system, x0; solver = :osqp)
    objective_history = Float64[]
    residual_history = Float64[]

    for _iter in 1:max_iter
        cost, grad = grad_cost(system, x0, U)
        @. W = U - rho * grad
        prox!(vec(U_next), feasible_set, vec(W), rho)

        residual = norm(U - U_next, Inf) / rho
        push!(objective_history, cost)
        push!(residual_history, residual)
        copyto!(U, U_next)

        if residual <= tol
            break
        end
    end

    return (
        U = U,
        objective_history = objective_history,
        residual_history = residual_history,
    )
end


function learned_PGM(
    model::LearnedPCF,
    system::LinearMPC;
    x0::Vector{Float64} = system.x0,
    rho::Float64 = model.rho,
    minimum_rho::Float64 = 1e-6,
    reduce_rho::Float64 = 0.5,
    increase_rho::Float64 = 1.05,
    max_iter::Int = 100,
    polish_iter::Int = 0,
    polish_rho = nothing,
    tol::Float64 = 1e-6,
    verbose = false,
)
    U = zeros(Float64, system.nu, system.N)
    W = similar(U)
    U_next = similar(U)
    res = similar(U)

    objective_history = Float64[]
    residual_history = Float64[]
    rho_history = Float64[]
    upper_bound_history = Float64[]

    start_time = time()
    solve_time = 0.0
    current_rho = rho

    @inbounds for iter in 1:max_iter
        cost, grad = grad_cost(system, x0, U)
        current_rho *= increase_rho

        current_rho, trial_cost, trial_upper = backtrack_stepsize!(
            current_rho,
            model,
            system,
            x0,
            U,
            cost,
            grad,
            W,
            U_next,
            res;
            minimum_rho = minimum_rho,
            reduce_rho = reduce_rho,
        )

        push!(objective_history, trial_cost)
        push!(rho_history, current_rho)
        push!(upper_bound_history, trial_upper)

        residual = norm(res, Inf) / current_rho
        push!(residual_history, residual)
        copyto!(U, U_next)
        solve_time = time() - start_time

        if residual <= tol
            if verbose
                println("Learned adaptive PGM converged at iteration $iter")
            end
            break
        end
    end

    polish_objective_history = Float64[]
    polish_residual_history = Float64[]
    if polish_iter > 0
        rho_for_polish = polish_rho === nothing ? current_rho : Float64(polish_rho)
        polish_sol = exact_PGM_polish(
            system,
            x0,
            U;
            rho = rho_for_polish,
            max_iter = polish_iter,
            tol = tol,
        )
        U = polish_sol.U
        polish_objective_history = polish_sol.objective_history
        polish_residual_history = polish_sol.residual_history
    end

    solve_time = time() - start_time

    return (
        U = U,
        X = rollout(system, x0, U),
        objective_history = objective_history,
        residual_history = residual_history,
        rho_history = rho_history,
        upper_bound_history = upper_bound_history,
        rho = current_rho,
        polish_objective_history = polish_objective_history,
        polish_residual_history = polish_residual_history,
        solve_time = solve_time,
    )
end
