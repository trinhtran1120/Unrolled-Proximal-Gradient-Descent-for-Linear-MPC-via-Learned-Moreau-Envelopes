"""Three-operator splitting for Moreau-smoothed Linear MPC.
minimize    f(U) + g(U) + h(U)
where
    f: condensed quadratic stage cost
    g: indicator function of the input box
    h: Moreau envelope of the state-feasible set indicator
"""

using LinearAlgebra
using OSQP
using JuMP
import MathOptInterface as MOI
using ProximalAlgorithms
using ProximalOperators

include("problem.jl")
include("../utils/preprocess.jl")

## Rollout
function rollout(problem::LinearMPC, x0::Vector{Float64}, U::Matrix{Float64})
    nx = problem.nx
    N = problem.N 
    A_ro = problem.A_ro
    B_ro = problem.B_ro

    X_stacked = A_ro * x0 + B_ro * vec(U)

    X = zeros(Float64, nx, N+1)
    X[:, 1] .= x0
    X[:, 2:end] .= reshape(X_stacked, nx, N)

    return X
end

StateConstraint(name::Symbol, problem, tol::Float64 = 1e-8) = StateConstraint(String(name), problem, tol)

function state_constraints(problem::LinearMPC, x0 = nothing)
    Gx = [problem.B_ro; -problem.B_ro]
    x0 === nothing && return Gx

    x_free = problem.A_ro * x0
    b_state = [
        fill(problem.xmax, problem.nx * problem.N) - x_free;
        -fill(problem.xmin, problem.nx * problem.N) + x_free
    ]

    return Gx, b_state
end


struct DifferentiableQuadratic
    H::Matrix{Float64}
    h::Vector{Float64}
end


function (f::DifferentiableQuadratic)(u)
    return 0.5 * dot(u, f.H * u) + dot(f.h, u)
end

function ProximalAlgorithms.value_and_gradient(f::DifferentiableQuadratic, u)
    grad = f.H * u + f.h
    return f(u), grad
end


function single_shooting_cost(problem::LinearMPC, x0::Vector{Float64})
    N = problem.N
    nu = problem.nu
    nx = problem.nx
    A_ro = problem.A_ro
    B_ro = problem.B_ro
    Q = problem.Q
    R = problem.R
            
    H = zeros(Float64, N*nu, N*nu)
    h = zeros(Float64, N*nu)

    for k in 1:N-1
        x_idx = ((k - 1) * nx + 1) : (k * nx)
        Ak = view(A_ro, x_idx, :)
        Bk = view(B_ro, x_idx, :)

        H .+= Bk' * Q * Bk
        h .+= Bk' * Q * Ak * x0
    end

    for k in 1:N
        u_idx = ((k - 1) * nu + 1) : (k * nu)
        H[u_idx, u_idx] .+= R 
    end

    return DifferentiableQuadratic(H, h)

end

function cost_cache(problem::LinearMPC, x0::Vector{Float64})
    f = single_shooting_cost(problem, x0)
    zero_U = zeros(Float64, problem.nu, problem.N)
    zero_X = rollout(problem, x0, zero_U)
    constant_cost = sum(problem.cost_func(zero_X[:, k], zero_U[:, k]) for k in 1:problem.N)
    return f, constant_cost
end

function cached_cost_gradient(
    f::DifferentiableQuadratic,
    constant_cost::Float64,
    problem::LinearMPC,
    U::Matrix{Float64},
)
    input = vec(U)
    Hu = f.H * input
    grad = Hu .+ f.h
    value = 0.5 * dot(input, Hu) + dot(f.h, input) + constant_cost
    return value, reshape(grad, problem.nu, problem.N)
end

## Moreau Envelope
struct MoreauEnvelope
    cost::DifferentiableQuadratic
    project::Function
    x0::Vector{Float64}
    mu::Float64
    data
end

function (f::MoreauEnvelope)(u)
    value, _ = ProximalAlgorithms.value_and_gradient(f, u)
    
    return value
end


function ProximalAlgorithms.value_and_gradient(f::MoreauEnvelope, u)
    """
    For a feasible input set C(x0), this evaluates the quadratic cost plus the
    Moreau envelope of the indicator function I_C:

        M_mu I_C(u) = min_{v in C(x0)} (1 / 2) ||v - u||_2^2

    with projection

        p = Pi_C(x0)(u) = argmin_{v in C(x0)} (1 / 2) ||v - u||_2^2.

    The returned value and gradient are

        phi(u) = l(u) + (1 / mu) M_mu I_C(u)
        grad phi(u) = grad l(u) + (1 / mu) (u - p),

    where l(u) = f.cost(u).
    """
    cost_value, cost_grad = ProximalAlgorithms.value_and_gradient(f.cost, u)
    projected_u, distance_value = f.project(f.x0, u)
    distance_grad = u .- projected_u
    value = cost_value + distance_value / f.mu
    grad = cost_grad .+ distance_grad ./ f.mu

    if f.data !== nothing
        append_projection_sample!(
            f.data,
            f.x0,
            u,
            projected_u,
        )
    end

    return value, grad
end

function append_projection_sample!(
    data,
    x0::AbstractVector,
    U_query::AbstractVector,
    V_star::AbstractVector,
)
    push!(get!(data, "x0", Vector{Float64}[]), copy(x0))
    push!(get!(data, "U_query", Vector{Float64}[]), copy(U_query))
    push!(get!(data, "V_star", Vector{Float64}[]), copy(V_star))

    return nothing
end

## objective computation
function evaluate_cost_gradient(problem::LinearMPC, x0::Vector{Float64}, U::Matrix{Float64})
    """Evaluate the cost and gradient for a given trajectory"""
    f, constant_cost = cost_cache(problem, x0)
    return cached_cost_gradient(f, constant_cost, problem, U)
end


## Projection onto the convex set C(x0)
function StateConstraint(name::String, problem::LinearMPC, tol::Float64)
    model = pick_solver(name, tol)
    A_ro = problem.A_ro
    B_ro = problem.B_ro
    nu = problem.nu
    N = problem.N
    nx = problem.nx
    xmax = problem.xmax
    xmin = problem.xmin

    @variable(model, V[1:N*nu])
    @variable(model, x0[i in 1:nx] in MOI.Parameter(problem.x0[i]))
    @variable(model, U_ref[i in 1:N*nu] in MOI.Parameter(0.0))

    # State bound
    X = A_ro * x0 + B_ro * V
    @constraint(model, X .<= fill(xmax, nx* N))
    @constraint(model, fill(xmin, nx*N) .<= X)
    @objective(model, Min, 0.5*sum((V[i] - U_ref[i])^2 for i in 1:N*nu))

    function solver(init::Vector{Float64}, q::AbstractVector{Float64}; verbose=false)
        set_parameter_value.(x0, init)
        set_parameter_value.(U_ref, q)

        optimize!(model)

        if verbose
            println(solution_summary(model))
        end

        V_star = value.(V)
        distance_value = objective_value(model)

        return V_star, distance_value
    end
    return solver
end


## Three-operator splitting
function tos_step!(
    X_iter::Vector{Float64},
    Y::Vector{Float64},
    residual_vector::Vector{Float64},
    cost::DifferentiableQuadratic,
    project,
    x0::Vector{Float64},
    Z::Vector{Float64},
    lower_U::Vector{Float64},
    upper_U::Vector{Float64},
    mu::Float64,
    gamma::Float64,
)
    X_iter .= clamp.(Z, lower_U, upper_U)
    V_star, distance_value = project(x0, X_iter)

    D = (X_iter .- V_star) ./ mu
    R = 2.0 .* X_iter .- Z .- gamma .* D
    cost_factor = cholesky(Symmetric(Matrix{Float64}(I, length(Z), length(Z)) + gamma * cost.H))
    Y .= cost_factor \ (R .- gamma .* cost.h)
    residual_vector .= Y .- X_iter

    return V_star, distance_value
end

function backtracking_step!(
    mu::Float64,
    gamma::Float64,
    X_iter::Vector{Float64},
    Y::Vector{Float64},
    residual_vector::Vector{Float64},
    cost::DifferentiableQuadratic,
    project,
    x0::Vector{Float64},
    Z::Vector{Float64},
    lower_U::Vector{Float64},
    upper_U::Vector{Float64},
    ;
    minimum_gamma::Float64 = 1e-6,
    reduce_gamma::Float64 = 0.5,
)
    candidate_gamma = gamma
    backtracks = 0

    while true
        V_star, distance_x = tos_step!(
            X_iter,
            Y,
            residual_vector,
            cost,
            project,
            x0,
            Z,
            lower_U,
            upper_U,
            mu,
            candidate_gamma,
        )
        _V_y, distance_y = project(x0, Y)

        h_x = distance_x / mu
        h_y = distance_y / mu
        linear_model = (dot(residual_vector, X_iter) - dot(residual_vector, V_star)) / mu
        quadratic_model = dot(residual_vector, residual_vector) / (2.0 * candidate_gamma)
        line_search_margin = 1e-12 * max(1.0, abs(h_x), abs(h_y))

        if h_y <= h_x + linear_model + quadratic_model + line_search_margin || candidate_gamma <= minimum_gamma
            residual = norm(residual_vector) / max(1.0, norm(X_iter))
            return candidate_gamma, residual, V_star, backtracks
        end

        candidate_gamma = max(minimum_gamma, candidate_gamma * reduce_gamma)
        backtracks += 1
    end
end

function PGM_solver(
    problem::LinearMPC;
    mu::Float64 = 0.1,
    gamma::Float64 = 0.1,
    relaxation::Float64 = 1.0,
    adaptive::Bool=true,
    minimum_gamma::Float64=1e-6,
    reduce_gamma::Float64=0.5,
    increase_gamma::Float64=1.0,
    max_iter::Int=1000,
    tol::Float64=1e-3
)
    mu > 0.0 || throw(ArgumentError("mu must be positive"))
    gamma > 0.0 || throw(ArgumentError("gamma must be positive"))
    relaxation > 0.0 || throw(ArgumentError("relaxation must be positive"))

    nu = problem.nu
    N = problem.N
    nU = nu * N
    Z0 = zeros(Float64, nU)
    lower_U = fill(problem.umin, nU)
    upper_U = fill(problem.umax, nU)
    project = StateConstraint("OSQP", problem, 1e-6)

    return @inbounds function solver(x0::Vector{Float64}; data = nothing, verbose = false)
        cost, constant_cost = cost_cache(problem, x0)
        current_gamma = gamma

        Z = copy(Z0)
        X_iter = similar(Z)
        Y = similar(Z)
        residual_vector = similar(Z)
        residual = Inf
        converged = false
        iter_done = 0

        for iter in 1:max_iter
            if adaptive
                current_gamma *= increase_gamma
                current_gamma, residual, V_star, _backtracks = backtracking_step!(
                    mu,
                    current_gamma,
                    X_iter,
                    Y,
                    residual_vector,
                    cost,
                    project,
                    x0,
                    Z,
                    lower_U,
                    upper_U,
                    minimum_gamma = minimum_gamma,
                    reduce_gamma = reduce_gamma,
                )
            else
                V_star, _distance_x = tos_step!(X_iter, Y, residual_vector, cost, project, x0, Z, lower_U, upper_U, mu, current_gamma)
                residual = norm(residual_vector) / max(1.0, norm(X_iter))
            end

            if data !== nothing
                append_projection_sample!(data, x0, X_iter, V_star)
            end

            if residual <= tol
                converged = true
                iter_done = iter
                if verbose
                    println("TOS converged at iteration $iter")
                end
                break
            end

            Z .+= relaxation .* residual_vector
            iter_done = iter
        end

        U = reshape(copy(X_iter), nu, N)
        X = rollout(problem, x0, U)
        _V_final, distance_final = project(x0, vec(U))
        objective = constant_cost + cost(vec(U)) + distance_final / mu

        if verbose && !converged
            println("TOS stopped after $iter_done iterations with residual $residual")
        end

        return U, X, objective
    end
end
