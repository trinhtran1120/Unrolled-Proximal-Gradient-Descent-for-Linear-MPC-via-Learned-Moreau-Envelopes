"""Proximal Gradient Method for Linear MPC.
minimize    f(U) + g(U)
where
    f: differentiable stage cost / objective
    g: indicator function of the feasible constraint set
"""

using LinearAlgebra
using ProximalAlgorithms
using ProximalCore
using ProximalOperators
using OSQP
using JuMP
import MathOptInterface as MOI

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

## Moreau Envelope
struct MoreauEnvelope
    cost::DifferentiableQuadratic
    project::Function
    x0::Vector{Float64}
    rho::Float64
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

        M_rho I_C(u) = min_{v in C(x0)} (1 / 2) ||v - u||_2^2

    with projection

        p = Pi_C(x0)(u) = argmin_{v in C(x0)} (1 / 2) ||v - u||_2^2.

    The returned value and gradient are

        phi(u) = l(u) + (1 / rho) M_rho I_C(u)
        grad phi(u) = grad l(u) + (1 / rho) (u - p),

    where l(u) = f.cost(u).
    """
    cost_value, cost_grad = ProximalAlgorithms.value_and_gradient(f.cost, u)
    projected_u, distance_value = f.project(f.x0, u)
    distance_grad = u .- projected_u
    value = cost_value + distance_value /f.rho
    grad = cost_grad .+ distance_grad ./ f.rho

    if f.data !== nothing
        push!(f.data["input"], copy(u))
        push!(f.data["parameter"], copy(f.x0))
        push!(f.data["proj"], copy(projected_u))
        push!(f.data["env"], distance_value)
        push!(f.data["grad"], copy(distance_grad))
    end

    return value, grad
end

## objective computation
function evaluate_cost_gradient(problem::LinearMPC, x0::Vector{Float64}, U::Matrix{Float64})
    """Evaluate the cost and gradient for a given trajectory"""
    f = single_shooting(problem, x0)
    val_cost, grad = ProximalAlgorithms.value_and_gradient(f, vec(U))

    zero_U = zeros(Float64, problem.nu, problem.N)
    zero_X = rollout(problem, x0, zero_U)

    constant_cost = sum(problem.cost_func(zero_X[:, k], zero_U[:, k]) for k in 1:problem.N)

    return val_cost + constant_cost, reshape(grad, problem.nu, problem.N)
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
    optimize!(model)

    function solver(init::Vector{Float64}, q::AbstractVector{Float64}; verbose=false)
        set_parameter_value.(x0, init)
        set_parameter_value.(U_ref, q)

        optimize!(model)

        if verbose
            println(solution_summary(model))
        end

        return value.(V), objective_value(model)
    end
    return solver
end


## PGM 
function PGM_solver(
    problem::LinearMPC;
    rho::Float64 = 0.1,
    gamma::Float64 = 0.1,
    adaptive::Bool=true,
    minimum_gamma::Float64=1e-6,
    reduce_gamma::Float64=0.5,
    increase_gamma::Float64=1.05,
    max_iter::Int=1000,
    tol::Float64=1e-3
)
    nu = problem.nu
    N = problem.N
    u0 = zeros(Float64, nu * N)
    u_solution = copy(u0)
    input_constraint = ProximalOperators.IndBox(
        fill(problem.umin, nu * N),
        fill(problem.umax, nu * N),
    )
    project = StateConstraint("OSQP", problem, 1e-6)

    return @inbounds function solver(x0::Vector{Float64}; data = nothing, verbose = false)
        cost = single_shooting_cost(problem, x0)
        f = MoreauEnvelope(cost, project, x0, rho, data)

        ffb_iter = ProximalAlgorithms.FastForwardBackwardIteration(
            f = f,
            g = input_constraint,
            x0 = u0,
            gamma = adaptive ? nothing : gamma,
            adaptive = adaptive,
            minimum_gamma = minimum_gamma,
            reduce_gamma = reduce_gamma,
            increase_gamma = increase_gamma,
        )

        for (iter, state) in enumerate(ffb_iter)
            u_solution .= state.z
            residual = norm(state.res, Inf) / state.gamma
            
            if data !== nothing
                haskey(data, "gamma") && push!(data["gamma"], state.gamma)
            end

            if residual <= tol
                if verbose
                    println("Converged at iteration $iter")
                end
                break
            end

            if iter >= max_iter
                break
            end
        end

        U = reshape(u_solution, nu, N)
        X = rollout(problem, x0, U)
        final_f = MoreauEnvelope(cost, project, x0, rho, nothing)
        objective = final_f(u_solution)

        return U, X, objective
    end
end