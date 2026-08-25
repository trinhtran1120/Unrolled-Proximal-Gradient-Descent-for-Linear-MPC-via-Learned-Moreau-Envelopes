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

include("problem.jl")
include("../utils/preprocess.jl")

## Rollout
function rollout(problem:LinearMPC, x0::Vector{Float64}, U::Matrix{Float64})
    nx = problem.nx
    N = problem.N 
    A_ro = problem.A_ro
    B_ro = problem.B_ro

    X_stacked = A_ro * x0 + B_ro * vec(U)

    X = zeros(Float64, nx, N+1)
    X[:, 1] == x0
    X[:, 2:end] .= reshape(X_stacked, nx, N)

    return X
end


struct DifferentiableQuadratic
    H::Matrix{Float64}
    h::Matrix{Float64}
end


function (f::DifferentiableQuadratic)(u)
    return 0.5 * dot(u, f.H * u) + dot(f.h, u)
end

function value_and_gradient(f::DifferentiableQuadratic, u)
    grad = f.H * u + f.H
    return f(u), grad
end


function single_shooting(problem::LinearMPC, x0::Vector{Float64})
    N = problem.N
    nu = problem.nu
            
    H = zeros(Float64, N*nu, N*nu)
    h = zeros(Float64, N*nu)

    for k in 1:N-1
        x_idx = (k - 1) * nx + 1: k * nx
        Ak = view(A_ro, x_idx, :)
        Bk = view(B_ro, x_idx, :)

        H .+= Bk' * Q * Bk
        h .+= Bk' * Q * Ak * x0
    end

    for k in 1:N
        u_idx = (k - 1) * nu: k * nu
        H[u_idx, u_idx] .+= R 
    end

    return DifferentiableQuadratic(H, h)

end

## Moreau Envelope
struct MoreauEnvelope
    cost::DifferentiableQuadratic
    prox::Function
    x0::Vector{Float64}
    rho::Float64
    data
end

function (f::MoreauEnvelope)(u)
    value, _ = value_and_gradient(f, u)
    
    return value
end


function value_and_gradient(f::MoreauEnvelope, u)
    cost

    return
end








## Projection onto the convex set C(x0)
function StateConstraint(name::String, problem::LinearMPC)
    model = pick_solver(name, tol=1e-6)
    A_ro = problem.A_ro
    B_ro = problem.B_ro
    nu = problem.nu
    N = problem.N

    @variable(model, V[1:N*nu])
    @variable(model, x0[i in 1:nx] in MOI.Parameter(problem.x0[i]))
    @variable(model, U_ref[i in 1:N*nu] in MOI.Parameter(0.0))

    # State bound
    X = A_ro * x0 + B_ro * V
    @constraint(model, X .<= fill(xmax, nx* N))
    @constraint(model, fill(xmin, nx*N) .<= X)
    @objective(model, Min, 0.5*sum((V[i] - U[i])^2) for i in 1:N*nu)
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


##