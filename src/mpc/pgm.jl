"""Proximal Gradient Method for Linear MPC.
minimize    f(U) + g(U)
where
    f: differentiable stage cost / objective
    g: indicator function of the feasible constraint set
"""

using LinearAlgebra
using OSQP
using JuMP
import MathOptInterface as MOI
using ProximalAlgorithms
using ProximalOperators

include("problem.jl")
include("../utils/preprocess.jl")


function StateConstraint(name, problem, tol::Float64 = 1e-8)
    solver_name = name isa Symbol ? String(name) : name
    solver_tol = max(tol, eps(Float64))
    model = pick_solver(solver_name, solver_tol)
    A_ro = problem.A_ro
    B_ro = problem.B_ro
    xmin = problem.xmin
    xmax = problem.xmax
    nx = problem.nx
    nu = problem.nu
    N = problem.N

    @variable(model, x0[i in 1:nx] in MOI.Parameter(problem.x0[i]))
    @variable(model, V[1:nu * N])
    @variable(model, U_ref[i in 1:nu * N] in MOI.Parameter(0.0))

    X = A_ro * x0 + B_ro * V
    @constraint(model, X .<= fill(xmax, nx * N))
    @constraint(model, fill(xmin, nx * N) .<= X)
    @objective(model, Min, 0.5 * sum((V[i] - U_ref[i])^2 for i in 1:nu * N))

    function solver(init::Vector{Float64}, q::AbstractVector{Float64}; verbose = false)
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


struct MoreauSmoothedCost
    cost::DifferentiableQuadratic
    project::Function
    x0::Vector{Float64}
    rho::Float64
    data
end

function (f::MoreauSmoothedCost)(u)
    value, _ = ProximalAlgorithms.value_and_gradient(f, u)
    return value
end

function ProximalAlgorithms.value_and_gradient(f::MoreauSmoothedCost, u)
    cost_value, cost_grad = ProximalAlgorithms.value_and_gradient(f.cost, u)
    projected_u, distance_value = f.project(f.x0, u)
    distance_grad = u .- projected_u
    value = cost_value + distance_value / f.rho
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


function rollout(problem::LinearMPC, x0::Vector{Float64}, U::Matrix{Float64})
    """Roll out the system from initial state x0 with control input U"""
    nx = problem.nx
    N = problem.N 

    X_stacked = problem.A_ro * x0 + problem.B_ro * vec(U)

    X = zeros(Float64, nx, N+1)
    X[:, 1] .= x0
    X[:, 2:end] .= reshape(X_stacked, nx, N)

    return X
end


function single_shooting_cost(problem::LinearMPC, x0::Vector{Float64})
    """Compute the cost of a single shooting trajectory"""
    nx = problem.nx
    nu = problem.nu
    N = problem.N
    Q = problem.Q
    R = problem.R

    H = zeros(Float64, N*nu, N*nu)
    h = zeros(Float64, N*nu)

    for k in 1:N-1
        x_idx = (k - 1) * nx + 1:k * nx

        Ak = view(problem.A_ro, x_idx, :)
        Bk = view(problem.B_ro, x_idx, :)

        H .+= Bk' * Q * Bk
        h .+= Bk' * Q * Ak * x0
    end

    for k in 1:N
        u_idx = (k - 1) * nu + 1:k * nu
        H[u_idx, u_idx] .+= R
    end

    return DifferentiableQuadratic(H, h)
end


function evaluate_cost_gradient(problem::LinearMPC, x0::Vector{Float64}, U::Matrix{Float64})
    """Evaluate the cost and its gradient for a given trajectory."""
    f = single_shooting_cost(problem, x0)
    variable_cost, grad = ProximalAlgorithms.value_and_gradient(f, vec(U))

    zero_U = zeros(Float64, problem.nu, problem.N)
    zero_X = rollout(problem, x0, zero_U)
    constant_cost = sum(problem.cost_func(zero_X[:, k], zero_U[:, k]) for k in 1:problem.N)

    return variable_cost + constant_cost, reshape(grad, problem.nu, problem.N)
end


function constraint(problem::LinearMPC; solver = "OSQP", tol::Float64 = 1e-8)
    """Return a projection solver for the state-feasible set C(x0)."""
    return StateConstraint(solver, problem, tol)
end


function PGM_solver(
    problem::LinearMPC;
    rho::Float64=0.1, #Moreau smoothing parameter
    gamma::Float64=0.01, #stepsize
    projection_tol::Float64=1e-8,
    adaptive::Bool=true, #use adaptive stepsize (backtracking)
    minimum_gamma::Float64=1e-6,
    reduce_gamma::Float64=0.5,
    increase_gamma::Float64=1.05,
    max_iter::Int=1000, #maximum number of iterations
    tol::Float64=1e-3, #tolerance for convergence
)
    """Solve the linear MPC problem using the Proximal Gradient Method"""

    nu = problem.nu
    N = problem.N
    u0 = zeros(Float64, nu * N)
    u_solution = copy(u0)
    input_constraint = ProximalOperators.IndBox(
        fill(problem.umin, nu * N),
        fill(problem.umax, nu * N),
    )
    project = constraint(problem; solver = "OSQP", tol = projection_tol)

    return @inbounds function solver(x0::Vector{Float64}; data = nothing, verbose = false)
        cost = single_shooting_cost(problem, x0)
        f = MoreauSmoothedCost(cost, project, x0, rho, data)

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
        final_f = MoreauSmoothedCost(cost, project, x0, rho, nothing)
        objective = final_f(u_solution)

        return U, X, objective
    end
end
