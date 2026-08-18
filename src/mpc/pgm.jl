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

    

function constraint(problem::LinearMPC, x0::Vector{Float64}; solver=:osqp)
    """Compute the constraints for the linear MPC problem"""
    A_ro = problem.A_ro
    B_ro = problem.B_ro
    xmin = problem.xmin
    xmax = problem.xmax
    umin = problem.umin
    umax = problem.umax
    nx = problem.nx
    nu = problem.nu
    N = problem.N

    x_lower = fill(xmin, nx * N) .- A_ro * x0
    x_upper = fill(xmax, nx * N) .- A_ro * x0

    u_lower = fill(umin, nu * N)
    u_upper = fill(umax, nu * N)

    return ProximalOperators.IndPolyhedral(
        x_lower,
        B_ro,
        x_upper,
        u_lower,
        u_upper;
        solver = solver,
    )
end


function PGM_solver(
    problem::LinearMPC;
    gamma::Float64=0.01, #stepsize
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

    return @inbounds function solver(x0::Vector{Float64}; data = nothing, verbose = false)
        f = single_shooting_cost(problem, x0)
        g = constraint(problem, x0; solver = :osqp)

        ffb_iter = ProximalAlgorithms.FastForwardBackwardIteration(
            f = f,
            g = g,
            x0 = u0,
            gamma = adaptive ? nothing : gamma,
            adaptive = adaptive,
            minimum_gamma = minimum_gamma,
            reduce_gamma = reduce_gamma,
            increase_gamma = increase_gamma,
        )

            for (iter, state) in enumerate(ffb_iter)
                u_solution .= state.z

                # Store psi(q; x0) = 0.5 * dist_F(x0)(q)^2. For an indicator, state.z = prox(g, state.y, gamma) = P_F(state.y), independent of gamma.
                d = state.y .- state.z                          # = y − P_F(y) (P_F: projection onto F)
                distance_grad = d
                distance_value = 0.5 * dot(d, d)                # = 0.5 * ||y - P_F(y)||^2

                if data !== nothing 
                        push!(data["q"], copy(state.y))
                        push!(data["x0"], copy(x0))
                        push!(data["proj"], copy(state.z))
                        push!(data["env"], distance_value)
                        push!(data["grad"], copy(distance_grad))
                end

                residual = norm(state.res, Inf) / state.gamma

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
            objective = f(u_solution)

            return U, X, objective
        end
end