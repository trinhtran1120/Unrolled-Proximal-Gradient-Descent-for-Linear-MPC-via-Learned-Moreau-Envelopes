using LinearAlgebra
using ProximalAlgorithms
using ProximalOperators

include("problem.jl")

function ProximalAlgorithms.value_and_gradient(
    f::ProximalOperators.Quadratic,
    x::AbstractVector,
)
    grad, value = ProximalOperators.gradient(f, x)
    return value, grad
end

function converted_matrices(problem::LinearMPC)
    nx = problem.nx
    nu = problem.nu
    N = problem.N 
    A = problem.A 
    B = problem.B 

    A_converted = zeros(Float64, N*nx, nx)
    B_converted = zeros(Float64, N*nx, N*nu)
    S = [zeros(Float64, nx, N*nu) for _ in 0:N]

    for k in 1:N 
        row_idx = (k - 1) * nx + 1:k * nx
        A_converted[row_idx, :] .= A^k

        for j in 1:k
            col_idx = (j - 1) * nu + 1:j*nu
            B_converted[row_idx, col_idx] .= A^(k - j) * B 
        end
        S[k + 1] .= B_converted[row_idx, :]   
    end
    
    return A_converted, B_converted, S
end

function rollout(problem::LinearMPC, x0::Vector{Float64}, U::Matrix{Float64})
    nx = problem.nx
    N = problem.N 
    
    A_ss, B_ss, _ = converted_matrices(problem)
    X_stacked = A_ss * x0 + B_ss * vec(U)

    X = zeros(Float64, nx, N + 1)
    X[:, 1] .= x0
    X[:, 2:end] .= reshape(X_stacked, nx, N)

    return X
end

function single_shooting_cost(problem::LinearMPC, x0::Vector{Float64})
    nx = problem.nx
    nu = problem.nu
    N = problem.N
    Q = problem.Q
    R = problem.R
    A_ss, B_ss, _ = converted_matrices(problem)

    H = zeros(Float64, N*nu, N*nu)
    h = zeros(Float64, N*nu)

    for k in 1:N-1
        x_idx = (k - 1) * nx + 1:k * nx

        Ak = view(A_ss, x_idx, :)
        Bk = view(B_ss, x_idx, :)

        H .+= Bk' * Q * Bk
        h .+= Bk' * Q * Ak * x0
    end

    for k in 1:N
        u_idx = (k - 1) * nu + 1:k * nu
        H[u_idx, u_idx] .+= R
    end

    return ProximalOperators.Quadratic(H, h)
end

function condensed_constraints(problem::LinearMPC, x0::Vector{Float64}; solver = :osqp)
    A_ss, B_ss, _ = converted_matrices(problem)

    x_lower = fill(problem.xmin, problem.nx * problem.N) .- A_ss * x0
    x_upper = fill(problem.xmax, problem.nx * problem.N) .- A_ss * x0

    u_lower = fill(problem.umin, problem.nu * problem.N)
    u_upper = fill(problem.umax, problem.nu * problem.N)

    return IndPolyhedral(
        x_lower,
        B_ss,
        x_upper,
        u_lower,
        u_upper;
        solver = solver,
    )
end


function PGM(
    problem::LinearMPC,
    x0::Vector{Float64};
    rho::Float64 = 0.001,
    max_iter::Int = 1000,
    tol::Float64 = 5e-4,
    verbose = false,
)
    f = single_shooting_cost(problem, x0)
    g = condensed_constraints(problem, x0; solver = :osqp)

    # ffb_iter = ProximalAlgorithms.FastForwardBackwardIteration(
    #     f = f,
    #     g = g,
    #     x0 = zeros(Float64, problem.nu * problem.N),
    #     gamma = rho,
    #     adaptive = true,
    #     minimum_gamma = 1e-6,
    #     reduce_gamma = 0.5,
    #     increase_gamma = 1.05,
    # )
    

    start_time = time()
    solve_time = 0.0
    u_solution = copy(ffb_iter.x0)
    iterations = 0
    objective_history = Float64[]
    W_data = Vector{Vector{Float64}}()
    ME_data = Float64[]
    ME_grad_data = Vector{Vector{Float64}}()
    gamma_data = Float64[]

    for (iter, state) in enumerate(ffb_iter)
        iterations = iter
        solve_time = time() - start_time
        u_solution .= state.z

        value, _gradient = ProximalAlgorithms.value_and_gradient(f, state.x)
        push!(objective_history, value)
        push!(W_data, copy(state.y))
        push!(ME_data, 0.5 / state.gamma * sum((state.z .- state.y) .^ 2))
        push!(ME_grad_data, (state.y .- state.z) ./ state.gamma)
        push!(gamma_data, state.gamma)

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

    U = reshape(u_solution, problem.nu, problem.N)
    X = rollout(problem, x0, U)

    return (
        U = U,
        X = X,
        objective_value = f(u_solution),
        objective_history = objective_history,
        W_data = W_data,
        ME_data = ME_data,
        ME_grad_data = ME_grad_data,
        gamma_data = gamma_data,
        iterations = iterations,
        solve_time = solve_time,
    )
end
