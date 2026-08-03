using LinearAlgebra
using ProximalOperators

include("problem.jl")

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

function grad_cost(problem::LinearMPC, x0::Vector{Float64}, U::Matrix{Float64})
    nx = problem.nx
    nu = problem.nu
    N = problem.N

    X = rollout(problem, x0, U)
    cost = sum(problem.cost_func(X[:, k], U[:, k]) for k in 1:N)

    grad = zeros(Float64, nu, N)
    p_next = zeros(Float64, nx)

    for k in N:-1:1
        xk = X[:, k]
        uk = U[:, k]

        grad[:, k] .= problem.R * uk + problem.B' * p_next
        p_next .= problem.Q * (xk - problem.xr) + problem.A' * p_next
    end

    return cost, grad
end


function projection( 
    problem::LinearMPC,
    x0_init::Vector{Float64},
    W_init::Matrix{Float64},
)
    nx = problem.nx
    nu = problem.nu
    N = problem.N

    A_ss, B_ss, _ = converted_matrices(problem)

    x_lower = repeat(problem.xmin, N) .- (A_ss * x0_init)
    x_upper = repeat(problem.xmax, N) .- (A_ss * x0_init)
    u_lower = repeat(problem.umin, N)
    u_upper = repeat(problem.umax, N)

    feasible_set = IndPolyhedral(
        x_lower,
        B_ss,
        x_upper,
        u_lower,
        u_upper;
        solver = :osqp,
    )

    function solver(init::Vector{Float64}, W_value::Matrix{Float64}; verbose=false)

        projected_u, _ = prox(feasible_set, vec(W_value))

        return reshape(projected_u, nu, N)
    end

    return solver
end

function PGM(
    problem::LinearMPC,
    x0::Vector{Float64};
    rho::Float64 = 0.05,
    max_iter::Int = 1000,
    tol::Float64 = 1e-6,
    verbose=false
)
    U = zeros(Float64, problem.nu, problem.N)

    project = projection(problem, x0, U)

    objective_history = Float64[]

    W_data = Vector{Vector{Float64}}()
    ME_data = Float64[]
    ME_grad_data = Vector{Vector{Float64}}()
    start_time = time()
    solve_time = 0.0

    for iter in 1:max_iter
        cost, grad = grad_cost(problem, x0, U)
        push!(objective_history, cost)

        W = U - rho * grad
        U_next = project(x0, W)
        moreau_value = 0.5 / rho * sum((U_next - W).^2)
        moreau_grad = (W - U_next) / rho

        # collect data
        push!(W_data, vec(copy(W)))
        push!(ME_data, moreau_value)
        push!(ME_grad_data, vec(copy(moreau_grad)))

        residual = norm(U_next - U)
        U .= U_next
        solve_time = time() - start_time

        if residual <= tol
            if verbose
                println("Converged at iteration $iter")
            end
            break
        end
    end

    return (
        U = U,
        X = rollout(problem, x0, U),
        objective_history = objective_history,
        W_data = W_data,
        ME_data = ME_data,
        ME_grad_data = ME_grad_data,
        solve_time = solve_time,
    )
end
