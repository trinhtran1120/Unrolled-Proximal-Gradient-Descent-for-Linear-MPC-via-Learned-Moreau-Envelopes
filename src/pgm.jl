using LinearAlgebra
using ProximalOperators

include("system.jl")

function converted_matrices(system::LinearMPC)
    nx = system.nx
    nu = system.nu
    N = system.N

    A_ss = zeros(Float64, N * nx, nx)
    B_ss = zeros(Float64, N * nx, N * nu)
    S = [zeros(Float64, nx, N * nu) for _ in 0:N]

    for k in 1:N
        row_range = (k - 1) * nx + 1:k * nx
        A_ss[row_range, :] .= system.A^k

        for j in 1:k
            col_range = (j - 1) * nu + 1:j * nu
            B_ss[row_range, col_range] .= system.A^(k - j) * system.B
        end

        S[k + 1] .= B_ss[row_range, :]
    end

    return A_ss, B_ss, S
end

function rollout(system::LinearMPC, x0::Vector{Float64}, U::Matrix{Float64})
    nx = system.nx
    N = system.N

    A_ss, B_ss, _ = converted_matrices(system)
    X_stacked = A_ss * x0 + B_ss * vec(U)

    X = zeros(Float64, nx, N + 1)
    X[:, 1] .= x0
    X[:, 2:end] .= reshape(X_stacked, nx, N)

    return X
end

function grad_cost(
    system::LinearMPC,
    x0::Vector{Float64},
    U::Matrix{Float64},
)
    nx = system.nx
    nu = system.nu
    N = system.N

    X = rollout(system, x0, U)
    cost = sum(system.cost_func(X[:, k], U[:, k]) for k in 1:N)

    grad = zeros(Float64, nu, N)
    p_next = zeros(Float64, nx)

    for k in N:-1:1
        xk = X[:, k]
        uk = U[:, k]

        grad[:, k] .= system.R * uk + system.B' * p_next
        p_next .= system.Q * xk + system.A' * p_next
    end

    return cost, grad
end

function projection(
    name,
    system::LinearMPC,
    x0_init::Vector{Float64},
    W_init::Matrix{Float64};
    tol::Float64 = 1e-6,
)

    nx = system.nx
    nu = system.nu
    N = system.N

    A_ss, B_ss, _ = converted_matrices(system)

    x_lower = fill(system.xmin, nx * N) .- A_ss * x0_init
    x_upper = fill(system.xmax, nx * N) .- A_ss * x0_init
    u_lower = fill(system.umin, nu * N)
    u_upper = fill(system.umax, nu * N)

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
    name,
    system::LinearMPC;
    x0::Vector{Float64} = system.x0,
    rho::Float64 = 0.05,
    max_iter::Int = 100,
    tol::Float64 = 1e-6,
    projection_tol::Float64 = 1e-6,
    verbose=false
)
    U = zeros(Float64, system.nu, system.N)
    project = projection(name, system, x0, U; tol = projection_tol)
    objective_history = Float64[]
    W_data = Vector{Vector{Float64}}()
    ME_data = Float64[]
    ME_grad_data = Vector{Vector{Float64}}()

    for iter in 1:max_iter
        cost, grad = grad_cost(system, x0, U)
        push!(objective_history, cost)

        W = U - rho * grad
        U_next = project(x0, W)
        moreau_value = 0.5 / rho * sum((U_next - W).^2)
        moreau_grad = (W - U_next) / rho

        push!(W_data, vec(copy(W)))
        push!(ME_data, moreau_value)
        push!(ME_grad_data, vec(copy(moreau_grad)))

        step_norm = norm(U_next - U)
        U .= U_next

        if step_norm <= tol
            if verbose
                println("Converged at iteration $iter")
            end
            break
        end
    end

    return (
        U = U,
        X = rollout(system, x0, U),
        objective_history = objective_history,
        W_data = W_data,
        ME_data = ME_data,
        ME_grad_data = ME_grad_data,
    )
end
