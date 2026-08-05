using LinearAlgebra

# Define the stage cost
struct StageCost
    Q::Matrix{Float64}
    R::Matrix{Float64}
end

function (obj::StageCost)(x::AbstractVector, u::AbstractVector)
    return 0.5 * dot(x, obj.Q * x) + 0.5 * dot(u, obj.R * u)
end

function convert_matrix(
    A::Matrix{Float64},
    B::Matrix{Float64},
    N::Int,
)
    nx = size(A, 1)
    nu = size(B, 2)

    A_ro = zeros(Float64, N * nx, nx)
    B_ro = zeros(Float64, N * nx, N * nu)

    for k in 1:N
        row_idx = (k - 1) * nx + 1:k * nx
        A_ro[row_idx, :] .= A^k

        for j in 1:k
            col_idx = (j - 1) * nu + 1:j * nu
            B_ro[row_idx, col_idx] .= A^(k - j) * B
        end
    end

    return A_ro, B_ro
end

# Define the MPC problem data
struct LinearMPC
    A::Matrix{Float64}
    B::Matrix{Float64}
    Q::Matrix{Float64}
    R::Matrix{Float64}

    x0::Vector{Float64}
    xmin::Float64
    xmax::Float64
    umin::Float64
    umax::Float64

    nx::Int
    nu::Int
    N::Int

    cost_func::StageCost

    A_ro::Matrix{Float64}
    B_ro::Matrix{Float64}
end



function mpc_problem()
    A = [2.0 -1.0; 1.0 0.2]
    B = [1.0; 0.0;;]
    Q = Matrix{Float64}(I, 2, 2)
    R = [2.0;;]

    x0 = [3.0, 1.0]
    xmin = -5.0
    xmax = 5.0
    umin = -1.0
    umax = 1.0

    nx = size(A, 1)
    nu = size(B, 2)
    N = 10

    cost_func = StageCost(Q, R)
    A_ro, B_ro = convert_matrix(A, B, N)

    return LinearMPC(
        A,
        B,
        Q,
        R,
        x0,
        xmin,
        xmax,
        umin,
        umax,
        nx,
        nu,
        N,
        cost_func,
        A_ro,
        B_ro,
    )
end
