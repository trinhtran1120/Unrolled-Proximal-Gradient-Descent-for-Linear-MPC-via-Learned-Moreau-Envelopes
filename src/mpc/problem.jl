using LinearAlgebra
using SparseArrays

# Utility function
speye(N) = spdiagm(ones(N))

# Define the stage cost
struct StageCost
    Q::Matrix{Float64}
    R::Matrix{Float64}
    xr::Vector{Float64}
end

function (obj::StageCost)(x::AbstractVector, u::AbstractVector)
    return 0.5 * dot(x-obj.xr, obj.Q * (x-obj.xr)) + 0.5 * dot(u, obj.R * u)
end

# Define the MPC problem data
struct LinearMPC
    A::Matrix{Float64}
    B::Matrix{Float64}
    Q::Matrix{Float64}
    R::Matrix{Float64}

    x0::Vector{Float64}
    # xmin::Float64
    # xmax::Float64
    # umin::Float64
    # umax::Float64
    xmin::Vector{Float64}
    xmax::Vector{Float64}
    umin::Vector{Float64}
    umax::Vector{Float64}

    nx::Int
    nu::Int
    N::Int

    cost_func::StageCost
end

function mpc_problem()
    # A = [2.0 -1.0; 1.0 0.2]
    # B = [1.0; 0.0;;]
    # Q = Matrix{Float64}(I, 2, 2)
    # R = [2.0;;]

    # x0 = [3.0, 1.0]
    # xmin = -5.0
    # xmax = 5.0
    # umin = -1.0
    # umax = 1.0

    # nx = size(A, 1)
    # nu = size(B, 2)
    # Discrete time model of a quadcopter
    A = [1       0       0   0   0   0   0.1     0       0    0       0       0;
        0       1       0   0   0   0   0       0.1     0    0       0       0;
        0       0       1   0   0   0   0       0       0.1  0       0       0;
        0.0488  0       0   1   0   0   0.0016  0       0    0.0992  0       0;
        0      -0.0488  0   0   1   0   0      -0.0016  0    0       0.0992  0;
        0       0       0   0   0   1   0       0       0    0       0       0.0992;
        0       0       0   0   0   0   1       0       0    0       0       0;
        0       0       0   0   0   0   0       1       0    0       0       0;
        0       0       0   0   0   0   0       0       1    0       0       0;
        0.9734  0       0   0   0   0   0.0488  0       0    0.9846  0       0;
        0      -0.9734  0   0   0   0   0      -0.0488  0    0       0.9846  0;
        0       0       0   0   0   0   0       0       0    0       0       0.9846] |> sparse
    B = [0      -0.0726  0       0.0726;
        -0.0726  0       0.0726  0;
        -0.0152  0.0152 -0.0152  0.0152;
        0      -0.0006 -0.0000  0.0006;
        0.0006  0      -0.0006  0;
        0.0106  0.0106  0.0106  0.0106;
        0      -1.4512  0       1.4512;
        -1.4512  0       1.4512  0;
        -0.3049  0.3049 -0.3049  0.3049;
        0      -0.0236  0       0.0236;
        0.0236  0      -0.0236  0;
        0.2107  0.2107  0.2107  0.2107] |> sparse
    (nx, nu) = size(B)
    # Constraints
    u0 = 10.5916
    umin = [9.6, 9.6, 9.6, 9.6] .- u0
    umax = [13, 13, 13, 13] .- u0
    xmin = [[-pi/6, -pi/6, -Inf, -Inf, -Inf, -1]; -Inf .* ones(6)]
    xmax = [[pi/6,  pi/6,  Inf,  Inf,  Inf, Inf]; Inf .* ones(6)]

    # Objective function
    Q = spdiagm([0, 0, 10, 10, 10, 10, 0, 0, 0, 5, 5, 5])
    # QN = Q
    R = 0.1 * speye(nu)

    # Initial and reference states
    x0 = zeros(12)
    xr = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    N = 10

    cost_func = StageCost(Q, R, xr)

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
    )
end
