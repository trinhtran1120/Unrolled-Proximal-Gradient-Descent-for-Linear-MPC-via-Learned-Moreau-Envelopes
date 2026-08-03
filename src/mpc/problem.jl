using LinearAlgebra

# Define the stage cost
struct StageCost
    Q::Matrix{Float64}
    R::Matrix{Float64}
end

function (obj::StageCost)(x::AbstractVector, u::AbstractVector)
    return 0.5 * dot(x, obj.Q * x) + 0.5 * dot(u, obj.R * u)
end

# Define the MPC problem data
struct LinearMPC
    A::Matrix{Float64}
    B::Matrix{Float64}
    Q::Matrix{Float64}
    R::Matrix{Float64}

    x0::Vector{Float64}
    xmin::Int
    xmax::Int
    umin::Int
    umax::Int

    nx::Int
    nu::Int
    N::Int

    cost_func::StageCost
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
