using LinearAlgebra

struct stage_cost
    Q::Matrix{Float64}
    R::Matrix{Float64}
end

function (obj::stage_cost)(x::AbstractVector, u::AbstractVector)
    return 0.5 * dot(x, obj.Q*x) + 0.5 * dot(u, obj.R*u)
end

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

    rho::Float64
    cost_func::stage_cost
end

function system_mag()
    A = [2 -1; 1 0.2]
    B = [1.0; 0.0;;]
    nx = size(A, 1)
    nu = size(B, 2)

    Q = Matrix{Float64}(I, nx, nx)
    R = [2.0;;]

    x0 = [3.0, 1.0]
    xmin = -5.0
    xmax = 5.0
    umin = -1.0
    umax = 1.0

    N = 10
    rho = 1.0
    cost_func = stage_cost(Q, R)

    return LinearMPC(
        A ,
        B ,
        Q ,
        R ,
        x0 ,
        xmin ,
        xmax ,
        umin ,
        umax ,
        nx ,
        nu ,
        N ,
        rho ,
        cost_func,
    )
end
