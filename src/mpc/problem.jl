using LinearAlgebra

speye(N::Integer) = Matrix{Float64}(I, N, N)
repeat_horizon(v::AbstractVector{Float64}, N::Integer) = repeat(v, N)
as_column(v::AbstractVector{Float64}) = reshape(v, :, 1)

# Define the stage cost
struct StageCost
    Q::Matrix{Float64}
    R::Matrix{Float64}
    xr::Vector{Float64}
end

function (obj::StageCost)(x::AbstractVector, u::AbstractVector)
    dx = x - obj.xr
    return dot(dx, obj.Q * dx) + dot(u, obj.R * u)
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
    xr::Vector{Float64}
    xmin::Vector{Float64}
    xmax::Vector{Float64}
    umin::Vector{Float64}
    umax::Vector{Float64}

    nx::Int
    nu::Int
    N::Int

    cost_func::StageCost

    A_ro::Matrix{Float64}
    B_ro::Matrix{Float64}
end


function mpc_problem()
    A = [
        1.0      0.0      0.0  0.0  0.0  0.0  0.1     0.0      0.0  0.0     0.0     0.0;
        0.0      1.0      0.0  0.0  0.0  0.0  0.0     0.1      0.0  0.0     0.0     0.0;
        0.0      0.0      1.0  0.0  0.0  0.0  0.0     0.0      0.1  0.0     0.0     0.0;
        0.0488   0.0      0.0  1.0  0.0  0.0  0.0016  0.0      0.0  0.0992  0.0     0.0;
        0.0     -0.0488   0.0  0.0  1.0  0.0  0.0    -0.0016   0.0  0.0     0.0992  0.0;
        0.0      0.0      0.0  0.0  0.0  1.0  0.0     0.0      0.0  0.0     0.0     0.0992;
        0.0      0.0      0.0  0.0  0.0  0.0  1.0     0.0      0.0  0.0     0.0     0.0;
        0.0      0.0      0.0  0.0  0.0  0.0  0.0     1.0      0.0  0.0     0.0     0.0;
        0.0      0.0      0.0  0.0  0.0  0.0  0.0     0.0      1.0  0.0     0.0     0.0;
        0.9734   0.0      0.0  0.0  0.0  0.0  0.0488  0.0      0.0  0.9846  0.0     0.0;
        0.0     -0.9734   0.0  0.0  0.0  0.0  0.0    -0.0488   0.0  0.0     0.9846  0.0;
        0.0      0.0      0.0  0.0  0.0  0.0  0.0     0.0      0.0  0.0     0.0     0.9846
    ]
    B = [
        0.0     -0.0726   0.0      0.0726;
       -0.0726   0.0      0.0726   0.0;
       -0.0152   0.0152  -0.0152   0.0152;
        0.0     -0.0006  -0.0000   0.0006;
        0.0006   0.0     -0.0006   0.0;
        0.0106   0.0106   0.0106   0.0106;
        0.0     -1.4512   0.0      1.4512;
       -1.4512   0.0      1.4512   0.0;
       -0.3049   0.3049  -0.3049   0.3049;
        0.0     -0.0236   0.0      0.0236;
        0.0236   0.0     -0.0236   0.0;
        0.2107   0.2107   0.2107   0.2107
    ]
    Q = Diagonal([0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0]) |> Matrix
    R = 0.1 * speye(size(B, 2))

    xr = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    x0 = zeros(12)

    u_hover = 10.5916
    umin = fill(9.6 - u_hover, 4)
    umax = fill(13.0 - u_hover, 4)

    xmin = [-pi / 6, -pi / 6, -Inf, -Inf, -Inf, -1.0, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf]
    xmax = [pi / 6, pi / 6, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf]

    nx = size(A, 1)
    nu = size(B, 2)
    N = 10

    cost_func = StageCost(Q, R, xr)
    A_ro, B_ro = convert_matrix(A, B, N)

    return LinearMPC(
        A,
        B,
        Q,
        R,
        x0,
        xr,
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