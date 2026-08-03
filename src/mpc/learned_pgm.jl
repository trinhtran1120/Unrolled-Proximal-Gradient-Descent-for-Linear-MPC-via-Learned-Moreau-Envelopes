using LinearAlgebra

include("pgm.jl")
include(joinpath(@__DIR__, "..", "utils", "utils.jl"))

function learned_input_dim(system::LinearMPC)
    return system.nu * system.N + system.nx
end

function validate_learned_pgm_dimensions(
    model::LearnedICNN,
    system::LinearMPC,
    W::Matrix{Float64},
    x0::Vector{Float64},
)
    expected_u_size = (system.nu, system.N)
    if size(W) != expected_u_size
        throw(DimensionMismatch("W must have size $expected_u_size, got $(size(W))"))
    end

    if length(x0) != system.nx
        throw(DimensionMismatch("x0 must have length $(system.nx), got $(length(x0))"))
    end

    expected_input_dim = learned_input_dim(system)
    model_input_dim = length(model.a)
    if model_input_dim != expected_input_dim
        throw(DimensionMismatch(
            "learned ICNN input dimension must be $expected_input_dim " *
            "(nu*N + nx = $(system.nu)*$(system.N) + $(system.nx)), got $model_input_dim",
        ))
    end
end

function learned_moreau_gradient(
    model::LearnedICNN,
    local_gradients::NTuple,
    system::LinearMPC,
    W::Matrix{Float64},
    x0::Vector{Float64},
)
    validate_learned_pgm_dimensions(model, system, W, x0)

    control_dim = system.nu * system.N
    input = vcat(vec(W), x0)
    full_gradient = learned_moreau_full_gradient(model, local_gradients, input)
    return reshape(full_gradient[1:control_dim], system.nu, system.N)
end

function learned_PGM(
    model::LearnedICNN,
    system::LinearMPC;
    x0::Vector{Float64} = system.x0,
    rho::Float64 = model.rho,
    max_iter::Int = 100,
    tol::Float64 = 1e-6,
    verbose = false,
)
    validate_learned_pgm_dimensions(model, system, zeros(Float64, system.nu, system.N), x0)

    U = zeros(Float64, system.nu, system.N)
    W = similar(U)
    U_next = similar(U)
    objective_history = Float64[]
    gradient_model = icnn_from_learned(model)
    local_gradients = (gradient_struct(gradient_model, 1, length(model.a)),)
    project = projection(system, x0, U)
    start_time = time()
    solve_time = 0.0

    @inbounds for iter in 1:max_iter
        cost, grad = grad_cost(system, x0, U)
        push!(objective_history, cost)

        @. W = U - rho * grad
        learned_grad = learned_moreau_gradient(model, local_gradients, system, W, x0)
        @. U_next = W - rho * learned_grad
        copyto!(U_next, project(x0, U_next))

        residual = norm(U_next - U)
        copyto!(U, U_next)
        solve_time = time() - start_time

        if residual <= tol
            if verbose
                println("Learned PGM converged at iteration $iter")
            end
            break
        end
    end

    copyto!(U, project(x0, U))
    solve_time = time() - start_time

    return (
        U = U,
        X = rollout(system, x0, U),
        objective_history = objective_history,
        solve_time = solve_time,
    )
end
