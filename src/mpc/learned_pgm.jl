using LinearAlgebra

include("pgm.jl")
include(joinpath(@__DIR__, "..", "utils", "utils.jl"))

function learned_moreau_gradient(
    model::LearnedICNN,
    local_gradients::NTuple,
    system::LinearMPC,
    W::Matrix{Float64},
    x0::Vector{Float64},
)
    input = vcat(vec(W), x0)
    full_gradient = learned_moreau_full_gradient(model, local_gradients, input)
    return reshape(full_gradient[1:(system.nu * system.N)], system.nu, system.N)
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
    U = zeros(Float64, system.nu, system.N)
    W = similar(U)
    U_next = similar(U)
    objective_history = Float64[]
    gradient_model = icnn_from_learned(model)
    local_gradients = (gradient_struct(gradient_model, 1, length(model.a)),)
    start_time = time()
    solve_time = 0.0

    @inbounds for iter in 1:max_iter
        cost, grad = grad_cost(system, x0, U)
        push!(objective_history, cost)

        @. W = U - rho * grad
        learned_grad = learned_moreau_gradient(model, local_gradients, system, W, x0)
        @. U_next = W - rho * learned_grad
        solve_time = time() - start_time

        residual = norm(U_next - U)
        copyto!(U, U_next)
        

        if residual <= tol
            if verbose
                println("Learned PGM converged at iteration $iter")
            end
            break
        end
    end

    return (
        U = U,
        X = rollout(system, x0, U),
        objective_history = objective_history,
        solve_time = solve_time,
    )
end
