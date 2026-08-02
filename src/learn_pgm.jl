using ForwardDiff
using LinearAlgebra

include("pgm.jl")
include("utlis.jl")

function learned_moreau_gradient(
    model::LearnedICNN,
    system::LinearMPC,
    W::Matrix{Float64},
    x0::Vector{Float64},
)
    input = vcat(vec(W), x0)
    full_gradient = ForwardDiff.gradient(z -> moreau_envelope(model, z), input)
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
    objective_history = Float64[]
    gradient_time_history = Float64[]

    for iter in 1:max_iter
        cost, grad = grad_cost(system, x0, U)
        push!(objective_history, cost)

        W = U - rho * grad
        gradient_start = time_ns()
        learned_grad = learned_moreau_gradient(model, system, W, x0)
        gradient_time = (time_ns() - gradient_start) / 1e9
        push!(gradient_time_history, gradient_time)
        U_next = W - rho * learned_grad

        step_norm = norm(U_next - U)
        U .= U_next

        if step_norm <= tol
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
        gradient_time_history = gradient_time_history,
        total_gradient_time = sum(gradient_time_history),
    )
end

function learned_PGM(
    model_path::AbstractString,
    system::LinearMPC;
    kwargs...,
)
    return learned_PGM(load_learned_icnn(model_path), system; kwargs...)
end
