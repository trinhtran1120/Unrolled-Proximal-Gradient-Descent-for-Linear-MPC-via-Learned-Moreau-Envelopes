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
    start_time = time()

    for iter in 1:max_iter
        cost, grad = grad_cost(system, x0, U)
        push!(objective_history, cost)

        W = U - rho * grad
        learned_grad = learned_moreau_gradient(model, system, W, x0)
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
    solve_time = time() - start_time

    return (
        U = U,
        X = rollout(system, x0, U),
        objective_history = objective_history,
        solve_time = solve_time,
    )
end

function learned_PGM(
    model_path::AbstractString,
    system::LinearMPC;
    kwargs...,
)
    return learned_PGM(load_learned_icnn(model_path), system; kwargs...)
end
