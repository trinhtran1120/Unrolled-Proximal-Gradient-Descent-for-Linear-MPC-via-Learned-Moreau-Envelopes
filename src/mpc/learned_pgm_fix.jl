using LinearAlgebra

include("pgm.jl")
include(joinpath(@__DIR__, "..", "utils", "utils.jl"))


function learned_grad_ME(
    model::LearnedPCF,
    system::LinearMPC,
    W::Matrix{Float64},
    x0::Vector{Float64},
    rho_backward::Real,
)
    input = vcat(vec(W), x0)
    full_gradient = learned_moreau_full_gradient(
        model,
        input,
        rho_backward,
    )
    control_dim = system.nu * system.N
    return reshape(full_gradient[1:control_dim], system.nu, system.N)
end


function learned_PGM(
    model::LearnedPCF,
    system::LinearMPC;
    x0::Vector{Float64} = system.x0,
    rho_forward::Float64 = model.rho,
    rho_backward::Float64 = model.rho,
    max_iter::Int = 100,
    tol::Float64 = 1e-6,
    verbose = false,
)
    U = zeros(Float64, system.nu, system.N)
    W = similar(U)
    U_next = similar(U)
    feasible_set = constraint(system, x0; solver = :osqp)

    objective_history = Float64[]
    residual_history = Float64[]

    start_time = time()
    solve_time = 0.0

    @inbounds for iter in 1:max_iter
        cost, grad = grad_cost(system, x0, U)
        push!(objective_history, cost)

        @. W = U - rho_forward * grad
        moreau_grad = learned_grad_ME(
            model,
            system,
            W,
            x0,
            rho_backward,
        )
        @. U_next = W - rho_backward * moreau_grad
        prox!(vec(U_next), feasible_set, vec(U_next), rho_backward)

        residual = norm(U_next - U, Inf)
        push!(residual_history, residual)
        copyto!(U, U_next)
        solve_time = time() - start_time

        if residual <= tol
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
        residual_history = residual_history,
        rho_forward = rho_forward,
        rho_backward = rho_backward,
        solve_time = solve_time,
    )
end
