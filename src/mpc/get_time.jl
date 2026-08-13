using LinearAlgebra
using ProximalAlgorithms
using ProximalOperators
using ProximalCore
using Printf

include("pgm.jl")
include("forward_backward.jl")
include("problem.jl")

rho = 0.005
max_iter = 2000
tol = 1e-2

problem = mpc_problem()

f = single_shooting_cost(problem, problem.x0)
g = constraint(problem, problem.x0; solver = :osqp)

u0 = zeros(Float64, problem.nu * problem.N)

fb_iter = ForwardBackwardIteration(
    f = f,
    g = g,
    x0 = u0,
    gamma = rho,
    adaptive = false,
)

final_state = nothing

for (iter, state) in enumerate(fb_iter)
    global final_state = state

    residual = norm(state.res, Inf) / state.gamma

    if residual <= tol
        println("Converged at iteration $iter")
        break
    end

    if iter >= max_iter
        println("Stopped at max_iter = $max_iter")
        break
    end
end

@printf("Forward time:       %.6f s\n", final_state.forward_time)
@printf("Backward time:      %.6f s\n", final_state.backward_time)
@printf("Total step time:    %.6f s\n", final_state.forward_time + final_state.backward_time)
@printf("Forward calls:      %d\n", final_state.forward_calls)
@printf("Backward calls:     %d\n", final_state.backward_calls)
@printf("Avg forward/call:   %.6e s\n", final_state.forward_time / final_state.forward_calls)
@printf("Avg backward/call:  %.6e s\n", final_state.backward_time / final_state.backward_calls)
