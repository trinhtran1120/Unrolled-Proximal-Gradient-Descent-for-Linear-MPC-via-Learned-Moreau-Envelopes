using LinearAlgebra
using ProximalAlgorithms
using ProximalOperators
using ProximalCore
using Printf

include("pgm.jl")
include("forward_backward.jl")
include("fast_forward_backward.jl")

function run_timed_iterator(name, iterator; max_iter::Int, tol::Float64)
    final_state = nothing
    final_iter = 0

    for (iter, state) in enumerate(iterator)
        final_state = state
        final_iter = iter

        residual = norm(state.res, Inf) / state.gamma

        if residual <= tol
            println("$name converged at iteration $iter")
            break
        end

        if iter >= max_iter
            println("$name stopped at max_iter = $max_iter")
            break
        end
    end
    println("\n$name timing")
    @printf("Iterations:         %d\n", final_iter)
    @printf("Forward time:       %.6f s\n", final_state.forward_time)
    @printf("Backward time:      %.6f s\n", final_state.backward_time)
    @printf("Total step time:    %.6f s\n", final_state.forward_time + final_state.backward_time)
    @printf("Forward calls:      %d\n", final_state.forward_calls)
    @printf("Backward calls:     %d\n", final_state.backward_calls)
    @printf("Avg forward/call:   %.6e s\n", final_state.forward_time / final_state.forward_calls)
    @printf("Avg backward/call:  %.6e s\n", final_state.backward_time / final_state.backward_calls)

    return nothing
end

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

ffb_iter = FastForwardBackwardIteration(
    f = f,
    g = g,
    x0 = u0,
    gamma = rho,
    adaptive = true,
)
println("=" ^30, "ForwardBackward", "=" ^30)
run_timed_iterator("ForwardBackward", fb_iter; max_iter = max_iter, tol = tol)
println("\n")
println("=" ^30, "FastForwardBackward", "=" ^30)
run_timed_iterator("FastForwardBackward", ffb_iter; max_iter = max_iter, tol = tol)
