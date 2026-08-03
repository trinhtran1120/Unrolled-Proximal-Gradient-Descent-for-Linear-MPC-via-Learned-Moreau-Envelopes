using JuMP
import MathOptInterface as MOI 

include("problem.jl")
include("../utils/preprocess.jl")

function mpc_solver(name, problem, tol)
    # Build model
    model = pick_solver(name)

    # Extract data
    A = problem.A 
    B = problem.B 
    x0_init = problem.x0
    xmin = problem.xmin
    xmax = problem.xmax
    umin = problem.umin
    umax = problem.umax
    nx = problem.nx
    nu = problem.nu
    N = problem.N
    cost_func = problem.cost_func
    
    @variable(model, xmin[i] <= x[i = 1:nx, j = 1:(N + 1)] <= xmax[i])
    @variable(model, umin[i] <= u[i = 1:nu, j = 1:N] <= umax[i])

    @variable(model, x0[i in 1:nx] in MOI.Parameter(x0_init[i]))

    @constraint(model, x[:, 1] .== x0)

    for k in 1:N
        @constraint(model, x[:, k+1] .== A * x[:, k] + B * u[:, k])
    end

    J = @expression(model, sum(cost_func(x[:,k], u[:, k]) for k in 1:N))
    @objective(model, Min, J)

    function solver(init::Vector{Float64}; verbose=false)
        set_parameter_value.(x0, init)
        optimize!(model)

        if verbose
            println(solution_summary(model))
        end

        # status = termination_status(model)
        # if !(status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED))
        #     error("MPC solve failed with status: $status")
        # end

        return value.(x), value.(u), solve_time(model), objective_value(model)
    end

    return solver
end
