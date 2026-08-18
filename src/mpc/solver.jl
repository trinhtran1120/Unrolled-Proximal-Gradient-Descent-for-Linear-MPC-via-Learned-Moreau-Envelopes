using JuMP
import MathOptInterface as MOI 

include("problem.jl")
include("../utils/preprocess.jl")

function mpc_solver(name, problem, tol)
    # Build model
    model = pick_solver(name, tol)

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
    
    @variable(model, xmin .<= x[1:nx, 1:N+1] .<= xmax)
    @variable(model, umin .<= u[1:nu, 1:N] .<= umax)

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



function initialization(name, problem, tol)
    model = pick_solver(name, tol)
    nx, nu, N = problem.nx, problem.nu, problem.N

    @variable(model, problem.xmin <= x[1:nx, 1:N+1] <= problem.xmax)
    @variable(model, problem.umin <= u[1:nu, 1:N] <= problem.umax)
    @variable(model, x0[i in 1:nx] in MOI.Parameter(problem.x0[i]))

    @constraint(model, x[:, 1] .== x0)
    for k in 1:N
        @constraint(model, x[:, k+1] .== problem.A * x[:, k] + problem.B * u[:, k])
    end
    @objective(model, Min, 0.0)

    return function (init)
        set_parameter_value.(x0, init)
        optimize!(model)
        return termination_status(model) == MOI.OPTIMAL &&
               primal_status(model) == MOI.FEASIBLE_POINT
    end
end
