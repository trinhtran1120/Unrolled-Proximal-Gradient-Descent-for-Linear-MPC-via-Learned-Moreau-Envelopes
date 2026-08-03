using JuMP
import MathOptInterface as MOI

function linear_mpc_solver(name, mpc_para, tol)
    model = pick_solver(name, tol)
    A = mpc_para.A 
    B = mpc_para.B 
    x0_init = mpc_para.x0
    xmin = mpc_para.xmin
    xmax = mpc_para.xmax
    umin = mpc_para.umin
    umax = mpc_para.umax
    nx = mpc_para.nx
    nu = mpc_para.nu
    N = mpc_para.N
    cost_func = mpc_para.cost_func

    @variable(model, umin <= u[1:nu, 1:N] <= umax)
    @variable(model, xmin <= x[1:nx, 1:N+1] <= xmax)

    @variable(model, x0[i in 1:nx] in MOI.Parameter(x0_init[i]))
    @constraint(model, x[:, 1] .== x0)

    for k in 1:N
        @constraint(model, x[:, k+1] .== A*x[:, k] + B*u[:, k])
    end

    J = @expression(model, sum(cost_func(x[:, k], u[:, k]) for k in 1:N))
    @objective(model, Min, J)

    function solver(init::Vector{Float64}; verbose=false)
        set_parameter_value.(x0, init)
        optimize!(model)

        if verbose
            println(solution_summary(model))
        end

        return JuMP.value.(model[:x]), JuMP.value.(model[:u]), objective_value(model)
    end

    return solver
end
