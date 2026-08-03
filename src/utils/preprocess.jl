using LinearAlgebra

function pick_solver(name, tol::Float64 = 1e-6)
    #All solvers should share the same tolerance for stopping criteria
    str = lowercase(name)

    if str == "ipopt"      #for LP, QP, NLP
        model = Model(Ipopt.Optimizer)
    elseif str == "madnlp" #for NLP
        model = Model(()->MadNLP.Optimizer())
        MOI.set(model, MOI.Silent(), false)
        # set_optimizer_attribute(model, "print_level", 5) 

        # set_optimizer_attribute(model, "blas_num_threads", nthreads())      
        set_optimizer_attribute(model, "tol", tol) 
        # set_optimizer_attribute(model, "nlp_scaling", false)
    elseif str == "gurobi"
        model = Model(Gurobi.Optimizer)
    elseif str =="mosek"
        model = Model(Mosek.Optimizer)
    elseif str == "osqp"   #for LP, QP 
        model = Model(OSQP.Optimizer); 
        set_optimizer_attribute(model, "eps_abs", tol); 
        set_optimizer_attribute(model, "eps_rel", tol)

    elseif str == "clarabel" #for NLP
        model = Model(Clarabel.Optimizer)

    elseif str == "ecos" #for NLP
        model = Model(ECOS.Optimizer)

    else
        error("Specified solver is not supported")
    end

    set_silent(model)

    return model
end  
