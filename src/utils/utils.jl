include("matrix_tools.jl")

using Base.Threads
using JSON3
using LinearAlgebra
using SparseArrays

struct LearnedICNN
    U::Vector{Matrix{Float64}}
    W::Vector{Matrix{Float64}}
    b::Vector{Vector{Float64}}
    v::Vector{Float64}
    a::Vector{Float64}
    c::Float64
    rho::Float64
    input_mean::Vector{Float64}
    input_std::Vector{Float64}
    env_scale::Float64
end

softplus(x) = log1p(exp(-abs(x))) + max(x, zero(x))
sigmoid(x) = x >= 0 ? inv(1 + exp(-x)) : exp(x) / (1 + exp(x))

function json_matrix(value)
    return Float64.(reduce(hcat, value)')
end

function json_vector(value)
    return Float64.(collect(value))
end

function load_learned_icnn(path::AbstractString)
    data = JSON3.read(read(path, String))
    input_dim = length(data["a"])

    if haskey(data, :normalization)
        normalization = data["normalization"]
        input_mean = json_vector(normalization["input_mean"])
        input_std = json_vector(normalization["input_std"])
        env_scale = Float64(normalization["env_scale"])
    else
        input_mean = zeros(Float64, input_dim)
        input_std = ones(Float64, input_dim)
        env_scale = 1.0
    end

    return LearnedICNN(
        [json_matrix(layer) for layer in data["U"]],
        [json_matrix(layer) for layer in data["W"]],
        [json_vector(layer) for layer in data["b"]],
        json_vector(data["v"]),
        json_vector(data["a"]),
        Float64(data["c"]),
        Float64(data["rho"]),
        input_mean,
        input_std,
        env_scale,
    )
end

function moreau_envelope(model::LearnedICNN, input::AbstractVector)
    normalized_input = (input .- model.input_mean) ./ model.input_std
    z = softplus.(model.U[1] * normalized_input .+ model.b[1])

    for layer_index in 2:length(model.U)
        z = softplus.(
            model.W[layer_index] * z .+
            model.U[layer_index] * normalized_input .+
            model.b[layer_index]
        )
    end

    raw_output = dot(model.v, z) + dot(model.a, normalized_input) + model.c
    return model.env_scale * softplus(raw_output)
end

function icnn_from_learned(model::LearnedICNN)
    layers = [
        ICNN_Layer(model.U[layer_index], model.W[layer_index], model.b[layer_index])
        for layer_index in 2:length(model.U)
    ]

    return ICNN(model.U[1], model.b[1], layers, model.v, model.a, model.c)
end

function load_model(fname::String)
    data   = JSON3.read(read(fname, String))
    vecf64 = (Vector{Float64} ∘ vec)
    model = (
        U = convert_to_matrix.(data["U"]),
        W = convert_to_matrix.(data["W"]),
        a = vecf64(data["a"]), 
        b = vecf64.(data["b"]),
        c = Float64(data["c"]),
        v = vecf64(data["v"])
    )
    return Float64(data["rho"]), model
end

function convert_to_matrix(L)
    v = copy(hcat(L...)')
    isempty(v) ? Float64[] : v
end


struct ICNN_Layer
    U::Matrix{Float64}
    W::Matrix{Float64}
    b::Vector{Float64}
end

@inbounds function (m::ICNN_Layer)(x::Matrix{Float64}, z::Matrix{Float64})
    s = m.W * z + m.U * x .+ m.b
    return map(softplus, s), s  # convex & nondecreasing
end


struct ICNN
    U0::Matrix{Float64}
    b0::Vector{Float64}
    layers::Vector{ICNN_Layer}
    v::Vector{Float64}
    a::Vector{Float64}
    c::Float64
end

@inbounds function (m::ICNN)(x::Matrix{Float64})::Matrix{Float64}
    z = softplus.(m.U0 * x .+ m.b0)  # first layer (no state W)
    for layer in m.layers
        z, _ = layer(x, z)
    end
    return m.v' * z .+ m.a' * x .+ m.c
end

mutable struct gradient_struct
    lenlay::Int
    m::ICNN
    s_store::NTuple
    σ_store::NTuple 
    z_store::NTuple

    init_grad_x::Matrix{Float64}
    init_dL_dz::Matrix{Float64}  
    grad_x_buf::Matrix{Float64} 
    dL_curr::Matrix{Float64}
    dL_next::Matrix{Float64}
end

function gradient_struct(m::ICNN, nbatch::Int, dim::Int)

    U0     = m.U0
    layers = m.layers
    lenlay = length(layers)

    layer_rows = hcat(size(U0, 1), [size(layer.W, 1) for layer in layers])
    store_len  = length(layer_rows)

    s_store     = ntuple(i -> zeros(Float64, layer_rows[i], nbatch), store_len)
    σ_store     = ntuple(i -> zeros(Float64, layer_rows[i], nbatch), store_len)
    z_store     = ntuple(i -> zeros(Float64, layer_rows[i], nbatch), store_len)
    init_grad_x = repeat(m.a, 1, nbatch)
    init_dL_dz  = repeat(m.v, 1, nbatch)

    grad_x_buf  = zeros(Float64, dim, nbatch)
    dL_curr     = zeros(Float64, size(m.v, 1), nbatch)
    dL_next     = zeros(Float64, size(m.v, 1), nbatch)

    return gradient_struct(lenlay, m, s_store, σ_store, z_store, init_grad_x, init_dL_dz, grad_x_buf, dL_curr, dL_next)
end
precompile(gradient_struct, (ICNN, Int, Int))


@inbounds function mini_batch(
    local_gradients::NTuple,
    batch::AbstractMatrix;
    batch_size::Int = size(batch, 2),
)
    data_size = size(batch, 2)
    n_mb      = div(data_size - 1, batch_size) + 1

    if length(local_gradients) < n_mb
        throw(ArgumentError("local_gradients must contain at least $n_mb gradient evaluators"))
    end

    out       = copy(batch)

    @threads for i in 1:n_mb
        if i == n_mb
            @views out[:, (data_size - batch_size + 1):data_size] .= local_gradients[i](batch[:, (data_size - batch_size + 1):data_size])
        else
            @views out[:, (i - 1) * batch_size + 1:i * batch_size] .= local_gradients[i](batch[:, (i - 1) * batch_size + 1:i * batch_size])
        end
    end
    return out
end


function (obj::gradient_struct)(x::AbstractMatrix{Float64})

    s_first = obj.s_store[1]
    nL = obj.lenlay

    mul!(s_first,      obj.m.U0, x)
    add_bias!(s_first, obj.m.b0)
    activation_sigma!(obj.z_store[1], obj.σ_store[1], s_first)

    for i in 1:nL
        layer  = obj.m.layers[i]
        s_next = obj.s_store[i+1]
        z_prev = obj.z_store[i]

        mul!(s_next, layer.W, z_prev)
        mmul_add_matrix!(s_next, layer.U, x)
        add_bias!(s_next, layer.b)
        activation_sigma!(obj.z_store[i+1], obj.σ_store[i+1], s_next)
    end

    copyto!(obj.dL_curr,    obj.init_dL_dz)
    copyto!(obj.grad_x_buf, obj.init_grad_x)

    for i in nL:-1:1
        layer = obj.m.layers[i]
        dL_ds = obj.σ_store[i+1]
        hadamard!(dL_ds, obj.dL_curr)
        mmul_add_matrix!(obj.grad_x_buf, layer.U', dL_ds)

        mul!(obj.dL_next, layer.W', dL_ds)
        obj.dL_curr, obj.dL_next = obj.dL_next, obj.dL_curr
    end

    dL_ds_first = obj.σ_store[1]
    hadamard!(dL_ds_first, obj.dL_curr)
    mmul_add_matrix!(obj.grad_x_buf, obj.m.U0', dL_ds_first)

    return obj.grad_x_buf
end

@inline function LU_decomp(x)
    return lu(x)
end
precompile(LU_decomp, (SparseMatrixCSC{Float64, Int64},))