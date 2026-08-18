using LinearAlgebra
using ForwardDiff
using JSON3
using Random

struct Section
    start::Int
    stop::Int
    shape::Vector{Int}
end

struct LearnedPCF
    input_dim::Int
    parameter_dim::Int
    sections::NTuple{3,Vector{Section}}
    W_psi::Vector{Matrix{Float64}}
    V_psi::Vector{Matrix{Float64}}
    omega_psi::Vector{Vector{Float64}}
    input_mean::Vector{Float64}
    input_std::Vector{Float64}
    parameter_mean::Vector{Float64}
    parameter_std::Vector{Float64}
    env_scale::Float64
    rho::Float64
end

softplus(x) = log1p(exp(-abs(x))) + max(x, zero(x))

function json_matrix(x)
    return Matrix{Float64}(reduce(hcat, Float64.(row) for row in x)')
end

function json_vector(x)
    return Vector{Float64}(x)
end

function json_sections(group)
    return [
        Section(Int(section.start), Int(section["end"]), Int.(section.shape))
        for section in group
    ]
end

function row_reshape(values::AbstractVector, shape::Vector{Int})
    if length(shape) == 1
        return collect(values)
    end
    rows, cols = shape
    return Matrix(reshape(values, cols, rows)')
end

function load_pcf(path::AbstractString)
    data = JSON3.read(read(path, String))
    data.model_type == "pcf" || throw(ArgumentError("expected model_type = pcf"))
    data.target == "squared_distance" || throw(ArgumentError("expected target = squared_distance"))

    normalization = data.normalization
    hyper = data.hyper
    rho = isnothing(data.rho_initial) ? Float64(data.gamma) : Float64(data.rho_initial)

    return LearnedPCF(
        Int(data.input_dim),
        Int(data.parameter_dim),
        (
            json_sections(data.sections[1]),
            json_sections(data.sections[2]),
            json_sections(data.sections[3]),
        ),
        [json_matrix(W) for W in hyper.W_psi],
        [json_matrix(V) for V in hyper.V_psi],
        [json_vector(omega) for omega in hyper.omega_psi],
        json_vector(normalization.input_mean),
        json_vector(normalization.input_std),
        json_vector(normalization.parameter_mean),
        json_vector(normalization.parameter_std),
        Float64(normalization.env_scale),
        rho,
    )
end

function pcf_psi(model::LearnedPCF, parameter::AbstractVector)
    out = model.V_psi[1] * parameter + model.omega_psi[1]
    for layer in 2:length(model.V_psi)
        out = softplus.(out)
        out = model.W_psi[layer - 1] * out + model.V_psi[layer] * parameter + model.omega_psi[layer]
    end
    return vec(out)
end

function pcf_unpack(model::LearnedPCF, emitted::AbstractVector)
    sections_W, sections_V, sections_omega = model.sections
    W = Matrix[]
    V = Matrix[]
    omega = Vector[]

    for section in sections_W
        raw = row_reshape(emitted[section.start + 1:section.stop], section.shape)
        push!(W, softplus.(raw) ./ section.shape[2])
    end
    for section in sections_V
        value = row_reshape(emitted[section.start + 1:section.stop], section.shape)
        push!(V, value ./ sqrt(section.shape[2]))
    end
    for section in sections_omega
        push!(omega, row_reshape(emitted[section.start + 1:section.stop], section.shape))
    end

    return W, V, omega
end

function pcf_value_norm(model::LearnedPCF, input::AbstractVector, parameter::AbstractVector)
    W, V, omega = pcf_unpack(model, pcf_psi(model, parameter))

    z = softplus.(V[1] * input + omega[1])
    for layer in 2:length(V)-1
        z = softplus.(W[layer - 1] * z + V[layer] * input + omega[layer])
    end

    return softplus(only(W[end] * z + V[end] * input + omega[end]))
end

function pcf_value(model::LearnedPCF, input::AbstractVector, parameter::AbstractVector)
    input_norm = (input .- model.input_mean) ./ model.input_std
    parameter_norm = (parameter .- model.parameter_mean) ./ model.parameter_std
    return model.env_scale * pcf_value_norm(model, input_norm, parameter_norm)
end

function pcf_grad(model::LearnedPCF, input::AbstractVector, parameter::AbstractVector)
    length(input) == model.input_dim || throw(DimensionMismatch("input has length $(length(input)); expected $(model.input_dim)"))
    length(parameter) == model.parameter_dim || throw(DimensionMismatch("parameter has length $(length(parameter)); expected $(model.parameter_dim)"))

    input_norm = (input .- model.input_mean) ./ model.input_std
    parameter_norm = (parameter .- model.parameter_mean) ./ model.parameter_std
    grad_norm = ForwardDiff.gradient(q -> pcf_value_norm(model, q, parameter_norm), input_norm)
    return grad_norm .* model.env_scale ./ model.input_std
end

function filter_zero!(data; env_tol, grad_tol, zero_to_nonzero_ratio, seed = 1234)
    sample_count = length(data["env"])
    is_nonzero = [
        env > env_tol || norm(grad, Inf) > grad_tol
        for (env, grad) in zip(data["env"], data["grad"])
    ]

    nonzero_indices = findall(is_nonzero)
    zero_indices = findall(!, is_nonzero)

    max_zero = min(length(zero_indices), round(Int, zero_to_nonzero_ratio * length(nonzero_indices)))
    rng = MersenneTwister(seed)
    retained_zero_indices = max_zero == 0 ? Int[] : shuffle(rng, zero_indices)[1:max_zero]
    retained_indices = sort!(vcat(nonzero_indices, retained_zero_indices))

    for key in keys(data)
        data[key] = data[key][retained_indices]
    end

    return (
        generated = sample_count,
        informative = length(nonzero_indices),
        retained_near_zero = length(retained_zero_indices),
        removed_near_zero = length(zero_indices) - length(retained_zero_indices),
        retained = length(retained_indices),
    )
end
