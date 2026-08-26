using LinearAlgebra
using JSON3
using SpecialFunctions: erf


struct ProjectionMLP
    input_dim::Int
    parameter_dim::Int
    slack_dim::Int
    output_dim::Int
    weights::Vector{Matrix{Float64}}
    biases::Vector{Vector{Float64}}
    Gx::Matrix{Float64}
    b_offset::Vector{Float64}
    b_theta::Matrix{Float64}
    E::Matrix{Float64}
    E_pinv::Matrix{Float64}
    activation::String
    rho::Float64
end

gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
selu(x) = 1.0507009873554805 * (x > 0 ? x : 1.6732632423543772 * (exp(x) - 1))

function activate(x, activation::String)
    if activation == "gelu"
        return gelu(x)
    elseif activation == "selu"
        return selu(x)
    elseif activation == "tanh"
        return tanh(x)
    else
        throw(ArgumentError("unsupported activation: $activation"))
    end
end

function json_matrix(x)
    return Matrix{Float64}(reduce(hcat, Float64.(row) for row in x)')
end

function json_vector(x)
    return Vector{Float64}(x)
end

function json_field(data, name::Symbol, default)
    return hasproperty(data, name) && !isnothing(getproperty(data, name)) ? getproperty(data, name) : default
end

function load_projection_mlp(path::AbstractString)
    data = JSON3.read(read(path, String))
    data.model_type == "projection_mlp" || throw(ArgumentError("expected model_type = projection_mlp"))

    feasibility = data.feasibility
    Gx = json_matrix(feasibility.g_matrix)
    b_theta = json_matrix(hasproperty(feasibility, :b_theta) ? feasibility.b_theta : feasibility.b_para)
    E = hasproperty(feasibility, :E) ? json_matrix(feasibility.E) : hcat(Gx, Matrix{Float64}(I, size(Gx, 1), size(Gx, 1)))
    E_pinv = hasproperty(feasibility, :E_pinv) ? json_matrix(feasibility.E_pinv) : ((E * E') \ E)'
    rho = Float64(json_field(data, :rho_initial, json_field(data, :rho, json_field(data, :gamma, 1.0))))
    activation = String(json_field(data, :activation, "gelu"))

    return ProjectionMLP(
        Int(data.input_dim),
        Int(data.parameter_dim),
        Int(data.slack_dim),
        Int(data.output_dim),
        [json_matrix(W) for W in data.weights],
        [json_vector(b) for b in data.biases],
        Gx,
        json_vector(feasibility.b_offset),
        b_theta,
        E,
        E_pinv,
        activation,
        rho,
    )
end

function projection_mlp_forward(model::ProjectionMLP, input::AbstractVector, parameter::AbstractVector)
    length(input) == model.input_dim ||
        throw(DimensionMismatch("input has length $(length(input)); expected $(model.input_dim)"))
    length(parameter) == model.parameter_dim ||
        throw(DimensionMismatch("parameter has length $(length(parameter)); expected $(model.parameter_dim)"))

    z = vcat(input, parameter)
    for layer in 1:length(model.weights)-1
        z = activate.(model.weights[layer] * z .+ model.biases[layer], Ref(model.activation))
    end
    z_bar = model.weights[end] * z .+ model.biases[end]
    b = model.b_offset .+ model.b_theta * parameter[1:size(model.b_theta, 2)]
    z_hat = z_bar .- model.E_pinv * (model.E * z_bar .- b)
    # z_hat = z_bar
    return z_hat[1:model.input_dim], z_hat[model.input_dim + 1:end]
end

function projection_mlp_gradient(model::ProjectionMLP, input::AbstractVector, parameter::AbstractVector)
    V_tilde, _ = projection_mlp_forward(model, input, parameter)
    return (input .- V_tilde) ./ model.rho
end
