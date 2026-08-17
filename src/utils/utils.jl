using JSON3
using LinearAlgebra


struct LearnedPCF
    W_psi::Vector{Matrix{Float64}}
    V_psi::Vector{Matrix{Float64}}
    omega_psi::Vector{Vector{Float64}}
    psi_matrices::Vector{Matrix{Float64}}
    psi_biases::Vector{Vector{Float64}}
    shapes::Vector{Vector{Int}}
    convex_widths::Vector{Int}
    q_dim::Int
    theta_dim::Int
    rho::Float64
    q_mean::Vector{Float64}
    q_std::Vector{Float64}
    theta_mean::Vector{Float64}
    theta_std::Vector{Float64}
    env_scale::Float64
    target::String
end


softplus(x) = log1p(exp(-abs(x))) + max(x, zero(x))
sigmoid(x) = x >= 0 ? inv(1 + exp(-x)) : exp(x) / (1 + exp(x))


function json_matrix(value)
    return Float64.(reduce(hcat, value)')
end


function json_vector(value)
    return Float64.(collect(value))
end


function json_int_vector(value)
    return Int.(collect(value))
end


function load_learned_pcf(path::AbstractString)
    data = JSON3.read(read(path, String))
    normalization = data["normalization"]
    hyper = data["hyper"]

    has_skip_psi = haskey(hyper, :W_psi) && haskey(hyper, :V_psi) && haskey(hyper, :omega_psi)
    W_psi = has_skip_psi ? [json_matrix(layer) for layer in hyper[:W_psi]] : Matrix{Float64}[]
    V_psi = has_skip_psi ? [json_matrix(layer) for layer in hyper[:V_psi]] : Matrix{Float64}[]
    omega_psi = has_skip_psi ? [json_vector(layer) for layer in hyper[:omega_psi]] : Vector{Float64}[]

    psi_matrix_key = haskey(hyper, :psi_matrices) ? :psi_matrices : :a
    psi_bias_key = haskey(hyper, :psi_biases) ? :psi_biases : :b
    psi_matrices = has_skip_psi ? Matrix{Float64}[] : [json_matrix(layer) for layer in hyper[psi_matrix_key]]
    psi_biases = has_skip_psi ? Vector{Float64}[] : [json_vector(layer) for layer in hyper[psi_bias_key]]

    return LearnedPCF(
        W_psi,
        V_psi,
        omega_psi,
        psi_matrices,
        psi_biases,
        [json_int_vector(shape) for shape in data["shapes"]],
        json_int_vector(data["convex_widths"]),
        Int(data["q_dim"]),
        Int(data["theta_dim"]),
        Float64(data["rho"]),
        json_vector(normalization["q_mean"]),
        json_vector(normalization["q_std"]),
        json_vector(normalization["theta_mean"]),
        json_vector(normalization["theta_std"]),
        Float64(normalization["env_scale"]),
        String(data["target"]),
    )
end


function pcf_hyper_forward(model::LearnedPCF, theta::AbstractVector)
    if !isempty(model.V_psi)
        out = model.V_psi[1] * theta .+ model.omega_psi[1]
        for layer_index in 2:length(model.V_psi)
            out = max.(0.0, out)
            out = (
                model.W_psi[layer_index - 1] * out
                .+ model.V_psi[layer_index] * theta
                .+ model.omega_psi[layer_index]
            )
        end
        return out
    end

    h = Vector{Float64}(theta)
    for layer_index in 1:length(model.psi_matrices)-1
        h = max.(
            0.0,
            model.psi_matrices[layer_index] * h .+ model.psi_biases[layer_index],
        )
    end
    return model.psi_matrices[end] * h .+ model.psi_biases[end]
end


function take_row_major_matrix(emitted::AbstractVector, offset::Int, rows::Int, cols::Int)
    last_index = offset + rows * cols - 1
    values = @view emitted[offset:last_index]
    return copy(reshape(values, cols, rows)')
end


function take_vector(emitted::AbstractVector, offset::Int, len::Int)
    last_index = offset + len - 1
    return collect(@view emitted[offset:last_index])
end


function unpack_pcf_weights(model::LearnedPCF, emitted::AbstractVector)
    offset = 1
    shape_idx = 1
    W = Matrix{Float64}[]
    V = Matrix{Float64}[]
    omega = Vector{Float64}[]

    for layer_idx in 1:length(model.convex_widths)
        if layer_idx > 1
            shape = model.shapes[shape_idx]
            push!(W, take_row_major_matrix(emitted, offset, shape[1], shape[2]))
            offset += prod(shape)
            shape_idx += 1
        end

        shape = model.shapes[shape_idx]
        push!(V, take_row_major_matrix(emitted, offset, shape[1], shape[2]))
        offset += prod(shape)
        shape_idx += 1

        shape = model.shapes[shape_idx]
        push!(omega, take_vector(emitted, offset, shape[1]))
        offset += prod(shape)
        shape_idx += 1
    end

    shape = model.shapes[shape_idx]
    W_out = take_row_major_matrix(emitted, offset, shape[1], shape[2])
    offset += prod(shape)
    shape_idx += 1

    shape = model.shapes[shape_idx]
    V_out = take_row_major_matrix(emitted, offset, shape[1], shape[2])
    offset += prod(shape)
    shape_idx += 1

    shape = model.shapes[shape_idx]
    omega_out = take_vector(emitted, offset, shape[1])

    return (W = W, V = V, omega = omega, W_out = W_out, V_out = V_out, omega_out = omega_out)
end


function pcf_value_and_grad_normalized(
    model::LearnedPCF,
    q_norm::AbstractVector,
    theta_norm::AbstractVector,
)
    weights = unpack_pcf_weights(model, pcf_hyper_forward(model, theta_norm))

    pre = weights.V[1] * q_norm .+ weights.omega[1]
    z = softplus.(pre)
    dz_dq = Diagonal(sigmoid.(pre)) * weights.V[1]

    for layer_idx in 2:length(weights.V)
        W_pos = softplus.(weights.W[layer_idx - 1])
        pre = W_pos * z .+ weights.V[layer_idx] * q_norm .+ weights.omega[layer_idx]
        dpre_dq = W_pos * dz_dq .+ weights.V[layer_idx]
        z = softplus.(pre)
        dz_dq = Diagonal(sigmoid.(pre)) * dpre_dq
    end

    W_out = vec(softplus.(weights.W_out[1, :]))
    value = dot(W_out, z) + dot(vec(weights.V_out[1, :]), q_norm) + weights.omega_out[1]
    grad_q = W_out' * dz_dq .+ vec(weights.V_out[1, :])'
    return value, vec(grad_q)
end


function moreau_envelope(model::LearnedPCF, input::AbstractVector, rho_backward::Real)
    q = input[1:model.q_dim]
    theta = input[model.q_dim+1:model.q_dim+model.theta_dim]
    q_norm = (q .- model.q_mean) ./ model.q_std
    theta_norm = (theta .- model.theta_mean) ./ model.theta_std
    value_norm, _ = pcf_value_and_grad_normalized(model, q_norm, theta_norm)
    return model.env_scale * value_norm
end


function learned_moreau_full_gradient(
    model::LearnedPCF,
    input::AbstractVector,
    rho_backward::Real,
)
    q = input[1:model.q_dim]
    theta = input[model.q_dim+1:model.q_dim+model.theta_dim]
    q_norm = (q .- model.q_mean) ./ model.q_std
    theta_norm = (theta .- model.theta_mean) ./ model.theta_std
    _, grad_q_norm = pcf_value_and_grad_normalized(model, q_norm, theta_norm)
    grad_q = grad_q_norm .* model.env_scale ./ model.q_std
    return vcat(grad_q, zeros(Float64, model.theta_dim))
end
