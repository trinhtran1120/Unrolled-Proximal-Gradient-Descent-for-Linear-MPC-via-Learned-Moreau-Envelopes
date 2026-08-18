using LinearAlgebra
using Random

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
