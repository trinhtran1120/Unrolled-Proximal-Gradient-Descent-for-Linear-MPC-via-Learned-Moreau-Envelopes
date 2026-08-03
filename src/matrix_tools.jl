@inline function add_bias!(mat, bias)
    cols = size(mat, 2)
    @inbounds @simd for j in 1:cols
        @views mat[:, j] .+= bias
    end
    return mat
end

@inline function activation_sigma!(activ, sigma_buf, preactiv)
    @inbounds @simd for i in eachindex(activ)
        val          = preactiv[i]
        activ[i]     = softplus(val)
        sigma_buf[i] = sigmoid(val)
    end
    return activ
end
# precompile(activation_sigma!, (Function, MMatrix{10,10,Float64,100}, MMatrix{10,10,Float64,100}))


@inline function hadamard!(dest, rhs)
    @inbounds @simd for i in eachindex(dest)
        dest[i] *= rhs[i]
    end
    return dest
end
# precompile(hadamard!, (MMatrix{10,10,Float64,100}, MMatrix{10,10,Float64,100}))



@inline function mmul_add_matrix!(dest, src1, src2)
    rows = size(dest, 1)
    cols = size(dest, 2)


    @inbounds for j in 1:cols, i in 1:rows
        @views dest[i, j] += dot(src1[i, :], src2[:,j])
    end
    return dest
end
# precompile(mul_add_matrix!, (MMatrix{10,10,Float64,100}, MMatrix{10,10,Float64,100}, MMatrix{10,10,Float64,100}))
