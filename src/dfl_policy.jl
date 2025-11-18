"""
    DFLPolicy{F,M}

A Decision-Focused Learning (DFL) policy that combines a statistical model with a combinatorial optimization algorithm.

# Fields
- `model::F`: Statistical model that predicts parameters
- `maximizer::M`: Optimization solver/maximizer
"""
struct DFLPolicy{F,M}
    model::F
    maximizer::M
end

function (p::DFLPolicy)(x; kwargs...)
    θ = p.model(x)
    y = p.maximizer(θ; kwargs...)
    return y
end
