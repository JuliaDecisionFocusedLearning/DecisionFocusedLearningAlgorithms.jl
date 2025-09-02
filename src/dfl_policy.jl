struct DFLPolicy{F,M}
    model::F
    maximizer::M
end

function (p::DFLPolicy)(x; kwargs...)
    θ = p.model(x)
    y = p.maximizer(θ; kwargs...)
    return y
end
