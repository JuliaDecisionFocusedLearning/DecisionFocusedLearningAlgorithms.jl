"""
$TYPEDEF

Decision-Focused Learning Policy combining a machine learning model and a combinatorial optimizer.
"""
struct DFLPolicy{ML,CO} <: AbstractPolicy
    "machine learning statistical model"
    statistical_model::ML
    "combinatorial optimizer"
    maximizer::CO
end

"""
$TYPEDSIGNATURES

Run the policy and get the next decision on the given input features.
"""
function (p::DFLPolicy)(features::AbstractArray; kwargs...)
    # Get predicted parameters from statistical model
    θ = p.statistical_model(features)
    # Use combinatorial optimizer to get decision
    y = p.maximizer(θ; kwargs...)
    return y
end

"""
$TYPEDSIGNATURES

Convenience overload: evaluate the optimality gap using a [`DFLPolicy`](@ref) directly,
instead of unpacking `policy.statistical_model` and `policy.maximizer`.
"""
function DecisionFocusedLearningBenchmarks.compute_gap(
    bench, dataset, policy::DFLPolicy, op=mean
)
    return DecisionFocusedLearningBenchmarks.compute_gap(
        bench, dataset, policy.statistical_model, policy.maximizer, op
    )
end
