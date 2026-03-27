"""
$TYPEDEF

An abstract type for decision-focused learning algorithms.
"""
abstract type AbstractAlgorithm end

"""
$TYPEDEF

An abstract type for imitation learning algorithms.

All subtypes must implement:
- `train_policy!(algorithm::AbstractImitationAlgorithm, policy::DFLPolicy, train_data; epochs, metrics)`
"""
abstract type AbstractImitationAlgorithm <: AbstractAlgorithm end

"""
$TYPEDSIGNATURES

Train a new DFLPolicy on a benchmark using any imitation learning algorithm.

Convenience wrapper that handles dataset generation, model initialization, and policy
creation. Returns the training history and the trained policy.

For dynamic benchmarks, use the algorithm-specific `train_policy` overload that accepts
environments and an anticipative policy.
"""
function train_policy(
    algorithm::AbstractImitationAlgorithm,
    benchmark::AbstractBenchmark;
    target_policy=nothing,
    dataset_size=30,
    epochs=100,
    metrics::Tuple=(),
    seed=nothing,
)
    dataset = generate_dataset(benchmark, dataset_size; target_policy)

    if any(s -> isnothing(s.y), dataset)
        error(
            "Training dataset contains unlabeled samples (y=nothing). " *
            "Provide a `target_policy` kwarg to label samples during dataset generation.",
        )
    end

    model = generate_statistical_model(benchmark; seed)
    maximizer = generate_maximizer(benchmark)
    policy = DFLPolicy(model, maximizer)

    history = train_policy!(
        algorithm, policy, dataset; epochs, metrics, maximizer_kwargs=s -> s.context
    )

    return history, policy
end
