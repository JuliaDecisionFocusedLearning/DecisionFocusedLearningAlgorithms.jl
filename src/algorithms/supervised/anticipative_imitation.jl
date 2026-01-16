"""
$TYPEDEF

Anticipative Imitation algorithm for supervised learning using anticipative solutions.

Trains a policy in a single shot using expert demonstrations from anticipative solutions.

Reference: <https://arxiv.org/abs/2304.00789>

# Fields
$TYPEDFIELDS
"""
@kwdef struct AnticipativeImitation{A} <: AbstractImitationAlgorithm
    "inner imitation algorithm for supervised learning"
    inner_algorithm::A = PerturbedFenchelYoungLossImitation()
end

"""
$TYPEDSIGNATURES

Train a DFLPolicy using the Anticipative Imitation algorithm on provided training environments.

# Core training method

Generates anticipative solutions from environments and trains the policy using supervised learning.
"""
function train_policy!(
    algorithm::AnticipativeImitation,
    policy::DFLPolicy,
    train_environments;
    anticipative_policy,
    epochs=10,
    metrics::Tuple=(),
    maximizer_kwargs=get_state,
)
    # Generate anticipative solutions as training data
    train_dataset = vcat(map(train_environments) do env
        v, y = anticipative_policy(env; reset_env=true)
        return y
    end...)

    # Delegate to inner algorithm
    return train_policy!(
        algorithm.inner_algorithm,
        policy,
        train_dataset;
        epochs,
        metrics,
        maximizer_kwargs=maximizer_kwargs,
    )
end

"""
$TYPEDSIGNATURES

Train a DFLPolicy using the Anticipative Imitation algorithm on a benchmark.

# Benchmark convenience wrapper

This high-level function handles all setup from the benchmark and returns a trained policy.
Uses anticipative solutions as expert demonstrations.
"""
function train_policy(
    algorithm::AnticipativeImitation,
    benchmark::AbstractStochasticBenchmark{true};
    dataset_size=30,
    split_ratio=(0.3, 0.3),
    epochs=10,
    metrics::Tuple=(),
    seed=nothing,
)
    # Generate instances and environments
    dataset = generate_dataset(benchmark, dataset_size)
    train_instances, validation_instances, _ = splitobs(dataset; at=split_ratio)
    train_environments = generate_environments(benchmark, train_instances)

    # Initialize model and create policy
    model = generate_statistical_model(benchmark; seed)
    maximizer = generate_maximizer(benchmark)
    policy = DFLPolicy(model, maximizer)

    # Define anticipative policy from benchmark
    anticipative_policy =
        (env; reset_env) -> generate_anticipative_solution(benchmark, env; reset_env)

    # Train policy
    history = train_policy!(
        algorithm,
        policy,
        train_environments;
        anticipative_policy=anticipative_policy,
        epochs=epochs,
        metrics=metrics,
    )

    return history, policy
end
