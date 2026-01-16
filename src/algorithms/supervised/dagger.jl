"""
$TYPEDEF

Dataset Aggregation (DAgger) algorithm for imitation learning.

Reference: <https://arxiv.org/abs/2402.04463>

# Fields
$TYPEDFIELDS
"""
@kwdef struct DAgger{A} <: AbstractImitationAlgorithm
    "inner imitation algorithm for supervised learning"
    inner_algorithm::A = PerturbedFenchelYoungLossImitation()
    "number of DAgger iterations"
    iterations::Int = 5
    "number of epochs per DAgger iteration"
    epochs_per_iteration::Int = 3
    "decay factor for mixing expert and learned policy"
    α_decay::Float64 = 0.9
end

"""
$TYPEDSIGNATURES

Train a DFLPolicy using the DAgger algorithm on the provided training environments.

# Core training method

Requires `train_environments` and `anticipative_policy` as keyword arguments.
"""
function train_policy!(
    algorithm::DAgger,
    policy::DFLPolicy,
    train_environments;
    anticipative_policy,
    metrics::Tuple=(),
    maximizer_kwargs=get_state,
)
    (; inner_algorithm, iterations, epochs_per_iteration, α_decay) = algorithm
    (; statistical_model, maximizer) = policy

    α = 1.0

    # Initial dataset from expert demonstrations
    train_dataset = vcat(map(train_environments) do env
        v, y = anticipative_policy(env; reset_env=true)
        return y
    end...)

    dataset = deepcopy(train_dataset)

    # Initialize combined history for all DAgger iterations
    combined_history = MVHistory()
    global_epoch = 0

    for iter in 1:iterations
        println("DAgger iteration $iter/$iterations (α=$(round(α, digits=3)))")

        # Train for epochs_per_iteration using inner algorithm
        iter_history = train_policy!(
            inner_algorithm,
            policy,
            dataset;
            epochs=epochs_per_iteration,
            metrics=metrics,
            maximizer_kwargs=maximizer_kwargs,
        )

        # Merge iteration history into combined history
        for key in keys(iter_history)
            epochs, values = get(iter_history, key)
            for i in eachindex(epochs)
                # Calculate global epoch number
                if iter == 1
                    # First iteration: use epochs as-is [0, 1, 2, ...]
                    global_epoch_value = epochs[i]
                else
                    # Later iterations: skip epoch 0 and renumber starting from global_epoch
                    if epochs[i] == 0
                        continue  # Skip epoch 0 for iterations > 1
                    end
                    # Map epoch 1 → global_epoch, epoch 2 → global_epoch+1, etc.
                    global_epoch_value = global_epoch + epochs[i] - 1
                end

                # For the epoch key, use global_epoch_value as both time and value
                # For other keys, use global_epoch_value as time and original value
                if key == :epoch
                    push!(combined_history, key, global_epoch_value, global_epoch_value)
                else
                    push!(combined_history, key, global_epoch_value, values[i])
                end
            end
        end

        # Update global_epoch for next iteration
        # After each iteration, advance by the number of non-zero epochs processed
        if iter == 1
            # First iteration processes all epochs [0, 1, ..., epochs_per_iteration]
            # Next iteration should start at epochs_per_iteration + 1
            global_epoch = epochs_per_iteration + 1
        else
            # Subsequent iterations skip epoch 0, so they process epochs_per_iteration epochs
            # Next iteration should start epochs_per_iteration later
            global_epoch += epochs_per_iteration
        end

        # Dataset update - collect new samples using mixed policy
        new_samples = eltype(dataset)[]
        for env in train_environments
            DecisionFocusedLearningBenchmarks.reset!(env; reset_rng=false)
            while !is_terminated(env)
                x_before = copy(observe(env)[1])
                _, anticipative_solution = anticipative_policy(env; reset_env=false)
                p = rand()
                target = anticipative_solution[1]
                x, state = observe(env)
                if size(target.x) != size(x)
                    @error "Mismatch between expert and observed state" size(target.x) size(
                        x
                    )
                end
                push!(new_samples, target)
                if p < α
                    action = target.y
                else
                    x, state = observe(env)
                    θ = statistical_model(x)
                    action = maximizer(θ; maximizer_kwargs(target)...)
                end
                step!(env, action)
            end
        end
        dataset = new_samples  # TODO: replay buffer
        α *= α_decay  # Decay factor for mixing expert and learned policy
    end

    return combined_history
end

"""
$TYPEDSIGNATURES

Train a DFLPolicy using the DAgger algorithm on a benchmark.

# Benchmark convenience wrapper

This high-level function handles all setup from the benchmark and returns a trained policy.
"""
function train_policy(
    algorithm::DAgger,
    benchmark::AbstractStochasticBenchmark{true};
    dataset_size=30,
    split_ratio=(0.3, 0.3, 0.4),
    metrics::Tuple=(),
    seed=0,
)
    # Generate dataset and environments
    dataset = generate_dataset(benchmark, dataset_size)
    train_instances, validation_instances, _ = splitobs(dataset; at=split_ratio)
    train_environments = generate_environments(benchmark, train_instances; seed)

    # Initialize model and create policy
    model = generate_statistical_model(benchmark)
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
        metrics=metrics,
        maximizer_kwargs=get_state,
    )

    return history, policy
end
