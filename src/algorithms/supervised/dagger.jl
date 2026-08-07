"""
$TYPEDEF

Dataset Aggregation (DAgger) algorithm for imitation learning.

Reference: <https://arxiv.org/abs/2402.04463>

# Fields
$TYPEDFIELDS
"""
@kwdef struct DAgger{A,S} <: AbstractImitationAlgorithm
    "inner imitation algorithm for supervised learning"
    inner_algorithm::A = PerturbedFenchelYoungLossImitation()
    "number of DAgger iterations"
    iterations::Int = 5
    "number of epochs per DAgger iteration"
    epochs_per_iteration::Int = 3
    "decay factor for mixing expert and learned policy"
    α_decay::Float64 = 0.9
    "random seed for the expert/policy mixing coin-flip (nothing = non-reproducible)"
    seed::S = nothing
    "maximum dataset size across iterations (nothing keeps all samples,
    an integer caps to the most recent N samples via FIFO)"
    max_dataset_size::Union{Int,Nothing} = nothing
end

# Helper function to collect one DAgger rollout pass over the training environments.
#
# Every environment starts a fresh episode; at each step the anticipative expert is queried
# from the current state and its solution is stored as a labeled sample. The action
# actually played is the expert one with probability `α`, the policy one otherwise.
function _collect_dagger_samples(
    policy::DFLPolicy,
    train_environments,
    anticipative_policy,
    α,
    rng;
    maximizer_kwargs=sample -> sample.context,
)
    (; statistical_model, maximizer) = policy
    new_samples = DataSample[]
    for env in train_environments
        # start a fresh episode from the wrapper's current rng state (no re-seeding),
        # so each DAgger iteration visits new scenarios
        DecisionFocusedLearningBenchmarks.reset!(env)
        while !is_terminated(env)
            anticipative_solution = anticipative_policy(env; reset_env=false)
            p = rand(rng)
            target = anticipative_solution[1]
            x, _ = observe(env)
            if size(target.x) != size(x)
                @error "Mismatch between expert and observed state" size(target.x) size(x)
            end
            push!(new_samples, target)
            if p < α
                action = target.y
            else
                θ = statistical_model(x)
                action = maximizer(θ; maximizer_kwargs(target)...)
            end
            step!(env, action)
        end
    end
    return new_samples
end

# Helper function to cap an aggregated dataset to its most recent samples (FIFO)
_clamp_dataset(dataset, max_dataset_size::Nothing) = dataset
_clamp_dataset(dataset, max_dataset_size::Int) = last(dataset, max_dataset_size)

"""
$TYPEDSIGNATURES

Train a DFLPolicy using the DAgger algorithm on the provided training environments.

# Core training method

Requires `train_environments` and `anticipative_policy` as keyword arguments.

`logger` is called with one line per iteration and per collection pass. It defaults to
`println`; pass a timestamping printer to make the phases of a long run legible, since
collecting expert labels can take hours per iteration and is otherwise silent.
"""
function train_policy!(
    algorithm::DAgger,
    policy::DFLPolicy,
    train_environments;
    anticipative_policy,
    metrics::Tuple=(),
    maximizer_kwargs=sample -> sample.context,
    logger=println,
)
    (; inner_algorithm, iterations, epochs_per_iteration, α_decay, seed) = algorithm
    (; statistical_model, maximizer) = policy

    rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)
    α = 1.0

    # Initial dataset from expert demonstrations
    train_dataset = vcat(map(train_environments) do env
        return anticipative_policy(env; reset_env=true)
    end...)

    dataset = deepcopy(train_dataset)

    # Initialize combined history for all DAgger iterations
    combined_history = MVHistory()
    epoch_offset = 0

    for iter in 1:iterations
        logger(
            "DAgger iteration $iter/$iterations (α=$(round(α, digits=3)), " *
            "dataset=$(length(dataset)) samples)",
        )

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
            local_epochs, values = get(iter_history, key)
            for i in eachindex(local_epochs)
                # Skip epoch 0 for all iterations after the first
                local_epochs[i] == 0 && epoch_offset > 0 && continue
                global_e = epoch_offset + local_epochs[i]
                push!(combined_history, key, global_e, key == :epoch ? global_e : values[i])
            end
        end

        epoch_offset += epochs_per_iteration

        # Dataset update - collect new samples using mixed policy.
        # Skipped on the last iteration: training happens at the *start* of an
        # iteration, so samples collected after the last one are never trained on. The
        # pass is pure waste, and an expensive one — it queries the expert solver at
        # every step of every training episode.
        iter == iterations && break

        t_collect = time()
        new_samples = _collect_dagger_samples(
            policy, train_environments, anticipative_policy, α, rng; maximizer_kwargs
        )
        dataset = _clamp_dataset(vcat(dataset, new_samples), algorithm.max_dataset_size)
        # The collection pass is the dominant cost of DAgger and grows sharply as α
        # decays (the learned policy drifts off-distribution, and the expert MILP from
        # those states gets much harder), so it is worth its own timing line.
        logger(
            "  collected $(length(new_samples)) samples in " *
            "$(round(time() - t_collect; digits=1))s " *
            "(dataset: $(length(dataset)) samples)",
        )
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
    benchmark::ExogenousDynamicBenchmark;
    dataset_size=30,
    metrics::Tuple=(),
    seed=nothing,
)
    # Generate environments
    train_environments = generate_environments(benchmark, dataset_size; seed)

    # Initialize model and create policy
    model = generate_statistical_model(benchmark; seed)
    maximizer = generate_maximizer(benchmark)
    policy = DFLPolicy(model, maximizer)

    # Define anticipative policy from benchmark
    anticipative_policy = generate_anticipative_solver(benchmark)

    # Train policy
    history = train_policy!(
        algorithm,
        policy,
        train_environments;
        anticipative_policy=anticipative_policy,
        metrics=metrics,
        maximizer_kwargs=sample -> sample.context,
    )

    return history, policy
end
