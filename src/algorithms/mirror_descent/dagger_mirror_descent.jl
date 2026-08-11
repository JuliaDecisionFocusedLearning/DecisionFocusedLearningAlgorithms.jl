"""
$TYPEDEF

Dataset Aggregation (DAgger) combined with Mirror Descent coordination, for **dynamic**
benchmarks.

Each iteration runs two passes over the training environments:

 1. **DAgger pass** — roll out the mixed expert/policy on every training environment and
    append the visited states to an aggregated buffer, capped by `max_dataset_size`.
    This decides *which states* the policy is trained on: the ones it actually visits.
 2. **Mirror descent pass** — relabel every state of the buffer by solving the perturbed
    parametric anticipative problem on freshly drawn scenarios, then run one supervised
    training step on those coordinated targets. This decides *which decision* is learned
    on each state: a relaxation of the SAA target that couples the stages through
    `θ = model(sample.x)`.

The expert labels collected by the DAgger pass are kept in the buffer, but the mirror
descent pass overwrites them before training: they are only used as-is by the
`imitation_start` step.

Everything benchmark-specific is dispatched on the type of `benchmark`, exactly as in
[`MirrorDescent`](@ref) — see its scenario-sampling [`train_policy!`](@ref) for the
calling conventions of `parametric_anticipative_solver`.

See [`DAgger`](@ref) and [`MirrorDescent`](@ref) for the two halves taken separately.

# Fields
$TYPEDFIELDS
"""
@kwdef struct DAggerMirrorDescent{A<:PerturbedFenchelYoungLossImitation} <:
              AbstractAlgorithm
    "inner imitation algorithm for supervised learning"
    inner_algorithm::A = PerturbedFenchelYoungLossImitation()
    "decay factor for mixing expert and learned policy"
    α_decay::Float64 = 0.9
    "maximum buffer size across iterations (nothing keeps all states,
    an integer caps to the most recent N states via FIFO)"
    max_dataset_size::Union{Int,Nothing} = nothing
end

# One DAgger + mirror descent iteration:
#   (a) aggregate the states visited by the mixed policy into `dataset`,
#   (b) recoordinate the whole buffer against freshly drawn scenarios,
#   (c) run one supervised training step on the coordinated targets.
# Returns the updated buffer alongside the training history.
function _dagger_mirror_descent_iteration!(
    algorithm,
    policy,
    benchmark,
    dataset,
    train_environments,
    anticipative_solver,
    perturbed_solver,
    is_minimization,
    α,
    rng;
    nb_scenarios,
    epochs,
    κ,
    metrics,
    maximizer_kwargs,
    verbose,
)
    new_samples = _collect_dagger_samples(
        policy, train_environments, anticipative_solver, α, rng; maximizer_kwargs
    )
    dataset = _clamp_dataset(vcat(dataset, new_samples), algorithm.max_dataset_size)
    verbose && println(
        "  buffer: $(length(dataset)) states ($(length(new_samples)) from this rollout)"
    )

    t_relabel = time()
    coordinated_dataset = _augment_with_sampled_scenarios(
        benchmark,
        dataset,
        policy.statistical_model,
        perturbed_solver,
        is_minimization,
        rng;
        κ,
        nb_scenarios,
    )
    verbose && println(
        "  relabelled $(length(coordinated_dataset)) states in " *
        "$(round(time() - t_relabel; digits=1))s",
    )
    history = train_policy!(
        algorithm.inner_algorithm,
        policy,
        coordinated_dataset;
        epochs,
        metrics,
        maximizer_kwargs,
    )
    return dataset, history
end

# Helper function to run the DAgger + mirror descent loop for a given number of iterations
function _dagger_mirror_descent_loop!(
    algorithm,
    policy,
    benchmark,
    input_dataset,
    train_environments,
    anticipative_solver,
    perturbed_solver,
    is_minimization;
    iterations,
    nb_scenarios,
    epochs,
    κ,
    α,
    metrics,
    maximizer_kwargs,
    verbose,
    rng,
)
    dataset = input_dataset
    histories = MVHistory[]
    for n_it in 1:iterations
        verbose && println(
            "DAgger mirror descent iteration $n_it / $iterations (α=$(round(α, digits=3)))",
        )
        dataset, history = _dagger_mirror_descent_iteration!(
            algorithm,
            policy,
            benchmark,
            dataset,
            train_environments,
            anticipative_solver,
            perturbed_solver,
            is_minimization,
            α,
            rng;
            nb_scenarios,
            epochs,
            κ,
            metrics,
            maximizer_kwargs,
            verbose,
        )
        push!(histories, history)
        α *= algorithm.α_decay  # Decay factor for mixing expert and learned policy
    end
    return histories
end

"""
$TYPEDSIGNATURES

Train a DFLPolicy using the [`DAggerMirrorDescent`](@ref) algorithm on the provided
training environments.

When `imitation_start=true`, the first iteration is a pure imitation step on
`train_dataset` (by default the anticipative demonstrations over `train_environments`).
Subsequent iterations are the DAgger + mirror descent loop, which keeps aggregating into
that same buffer.

# Arguments
- `anticipative_solver`: `(env; reset_env, kwargs...) -> Vector{DataSample}`, the expert
queried at every step of the DAgger rollouts
- `parametric_anticipative_solver`: parametric anticipative oracle, called with the
convention of its own benchmark family (see [`MirrorDescent`](@ref))
- `train_dataset`: initial buffer (defaults to the anticipative demonstrations)
- `nb_scenarios=1`: scenarios drawn per state and per iteration in the coordination pass
- `iterations=10`: total number of iterations (includes the imitation step when
`imitation_start=true`)
- `epochs=10`: number of inner training epochs per iteration
- `κ=1.0`: scaling factor applied to `θ` before passing it to the perturbed solver
- `α=1.0`: initial probability of playing the expert action during a rollout, decayed by
`algorithm.α_decay` after each iteration
- `metrics::Tuple=()`: metrics forwarded to the inner training algorithm
- `verbose=false`: if true, prints progress at each iteration
- `imitation_start=true`: if true, run a pure imitation step on `train_dataset` as the
first iteration
"""
function train_policy!(
    algorithm::DAggerMirrorDescent,
    policy::DFLPolicy,
    benchmark::AbstractDynamicBenchmark,
    train_environments,
    anticipative_solver,
    parametric_anticipative_solver;
    train_dataset=reduce(
        vcat, (anticipative_solver(env; reset_env=true) for env in train_environments)
    ),
    nb_scenarios=1,
    epochs=10,
    iterations=10,
    κ=1.0,
    α=1.0,
    metrics::Tuple=(),
    maximizer_kwargs=sample -> sample.context,
    verbose::Bool=false,
    imitation_start::Bool=true,
    is_minimization::Bool=is_minimization_problem(benchmark),
    rng=Random.default_rng(),
)
    perturbed_solver = _perturbed_parametric_solver(
        benchmark, parametric_anticipative_solver, algorithm.inner_algorithm, κ
    )

    histories = MVHistory[]
    loop_iters = iterations
    if imitation_start
        verbose && println("Imitation step")
        push!(
            histories,
            train_policy!(
                algorithm.inner_algorithm,
                policy,
                train_dataset;
                epochs,
                metrics,
                maximizer_kwargs,
            ),
        )
        loop_iters -= 1
    end
    loop_iters >= 1 || return histories

    append!(
        histories,
        _dagger_mirror_descent_loop!(
            algorithm,
            policy,
            benchmark,
            train_dataset,
            train_environments,
            anticipative_solver,
            perturbed_solver,
            is_minimization;
            iterations=loop_iters,
            nb_scenarios,
            epochs,
            κ,
            α,
            metrics,
            maximizer_kwargs,
            verbose,
            rng,
        ),
    )
    return histories
end

"""
$TYPEDSIGNATURES

Generate environments for the provided **dynamic** benchmark and train a DFLPolicy using
the [`DAggerMirrorDescent`](@ref) algorithm.

This high-level wrapper builds every component (`model`, `maximizer`,
`anticipative_solver`, `parametric_anticipative_solver`, `train_environments`,
`train_dataset`) from the benchmark, each exposed as an optional keyword so callers can
override any of them without dropping to [`train_policy!`](@ref).

The benchmark must implement `build_environment(benchmark, sample, scenario)`, used by the
coordination pass to rebuild an environment at the state of each buffered sample.

# Arguments
- `dataset_size=10`: number of training environments
(used when `train_environments` is not provided)
- `seed=nothing`: random seed for reproducibility
(used in `model`, `train_environments` and the rollout/scenario `rng`)

See [`train_policy!`](@ref) for the remaining arguments.
"""
function train_policy(
    algorithm::DAggerMirrorDescent,
    benchmark::AbstractDynamicBenchmark;
    dataset_size=10,
    nb_scenarios=1,
    seed=nothing,
    model=generate_statistical_model(benchmark; seed=seed),
    maximizer=generate_maximizer(benchmark),
    anticipative_solver=generate_anticipative_solver(benchmark),
    parametric_anticipative_solver=generate_parametric_anticipative_solver(benchmark),
    train_environments=generate_environments(benchmark, dataset_size; seed=seed),
    train_dataset=reduce(
        vcat, (anticipative_solver(env; reset_env=true) for env in train_environments)
    ),
    epochs=10,
    iterations=10,
    κ=1.0,
    α=1.0,
    metrics::Tuple=(),
    maximizer_kwargs=sample -> sample.context,
    verbose::Bool=false,
    imitation_start::Bool=true,
    is_minimization::Bool=is_minimization_problem(benchmark),
)
    policy = DFLPolicy(model, maximizer)
    histories = train_policy!(
        algorithm,
        policy,
        benchmark,
        train_environments,
        anticipative_solver,
        parametric_anticipative_solver;
        train_dataset,
        nb_scenarios,
        epochs,
        iterations,
        κ,
        α,
        metrics,
        maximizer_kwargs,
        verbose,
        imitation_start,
        is_minimization,
        rng=isnothing(seed) ? Random.default_rng() : MersenneTwister(seed),
    )
    return histories, policy
end
