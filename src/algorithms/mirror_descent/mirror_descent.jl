"""
$TYPEDEF

Mirror Descent algorithm for learning coordinated solutions.

Reference: <https://arxiv.org/abs/2505.04757>

# Fields
$TYPEDFIELDS
"""
@kwdef struct MirrorDescent{A<:PerturbedFenchelYoungLossImitation} <: AbstractAlgorithm
    "inner imitation algorithm for supervised learning"
    inner_algorithm::A = PerturbedFenchelYoungLossImitation()
end

# ---------------------------------------------------------------------------
# Scenario handling
#
# Every mirror descent iteration relabels each sample by solving the perturbed parametric
# anticipative problem under a scenario. That scenario is either the one already carried by
# the sample — reused for reproducibility, the default — or a freshly drawn one, the "SAA
# relaxed by resampling" reading of mirror descent. Reuse is generic (any sample that already
# has a `.scenario`, from `generate_dataset` or a previous mirror descent round, qualifies);
# only drawing a *fresh* one is benchmark-specific.
# ---------------------------------------------------------------------------

"""
    _draw_scenario(benchmark, rng, sample; use_stored_scenario=true) -> scenario

Return the scenario to solve `sample` under. When `use_stored_scenario=true` (the default)
and `sample` already carries one, reuse it. Otherwise draw a fresh one from `benchmark` —
see [`_sample_fresh_scenario`](@ref) for the benchmark-specific part.
"""
function _draw_scenario(benchmark, rng, sample; use_stored_scenario::Bool=true)
    if use_stored_scenario && hasproperty(sample, :scenario)
        return sample.scenario
    end
    return _sample_fresh_scenario(benchmark, rng, sample)
end

# Stochastic benchmarks draw scenarios from the sample context.
function _sample_fresh_scenario(benchmark::ExogenousStochasticBenchmark, rng, sample)
    return generate_scenario(benchmark, rng; sample.context...)
end

# Dynamic benchmarks draw a whole episode scenario straight from the benchmark: the state
# lives in the sample, so there is no context to spread.
function _sample_fresh_scenario(benchmark::AbstractDynamicBenchmark, rng, ::DataSample)
    return generate_scenario(benchmark; rng)
end

# Solve the parametric anticipative subproblem for `sample` under `scenario`, given the
# already-signed score `signed_θ`.
# Stochastic solvers follow `(θ, scenario; context...) -> y` and need nothing else.
function _parametric_target(
    ::ExogenousStochasticBenchmark, perturbed_solver, signed_θ, sample, scenario
)
    return perturbed_solver(signed_θ; scenario=scenario, sample.context...)
end

# Dynamic solvers follow `(θ, scenario, environment; kwargs...) -> trajectory`: they take the
# environment as a third positional argument and solve from its current state, returning the
# whole anticipative trajectory. So we rebuild an environment sitting at the state of
# `sample` — cheap next to the solve it feeds — and keep only the first-stage decision.
function _parametric_target(
    benchmark::AbstractDynamicBenchmark, perturbed_solver, signed_θ, sample, scenario
)
    environment = SeededEnvironment(build_environment(benchmark, sample, scenario))
    return perturbed_solver(signed_θ; scenario=scenario, environment=environment)
end

# Wrap the benchmark's parametric anticipative solver in a perturbed layer exposing the
# `(θ; scenario, kwargs...) -> y` interface `_parametric_target` calls it through.
function _perturbed_parametric_solver(
    ::ExogenousStochasticBenchmark, parametric_anticipative_solver, inner_algorithm, κ
)
    (; nb_samples, ε, threaded, seed) = inner_algorithm
    return PerturbedAdditive(
        (θ; scenario, kwargs...) -> parametric_anticipative_solver(θ, scenario; kwargs...);
        ε=κ * ε,
        nb_samples=nb_samples,
        seed=seed,
        threaded=threaded,
    )
end

function _perturbed_parametric_solver(
    ::AbstractDynamicBenchmark, parametric_anticipative_solver, inner_algorithm, κ
)
    (; nb_samples, ε, threaded, seed) = inner_algorithm
    return PerturbedAdditive(
        (θ; scenario, environment, kwargs...) ->
            first(parametric_anticipative_solver(θ, scenario, environment; kwargs...)).y;
        ε=κ * ε,
        nb_samples=nb_samples,
        seed=seed,
        threaded=threaded,
    )
end

# Augment a dataset by solving the perturbed parametric problem on each sample.
# With `use_stored_scenario=true` (default) and a sample that already carries a scenario,
# that scenario is fixed — it is *the* decision problem for this sample — so drawing it
# `nb_scenarios` times would only replay perturbation noise, not coordinate against
# different scenarios. Only one `y` is produced in that case. Otherwise (no stored scenario,
# or `use_stored_scenario=false`) `nb_scenarios` independent scenarios are drawn, the
# "relaxed SAA" reading of mirror descent, growing the dataset to `nb_scenarios *
# length(dataset)`.
function _augment_with_sampled_scenarios(
    benchmark,
    dataset,
    model,
    perturbed_solver,
    is_minimization,
    rng;
    κ,
    nb_scenarios,
    use_stored_scenario::Bool=true,
)
    augmented = DataSample[]
    sizehint!(augmented, nb_scenarios * length(dataset))
    for sample in dataset
        θ = model(sample.x)
        signed_θ = is_minimization ? -κ * θ : κ * θ
        draws = use_stored_scenario && hasproperty(sample, :scenario) ? 1 : nb_scenarios
        for _ in 1:draws
            scenario = _draw_scenario(benchmark, rng, sample; use_stored_scenario)
            y = _parametric_target(benchmark, perturbed_solver, signed_θ, sample, scenario)
            push!(augmented, DataSample(sample; y, extra=merge(sample.extra, (; scenario))))
        end
    end
    return augmented
end

function _mirror_descent_loop_sampled(
    algorithm,
    policy,
    benchmark,
    input_dataset,
    perturbed_solver,
    is_minimization;
    md_iters,
    nb_scenarios,
    epochs,
    κ,
    metrics,
    verbose,
    rng,
    use_stored_scenario,
)
    return map(1:md_iters) do md_it
        verbose && println("Mirror descent iteration $md_it / $md_iters")
        t_relabel = time()
        dataset = _augment_with_sampled_scenarios(
            benchmark,
            input_dataset,
            policy.statistical_model,
            perturbed_solver,
            is_minimization,
            rng;
            κ,
            nb_scenarios,
            use_stored_scenario,
        )
        verbose && println(
            "  relabelled $(length(dataset)) states in " *
            "$(round(time() - t_relabel; digits=1))s",
        )
        return train_policy!(algorithm.inner_algorithm, policy, dataset; epochs, metrics)
    end
end

"""
$TYPEDSIGNATURES

Train a DFLPolicy using the Mirror Descent algorithm.

`train_dataset` must already be labeled — see [`generate_train_dataset`](@ref) for how the
high-level [`train_policy`](@ref) wrapper builds one (imitation demonstrations for
stochastic benchmarks, anticipative trajectories for dynamic ones). Mirror descent
iterations never need `train_dataset` pre-labeled anyway, since every sample gets relabeled
by solving the perturbed parametric anticipative problem under a scenario — reused from the
sample when `use_stored_scenario=true` (the default) and present, otherwise drawn fresh from
`benchmark`.

Everything benchmark-specific is dispatched on the type of `benchmark`, so
`parametric_anticipative_solver` is used with the calling convention of its own benchmark
family and needs no adapter:

| | stochastic benchmark | dynamic benchmark |
|---|---|---|
| fresh scenario drawn as | `generate_scenario(benchmark, rng; sample.context...)` | `generate_scenario(benchmark; rng)` |
| solver called as | `(θ, scenario; sample.context...) -> y` | `(θ, scenario, environment) -> trajectory` |

Dynamic solvers solve from the current state of the environment they are given, so each
sample gets a throwaway environment rebuilt at its own state through
`build_environment(benchmark, sample, scenario)`, and only the first-stage decision of the
returned trajectory is kept.

# Arguments
- `use_stored_scenario=true`: reuse `sample.scenario` when present instead of drawing a
fresh one — see [`_draw_scenario`](@ref)
- `nb_scenarios=1`: scenarios drawn per sample and per iteration, ignored for samples that
reuse a stored scenario (see [`_augment_with_sampled_scenarios`](@ref))
- `iterations=10`: total number of mirror descent iterations (includes the imitation step
when `imitation_start=true`)
- `epochs=10`: number of inner training epochs per mirror descent iteration
- `κ=1.0`: scaling factor applied to `θ` before passing it to the perturbed solver
- `metrics::Tuple=()`: metrics forwarded to the inner training algorithm
- `verbose=false`: if true, prints progress at each iteration
- `imitation_start=true`: if true, run a pure imitation step on `train_dataset` as the first
iteration
- `is_minimization=is_minimization_problem(benchmark)`: set to false if the objective is a
maximization problem
"""
function train_policy!(
    algorithm::MirrorDescent,
    policy::DFLPolicy,
    benchmark::AbstractBenchmark,
    train_dataset,
    parametric_anticipative_solver;
    use_stored_scenario::Bool=true,
    nb_scenarios=1,
    epochs=10,
    iterations=10,
    κ=1.0,
    metrics::Tuple=(),
    verbose::Bool=false,
    imitation_start::Bool=true,
    is_minimization::Bool=is_minimization_problem(benchmark),
    rng=Random.default_rng(),
)
    perturbed_solver = _perturbed_parametric_solver(
        benchmark, parametric_anticipative_solver, algorithm.inner_algorithm, κ
    )

    histories = MVHistory[]
    md_iters = iterations
    if imitation_start
        verbose && println("Imitation step")
        push!(
            histories,
            train_policy!(
                algorithm.inner_algorithm, policy, train_dataset; epochs, metrics
            ),
        )
        md_iters -= 1
    end
    md_iters >= 1 || return histories

    append!(
        histories,
        _mirror_descent_loop_sampled(
            algorithm,
            policy,
            benchmark,
            train_dataset,
            perturbed_solver,
            is_minimization;
            md_iters,
            nb_scenarios,
            epochs,
            κ,
            metrics,
            verbose,
            rng,
            use_stored_scenario,
        ),
    )
    return histories
end

# ---------------------------------------------------------------------------
# Default training data for the high-level `train_policy` wrapper below, dispatched on
# benchmark family so the single public method stays family-agnostic. Both branches return a
# fully labeled dataset, ready for `train_policy!` whether or not `imitation_start` is used.
# ---------------------------------------------------------------------------

"""
    generate_train_dataset(benchmark, anticipative_solver; dataset_size, nb_scenarios,
                            contexts_per_instance, train_environments, seed) -> Vector{DataSample}

Build the default `train_dataset` for [`train_policy`](@ref)`(::MirrorDescent, benchmark)`.

**Stochastic benchmarks**: draws `dataset_size` instances (`contexts_per_instance` contexts
× `nb_scenarios` scenarios each, via [`generate_dataset`](@ref)) and labels every sample by
solving `anticipative_solver(sample.scenario; sample.context...)`.

**Dynamic benchmarks**: generates `dataset_size` fresh `train_environments` (or reuses the
ones given) and concatenates the anticipative trajectory of each — already labeled.
"""
function generate_train_dataset(
    benchmark::ExogenousStochasticBenchmark,
    anticipative_solver;
    dataset_size,
    nb_scenarios,
    contexts_per_instance,
    train_environments,
    seed,
)
    train_environments === nothing || error(
        "`train_environments` only applies to dynamic benchmarks; got a stochastic " *
        "benchmark. Pass `train_dataset` directly instead.",
    )
    unlabeled = generate_dataset(
        benchmark, dataset_size; nb_scenarios, contexts_per_instance, seed
    )
    return map(unlabeled) do sample
        y = anticipative_solver(sample.scenario; sample.context...)
        return DataSample(sample; y=y)
    end
end

function generate_train_dataset(
    benchmark::AbstractDynamicBenchmark,
    anticipative_solver;
    dataset_size,
    nb_scenarios,
    contexts_per_instance,
    train_environments,
    seed,
)
    envs = something(
        train_environments, generate_environments(benchmark, dataset_size; seed)
    )
    return reduce(vcat, (anticipative_solver(env; reset_env=true) for env in envs))
end

"""
$TYPEDSIGNATURES

Generate a dataset for the provided benchmark and train a DFLPolicy using the Mirror
Descent algorithm.

This high-level wrapper builds every component (`model`, `maximizer`,
`anticipative_solver`, `parametric_anticipative_solver`, `train_dataset`) from the
benchmark, each exposed as an optional keyword so callers can override any of them without
dropping to [`train_policy!`](@ref).

`contexts_per_instance` and the dataset-sizing use of `nb_scenarios` only affect the default
`train_dataset` on stochastic benchmarks; `train_environments` only affects it on dynamic
ones — see [`generate_train_dataset`](@ref).

# Arguments
- `dataset_size=30`: number of samples (stochastic) or training environments (dynamic) —
used when `train_dataset` is not provided
- `nb_scenarios=1`: for the default `train_dataset` (stochastic only): scenarios per
context; for the mirror descent loop (both families): scenarios drawn per sample and per
iteration
- `contexts_per_instance=1`: context draws per instance, stochastic only
(used when `train_dataset` is not provided)
- `train_environments=nothing`: training environments, dynamic only
(used when `train_dataset` is not provided; defaults to `dataset_size` fresh ones)
- `seed=nothing`: random seed for reproducibility
(used in `model`, `train_dataset` and the mirror descent `rng` when not provided)
- `model`: statistical model to wrap in the policy
(defaults to `generate_statistical_model(benchmark; seed)`)
- `maximizer`: combinatorial oracle to wrap in the policy
(defaults to `generate_maximizer(benchmark)`)
- `anticipative_solver`: oracle used to build the default `train_dataset` — see
[`generate_train_dataset`](@ref) (defaults to `generate_anticipative_solver(benchmark)`)
- `parametric_anticipative_solver`: parametric oracle wrapped in `PerturbedAdditive` for
mirror-descent iterations (defaults to `generate_parametric_anticipative_solver(benchmark)`)
- `train_dataset`: training dataset (defaults to `generate_train_dataset(benchmark,
anticipative_solver; ...)`)
- `use_stored_scenario=true`: reuse each sample's stored scenario when present instead of
drawing a fresh one — see [`_draw_scenario`](@ref)
- `epochs=10`: number of inner training epochs per mirror descent iteration
- `iterations=10`: total number of mirror descent iterations
- `κ=1.0`: scaling factor applied to `θ` before passing it to the perturbed solver
- `metrics::Tuple=()`: metrics forwarded to the inner training algorithm
- `verbose=false`: if true, prints progress at each iteration
- `imitation_start=true`: if true, run a pure imitation step on `train_dataset` as the first
iteration
"""
function train_policy(
    algorithm::MirrorDescent,
    benchmark::Union{ExogenousStochasticBenchmark,AbstractDynamicBenchmark};
    dataset_size=30,
    nb_scenarios=1,
    contexts_per_instance=1,
    train_environments=nothing,
    seed=nothing,
    model=generate_statistical_model(benchmark; seed=seed),
    maximizer=generate_maximizer(benchmark),
    anticipative_solver=generate_anticipative_solver(benchmark),
    parametric_anticipative_solver=generate_parametric_anticipative_solver(benchmark),
    train_dataset=generate_train_dataset(
        benchmark,
        anticipative_solver;
        dataset_size,
        nb_scenarios,
        contexts_per_instance,
        train_environments,
        seed,
    ),
    use_stored_scenario::Bool=true,
    epochs=10,
    iterations=10,
    κ=1.0,
    metrics::Tuple=(),
    verbose::Bool=false,
    imitation_start::Bool=true,
    is_minimization::Bool=is_minimization_problem(benchmark),
)
    policy = DFLPolicy(model, maximizer)
    histories = train_policy!(
        algorithm,
        policy,
        benchmark,
        train_dataset,
        parametric_anticipative_solver;
        use_stored_scenario,
        nb_scenarios,
        epochs,
        iterations,
        κ,
        metrics,
        verbose,
        imitation_start,
        is_minimization,
        rng=isnothing(seed) ? Random.default_rng() : MersenneTwister(seed),
    )
    return histories, policy
end
