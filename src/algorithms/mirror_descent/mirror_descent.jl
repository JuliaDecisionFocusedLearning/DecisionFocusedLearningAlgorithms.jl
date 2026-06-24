"""
$TYPEDEF

Mirror Descent algorithm for learning coordinated solutions.

This algorithm is designed for stochastic benchmarks.

Reference: <https://arxiv.org/abs/2505.04757>

# Fields
$TYPEDFIELDS
"""
@kwdef struct MirrorDescent{A<:PerturbedFenchelYoungLossImitation} <: AbstractAlgorithm
    "inner imitation algorithm for supervised learning"
    inner_algorithm::A = PerturbedFenchelYoungLossImitation()
end

# Helper function to augment a dataset with anticipative solutions
function _augment_with_anticipative(dataset, anticipative_solver)
    return map(dataset) do sample
        y = anticipative_solver(sample.scenario; sample.context...)
        return DataSample(sample; y=y)
    end
end

# Helper function to create a perturbed sample
function _perturbed_sample(sample, model, perturbed_solver, is_minimization, κ)
    θ = model(sample.x)
    signed_θ = is_minimization ? -κ * θ : κ * θ
    y = perturbed_solver(signed_θ; scenario=sample.scenario, sample.context...)
    return DataSample(sample; y=y)
end

# Helper function to augment a dataset with perturbed solutions
function _augment_with_perturbed(dataset, model, perturbed_solver, is_minimization; κ=1.0)
    return map(dataset) do sample
        return _perturbed_sample(sample, model, perturbed_solver, is_minimization, κ)
    end
end

# Helper function to augment a dataset with perturbed solutions in-place
function _augment_with_perturbed!(dataset, model, perturbed_solver, is_minimization; κ=1.0)
    for i in eachindex(dataset)
        dataset[i] = _perturbed_sample(
            dataset[i], model, perturbed_solver, is_minimization, κ
        )
    end
    return dataset
end

# Helper function to run the mirror descent loop for a given number of iterations
function _mirror_descent_loop(
    algorithm,
    policy,
    input_dataset,
    perturbed_solver,
    is_minimization;
    md_iters,
    epochs,
    κ,
    metrics,
    verbose,
)
    # Allocate the perturbed dataset once. Subsequent iterations mutate in place.
    dataset = _augment_with_perturbed(
        input_dataset, policy.statistical_model, perturbed_solver, is_minimization; κ
    )
    return map(1:md_iters) do n_it
        verbose && println("Mirror descent iteration $n_it / $md_iters")
        if n_it > 1
            _augment_with_perturbed!(
                dataset, policy.statistical_model, perturbed_solver, is_minimization; κ
            )
        end
        return train_policy!(algorithm.inner_algorithm, policy, dataset; epochs, metrics)
    end
end

"""
$TYPEDSIGNATURES

Train a DFLPolicy using the Mirror Descent algorithm on a provided training dataset.

When `imitation_start=true`, the first iteration is a pure imitation step using
`anticipative_solver`. Subsequent iterations are the mirror descent loop.

# Arguments
- `iterations=10`: total number of mirror descent iterations (includes the imitation step
when `imitation_start=true`)
- `epochs=10`: number of inner training epochs per mirror descent iteration
- `κ=1.0`: scaling factor applied to `θ` before passing it to the perturbed solver
- `metrics::Tuple=()`: metrics forwarded to the inner training algorithm
- `verbose=false`: if true, prints progress at each iteration
- `imitation_start=true`: if true, run a pure imitation step against the
  anticipative solver as the first iteration
- `is_minimization=true`: set to false if the objective is a maximization problem
"""
function train_policy!(
    algorithm::MirrorDescent,
    policy::DFLPolicy,
    train_dataset,
    anticipative_solver,
    parametric_anticipative_solver;
    epochs=10,
    iterations=10,
    κ=1.0,
    metrics::Tuple=(),
    verbose::Bool=false,
    imitation_start::Bool=true,
    is_minimization::Bool=true,
)
    (; nb_samples, ε, threaded, seed) = algorithm.inner_algorithm
    perturbed_anticipative_solver = PerturbedAdditive(
        (θ; scenario, kwargs...) -> parametric_anticipative_solver(θ, scenario; kwargs...);
        ε=κ * ε,
        nb_samples=nb_samples,
        seed=seed,
        threaded=threaded,
    )

    if imitation_start
        verbose && println("Imitation step")
        dataset = _augment_with_anticipative(train_dataset, anticipative_solver)
        h_imitation = train_policy!(
            algorithm.inner_algorithm, policy, dataset; epochs, metrics
        )
        md_iters = iterations - 1
        md_iters >= 1 || return [h_imitation]
        rest = _mirror_descent_loop(
            algorithm,
            policy,
            dataset,
            perturbed_anticipative_solver,
            is_minimization;
            md_iters,
            epochs,
            κ,
            metrics,
            verbose,
        )
        return pushfirst!(rest, h_imitation)
    end

    # else
    return _mirror_descent_loop(
        algorithm,
        policy,
        train_dataset,
        perturbed_anticipative_solver,
        is_minimization;
        md_iters=iterations,
        epochs,
        κ,
        metrics,
        verbose,
    )
end

"""
$TYPEDSIGNATURES

Generate a dataset for the provided benchmark and train a DFLPolicy using the Mirror Descent algorithm.

This high-level wrapper builds every component (`model`, `maximizer`,
`anticipative_solver`, `parametric_anticipative_solver`, `train_dataset`) from the
benchmark, each exposed as an optional keyword so callers can override any of them
without dropping to [`train_policy!`](@ref).

# Arguments
- `dataset_size=30`: number of samples in the training dataset
(used when `train_dataset` is not provided)
- `nb_scenarios=1`: number of scenarios per instance
(used when `train_dataset` is not provided)
- `context_per_instance=1`: number of contexts per instance
(used when `train_dataset` is not provided)
- `seed=nothing`: random seed for reproducibility
(used in `model` and `train_dataset` when not provided)
- `model`: statistical model to wrap in the policy
(defaults to `generate_statistical_model(benchmark; seed)`)
- `maximizer`: combinatorial oracle to wrap in the policy
(defaults to `generate_maximizer(benchmark)`)
- `anticipative_solver`: oracle used in pure-imitation iterations
(defaults to `generate_anticipative_solver(benchmark)`)
- `parametric_anticipative_solver`: parametric oracle wrapped in `PerturbedAdditive` for
mirror-descent iterations (defaults to `generate_parametric_anticipative_solver(benchmark)`)
- `train_dataset`: training dataset (defaults to `generate_dataset(benchmark, dataset_size; ...)`)
- `epochs=10`: number of inner training epochs per mirror descent iteration
- `iterations=10`: total number of mirror descent iterations
- `κ=1.0`: scaling factor applied to `θ` before passing it to the perturbed solver
- `metrics::Tuple=()`: metrics forwarded to the inner training algorithm
- `verbose=false`: if true, prints a banner at each iteration
- `imitation_start=true`: if true, run a pure imitation step against the anticipative solver as the
first iteration
"""
function train_policy(
    algorithm::MirrorDescent,
    benchmark::ExogenousStochasticBenchmark;
    dataset_size=30,
    nb_scenarios=1,
    context_per_instance=1,
    seed=nothing,
    model=generate_statistical_model(benchmark; seed=seed),
    maximizer=generate_maximizer(benchmark),
    anticipative_solver=generate_anticipative_solver(benchmark),
    parametric_anticipative_solver=generate_parametric_anticipative_solver(benchmark),
    train_dataset=generate_dataset(
        benchmark,
        dataset_size;
        nb_scenarios=nb_scenarios,
        contexts_per_instance=context_per_instance,
        seed=seed,
    ),
    epochs=10,
    iterations=10,
    κ=1.0,
    metrics::Tuple=(),
    verbose::Bool=false,
    imitation_start::Bool=true,
)
    policy = DFLPolicy(model, maximizer)

    histories_per_iteration = train_policy!(
        algorithm,
        policy,
        train_dataset,
        anticipative_solver,
        parametric_anticipative_solver;
        epochs=epochs,
        iterations=iterations,
        κ=κ,
        metrics=metrics,
        verbose=verbose,
        imitation_start=imitation_start,
        is_minimization=is_minimization_problem(benchmark),
    )

    return histories_per_iteration, policy
end
