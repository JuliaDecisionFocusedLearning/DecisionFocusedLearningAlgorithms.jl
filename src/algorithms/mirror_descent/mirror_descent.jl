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

"""
$TYPEDSIGNATURES

Train a DFLPolicy using the Mirror Descent algorithm on a provided training dataset.

# Core training method

# Arguments
- `epochs`: number of training epochs per iteration
- `iterations`: number of mirror descent iterations
- `κ`: scaling factor for the perturbation magnitude
- `metrics`: tuple of metrics to track during training
- `verbose`: if true, prints progress at each iteration
- `imitation_start`: if true, the first iteration uses pure imitation learning (no perturbation)
"""

function train_policy!(
    benchmark::ExogenousStochasticBenchmark,
    algorithm::MirrorDescent,
    policy::DFLPolicy,
    train_dataset,
    anticipative_solver,
    perturbed_anticipative_solver;
    epochs=10,
    iterations=10,
    κ=1.0,
    metrics::Tuple=(),
    verbose::Bool=false,
    imitation_start::Bool=true,
)
    augmented_dataset = train_dataset
    return map(1:iterations) do n_it
        if verbose
            println("Iteration $n_it / $iterations")
        end

        perturb = n_it > 1 || !imitation_start

        augmented_dataset = augment_dataset(
            benchmark,
            augmented_dataset,
            policy.statistical_model,
            anticipative_solver,
            perturbed_anticipative_solver;
            κ=κ,
            perturb=perturb,
        )

        train_policy!(
            algorithm.inner_algorithm,
            policy,
            augmented_dataset;
            epochs=epochs,
            metrics=metrics,
            maximizer_kwargs=sample -> sample.context,
        )
    end
end

"""
$TYPEDSIGNATURES

Generate a dataset for the provided benchmark and train a DFLPolicy using the Mirror Descent algorithm.

# Benchmark convenience wrapper

This high-level function handles all setup from the benchmark and returns a trained policy.

# Arguments
- `dataset_size`: number of samples in the training dataset
- `epochs`: number of training epochs per iteration
- `iterations`: number of mirror descent iterations
- `κ`: scaling factor for the perturbation magnitude
- `metrics`: tuple of metrics to track during training
- `seed`: random seed for reproducibility
- `verbose`: if true, prints progress at each iteration
- `imitation_start`: if true, the first iteration uses pure imitation learning (no perturbation)
- `model_kwargs`: additional keyword arguments passed to `generate_statistical_model`
- `maximizer_kwargs`: additional keyword arguments passed to `generate_maximizer`
- `solver_kwargs`: additional keyword arguments passed to `generate_anticipative_solver` and `generate_parametric_anticipative_solver`
- `nb_scenarios`: number of scenarios per instance. 
- `context_per_instance`: number of contexts per instance. 
"""

function train_policy(
    algorithm::MirrorDescent,
    benchmark::ExogenousStochasticBenchmark;
    dataset_size=30,
    epochs=10,
    iterations=10,
    κ=1.0,
    metrics::Tuple=(),
    seed=nothing,
    verbose::Bool=false,
    imitation_start::Bool=true,
    model_kwargs=(;),
    maximizer_kwargs=(;),
    solver_kwargs=(;),
    nb_scenarios=1,
    context_per_instance=1,
)
    train_dataset = generate_dataset(
        benchmark,
        dataset_size;
        nb_scenarios=nb_scenarios,
        contexts_per_instance=context_per_instance,
        seed=seed,
    )

    model = generate_statistical_model(benchmark; seed=seed, model_kwargs...)
    maximizer = generate_maximizer(benchmark; maximizer_kwargs...)
    policy = DFLPolicy(model, maximizer)

    anticipative_solver = generate_anticipative_solver(benchmark; solver_kwargs...)
    parametric_anticipative_solver = generate_parametric_anticipative_solver(
        benchmark; solver_kwargs...
    )
    (; nb_samples, ε, threaded, seed) = algorithm.inner_algorithm
    perturbed_anticipative_solver = PerturbedAdditive(
        (θ; scenario, kwargs...) -> parametric_anticipative_solver(θ, scenario; kwargs...);
        ε=κ * ε,
        nb_samples=nb_samples,
        seed=seed,
        threaded=threaded,
    )

    histories_per_iteration = train_policy!(
        benchmark,
        algorithm,
        policy,
        train_dataset,
        anticipative_solver,
        perturbed_anticipative_solver;
        epochs=epochs,
        iterations=iterations,
        κ=κ,
        metrics=metrics,
        verbose=verbose,
        imitation_start=imitation_start,
    )

    return histories_per_iteration, policy
end

function augment_dataset(
    bench::ExogenousStochasticBenchmark,
    train_dataset::AbstractArray,
    model,
    anticipative_solver,
    perturbed_anticipative_solver;
    κ=1.0,
    perturb=false,
)
    return _augment_dataset(
        Val(fieldtype(eltype(train_dataset), :y) !== Nothing),
        bench,
        train_dataset,
        model,
        anticipative_solver,
        perturbed_anticipative_solver;
        κ=κ,
        perturb=perturb,
    )
end

# Raw dataset (samples have no y) → create new DataSamples
function _augment_dataset(
    ::Val{false},
    bench,
    train_dataset,
    model,
    anticipative_solver,
    perturbed_anticipative_solver;
    κ=1.0,
    perturb=false,
)
    return map(train_dataset) do sample
        θ = model(sample.x)
        if perturb
            if is_minimization_problem(bench)
                y = perturbed_anticipative_solver(
                    -κ * θ; scenario=sample.scenario, sample.context...
                )
            else
                y = perturbed_anticipative_solver(
                    κ * θ; scenario=sample.scenario, sample.context...
                )
            end
        else
            y = anticipative_solver(sample.scenario; sample.context...)
        end
        DataSample(sample; y=y)
    end
end

# Augmented dataset (samples already have y) → update y in place
function _augment_dataset(
    ::Val{true},
    bench,
    train_dataset,
    model,
    anticipative_solver,
    perturbed_anticipative_solver;
    κ=1.0,
    perturb=false,
)
    for (i, sample) in enumerate(train_dataset)
        θ = model(sample.x)
        if perturb
            if is_minimization_problem(bench)
                y = perturbed_anticipative_solver(
                    -κ * θ; scenario=sample.scenario, sample.context...
                )
            else
                y = perturbed_anticipative_solver(
                    κ * θ; scenario=sample.scenario, sample.context...
                )
            end
        else
            y = anticipative_solver(sample.scenario; sample.context...)
        end
        ET = eltype(sample.y)
        y_converted = convert(typeof(sample.y), ET <: Integer ? round.(ET, y) : y)
        train_dataset[i] = DataSample(sample; y=y_converted)
    end
    return train_dataset
end
