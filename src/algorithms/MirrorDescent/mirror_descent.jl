"""
$TYPEDEF

Mirror Descent algorithm for learning coordinated solutions.

This algorithm is designed for stochastic benchmarks.

Reference: <https://arxiv.org/abs/2505.04757>

# Fields
$TYPEDFIELDS
"""
@kwdef struct MirrorDescent{A} <: AbstractImitationAlgorithm
    "inner imitation algorithm for supervised learning"
    inner_algorithm::A = PerturbedFenchelYoungLossImitation()
end

"""
$TYPEDSIGNATURES
Generate a dataset for the provided benchmark and train a DFLPolicy using the Mirror Descent algorithm.

# Core training method
"""


function train_policy(
    algorithm::MirrorDescent,
    benchmark::ExogenousStochasticBenchmark;
    dataset_size=30,
    epochs=10,
    iterations=10,
    κ = 1.0,
    metrics::Tuple=(),
    seed=nothing,
)

    train_dataset = generate_dataset(benchmark, dataset_size; seed=seed)

    # Initialize model and create policy
    model = generate_statistical_model(benchmark; seed=seed)
    maximizer = generate_maximizer(benchmark)
    policy = DFLPolicy(model, maximizer)

    # vector because we store one history per iteration
    histories_per_iteration = MVHistory[]

    anticipative_solver = generate_anticipative_solver(benchmark;) 
    parametric_anticipative_solver = generate_parametric_anticipative_solver(benchmark;) 

    # perturb = true correspond to "real" iterations of mirror descent
    # we compute solutions with the penalized anticipative solver  + perturbation

    # perturb = false correspond to imitation learning
    # we use the anticipative solver without perturbation
    # usefull to start with one iteration of pure imitation learning
    perturb = false

    # Train policy
    for n_it in 1:iterations
        println("Iteration $n_it / $iterations")

        if n_it > 1
            perturb = true
        end


        # Generate anticipative solutions as training data
        augmented_dataset = augment_dataset(
            algorithm.inner_algorithm, benchmark, train_dataset, model, anticipative_solver, parametric_anticipative_solver;
            κ = κ, perturb = perturb
        )


        # Train policy on augmented dataset
        history = train_policy!(
            algorithm.inner_algorithm,
            policy,
            augmented_dataset;
            epochs = epochs,
            metrics = metrics,
            maximizer_kwargs=sample -> sample.context,
        )

        push!(histories_per_iteration, history)
    end

    return histories_per_iteration, policy
end


function augment_dataset(
    algorithm::PerturbedFenchelYoungLossImitation,
    bench::AbstractStochasticBenchmark,
    train_dataset::AbstractArray,
    model,
    anticipative_solver,
    parametric_anticipative_solver;
    κ = 1.0,
    perturb = false
)

    (; nb_samples, ε, threaded, training_optimizer, seed) = algorithm

    augmented_dataset = Vector{DataSample}()

    if perturb
        perturbed_maximizer = PerturbedAdditive(
            parametric_anticipative_solver; ε=κ*ε, nb_samples=nb_samples
        )
    end


    for sample in train_dataset

        θ = model(sample.x)

        if perturb
            if is_minimization_problem(bench)
                y = perturbed_maximizer(-κ*θ; scenario = sample.scenario, context = sample) 
            else
                y = perturbed_maximizer(κ*θ; scenario = sample.scenario, context = sample)
            end
        else
            y = anticipative_solver(sample.scenario; context = sample)
        end

        augmented_datasample = DataSample(;
            x = sample.x,
            y,
            instance = sample.context,
            extra = sample.extra
        )

        push!(augmented_dataset, augmented_datasample)
    end

    return augmented_dataset
end