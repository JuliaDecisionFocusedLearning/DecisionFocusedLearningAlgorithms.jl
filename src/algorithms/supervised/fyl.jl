# TODO: best_model saving method, using default metric validation loss, overwritten in dagger
# TODO: batch training option
# TODO: parallelize loss computation on validation set
# TODO: have supervised learning training method, where fyl_train calls it, therefore we can easily test new supervised losses if needed

"""
$TYPEDEF

Structured imitation learning with a perturbed Fenchel-Young loss.

Reference: <https://arxiv.org/abs/2002.08676>

# Fields
$TYPEDFIELDS
"""
@kwdef struct PerturbedFenchelYoungLossImitation{O,S} <: AbstractImitationAlgorithm
    "number of perturbation samples"
    nb_samples::Int = 10
    "perturbation magnitude"
    ε::Float64 = 0.1
    "whether to use threading for perturbations"
    threaded::Bool = true
    "optimizer used for training"
    training_optimizer::O = Adam()
    "random seed for perturbations"
    seed::S = nothing
    "whether to use multiplicative perturbation (else additive)"
    use_multiplicative_perturbation::Bool = false
end

"""
$TYPEDSIGNATURES

Train a DFLPolicy using the Perturbed Fenchel-Young Loss Imitation Algorithm.

The `train_dataset` should be a `DataLoader` for batched training. Gradients are computed 
from the sum of losses across each batch before updating model parameters.

For unbatched training with a `Vector{DataSample}`, use the convenience method that 
automatically wraps the data in a DataLoader with batchsize=1.
"""
function train_policy!(
    algorithm::PerturbedFenchelYoungLossImitation,
    policy::DFLPolicy,
    train_dataset::DataLoader;
    epochs=100,
    metrics::Tuple=(),
    maximizer_kwargs=get_info,
)
    (; nb_samples, ε, threaded, training_optimizer, seed) = algorithm
    (; statistical_model, maximizer) = policy

    perturbed = if algorithm.use_multiplicative_perturbation
        PerturbedMultiplicative(maximizer; nb_samples, ε, threaded, seed)
    else
        PerturbedAdditive(maximizer; nb_samples, ε, threaded, seed)
    end
    loss = FenchelYoungLoss(perturbed)

    opt_state = Flux.setup(training_optimizer, statistical_model)

    history = MVHistory()

    train_loss_metric = FYLLossMetric(train_dataset.data, :training_loss)

    # Initial metric evaluation and training loss (epoch 0)
    context = TrainingContext(;
        policy=policy, epoch=0, loss=loss, maximizer_kwargs=maximizer_kwargs
    )
    push!(history, :training_loss, 0, evaluate!(train_loss_metric, context))
    evaluate_metrics!(history, metrics, context)

    @showprogress for epoch in 1:epochs
        next_epoch!(context)
        for batch in train_dataset
            val, grads = Flux.withgradient(statistical_model) do m
                mean(
                    loss(m(sample.x), sample.y; maximizer_kwargs(sample)...) for
                    sample in batch
                )
            end
            Flux.update!(opt_state, statistical_model, grads[1])
            update!(train_loss_metric, val)
        end

        # Log metrics
        push!(history, :training_loss, epoch, compute!(train_loss_metric))
        evaluate_metrics!(history, metrics, context)
    end

    return history
end

"""
$TYPEDSIGNATURES

Train a DFLPolicy using the Perturbed Fenchel-Young Loss Imitation Algorithm with unbatched data.

This convenience method wraps the dataset in a `DataLoader` with batchsize=1 and delegates 
to the batched training method. For custom batching behavior, create your own `DataLoader` 
and use the batched method directly.
"""
function train_policy!(
    algorithm::PerturbedFenchelYoungLossImitation,
    policy::DFLPolicy,
    train_dataset::AbstractArray{<:DataSample};
    epochs=100,
    metrics::Tuple=(),
    maximizer_kwargs=get_info,
)
    data_loader = DataLoader(train_dataset; batchsize=1, shuffle=false)
    return train_policy!(
        algorithm,
        policy,
        data_loader;
        epochs=epochs,
        metrics=metrics,
        maximizer_kwargs=maximizer_kwargs,
    )
end

"""
$TYPEDSIGNATURES

Train a DFLPolicy using the Perturbed Fenchel-Young Loss Imitation Algorithm on a benchmark.

# Benchmark convenience wrapper

This high-level function handles all setup from the benchmark and returns a trained policy.
"""
function train_policy(
    algorithm::PerturbedFenchelYoungLossImitation,
    benchmark::AbstractBenchmark;
    dataset_size=30,
    split_ratio=(0.3, 0.3),
    epochs=100,
    metrics::Tuple=(),
    seed=nothing,
)
    # Generate dataset and split
    dataset = generate_dataset(benchmark, dataset_size)
    train_instances, _, _ = splitobs(dataset; at=split_ratio)

    # Initialize model and create policy
    model = generate_statistical_model(benchmark; seed)
    maximizer = generate_maximizer(benchmark)
    policy = DFLPolicy(model, maximizer)

    # Train policy
    history = train_policy!(
        algorithm, policy, train_instances; epochs, metrics, maximizer_kwargs=get_info
    )

    return history, policy
end
