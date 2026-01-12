# TODO: best_model saving method, using default metric validation loss, overwritten in dagger
# TODO: batch training option
# TODO: parallelize loss computation on validation set
# TODO: have supervised learning training method, where fyl_train calls it, therefore we can easily test new supervised losses if needed

"""
$TYPEDEF

Structured imitation learning with a perturbed Fenchel-Young loss.

# Fields
$TYPEDFIELDS
"""
@kwdef struct PerturbedImitationAlgorithm{O,S} <: AbstractImitationAlgorithm
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
end

"""
$TYPEDSIGNATURES

Train a model using the Perturbed Imitation Algorithm on the provided training dataset.
"""
function train_policy!(
    algorithm::PerturbedImitationAlgorithm,
    model,
    maximizer,
    train_dataset::AbstractArray{<:DataSample};
    epochs=100,
    maximizer_kwargs=get_info,
    metrics::Tuple=(),
)
    (; nb_samples, ε, threaded, training_optimizer, seed) = algorithm
    perturbed = PerturbedAdditive(maximizer; nb_samples, ε, threaded, seed)
    loss = FenchelYoungLoss(perturbed)

    opt_state = Flux.setup(training_optimizer, model)

    history = MVHistory()

    train_loss_metric = FYLLossMetric(train_dataset, :training_loss)

    # Initial metric evaluation and training loss (epoch 0)
    context = TrainingContext(;
        model=model,
        epoch=0,
        maximizer=maximizer,
        maximizer_kwargs=maximizer_kwargs,
        loss=loss,
    )
    push!(history, :training_loss, 0, evaluate!(train_loss_metric, context))
    evaluate_metrics!(history, metrics, context)

    @showprogress for epoch in 1:epochs
        next_epoch!(context)
        # Training step
        for sample in train_dataset
            (; x, y) = sample
            val, grads = Flux.withgradient(model) do m
                loss(m(x), y; maximizer_kwargs(sample)...)
            end
            Flux.update!(opt_state, model, grads[1])
            update!(train_loss_metric, val)
        end

        # Log metrics
        push!(history, :training_loss, epoch, compute!(train_loss_metric))
        evaluate_metrics!(history, metrics, context)
    end

    # Plot training loss (or first metric if available)
    # if !isempty(metrics)
    #     X, Y = get(history, metrics[1].name)
    #     println(lineplot(X, Y; xlabel="Epoch", ylabel=string(metrics[1].name)))
    # else
    #     X, Y = get(history, :training_loss)
    #     println(lineplot(X, Y; xlabel="Epoch", ylabel="Training Loss"))
    # end
    return history
end

function fyl_train_model(
    initial_model,
    maximizer,
    train_dataset;
    algorithm=PerturbedImitationAlgorithm(),
    kwargs...,
)
    model = deepcopy(initial_model)
    history = train_policy!(algorithm, model, maximizer, train_dataset; kwargs...)
    return history, model
end
