# TODO: best_model saving method, using default metric validation loss, overwritten in dagger
# TODO: batch training option
# TODO: parallelize loss computation on validation set
# TODO: have supervised learning training method, where fyl_train calls it, therefore we can easily test new supervised losses if needed

@kwdef struct PerturbedImitationAlgorithm{O,S}
    nb_samples::Int = 10
    ε::Float64 = 0.1
    threaded::Bool = true
    training_optimizer::O = Adam()
    seed::S = nothing
end

reset!(algorithm::PerturbedImitationAlgorithm) = empty!(algorithm.history)

function train_policy!(
    algorithm::PerturbedImitationAlgorithm,
    model,
    maximizer,
    train_dataset::AbstractArray{<:DataSample};
    epochs=100,
    maximizer_kwargs=get_info,
    metrics::Tuple=(),
    reset=false,
)
    reset && reset!(algorithm)
    (; nb_samples, ε, threaded, training_optimizer, seed) = algorithm
    perturbed = PerturbedAdditive(maximizer; nb_samples, ε, threaded, seed)
    loss = FenchelYoungLoss(perturbed)

    opt_state = Flux.setup(training_optimizer, model)

    history = MVHistory()

    train_loss_metric = LossAccumulator(:training_loss)

    # Store initial losses (epoch 0)
    # Epoch 0
    for sample in train_dataset
        (; x, y) = sample
        val = loss(model(x), y; maximizer_kwargs(sample)...)
        update!(train_loss_metric, val)
    end
    push!(history, :training_loss, 0, compute(train_loss_metric))
    reset!(train_loss_metric)

    # Initial metric evaluation
    context = TrainingContext(; model=model, epoch=0, maximizer=maximizer, loss=loss)
    run_metrics!(history, metrics, context)

    @showprogress for epoch in 1:epochs
        # Training step
        for sample in train_dataset
            (; x, y) = sample
            val, grads = Flux.withgradient(model) do m
                loss(m(x), y; maximizer_kwargs(sample)...)
            end
            Flux.update!(opt_state, model, grads[1])
            update!(train_loss_metric, val)
        end

        # Store training loss
        push!(history, :training_loss, epoch, compute(train_loss_metric))
        reset!(train_loss_metric)

        # Evaluate all metrics - update epoch in context
        context.epoch = epoch
        run_metrics!(history, metrics, context)
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
