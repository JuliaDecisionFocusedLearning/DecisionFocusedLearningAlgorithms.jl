# TODO: every N epochs
# TODO: best_model saving method, using default metric validation loss, overwritten in dagger
# TODO: Implement validation loss as a metric callback
# TODO: batch training option
# TODO: parallelize loss computation on validation set
# TODO: have supervised learning training method, where fyl_train calls it, therefore we can easily test new supervised losses if needed

@kwdef struct PerturbedImitationAlgorithm{O}
    nb_samples::Int = 10
    ε::Float64 = 0.1
    threaded::Bool = true
    training_optimizer::O = Adam()
end

function FYLLossMetric(
    algorithm::PerturbedImitationAlgorithm, dataset, name::Symbol, maximizer
)
    perturbed = PerturbedAdditive(
        maximizer;
        nb_samples=algorithm.nb_samples,
        ε=algorithm.ε,
        threaded=algorithm.threaded,
    )
    loss = FenchelYoungLoss(perturbed)
    return FYLLossMetric(loss, dataset, name)
end

reset!(algorithm::PerturbedImitationAlgorithm) = empty!(algorithm.history)

function train_policy!(
    algorithm::PerturbedImitationAlgorithm,
    model,
    maximizer,
    train_dataset::AbstractArray{<:DataSample},
    validation_dataset;
    epochs=100,
    maximizer_kwargs=get_info,
    metrics::Tuple=(),
    reset=false,
)
    reset && reset!(algorithm)
    (; nb_samples, ε, threaded, training_optimizer) = algorithm
    perturbed = PerturbedAdditive(maximizer; nb_samples, ε, threaded)
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
    context = TrainingContext(; model=model, epoch=0, maximizer=maximizer)

    # Evaluate all metrics
    for metric in metrics
        value = evaluate!(metric, context)
        push!(history, metric.name, 0, value)
    end

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

        # Evaluate all metrics
        context = TrainingContext(; model=model, epoch=epoch, maximizer=maximizer)

        for metric in metrics
            value = evaluate!(metric, context)
            push!(history, metric.name, epoch, value)
        end
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
    initial_model, maximizer, train_dataset, validation_dataset; kwargs...
)
    model = deepcopy(initial_model)
    return fyl_train_model!(model, maximizer, train_dataset, validation_dataset; kwargs...),
    model
end

function baty_train_model(
    b::AbstractStochasticBenchmark{true};
    epochs=10,
    metrics::Tuple=(),
)
    # Generate instances and environments
    dataset = generate_dataset(b, 30)
    train_instances, validation_instances, _ = splitobs(dataset; at=(0.3, 0.3))
    train_environments = generate_environments(b, train_instances)
    validation_environments = generate_environments(b, validation_instances)

    # Generate anticipative solutions
    train_dataset = vcat(
        map(train_environments) do env
            v, y = generate_anticipative_solution(b, env; reset_env=true)
            return y
        end...
    )

    val_dataset = vcat(map(validation_environments) do env
        v, y = generate_anticipative_solution(b, env; reset_env=true)
        return y
    end...)

    # Initialize model and maximizer
    model = generate_statistical_model(b)
    maximizer = generate_maximizer(b)

    # Train with metrics
    history = fyl_train_model!(
        model,
        maximizer,
        train_dataset,
        val_dataset;
        epochs=epochs,
        metrics=metrics,
        maximizer_kwargs=get_state,
    )

    return history, model
end
