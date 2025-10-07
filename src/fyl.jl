# TODO: every N epochs
# TODO: best_model saving method, using default metric validation loss, overwritten in dagger
# TODO: Implement validation loss as a metric callback
# TODO: batch training option
# TODO: parallelize loss computation on validation set
# TODO: have supervised learning training method, where fyl_train calls it, therefore we can easily test new supervised losses if needed
# TODO: easier way to define and provide metrics

function fyl_train_model!(
    model,
    maximizer,
    train_dataset::AbstractArray{<:DataSample},
    validation_dataset;
    epochs=100,
    maximizer_kwargs=(sample -> (; instance=sample.info)),
    metrics_callbacks::NamedTuple=NamedTuple(),
)
    perturbed = PerturbedAdditive(maximizer; nb_samples=50, ε=0.0, threaded=true, seed=0)
    loss = FenchelYoungLoss(perturbed)

    optimizer = Adam()
    opt_state = Flux.setup(optimizer, model)

    total_loss = 0.0
    for sample in validation_dataset
        (; x, y) = sample
        total_loss += loss(model(x), y; maximizer_kwargs(sample)...)
    end
    loss_history = [total_loss / length(validation_dataset)]

    total_train_loss = 0.0
    for sample in train_dataset
        (; x, y) = sample
        total_train_loss += loss(model(x), y; maximizer_kwargs(sample)...)
    end

    # Initialize metrics history with epoch 0 for type stability
    metrics_history = _initialize_nested_metrics(metrics_callbacks, model, maximizer, 0)

    # Add validation loss to metrics
    metrics_history = merge(
        metrics_history,
        (;
            validation_loss=[total_loss / length(validation_dataset)],
            training_loss=[total_train_loss / length(train_dataset)],
        ),
    )

    @showprogress for epoch in 1:epochs
        l = 0
        for sample in train_dataset
            (; x, y) = sample
            val, grads = Flux.withgradient(model) do m
                loss(m(x), y; maximizer_kwargs(sample)...)
            end
            l += val
            Flux.update!(opt_state, model, grads[1])
        end
        # Evaluate on validation set
        total_loss = 0.0
        for sample in validation_dataset
            (; x, y) = sample
            total_loss += loss(model(x), y; maximizer_kwargs(sample)...)
        end
        push!(loss_history, total_loss / length(validation_dataset))
        push!(metrics_history.validation_loss, total_loss / length(validation_dataset))
        # push!(metrics_history.training_loss, l / length(train_dataset))

        total_loss = 0.0
        for sample in train_dataset
            (; x, y) = sample
            total_loss += loss(model(x), y; maximizer_kwargs(sample)...)
        end
        push!(metrics_history.training_loss, total_loss / length(train_dataset))

        # Call metrics callbacks
        if !isempty(metrics_callbacks)
            epoch_metrics = _call_nested_callbacks(
                metrics_callbacks, model, maximizer, epoch
            )
            _push_nested_metrics!(metrics_history, epoch_metrics)
        end
    end
    println(
        lineplot(metrics_history.validation_loss; xlabel="Epoch", ylabel="Validation Loss")
    )
    return metrics_history
end

function fyl_train_model(b::AbstractBenchmark; kwargs...)
    dataset = generate_dataset(b, 20)
    train_dataset, validation_dataset, _ = splitobs(dataset; at=(0.3, 0.3, 0.4))
    model = generate_statistical_model(b)
    maximizer = generate_maximizer(b)
    return fyl_train_model!(model, maximizer, train_dataset, validation_dataset; kwargs...)
end

function baty_train_model(b::AbstractStochasticBenchmark{true})
    dataset = generate_dataset(b, 30)
    train_instances, validation_instances, test_instances = splitobs(
        dataset; at=(0.3, 0.3, 0.4)
    )
    train_environments = generate_environments(b, train_instances)
    validation_environments = generate_environments(b, validation_instances)

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

    model = generate_statistical_model(b)
    maximizer = generate_maximizer(b)

    return fyl_train_model!(model, maximizer, train_dataset, val_dataset; epochs=10)
end
