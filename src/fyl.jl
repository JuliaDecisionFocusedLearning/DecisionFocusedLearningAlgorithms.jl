# TODO: every N epochs
# TODO: best_model saving method, using default metric validation loss, overwritten in dagger
# TODO: Implement validation loss as a metric callback
# TODO: batch training option
# TODO: parallelize loss computation on validation set
# TODO: have supervised learning training method, where fyl_train calls it, therefore we can easily test new supervised losses if needed

function fyl_train_model!(
    model,
    maximizer,
    train_dataset::AbstractArray{<:DataSample},
    validation_dataset;
    epochs=100,
    maximizer_kwargs=get_info,
    callbacks::Vector{<:TrainingCallback}=TrainingCallback[],
)
    perturbed = PerturbedAdditive(maximizer; nb_samples=10, ε=0.1, threaded=true)  # ! hardcoded
    loss = FenchelYoungLoss(perturbed)

    optimizer = Adam()  # ! hardcoded
    opt_state = Flux.setup(optimizer, model)

    # Initialize metrics storage with MVHistory
    history = MVHistory()

    # Compute initial losses
    initial_val_loss = mean([
        loss(model(sample.x), sample.y; maximizer_kwargs(sample)...) for
        sample in validation_dataset
    ])
    initial_train_loss = mean([
        loss(model(sample.x), sample.y; maximizer_kwargs(sample)...) for
        sample in train_dataset
    ])

    # Store initial losses (epoch 0)
    push!(history, :training_loss, 0, initial_train_loss)
    push!(history, :validation_loss, 0, initial_val_loss)

    # Initial callback evaluation
    context = TrainingContext(;
        model=model,
        epoch=0,
        maximizer=maximizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_loss=initial_train_loss,
        val_loss=initial_val_loss,
    )
    run_callbacks!(history, callbacks, context)

    @showprogress for epoch in 1:epochs
        # Training step
        epoch_train_loss = 0.0
        for sample in train_dataset
            (; x, y) = sample
            val, grads = Flux.withgradient(model) do m
                loss(m(x), y; maximizer_kwargs(sample)...)
            end
            epoch_train_loss += val
            Flux.update!(opt_state, model, grads[1])
        end
        avg_train_loss = epoch_train_loss / length(train_dataset)

        # Validation step
        epoch_val_loss = 0.0
        for sample in validation_dataset
            (; x, y) = sample
            epoch_val_loss += loss(model(x), y; maximizer_kwargs(sample)...)
        end
        avg_val_loss = epoch_val_loss / length(validation_dataset)

        # Store losses
        push!(history, :training_loss, epoch, avg_train_loss)
        push!(history, :validation_loss, epoch, avg_val_loss)

        # Run callbacks
        context = TrainingContext(;
            model=model,
            epoch=epoch,
            maximizer=maximizer,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
        )
        run_callbacks!(history, callbacks, context)
    end

    # Get validation loss values for plotting
    a, b = get(history, :validation_loss)
    println(lineplot(a, b; xlabel="Epoch", ylabel="Validation Loss"))
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
    callbacks::Vector{<:TrainingCallback}=TrainingCallback[],
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

    # Train with callbacks
    history = fyl_train_model!(
        model,
        maximizer,
        train_dataset,
        val_dataset;
        epochs=epochs,
        callbacks=callbacks,
        maximizer_kwargs=get_state,
    )

    return history, model
end
