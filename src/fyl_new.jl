# New implementation using the callback system with MVHistory

function fyl_train_model!(
    model,
    maximizer,
    train_dataset::AbstractArray{<:DataSample},
    validation_dataset;
    epochs=100,
    maximizer_kwargs=(sample -> (; instance=sample.info)),
    callbacks::Vector{<:TrainingCallback}=TrainingCallback[],
)
    perturbed = PerturbedAdditive(maximizer; nb_samples=50, ε=0.1, threaded=true)
    loss = FenchelYoungLoss(perturbed)

    optimizer = Adam()
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
    context = (
        epoch=0,
        model=model,
        maximizer=maximizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
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
        context = (
            epoch=epoch,
            model=model,
            maximizer=maximizer,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
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
