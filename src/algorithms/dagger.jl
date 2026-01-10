
function DAgger_train_model!(
    model,
    maximizer,
    train_environments,
    validation_environments,
    anticipative_policy;
    iterations=5,
    fyl_epochs=3,
    metrics::Tuple=(),
    algorithm::PerturbedImitationAlgorithm=PerturbedImitationAlgorithm(),
    maximizer_kwargs=get_state,
)
    α = 1.0
    train_dataset = vcat(map(train_environments) do env
        v, y = anticipative_policy(env; reset_env=true)
        return y
    end...)
    val_dataset = vcat(map(validation_environments) do env
        v, y = anticipative_policy(env; reset_env=true)
        return y
    end...)

    dataset = deepcopy(train_dataset)

    # Initialize combined history for all DAgger iterations
    combined_history = MVHistory()
    global_epoch = 0

    for iter in 1:iterations
        println("DAgger iteration $iter/$iterations (α=$(round(α, digits=3)))")

        # Train for fyl_epochs
        iter_history = train_policy!(
            algorithm,
            model,
            maximizer,
            dataset,
            val_dataset;
            epochs=fyl_epochs,
            metrics=metrics,
            maximizer_kwargs=maximizer_kwargs,
        )

        # Merge iteration history into combined history
        for key in keys(iter_history)
            epochs, values = get(iter_history, key)
            for i in 1:length(epochs)
                # Calculate global epoch number
                if iter == 1
                    # First iteration: use epochs as-is [0, 1, 2, ...]
                    global_epoch_value = epochs[i]
                else
                    # Later iterations: skip epoch 0 and renumber starting from global_epoch
                    if epochs[i] == 0
                        continue  # Skip epoch 0 for iterations > 1
                    end
                    # Map epoch 1 → global_epoch, epoch 2 → global_epoch+1, etc.
                    global_epoch_value = global_epoch + epochs[i] - 1
                end

                # For the epoch key, use global_epoch_value as both time and value
                # For other keys, use global_epoch_value as time and original value
                if key == :epoch
                    push!(combined_history, key, global_epoch_value, global_epoch_value)
                else
                    push!(combined_history, key, global_epoch_value, values[i])
                end
            end
        end

        # Update global_epoch for next iteration
        # After each iteration, advance by the number of non-zero epochs processed
        if iter == 1
            # First iteration processes all epochs [0, 1, ..., fyl_epochs]
            # Next iteration should start at fyl_epochs + 1
            global_epoch = fyl_epochs + 1
        else
            # Subsequent iterations skip epoch 0, so they process fyl_epochs epochs
            # Next iteration should start fyl_epochs later
            global_epoch += fyl_epochs
        end

        # Dataset update - collect new samples using mixed policy
        new_samples = eltype(dataset)[]
        for env in train_environments
            DecisionFocusedLearningBenchmarks.reset!(env; reset_rng=false)
            while !is_terminated(env)
                x_before = copy(observe(env)[1])
                _, anticipative_solution = anticipative_policy(env; reset_env=false)
                p = rand()
                target = anticipative_solution[1]
                x, state = observe(env)
                if size(target.x) != size(x)
                    @error "Mismatch between expert and observed state" size(target.x) size(
                        x
                    )
                end
                push!(new_samples, target)
                if p < α
                    action = target.y
                else
                    x, state = observe(env)
                    θ = model(x)
                    action = maximizer(θ; instance=state)  # ! not benchmark generic
                end
                step!(env, action)
            end
        end
        dataset = new_samples  # TODO: replay buffer
        α *= 0.9  # Decay factor for mixing expert and learned policy
    end

    return combined_history
end

function DAgger_train_model(b::AbstractStochasticBenchmark{true}; kwargs...)
    dataset = generate_dataset(b, 30)
    train_instances, validation_instances, _ = splitobs(dataset; at=(0.3, 0.3, 0.4))
    train_environments = generate_environments(b, train_instances; seed=0)
    validation_environments = generate_environments(b, validation_instances)
    model = generate_statistical_model(b)
    maximizer = generate_maximizer(b)
    anticipative_policy =
        (env; reset_env) -> generate_anticipative_solution(b, env; reset_env)
    history = DAgger_train_model!(
        model,
        maximizer,
        train_environments,
        validation_environments,
        anticipative_policy;
        kwargs...,
    )
    return history, model
end
