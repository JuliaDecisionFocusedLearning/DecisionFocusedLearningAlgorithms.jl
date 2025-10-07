
function DAgger_train_model!(
    model,
    maximizer,
    train_environments,
    validation_environments,
    anticipative_policy;
    iterations=5,
    fyl_epochs=3,
    metrics_callbacks::NamedTuple=NamedTuple(),
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
    all_metrics = []
    for iter in 1:iterations
        println("DAgger iteration $iter")
        metrics = fyl_train_model!(
            model,
            maximizer,
            dataset,
            val_dataset;
            epochs=fyl_epochs,
            metrics_callbacks=metrics_callbacks,
        )
        push!(all_metrics, metrics)
        new_samples = eltype(dataset)[]
        # Dataset update
        for env in train_environments
            reset!(env; reset_rng=false)
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
                    action = target.y_true
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

    return _flatten_dagger_metrics(all_metrics)
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
    return DAgger_train_model!(
        model,
        maximizer,
        train_environments,
        validation_environments,
        anticipative_policy;
        kwargs...,
    )
end
