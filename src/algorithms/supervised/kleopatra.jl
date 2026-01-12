function kleopatra_train_model(
    b::AbstractStochasticBenchmark{true};
    epochs=10,
    metrics::Tuple=(),
    algorithm::PerturbedImitationAlgorithm=PerturbedImitationAlgorithm(),
)
    # Generate instances and environments
    dataset = generate_dataset(b, 30)
    train_instances, validation_instances, _ = splitobs(dataset; at=(0.3, 0.3))
    train_environments = generate_environments(b, train_instances)

    # Generate anticipative solutions
    train_dataset = vcat(
        map(train_environments) do env
            v, y = generate_anticipative_solution(b, env; reset_env=true)
            return y
        end...
    )

    # Initialize model and maximizer
    model = generate_statistical_model(b)
    maximizer = generate_maximizer(b)

    # Train with algorithm
    history = train_policy!(
        algorithm,
        model,
        maximizer,
        train_dataset;
        epochs=epochs,
        metrics=metrics,
        maximizer_kwargs=get_state,
    )

    return history, model
end
