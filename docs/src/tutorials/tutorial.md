```@meta
EditURL = "tutorial.jl"
```

Tutorial

````@example tutorial
using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks
using MLUtils: splitobs
using Plots

b = ArgmaxBenchmark()
dataset = generate_dataset(b, 100)
train_instances, validation_instances, test_instances = splitobs(
    dataset; at=(0.3, 0.3, 0.4)
)

model = generate_statistical_model(b; seed=0)
maximizer = generate_maximizer(b)
````

Compute initial gap

````@example tutorial
initial_gap = compute_gap(b, test_instances, model, maximizer)
println("Initial test gap: $initial_gap")
````

Configure the training algorithm

````@example tutorial
algorithm = PerturbedImitationAlgorithm(;
    nb_samples=10, ε=0.1, threaded=true, seed=0
)
````

Define metrics to track during training

````@example tutorial
validation_loss_metric = FYLLossMetric(validation_instances, :validation_loss)
````

Validation gap metric

````@example tutorial
val_gap_metric = FunctionMetric(:val_gap, validation_instances) do ctx, data
    compute_gap(b, data, ctx.model, ctx.maximizer)
end
````

Test gap metric

````@example tutorial
test_gap_metric = FunctionMetric(:test_gap, test_instances) do ctx, data
    compute_gap(b, data, ctx.model, ctx.maximizer)
end
````

Combine metrics

````@example tutorial
metrics = (validation_loss_metric, val_gap_metric, test_gap_metric)
````

Train the model

````@example tutorial
fyl_model = deepcopy(model)
history = train_policy!(
    algorithm,
    fyl_model,
    maximizer,
    train_instances,
    validation_instances;
    epochs=100,
    metrics=metrics,
)
````

Plot validation and test gaps

````@example tutorial
val_gap_epochs, val_gap_values = get(history, :val_gap)
test_gap_epochs, test_gap_values = get(history, :test_gap)

plot(
    [val_gap_epochs, test_gap_epochs],
    [val_gap_values, test_gap_values];
    labels=["Val Gap" "Test Gap"],
    xlabel="Epoch",
    ylabel="Gap",
    title="Gap Evolution During Training",
)
````

Plot validation loss

````@example tutorial
train_loss_epochs, train_loss_values = get(history, :training_loss)
val_loss_epochs, val_loss_values = get(history, :validation_loss)

plot(
    [train_loss_epochs, val_loss_epochs],
    [train_loss_values, val_loss_values];
    labels=["Training Loss" "Validation Loss"],
    xlabel="Epoch",
    ylabel="Loss",
    title="Loss Evolution During Training",
)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

