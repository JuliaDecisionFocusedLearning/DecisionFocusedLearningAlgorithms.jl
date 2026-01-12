using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks

using Flux
using InferOpt
using MLUtils
using Plots

b = ArgmaxBenchmark(; seed=42)
initial_model = generate_statistical_model(b; seed=0)
maximizer = generate_maximizer(b)
dataset = generate_dataset(b, 100; seed=0);
train_dataset, val_dataset = splitobs(dataset; at=(0.5, 0.5));

algorithm = PerturbedImitationAlgorithm(;
    nb_samples=20, ε=0.1, threaded=true, training_optimizer=Adam(), seed=0
)

validation_metric = FYLLossMetric(val_dataset, :validation_loss);
epoch_metric = FunctionMetric(ctx -> ctx.epoch, :current_epoch)

dual_gap_metric = FunctionMetric(:dual_gap, (train_dataset, val_dataset)) do ctx, datasets
    _train_dataset, _val_dataset = datasets
    train_gap = compute_gap(b, _train_dataset, ctx.model, ctx.maximizer)
    val_gap = compute_gap(b, _val_dataset, ctx.model, ctx.maximizer)
    return (train_gap=train_gap, val_gap=val_gap)
end

gap_metric = FunctionMetric(:validation_gap, val_dataset) do ctx, data
    compute_gap(b, data, ctx.model, ctx.maximizer)
end
periodic_gap = PeriodicMetric(gap_metric, 5)

gap_metric_offset = FunctionMetric(:delayed_gap, val_dataset) do ctx, data
    compute_gap(b, data, ctx.model, ctx.maximizer)
end
delayed_periodic_gap = PeriodicMetric(gap_metric_offset, 5; offset=10)

# Combine metrics
metrics = (
    validation_metric,
    epoch_metric,
    dual_gap_metric,       # Outputs both train_gap and val_gap every epoch
    periodic_gap,          # Outputs validation_gap every 5 epochs
    delayed_periodic_gap,  # Outputs delayed_gap every 5 epochs starting at epoch 10
);

model = deepcopy(initial_model)
history = train_policy!(
    algorithm, model, maximizer, train_dataset; epochs=50, metrics=metrics
)
X_train, Y_train = get(history, :training_loss)
X_val, Y_val = get(history, :validation_loss)
plot(
    X_train,
    Y_train;
    xlabel="Epoch",
    label="Training Loss",
    title="Training Loss over Epochs",
);
plot!(
    X_val,
    Y_val;
    xlabel="Epoch",
    label="Validation Loss",
    title="Validation Loss over Epochs",
)

plot(get(history, :validation_gap); xlabel="Epoch", title="Validation Gap over Epochs")
