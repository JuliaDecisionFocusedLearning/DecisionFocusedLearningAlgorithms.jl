# Tutorial
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

compute_gap(b, test_instances, model, maximizer)

metrics_callbacks = (;
    :time => (model, maximizer, epoch) -> (epoch_time = time()),
    :gap => (;
        :val =>
            (model, maximizer, epoch) ->
                (gap = compute_gap(b, validation_instances, model, maximizer)),
        :test =>
            (model, maximizer, epoch) ->
                (gap = compute_gap(b, test_instances, model, maximizer)),
    ),
)

fyl_model = deepcopy(model)
log = fyl_train_model!(
    fyl_model,
    maximizer,
    train_instances,
    validation_instances;
    epochs=100,
    metrics_callbacks,
)

log[:gap]
plot(
    [log[:gap].val, log[:gap].test];
    labels=["Val Gap" "Test Gap"],
    xlabel="Epoch",
    ylabel="Gap",
)
plot(log[:validation_loss])
