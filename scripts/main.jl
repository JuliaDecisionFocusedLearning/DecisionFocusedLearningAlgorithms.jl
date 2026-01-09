using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks

using Flux
using InferOpt
using MLUtils
using Plots

b = ArgmaxBenchmark()
initial_model = generate_statistical_model(b; seed=0)
maximizer = generate_maximizer(b)
dataset = generate_dataset(b, 100; seed=0);
train_dataset, val_dataset = splitobs(dataset; at=(0.5, 0.5));

algorithm = PerturbedImitationAlgorithm(;
    nb_samples=20, ε=0.1, threaded=true, training_optimizer=Adam()
)

validation_metric = FYLLossMetric(algorithm, val_dataset, :validation_loss, maximizer);

model = deepcopy(initial_model)
history = train_policy!(
    algorithm,
    model,
    maximizer,
    train_dataset,
    val_dataset;
    epochs=50,
    metrics=(validation_metric,),
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
