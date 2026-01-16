# # Basic Tutorial: Training with FYL on Argmax Benchmark
#
# This tutorial demonstrates the basic workflow for training a policy
# using the Perturbed Fenchel-Young Loss algorithm.

# ## Setup
using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks
using MLUtils: splitobs
using Plots

# ## Create Benchmark and Data
b = ArgmaxBenchmark()
dataset = generate_dataset(b, 100)
train_data, val_data, test_data = splitobs(dataset; at=(0.3, 0.3, 0.4))

# ## Create Policy
model = generate_statistical_model(b; seed=0)
maximizer = generate_maximizer(b)
policy = DFLPolicy(model, maximizer)

# ## Configure Algorithm
algorithm = PerturbedFenchelYoungLossImitation(;
    nb_samples=10, ε=0.1, threaded=true, seed=0
)

# ## Define Metrics to track during training
validation_loss_metric = FYLLossMetric(val_data, :validation_loss)

val_gap_metric = FunctionMetric(:val_gap, val_data) do ctx, data
    compute_gap(b, data, ctx.policy.statistical_model, ctx.policy.maximizer)
end

test_gap_metric = FunctionMetric(:test_gap, test_data) do ctx, data
    compute_gap(b, data, ctx.policy.statistical_model, ctx.policy.maximizer)
end

metrics = (validation_loss_metric, val_gap_metric, test_gap_metric)

# ## Train the Policy
history = train_policy!(algorithm, policy, train_data; epochs=100, metrics=metrics)

# ## Plot Results
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

# Plot loss evolution
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
