# Using MVHistory for Metrics Storage

using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks
using MLUtils: splitobs
using ValueHistories
using Plots

b = ArgmaxBenchmark()
dataset = generate_dataset(b, 100)
train_instances, val_instances, test_instances = splitobs(dataset; at=(0.3, 0.3, 0.4))

model = generate_statistical_model(b; seed=0)
maximizer = generate_maximizer(b)

compute_gap_fn = (m, max, data) -> compute_gap(b, data, m, max)

# Define callbacks
callbacks = [
    Metric(:gap, compute_gap_fn; on=:both),
    Metric(:test_gap, compute_gap_fn; on=test_instances),
]

# Train and get MVHistory back
history = fyl_train_model!(
    model, maximizer, train_instances, val_instances; epochs=100, callbacks=callbacks
)

# ============================================================================
# Working with MVHistory - Much Cleaner!
# ============================================================================

# Get values and iterations
epochs, train_losses = get(history, :training_loss)
epochs, val_losses = get(history, :validation_loss)
epochs, train_gaps = get(history, :train_gap)
epochs, val_gaps = get(history, :val_gap)
test_epochs, test_gaps = get(history, :test_gap)

# Plot multiple metrics
plot(epochs, train_losses; label="Train Loss")
plot!(epochs, val_losses; label="Val Loss")

plot(epochs, train_gaps; label="Train Gap")
plot!(epochs, val_gaps; label="Val Gap")
plot!(test_epochs, test_gaps; label="Test Gap")

using JLD2
@save "training_history.jld2" history
@load "training_history.jld2" history
