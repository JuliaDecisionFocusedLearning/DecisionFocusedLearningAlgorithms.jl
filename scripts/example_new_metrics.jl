# Example: Using the New Metric System

using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks
using MLUtils

# Setup benchmark and data
benchmark = ArgmaxBenchmark()
dataset = generate_dataset(benchmark, 100)
train_data, val_data = splitobs(dataset; at=(0.6, 0.4))

# Initialize model and algorithm
initial_model = generate_statistical_model(benchmark)
maximizer = generate_maximizer(benchmark)
algorithm = PerturbedImitationAlgorithm(; nb_samples=10, ε=0.1, threaded=true)

# Create metrics
# 1. Validation loss metric (stores validation dataset)
val_loss_metric = FYLLossMetric(algorithm, val_data, :validation_loss, maximizer)

# 2. Simple function metrics (no data stored)
epoch_metric = FunctionMetric(ctx -> ctx.epoch, :current_epoch)

# 3. Metrics with stored data
gap_metric = FunctionMetric(
    ctx -> compute_gap(benchmark, val_data, ctx.model, ctx.maximizer), :validation_gap
)

# Combine all metrics
metrics = (val_loss_metric, epoch_metric, gap_metric)

# Train with metrics
model = deepcopy(initial_model)
history = train_policy!(
    algorithm, model, maximizer, train_data, val_data; epochs=50, metrics=metrics
)

println("\n=== Training Complete ===")
println("Metrics tracked: ", keys(history))
println("\nFinal epoch: ", last(get(history, :current_epoch)[2]))
println("Final validation loss: ", last(get(history, :validation_loss)[2]))
println("Final validation gap: ", last(get(history, :validation_gap)[2]))

plot(get(history, :validation_gap))