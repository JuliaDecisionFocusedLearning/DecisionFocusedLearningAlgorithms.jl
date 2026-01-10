using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks

using Flux
using InferOpt
using MLUtils
using Plots

# Create Dynamic Vehicle Scheduling Problem benchmark
b = DynamicVehicleSchedulingBenchmark(; two_dimensional_features=true)

# Generate dataset and environments
dataset = generate_dataset(b, 9)
train_instances, val_instances, test_instances = splitobs(dataset; at=(0.5, 0.3, 0.2))

train_envs = generate_environments(b, train_instances; seed=0)
val_envs = generate_environments(b, val_instances; seed=1)

# Initialize model and maximizer
initial_model = generate_statistical_model(b; seed=0)
maximizer = generate_maximizer(b)

# Define anticipative (expert) policy
anticipative_policy = (env; reset_env) -> generate_anticipative_solution(b, env; reset_env)

# Configure training algorithm
algorithm = PerturbedImitationAlgorithm(;
    nb_samples=10, ε=0.1, threaded=true, training_optimizer=Adam(0.001), seed=0
)

# Define metrics to track during training
epoch_metric = FunctionMetric(ctx -> ctx.epoch, :current_epoch)

# You can add validation metrics if you have a validation function
# For now, we'll just track epochs
metrics = (epoch_metric,)

# Train using DAgger
println("Starting DAgger training on Dynamic Vehicle Scheduling Problem...")
model = deepcopy(initial_model)

history = DAgger_train_model!(
    model,
    maximizer,
    train_envs,
    val_envs,
    anticipative_policy;
    iterations=5,
    fyl_epochs=10,
    metrics=metrics,
    algorithm=algorithm,
)

# Plot training progress
X_train, Y_train = get(history, :training_loss)
plot(
    X_train,
    Y_train;
    xlabel="Epoch",
    ylabel="Training Loss",
    label="Training Loss",
    title="DAgger Training on Dynamic VSP",
    legend=:topright,
)

# Plot epoch tracking if available
if haskey(history, :current_epoch)
    X_epoch, Y_epoch = get(history, :current_epoch)
    println("Tracked epochs: ", Y_epoch)
end

println("\nTraining completed!")
println("Final training loss: ", Y_train[end])
println("Total epochs: ", length(Y_train) - 1)  # -1 because epoch 0 is included
