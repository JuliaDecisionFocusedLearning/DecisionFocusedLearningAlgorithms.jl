# # Training on Warcraft Shortest Path
#
# This tutorial demonstrates how to train a decision-focused learning policy
# on the Warcraft shortest path benchmark using the Perturbed Fenchel-Young Loss
# Imitation algorithm.

# ## Setup
#
# First, let's load the required packages:

using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks
using Flux
using MLUtils
using Plots
using Statistics

# ## Benchmark Setup
#
# The Warcraft benchmark involves predicting edge costs in a grid graph for shortest path problems.
# We'll create a benchmark instance and generate training data:

benchmark = WarcraftBenchmark()
dataset = generate_dataset(benchmark, 50)

# Split the dataset into training, validation, and test sets:
train_data, val_data = dataset[1:45], dataset[46:end]

# ## Creating a Policy
#
# A `DFLPolicy` combines a statistical model (neural network) with a combinatorial optimizer.
# The benchmark provides utilities to generate appropriate models and optimizers:

model = generate_statistical_model(benchmark)
maximizer = generate_maximizer(benchmark; dijkstra=true)
policy = DFLPolicy(model, maximizer)

# ## Configuring the Algorithm
#
# We'll use the Perturbed Fenchel-Young Loss Imitation algorithm:

algorithm = PerturbedFenchelYoungLossImitation(;
    nb_samples=100,          # Number of perturbation samples for gradient estimation
    ε=0.2,                  # Perturbation magnitude
    threaded=true,          # Use multi-threading for perturbations
    training_optimizer=Adam(1e-3),  # Flux optimizer with learning rate
    seed=42,                 # Random seed for reproducibility
    use_multiplicative_perturbation=true,  # Use multiplicative perturbations
)

# ## Setting Up Metrics
#
# We'll track several metrics during training:

# Validation loss metric
val_loss_metric = FYLLossMetric(val_data, :validation_loss)

# Validation gap metric
val_gap_metric = FunctionMetric(:val_gap, val_data) do ctx, data
    compute_gap(benchmark, data, ctx.policy.statistical_model, ctx.policy.maximizer)
end

# ## Training
#
# Now we train the policy:

data_loader = DataLoader(train_data; batchsize=50)
history = train_policy!(
    algorithm, policy, data_loader; epochs=50, metrics=(val_loss_metric, val_gap_metric)
)
# ## Results Analysis
#
# Let's examine the training progress:

# Extract training history
train_loss_epochs, train_loss_values = get(history, :training_loss)
val_loss_epochs, val_loss_values = get(history, :validation_loss)
val_gap_epochs, val_gap_values = get(history, :val_gap)

# Plot training and validation loss
p1 = plot(
    train_loss_epochs,
    train_loss_values;
    label="Training",
    xlabel="Epoch",
    ylabel="FYL Loss",
    title="Training Progress",
    linewidth=2,
)
plot!(p1, val_loss_epochs, val_loss_values; label="Validation", linewidth=2)

# Plot gap evolution
p2 = plot(
    val_gap_epochs,
    val_gap_values;
    label="Validation Gap",
    xlabel="Epoch",
    ylabel="Gap (Regret)",
    title="Decision Quality",
    linewidth=2,
)
