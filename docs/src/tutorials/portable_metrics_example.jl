# Example: Writing Portable Metrics

using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks
using MLUtils

# Setup benchmark
benchmark = DynamicVehicleSchedulingBenchmark(; two_dimensional_features=false)
dataset = generate_dataset(benchmark, 50)
train_data, val_data, test_data = splitobs(dataset; at=(0.5, 0.25, 0.25))

# ============================================================================
# Example 1: Simple portable metrics (work with ALL algorithms)
# ============================================================================

# These metrics only use core context fields, so they work everywhere
portable_callbacks = [
    # Compute gap on validation set
    Metric(
        :gap,
        ctx -> compute_gap(benchmark, ctx.validation_dataset, ctx.model, ctx.maximizer),
    ),

    # Compute gap on training set
    Metric(
        :gap,
        ctx -> compute_gap(benchmark, ctx.train_dataset, ctx.model, ctx.maximizer);
        on=:train,
    ),

    # Loss improvement from epoch 0
    Metric(:loss_improvement, ctx -> begin
        if ctx.epoch == 0
            return 0.0
        end
        # You could store initial loss in a closure or use history
        return ctx.val_loss
    end; on=:none),

    # Loss ratio (overfitting indicator)
    Metric(:loss_ratio, ctx -> ctx.val_loss / ctx.train_loss; on=:none),

    # Just track epoch (useful for debugging)
    Metric(:epoch, ctx -> ctx.epoch; on=:none),
]

# ============================================================================
# Example 2: Use the SAME callbacks with different algorithms
# ============================================================================

# Train with FYL
println("Training with FYL...")
model_fyl = generate_statistical_model(benchmark)
maximizer = generate_maximizer(benchmark)

history_fyl, trained_model_fyl = fyl_train_model(
    model_fyl,
    maximizer,
    train_data,
    val_data;
    epochs=10,
    callbacks=portable_callbacks,  # Same callbacks!
)

# Train with DAgger
println("\nTraining with DAgger...")
model_dagger = generate_statistical_model(benchmark)

train_instances = [sample.info for sample in train_data]
val_instances = [sample.info for sample in val_data]
train_envs = generate_environments(benchmark, train_instances)
val_envs = generate_environments(benchmark, val_instances)

anticipative_policy =
    (env; reset_env) -> generate_anticipative_solution(benchmark, env; reset_env)

history_dagger, trained_model_dagger = DAgger_train_model!(
    model_dagger,
    maximizer,
    train_envs,
    val_envs,
    anticipative_policy;
    iterations=3,
    fyl_epochs=5,
    callbacks=portable_callbacks,  # Same callbacks work here too!
    maximizer_kwargs=(sample -> (; instance=sample.info.state)),
)

# ============================================================================
# Example 3: Extract and compare results
# ============================================================================

using Plots

# FYL results
fyl_epochs, fyl_gap = get(history_fyl, :val_gap)
fyl_loss_epochs, fyl_loss = get(history_fyl, :validation_loss)

# DAgger results  
dagger_epochs, dagger_gap = get(history_dagger, :val_gap)
dagger_loss_epochs, dagger_loss = get(history_dagger, :validation_loss)

# Plot gap comparison
plot(
    fyl_epochs,
    fyl_gap;
    label="FYL",
    xlabel="Epoch",
    ylabel="Validation Gap",
    title="Gap Comparison",
    linewidth=2,
)
plot!(dagger_epochs, dagger_gap; label="DAgger", linewidth=2)
savefig("gap_comparison.png")

# Plot loss comparison
plot(
    fyl_loss_epochs,
    fyl_loss;
    label="FYL",
    xlabel="Epoch",
    ylabel="Validation Loss",
    title="Loss Comparison",
    linewidth=2,
)
plot!(dagger_loss_epochs, dagger_loss; label="DAgger", linewidth=2)
savefig("loss_comparison.png")

println("\nResults:")
println("FYL final gap: ", fyl_gap[end])
println("DAgger final gap: ", dagger_gap[end])
println("FYL final loss: ", fyl_loss[end])
println("DAgger final loss: ", dagger_loss[end])

# ============================================================================
# Example 4: Algorithm-specific metrics (opt-in)
# ============================================================================

# These metrics check for algorithm-specific fields
dagger_specific_callbacks = [
    # Include all portable metrics
    portable_callbacks...,

    # DAgger-specific: track mixing parameter α
    Metric(:alpha, ctx -> begin
        if haskey(ctx, :α)
            return ctx.α
        else
            return NaN  # Not a DAgger algorithm
        end
    end; on=:none),
]

# This works with DAgger (will track α)
history_dagger2, model_dagger2 = DAgger_train_model!(
    generate_statistical_model(benchmark),
    maximizer,
    train_envs,
    val_envs,
    anticipative_policy;
    iterations=3,
    fyl_epochs=5,
    callbacks=dagger_specific_callbacks,
)

# Check if α was tracked
if haskey(history_dagger2, :alpha)
    α_epochs, α_values = get(history_dagger2, :alpha)
    println("\nDAgger α decay: ", α_values)
end

# This also works with FYL (α will be NaN, but no error)
history_fyl2, model_fyl2 = fyl_train_model(
    generate_statistical_model(benchmark),
    maximizer,
    train_data,
    val_data;
    epochs=10,
    callbacks=dagger_specific_callbacks,  # Same callbacks, graceful degradation
)

# ============================================================================
# Example 5: Reusable metric functions
# ============================================================================

# Define a reusable metric function
function create_gap_metric(benchmark; on=:validation)
    return Metric(
        :gap,
        ctx -> begin
            dataset = on == :validation ? ctx.validation_dataset : ctx.train_dataset
            return compute_gap(benchmark, dataset, ctx.model, ctx.maximizer)
        end;
        on=on,
    )
end

# Use it with different algorithms
gap_val = create_gap_metric(benchmark; on=:validation)
gap_train = create_gap_metric(benchmark; on=:train)

callbacks = [gap_val, gap_train]

# Works everywhere!
fyl_train_model(model_fyl, maximizer, train_data, val_data; epochs=10, callbacks=callbacks)
DAgger_train_model!(
    model_dagger,
    maximizer,
    train_envs,
    val_envs,
    anticipative_policy;
    iterations=3,
    fyl_epochs=5,
    callbacks=callbacks,
)

println("\n✅ All examples completed successfully!")
println("Key takeaway: Write metrics once, use them with ANY algorithm!")
