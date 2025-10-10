# Simplified Metric Signature - Just (data, context)!

using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks
using MLUtils: splitobs

b = ArgmaxBenchmark()
dataset = generate_dataset(b, 100)
train, val, test = splitobs(dataset; at=(0.3, 0.3, 0.4))
model = generate_statistical_model(b)
maximizer = generate_maximizer(b)

# ============================================================================
# NEW: Metric functions take just 2 arguments: (data, context)
# Everything you need is in context!
# ============================================================================

# Simple metric - model and maximizer from context
compute_gapp = (data, ctx) -> compute_gap(b, data, ctx.model, ctx.maximizer)

# Complex metric - access other datasets from context
compute_ratio =
    (data, ctx) -> begin
        train_gap = compute_gap(b, ctx.train_dataset, ctx.model, ctx.maximizer)
        val_gap = compute_gap(b, data, ctx.model, ctx.maximizer)
        return train_gap / val_gap
    end

# Context-only metrics - ignore data completely
get_epoch = (_, ctx) -> ctx.epoch

# ============================================================================
# Usage Examples
# ============================================================================

callbacks = [
    # Default: on=:validation
    Metric(:gap, compute_gap),
    # Creates: val_gap

    # Automatic train and validation
    Metric(:gap, compute_gapp; on=:both),
    # Creates: train_gap, val_gap

    # Specific test set
    Metric(:test_gap, compute_gapp; on=test),
    # Creates: test_gap

    # Complex metric using context
    Metric(:gap_ratio, compute_ratio),
    # Creates: val_gap_ratio

    # Context-only metrics
    Metric(:current_epoch, get_epoch),
]

# Note: training_loss and validation_loss are automatically tracked in history!
# Access them with: get(history, :training_loss), get(history, :validation_loss)

history = fyl_train_model!(model, maximizer, train, val; epochs=100, callbacks=callbacks)

# ============================================================================
# Why This is Better
# ============================================================================

# BEFORE: Redundant parameters (4 arguments)
# metric_fn(model, maximizer, data, context)
# - model and maximizer are ALSO in context (redundant!)
# - Longer signature
# - More typing

# AFTER: Clean and minimal (2 arguments)
# metric_fn(data, context)
# - Get model from ctx.model
# - Get maximizer from ctx.maximizer
# - Everything in one place (context)
# - Shorter, cleaner

# ============================================================================
# Real-World Example
# ============================================================================

# Define your metric functions
compute_gap = (data, ctx) -> compute_gap(benchmark, data, ctx.model, ctx.maximizer)
compute_regret = (data, ctx) -> compute_regret(benchmark, data, ctx.model, ctx.maximizer)

# Metric that uses multiple datasets
overfitting_indicator =
    (data, ctx) -> begin
        train_metric = compute_gap(b, ctx.train_dataset, ctx.model, ctx.maximizer)
        val_metric = compute_gap(b, ctx.validation_dataset, ctx.model, ctx.maximizer)
        return val_metric - train_metric
    end

# Metric that evaluates policy on environments
eval_policy = (envs, ctx) -> begin
    policy = Policy("", "", PolicyWrapper(ctx.model))
    rewards, _ = evaluate_policy!(policy, envs, 100)
    return mean(rewards)
end

test_envs = generate_environments(b, test)

callbacks = [
    Metric(:gap, compute_gap; on=:both),
    Metric(:regret, compute_regret; on=:both),
    Metric(:test_gap, compute_gap; on=test),
    Metric(:overfitting, overfitting_indicator),
    Metric(:test_reward, eval_policy; on=test_envs),
]

# ============================================================================
# Metric Library Pattern
# ============================================================================

# Create a module with all your metrics
module MyMetrics
gap(data, ctx) = compute_gap(benchmark, data, ctx.model, ctx.maximizer)
regret(data, ctx) = compute_regret(benchmark, data, ctx.model, ctx.maximizer)

# More complex metrics
overfitting(data, ctx) = begin
    train = gap(ctx.train_dataset, ctx)
    val = gap(ctx.validation_dataset, ctx)
    return val - train
end
end

# Use them
callbacks = [
    Metric(:gap, MyMetrics.gap; on=:both),
    Metric(:regret, MyMetrics.regret; on=:both),
    Metric(:overfitting, MyMetrics.overfitting),
]

# ============================================================================
# Migration from 4-argument signature
# ============================================================================

# If you have old 4-argument functions:
old_metric = (model, max, data, ctx) -> compute_gap(b, data, model, max)

# Convert to new 2-argument:
new_metric = (data, ctx) -> compute_gap(b, data, ctx.model, ctx.maximizer)

# Or just update inline:
Metric(:gap, (data, ctx) -> compute_gap(b, data, ctx.model, ctx.maximizer); on=:both)

# ============================================================================
# Benefits Summary
# ============================================================================

# ✅ Cleaner: 2 arguments instead of 4
# ✅ Less redundancy: No duplicate model/maximizer
# ✅ Consistent: Everything from context
# ✅ Simpler: Less to type and remember
# ✅ Flexible: Context has everything you need
