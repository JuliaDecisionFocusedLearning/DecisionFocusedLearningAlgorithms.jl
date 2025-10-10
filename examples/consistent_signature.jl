# Consistent Metric Function Signature

using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks
using MLUtils: splitobs
using Statistics

b = ArgmaxBenchmark()
dataset = generate_dataset(b, 100)
train_instances, val_instances, test_instances = splitobs(dataset; at=(0.3, 0.3, 0.4))

model = generate_statistical_model(b; seed=0)
maximizer = generate_maximizer(b)

# ============================================================================
# NEW: ALL metric functions have the SAME signature!
# (model, maximizer, data, context) -> value
# ============================================================================

# Simple metric - just uses model, maximizer, and data
compute_gap = (model, max, data, ctx) -> compute_gap(b, data, model, max)

# Metric that also uses context
compute_gap_ratio =
    (model, max, data, ctx) -> begin
        # data is the dataset from 'on' parameter
        # context gives access to everything else
        train_gap = compute_gap(b, ctx.train_dataset, model, max)
        data_gap = compute_gap(b, data, model, max)
        return train_gap / data_gap
    end

# Metric that ignores data, just uses context
get_epoch = (model, max, data, ctx) -> ctx.epoch

# Metric that uses everything
complex_metric = (model, max, data, ctx) -> begin
    # Can access:
    # - model, max (always provided)
    # - data (the dataset from 'on')
    # - ctx.epoch
    # - ctx.train_dataset, ctx.validation_dataset
    # - ctx.training_loss, ctx.validation_loss
    gap = compute_gap(b, data, model, max)
    return gap * ctx.epoch  # silly example, but shows flexibility
end

# ============================================================================
# Usage - Same function signature works everywhere!
# ============================================================================

callbacks = [
    # on=:validation (default) - data will be validation_dataset
    Metric(:gap, compute_gap),
    # Creates: val_gap

    # on=:both - function called twice with train and val datasets
    Metric(:gap, compute_gap; on=:both),
    # Creates: train_gap, val_gap

    # on=test_instances - data will be test_instances
    Metric(:test_gap, compute_gap; on=test_instances),
    # Creates: test_gap

    # Complex metric using context
    Metric(:gap_ratio, compute_gap_ratio; on=:validation),
    # Creates: val_gap_ratio

    # Ignore data parameter completely
    Metric(:current_epoch, get_epoch),
    # Creates: val_current_epoch (on=:validation by default)
]

# ============================================================================
# Benefits of Consistent Signature
# ============================================================================

# ✅ ALWAYS the same signature: (model, max, data, ctx) -> value
# ✅ No confusion about what arguments metric_fn receives
# ✅ Easy to write - just follow one pattern
# ✅ Easy to compose - all functions compatible
# ✅ Full flexibility - context gives access to everything
# ✅ Can ignore unused parameters (data or parts of context)

# ============================================================================
# Comparison: OLD vs NEW
# ============================================================================

# OLD (inconsistent signatures):
# on=nothing    → metric_fn(context)                     # 1 arg
# on=:both      → metric_fn(model, maximizer, dataset)   # 3 args
# on=data       → metric_fn(model, maximizer, data)      # 3 args
# 😕 Confusing! Different signatures for different modes!

# NEW (consistent signature):
# Always: metric_fn(model, maximizer, data, context)     # 4 args
# ✨ Clear! Same signature everywhere!

# ============================================================================
# Practical Example: Define metrics once, use everywhere
# ============================================================================

# Define your metrics library with consistent signature
module MyMetrics
gap(model, max, data, ctx) = compute_gap(benchmark, data, model, max)
regret(model, max, data, ctx) = compute_regret(benchmark, data, model, max)
accuracy(model, max, data, ctx) = compute_accuracy(benchmark, data, model, max)

# Complex metric using context
function overfitting_indicator(model, max, data, ctx)
    train_metric = gap(model, max, ctx.train_dataset, ctx)
    val_metric = gap(model, max, ctx.validation_dataset, ctx)
    return val_metric - train_metric
end
end

# Use them easily
callbacks = [
    Metric(:gap, MyMetrics.gap; on=:both),
    Metric(:regret, MyMetrics.regret; on=:both),
    Metric(:test_accuracy, MyMetrics.accuracy; on=test_instances),
    Metric(:overfitting, MyMetrics.overfitting_indicator),
]

# ============================================================================
# Advanced: Higher-order functions
# ============================================================================

# Create a metric factory that returns properly-signed functions
function dataset_metric(benchmark, compute_fn)
    return (model, max, data, ctx) -> compute_fn(benchmark, data, model, max)
end

# Use it
callbacks = [
    Metric(:gap, dataset_metric(b, compute_gap); on=:both),
    Metric(:regret, dataset_metric(b, compute_regret); on=:both),
]

# ============================================================================
# Migration Helper
# ============================================================================

# If you have old-style functions: (model, max, data) -> value
# Wrap them easily:
old_compute_gap = (model, max, data) -> compute_gap(b, data, model, max)

# Convert to new signature:
new_compute_gap = (model, max, data, ctx) -> old_compute_gap(model, max, data)
# Or more concisely:
new_compute_gap = (model, max, data, _) -> old_compute_gap(model, max, data)

Metric(:gap, new_compute_gap; on=:both)
