"""
$TYPEDEF

A flexible metric that wraps a user-defined function.

This metric allows users to define custom metrics using functions. The function
receives the training context and optionally any stored data. It can return:
- A single value (stored with `metric.name`)
- A `NamedTuple` (each key-value pair stored separately)

# Fields
$TYPEDFIELDS

# Examples
```julia
# Simple metric using only context
epoch_metric = FunctionMetric(ctx -> ctx.epoch, :current_epoch)

# Metric with stored data (dataset)
gap_metric = FunctionMetric(:val_gap, val_data) do ctx, data
    compute_gap(benchmark, data, ctx.model, ctx.maximizer)
end

# Metric returning multiple values
dual_gap = FunctionMetric(:gaps, (train_data, val_data)) do ctx, datasets
    train_ds, val_ds = datasets
    return (
        train_gap = compute_gap(benchmark, train_ds, ctx.model, ctx.maximizer),
        val_gap = compute_gap(benchmark, val_ds, ctx.model, ctx.maximizer)
    )
end
```

# See also
- [`PeriodicMetric`](@ref) - Wrap a metric to evaluate periodically
- [`evaluate!`](@ref)
"""
struct FunctionMetric{F,D} <: AbstractMetric
    "function with signature `(context) -> value` or `(context, data) -> value`"
    metric_fn::F
    "identifier for the metric"
    name::Symbol
    "optional data stored in the metric (default: `nothing`)"
    data::D
end

"""
$TYPEDSIGNATURES

Construct a FunctionMetric without stored data.

The function should have signature `(context) -> value`.

# Arguments
- `metric_fn::Function` - Function to compute the metric
- `name::Symbol` - Identifier for the metric
"""
function FunctionMetric(metric_fn::F, name::Symbol) where {F}
    return FunctionMetric{F,Nothing}(metric_fn, name, nothing)
end

"""
$TYPEDSIGNATURES

Evaluate the function metric by calling the stored function.

# Arguments
- `metric::FunctionMetric` - The metric to evaluate
- `context` - TrainingContext with current training state

# Returns
- The value returned by `metric.metric_fn` (can be single value or NamedTuple)
```
"""
function evaluate!(metric::FunctionMetric, context::TrainingContext)
    if isnothing(metric.data)
        return metric.metric_fn(context)
    else
        return metric.metric_fn(context, metric.data)
    end
end
