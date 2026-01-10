"""
    FunctionMetric{F,D} <: AbstractMetric

A flexible metric that wraps a user-defined function.

This metric allows users to define custom metrics using functions. The function
receives the training context and optionally any stored data. It can return:
- A single value (stored with `metric.name`)
- A `NamedTuple` (each key-value pair stored separately)

# Fields
- `name::Symbol` - Identifier for the metric
- `metric_fn::F` - Function with signature `(context) -> value` or `(context, data) -> value`
- `data::D` - Optional data stored in the metric (default: `nothing`)

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
    metric_fn::F
    name::Symbol
    data::D
end

"""
    FunctionMetric(metric_fn::Function, name::Symbol)

Construct a FunctionMetric without stored data.

The function should have signature `(context) -> value`.

# Arguments
- `metric_fn::Function` - Function to compute the metric
- `name::Symbol` - Identifier for the metric

# Examples
```julia
# Track current epoch
epoch_metric = FunctionMetric(ctx -> ctx.epoch, :epoch)

# Track model parameter norm
param_norm = FunctionMetric(:param_norm) do ctx
    sum(abs2, Flux.params(ctx.model))
end
```
"""
function FunctionMetric(metric_fn::F, name::Symbol) where {F}
    return FunctionMetric{F,Nothing}(metric_fn, name, nothing)
end

"""
    FunctionMetric(name::Symbol, metric_fn::Function, data)

Construct a FunctionMetric with stored data.

The function should have signature `(context, data) -> value`.

# Arguments
- `name::Symbol` - Identifier for the metric
- `metric_fn::Function` - Function to compute the metric
- `data` - Data to store in the metric (e.g., dataset, environments)

# Examples
```julia
# Gap metric with validation dataset
gap = FunctionMetric(:val_gap, val_dataset) do ctx, data
    compute_gap(benchmark, data, ctx.model, ctx.maximizer)
end

# Multiple datasets
dual_gap = FunctionMetric(:gaps, (train_data, val_data)) do ctx, datasets
    train_ds, val_ds = datasets
    return (train_gap=compute_gap(...), val_gap=compute_gap(...))
end
```
"""
# Constructor with data - uses default struct constructor FunctionMetric{F,D}(name, metric_fn, data)

"""
    evaluate!(metric::FunctionMetric, context)

Evaluate the function metric by calling the stored function.

# Arguments
- `metric::FunctionMetric` - The metric to evaluate
- `context` - TrainingContext with current training state

# Returns
- The value returned by `metric.metric_fn` (can be single value or NamedTuple)

# Examples
```julia
metric = FunctionMetric(ctx -> ctx.epoch, :epoch)
context = TrainingContext(model=model, epoch=5, maximizer=maximizer)
value = evaluate!(metric, context)  # Returns 5
```
"""
function evaluate!(metric::FunctionMetric, context)
    if isnothing(metric.data)
        return metric.metric_fn(context)
    else
        return metric.metric_fn(context, metric.data)
    end
end
