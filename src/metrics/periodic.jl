"""
$TYPEDEF

Wrapper that evaluates a metric only every N epochs.

This is useful for expensive metrics that don't need to be computed every epoch
(e.g., gap computation, test set evaluation).

# Fields
$TYPEDFIELDS

# Behavior
The metric is evaluated when `(epoch - offset) % frequency == 0`.
On other epochs, `evaluate!` returns `nothing` (which is skipped by `evaluate_metrics!`).

# Examples
```julia
# Evaluate gap every 5 epochs (at epochs 0, 5, 10, 15, ...)
gap_metric = FunctionMetric(:val_gap, val_data) do ctx, data
    compute_gap(benchmark, data, ctx.model, ctx.maximizer)
end
periodic_gap = PeriodicMetric(gap_metric, 5)

# Start at epoch 10, then every 5 epochs (at epochs 10, 15, 20, ...)
delayed_gap = PeriodicMetric(gap_metric, 5; offset=10)

# Evaluate only at final epoch (epoch 100 with offset=100, frequency=1)
final_test = PeriodicMetric(test_metric, 1; offset=100)
```

# See also
- [`FunctionMetric`](@ref)
- [`evaluate!`](@ref)
- [`evaluate_metrics!`](@ref)
"""
struct PeriodicMetric{M<:AbstractMetric} <: AbstractMetric
    "the wrapped metric to evaluate periodically"
    metric::M
    "evaluate every N epochs"
    frequency::Int
    "offset for the first evaluation"
    offset::Int
end

"""
$TYPEDSIGNATURES

Construct a PeriodicMetric that evaluates the wrapped metric every N epochs.
"""
function PeriodicMetric(metric::M, frequency::Int; offset::Int=0) where {M<:AbstractMetric}
    return PeriodicMetric{M}(metric, frequency, offset)
end

"""
$TYPEDSIGNATURES

Construct a PeriodicMetric from a function to be wrapped.
"""
function PeriodicMetric(metric_fn, frequency::Int; offset::Int=0)
    metric = FunctionMetric(metric_fn, :periodic_metric)
    return PeriodicMetric{typeof(metric)}(metric, frequency, offset)
end

"""
$TYPEDSIGNATURES

Delegate `name` property to the wrapped metric for seamless integration.
"""
function Base.getproperty(pm::PeriodicMetric, s::Symbol)
    if s === :name
        return getfield(pm, :metric).name
    else
        return getfield(pm, s)
    end
end

"""
$TYPEDSIGNATURES

List available properties of PeriodicMetric.
"""
function Base.propertynames(pm::PeriodicMetric, private::Bool=false)
    return (:metric, :frequency, :offset, :name)
end

"""
$TYPEDSIGNATURES

Evaluate the wrapped metric only if the current epoch matches the frequency pattern.

# Arguments
- `pm::PeriodicMetric` - The periodic metric wrapper
- `context` - TrainingContext with current epoch

# Returns
- The result of `evaluate!(pm.metric, context)` if epoch matches the pattern
- `nothing` otherwise (which is skipped by `evaluate_metrics!`)

# Examples
```julia
periodic = PeriodicMetric(gap_metric, 5)

# At epoch 0, 5, 10, 15, ... → evaluates the metric
# At epoch 1, 2, 3, 4, 6, ... → returns nothing
context = TrainingContext(model=model, epoch=5, maximizer=maximizer)
result = evaluate!(periodic, context)  # Evaluates gap_metric
```
"""
function evaluate!(pm::PeriodicMetric, context)
    if (context.epoch - pm.offset) % pm.frequency == 0
        return evaluate!(pm.metric, context)
    else
        return nothing  # Skip evaluation on this epoch
    end
end
