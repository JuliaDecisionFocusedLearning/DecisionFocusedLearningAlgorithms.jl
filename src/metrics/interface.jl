"""
    AbstractMetric

Abstract base type for all metrics used during training.

All concrete metric types should implement:
- `evaluate!(metric, context)` - Evaluate the metric given a training context
- Optionally: `reset!(metric)`, `update!(metric, ...)`, `compute(metric)`

# See also
- [`LossAccumulator`](@ref)
- [`FYLLossMetric`](@ref)
- [`FunctionMetric`](@ref)
- [`PeriodicMetric`](@ref)
"""
abstract type AbstractMetric end

"""
    reset!(metric::AbstractMetric)

Reset the internal state of a metric. Used for accumulator-style metrics.
"""
function reset!(metric::AbstractMetric) end

"""
    update!(metric::AbstractMetric; kwargs...)

Update the metric with new data. Used for accumulator-style metrics during training.
"""
function update!(metric::AbstractMetric; kwargs...) end

"""
    evaluate!(metric::AbstractMetric, context)

Evaluate the metric given the current training context.

# Arguments
- `metric::AbstractMetric` - The metric to evaluate
- `context::TrainingContext` - Current training state (model, epoch, maximizer, etc.)

# Returns
Can return:
- A single value (Float64, Int, etc.) - stored with `metric.name`
- A `NamedTuple` - each key-value pair stored separately
- `nothing` - skipped (e.g., periodic metrics on off-epochs)
"""
function evaluate!(metric::AbstractMetric, context) end

"""
    compute(metric::AbstractMetric)

Compute the final metric value from accumulated data. Used for accumulator-style metrics.
"""
function compute(metric::AbstractMetric) end

# ============================================================================
# Metric storage helpers
# ============================================================================

"""
    _store_metric_value!(history, metric_name, epoch, value)

Internal helper to store a single metric value in the history.
"""
function _store_metric_value!(history, metric_name, epoch, value)
    try
        push!(history, metric_name, epoch, value)
    catch e
        throw(
            ErrorException(
                "Failed to store metric '$metric_name' at epoch $epoch: $(e.msg)"
            ),
        )
    end
    return nothing
end

"""
    _store_metric_value!(history, metric_name, epoch, value::NamedTuple)

Internal helper to store multiple metric values from a NamedTuple.
Each key-value pair is stored separately in the history.
"""
function _store_metric_value!(
    history::MVHistory, metric_name::Symbol, epoch::Int, value::NamedTuple
)
    for (key, val) in pairs(value)
        push!(history, key, epoch, val)
    end
    return nothing
end

"""
    _store_metric_value!(history, metric_name, epoch, ::Nothing)

Internal helper that skips storing when value is `nothing`.
Used by periodic metrics on epochs when they're not evaluated.
"""
function _store_metric_value!(
    history::MVHistory, metric_name::Symbol, epoch::Int, ::Nothing
)
    return nothing
end

"""
    run_metrics!(history, metrics::Tuple, context)

Evaluate all metrics and store their results in the history.

This function handles three types of metric returns through multiple dispatch:
- **Single value**: Stored with the metric's name
- **NamedTuple**: Each key-value pair stored separately (for metrics that compute multiple values)
- **nothing**: Skipped (e.g., periodic metrics on epochs when not evaluated)

# Arguments
- `history` - MVHistory object to store metric values
- `metrics::Tuple` - Tuple of AbstractMetric instances to evaluate
- `context` - TrainingContext with current training state (model, epoch, maximizer, etc.)

# Examples
```julia
# Create metrics
val_loss = FYLLossMetric(val_dataset, :validation_loss)
epoch_metric = FunctionMetric(ctx -> ctx.epoch, :current_epoch)

# Evaluate and store
context = TrainingContext(model=model, epoch=5, maximizer=maximizer)
run_metrics!(history, (val_loss, epoch_metric), context)
```

# See also
- [`AbstractMetric`](@ref)
- [`evaluate!`](@ref)
"""
function run_metrics!(history::MVHistory, metrics::Tuple, context::TrainingContext)
    for metric in metrics
        value = evaluate!(metric, context)
        _store_metric_value!(history, metric.name, context.epoch, value)
    end
    return nothing
end
