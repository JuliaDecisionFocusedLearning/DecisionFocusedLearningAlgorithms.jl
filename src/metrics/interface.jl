"""
$TYPEDEF

Abstract base type for all metrics used during training.

All concrete metric types should implement:
- `evaluate!(metric, context)` - Evaluate the metric given a training context

# See also
- [`LossAccumulator`](@ref)
- [`FYLLossMetric`](@ref)
- [`FunctionMetric`](@ref)
- [`PeriodicMetric`](@ref)
"""
abstract type AbstractMetric end

"""
    evaluate!(metric::AbstractMetric, context::TrainingContext)

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
function evaluate! end

# ============================================================================
# Metric storage helpers
# ============================================================================

"""
$TYPEDSIGNATURES

Internal helper to store a single metric value in the history.
"""
function _store_metric_value!(
    history::MVHistory, metric_name::Symbol, epoch::Int, value::Number
)
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
$TYPEDSIGNATURES

Internal helper to store multiple metric values from a NamedTuple.
Each key-value pair is stored separately in the history.
"""
function _store_metric_value!(history::MVHistory, ::Symbol, epoch::Int, value::NamedTuple)
    for (key, val) in pairs(value)
        _store_metric_value!(history, Symbol(key), epoch, val)
    end
    return nothing
end

"""
$TYPEDSIGNATURES

Internal helper that skips storing when value is `nothing`.
Used by periodic metrics on epochs when they're not evaluated.
"""
function _store_metric_value!(::MVHistory, ::Symbol, ::Int, ::Nothing)
    return nothing
end

"""
$TYPEDSIGNATURES

Evaluate all metrics and store their results in the history.

This function handles three types of metric returns through multiple dispatch:
- **Single value**: Stored with the metric's name
- **NamedTuple**: Each key-value pair stored separately (for metrics that compute multiple values)
- **nothing**: Skipped (e.g., periodic metrics on epochs when not evaluated)

# Arguments
- `history::MVHistory` - MVHistory object to store metric values
- `metrics::Tuple` - Tuple of AbstractMetric instances to evaluate
- `context::TrainingContext` - TrainingContext with current training state (policy, epoch, etc.)

# Examples
```julia
# Create metrics
val_loss = FYLLossMetric(val_dataset, :validation_loss)
epoch_metric = FunctionMetric(ctx -> ctx.epoch, :current_epoch)

# Evaluate and store
context = TrainingContext(policy=policy, epoch=5)
evaluate_metrics!(history, (val_loss, epoch_metric), context)
```

# See also
- [`AbstractMetric`](@ref)
- [`evaluate!`](@ref)
"""
function evaluate_metrics!(history::MVHistory, metrics::Tuple, context::TrainingContext)
    for metric in metrics
        value = evaluate!(metric, context)
        _store_metric_value!(history, metric.name, context.epoch, value)
    end
    return nothing
end
