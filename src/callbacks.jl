"""
    TrainingCallback

Abstract type for training callbacks. Callbacks are called at specific points during training
to compute metrics, log information, or modify training behavior.

# Interface
Implement `on_epoch_end` for your callback type:
- `on_epoch_end(callback, context)` - called after each training epoch

# Context Structure

All training algorithms provide a context NamedTuple with the following **core fields**:

## Required Fields (Always Present)
- `epoch::Int` - Current epoch number (0-indexed, where 0 is pre-training)
- `model` - The model being trained
- `maximizer` - The optimization solver/maximizer
- `train_dataset` - Training dataset
- `validation_dataset` - Validation dataset
- `train_loss::Float64` - Average training loss for this epoch
- `val_loss::Float64` - Average validation loss for this epoch

## Optional Fields (Algorithm-Specific)
Different algorithms may provide additional fields. Check with `haskey(context, :field_name)`:

**DAgger-Specific:**
- `α::Float64` - Expert/learner mixing parameter
- `dagger_iteration::Int` - Current DAgger iteration
- `expert_policy` - Expert policy function
- `train_environments` - Training environments
- `validation_environments` - Validation environments

**Future Algorithms:**
Other algorithms will add their own specific fields as needed.

# Writing Portable Metrics

To write metrics that work across all algorithms, use only the core fields:

```julia
# Works with any algorithm
Metric(:gap, ctx -> compute_gap(benchmark, ctx.validation_dataset, ctx.model, ctx.maximizer))

# Works with any algorithm
Metric(:loss_ratio, ctx -> ctx.val_loss / ctx.train_loss; on=:none)
```

To write algorithm-specific metrics, check for optional fields:

```julia
# DAgger-specific metric
Metric(:alpha, ctx -> haskey(ctx, :α) ? ctx.α : NaN; on=:none)
```

# See Also
- [`Metric`](@ref) - Generic callback for computing metrics
- [`on_epoch_end`](@ref) - Callback interface method
"""
abstract type TrainingCallback end

"""
    on_epoch_end(callback::TrainingCallback, context)

Called at the end of each training epoch. Should return a `NamedTuple` of metrics
or `nothing` if no metrics to record.

# Arguments
- `callback`: The callback instance
- `context`: NamedTuple with training state (epoch, model, datasets, losses, etc.)

# Returns
- `NamedTuple` with metric name(s) and value(s), or `nothing`

# Example
```julia
function on_epoch_end(cb::MyCallback, context)
    metric_value = compute_metric(context.model, context.validation_dataset)
    return (my_metric = metric_value,)
end
```
"""
function on_epoch_end(::TrainingCallback, context)
    return nothing
end

# ============================================================================
# Built-in Callbacks
# ============================================================================

"""
    Metric(name::Symbol, metric_fn; on=:validation)

Generic callback for computing metrics during training.

# Arguments
- `name`: Base name for the metric
- `metric_fn`: Function with signature `(data, context) -> value`
  - `data`: The data to compute metric on (from `on` parameter)
  - `context`: Full training context with model, maximizer, datasets, epoch, losses, etc.
- `on`: What data to use (default: `:validation`)
  - `:train` - use `context.train_dataset`, creates `train_<name>` metric
  - `:validation` - use `context.validation_dataset`, creates `val_<name>` metric
  - `:both` - compute on both, creates `train_<name>` and `val_<name>` metrics
  - Any other value - use that data directly, creates `name` metric

# Examples
```julia
# Most common: compute on validation set
Metric(:gap, (data, ctx) -> compute_gap(benchmark, data, ctx.model, ctx.maximizer))
# Creates: val_gap (default on=:validation)

# Compute on both train and validation
Metric(:gap, (data, ctx) -> compute_gap(benchmark, data, ctx.model, ctx.maximizer); on=:both)
# Creates: train_gap and val_gap

# Compute on specific dataset (e.g., test set)
Metric(:test_gap, (data, ctx) -> compute_gap(benchmark, data, ctx.model, ctx.maximizer);
       on=test_instances)
# Creates: test_gap

# Use context for complex metrics
Metric(:gap_ratio, (data, ctx) -> begin
    train_gap = compute_gap(b, ctx.train_dataset, ctx.model, ctx.maximizer)
    val_gap = compute_gap(b, data, ctx.model, ctx.maximizer)
    return train_gap / val_gap
end)

# If you don't need data parameter, just ignore it
Metric(:epoch, (data, ctx) -> ctx.epoch)
```
"""
struct Metric <: TrainingCallback
    name::Symbol
    metric_fn::Function
    on::Any  # :train, :validation, :both, or any data (dataset, environments, etc.)

    function Metric(name::Symbol, metric_fn; on=:validation)
        return new(name, metric_fn, on)
    end
end

function on_epoch_end(cb::Metric, context)
    try
        if cb.on == :train
            # Apply to training dataset
            value = cb.metric_fn(context.train_dataset, context)
            return NamedTuple{(Symbol("train_$(cb.name)"),)}((value,))

        elseif cb.on == :validation
            # Apply to validation dataset
            value = cb.metric_fn(context.validation_dataset, context)
            return NamedTuple{(Symbol("val_$(cb.name)"),)}((value,))

        elseif cb.on == :both || cb.on == [:train, :validation]
            # Apply to both datasets
            train_value = cb.metric_fn(context.train_dataset, context)
            val_value = cb.metric_fn(context.validation_dataset, context)
            return (;
                Symbol("train_$(cb.name)") => train_value,
                Symbol("val_$(cb.name)") => val_value,
            )

        else
            # Apply to provided data (dataset, environments, etc.)
            value = cb.metric_fn(cb.on, context)
            return NamedTuple{(cb.name,)}((value,))
        end

    catch e
        @warn "Metric $(cb.name) failed at epoch $(context.epoch)" exception = (
            e, catch_backtrace()
        )
        return nothing
    end
end

# ============================================================================
# Helper functions
# ============================================================================

"""
    run_callbacks!(history, callbacks::Vector{<:TrainingCallback}, context)

Run all callbacks and store their metrics in the history.

# Arguments
- `history`: MVHistory object to store metrics
- `callbacks`: Vector of callbacks to run
- `context`: Training context (epoch, model, datasets, etc.)
"""
function run_callbacks!(history, callbacks::Vector{<:TrainingCallback}, context)
    for callback in callbacks
        metrics = on_epoch_end(callback, context)
        if !isnothing(metrics)
            for (name, value) in pairs(metrics)
                push!(history, name, context.epoch, value)
            end
        end
    end
    return nothing
end

"""
    get_metric_names(callbacks::Vector{<:TrainingCallback})

Extract metric names from callbacks. For Metric with on=:both,
this will return both train_ and val_ prefixed names.
"""
function get_metric_names(callbacks::Vector{<:TrainingCallback})
    names = Symbol[]
    for callback in callbacks
        if isa(callback, Metric)
            # Handle different on modes
            if isnothing(callback.on)
                push!(names, callback.name)
            elseif callback.on == :train
                push!(names, Symbol("train_$(callback.name)"))
            elseif callback.on == :validation
                push!(names, Symbol("val_$(callback.name)"))
            elseif callback.on == :both || callback.on == [:train, :validation]
                push!(names, Symbol("train_$(callback.name)"))
                push!(names, Symbol("val_$(callback.name)"))
            else
                # Custom data (dataset, environments, etc.)
                push!(names, callback.name)
            end
        elseif hasfield(typeof(callback), :name)
            # Generic fallback for custom callbacks
            push!(names, callback.name)
        end
    end
    return names
end
