# TODO: optional (line)plot utils
abstract type AbstractMetric end

function reset!(metric::AbstractMetric) end
function update!(metric::AbstractMetric; kwargs...) end
function evaluate!(metric::AbstractMetric, context) end
function compute(metric::AbstractMetric) end

mutable struct LossAccumulator <: AbstractMetric
    const name::Symbol
    total_loss::Float64
    count::Int
end

function LossAccumulator(name::Symbol=:training_loss)
    return LossAccumulator(name, 0.0, 0)
end

function reset!(metric::LossAccumulator)
    metric.total_loss = 0.0
    return metric.count = 0
end

function update!(metric::LossAccumulator, loss_value::Float64)
    metric.total_loss += loss_value
    return metric.count += 1
end

function compute(metric::LossAccumulator; reset::Bool=true)
    value = metric.count == 0 ? 0.0 : metric.total_loss / metric.count
    reset && reset!(metric)
    return value
end

mutable struct FYLLossMetric{L<:FenchelYoungLoss,D} <: AbstractMetric
    const loss::L
    const name::Symbol
    const dataset::D
    total_loss::Float64
    count::Int
end

function FYLLossMetric(loss::FenchelYoungLoss, dataset, name::Symbol=:fyl_loss)
    return FYLLossMetric(loss, name, dataset, 0.0, 0)
end

# Reset the stored history
function reset!(metric::FYLLossMetric)
    metric.total_loss = 0.0
    return metric.count = 0
end

# Online update and accumulation of the FYL loss
function update!(metric::FYLLossMetric, θ, y_target; kwargs...)
    l = metric.loss(θ, y_target; kwargs...)
    metric.total_loss += l
    metric.count += 1
    return l
end

# Evaluate average FYL loss over a dataset using context
function evaluate!(metric::FYLLossMetric, context)
    reset!(metric)
    for sample in metric.dataset
        θ = context.model(sample.x)
        y_target = sample.y
        update!(metric, θ, y_target; sample.info...)
    end
    return compute(metric)
end

# Compute final average FYL loss
function compute(metric::FYLLossMetric)
    return metric.count == 0 ? 0.0 : metric.total_loss / metric.count
end

"""
    FunctionMetric{F,D}

A metric that wraps a user-defined function with signature `(context) -> value`.
Stores any needed data internally (e.g., dataset, environments).

# Fields
- `name::Symbol` - metric identifier  
- `metric_fn::F` - function with signature `(context) -> value`
- `data::D` - optional data stored in the metric (default: nothing)

# Examples
```julia
# Simple metric using only context
FunctionMetric(:epoch, ctx -> ctx.epoch)

# Metric with stored dataset
FunctionMetric(:val_gap, ctx -> compute_gap(benchmark, ctx.model, ctx.maximizer), validation_dataset)

# Metric with custom function
FunctionMetric(:custom, validation_dataset) do ctx, data
    # compute something with ctx.model, ctx.maximizer, and data
end
```
"""
struct FunctionMetric{F,D} <: AbstractMetric
    name::Symbol
    metric_fn::F
    data::D
end

# Constructor without data (stores nothing)
function FunctionMetric(metric_fn::F, name::Symbol) where {F}
    return FunctionMetric{F,Nothing}(name, metric_fn, nothing)
end

# Constructor with data - uses default struct constructor FunctionMetric{F,D}(name, metric_fn, data)

# Evaluate the function metric
function evaluate!(metric::FunctionMetric, context)
    if isnothing(metric.data)
        return metric.metric_fn(context)
    else
        return metric.metric_fn(context, metric.data)
    end
end
