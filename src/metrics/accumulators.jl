"""
$TYPEDEF

Accumulates loss values during training and computes their average.

This metric is used internally by training loops to track training loss.
It accumulates loss values via `update!` calls and computes the average via `compute`.

# Fields
$TYPEDFIELDS

# Examples
```julia
metric = LossAccumulator(:training_loss)

# During training
for sample in dataset
    loss_value = compute_loss(model, sample)
    update!(metric, loss_value)
end

# Get average and reset
avg_loss = compute(metric)  # Automatically resets
```

# See also
- [`FYLLossMetric`](@ref)
- [`reset!`](@ref)
- [`update!`](@ref)
- [`compute`](@ref)
"""
mutable struct LossAccumulator
    "Identifier for this metric (e.g., `:training_loss`)"
    const name::Symbol
    "Running sum of loss values"
    total_loss::Float64
    "Number of samples accumulated"
    count::Int
end

"""
$TYPEDSIGNATURES

Construct a LossAccumulator with the given name.
Initializes total loss and count to zero.
"""
function LossAccumulator(name::Symbol=:training_loss)
    return LossAccumulator(name, 0.0, 0)
end

"""
$TYPEDSIGNATURES

Reset the accumulator to its initial state (zero total loss and count).

# Examples
```julia
metric = LossAccumulator()
update!(metric, 1.5)
update!(metric, 2.0)
reset!(metric)  # total_loss = 0.0, count = 0
```
"""
function reset!(metric::LossAccumulator)
    metric.total_loss = 0.0
    return metric.count = 0
end

"""
$TYPEDSIGNATURES

Add a loss value to the accumulator.

# Examples
```julia
metric = LossAccumulator()
update!(metric, 1.5)
update!(metric, 2.0)
compute(metric)  # Returns 1.75
```
"""
function update!(metric::LossAccumulator, loss_value::Float64)
    metric.total_loss += loss_value
    return metric.count += 1
end

"""
$TYPEDSIGNATURES

Compute the average loss from accumulated values.

# Arguments
- `metric::LossAccumulator` - The accumulator to compute from
- `reset::Bool` - Whether to reset the accumulator after computing (default: `true`)

# Returns
- `Float64` - Average loss (or 0.0 if no values accumulated)

# Examples
```julia
metric = LossAccumulator()
update!(metric, 1.5)
update!(metric, 2.5)
avg = compute(metric)  # Returns 2.0, then resets
```
"""
function compute(metric::LossAccumulator; reset::Bool=true)
    value = metric.count == 0 ? 0.0 : metric.total_loss / metric.count
    reset && reset!(metric)
    return value
end

# ============================================================================

"""
    FYLLossMetric{D} <: AbstractMetric

Metric for evaluating Fenchel-Young Loss over a dataset.

This metric stores a dataset and computes the average Fenchel-Young Loss
when `evaluate!` is called. Useful for tracking validation loss during training.
Can also be used in the algorithms to accumulate loss over training data.

# Fields
- `dataset::D` - Dataset to evaluate on (stored internally)
- `accumulator::LossAccumulator` - Embedded accumulator holding `name`, `total_loss`, and `count`.

# Examples
```julia
# Create metric with validation dataset
val_metric = FYLLossMetric(val_dataset, :validation_loss)

# Evaluate during training (called by run_metrics!)
context = TrainingContext(model=model, epoch=5, maximizer=maximizer, loss=loss)
avg_loss = evaluate!(val_metric, context)
```

# See also
- [`LossAccumulator`](@ref)
- [`FunctionMetric`](@ref)
"""
struct FYLLossMetric{D} <: AbstractMetric
    dataset::D
    accumulator::LossAccumulator
end

"""
    FYLLossMetric(dataset, name::Symbol=:fyl_loss)

Construct a FYLLossMetric for a given dataset.

# Arguments
- `dataset` - Dataset to evaluate on (should have samples with `.x`, `.y`, and `.info` fields)
- `name::Symbol` - Identifier for the metric (default: `:fyl_loss`)

# Examples
```julia
val_metric = FYLLossMetric(val_dataset, :validation_loss)
test_metric = FYLLossMetric(test_dataset, :test_loss)
```
"""
function FYLLossMetric(dataset, name::Symbol=:fyl_loss)
    return FYLLossMetric(dataset, LossAccumulator(name))
end

"""
    reset!(metric::FYLLossMetric)

Reset the metric's accumulated loss to zero.
"""
function reset!(metric::FYLLossMetric)
    return reset!(metric.accumulator)
end

function Base.getproperty(metric::FYLLossMetric, s::Symbol)
    if s === :name
        return metric.accumulator.name
    else
        return getfield(metric, s)
    end
end

"""
    update!(metric::FYLLossMetric, loss::FenchelYoungLoss, θ, y_target; kwargs...)

Update the metric with a single loss computation.

# Arguments
- `metric::FYLLossMetric` - The metric to update
- `loss::FenchelYoungLoss` - Loss function to use
- `θ` - Model prediction
- `y_target` - Target value
- `kwargs...` - Additional arguments passed to loss function

# Returns
- The computed loss value
"""
function update!(metric::FYLLossMetric, loss::FenchelYoungLoss, θ, y_target; kwargs...)
    l = loss(θ, y_target; kwargs...)
    update!(metric.accumulator, l)
    return l
end

"""
    evaluate!(metric::FYLLossMetric, context)

Evaluate the average Fenchel-Young Loss over the stored dataset.

This method iterates through the dataset, computes predictions using `context.model`,
and accumulates losses using `context.loss`. The dataset should be stored in the metric.

# Arguments
- `metric::FYLLossMetric` - The metric to evaluate
- `context` - TrainingContext with `model`, `loss`, and other fields

# Returns
- `Float64` - Average loss over the dataset

# Examples
```julia
val_metric = FYLLossMetric(val_dataset, :validation_loss)
context = TrainingContext(model=model, epoch=5, maximizer=maximizer, loss=loss)
avg_loss = evaluate!(val_metric, context)
```
"""
function evaluate!(metric::FYLLossMetric, context::TrainingContext)
    reset!(metric)
    for sample in metric.dataset
        θ = context.model(sample.x)
        y_target = sample.y
        update!(metric, context.loss, θ, y_target; sample.info...)
    end
    return compute(metric)
end

"""
    compute(metric::FYLLossMetric)

Compute the average loss from accumulated values.

# Returns
- `Float64` - Average loss (or 0.0 if no values accumulated)
"""
function compute(metric::FYLLossMetric)
    return compute(metric.accumulator)
end
