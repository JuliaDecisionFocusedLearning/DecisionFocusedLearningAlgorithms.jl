"""
    LossAccumulator <: AbstractMetric

Accumulates loss values during training and computes their average.

This metric is used internally by training loops to track training loss.
It accumulates loss values via `update!` calls and computes the average via `compute`.

# Fields
- `name::Symbol` - Identifier for this metric (e.g., `:training_loss`)
- `total_loss::Float64` - Running sum of loss values
- `count::Int` - Number of samples accumulated

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
mutable struct LossAccumulator <: AbstractMetric
    const name::Symbol
    total_loss::Float64
    count::Int
end

"""
    LossAccumulator(name::Symbol=:training_loss)

Construct a LossAccumulator with the given name.

# Arguments
- `name::Symbol` - Identifier for the metric (default: `:training_loss`)

# Examples
```julia
train_metric = LossAccumulator(:training_loss)
val_metric = LossAccumulator(:validation_loss)
```
"""
function LossAccumulator(name::Symbol=:training_loss)
    return LossAccumulator(name, 0.0, 0)
end

"""
    reset!(metric::LossAccumulator)

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
    update!(metric::LossAccumulator, loss_value::Float64)

Add a loss value to the accumulator.

# Arguments
- `metric::LossAccumulator` - The accumulator to update
- `loss_value::Float64` - Loss value to add

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
    compute(metric::LossAccumulator; reset::Bool=true)

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

# Fields
- `name::Symbol` - Identifier for this metric (e.g., `:validation_loss`)
- `dataset::D` - Dataset to evaluate on (stored internally)
- `total_loss::Float64` - Running sum during evaluation
- `count::Int` - Number of samples evaluated

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
mutable struct FYLLossMetric{D} <: AbstractMetric
    const name::Symbol
    const dataset::D
    total_loss::Float64
    count::Int
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
    return FYLLossMetric(name, dataset, 0.0, 0)
end

"""
    reset!(metric::FYLLossMetric)

Reset the metric's accumulated loss to zero.
"""
function reset!(metric::FYLLossMetric)
    metric.total_loss = 0.0
    return metric.count = 0
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
    metric.total_loss += l
    metric.count += 1
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
function evaluate!(metric::FYLLossMetric, context)
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
    return metric.count == 0 ? 0.0 : metric.total_loss / metric.count
end
