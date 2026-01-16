# Algorithm Interface

This page describes the unified interface for Decision-Focused Learning algorithms provided by this package.

## Core Concepts

### DFLPolicy

The [`DFLPolicy`](@ref) is the central abstraction that encapsulates a decision-focused learning policy. It combines:
- A **statistical model** (typically a neural network) that predicts parameters from input features
- A **combinatorial optimizer** (maximizer) that solves optimization problems using the predicted parameters

```julia
policy = DFLPolicy(
    Chain(Dense(input_dim => hidden_dim, relu), Dense(hidden_dim => output_dim)),
    my_optimizer
)
```

### Training Interface

All algorithms in this package follow a unified training interface with two main functions:

#### Core Training Method

```julia
history = train_policy!(algorithm, policy, training_data; epochs=100, metrics=(), maximizer_kwargs=get_info)
```

**Arguments:**
- `algorithm`: An algorithm instance (e.g., `PerturbedFenchelYoungLossImitation`, `DAgger`, `AnticipativeImitation`)
- `policy::DFLPolicy`: The policy to train (contains the model and maximizer)
- `training_data`: Either a dataset of `DataSample` objects or `Environment` (depends on algorithm)
- `epochs::Int`: Number of training epochs (default: 100)
- `metrics::Tuple`: Metrics to evaluate during training (default: empty)
- `maximizer_kwargs::Function`: Function that extracts keyword arguments for the maximizer from data samples (default: `get_info`)

**Returns:**
- `history::MVHistory`: Training history containing loss values and metric evaluations

#### Benchmark Convenience Wrapper

```julia
result = train_policy(algorithm, benchmark; dataset_size=30, split_ratio=(0.3, 0.3), epochs=100, metrics=())
```

This high-level function handles all setup from a benchmark and returns a trained policy along with training history.

**Arguments:**
- `algorithm`: An algorithm instance
- `benchmark::AbstractBenchmark`: A benchmark from DecisionFocusedLearningBenchmarks.jl
- `dataset_size::Int`: Number of instances to generate
- `split_ratio::Tuple`: Train/validation/test split ratios
- `epochs::Int`: Number of training epochs
- `metrics::Tuple`: Metrics to track during training

**Returns:**
- `(; policy, history)`: Named tuple with trained policy and training history

## Metrics

Metrics allow you to track additional quantities during training.

### Built-in Metrics

#### FYLLossMetric

Evaluates Fenchel-Young loss on a validation dataset.

```julia
val_metric = FYLLossMetric(validation_data, :validation_loss)
```

#### FunctionMetric

Custom metric defined by a function.

```julia
# Simple metric (no stored data)
epoch_metric = FunctionMetric(ctx -> ctx.epoch, :epoch)

# Metric with stored data
gap_metric = FunctionMetric(:validation_gap, validation_data) do ctx, data
    compute_gap(benchmark, data, ctx.policy.statistical_model, ctx.policy.maximizer)
end
```

### TrainingContext

Metrics receive a `TrainingContext` object containing:
- `policy::DFLPolicy`: The policy being trained
- `epoch::Int`: Current epoch number
- `maximizer_kwargs::Function`: Maximizer kwargs extractor
- `other_fields`: Algorithm-specific fields (e.g., `loss` for FYL)

Access policy components:
```julia
ctx.policy.statistical_model  # Neural network
ctx.policy.maximizer          # Combinatorial optimizer
```
