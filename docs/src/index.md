# DecisionFocusedLearningAlgorithms

Documentation for [DecisionFocusedLearningAlgorithms](https://github.com/JuliaDecisionFocusedLearning/DecisionFocusedLearningAlgorithms.jl).

## Overview

This package provides a unified interface for training decision-focused learning algorithms that combine machine learning with combinatorial optimization. It implements several state-of-the-art algorithms for learning to predict parameters of optimization problems.

### Key Features

- **Unified Interface**: Consistent API across all algorithms via `train_policy!`
- **Policy-Centric Design**: `DFLPolicy` encapsulates statistical models and optimizers
- **Flexible Metrics**: Track custom metrics during training
- **Benchmark Integration**: Seamless integration with DecisionFocusedLearningBenchmarks.jl

### Quick Start

```julia
using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks

# Create a policy
benchmark = ArgmaxBenchmark()
model = generate_statistical_model(benchmark)
maximizer = generate_maximizer(benchmark)
policy = DFLPolicy(model, maximizer)

# Train with FYL algorithm
algorithm = PerturbedFenchelYoungLossImitation()
result = train_policy(algorithm, benchmark; epochs=50)
```

See the [Interface Guide](interface.md) and [Tutorials](tutorials/tutorial.md) for more details.

## Available Algorithms

- **Perturbed Fenchel-Young Loss Imitation**: Differentiable imitation learning with perturbed optimization
- **AnticipativeImitation**: Imitation of anticipative solutions for dynamic problems
- **DAgger**: DAgger algorithm for dynamic problems
