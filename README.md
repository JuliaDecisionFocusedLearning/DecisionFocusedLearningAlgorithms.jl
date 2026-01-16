# DecisionFocusedLearningAlgorithms

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaDecisionFocusedLearning.github.io/DecisionFocusedLearningAlgorithms.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaDecisionFocusedLearning.github.io/DecisionFocusedLearningAlgorithms.jl/dev/)
[![Build Status](https://github.com/JuliaDecisionFocusedLearning/DecisionFocusedLearningAlgorithms.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaDecisionFocusedLearning/DecisionFocusedLearningAlgorithms.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaDecisionFocusedLearning/DecisionFocusedLearningAlgorithms.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaDecisionFocusedLearning/DecisionFocusedLearningAlgorithms.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

> [!WARNING]  
>  This package is currently under active development. The API may change in future releases.
>  Please refer to the [documentation](https://JuliaDecisionFocusedLearning.github.io/DecisionFocusedLearningAlgorithms.jl/stable/) for the latest updates.

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

See the [documentation](https://JuliaDecisionFocusedLearning.github.io/DecisionFocusedLearningAlgorithms.jl/stable/) for more details.

## Available Algorithms

- **Perturbed Fenchel-Young Loss Imitation**: Differentiable imitation learning with perturbed optimization
- **AnticipativeImitation**: Imitation of anticipative solutions for dynamic problems
- **DAgger**: DAgger algorithm for dynamic problems