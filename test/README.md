# Test Suite Documentation

## Overview

The test suite for DecisionFocusedLearningAlgorithms.jl validates the training functions and callback system.

## Test Files

### `runtests.jl`
Main test runner that includes:
- Code quality checks (Aqua.jl)
- Linting (JET.jl)
- Code formatting (JuliaFormatter.jl)
- Training and callback tests

### `training_tests.jl`
Comprehensive tests for the training system covering:

## Test Coverage

### 1. FYL Training Tests

#### `FYL Training - Basic`
- ✅ Basic training runs without error
- ✅ Returns MVHistory object
- ✅ Tracks training and validation losses
- ✅ Proper epoch indexing (0-based)
- ✅ Loss values are Float64

#### `FYL Training - With Callbacks`
- ✅ Callbacks are executed
- ✅ Custom metrics are recorded in history
- ✅ Multiple callbacks work together
- ✅ Epoch tracking works correctly

#### `FYL Training - Callback on=:both`
- ✅ Train and validation metrics both computed
- ✅ Correct naming with train_/val_ prefixes
- ✅ Both datasets processed

#### `FYL Training - Context Fields`
- ✅ All core context fields present
- ✅ Correct types for context fields
- ✅ Context structure is consistent
- ✅ Required fields: epoch, model, maximizer, datasets, losses

#### `FYL Training - fyl_train_model (non-mutating)`
- ✅ Returns both history and model
- ✅ Original model not mutated
- ✅ Trained model is a copy

#### `Callback Error Handling`
- ✅ Training continues when callback fails
- ✅ Failed metrics not added to history
- ✅ Warning issued for failed callbacks

#### `Multiple Callbacks`
- ✅ Multiple callbacks run successfully
- ✅ All metrics tracked independently
- ✅ Different callback types (dataset-based, context-only)

### 2. DAgger Training Tests

#### `DAgger - Basic Training`
- ✅ Training runs without error
- ✅ Returns MVHistory
- ✅ Tracks losses across iterations
- ✅ Epoch numbers increment correctly across DAgger iterations

#### `DAgger - With Callbacks`
- ✅ Callbacks work with DAgger
- ✅ Metrics tracked across iterations
- ✅ Epoch continuity maintained

#### `DAgger - Convenience Function`
- ✅ Benchmark-based function works
- ✅ Returns history and model
- ✅ Creates datasets and environments automatically

### 3. Callback System Tests

#### `Metric Construction`
- ✅ Default parameters (on=:validation)
- ✅ Custom 'on' parameter
- ✅ Different 'on' modes (:train, :both, :none)

#### `on_epoch_end Interface`
- ✅ Returns NamedTuple of metrics
- ✅ Correct metric values computed
- ✅ Context passed correctly

#### `get_metric_names`
- ✅ Extracts correct metric names
- ✅ Handles train_/val_ prefixes
- ✅ Works with different 'on' modes

#### `run_callbacks!`
- ✅ Executes all callbacks
- ✅ Stores metrics in history
- ✅ Correct epoch association

### 4. Integration Tests

#### `Portable Metrics Across Algorithms`
- ✅ Same callback works with FYL and DAgger
- ✅ Core context fields are consistent
- ✅ Portable metric definition

#### `Loss Values in Context`
- ✅ train_loss present in context
- ✅ val_loss present in context
- ✅ Both are positive Float64 values
- ✅ Can be used to compute derived metrics

## Running Tests

### Run All Tests
```bash
julia --project -e 'using Pkg; Pkg.test()'
```

### Run Specific Test File
```julia
using Pkg
Pkg.activate(".")
include("test/training_tests.jl")
```

### Run Tests in REPL
```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.test()
```

## Test Benchmarks Used

- **ArgmaxBenchmark**: Fast, simple benchmark for quick tests
- **DynamicVehicleSchedulingBenchmark**: More complex, tests sequential decision making

Small dataset sizes (10-30 samples) are used for speed while maintaining test coverage.

## What's Tested

### Core Functionality
- ✅ Training loop execution
- ✅ Gradient computation and model updates
- ✅ Loss computation on train/val sets
- ✅ Callback execution at correct times
- ✅ History storage and retrieval

### Callback System
- ✅ Metric computation with different 'on' modes
- ✅ Context structure and field availability
- ✅ Error handling and graceful degradation
- ✅ Multiple callback interaction
- ✅ Portable callback definitions

### API Consistency
- ✅ FYL and DAgger use same callback interface
- ✅ Context fields are consistent across algorithms
- ✅ Return types are correct
- ✅ Non-mutating variants work correctly

### Edge Cases
- ✅ Failing callbacks don't crash training
- ✅ Empty callback list works
- ✅ Epoch 0 (pre-training) handled correctly
- ✅ Single epoch training works

## Expected Test Duration

- **Code quality tests**: ~10-20 seconds
- **Training tests**: ~30-60 seconds
- **Total**: ~1-2 minutes

Tests are designed to be fast while providing comprehensive coverage.

## Common Issues

### Slow Tests
If tests are slow, reduce dataset sizes in `training_tests.jl`:
- `generate_dataset(benchmark, 10)` instead of 30
- Fewer epochs (2-3 instead of 5)
- Fewer DAgger iterations

### Missing Dependencies
Ensure all dependencies are installed:
```julia
using Pkg
Pkg.instantiate()
```

### GPU-Related Issues
Tests run on CPU. If GPU issues occur, set:
```julia
ENV["JULIA_CUDA_USE_BINARYBUILDER"] = "false"
```

## Adding New Tests

When adding new features, add tests to `training_tests.jl`:

1. **Add test group**: `@testset "Feature Name" begin ... end`
2. **Test basic functionality**: Does it run without error?
3. **Test correctness**: Are results correct?
4. **Test edge cases**: What happens with unusual inputs?
5. **Test integration**: Does it work with existing features?

## Continuous Integration

Tests run automatically on:
- Push to main branch
- Pull requests
- Scheduled daily runs

See `.github/workflows/CI.yml` for CI configuration.
