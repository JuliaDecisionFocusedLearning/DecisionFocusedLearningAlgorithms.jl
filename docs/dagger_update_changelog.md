# DAgger Update to New Callback System - Changelog

**Date:** November 13, 2025  
**Updated Files:**
- `src/dagger.jl`
- `scripts/main.jl`
- `src/utils/metrics.jl` (marked deprecated functions)

---

## Summary

Updated `DAgger_train_model!` and `DAgger_train_model` to use the new callback system (Vector of `TrainingCallback` objects) instead of the old nested NamedTuple system. This achieves API consistency across all training functions.

---

## Changes Made

### 1. `src/dagger.jl` - `DAgger_train_model!` Function

#### Before (Old System)
```julia
function DAgger_train_model!(
    model,
    maximizer,
    train_environments,
    validation_environments,
    anticipative_policy;
    iterations=5,
    fyl_epochs=3,
    metrics_callbacks::NamedTuple=NamedTuple(),  # ❌ Old system
)
    # ...
    all_metrics = []
    for iter in 1:iterations
        metrics = fyl_train_model!(
            model,
            maximizer,
            dataset,
            val_dataset;
            epochs=fyl_epochs,
            metrics_callbacks=metrics_callbacks,  # ❌ Old system
        )
        push!(all_metrics, metrics)
        # ...
    end
    return _flatten_dagger_metrics(all_metrics)  # ❌ Old system
end
```

#### After (New System)
```julia
function DAgger_train_model!(
    model,
    maximizer,
    train_environments,
    validation_environments,
    anticipative_policy;
    iterations=5,
    fyl_epochs=3,
    callbacks::Vector{<:TrainingCallback}=TrainingCallback[],  # ✅ New system
    maximizer_kwargs=(sample -> (; instance=sample.info)),
)
    # ...
    combined_history = MVHistory()  # ✅ Combined history
    global_epoch = 0
    
    for iter in 1:iterations
        println("DAgger iteration $iter/$iterations (α=$(round(α, digits=3)))")
        
        iter_history = fyl_train_model!(
            model,
            maximizer,
            dataset,
            val_dataset;
            epochs=fyl_epochs,
            callbacks=callbacks,  # ✅ New system
            maximizer_kwargs=maximizer_kwargs,
        )
        
        # Merge iteration history into combined history
        # Skip epoch 0 for iterations > 1 to avoid duplication
        for key in keys(iter_history)
            epochs, values = get(iter_history, key)
            start_idx = (iter == 1) ? 1 : 2
            for i in start_idx:length(epochs)
                push!(combined_history, key, global_epoch + epochs[i], values[i])
            end
        end
        global_epoch += fyl_epochs
        # ...
    end
    
    return combined_history  # ✅ Returns MVHistory
end
```

**Key Improvements:**
- ✅ Uses new callback system (`callbacks::Vector{<:TrainingCallback}`)
- ✅ Returns `MVHistory` instead of flattened NamedTuple
- ✅ Properly tracks global epoch numbers across DAgger iterations
- ✅ Skips duplicate epoch 0 for iterations > 1
- ✅ Improved progress messages showing α decay
- ✅ Added `maximizer_kwargs` parameter for consistency with FYL

---

### 2. `src/dagger.jl` - `DAgger_train_model` Function

#### Before
```julia
function DAgger_train_model(b::AbstractStochasticBenchmark{true}; kwargs...)
    # ...
    return DAgger_train_model!(...)  # Returned history directly
end
```

#### After
```julia
function DAgger_train_model(b::AbstractStochasticBenchmark{true}; kwargs...)
    # ...
    history = DAgger_train_model!(...)
    return history, model  # ✅ Returns (history, model) tuple like fyl_train_model
end
```

**Key Improvements:**
- ✅ Consistent return signature with `fyl_train_model`
- ✅ Returns both history and trained model

---

### 3. `scripts/main.jl` - Example Script Update

#### Before
```julia
metrics_callbacks = (;
    obj=(model, maximizer, epoch) ->
        mean(evaluate_policy!(policy, test_environments, 1)[1])
)

fyl_loss = fyl_train_model!(
    fyl_model, maximizer, train_dataset, val_dataset; 
    epochs=100, metrics_callbacks
)

dagger_loss = DAgger_train_model!(
    dagger_model, maximizer, train_environments, validation_environments,
    anticipative_policy; iterations=10, fyl_epochs=10, metrics_callbacks
)

# Plotting with old API
plot(0:100, [fyl_loss.obj[1:end], dagger_loss.obj[1:end]]; ...)
```

#### After
```julia
callbacks = [
    Metric(:obj, (data, ctx) -> 
        mean(evaluate_policy!(policy, test_environments, 1)[1])
    )
]

fyl_history = fyl_train_model!(
    fyl_model, maximizer, train_dataset, val_dataset; 
    epochs=100, callbacks
)

dagger_history = DAgger_train_model!(
    dagger_model, maximizer, train_environments, validation_environments,
    anticipative_policy; iterations=10, fyl_epochs=10, callbacks=callbacks
)

# Plotting with new API
fyl_epochs, fyl_obj_values = get(fyl_history, :val_obj)
dagger_epochs, dagger_obj_values = get(dagger_history, :val_obj)
plot([fyl_epochs, dagger_epochs], [fyl_obj_values, dagger_obj_values]; ...)
```

**Key Improvements:**
- ✅ Uses new `Metric` callback instead of NamedTuple
- ✅ Uses `MVHistory.get()` API to extract metrics
- ✅ More explicit and type-safe
- ✅ Same callback definition for both FYL and DAgger

---

### 4. `src/utils/metrics.jl` - Marked Old Functions as Deprecated

Added deprecation notice at the top:

```julia
# NOTE: The functions below are deprecated and only kept for backward compatibility
# with the old nested NamedTuple callback system (used in fyl.jl, not fyl_new.jl).
# They can be removed once fyl.jl is fully removed from the codebase.

# Helper functions for nested callbacks (DEPRECATED - for old system only)
```

The following functions are now deprecated:
- `_flatten_callbacks`
- `_unflatten_metrics`
- `_initialize_nested_metrics`
- `_call_nested_callbacks`
- `_push_nested_metrics!`
- `_flatten_dagger_metrics`

These can be safely removed once `fyl.jl` is deleted.

---

## Migration Guide

### For Users Upgrading Existing Code

#### Old API (DAgger with NamedTuple callbacks)
```julia
metrics_callbacks = (;
    gap = (m, max, e) -> compute_gap(benchmark, val_data, m, max),
    obj = (m, max, e) -> mean(evaluate_policy!(policy, test_envs, 1)[1])
)

history = DAgger_train_model!(
    model, maximizer, train_envs, val_envs, anticipative_policy;
    iterations=10, fyl_epochs=10, metrics_callbacks
)

# Access metrics
gap_values = history.gap
obj_values = history.obj
```

#### New API (DAgger with TrainingCallback)
```julia
callbacks = [
    Metric(:gap, (data, ctx) -> 
        compute_gap(benchmark, data, ctx.model, ctx.maximizer)),
    Metric(:obj, (data, ctx) -> 
        mean(evaluate_policy!(policy, test_envs, 1)[1]))
]

history = DAgger_train_model!(
    model, maximizer, train_envs, val_envs, anticipative_policy;
    iterations=10, fyl_epochs=10, callbacks=callbacks
)

# Access metrics
epochs, gap_values = get(history, :val_gap)
epochs, obj_values = get(history, :val_obj)
```

**Key Differences:**
1. ❌ `metrics_callbacks::NamedTuple` → ✅ `callbacks::Vector{<:TrainingCallback}`
2. ❌ Function signature `(model, maximizer, epoch)` → ✅ `(data, context)`
3. ❌ Direct field access `history.gap` → ✅ `get(history, :val_gap)`
4. ❌ Returns flattened NamedTuple → ✅ Returns MVHistory object
5. ✅ Automatic `val_` prefix for metrics using validation data

---

## Benefits of the Update

### 1. **API Consistency**
- ✅ FYL and DAgger now use the same callback system
- ✅ Users learn one API, use everywhere
- ✅ Callbacks are reusable across different training methods

### 2. **Better Type Safety**
- ✅ `TrainingCallback` abstract type provides structure
- ✅ Compile-time checking of callback types
- ✅ Better IDE support and autocomplete

### 3. **Improved Extensibility**
- ✅ Easy to add new callback types (early stopping, checkpointing, etc.)
- ✅ Callbacks can be packaged and shared
- ✅ Clear interface for custom callbacks

### 4. **Standard Library Integration**
- ✅ `MVHistory` is a well-tested package
- ✅ Better plotting support
- ✅ Standard API familiar to Julia ML users

### 5. **Better Error Handling**
- ✅ Graceful degradation when callbacks fail
- ✅ Clear error messages
- ✅ Training continues even if a metric fails

---

## Validation

### Tests Passed
- ✅ No syntax errors in updated files
- ✅ No import/export errors
- ✅ Code passes Julia linter

### Manual Testing Required
- ⚠️ Run `scripts/main.jl` to verify end-to-end functionality
- ⚠️ Test with custom callbacks
- ⚠️ Verify metric values are correct
- ⚠️ Check plot generation

### Recommended Test Script
```julia
using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks

b = DynamicVehicleSchedulingBenchmark(; two_dimensional_features=false)

# Test with callbacks
callbacks = [
    Metric(:test_metric, (data, ctx) -> ctx.epoch * 1.5)
]

history, model = DAgger_train_model(b; 
    iterations=3, 
    fyl_epochs=2, 
    callbacks=callbacks
)

# Verify structure
@assert history isa MVHistory
@assert haskey(history, :training_loss)
@assert haskey(history, :validation_loss)
@assert haskey(history, :val_test_metric)

# Verify epoch continuity
epochs, _ = get(history, :training_loss)
@assert epochs == 0:6  # 3 iterations × 2 epochs + epoch 0

println("✅ All tests passed!")
```

---

## Next Steps

### Immediate
1. ✅ **Done:** Update DAgger to new callback system
2. ⚠️ **TODO:** Run test script to verify functionality
3. ⚠️ **TODO:** Update any other example scripts using DAgger

### Short Term
4. ⚠️ **TODO:** Add unit tests for DAgger callback integration
5. ⚠️ **TODO:** Update documentation/tutorials
6. ⚠️ **TODO:** Consider removing `fyl.jl` entirely (if not needed)

### Long Term
7. ⚠️ **TODO:** Remove deprecated functions from `utils/metrics.jl`
8. ⚠️ **TODO:** Add more callback types (EarlyStopping, ModelCheckpoint)
9. ⚠️ **TODO:** Write migration guide in docs

---

## Breaking Changes

### ⚠️ This is a Breaking Change

Code using the old DAgger API will need to be updated:

```julia
# ❌ This will no longer work:
metrics_callbacks = (gap = (m, max, e) -> ...,)
DAgger_train_model!(...; metrics_callbacks=metrics_callbacks)

# ✅ Use this instead:
callbacks = [Metric(:gap, (data, ctx) -> ...)]
DAgger_train_model!(...; callbacks=callbacks)
```

### Deprecation Path

1. **Current:** Old API removed, new API required
2. **Alternative:** Could add deprecation warning if needed:
   ```julia
   function DAgger_train_model!(...; metrics_callbacks=nothing, callbacks=TrainingCallback[], ...)
       if !isnothing(metrics_callbacks)
           @warn "metrics_callbacks is deprecated. Use callbacks= instead." maxlog=1
           # Convert old to new format (if feasible)
       end
       # ...
   end
   ```

---

## Files Changed

1. **`src/dagger.jl`** - Main DAgger implementation
   - Updated `DAgger_train_model!` signature and implementation
   - Updated `DAgger_train_model` return value
   - ~60 lines changed

2. **`scripts/main.jl`** - Example script
   - Updated to use new callback API
   - Updated plotting code for MVHistory
   - ~40 lines changed

3. **`src/utils/metrics.jl`** - Helper functions
   - Added deprecation notice
   - ~5 lines changed

**Total:** ~105 lines changed across 3 files

---

**End of Changelog**
