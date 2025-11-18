# Analysis of the New Callback System

**Date:** November 13, 2025  
**Analyzed Files:** `src/fyl_new.jl`, `src/callbacks.jl`, `src/dagger.jl`

## Executive Summary

The new callback-based training system represents a **step in the right direction** with cleaner architecture and better extensibility. However, it suffers from incomplete implementation, API inconsistencies, and missing essential features common in modern ML frameworks.

**Grade: B-**

---

## ✅ Strengths

### 1. Cleaner Architecture
- **Clear separation of concerns**: Callbacks are independent, reusable modules
- **Standard storage**: `MVHistory` is more conventional than nested NamedTuples
- **Simpler mental model**: Easier to understand than the old nested callback system

### 2. Better Extensibility
```julia
# Easy to add new metrics
callbacks = [
    Metric(:gap, (data, ctx) -> compute_gap(b, data, ctx.model, ctx.maximizer)),
    Metric(:custom, (data, ctx) -> my_custom_metric(ctx.model))
]
```
- Adding new metrics is straightforward with the `Metric` class
- `TrainingCallback` abstract type enables custom callback development
- Users can compose multiple callbacks without complex nested structures

### 3. Improved Error Handling
```julia
catch e
    @warn "Metric $(cb.name) failed at epoch $(context.epoch)" exception = (
        e, catch_backtrace()
    )
    return nothing
end
```
- Graceful degradation when metrics fail
- Training continues even if a callback encounters an error
- Clear warning messages

### 4. More Predictable Naming
- Automatic `train_`/`val_` prefixes based on `on` parameter
- Less cognitive overhead for users
- Consistent naming convention across metrics

---

## ❌ Critical Issues

### 1. API Inconsistency Between FYL and DAgger ⚠️ **BLOCKER**

**Problem:** The two main training functions use incompatible callback systems!

```julia
# fyl_new.jl uses Vector of TrainingCallback objects
fyl_train_model!(model, maximizer, train, val; 
                 callbacks::Vector{<:TrainingCallback}=TrainingCallback[])

# dagger.jl STILL uses the old NamedTuple system!
DAgger_train_model!(model, maximizer, ...; 
                    metrics_callbacks::NamedTuple=NamedTuple())
```

**Impact:**
- Confusing for users - which API should they learn?
- Breaks composability - can't reuse callbacks across algorithms
- Creates maintenance burden - two systems to maintain
- Suggests incomplete migration

**Fix Required:** Update `DAgger_train_model!` to use the new callback system immediately.

---

### 2. Context Missing Current Loss Values

**Problem:** Callbacks cannot access the current epoch's losses without recomputing them.

```julia
# Current implementation
context = (
    epoch=epoch,
    model=model,
    maximizer=maximizer,
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
)
```

**Why This Matters:**
- Metrics that depend on loss (e.g., loss ratios, relative improvements) must recompute
- Wasteful and inefficient
- Early stopping callbacks need loss values

**Should Be:**
```julia
context = (
    epoch=epoch,
    model=model,
    maximizer=maximizer,
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    train_loss=avg_train_loss,  # ADD
    val_loss=avg_val_loss,      # ADD
)
```

---

### 3. Hardcoded Hyperparameters

**Problem:** Critical training parameters cannot be customized.

```julia
# Hardcoded in function body
perturbed = PerturbedAdditive(maximizer; nb_samples=10, ε=0.1, threaded=true)
optimizer = Adam()
```

**What's Missing:**
- ❌ Cannot change perturbation strategy
- ❌ Cannot adjust number of samples
- ❌ Cannot tune epsilon value
- ❌ Cannot use different optimizers (AdamW, SGD, etc.)
- ❌ Cannot set learning rate
- ❌ Cannot disable threading

**Impact:**
- Users stuck with one configuration
- Cannot reproduce papers that use different settings
- Limits experimental flexibility

**Recommended Fix:**
```julia
function fyl_train_model!(
    model,
    maximizer,
    train_dataset,
    validation_dataset;
    epochs=100,
    optimizer=Adam(),
    nb_samples=10,
    ε=0.1,
    threaded=true,
    maximizer_kwargs=(sample -> (; instance=sample.info)),
    callbacks::Vector{<:TrainingCallback}=TrainingCallback[],
)
```

---

### 4. Inefficient and Inconsistent Loss Computation

**Problem:** Mixed approaches for computing losses.

Initial losses (list comprehension):
```julia
initial_val_loss = mean([
    loss(model(sample.x), sample.y; maximizer_kwargs(sample)...) for
    sample in validation_dataset
])
```

Training loop (accumulation):
```julia
epoch_val_loss = 0.0
for sample in validation_dataset
    epoch_val_loss += loss(model(x), y; maximizer_kwargs(sample)...)
end
avg_val_loss = epoch_val_loss / length(validation_dataset)
```

**Issues:**
- Inconsistency is confusing
- List comprehension allocates unnecessary array
- Memory inefficient for large datasets

**Fix:** Use accumulation pattern consistently.

---

### 5. No Mini-Batch Support

**Problem:** Only supports online learning (one sample at a time).

```julia
for sample in train_dataset
    val, grads = Flux.withgradient(model) do m
        loss(m(x), y; maximizer_kwargs(sample)...)
    end
    Flux.update!(opt_state, model, grads[1])  # Update after EVERY sample
end
```

**Why This is Bad:**
- Slow convergence
- Noisy gradients
- Not standard practice in modern ML
- Cannot leverage GPU batching efficiently
- Inefficient for large datasets

**Standard Approach:**
```julia
for batch in DataLoader(train_dataset; batchsize=32, shuffle=true)
    # Accumulate gradients over batch
    # Single update per batch
end
```

---

### 6. Awkward Metric Function Signature

**Current Design:**
```julia
Metric(:gap, (data, ctx) -> compute_gap(benchmark, data, ctx.model, ctx.maximizer))
```

**Issues:**
1. **Confusing `data` parameter**: Its meaning changes based on `on` value
   - `on=:train` → `data = train_dataset`
   - `on=:validation` → `data = validation_dataset`  
   - `on=:both` → function called twice with different data
   - `on=custom_data` → `data = custom_data`

2. **Repetitive code**: Must extract `model`, `maximizer` from context every time

3. **No type safety**: Function signature not enforced

4. **Not discoverable**: Users must read docs to understand signature

**Better Alternative:**
```julia
# Option 1: Pass full context, let metric extract what it needs
Metric(:gap, ctx -> compute_gap(benchmark, ctx.validation_dataset, ctx.model, ctx.maximizer))

# Option 2: Declare dependencies explicitly
Metric(:gap, compute_gap; 
       on=:validation, 
       needs=[:model, :maximizer],
       args=(benchmark,))
```

---

### 7. Missing Standard ML Features

The implementation lacks features that are **table stakes** in modern ML frameworks:

#### Early Stopping
```julia
# Users cannot do this:
callbacks = [
    EarlyStopping(patience=10, metric=:val_loss, mode=:min)
]
```

#### Model Checkpointing
```julia
# Users cannot do this:
callbacks = [
    ModelCheckpoint(path="best_model.bson", metric=:val_loss, mode=:min)
]
```

#### Learning Rate Scheduling
```julia
# No support for:
LearningRateScheduler(schedule = epoch -> 0.001 * 0.95^epoch)
ReduceLROnPlateau(patience=5, factor=0.5)
```

#### Other Missing Features
- ❌ Gradient clipping (risk of exploding gradients)
- ❌ Logging frequency control (always every epoch)
- ❌ Warmup epochs
- ❌ Progress bar customization
- ❌ TensorBoard logging
- ❌ Validation frequency control (always every epoch)

---

### 8. Return Value Convention

**Problem:** Non-obvious return order and type.

```julia
function fyl_train_model(...)
    model = deepcopy(initial_model)
    return fyl_train_model!(...), model
end
```

Returns `(history, model)` as a tuple.

**Issues:**
- Order not obvious from function name
- Positional unpacking error-prone: `h, m = fyl_train_model(...)` vs `m, h = ...`?
- Inconsistent with other Julia ML libraries

**Better Options:**

**Option 1: Named Tuple**
```julia
return (model=model, history=history)
# Usage: result.model, result.history
```

**Option 2: Follow Flux Convention**
```julia
return model, history  # Model first (most important)
```

**Option 3: Struct**
```julia
struct TrainingResult
    model
    history
    best_epoch::Int
    best_val_loss::Float64
end
```

---

### 9. Forced Plotting Side Effect

**Problem:** Always prints a plot to stdout.

```julia
# At end of function
println(lineplot(a, b; xlabel="Epoch", ylabel="Validation Loss"))
```

**Issues:**
- ❌ Cannot disable
- ❌ Clutters output in batch jobs
- ❌ Unnecessary in automated experiments
- ❌ Not helpful in notebooks (users want actual plots)
- ❌ Violates principle of least surprise

**Fix:** Make optional with `verbose` parameter.

```julia
function fyl_train_model!(
    # ... existing args ...
    verbose::Bool=true,
)
    # ... training code ...
    
    if verbose
        a, b = get(history, :validation_loss)
        println(lineplot(a, b; xlabel="Epoch", ylabel="Validation Loss"))
    end
    
    return history
end
```

---

### 10. No Documentation

**Problem:** Function lacks docstring.

```julia
function fyl_train_model!(  # ← No docstring!
    model,
    maximizer,
    train_dataset::AbstractArray{<:DataSample},
    # ...
```

**What's Missing:**
- Parameter descriptions
- Return value documentation
- Usage examples
- Callback system explanation
- Link to callback documentation

**Example of What's Needed:**
````julia
"""
    fyl_train_model!(model, maximizer, train_dataset, validation_dataset; kwargs...)

Train a model using Fenchel-Young Loss with decision-focused learning.

# Arguments
- `model`: Neural network model to train (will be modified in-place)
- `maximizer`: Optimization solver for computing decisions
- `train_dataset::AbstractArray{<:DataSample}`: Training data
- `validation_dataset`: Validation data for evaluation

# Keywords
- `epochs::Int=100`: Number of training epochs
- `maximizer_kwargs::Function`: Function mapping sample to maximizer kwargs
- `callbacks::Vector{<:TrainingCallback}`: Callbacks for metrics/logging

# Returns
- `MVHistory`: Training history containing losses and metrics

# Examples
```julia
# Basic usage
history = fyl_train_model!(model, maximizer, train_data, val_data; epochs=50)

# With custom metrics
callbacks = [
    Metric(:gap, (data, ctx) -> compute_gap(benchmark, data, ctx.model, ctx.maximizer))
]
history = fyl_train_model!(model, maximizer, train_data, val_data; 
                           epochs=100, callbacks=callbacks)

# Access results
val_losses = get(history, :validation_loss)
gap_values = get(history, :val_gap)
```

See also: [`TrainingCallback`](@ref), [`Metric`](@ref), [`fyl_train_model`](@ref)
"""
````

---

## 🔶 Design Concerns

### 1. Callback vs Metric Naming Confusion

**Problem:** `Metric` is a callback, but the naming suggests they're different concepts.

```julia
abstract type TrainingCallback end
struct Metric <: TrainingCallback  # Metric is-a Callback
```

**Confusion:**
- Are metrics different from callbacks?
- Can callbacks do more than just metrics?
- Why inherit from `TrainingCallback` if it's just a `Metric`?

**Clarity Improvement:**
```julia
# Option 1: Keep as is but document clearly
# Option 2: Rename to MetricCallback
struct MetricCallback <: TrainingCallback

# Option 3: Make distinction explicit
abstract type TrainingCallback end
abstract type MetricCallback <: TrainingCallback end
struct SimpleMetric <: MetricCallback
struct EarlyStopping <: TrainingCallback  # Not a metric
```

---

### 2. Direct History Manipulation

**Problem:** Both the trainer and callbacks push to the same history object.

```julia
# In trainer
push!(history, :training_loss, epoch, avg_train_loss)

# In callback
function run_callbacks!(history, callbacks, context)
    for callback in callbacks
        metrics = on_epoch_end(callback, context)
        if !isnothing(metrics)
            for (name, value) in pairs(metrics)
                push!(history, name, context.epoch, value)  # Same object!
            end
        end
    end
end
```

**Risks:**
- Naming conflicts (callback could override `:training_loss`)
- No validation of metric names
- Hard to track what came from where
- Callbacks could corrupt history

**Better Separation:**
```julia
# Callbacks return metrics, trainer handles history
function run_callbacks!(history, callbacks, context)
    for callback in callbacks
        metrics = on_epoch_end(callback, context)
        if !isnothing(metrics)
            # Validate no conflicts with reserved names
            if any(name in [:training_loss, :validation_loss] for name in keys(metrics))
                error("Callback metric name conflicts with reserved names")
            end
            # Store safely
            for (name, value) in pairs(metrics)
                push!(history, name, context.epoch, value)
            end
        end
    end
end
```

---

### 3. No Test Dataset Support

**Problem:** Only `train_dataset` and `validation_dataset` are in the API.

```julia
function fyl_train_model!(
    model,
    maximizer,
    train_dataset::AbstractArray{<:DataSample},
    validation_dataset;  # Only train and val
    # ...
```

**Workaround is Clunky:**
```julia
# User must do this:
test_dataset = ...
callbacks = [
    Metric(:test_gap, (data, ctx) -> compute_gap(b, data, ctx.model, ctx.maximizer);
           on=test_dataset)  # Pass test set directly
]
```

**Better API:**
```julia
function fyl_train_model!(
    model,
    maximizer,
    train_dataset,
    validation_dataset;
    test_dataset=nothing,  # Optional test set
    # ...
)
```

Then metrics can use `on=:test`.

---

## 💡 Recommendations

### Immediate Priority (Fix Before Release)

1. **✅ Update DAgger to use new callback system**
   - Critical for API consistency
   - Blocks adoption of new system
   - Update all example scripts

2. **✅ Add loss values to context**
   ```julia
   context = merge(context, (train_loss=avg_train_loss, val_loss=avg_val_loss,))
   ```

3. **✅ Make hyperparameters configurable**
   - Add optimizer parameter
   - Add perturbation parameters (nb_samples, ε)
   - Add learning rate

### High Priority (Before v1.0)

4. **Add mini-batch support**
   ```julia
   function fyl_train_model!(
       # ...
       batch_size::Int=1,  # Default to online learning for compatibility
   )
   ```

5. **Implement essential callbacks**
   - `EarlyStopping(patience, metric, mode)`
   - `ModelCheckpoint(path, metric, mode)`
   - `LearningRateScheduler(schedule)`

6. **Make plotting optional**
   ```julia
   verbose::Bool=true,
   plot_loss::Bool=verbose,
   ```

7. **Add comprehensive docstrings**
   - Function-level docs
   - Parameter descriptions
   - Usage examples

### Medium Priority (Quality of Life)

8. **Improve error messages**
   ```julia
   try
       value = cb.metric_fn(context.validation_dataset, context)
   catch e
       @error "Metric '$(cb.name)' failed at epoch $(context.epoch)" exception=(e, catch_backtrace())
       @info "Context available: $(keys(context))"
       @info "Callback type: $(typeof(cb))"
       rethrow()  # Or return nothing, depending on desired behavior
   end
   ```

9. **Add metric name validation**
   ```julia
   reserved_names = [:training_loss, :validation_loss, :epoch]
   metric_names = get_metric_names(callbacks)
   conflicts = intersect(metric_names, reserved_names)
   if !isempty(conflicts)
       error("Callback metric names conflict with reserved names: $conflicts")
   end
   ```

10. **Return named tuple instead of tuple**
    ```julia
    return (model=model, history=history)
    ```

### Low Priority (Nice to Have)

11. **Add test dataset support**
    ```julia
    test_dataset=nothing
    ```

12. **Add progress bar customization**
    ```julia
    show_progress::Bool=true,
    progress_prefix::String="Training",
    ```

13. **Add TensorBoard logging callback**
    ```julia
    TensorBoardLogger(logdir="runs/experiment_1")
    ```

14. **Consider a TrainingConfig struct**
    ```julia
    struct TrainingConfig
        epochs::Int
        optimizer
        batch_size::Int
        nb_samples::Int
        ε::Float64
        # ... etc
    end
    ```

---

## 📊 Comparison: Old vs New System

| Aspect | Old System (`fyl.jl`) | New System (`fyl_new.jl`) |
|--------|----------------------|--------------------------|
| **Callback API** | Nested NamedTuples | `TrainingCallback` objects |
| **Storage** | Nested NamedTuples | `MVHistory` |
| **Extensibility** | ⚠️ Awkward | ✅ Good |
| **Error Handling** | ❌ No try-catch | ✅ Graceful degradation |
| **Naming** | Manual | ✅ Automatic prefixes |
| **Type Safety** | ❌ Runtime checks | ✅ Abstract types |
| **Discoverability** | ❌ Poor | ⚠️ Better but needs docs |
| **DAgger Support** | ✅ Yes | ❌ Not yet updated |
| **Documentation** | ❌ Minimal | ❌ None yet |
| **Hyperparameters** | ❌ Hardcoded | ❌ Still hardcoded |
| **Batching** | ❌ No | ❌ No |

**Verdict:** New system is architecturally superior but incompletely implemented.

---

## 🎯 Overall Assessment

### What Works Well
- ✅ Callback abstraction is clean and extensible
- ✅ `MVHistory` is a solid choice for metric storage
- ✅ Error handling in callbacks prevents total failure
- ✅ Automatic metric naming reduces boilerplate

### Critical Blockers
- 🚫 **DAgger not updated** - API split is confusing
- 🚫 **No hyperparameter configuration** - Limits experimentation
- 🚫 **Missing essential callbacks** - Early stopping, checkpointing

### Missing Features
- ⚠️ No mini-batch training
- ⚠️ Context missing loss values
- ⚠️ No documentation
- ⚠️ Forced plotting output

### Verdict

The new callback system shows **promise** but is **not production-ready**. The biggest issue is the incomplete migration - DAgger still uses the old system, creating a confusing API split.

**Recommended Action Plan:**
1. Update DAgger immediately
2. Add essential hyperparameters
3. Include loss in context
4. Add basic documentation
5. Then consider it ready for testing

After these changes, the system would merit a **B+** grade and be ready for wider use.

---

## 📝 Code Examples

### Current Usage (New System)
```julia
using DecisionFocusedLearningAlgorithms

callbacks = [
    Metric(:gap, (data, ctx) -> compute_gap(benchmark, data, ctx.model, ctx.maximizer))
]

history = fyl_train_model!(
    model,
    maximizer,
    train_dataset,
    validation_dataset;
    epochs=100,
    callbacks=callbacks
)

# Access results
val_loss = get(history, :validation_loss)
gap = get(history, :val_gap)
```

### Proposed Improved Usage
```julia
using DecisionFocusedLearningAlgorithms

callbacks = [
    Metric(:gap, compute_gap_metric),
    EarlyStopping(patience=10, metric=:val_loss),
    ModelCheckpoint("best_model.bson", metric=:val_gap, mode=:min),
]

result = fyl_train_model!(
    model,
    maximizer,
    train_dataset,
    validation_dataset;
    test_dataset=test_dataset,
    epochs=100,
    batch_size=32,
    optimizer=Adam(0.001),
    callbacks=callbacks,
    verbose=true
)

# Access with named fields
best_model = result.best_model
final_model = result.model
history = result.history
```

---

## 🔍 Additional Notes

### Performance Considerations
- Current online learning (batch_size=1) is inefficient
- Loss computation could be parallelized
- Consider GPU support for batch operations

### Compatibility
- Breaking change from old system
- Need migration guide for users
- Consider deprecation warnings

### Testing
- No unit tests for callback system visible
- Need tests for:
  - Callback error handling
  - Metric name conflicts
  - History storage correctness
  - DAgger integration

### Documentation Needs
- Tutorial on writing custom callbacks
- Examples of common use cases
- API reference
- Migration guide from old system

---

**End of Analysis**
