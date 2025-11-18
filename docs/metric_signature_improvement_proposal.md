# Metric Function Signature Improvement Proposal

**Date:** November 13, 2025  
**Status:** Proposal / Discussion Document  
**Related:** Issue #6 from callback_system_analysis.md

---

## Problem Statement

The current `Metric` callback has an awkward function signature that is:
1. **Confusing**: The `data` parameter's meaning changes based on the `on` value
2. **Verbose**: Users must manually extract common items from context every time
3. **Error-prone**: No type checking on the function signature
4. **Not discoverable**: Users must read documentation to understand `(data, ctx)` signature

### Current API

```julia
# Current implementation
Metric(:gap, (data, ctx) -> compute_gap(benchmark, data, ctx.model, ctx.maximizer))
```

**Problems:**
- What is `data`? Is it train, validation, test, or something else?
- Must always extract `model` and `maximizer` from context
- Function signature not enforced - could accidentally break
- Not clear which parameters are available in context

---

## Proposed Solutions

I propose **three alternative approaches** (not mutually exclusive):

### Option 1: Context-Only Signature (Simplest)
### Option 2: Declarative Dependencies (Most Flexible)
### Option 3: Multiple Dispatch (Most Julian)

Let me detail each option:

---

## Option 1: Context-Only Signature

### Concept
Remove the confusing `data` parameter entirely. Users get full context and extract what they need.

### Implementation

```julia
struct Metric <: TrainingCallback
    name::Symbol
    metric_fn::Function  # Signature: (context) -> value
    on::Symbol  # :train, :validation, :both, :none
    
    function Metric(name::Symbol, metric_fn; on=:validation)
        new(name, metric_fn, on)
    end
end

function on_epoch_end(cb::Metric, context)
    try
        if cb.on == :train
            value = cb.metric_fn(context)
            return (Symbol("train_$(cb.name)") => value,)
            
        elseif cb.on == :validation
            value = cb.metric_fn(context)
            return (Symbol("val_$(cb.name)") => value,)
            
        elseif cb.on == :both
            # Call metric twice with modified context
            train_ctx = merge(context, (active_dataset=context.train_dataset,))
            val_ctx = merge(context, (active_dataset=context.validation_dataset,))
            return (
                Symbol("train_$(cb.name)") => cb.metric_fn(train_ctx),
                Symbol("val_$(cb.name)") => cb.metric_fn(val_ctx),
            )
            
        elseif cb.on == :none
            # Context-only metric (e.g., learning rate, epoch number)
            value = cb.metric_fn(context)
            return (cb.name => value,)
        end
    catch e
        @warn "Metric $(cb.name) failed" exception=(e, catch_backtrace())
        return nothing
    end
end
```

### Usage

```julia
# Simple validation metric
Metric(:gap, ctx -> compute_gap(benchmark, ctx.validation_dataset, ctx.model, ctx.maximizer))

# Train and validation
Metric(:gap, ctx -> compute_gap(benchmark, ctx.active_dataset, ctx.model, ctx.maximizer); on=:both)

# Context-only metric
Metric(:learning_rate, ctx -> ctx.optimizer.eta; on=:none)
Metric(:epoch, ctx -> ctx.epoch; on=:none)

# Complex metric using multiple context fields
Metric(:gap_improvement, ctx -> begin
    current_gap = compute_gap(benchmark, ctx.validation_dataset, ctx.model, ctx.maximizer)
    baseline_gap = ctx.baseline_gap  # Could be in context
    return (baseline_gap - current_gap) / baseline_gap
end)
```

### Pros & Cons

✅ **Pros:**
- Simpler signature: just `(context) -> value`
- No confusion about what `data` means
- `active_dataset` makes it explicit which dataset is being used
- Easy to understand and teach

❌ **Cons:**
- For `:both`, metric function is called twice (slight overhead)
- Need to remember to use `ctx.active_dataset` when `on=:both`
- Less flexible than current system

---

## Option 2: Declarative Dependencies

### Concept
Users declare what they need, and the callback system extracts and validates it for them.

### Implementation

```julia
struct Metric <: TrainingCallback
    name::Symbol
    metric_fn::Function
    on::Symbol  # :train, :validation, :both, :none
    needs::Vector{Symbol}  # [:model, :maximizer, :dataset, :epoch, etc.]
    extra_args::Tuple  # Additional arguments to pass to metric_fn
    
    function Metric(name::Symbol, metric_fn; on=:validation, needs=Symbol[], args=())
        new(name, metric_fn, on, needs, args)
    end
end

function on_epoch_end(cb::Metric, context)
    try
        # Extract only what's needed
        kwargs = NamedTuple()
        for key in cb.needs
            if key == :dataset
                # Special handling: dataset depends on 'on'
                if cb.on == :train
                    kwargs = merge(kwargs, (dataset=context.train_dataset,))
                elseif cb.on == :validation
                    kwargs = merge(kwargs, (dataset=context.validation_dataset,))
                end
            elseif haskey(context, key)
                kwargs = merge(kwargs, (key => context[key],))
            else
                @warn "Metric $(cb.name) requested '$key' but it's not in context"
            end
        end
        
        if cb.on == :train
            value = cb.metric_fn(cb.extra_args...; kwargs...)
            return (Symbol("train_$(cb.name)") => value,)
            
        elseif cb.on == :validation
            value = cb.metric_fn(cb.extra_args...; kwargs...)
            return (Symbol("val_$(cb.name)") => value,)
            
        elseif cb.on == :both
            # Call with train dataset
            train_kwargs = merge(kwargs, (dataset=context.train_dataset,))
            train_val = cb.metric_fn(cb.extra_args...; train_kwargs...)
            
            # Call with validation dataset
            val_kwargs = merge(kwargs, (dataset=context.validation_dataset,))
            val_val = cb.metric_fn(cb.extra_args...; val_kwargs...)
            
            return (
                Symbol("train_$(cb.name)") => train_val,
                Symbol("val_$(cb.name)") => val_val,
            )
        end
    catch e
        @warn "Metric $(cb.name) failed" exception=(e, catch_backtrace())
        return nothing
    end
end
```

### Usage

```julia
# Define metric function with clear signature
function compute_gap_metric(benchmark; dataset, model, maximizer)
    return compute_gap(benchmark, dataset, model, maximizer)
end

# Use with declarative dependencies
Metric(:gap, compute_gap_metric; 
       on=:validation,
       needs=[:dataset, :model, :maximizer],
       args=(benchmark,))

# Simpler version without needs (context-only)
Metric(:epoch, ctx -> ctx.epoch; on=:none)

# Multiple dependencies
function compute_loss_ratio(; train_loss, val_loss)
    return val_loss / train_loss
end

Metric(:loss_ratio, compute_loss_ratio;
       on=:none,
       needs=[:train_loss, :val_loss])

# Benchmark-generic version
struct GapMetric
    benchmark
end

function (gm::GapMetric)(; dataset, model, maximizer)
    return compute_gap(gm.benchmark, dataset, model, maximizer)
end

Metric(:gap, GapMetric(benchmark);
       on=:both,
       needs=[:dataset, :model, :maximizer])
```

### Pros & Cons

✅ **Pros:**
- **Type-safe**: Can validate that metric_fn has correct signature
- **Self-documenting**: `needs` shows exactly what's required
- **Flexible**: Can pass extra args via `args=`
- **Clear separation**: Metric function doesn't need to know about context structure
- **Reusable**: Metric functions can be defined once and reused

❌ **Cons:**
- More complex implementation
- Requires users to understand `needs` concept
- More verbose for simple metrics
- Need to handle special cases (like `:dataset` mapping)

---

## Option 3: Multiple Dispatch (Most Julian)

### Concept
Use Julia's multiple dispatch to create different `Metric` constructors for different use cases.

### Implementation

```julia
# Base type
abstract type TrainingCallback end

struct Metric{F} <: TrainingCallback
    name::Symbol
    metric_fn::F
    on::Symbol
end

# Constructor 1: Simple function with context
function Metric(name::Symbol, fn::Function; on=:validation)
    return Metric{typeof(fn)}(name, fn, on)
end

# Constructor 2: Callable struct (for metrics with state/parameters)
function Metric(name::Symbol, callable; on=:validation)
    return Metric{typeof(callable)}(name, callable, on)
end

# Dispatch on epoch_end based on metric type and 'on' value
function on_epoch_end(cb::Metric, context)
    try
        if cb.on == :validation
            value = compute_metric_value(cb.metric_fn, context, context.validation_dataset)
            return (Symbol("val_$(cb.name)") => value,)
            
        elseif cb.on == :train
            value = compute_metric_value(cb.metric_fn, context, context.train_dataset)
            return (Symbol("train_$(cb.name)") => value,)
            
        elseif cb.on == :both
            train_val = compute_metric_value(cb.metric_fn, context, context.train_dataset)
            val_val = compute_metric_value(cb.metric_fn, context, context.validation_dataset)
            return (
                Symbol("train_$(cb.name)") => train_val,
                Symbol("val_$(cb.name)") => val_val,
            )
            
        elseif cb.on == :none
            value = compute_metric_value(cb.metric_fn, context, nothing)
            return (cb.name => value,)
        end
    catch e
        @warn "Metric $(cb.name) failed" exception=(e, catch_backtrace())
        return nothing
    end
end

# Multiple dispatch for different metric function types

# For simple functions: f(context) -> value
function compute_metric_value(fn::Function, context, ::Nothing)
    return fn(context)
end

# For dataset metrics: f(dataset, context) -> value
function compute_metric_value(fn::Function, context, dataset)
    if applicable(fn, dataset, context)
        return fn(dataset, context)
    elseif applicable(fn, context)
        return fn(context)
    else
        error("Metric function doesn't accept (dataset, context) or (context)")
    end
end

# For callable structs with parameters
struct GapMetric
    benchmark
end

function (gm::GapMetric)(dataset, context)
    return compute_gap(gm.benchmark, dataset, context.model, context.maximizer)
end

function compute_metric_value(callable, context, dataset)
    if applicable(callable, dataset, context)
        return callable(dataset, context)
    elseif applicable(callable, context)
        return callable(context)
    else
        error("Callable doesn't accept (dataset, context) or (context)")
    end
end
```

### Usage

```julia
# Option A: Simple lambda with dataset and context
Metric(:gap, (dataset, ctx) -> compute_gap(b, dataset, ctx.model, ctx.maximizer))

# Option B: Context-only for non-dataset metrics
Metric(:epoch, ctx -> ctx.epoch; on=:none)
Metric(:learning_rate, ctx -> ctx.learning_rate; on=:none)

# Option C: Callable struct (best for reusability)
struct GapMetric
    benchmark
end

function (gm::GapMetric)(dataset, context)
    return compute_gap(gm.benchmark, dataset, context.model, context.maximizer)
end

gap_metric = GapMetric(benchmark)
Metric(:gap, gap_metric; on=:both)

# Option D: Pre-defined metric types
struct ModelCheckpointMetric
    filepath::String
    mode::Symbol  # :min or :max
end

function (mcm::ModelCheckpointMetric)(context)
    # Save model if it's the best so far
    # ... implementation ...
end

Metric(:checkpoint, ModelCheckpointMetric("best_model.bson", :min); on=:none)
```

### Pros & Cons

✅ **Pros:**
- **Very Julian**: Uses multiple dispatch naturally
- **Flexible**: Supports both `(dataset, ctx)` and `(ctx)` signatures
- **Backward compatible**: Can keep current API
- **Type-safe**: Dispatch checks at compile time
- **Encourages good design**: Callable structs for complex metrics

❌ **Cons:**
- More complex implementation with multiple dispatch paths
- Users need to understand when to use which signature
- `applicable` checks add slight runtime overhead
- May be harder to debug when dispatch fails

---

## Comparison Matrix

| Feature | Current | Option 1 | Option 2 | Option 3 |
|---------|---------|----------|----------|----------|
| **Simplicity** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Type Safety** | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Discoverability** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Flexibility** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Performance** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Maintainability** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Learning Curve** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Backward Compat** | - | ❌ | ❌ | ✅ (partial) |

---

## Recommendation: Hybrid Approach

I recommend a **combination of Option 1 and Option 3**:

### Proposed Design

```julia
struct Metric{F} <: TrainingCallback
    name::Symbol
    metric_fn::F
    on::Symbol
    
    function Metric(name::Symbol, fn; on=:validation)
        new{typeof(fn)}(name, fn, on)
    end
end

function on_epoch_end(cb::Metric, context)
    try
        if cb.on == :validation
            value = call_metric(cb.metric_fn, context, :validation)
            return (Symbol("val_$(cb.name)") => value,)
            
        elseif cb.on == :train
            value = call_metric(cb.metric_fn, context, :train)
            return (Symbol("train_$(cb.name)") => value,)
            
        elseif cb.on == :both
            train_val = call_metric(cb.metric_fn, context, :train)
            val_val = call_metric(cb.metric_fn, context, :validation)
            return (
                Symbol("train_$(cb.name)") => train_val,
                Symbol("val_$(cb.name)") => val_val,
            )
            
        else  # :none or custom
            value = call_metric(cb.metric_fn, context, cb.on)
            return (cb.name => value,)
        end
    catch e
        @warn "Metric $(cb.name) failed at epoch $(context.epoch)" exception=(e, catch_backtrace())
        return nothing
    end
end

# Multiple dispatch for different signatures

# Signature 1: f(context) -> value
# Best for: epoch number, learning rate, loss ratios, etc.
function call_metric(fn::Function, context, ::Symbol)
    if applicable(fn, context)
        return fn(context)
    else
        error("Metric function must accept (context) or (dataset, context)")
    end
end

# Signature 2: f(dataset, context) -> value  
# Best for: metrics that need a specific dataset
function call_metric(fn::Function, context, dataset_key::Symbol)
    dataset = if dataset_key == :validation
        context.validation_dataset
    elseif dataset_key == :train
        context.train_dataset
    else
        get(context, dataset_key, nothing)
    end
    
    # Try both signatures
    if applicable(fn, dataset, context)
        return fn(dataset, context)
    elseif applicable(fn, context)
        return fn(context)
    else
        error("Metric function must accept (dataset, context) or (context)")
    end
end

# For callable structs
function call_metric(obj, context, dataset_key::Symbol)
    # Same logic as function but with obj instead of fn
    dataset = if dataset_key == :validation
        context.validation_dataset
    elseif dataset_key == :train
        context.train_dataset
    else
        get(context, dataset_key, nothing)
    end
    
    if applicable(obj, dataset, context)
        return obj(dataset, context)
    elseif applicable(obj, context)
        return obj(context)
    else
        error("Metric callable must accept (dataset, context) or (context)")
    end
end
```

### Usage Examples

```julia
# Use case 1: Simple context-only metric
Metric(:epoch, ctx -> ctx.epoch; on=:none)

# Use case 2: Dataset-dependent metric (current style, still works!)
Metric(:gap, (dataset, ctx) -> compute_gap(b, dataset, ctx.model, ctx.maximizer))

# Use case 3: Reusable callable struct
struct GapMetric
    benchmark
end

(gm::GapMetric)(dataset, ctx) = compute_gap(gm.benchmark, dataset, ctx.model, ctx.maximizer)

Metric(:gap, GapMetric(benchmark); on=:both)

# Use case 4: Complex metric using multiple context fields
Metric(:loss_improvement, ctx -> begin
    current = ctx.val_loss
    initial = ctx.initial_val_loss
    return (initial - current) / initial
end; on=:none)

# Use case 5: Test dataset (custom dataset)
test_dataset = ...
Metric(:test_gap, (dataset, ctx) -> compute_gap(b, dataset, ctx.model, ctx.maximizer);
       on=:test_dataset)  # Would need to add test_dataset to context
```

---

## Implementation Plan

### Phase 1: Add Support (Non-Breaking)
1. ✅ Add `call_metric` helper with multiple dispatch
2. ✅ Support both `(context)` and `(dataset, context)` signatures
3. ✅ Add tests for both signatures
4. ✅ Update documentation with examples

### Phase 2: Encourage Migration (Soft Deprecation)
1. ✅ Add examples using new `(context)` signature
2. ✅ Update tutorials to show both patterns
3. ⚠️ Add note that `(context)` is preferred for simple metrics

### Phase 3: Improve Developer Experience
1. ✅ Add helpful error messages when signature is wrong
2. ✅ Add `@assert applicable(...)` checks with clear messages
3. ✅ Create common metric function library

### Example Error Messages

```julia
try
    return fn(dataset, context)
catch MethodError
    error("""
    Metric function $(cb.name) failed with signature (dataset, context).
    
    Possible fixes:
    1. Define your function to accept (dataset, context):
       (dataset, ctx) -> compute_metric(dataset, ctx.model)
    
    2. Or use context-only signature if you don't need dataset:
       ctx -> compute_metric(ctx.validation_dataset, ctx.model)
    
    3. For callable structs, implement:
       (obj::MyMetric)(dataset, context) = ...
    """)
end
```

---

## Additional Improvements

### 1. Add Standard Context Fields

Extend context to include commonly-needed values:

```julia
context = (
    epoch=epoch,
    model=model,
    maximizer=maximizer,
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    train_loss=avg_train_loss,              # NEW
    val_loss=avg_val_loss,                  # NEW
    optimizer=optimizer,                    # NEW
    learning_rate=get_learning_rate(opt),   # NEW
)
```

### 2. Create Common Metric Library

```julia
# In src/callbacks/metrics.jl

"""Pre-defined metrics for common use cases"""

struct GapMetric
    benchmark
end

(gm::GapMetric)(dataset, ctx) = compute_gap(gm.benchmark, dataset, ctx.model, ctx.maximizer)

struct RegretMetric
    benchmark
end

(rm::RegretMetric)(dataset, ctx) = compute_regret(rm.benchmark, dataset, ctx.model, ctx.maximizer)

struct LossImprovementMetric end

function (lim::LossImprovementMetric)(ctx)
    if !haskey(ctx, :initial_val_loss)
        return 0.0
    end
    return (ctx.initial_val_loss - ctx.val_loss) / ctx.initial_val_loss
end

# Usage:
callbacks = [
    Metric(:gap, GapMetric(benchmark); on=:both),
    Metric(:regret, RegretMetric(benchmark)),
    Metric(:improvement, LossImprovementMetric(); on=:none),
]
```

### 3. Add Type Annotations Helper

```julia
"""
Helper to validate metric function signatures at callback creation time
"""
function validate_metric_signature(fn, on::Symbol)
    # Try to compile the function with expected types
    # This gives early errors instead of runtime errors
    
    if on in [:train, :validation, :both]
        if !hasmethod(fn, Tuple{Any, NamedTuple}) && !hasmethod(fn, Tuple{NamedTuple})
            @warn """
            Metric function may have incorrect signature.
            Expected: (dataset, context) or (context)
            This check is best-effort and may have false positives.
            """
        end
    end
end

# Call in constructor
function Metric(name::Symbol, fn; on=:validation)
    validate_metric_signature(fn, on)
    new{typeof(fn)}(name, fn, on)
end
```

---

## Migration Guide

### From Current API

```julia
# OLD (Current)
Metric(:gap, (data, ctx) -> compute_gap(benchmark, data, ctx.model, ctx.maximizer))

# NEW (Recommended - Option 1: Context-only)
Metric(:gap, ctx -> compute_gap(benchmark, ctx.validation_dataset, ctx.model, ctx.maximizer))

# NEW (Alternative - Option 2: Keep dataset param, clearer naming)
Metric(:gap, (dataset, ctx) -> compute_gap(benchmark, dataset, ctx.model, ctx.maximizer))

# NEW (Best - Option 3: Reusable callable struct)
struct GapMetric
    benchmark
end
(gm::GapMetric)(dataset, ctx) = compute_gap(gm.benchmark, dataset, ctx.model, ctx.maximizer)

Metric(:gap, GapMetric(benchmark); on=:both)
```

---

## Summary

**Best Approach: Hybrid (Option 1 + Option 3)**

**Why:**
1. ✅ Supports both simple `(context)` and explicit `(dataset, context)` signatures
2. ✅ Uses Julia's multiple dispatch naturally
3. ✅ Backward compatible with current usage
4. ✅ Encourages good practices (callable structs for reusable metrics)
5. ✅ Clear error messages guide users
6. ✅ Self-documenting code

**Implementation Priority:**
1. **High**: Add `call_metric` multiple dispatch helper
2. **High**: Add context fields (train_loss, val_loss, etc.)
3. **Medium**: Create common metrics library
4. **Medium**: Add validation and better error messages
5. **Low**: Add type annotation helpers

**Impact:**
- 📉 Reduces boilerplate for simple metrics
- 📈 Improves code reusability
- 📈 Better error messages and debugging
- 📈 More Pythonic for users coming from PyTorch/TensorFlow
- 📈 More Julian for experienced Julia users

