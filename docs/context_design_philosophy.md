# Context Design Philosophy: Generic vs. Easy-to-Use

**Date:** November 13, 2025  
**Author:** Discussion with taleboy  
**Topic:** How to design a context system that works across multiple algorithms while remaining user-friendly

---

## The Core Problem

You want to implement multiple training algorithms (FYL, DAgger, SPO+, QPTL, IntOpt, etc.), but:

1. **Different algorithms need different information**
   - FYL: model, maximizer, datasets, loss
   - DAgger: model, maximizer, environments, expert policy, α (mixing parameter)
   - SPO+: model, maximizer, datasets, cost vectors
   - IntOpt: model, maximizer, datasets, interpolation schedule
   - Imitation Learning: model, expert trajectories, behavior cloning parameters

2. **Users want simple metrics that work everywhere**
   ```julia
   # User wants to write this ONCE:
   Metric(:gap, ctx -> compute_gap(benchmark, ctx.validation_dataset, ctx.model, ctx.maximizer))
   
   # And use it with ANY algorithm:
   fyl_train_model!(...; callbacks=[gap_metric])
   dagger_train_model!(...; callbacks=[gap_metric])
   spo_train_model!(...; callbacks=[gap_metric])
   ```

3. **Question: How can context be both flexible AND consistent?**

---

## Solution: Layered Context Design

### Concept: Core Context + Algorithm-Specific Extensions

```
┌─────────────────────────────────────────────────────┐
│  Core Context (Always Present)                      │
│  - epoch, model, maximizer                          │
│  - train_dataset, validation_dataset                │
│  - train_loss, val_loss                             │
├─────────────────────────────────────────────────────┤
│  Algorithm-Specific Extensions (Optional)           │
│  - DAgger: α, expert_policy, environments           │
│  - SPO+: cost_vectors, perturbed_costs              │
│  - IntOpt: interpolation_weight                     │
└─────────────────────────────────────────────────────┘
```

### Implementation Strategy

```julia
# Define a base context type
struct TrainingContext
    # Core fields (always present)
    epoch::Int
    model
    maximizer
    train_dataset
    validation_dataset
    train_loss::Float64
    val_loss::Float64
    
    # Extensions (algorithm-specific, stored as NamedTuple)
    extensions::NamedTuple
end

# Easy constructor
function TrainingContext(; epoch, model, maximizer, train_dataset, validation_dataset, 
                          train_loss, val_loss, kwargs...)
    extensions = NamedTuple(kwargs)
    return TrainingContext(epoch, model, maximizer, train_dataset, validation_dataset,
                          train_loss, val_loss, extensions)
end

# Make it behave like a NamedTuple for easy access
Base.getproperty(ctx::TrainingContext, sym::Symbol) = begin
    # First check core fields
    if sym in fieldnames(TrainingContext)
        return getfield(ctx, sym)
    # Then check extensions
    elseif haskey(getfield(ctx, :extensions), sym)
        return getfield(ctx, :extensions)[sym]
    else
        error("Field $sym not found in context")
    end
end

Base.haskey(ctx::TrainingContext, sym::Symbol) = begin
    sym in fieldnames(TrainingContext) || haskey(getfield(ctx, :extensions), sym)
end

# Helper to get all available keys
function Base.keys(ctx::TrainingContext)
    core_keys = fieldnames(TrainingContext)[1:end-1]  # Exclude :extensions
    ext_keys = keys(getfield(ctx, :extensions))
    return (core_keys..., ext_keys...)
end
```

---

## Usage Across Different Algorithms

### 1. FYL (Simple Case)

```julia
function fyl_train_model!(model, maximizer, train_dataset, validation_dataset; 
                          epochs=100, callbacks=TrainingCallback[])
    # ...training loop...
    
    for epoch in 1:epochs
        # Training
        avg_train_loss, avg_val_loss = train_epoch!(...)
        
        # Create context with ONLY core fields
        context = TrainingContext(
            epoch=epoch,
            model=model,
            maximizer=maximizer,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            # No extensions needed for FYL
        )
        
        run_callbacks!(history, callbacks, context)
    end
end
```

### 2. DAgger (With Extensions)

```julia
function DAgger_train_model!(model, maximizer, train_environments, validation_environments,
                            anticipative_policy; iterations=5, fyl_epochs=3, 
                            callbacks=TrainingCallback[])
    α = 1.0
    
    for iter in 1:iterations
        # Generate dataset from current policy mix
        dataset = generate_mixed_dataset(environments, α, anticipative_policy, model, maximizer)
        
        # Train with FYL
        for epoch in 1:fyl_epochs
            avg_train_loss, avg_val_loss = train_epoch!(...)
            
            global_epoch = (iter - 1) * fyl_epochs + epoch
            
            # Create context with DAgger-specific extensions
            context = TrainingContext(
                epoch=global_epoch,
                model=model,
                maximizer=maximizer,
                train_dataset=dataset,
                validation_dataset=validation_dataset,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                # DAgger-specific extensions
                α=α,
                dagger_iteration=iter,
                expert_policy=anticipative_policy,
                train_environments=train_environments,
                validation_environments=validation_environments,
            )
            
            run_callbacks!(history, callbacks, context)
        end
        
        α *= 0.9  # Decay
    end
end
```

### 3. SPO+ (Different Extensions)

```julia
function spo_plus_train_model!(model, maximizer, train_dataset, validation_dataset;
                               epochs=100, callbacks=TrainingCallback[])
    
    for epoch in 1:epochs
        # SPO+ specific training
        avg_train_loss, avg_val_loss, avg_cost = train_epoch_spo!(...)
        
        # Create context with SPO+-specific extensions
        context = TrainingContext(
            epoch=epoch,
            model=model,
            maximizer=maximizer,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            # SPO+-specific extensions
            avg_decision_cost=avg_cost,
            gradient_type=:spo_plus,
        )
        
        run_callbacks!(history, callbacks, context)
    end
end
```

---

## User-Friendly Metric Writing

### Generic Metrics (Work Everywhere)

Users can write metrics that **only use core fields**:

```julia
# ✅ This works with ANY algorithm
Metric(:gap, ctx -> compute_gap(benchmark, ctx.validation_dataset, ctx.model, ctx.maximizer))

# ✅ This works with ANY algorithm
Metric(:loss_improvement, ctx -> begin
    if ctx.epoch == 0
        return 0.0
    end
    return (ctx.val_loss - previous_loss) / previous_loss
end; on=:none)

# ✅ This works with ANY algorithm
Metric(:epoch, ctx -> ctx.epoch; on=:none)
```

### Algorithm-Specific Metrics (Opt-In)

Users can write metrics that check for algorithm-specific fields:

```julia
# DAgger-specific: monitor mixing parameter
Metric(:alpha, ctx -> begin
    if haskey(ctx, :α)
        return ctx.α
    else
        return missing  # Or NaN, or skip this metric
    end
end; on=:none)

# Or with error handling
Metric(:alpha, ctx -> get(ctx.extensions, :α, NaN); on=:none)

# SPO+-specific: monitor decision cost
Metric(:decision_cost, ctx -> begin
    haskey(ctx, :avg_decision_cost) || return NaN
    return ctx.avg_decision_cost
end; on=:none)
```

### Smart Metrics (Adapt to Context)

```julia
# Metric that uses algorithm-specific info if available
Metric(:detailed_gap, ctx -> begin
    gap = compute_gap(benchmark, ctx.validation_dataset, ctx.model, ctx.maximizer)
    
    # If we have environments (DAgger), compute trajectory-based gap
    if haskey(ctx, :validation_environments)
        traj_gap = compute_trajectory_gap(benchmark, ctx.validation_environments, ctx.model)
        return (standard_gap=gap, trajectory_gap=traj_gap)
    end
    
    return gap
end)
```

---

## Benefits of This Design

### 1. ✅ **Consistency**: Core fields always available
```julia
# These fields are GUARANTEED to exist in any training algorithm:
ctx.epoch
ctx.model
ctx.maximizer
ctx.train_dataset
ctx.validation_dataset
ctx.train_loss
ctx.val_loss
```

### 2. ✅ **Flexibility**: Algorithms can add whatever they need
```julia
# DAgger adds:
ctx.α
ctx.expert_policy
ctx.train_environments

# SPO+ adds:
ctx.avg_decision_cost
ctx.gradient_type

# Your future algorithm adds:
ctx.whatever_you_need
```

### 3. ✅ **Discoverability**: Easy to see what's available
```julia
# User can inspect context
println(keys(ctx))
# Output: (:epoch, :model, :maximizer, :train_dataset, :validation_dataset, 
#          :train_loss, :val_loss, :α, :dagger_iteration, :expert_policy, ...)

# Or check if a field exists
if haskey(ctx, :α)
    println("This is DAgger training with α = $(ctx.α)")
end
```

### 4. ✅ **Safety**: Clear errors when accessing missing fields
```julia
# If you try to access a field that doesn't exist:
ctx.nonexistent_field
# Error: Field nonexistent_field not found in context
# Available fields: epoch, model, maximizer, ..., α, expert_policy
```

### 5. ✅ **Backward Compatibility**: Adding new algorithms doesn't break old metrics
```julia
# Old metric written for FYL
old_metric = Metric(:gap, ctx -> compute_gap(b, ctx.validation_dataset, ctx.model, ctx.maximizer))

# Still works with new algorithms!
fyl_train_model!(...; callbacks=[old_metric])
dagger_train_model!(...; callbacks=[old_metric])
spo_train_model!(...; callbacks=[old_metric])
future_algorithm_train_model!(...; callbacks=[old_metric])
```

---

## Alternative: Even Simpler (Just NamedTuple)

If you want to keep it super simple, you could just use a NamedTuple with conventions:

```julia
# Core fields (convention: ALWAYS include these)
context = (
    epoch=epoch,
    model=model,
    maximizer=maximizer,
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    train_loss=avg_train_loss,
    val_loss=avg_val_loss,
    # Algorithm-specific (optional)
    α=α,
    expert_policy=expert_policy,
)

# Pros:
# ✅ Extremely simple
# ✅ No new types needed
# ✅ Works with existing code

# Cons:
# ❌ No validation that core fields exist
# ❌ Typos won't be caught
# ❌ Less discoverability
```

**Recommendation**: Start with NamedTuple (simpler), then create `TrainingContext` struct later if needed.

---

## Recommended Best Practice

### 1. **Document Core Context Fields**

Create a clear spec in your documentation:

```julia
"""
# Training Context

All training algorithms must provide these core fields:

## Required Fields
- `epoch::Int` - Current training epoch (0-indexed)
- `model` - The model being trained
- `maximizer` - The optimization solver/maximizer
- `train_dataset` - Training dataset
- `validation_dataset` - Validation dataset  
- `train_loss::Float64` - Average training loss for this epoch
- `val_loss::Float64` - Average validation loss for this epoch

## Optional Fields (Algorithm-Specific)
Algorithms may add additional fields as needed. Check with `haskey(ctx, :field_name)`.

Common optional fields:
- `test_dataset` - Test dataset (if available)
- `optimizer` - The optimizer instance
- `learning_rate::Float64` - Current learning rate

### DAgger-Specific
- `α::Float64` - Expert/learner mixing parameter
- `dagger_iteration::Int` - Current DAgger iteration
- `expert_policy` - Expert policy function
- `train_environments` - Training environments
- `validation_environments` - Validation environments

### SPO+-Specific  
- `avg_decision_cost::Float64` - Average decision quality
- `gradient_type::Symbol` - Type of gradient (:spo_plus, :blackbox, etc.)
"""
```

### 2. **Provide Helper Functions for Common Patterns**

```julia
# Helper to safely get optional fields
function get_context_field(ctx, field::Symbol, default=nothing)
    haskey(ctx, field) ? ctx[field] : default
end

# Helper to check if this is a specific algorithm
is_dagger_context(ctx) = haskey(ctx, :α) && haskey(ctx, :expert_policy)
is_spo_context(ctx) = haskey(ctx, :gradient_type) && ctx.gradient_type == :spo_plus

# Usage in metrics:
Metric(:alpha, ctx -> get_context_field(ctx, :α, NaN); on=:none)

Metric(:method, ctx -> begin
    if is_dagger_context(ctx)
        return "DAgger (α=$(ctx.α))"
    elseif is_spo_context(ctx)
        return "SPO+"
    else
        return "FYL"
    end
end; on=:none)
```

### 3. **Create a Metric Library with Helpers**

```julia
# src/callbacks/common_metrics.jl

"""
Creates a gap metric that works with any algorithm.
Automatically uses environments if available (for DAgger), otherwise uses dataset.
"""
function gap_metric(benchmark; name=:gap, on=:validation)
    return Metric(name, ctx -> begin
        # Try to use environments if available (more accurate for sequential problems)
        env_key = on == :validation ? :validation_environments : :train_environments
        dataset_key = on == :validation ? :validation_dataset : :train_dataset
        
        if haskey(ctx, env_key)
            # Trajectory-based gap (for DAgger)
            return compute_trajectory_gap(benchmark, ctx[env_key], ctx.model, ctx.maximizer)
        else
            # Dataset-based gap (for FYL, SPO+, etc.)
            return compute_gap(benchmark, ctx[dataset_key], ctx.model, ctx.maximizer)
        end
    end; on=on)
end

# Usage:
callbacks = [
    gap_metric(benchmark),  # Works with FYL, DAgger, SPO+, etc.
]
```

---

## Example: Complete Multi-Algorithm Workflow

```julia
using DecisionFocusedLearningAlgorithms

# Setup
benchmark = DynamicVehicleSchedulingBenchmark()
dataset = generate_dataset(benchmark, 100)
train_data, val_data, test_data = splitobs(dataset; at=(0.6, 0.2, 0.2))

# Define metrics that work with ANY algorithm
callbacks = [
    gap_metric(benchmark; on=:validation),
    gap_metric(benchmark; on=:train),
    Metric(:epoch, ctx -> ctx.epoch; on=:none),
    Metric(:loss_ratio, ctx -> ctx.val_loss / ctx.train_loss; on=:none),
]

# Train with FYL
model_fyl = generate_statistical_model(benchmark)
maximizer = generate_maximizer(benchmark)
history_fyl, model_fyl = fyl_train_model(
    model_fyl, maximizer, train_data, val_data;
    epochs=100,
    callbacks=callbacks  # Same callbacks!
)

# Train with DAgger
model_dagger = generate_statistical_model(benchmark)
train_envs = generate_environments(benchmark, train_instances)
val_envs = generate_environments(benchmark, val_instances)
history_dagger, model_dagger = DAgger_train_model(
    model_dagger, maximizer, train_envs, val_envs, anticipative_policy;
    iterations=10,
    fyl_epochs=10,
    callbacks=callbacks  # Same callbacks work!
)

# Train with SPO+ (future)
model_spo = generate_statistical_model(benchmark)
history_spo, model_spo = spo_plus_train_model(
    model_spo, maximizer, train_data, val_data;
    epochs=100,
    callbacks=callbacks  # Same callbacks work!
)

# Compare results
using Plots
plot(get(history_fyl, :val_gap)..., label="FYL")
plot!(get(history_dagger, :val_gap)..., label="DAgger")
plot!(get(history_spo, :val_gap)..., label="SPO+")
```

---

## Decision: What to Implement Now

### Phase 1 (Immediate - Keep it Simple)
```julia
# Just use NamedTuple with documented conventions
context = (
    epoch=epoch,
    model=model,
    maximizer=maximizer,
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    train_loss=avg_train_loss,
    val_loss=avg_val_loss,
    # ... any algorithm-specific fields ...
)
```

**Action Items:**
1. ✅ Document required core fields in callbacks.jl docstring
2. ✅ Add `train_loss` and `val_loss` to context (currently missing!)
3. ✅ Update DAgger to include algorithm-specific fields (α, expert_policy, etc.)
4. ✅ Create examples showing how to write generic metrics

### Phase 2 (Short-term - Add Helpers)
```julia
# Add helper functions
get_context_field(ctx, :α, NaN)
is_dagger_context(ctx)

# Add common metric factory functions
gap_metric(benchmark)
regret_metric(benchmark)
```

### Phase 3 (Long-term - If Needed)
```julia
# Create TrainingContext struct for better validation
struct TrainingContext
    # ... as described above ...
end
```

Only do this if you find yourself repeatedly having issues with missing fields or typos.

---

## Summary: The Answer to Your Question

> How can I be generic + easy to use at the same time?

**Answer: Use a convention-based approach with a core set of required fields.**

### The Strategy:
1. **Define a "core context contract"** - 7 required fields that EVERY algorithm must provide
2. **Allow arbitrary extensions** - Algorithms can add whatever else they need
3. **Write metrics against the core** - Most metrics only use core fields → work everywhere
4. **Opt-in to algorithm-specific features** - Advanced users can check for and use extensions

### The Key Insight:
**You don't need to make context work for EVERY possible use case. You just need to make the COMMON cases (80%) work everywhere, and allow the SPECIAL cases (20%) to be handled explicitly.**

### Concrete Next Steps:
1. Add `train_loss` and `val_loss` to FYL and DAgger contexts
2. Document the core context fields in the `TrainingCallback` docstring
3. Create 2-3 example metrics in the docs that work with any algorithm
4. When you add a new algorithm, just follow the same pattern

**This way:** Users write simple metrics once, they work everywhere, and you maintain flexibility for algorithm-specific features. 🎯

