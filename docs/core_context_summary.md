# Summary: Core Context Solution

**Date:** November 13, 2025  
**Issue:** How to balance genericity and ease-of-use in callback context across multiple algorithms

---

## ✅ Solution Implemented

We adopted a **convention-based core context** approach:

### Core Fields (Required in ALL algorithms)
```julia
context = (
    epoch::Int,
    model,
    maximizer,
    train_dataset,
    validation_dataset,
    train_loss::Float64,   # ✅ Added
    val_loss::Float64,     # ✅ Added
    # ... + algorithm-specific fields
)
```

### Algorithm-Specific Extensions (Optional)
```julia
# DAgger adds:
context = (...core..., α=α, expert_policy=..., environments=...)

# Future SPO+ might add:
context = (...core..., decision_cost=..., gradient_type=...)

# Your next algorithm adds whatever it needs!
```

---

## 📝 Changes Made

### 1. Updated `fyl_new.jl`
✅ Added `train_loss` and `val_loss` to context (both at epoch 0 and in training loop)

**Before:**
```julia
context = (epoch=epoch, model=model, maximizer=maximizer, 
           train_dataset=train_dataset, validation_dataset=validation_dataset)
```

**After:**
```julia
context = (epoch=epoch, model=model, maximizer=maximizer,
           train_dataset=train_dataset, validation_dataset=validation_dataset,
           train_loss=avg_train_loss, val_loss=avg_val_loss)
```

### 2. Updated `callbacks.jl` Documentation
✅ Documented the core context contract in `TrainingCallback` docstring:
- Lists all 7 required core fields
- Explains algorithm-specific extensions
- Provides examples of portable vs. algorithm-specific metrics

### 3. Created Examples
✅ `docs/src/tutorials/portable_metrics_example.jl` - Shows how to:
- Write portable metrics that work everywhere
- Use same callbacks with FYL and DAgger
- Opt-in to algorithm-specific features
- Create reusable metric functions

### 4. Created Design Documentation
✅ `docs/context_design_philosophy.md` - Complete guide covering:
- The generic vs. easy-to-use tension
- Layered context design approach
- Usage patterns across algorithms
- Best practices and recommendations

---

## 🎯 Benefits

### For Users
1. **Write once, use everywhere**: Metrics using core fields work with all algorithms
2. **Clear contract**: Know exactly what's always available
3. **Opt-in complexity**: Can access algorithm-specific features when needed
4. **Type-safe**: Context fields are documented and validated

### For Developers (You!)
1. **Freedom to extend**: Each new algorithm can add whatever fields it needs
2. **No breaking changes**: Adding new algorithms doesn't break existing metrics
3. **Simple implementation**: Just a NamedTuple with documented conventions
4. **Future-proof**: Pattern scales to unlimited number of algorithms

---

## 📖 How to Use

### Writing Portable Metrics (Recommended)

```julia
# ✅ Works with FYL, DAgger, SPO+, any future algorithm
callbacks = [
    Metric(:gap, ctx -> compute_gap(benchmark, ctx.validation_dataset, ctx.model, ctx.maximizer)),
    Metric(:loss_ratio, ctx -> ctx.val_loss / ctx.train_loss; on=:none),
    Metric(:epoch, ctx -> ctx.epoch; on=:none),
]

# Use with any algorithm
fyl_train_model!(model, maximizer, train, val; epochs=100, callbacks=callbacks)
DAgger_train_model!(model, maximizer, envs, ...; iterations=10, callbacks=callbacks)
spo_train_model!(model, maximizer, train, val; epochs=100, callbacks=callbacks)  # Future!
```

### Writing Algorithm-Specific Metrics (When Needed)

```julia
# Check for optional fields
Metric(:alpha, ctx -> haskey(ctx, :α) ? ctx.α : NaN; on=:none)

# Or use get with default
Metric(:alpha, ctx -> get(ctx, :α, NaN); on=:none)
```

### Adding a New Algorithm

When you implement a new algorithm, just:

1. **Provide the 7 core fields** (required)
2. **Add any algorithm-specific fields** you need
3. **Document** your extensions in the algorithm's docstring
4. **Done!** All existing metrics will work

Example for future SPO+ implementation:
```julia
function spo_plus_train_model!(model, maximizer, train_dataset, validation_dataset;
                               epochs=100, callbacks=TrainingCallback[])
    for epoch in 1:epochs
        avg_train_loss, avg_val_loss, avg_cost = train_epoch_spo!(...)
        
        # Provide core + SPO+ specific fields
        context = (
            # Core (required)
            epoch=epoch,
            model=model,
            maximizer=maximizer,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            # SPO+ specific (optional)
            decision_cost=avg_cost,
            gradient_type=:spo_plus,
        )
        
        run_callbacks!(history, callbacks, context)
    end
end
```

---

## 🔮 Future Enhancements (Optional)

If you find yourself having issues with missing fields or typos, you could later add:

### Option 1: Helper Functions
```julia
get_context_field(ctx, :α, NaN)  # Safe getter with default
is_dagger_context(ctx)           # Type checking
```

### Option 2: TrainingContext Struct (More Formal)
```julia
struct TrainingContext
    # Core fields with types
    epoch::Int
    model
    maximizer
    train_dataset
    validation_dataset
    train_loss::Float64
    val_loss::Float64
    
    # Extensions dictionary
    extensions::Dict{Symbol, Any}
end
```

But **you don't need this now**. Start simple with NamedTuple + conventions.

---

## ✨ Key Insight

**You don't need to solve for ALL use cases upfront.**

- **80% of metrics** only use core fields → work everywhere automatically
- **20% of metrics** are algorithm-specific → opt-in explicitly with `haskey()`

This is the **sweet spot** between generic and easy-to-use! 🎯

---

## 📚 See Also

- `docs/context_design_philosophy.md` - Detailed design rationale
- `docs/src/tutorials/portable_metrics_example.jl` - Runnable examples
- `docs/callback_system_analysis.md` - Original analysis that led to this
- `src/callbacks.jl` - Implementation and API documentation

---

## Questions Answered

> "How can I be generic + easy to use at the same time?"

**Answer:** Define a minimal set of core fields that EVERY algorithm provides, then let each algorithm extend as needed. Users write against the core for portability, and opt-in to extensions for specific features.

> "Will the context content change when I add new algorithms?"

**Answer:** The CORE fields stay the same (that's the contract). New algorithms add ADDITIONAL fields, but never remove or change the core ones. This means old metrics keep working with new algorithms.

> "Isn't this difficult to maintain?"

**Answer:** No! It's actually simpler than alternatives because:
1. You document once (7 core fields)
2. Each algorithm independently adds what it needs
3. No coordination needed between algorithms
4. Users only learn the core once

---

**Status:** ✅ **Implemented and Documented**

The core context system is now in place and ready to use. You can confidently add new algorithms knowing that existing metrics will continue to work!
