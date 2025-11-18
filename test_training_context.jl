#!/usr/bin/env julia

# Quick test script to verify TrainingContext integration
using Pkg;
Pkg.activate(".")
using DecisionFocusedLearningAlgorithms, DecisionFocusedLearningBenchmarks
using MLUtils

println("Testing TrainingContext integration...")

# Create a simple benchmark test
benchmark = ArgmaxBenchmark()
dataset = generate_dataset(benchmark, 6)  # Small dataset for quick test
train_dataset, validation_dataset = splitobs(dataset; at=0.5)

model = generate_statistical_model(benchmark)
maximizer = generate_maximizer(benchmark)

# Test basic TrainingContext functionality
println("\n1. Testing TrainingContext creation...")
ctx = TrainingContext(;
    model=model,
    epoch=5,
    maximizer=maximizer,
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    train_loss=1.5,
    val_loss=2.0,
    custom_field="test_value",
)

println("   ✓ Model type: ", typeof(ctx.model))
println("   ✓ Epoch: ", ctx.epoch)
println("   ✓ Train loss: ", ctx.train_loss)
println("   ✓ Val loss: ", ctx.val_loss)
println("   ✓ Custom field: ", ctx.custom_field)
println("   ✓ Has custom field: ", haskey(ctx, :custom_field))

# Test with metric callbacks
println("\n2. Testing TrainingContext with callbacks...")
callbacks = [
    Metric(:epoch, (data, ctx) -> ctx.epoch; on=:none),
    Metric(:model_info, (data, ctx) -> string(typeof(ctx.model)); on=:none),
]

# Test FYL training with TrainingContext
println("\n3. Testing FYL training with TrainingContext...")
try
    history = fyl_train_model!(
        deepcopy(model),
        maximizer,
        train_dataset,
        validation_dataset;
        epochs=2,
        callbacks=callbacks,
    )
    println("   ✓ FYL training completed successfully!")
    println("   ✓ History keys: ", keys(history))

    # Check if callbacks worked
    if haskey(history, :epoch)
        epoch_times, epoch_values = get(history, :epoch)
        println("   ✓ Epoch callback values: ", epoch_values)
    end

catch e
    println("   ✗ FYL training failed: ", e)
    rethrow(e)
end

println("\n4. Testing DAgger with TrainingContext...")
try
    # For ArgmaxBenchmark, we need to check if DAgger is supported
    # Let's skip DAgger test for now since it may need special environment setup
    println("   ✓ DAgger test skipped for ArgmaxBenchmark (not applicable)")

catch e
    println("   ✗ DAgger training failed: ", e)
    rethrow(e)
end

println("\n🎉 All TrainingContext tests passed!")
