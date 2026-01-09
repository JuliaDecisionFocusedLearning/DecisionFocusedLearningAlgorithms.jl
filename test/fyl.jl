
using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks
using MLUtils
using Test
using ValueHistories

@testset "Training Functions" begin
    # Setup - use a simple benchmark for fast tests
    benchmark = ArgmaxBenchmark()
    dataset = generate_dataset(benchmark, 30)
    train_data, val_data, test_data = splitobs(dataset; at=(0.6, 0.2))

    @testset "FYL Training - Basic" begin
        model = generate_statistical_model(benchmark)
        maximizer = generate_maximizer(benchmark)

        # Test basic training runs without error
        history = fyl_train_model!(
            model, maximizer, train_data, val_data; epochs=3, callbacks=TrainingCallback[]
        )

        # Check that history is returned
        @test history isa MVHistory

        # Check that losses are tracked
        @test haskey(history, :training_loss)
        @test haskey(history, :validation_loss)

        # Check epochs (0-indexed: 0, 1, 2, 3)
        train_epochs, train_losses = get(history, :training_loss)
        @test length(train_epochs) == 4  # epoch 0 + 3 training epochs
        @test train_epochs[1] == 0
        @test train_epochs[end] == 3

        # Check that losses are Float64
        @test all(isa(l, Float64) for l in train_losses)

        val_epochs, val_losses = get(history, :validation_loss)
        @test length(val_epochs) == 4
        @test all(isa(l, Float64) for l in val_losses)
    end

    @testset "FYL Training - With Callbacks" begin
        model = generate_statistical_model(benchmark)
        maximizer = generate_maximizer(benchmark)

        # Create simple callbacks
        callbacks = [
            Metric(
                :gap, (data, ctx) -> compute_gap(benchmark, data, ctx.model, ctx.maximizer)
            ),
            Metric(:epoch, (data, ctx) -> ctx.epoch; on=:none),
        ]

        history = fyl_train_model!(
            model, maximizer, train_data, val_data; epochs=3, callbacks=callbacks
        )

        # Check callback metrics are recorded
        @test haskey(history, :val_gap)
        @test haskey(history, :epoch)

        # Check gap values exist
        gap_epochs, gap_values = get(history, :val_gap)
        @test length(gap_epochs) == 4  # epoch 0 + 3 epochs
        @test all(isa(g, AbstractFloat) for g in gap_values)

        # Check epoch tracking
        epoch_epochs, epoch_values = get(history, :epoch)
        @test epoch_values == [0, 1, 2, 3]
    end

    @testset "FYL Training - Callback on=:both" begin
        model = generate_statistical_model(benchmark)
        maximizer = generate_maximizer(benchmark)

        callbacks = [
            Metric(
                :gap,
                (data, ctx) -> compute_gap(benchmark, data, ctx.model, ctx.maximizer);
                on=:both,
            ),
        ]

        history = fyl_train_model!(
            model, maximizer, train_data, val_data; epochs=2, callbacks=callbacks
        )

        # Check both train and val metrics exist
        @test haskey(history, :train_gap)
        @test haskey(history, :val_gap)

        train_gap_epochs, train_gap_values = get(history, :train_gap)
        val_gap_epochs, val_gap_values = get(history, :val_gap)

        @test length(train_gap_epochs) == 3  # epoch 0, 1, 2
        @test length(val_gap_epochs) == 3
    end

    @testset "FYL Training - Context Fields" begin
        model = generate_statistical_model(benchmark)
        maximizer = generate_maximizer(benchmark)

        # Callback that checks context structure
        context_checker = Metric(
            :context_check,
            (data, ctx) -> begin
                # Check all required core fields exist
                @test haskey(ctx, :epoch)
                @test haskey(ctx, :model)
                @test haskey(ctx, :maximizer)
                @test haskey(ctx, :train_dataset)
                @test haskey(ctx, :validation_dataset)
                @test haskey(ctx, :train_loss)
                @test haskey(ctx, :val_loss)

                # Check types
                @test ctx.epoch isa Int
                @test ctx.train_loss isa Float64
                @test ctx.val_loss isa Float64

                return 1.0  # dummy value
            end;
            on=:none,
        )

        history = fyl_train_model!(
            model, maximizer, train_data, val_data; epochs=2, callbacks=[context_checker]
        )

        @test haskey(history, :context_check)
    end

    @testset "FYL Training - fyl_train_model (non-mutating)" begin
        initial_model = generate_statistical_model(benchmark)
        maximizer = generate_maximizer(benchmark)

        # Test non-mutating version
        history, trained_model = fyl_train_model(
            initial_model, maximizer, train_data, val_data; epochs=2
        )

        @test history isa MVHistory
        @test trained_model !== initial_model  # Should be a copy

        # Check history structure
        @test haskey(history, :training_loss)
        @test haskey(history, :validation_loss)
    end

    @testset "Callback Error Handling" begin
        model = generate_statistical_model(benchmark)
        maximizer = generate_maximizer(benchmark)

        # Create a callback that fails
        failing_callback = Metric(
            :failing, (data, ctx) -> begin
                error("Intentional error for testing")
            end
        )

        # Should not crash, just warn
        history = fyl_train_model!(
            model, maximizer, train_data, val_data; epochs=2, callbacks=[failing_callback]
        )

        # Training should complete
        @test history isa MVHistory
        @test haskey(history, :training_loss)

        # Failed metric should not be in history
        @test !haskey(history, :val_failing)
    end

    @testset "Multiple Callbacks" begin
        model = generate_statistical_model(benchmark)
        maximizer = generate_maximizer(benchmark)

        callbacks = [
            Metric(
                :gap, (data, ctx) -> compute_gap(benchmark, data, ctx.model, ctx.maximizer)
            ),
            Metric(:loss_ratio, (data, ctx) -> ctx.val_loss / ctx.train_loss; on=:none),
            Metric(:epoch_squared, (data, ctx) -> Float64(ctx.epoch^2); on=:none),
        ]

        history = fyl_train_model!(
            model, maximizer, train_data, val_data; epochs=3, callbacks=callbacks
        )

        # All metrics should be tracked
        @test haskey(history, :val_gap)
        @test haskey(history, :loss_ratio)
        @test haskey(history, :epoch_squared)

        # Check epoch_squared values
        _, epoch_sq_values = get(history, :epoch_squared)
        @test epoch_sq_values == [0.0, 1.0, 4.0, 9.0]
    end
end
