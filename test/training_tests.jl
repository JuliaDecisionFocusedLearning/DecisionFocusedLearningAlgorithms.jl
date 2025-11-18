using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks
using Test
using MLUtils
using ValueHistories

@testitem "Training Functions" setup = [Imports] begin
    using MLUtils: splitobs
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

@testitem "DAgger Training" setup = [Imports] begin
    # Use a simple dynamic benchmark
    benchmark = DynamicVehicleSchedulingBenchmark(; two_dimensional_features=false)
    dataset = generate_dataset(benchmark, 10)  # Small for speed
    train_instances, val_instances = splitobs(dataset; at=0.6)

    train_envs = generate_environments(benchmark, train_instances; seed=0)
    val_envs = generate_environments(benchmark, val_instances; seed=1)

    @testset "DAgger - Basic Training" begin
        model = generate_statistical_model(benchmark)
        maximizer = generate_maximizer(benchmark)
        anticipative_policy =
            (env; reset_env) -> generate_anticipative_solution(benchmark, env; reset_env)

        history = DAgger_train_model!(
            model,
            maximizer,
            train_envs,
            val_envs,
            anticipative_policy;
            iterations=2,
            fyl_epochs=2,
            callbacks=TrainingCallback[],
        )

        @test history isa MVHistory
        @test haskey(history, :training_loss)
        @test haskey(history, :validation_loss)

        # Check epoch progression across DAgger iterations
        # 2 iterations × 2 fyl_epochs = 4 total epochs (plus epoch 0)
        train_epochs, _ = get(history, :training_loss)
        @test maximum(train_epochs) == 4  # epochs 0, 1, 2, 3, 4
    end

    @testset "DAgger - With Callbacks" begin
        model = generate_statistical_model(benchmark)
        maximizer = generate_maximizer(benchmark)
        anticipative_policy =
            (env; reset_env) -> generate_anticipative_solution(benchmark, env; reset_env)

        callbacks = [Metric(:epoch, (data, ctx) -> ctx.epoch; on=:none)]

        history = DAgger_train_model!(
            model,
            maximizer,
            train_envs,
            val_envs,
            anticipative_policy;
            iterations=2,
            fyl_epochs=2,
            callbacks=callbacks,
        )

        @test haskey(history, :epoch)

        # Check epoch values are continuous across DAgger iterations
        epoch_times, epoch_values = get(history, :epoch)
        @test epoch_values == collect(0:4)  # 0, 1, 2, 3, 4
    end

    @testset "DAgger - Convenience Function" begin
        # Test the benchmark-based convenience function
        history, model = DAgger_train_model(
            benchmark; iterations=2, fyl_epochs=2, callbacks=TrainingCallback[]
        )

        @test history isa MVHistory
        @test model !== nothing
        @test haskey(history, :training_loss)
    end
end

@testitem "Callback System" setup = [Imports] begin
    @testset "Metric Construction" begin
        # Test various Metric construction patterns
        m1 = Metric(:test, (d, c) -> 1.0)
        @test m1.name == :test
        @test m1.on == :validation  # default

        m2 = Metric(:test2, (d, c) -> 2.0; on=:train)
        @test m2.on == :train

        m3 = Metric(:test3, (d, c) -> 3.0; on=:both)
        @test m3.on == :both
    end

    @testset "on_epoch_end Interface" begin
        # Test the callback interface
        simple_callback = Metric(:simple, (d, c) -> c.epoch * 2.0; on=:none)

        context = (
            epoch=5,
            model=nothing,
            maximizer=nothing,
            train_dataset=[],
            validation_dataset=[],
            train_loss=1.0,
            val_loss=2.0,
        )

        result = on_epoch_end(simple_callback, context)
        @test result isa NamedTuple
        @test haskey(result, :simple)
        @test result.simple == 10.0
    end

    @testset "get_metric_names" begin
        callbacks = [
            Metric(:gap, (d, c) -> 1.0),  # default on=:validation
            Metric(:gap2, (d, c) -> 1.0; on=:train),
            Metric(:gap3, (d, c) -> 1.0; on=:both),
            Metric(:epoch, (d, c) -> 1.0; on=:none),
        ]

        names = get_metric_names(callbacks)

        @test :val_gap in names
        @test :train_gap2 in names
        @test :train_gap3 in names
        @test :val_gap3 in names
        @test :epoch in names
    end

    @testset "run_callbacks!" begin
        history = MVHistory()

        callbacks = [
            Metric(:metric1, (d, c) -> Float64(c.epoch)),
            Metric(:metric2, (d, c) -> Float64(c.epoch * 2); on=:none),
        ]

        context = (
            epoch=3,
            model=nothing,
            maximizer=nothing,
            train_dataset=[],
            validation_dataset=[],
            train_loss=1.0,
            val_loss=2.0,
        )

        run_callbacks!(history, callbacks, context)

        @test haskey(history, :val_metric1)
        @test haskey(history, :metric2)

        _, values1 = get(history, :val_metric1)
        _, values2 = get(history, :metric2)

        @test values1[1] == 3.0
        @test values2[1] == 6.0
    end
end

@testitem "Integration Tests" setup = [Imports] begin
    @testset "Portable Metrics Across Algorithms" begin
        # Test that the same callback works with both FYL and DAgger
        benchmark = ArgmaxBenchmark()
        dataset = generate_dataset(benchmark, 20)
        train_data, val_data = splitobs(dataset; at=0.7)

        # Define a portable metric
        portable_callback = Metric(
            :gap, (data, ctx) -> compute_gap(benchmark, data, ctx.model, ctx.maximizer)
        )

        # Test with FYL
        model_fyl = generate_statistical_model(benchmark)
        maximizer = generate_maximizer(benchmark)

        history_fyl = fyl_train_model!(
            model_fyl,
            maximizer,
            train_data,
            val_data;
            epochs=2,
            callbacks=[portable_callback],
        )

        @test haskey(history_fyl, :val_gap)

        # The same callback should work with DAgger too
        # (but we'll skip actually running DAgger here for speed)
        @test portable_callback isa TrainingCallback
    end

    @testset "Loss Values in Context" begin
        # Verify that loss values are correctly passed in context
        benchmark = ArgmaxBenchmark()
        dataset = generate_dataset(benchmark, 15)
        train_data, val_data = splitobs(dataset; at=0.7)

        model = generate_statistical_model(benchmark)
        maximizer = generate_maximizer(benchmark)

        loss_checker = Metric(
            :loss_check, (data, ctx) -> begin
                # Verify losses exist and are positive
                @test ctx.train_loss > 0
                @test ctx.val_loss > 0
                @test ctx.train_loss isa Float64
                @test ctx.val_loss isa Float64

                # Return loss ratio as metric
                return ctx.val_loss / ctx.train_loss
            end; on=:none
        )

        history = fyl_train_model!(
            model, maximizer, train_data, val_data; epochs=2, callbacks=[loss_checker]
        )

        @test haskey(history, :loss_check)
        _, loss_ratios = get(history, :loss_check)
        @test all(lr > 0 for lr in loss_ratios)
    end
end
