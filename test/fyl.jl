
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
            model, maximizer, train_data, val_data; epochs=3, metrics=()
        )

        # Check that history is returned
        @test history isa MVHistory

        # Check that training loss is tracked
        @test haskey(history, :training_loss)

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

    @testset "FYL Training - With Metrics" begin
        model = generate_statistical_model(benchmark)
        maximizer = generate_maximizer(benchmark)

        # Create loss metric using FenchelYoungLoss
        using InferOpt: FenchelYoungLoss, PerturbedAdditive
        perturbed = PerturbedAdditive(maximizer; nb_samples=10, ε=0.1)
        loss = FenchelYoungLoss(perturbed)
        val_loss_metric = FYLLossMetric(loss, val_data, :validation_loss)

        # Create custom function metrics
        epoch_metric = FunctionMetric(:epoch, ctx -> ctx.epoch)

        # Create metric with stored data
        gap_metric = FunctionMetric(:val_gap, val_data) do ctx, data
            compute_gap(benchmark, data, ctx.model, ctx.maximizer)
        end

        metrics = (val_loss_metric, epoch_metric, gap_metric)

        history = fyl_train_model!(
            model, maximizer, train_data, val_data; epochs=3, metrics=metrics
        )

        # Check metrics are recorded
        @test haskey(history, :validation_loss)
        @test haskey(history, :epoch)
        @test haskey(history, :val_gap)

        # Check validation loss values
        val_epochs, val_values = get(history, :validation_loss)
        @test length(val_epochs) == 4  # epoch 0 + 3 epochs
        @test all(isa(v, AbstractFloat) for v in val_values)

        # Check epoch tracking
        epoch_epochs, epoch_values = get(history, :epoch)
        @test epoch_values == [0, 1, 2, 3]

        # Check gap tracking
        gap_epochs, gap_values = get(history, :val_gap)
        @test length(gap_epochs) == 4
        @test all(isa(g, AbstractFloat) for g in gap_values)
    end

    @testset "FYL Training - Context Fields" begin
        model = generate_statistical_model(benchmark)
        maximizer = generate_maximizer(benchmark)

        # Metric that checks context structure
        context_checker = FunctionMetric(
            :context_check, (ctx) -> begin
                # Check required core fields exist
                @test hasproperty(ctx, :epoch)
                @test hasproperty(ctx, :model)
                @test hasproperty(ctx, :maximizer)

                # Check types
                @test ctx.epoch isa Int
                @test ctx.model !== nothing
                @test ctx.maximizer isa Function

                return 1.0  # dummy value
            end
        )

        history = fyl_train_model!(
            model, maximizer, train_data, val_data; epochs=2, metrics=(context_checker,)
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
    end

    @testset "Multiple Metrics" begin
        model = generate_statistical_model(benchmark)
        maximizer = generate_maximizer(benchmark)

        metrics = (FunctionMetric(:epoch_squared, ctx -> Float64(ctx.epoch^2)),)

        history = fyl_train_model!(
            model, maximizer, train_data, val_data; epochs=3, metrics=metrics
        )

        # Metric should be tracked
        @test haskey(history, :epoch_squared)

        # Check epoch_squared values
        _, epoch_sq_values = get(history, :epoch_squared)
        @test epoch_sq_values == [0.0, 1.0, 4.0, 9.0]
    end
end
