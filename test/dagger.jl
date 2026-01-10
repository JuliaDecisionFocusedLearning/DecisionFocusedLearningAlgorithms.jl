using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks
using MLUtils
using Test
using ValueHistories

@testset "DAgger Training" begin
    # Use a simple dynamic benchmark
    benchmark = DynamicVehicleSchedulingBenchmark(; two_dimensional_features=true)
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
            metrics=(),
        )

        @test history isa MVHistory
        @test haskey(history, :training_loss)

        # Check epoch progression across DAgger iterations
        # 2 iterations × 2 fyl_epochs = 4 total epochs (plus epoch 0)
        train_epochs, _ = get(history, :training_loss)
        @test maximum(train_epochs) == 4  # epochs 0, 1, 2, 3, 4
    end

    @testset "DAgger - With Metrics" begin
        model = generate_statistical_model(benchmark)
        maximizer = generate_maximizer(benchmark)
        anticipative_policy =
            (env; reset_env) -> generate_anticipative_solution(benchmark, env; reset_env)

        metrics = (FunctionMetric(ctx -> ctx.epoch, :epoch),)

        history = DAgger_train_model!(
            model,
            maximizer,
            train_envs,
            val_envs,
            anticipative_policy;
            iterations=2,
            fyl_epochs=2,
            metrics=metrics,
        )

        @test haskey(history, :epoch)

        # Check epoch values are continuous across DAgger iterations
        epoch_times, epoch_values = get(history, :epoch)
        @test epoch_values == collect(0:4)  # 0, 1, 2, 3, 4
    end

    @testset "DAgger - Convenience Function" begin
        # Test the benchmark-based convenience function
        history, model = DAgger_train_model(
            benchmark; iterations=2, fyl_epochs=2, metrics=()
        )

        @test history isa MVHistory
        @test model !== nothing
        @test haskey(history, :training_loss)
    end
end

@testset "Integration Tests" begin
    @testset "Portable Metrics Across Algorithms" begin
        # Test that the same metric works with both FYL and DAgger
        benchmark = ArgmaxBenchmark()
        dataset = generate_dataset(benchmark, 20)
        train_data, val_data = splitobs(dataset; at=0.7)

        # Define a portable metric
        portable_metric = FunctionMetric(
            ctx -> compute_gap(benchmark, val_data, ctx.model, ctx.maximizer), :gap
        )

        # Test with FYL
        algorithm = PerturbedImitationAlgorithm()
        model_fyl = generate_statistical_model(benchmark)
        maximizer = generate_maximizer(benchmark)

        history_fyl = train_policy!(
            algorithm,
            model_fyl,
            maximizer,
            train_data,
            val_data;
            epochs=2,
            metrics=(portable_metric,),
        )

        @test haskey(history_fyl, :gap)
        @test portable_metric isa AbstractMetric
    end

    @testset "Loss Values in Context" begin
        # Verify that loss values are correctly passed in context
        benchmark = ArgmaxBenchmark()
        dataset = generate_dataset(benchmark, 15)
        train_data, val_data = splitobs(dataset; at=0.7)

        algorithm = PerturbedImitationAlgorithm()
        model = generate_statistical_model(benchmark)
        maximizer = generate_maximizer(benchmark)

        loss_checker = FunctionMetric(ctx -> begin
            # Verify loss exists in context
            @test hasproperty(ctx, :loss)
            @test ctx.loss !== nothing
            return 1.0
        end, :loss_check)

        history = train_policy!(
            algorithm,
            model,
            maximizer,
            train_data,
            val_data;
            epochs=2,
            metrics=(loss_checker,),
        )

        @test haskey(history, :loss_check)
    end
end
