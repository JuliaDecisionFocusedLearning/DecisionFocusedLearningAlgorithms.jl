using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks
using Test
using ValueHistories
using Statistics: mean

@testset "MirrorDescent Training" begin
    @testset "MirrorDescent - ContextualStochasticArgmax basic" begin
        benchmark = ContextualStochasticArgmaxBenchmark()
        algorithm = MirrorDescent()

        histories, policy = train_policy(
            algorithm, benchmark; dataset_size=5, epochs=2, iterations=2, seed=0
        )

        @test histories isa Vector
        @test length(histories) == 2
        @test all(h isa MVHistory for h in histories)
        @test all(haskey(h, :training_loss) for h in histories)
        @test policy isa DFLPolicy
    end

    @testset "MirrorDescent - StochasticVehicleScheduling basic" begin
        benchmark = StochasticVehicleSchedulingBenchmark()
        algorithm = MirrorDescent()

        histories, policy = train_policy(
            algorithm, benchmark; dataset_size=1, epochs=2, iterations=2, seed=0
        )

        @test histories isa Vector
        @test length(histories) == 2
        @test all(h isa MVHistory for h in histories)
        @test all(haskey(h, :training_loss) for h in histories)
        @test policy isa DFLPolicy
    end

    @testset "MirrorDescent - imitation_start=false" begin
        benchmark = ContextualStochasticArgmaxBenchmark()
        algorithm = MirrorDescent()

        histories, policy = train_policy(
            algorithm,
            benchmark;
            dataset_size=5,
            epochs=2,
            iterations=2,
            seed=0,
            imitation_start=false,
        )

        @test histories isa Vector
        @test length(histories) == 2
        @test policy isa DFLPolicy
    end

    @testset "MirrorDescent - performance improves over iterations" begin
        benchmark = ContextualStochasticArgmaxBenchmark()
        algorithm = MirrorDescent()

        val_dataset = generate_dataset(benchmark, 100; seed=99)

        val_metric = FunctionMetric(:val_obj, val_dataset) do ctx, data
            vals = map(data) do s
                θ = ctx.policy.statistical_model(s.x)
                y = ctx.policy.maximizer(θ; s.context...)
                Float64(DecisionFocusedLearningBenchmarks.objective_value(benchmark, s, y))
            end
            (val_obj=mean(vals),)
        end

        histories, policy = train_policy(
            algorithm,
            benchmark;
            dataset_size=20,
            epochs=3,
            iterations=5,
            seed=0,
            metrics=(val_metric,),
        )

        val_objs = [get(histories[i], :val_obj)[2][end] for i in 1:5]

        # Performance should improve at each iteration
        @test (val_objs[4] > val_objs[1])
    end

    @testset "MirrorDescent - with metrics" begin
        benchmark = ContextualStochasticArgmaxBenchmark()
        algorithm = MirrorDescent()

        metrics = (FunctionMetric(ctx -> ctx.epoch, :epoch),)

        histories, policy = train_policy(
            algorithm,
            benchmark;
            dataset_size=5,
            epochs=2,
            iterations=2,
            seed=0,
            metrics=metrics,
        )

        @test all(haskey(h, :epoch) for h in histories)
    end
end
