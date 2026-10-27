using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningAlgorithms: perturbed_layer
using DecisionFocusedLearningBenchmarks
using InferOpt: InferOpt, FenchelYoungLoss, PerturbedAdditive
using Random
using Statistics
using Test

# `Z ∼ N(0, I)` through `PerturbedLaw`: reference to check the layer against
# `PerturbedAdditive`. Production code keeps `PerturbedAdditive` for this law.
struct StandardNormalLaw <: AbstractPerturbationLaw end
DecisionFocusedLearningAlgorithms.rand_perturbation(::StandardNormalLaw, rng, θ, m; kwargs...) =
    randn(rng, length(θ), m)

@testset "PerturbedLaw" begin
    benchmark = ArgmaxBenchmark()
    maximizer = generate_maximizer(benchmark)
    θ = randn(MersenneTwister(1), 10)
    y_true = maximizer(randn(MersenneTwister(2), 10))
    kw = (; ε=0.7, nb_samples=20, seed=3)

    @testset "N(0, I) reproduces PerturbedAdditive" begin
        reference = PerturbedAdditive(maximizer; kw...)
        for law in (StandardNormalLaw(), GaussianCombinationLaw((θ; kwargs...) -> (zeros(length(θ)), zeros(length(θ), 0))))
            perturbed = PerturbedLaw(maximizer, law; kw...)
            @test perturbed(θ) == reference(θ)
            l_ref, g_ref = InferOpt.fenchel_young_loss_and_grad(
                FenchelYoungLoss(reference), θ, y_true
            )
            l, g = InferOpt.fenchel_young_loss_and_grad(FenchelYoungLoss(perturbed), θ, y_true)
            @test l ≈ l_ref
            @test g ≈ g_ref
        end
    end

    @testset "a mean shift is a shift of θ" begin
        μ = randn(MersenneTwister(4), 10)
        law = GaussianCombinationLaw((θ; kwargs...) -> (μ, zeros(length(θ), 0)))
        reference = PerturbedAdditive(maximizer; kw...)
        perturbed = PerturbedLaw(maximizer, law; kw...)
        @test perturbed(θ) ≈ reference(θ .+ kw.ε .* μ)
        _, g_ref = InferOpt.fenchel_young_loss_and_grad(
            FenchelYoungLoss(reference), θ .+ kw.ε .* μ, y_true
        )
        _, g = InferOpt.fenchel_young_loss_and_grad(FenchelYoungLoss(perturbed), θ, y_true)
        @test g ≈ g_ref
    end

    @testset "GaussianCombinationLaw: mean and covariance" begin
        μ = collect(1.0:4.0)
        V = [1.0 0.0; -1.0 0.0; 0.0 2.0; 0.0 0.0]
        law = GaussianCombinationLaw((θ; kwargs...) -> (μ, V))
        Z = rand_perturbation(law, MersenneTwister(0), zeros(4), 200_000)
        @test vec(mean(Z; dims=2)) ≈ μ atol = 0.02
        @test cov(Z; dims=2) ≈ [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1] + V * V' atol = 0.05
    end

    @testset "perturbed_layer" begin
        algorithm = PerturbedFenchelYoungLossImitation(; nb_samples=5, ε=0.5)
        @test perturbed_layer(algorithm, maximizer) isa InferOpt.PerturbedOracle
        with_law = PerturbedFenchelYoungLossImitation(;
            nb_samples=5, ε=0.5, perturbation_law=StandardNormalLaw()
        )
        layer = perturbed_layer(with_law, maximizer; ε=2.0)
        @test layer isa PerturbedLaw
        @test layer.ε == 2.0
        @test perturbed_layer(with_law, maximizer; law=nothing) isa InferOpt.PerturbedOracle
        @test perturbed_layer(algorithm, maximizer; law=StandardNormalLaw()) isa PerturbedLaw
        multiplicative = PerturbedFenchelYoungLossImitation(;
            perturbation_law=StandardNormalLaw(), use_multiplicative_perturbation=true
        )
        @test_throws ErrorException perturbed_layer(multiplicative, maximizer)
    end

    @testset "MirrorDescent relabelling law" begin
        inner = PerturbedFenchelYoungLossImitation(; perturbation_law=StandardNormalLaw())
        @test MirrorDescent(; inner_algorithm=inner).relabel_perturbation_law isa StandardNormalLaw
        @test isnothing(MirrorDescent().relabel_perturbation_law)
        @test isnothing(
            MirrorDescent(; inner_algorithm=inner, relabel_perturbation_law=nothing).relabel_perturbation_law,
        )
        @test DAggerMirrorDescent(; inner_algorithm=inner).relabel_perturbation_law isa StandardNormalLaw
    end

    @testset "training with a law" begin
        dataset = generate_dataset(benchmark, 20)
        policy = DFLPolicy(generate_statistical_model(benchmark), maximizer)
        algorithm = PerturbedFenchelYoungLossImitation(;
            nb_samples=5, perturbation_law=StandardNormalLaw()
        )
        history = train_policy!(algorithm, policy, dataset; epochs=3)
        _, losses = get(history, :training_loss)
        @test length(losses) == 4
        @test all(isfinite, losses)
    end
end
