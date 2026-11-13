# Relabelled targets of mirror descent in the decision space (`:y`, historical) or in the
# feature space of the maximizer (`:g`, the moment the Fenchel-Young loss compares with).
using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks
using Test

const DFLA = DecisionFocusedLearningAlgorithms
const InferOpt_ = DFLA.InferOpt

# One item, y ∈ {0, …, 6}; g(y) = [y; −max(0, y − 4)] has a kink at level 4, like the
# staircase block of the dynamic replenishment maximizer.
toy_g(y; kwargs...) = [y[1], -max(0, y[1] - 4)]
const TOY_CANDIDATES = [[Float64(k)] for k in 0:6]
toy_argmax(θ; kwargs...) = TOY_CANDIDATES[argmax([sum(θ .* toy_g(c)) for c in TOY_CANDIDATES])]
const TOY_MAXIMIZER = InferOpt_.LinearMaximizer(toy_argmax; g=toy_g)

@testset "Relabelled target: decision mean (:y) or moment of g (:g)" begin
    ys = [[2.0], [6.0]]
    w = [0.5, 0.5]

    ty = DFLA._relabel_target(ys, w, :y, TOY_MAXIMIZER, (;))
    @test ty == [4.0]
    # what the loss makes of the historical target: g(ȳ) never uses the level above 4
    @test InferOpt_.apply_g(TOY_MAXIMIZER, ty) == [4.0, 0.0]

    tg = DFLA._relabel_target(ys, w, :g, TOY_MAXIMIZER, (;))
    @test tg isa DFLA.GSpaceTarget
    @test tg.g == [4.0, -1.0]          # E[g(y)]: the atoms use it half the time
    @test tg.h == 0.0
    @test collect(tg) == tg.g           # behaves as the vector E[g(y)]
    @test InferOpt_.apply_g(TOY_MAXIMIZER, tg) == tg.g
    θ = [0.3, 1.7]
    @test InferOpt_.objective_value(TOY_MAXIMIZER, θ, tg) ≈ sum(θ .* tg.g)
    @test InferOpt_.apply_g(toy_argmax, tg) == tg.g      # generic oracle (g = identity)

    # g linear (the identity of a plain `LinearMaximizer`): same numbers as the decision mean
    linear = InferOpt_.LinearMaximizer(toy_argmax)
    @test DFLA._relabel_target(ys, w, :g, linear, (;)).g ==
        DFLA._relabel_target(ys, w, :y, linear, (;))

    @test_throws ArgumentError DFLA._relabel_target(ys, w, :z, TOY_MAXIMIZER, (;))
    @test_throws ArgumentError DFLA._relabel_target(ys, w, :g, nothing, (;))
    @test_throws ArgumentError MirrorDescent(; target_space=:z) |>
        a -> DFLA._check_target_space(a.target_space)
end

@testset "Fenchel-Young losses read a GSpaceTarget as a target in g-space" begin
    # When all atoms agree the moment is g(y): loss and gradient must be those of y.
    θ = [0.8, 2.0]
    y = [5.0]
    t = DFLA.GSpaceTarget(collect(Float64, toy_g(y)), 0.0)
    additive = InferOpt_.FenchelYoungLoss(
        InferOpt_.PerturbedAdditive(TOY_MAXIMIZER; ε=0.5, nb_samples=4, seed=0)
    )
    l1, g1 = InferOpt_.fenchel_young_loss_and_grad(additive, θ, y)
    l2, g2 = InferOpt_.fenchel_young_loss_and_grad(additive, θ, t)
    @test l1 ≈ l2
    @test g1 ≈ g2
    @test additive(θ, y) ≈ additive(θ, t)

    # the layer used with a perturbation law (P / covone in DynamicStoreReplenishment)
    standard = GaussianCombinationLaw((θ; kwargs...) -> (zeros(length(θ)), zeros(length(θ), 0)))
    law = InferOpt_.FenchelYoungLoss(
        PerturbedLaw(TOY_MAXIMIZER, standard; ε=0.5, nb_samples=4, seed=0)
    )
    l3, g3 = InferOpt_.fenchel_young_loss_and_grad(law, θ, y)
    l4, g4 = InferOpt_.fenchel_young_loss_and_grad(law, θ, t)
    @test l3 ≈ l4
    @test g3 ≈ g4

    # Spread atoms: the target terms differ by ⟨θ, E[g(y)] − g(ȳ)⟩ (Jensen, here < 0 on the
    # kinked coordinate), the prediction terms are the same.
    ty = [4.0]
    tg = DFLA.GSpaceTarget([4.0, -1.0], 0.0)
    ly, gy = InferOpt_.fenchel_young_loss_and_grad(additive, θ, ty)
    lg, gg = InferOpt_.fenchel_young_loss_and_grad(additive, θ, tg)
    @test lg - ly ≈ -sum(θ .* ([4.0, -1.0] .- [4.0, 0.0]))
    @test gg - gy ≈ [4.0, 0.0] .- [4.0, -1.0]
end

@testset "MirrorDescent target_space = :g is :y when g is linear" begin
    # ContextualStochasticArgmax: linear oracle, so both target spaces give the same targets,
    # hence the same training (reseeded perturbations for reproducibility).
    benchmark = ContextualStochasticArgmaxBenchmark()
    train_md(space) = train_policy(
        MirrorDescent(;
            inner_algorithm=PerturbedFenchelYoungLossImitation(; seed=0, threaded=false),
            target_space=space,
        ),
        benchmark;
        dataset_size=5,
        epochs=2,
        iterations=2,
        seed=0,
    )
    hy, _ = train_md(:y)
    hg, _ = train_md(:g)
    for (a, b) in zip(hy, hg)
        # `get` of ValueHistories (loaded by DecisionFocusedLearningAlgorithms): (epochs, values)
        @test get(a, :training_loss)[2] ≈ get(b, :training_loss)[2]
    end
end

@testset "Perturbations: seed reset at every call (historical) or dedicated rng" begin
    using Random: Random, Xoshiro
    identity_oracle(θ; kwargs...) = θ
    algo(; kw...) = PerturbedFenchelYoungLossImitation(; nb_samples=3, ε=1.0, threaded=false, kw...)
    atoms(layer, θ) =
        InferOpt_.DifferentiableExpectations.empirical_predistribution(layer.reinforce, θ).atoms

    # historical: `seed` resets the rng before every call, so every call draws the same Z
    seeded = DFLA.perturbed_layer(algo(; seed=0), identity_oracle)
    @test atoms(seeded, zeros(4)) == atoms(seeded, zeros(4))
    @test atoms(seeded, fill(10.0, 4)) .- Ref(fill(10.0, 4)) ≈ atoms(seeded, zeros(4))

    # dedicated rng: fresh draws at every call, reproducible from the rng's seed
    fresh = DFLA.perturbed_layer(algo(; rng=Xoshiro(0)), identity_oracle)
    b1 = atoms(fresh, zeros(4))
    @test atoms(fresh, zeros(4)) != b1
    @test atoms(DFLA.perturbed_layer(algo(; rng=Xoshiro(0)), identity_oracle), zeros(4)) == b1
    # ... and the global rng is left alone
    Random.seed!(7); r1 = rand()
    Random.seed!(7); fresh(zeros(4)); r2 = rand()
    @test r1 == r2

    # the same holds for a layer with a perturbation law (P / covone)
    standard = GaussianCombinationLaw((θ; kwargs...) -> (zeros(length(θ)), zeros(length(θ), 0)))
    with_law = DFLA.perturbed_layer(algo(; rng=Xoshiro(1), perturbation_law=standard), identity_oracle)
    @test DFLA._perturbed_parameters(with_law, zeros(4)).atoms !=
        DFLA._perturbed_parameters(with_law, zeros(4)).atoms

    # mirror descent's relabelling layer shares the inner algorithm's rng: no reset either
    md = MirrorDescent(; inner_algorithm=algo(; rng=Xoshiro(2)))
    relabel = DFLA._perturbed_parametric_solver(
        ContextualStochasticArgmaxBenchmark(), (θ, scenario; kwargs...) -> θ, md, 1.0
    )
    @test DFLA._relabel_atoms(relabel, zeros(4)).atoms != DFLA._relabel_atoms(relabel, zeros(4)).atoms
end
