"""
$TYPEDEF

Law of the perturbation `Z` in `θ + εZ`, for laws InferOpt cannot express: InferOpt's
`PerturbedAdditive` draws `Z ∼ N(0, I)`, and its sampler only sees `θ`. A standard normal
`Z` should keep going through `PerturbedAdditive` (`perturbation_law = nothing`); a law
is only needed for anything else.

A law may depend on the instance: subtypes implement
[`rand_perturbation`](@ref)`(law, rng, θ, nb_samples; kwargs...)`, which receives the
keyword arguments of the maximizer (e.g. the state of a dynamic benchmark).
"""
abstract type AbstractPerturbationLaw end

"""
    rand_perturbation(law, rng, θ, nb_samples; kwargs...) -> Matrix

Draw `nb_samples` i.i.d. realizations of `Z`, one per column of a
`length(θ) × nb_samples` matrix. `kwargs` are those passed to the maximizer.
"""
function rand_perturbation end

"""
$TYPEDEF

Gaussian law built as a combination of independent standard Gaussian vectors:

    Z = μ + Z₀ + Σₖ ξₖ vₖ,   Z₀ ∼ N(0, I),   ξₖ ∼ N(0, 1)   ⇒   Z ∼ N(μ, I + V Vᵀ)

where `(μ, V) = components(θ; kwargs...)`, `V` holding the vectors `vₖ` (scales included)
as columns. The identity part `Z₀` is always there, so that `Z` has a positive density
whatever `V` (rank deficient included): the perturbed Fenchel-Young loss stays valid.

`Z₀` is drawn first, in one `randn(rng, d, nb_samples)` block like `PerturbedAdditive`,
then the `ξ`; nothing else is drawn when `V` has no column.

# Fields
$TYPEDFIELDS
"""
struct GaussianCombinationLaw{F} <: AbstractPerturbationLaw
    "`(θ; kwargs...) -> (μ, V)`: mean vector and `d × K` matrix of directions"
    components::F
end

function rand_perturbation(law::GaussianCombinationLaw, rng, θ, nb_samples; kwargs...)
    μ, V = law.components(θ; kwargs...)
    Z = randn(rng, length(θ), nb_samples)
    Z .+= μ
    size(V, 2) > 0 && (Z .+= V * randn(rng, size(V, 2), nb_samples))
    return Z
end

"""
$TYPEDEF

Perturbed optimizer `θ ↦ E[f(θ + εZ)]` for an arbitrary [`AbstractPerturbationLaw`](@ref)
of `Z`, compatible with `FenchelYoungLoss`.

The Fenchel-Young loss of an additive perturbation does not depend on the law of `Z`:
`F(θ) = E[max_y ⟨θ + εZ, g(y)⟩]` has gradient `E[g(y*(θ + εZ))]` (Danskin) as soon as `Z`
has a density, which is what InferOpt computes for `PerturbedAdditive`. Only the sampling
differs, hence this layer rather than a fork of InferOpt.

Sampling mimics `PerturbedAdditive`: the rng is reseeded with `seed` at every call (common
random numbers across calls), and the atoms are mapped with the same threaded helper.

# Fields
$TYPEDFIELDS
"""
struct PerturbedLaw{F,L<:AbstractPerturbationLaw,R<:Random.AbstractRNG,S} <:
       InferOpt.AbstractOptimizationLayer
    "maximizer `(θ; kwargs...) -> y`"
    maximizer::F
    "law of the perturbation `Z`"
    law::L
    "perturbation size"
    ε::Float64
    "number of perturbation samples"
    nb_samples::Int
    "whether to map the samples over threads"
    threaded::Bool
    "seed of `rng`, reset at every call (`nothing`: never reset)"
    seed::S
    "random number generator"
    rng::R
end

function PerturbedLaw(
    maximizer,
    law::AbstractPerturbationLaw;
    ε=1.0,
    nb_samples=1,
    threaded=false,
    seed=nothing,
    rng=Random.default_rng(),
)
    return PerturbedLaw(maximizer, law, Float64(ε), nb_samples, threaded, seed, rng)
end

InferOpt.get_maximizer(perturbed::PerturbedLaw) = perturbed.maximizer

# Uniform distribution over the perturbed parameters `θ + εZₘ`, `m = 1..nb_samples`.
function _perturbed_parameters(perturbed::PerturbedLaw, θ; kwargs...)
    (; law, ε, nb_samples, threaded, seed, rng) = perturbed
    isnothing(seed) || Random.seed!(rng, seed)
    Z = rand_perturbation(law, rng, θ, nb_samples; kwargs...)
    size(Z) == (length(θ), nb_samples) || throw(
        DimensionMismatch(
            "rand_perturbation returned a $(size(Z)) matrix, expected " *
            "$((length(θ), nb_samples))",
        ),
    )
    ηs = [θ .+ ε .* Z[:, m] for m in 1:nb_samples]
    return FixedAtomsProbabilityDistribution(ηs; threaded)
end

function (perturbed::PerturbedLaw)(θ::AbstractArray; kwargs...)
    η_dist = _perturbed_parameters(perturbed, θ; kwargs...)
    return mean(map(η -> perturbed.maximizer(η; kwargs...), η_dist))
end

function InferOpt.fenchel_young_loss_and_grad(
    fyl::FenchelYoungLoss{<:PerturbedLaw},
    θ::AbstractArray,
    y_true::AbstractArray;
    kwargs...,
)
    perturbed = fyl.optimization_layer
    maximizer = perturbed.maximizer
    η_dist = _perturbed_parameters(perturbed, θ; kwargs...)
    y_dist = map(η -> maximizer(η; kwargs...), η_dist)
    F = mean(
        InferOpt.objective_value(maximizer, η, y; kwargs...) for
        (η, y) in zip(η_dist.atoms, y_dist.atoms)
    )
    ŷ = mean(y -> InferOpt.apply_g(maximizer, y; kwargs...), y_dist)
    l = F - InferOpt.objective_value(maximizer, θ, y_true; kwargs...)
    g = ŷ - InferOpt.apply_g(maximizer, y_true; kwargs...)
    return l, g
end
