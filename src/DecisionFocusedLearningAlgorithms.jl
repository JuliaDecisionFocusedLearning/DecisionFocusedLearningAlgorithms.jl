module DecisionFocusedLearningAlgorithms

using DecisionFocusedLearningBenchmarks
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Flux: Flux, Adam
using InferOpt: InferOpt, FenchelYoungLoss, PerturbedAdditive
using MLUtils: splitobs
using ProgressMeter: @showprogress
using Statistics: mean
using UnicodePlots: lineplot
using ValueHistories: MVHistory

include("utils.jl")
include("training_context.jl")

include("metrics/interface.jl")
include("metrics/accumulators.jl")
include("metrics/function_metric.jl")
include("metrics/periodic.jl")

include("algorithms/abstract_algorithm.jl")
include("algorithms/supervised/fyl.jl")
include("algorithms/supervised/kleopatra.jl")
include("algorithms/supervised/dagger.jl")

export TrainingContext

export AbstractMetric,
    FYLLossMetric,
    FunctionMetric,
    PeriodicMetric,
    LossAccumulator,
    reset!,
    update!,
    evaluate!,
    compute!,
    evaluate_metrics!

export fyl_train_model, kleopatra_train_model, DAgger_train_model!, DAgger_train_model
export PerturbedImitationAlgorithm, train_policy!

end
