module DecisionFocusedLearningAlgorithms

using DecisionFocusedLearningBenchmarks
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Flux: Flux, Adam
using InferOpt: InferOpt, FenchelYoungLoss, PerturbedAdditive, PerturbedMultiplicative
using MLUtils: splitobs, DataLoader
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

include("policies/abstract_policy.jl")
include("policies/dfl_policy.jl")

include("algorithms/abstract_algorithm.jl")
include("algorithms/supervised/fyl.jl")
include("algorithms/supervised/anticipative_imitation.jl")
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

export PerturbedFenchelYoungLossImitation,
    DAgger, AnticipativeImitation, train_policy!, train_policy
export AbstractPolicy, DFLPolicy

end
