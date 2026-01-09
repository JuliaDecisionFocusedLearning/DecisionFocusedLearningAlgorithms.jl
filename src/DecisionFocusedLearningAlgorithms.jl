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
# include("dfl_policy.jl")
# include("callbacks.jl")
include("metric.jl")
include("fyl.jl")
include("dagger.jl")

export fyl_train_model!,
    fyl_train_model, baty_train_model, DAgger_train_model!, DAgger_train_model
export TrainingCallback, Metric, on_epoch_end, get_metric_names, run_callbacks!
export TrainingContext, update_context

export AbstractMetric,
    FYLLossMetric, FunctionMetric, LossAccumulator, reset!, update!, evaluate!, compute
export PerturbedImitationAlgorithm, train_policy!

end
