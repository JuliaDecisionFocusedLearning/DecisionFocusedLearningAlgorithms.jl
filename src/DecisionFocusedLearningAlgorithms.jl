module DecisionFocusedLearningAlgorithms

using DecisionFocusedLearningBenchmarks
const DVSP = DecisionFocusedLearningBenchmarks.DynamicVehicleScheduling
using Flux: Flux, Adam
using InferOpt: InferOpt, FenchelYoungLoss, PerturbedAdditive
using MLUtils: splitobs
using ProgressMeter: @showprogress
using UnicodePlots: lineplot

include("utils/metrics.jl")
include("fyl.jl")
include("dagger.jl")

export fyl_train_model!,
    fyl_train_model, baty_train_model, DAgger_train_model!, DAgger_train_model

end
