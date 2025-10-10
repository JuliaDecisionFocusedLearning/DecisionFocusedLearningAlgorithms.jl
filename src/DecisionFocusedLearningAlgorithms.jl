module DecisionFocusedLearningAlgorithms

using DecisionFocusedLearningBenchmarks
const DVSP = DecisionFocusedLearningBenchmarks.DynamicVehicleScheduling
using Flux: Flux, Adam
using InferOpt: InferOpt, FenchelYoungLoss, PerturbedAdditive
using MLUtils: splitobs
using ProgressMeter: @showprogress
using Statistics: mean
using UnicodePlots: lineplot
using ValueHistories: MVHistory

include("callbacks.jl")
include("utils/metrics.jl")
include("fyl_new.jl")
include("dagger.jl")

export fyl_train_model!,
    fyl_train_model, baty_train_model, DAgger_train_model!, DAgger_train_model
export TrainingCallback, Metric, on_epoch_end

end
