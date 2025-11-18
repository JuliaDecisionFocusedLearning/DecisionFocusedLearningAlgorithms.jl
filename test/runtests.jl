using TestItemRunner

@testsnippet Imports begin
    using DecisionFocusedLearningAlgorithms
    using DecisionFocusedLearningBenchmarks
    using MLUtils: splitobs
    using Random
    using ValueHistories
end

@run_package_tests verbose = true

# using DecisionFocusedLearningAlgorithms
# using Test
# using Aqua
# using JET
# using JuliaFormatter

# @testset "DecisionFocusedLearningAlgorithms.jl" begin
#     @testset "Code quality (Aqua.jl)" begin
#         Aqua.test_all(
#             DecisionFocusedLearningAlgorithms;
#             ambiguities=false,
#             deps_compat=(check_extras = false),
#         )
#     end
#     @testset "Code linting (JET.jl)" begin
#         JET.test_package(DecisionFocusedLearningAlgorithms; target_defined_modules=true)
#     end
#     @testset "Code formatting (JuliaFormatter.jl)" begin
#         @test JuliaFormatter.format(
#             DecisionFocusedLearningAlgorithms; verbose=false, overwrite=false
#         )
#     end

#     # Training and callback tests
#     include("training_tests.jl")
# end
