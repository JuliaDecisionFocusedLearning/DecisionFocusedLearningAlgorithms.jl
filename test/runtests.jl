using DecisionFocusedLearningAlgorithms
using Test
using Aqua
using JET

@testset "DecisionFocusedLearningAlgorithms.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(DecisionFocusedLearningAlgorithms)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(DecisionFocusedLearningAlgorithms; target_defined_modules = true)
    end
    # Write your tests here.
end
