using DecisionFocusedLearningAlgorithms
using Test
using Aqua
using JET
using JuliaFormatter

@testset "DecisionFocusedLearningAlgorithms.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(
            DecisionFocusedLearningAlgorithms;
            ambiguities=false,
            deps_compat=(check_extras = false),
        )
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(DecisionFocusedLearningAlgorithms; target_defined_modules=true)
    end
    # Write your tests here.
    @testset "Code formatting (JuliaFormatter.jl)" begin
        @test JuliaFormatter.format(
            DecisionFocusedLearningAlgorithms; verbose=false, overwrite=false
        )
    end
end
