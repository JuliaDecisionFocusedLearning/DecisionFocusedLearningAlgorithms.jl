using Test
using DecisionFocusedLearningAlgorithms

@testset "DecisionFocusedLearningAlgorithms tests" begin
    @testset "Code quality" begin
        include("code.jl")
    end

    @testset "FYL" begin
        include("fyl.jl")
    end

    @testset "DAgger" begin
        include("dagger.jl")
    end
end
