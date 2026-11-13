using Test
using DecisionFocusedLearningAlgorithms

@testset "DecisionFocusedLearningAlgorithms tests" begin
    @testset "Code quality" begin
        include("code.jl")
    end

    @testset "FYL" begin
        include("fyl.jl")
    end

    @testset "Perturbation law" begin
        include("perturbation_law.jl")
    end

    @testset "DAgger" begin
        include("dagger.jl")
    end

    @testset "MirrorDescent" begin
        include("mirror_descent.jl")
    end

    @testset "MirrorDescent relabelled targets" begin
        include("relabel_target.jl")
    end
end
