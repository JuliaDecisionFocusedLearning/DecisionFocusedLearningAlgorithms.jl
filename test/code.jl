using Aqua
using Documenter
using JET
using JuliaFormatter

using DecisionFocusedLearningAlgorithms

@testset "Aqua" begin
    Aqua.test_all(
        DecisionFocusedLearningAlgorithms;
        ambiguities=false,
        deps_compat=(check_extras = false),
    )
end

@testset "JET" begin
    JET.test_package(DecisionFocusedLearningAlgorithms; target_defined_modules=true)
end

@testset "JuliaFormatter" begin
    @test JuliaFormatter.format(
        DecisionFocusedLearningAlgorithms; verbose=false, overwrite=false
    )
end

@testset "Documenter" begin
    Documenter.doctest(DecisionFocusedLearningAlgorithms)
end
