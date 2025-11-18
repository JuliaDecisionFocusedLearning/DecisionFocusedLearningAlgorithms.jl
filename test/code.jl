@testitem "Aqua" begin
    using Aqua
    Aqua.test_all(
        DecisionFocusedLearningAlgorithms;
        ambiguities=false,
        deps_compat=(check_extras = false),
    )
end

@testitem "JET" begin
    using DecisionFocusedLearningAlgorithms
    using JET
    JET.test_package(DecisionFocusedLearningAlgorithms; target_defined_modules=true)
end

@testitem "JuliaFormatter" begin
    using DecisionFocusedLearningAlgorithms
    using JuliaFormatter
    @test JuliaFormatter.format(
        DecisionFocusedLearningAlgorithms; verbose=false, overwrite=false
    )
end

@testitem "Documenter" begin
    using DecisionFocusedLearningAlgorithms
    using Documenter

    Documenter.doctest(DecisionFocusedLearningAlgorithms)
end
