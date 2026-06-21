using Aqua
using Documenter
using JET

using DecisionFocusedLearningAlgorithms

@testset "Aqua" begin
    Aqua.test_all(
        DecisionFocusedLearningAlgorithms;
        ambiguities=false,
        deps_compat=(check_extras = false),
    )
end

@testset "JET" begin
    JET.test_package(
        DecisionFocusedLearningAlgorithms;
        target_modules=[DecisionFocusedLearningAlgorithms],
    )
end

@testset "Documenter" begin
    Documenter.doctest(DecisionFocusedLearningAlgorithms)
end
