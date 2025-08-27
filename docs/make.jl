using DecisionFocusedLearningAlgorithms
using Documenter

DocMeta.setdocmeta!(DecisionFocusedLearningAlgorithms, :DocTestSetup, :(using DecisionFocusedLearningAlgorithms); recursive=true)

makedocs(;
    modules=[DecisionFocusedLearningAlgorithms],
    authors="Members of JuliaDecisionFocusedLearning and contributors",
    sitename="DecisionFocusedLearningAlgorithms.jl",
    format=Documenter.HTML(;
        canonical="https://JuliaDecisionFocusedLearning.github.io/DecisionFocusedLearningAlgorithms.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaDecisionFocusedLearning/DecisionFocusedLearningAlgorithms.jl",
    devbranch="main",
)
