using DecisionFocusedLearningAlgorithms
using Documenter

DocMeta.setdocmeta!(
    DecisionFocusedLearningAlgorithms,
    :DocTestSetup,
    :(using DecisionFocusedLearningAlgorithms);
    recursive=true,
)

tutorial_dir = joinpath(@__DIR__, "src", "tutorials")

include_tutorial = true

if include_tutorial
    for file in tutorial_files
        filepath = joinpath(tutorial_dir, file)
        Literate.markdown(filepath, md_dir; documenter=true, execute=false)
    end
end

makedocs(;
    modules=[DecisionFocusedLearningAlgorithms],
    authors="Members of JuliaDecisionFocusedLearning and contributors",
    sitename="DecisionFocusedLearningAlgorithms.jl",
    format=Documenter.HTML(;
        canonical="https://JuliaDecisionFocusedLearning.github.io/DecisionFocusedLearningAlgorithms.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md", "Tutorials" => include_tutorial ? md_tutorial_files : []],
)

deploydocs(;
    repo="github.com/JuliaDecisionFocusedLearning/DecisionFocusedLearningAlgorithms.jl",
    devbranch="main",
)
