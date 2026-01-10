using DecisionFocusedLearningAlgorithms
using Documenter
using Literate

DocMeta.setdocmeta!(
    DecisionFocusedLearningAlgorithms,
    :DocTestSetup,
    :(begin
        using DecisionFocusedLearningAlgorithms
        using DecisionFocusedLearningBenchmarks
        using Flux
        using MLUtils
        using Plots
    end),
    recursive=true,
)

# Generate markdown files from tutorial scripts
tutorial_dir = joinpath(@__DIR__, "src", "tutorials")
tutorial_files = filter(f -> endswith(f, ".jl"), readdir(tutorial_dir))

# Convert .jl tutorial files to markdown
for file in tutorial_files
    filepath = joinpath(tutorial_dir, file)
    Literate.markdown(filepath, tutorial_dir; documenter=true, execute=false)
end

# Get list of generated markdown files for the docs
md_tutorial_files = [
    "tutorials/" * replace(file, ".jl" => ".md") for file in tutorial_files
]

makedocs(;
    modules=[DecisionFocusedLearningAlgorithms],
    authors="Members of JuliaDecisionFocusedLearning and contributors",
    sitename="DecisionFocusedLearningAlgorithms.jl",
    format=Documenter.HTML(;
        canonical="https://JuliaDecisionFocusedLearning.github.io/DecisionFocusedLearningAlgorithms.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md", "Tutorials" => md_tutorial_files],
)

deploydocs(;
    repo="github.com/JuliaDecisionFocusedLearning/DecisionFocusedLearningAlgorithms.jl",
    devbranch="main",
)
