using DecisionFocusedLearningAlgorithms
using Documenter
using Literate

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
    joinpath("tutorials", replace(file, ".jl" => ".md")) for file in tutorial_files
]

makedocs(;
    modules=[DecisionFocusedLearningAlgorithms],
    authors="Members of JuliaDecisionFocusedLearning and contributors",
    sitename="DecisionFocusedLearningAlgorithms.jl",
    format=Documenter.HTML(; size_threshold=typemax(Int)),
    pages=[
        "Home" => "index.md",
        "Interface Guide" => "interface.md",
        "Tutorials" => md_tutorial_files,
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaDecisionFocusedLearning/DecisionFocusedLearningAlgorithms.jl",
    devbranch="main",
)

for file in md_tutorial_files
    rm(joinpath(@__DIR__, "src", file))
end
