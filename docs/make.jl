using ReduceWindows
using Documenter

DocMeta.setdocmeta!(ReduceWindows, :DocTestSetup, :(using ReduceWindows); recursive=true)

makedocs(;
    modules=[ReduceWindows],
    authors="Jan Weidner <jw3126@gmail.com> and contributors",
    repo="https://github.com/jw3126/ReduceWindows.jl/blob/{commit}{path}#{line}",
    sitename="ReduceWindows.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jw3126.github.io/ReduceWindows.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jw3126/ReduceWindows.jl",
    devbranch="main",
)
