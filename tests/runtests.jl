using Test
using SPIRA

@testset "SPIRA.jl" begin
    # Include all test files
    include("test_core.jl")
    include("test_render.jl")
end
