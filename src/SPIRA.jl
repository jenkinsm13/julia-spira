module SPIRA

using LinearAlgebra
using Random
using Images
using StaticArrays
using FileIO
using Base.Threads

# Export main types and functions
export Scene, Camera, Material, Sphere, Ray
export render, create_scene
export render_hybrid_gpu, render_with_cpu

# Include GPU-accelerated implementation only when Metal is available
if Base.find_package("Metal") !== nothing
    include("spira-metal-optimized.jl")
else
    @info "Metal.jl not found; GPU rendering disabled"
end

end # module
