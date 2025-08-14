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

# Include all source files
include("spira-metal-optimized.jl")

end # module
