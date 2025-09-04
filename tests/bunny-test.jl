# Simple test script to render a scene with Stanford bunny
# Run Plots headless during tests
ENV["GKSwstype"] = "100"

# Include the example raytracer relative to this file
include(joinpath(@__DIR__, "../examples/julia-raytracer.jl"))
using Test

function download_bunny_if_needed()
    bunny_path = expanduser("~/Downloads/bunny.obj")
    mkpath(dirname(bunny_path))
    
    # Check if the bunny file already exists
    if !isfile(bunny_path)
        println("Stanford bunny model not found, downloading it...")
        # URL for the Stanford bunny from Stanford's repository
        bunny_url = "https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj"
        
        try
            download(bunny_url, bunny_path)
            println("Downloaded Stanford bunny to $bunny_path")
        catch e
            println("Failed to download bunny: $e")
            println("Please manually download the Stanford bunny OBJ from:")
            println("https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj")
            println("And save it as $bunny_path")
            return false
        end
    else
        println("Stanford bunny model found at $bunny_path")
    end
    
    return true
end

# Run a small render of the high‑poly bunny and verify output
function run_bunny_render_test()
    width = 64
    height = 64
    samples = 1
    output_file = ""

    # Create a scene with the bunny mesh
    println("Creating scene with Stanford bunny...")
    scene, camera = create_scene_with_obj()

    # Render the scene without displaying it
    println("Rendering bunny scene with $samples sample per pixel...")
    image, _ = render_example(
        width=width,
        height=height,
        samples=samples,
        interactive=false,
        output_file=output_file,
        scene=scene,
        camera=camera,
    )

    return size(image) == (height, width)
end

@testset "Bunny render" begin
    @test download_bunny_if_needed()
    @test run_bunny_render_test()
end

