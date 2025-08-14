# Simple test script to render a scene with Stanford bunny
include("julia-raytracer.jl")

function download_bunny_if_needed()
    bunny_path = expanduser("~/Downloads/bunny.obj")
    
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

# Ensure we have the Stanford bunny
if download_bunny_if_needed()
    # Set up parameters for a quick render to test OBJ loading
    width = 640
    height = 360
    samples = 50
    output_file = "bunny_render.exr"
    
    # Create a scene with the bunny
    println("Creating scene with Stanford bunny...")
    scene, camera = create_scene_with_obj()
    
    # Render the scene
    println("Rendering bunny scene with $samples samples per pixel...")
    render_example(
        width=width,
        height=height,
        samples=samples,
        interactive=true,
        output_file=output_file,
        scene=scene,
        camera=camera
    )
else
    println("Cannot render without the bunny model.")
end