#!/usr/bin/env julia

# Tiny test script for the Metal-optimized ray tracer
# Uses absolutely minimal settings for fastest testing

include("spira-metal-optimized.jl")

# Override main function with minimal settings
function tiny_test()
    width = 128  # Tiny resolution
    height = 72
    samples = 1  # Single sample
    max_depth = 1  # Just primary rays
    
    # Create camera
    aspect_ratio = Float32(width / height)
    lookfrom = Point3(Float32(0.0), Float32(1.0), Float32(3.0))
    lookat = Point3(Float32(0.0), Float32(0.0), Float32(0.0))
    vup = Point3(Float32(0.0), Float32(1.0), Float32(0.0))
    camera = Camera(lookfrom, lookat, vup, Float32(40.0), aspect_ratio)
    
    # Create scene
    scene = create_scene()
    
    println("Running tiny test render with Metal optimized ray tracer...")
    println("Resolution: $(width)x$(height), Samples: $samples, Depth: $max_depth")
    
    # Render
    render(scene, camera, width, height, 
          samples_per_pixel=samples, 
          max_depth=max_depth, 
          output_path="metal_tiny_test.png")
    
    println("Test complete! Check metal_tiny_test.png for results.")
end

# Run the test
tiny_test()