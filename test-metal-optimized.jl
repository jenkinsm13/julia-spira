#!/usr/bin/env julia

# Simple test script for the Metal-optimized ray tracer
# Uses minimal settings for quick testing

include("spira-metal-optimized.jl")

# Override main function with more minimal settings
function quick_test()
    width = 320  # Very small resolution for quick test
    height = 180
    samples = 4  # Minimal samples
    max_depth = 2  # Limited bounces
    
    # Create camera
    aspect_ratio = Float32(width / height)
    lookfrom = Point3(Float32(0.0), Float32(1.0), Float32(3.0))
    lookat = Point3(Float32(0.0), Float32(0.0), Float32(0.0))
    vup = Point3(Float32(0.0), Float32(1.0), Float32(0.0))
    camera = Camera(lookfrom, lookat, vup, Float32(40.0), aspect_ratio)
    
    # Create scene
    scene = create_scene()
    
    println("Running quick test render with Metal optimized ray tracer...")
    println("Resolution: $(width)x$(height), Samples: $samples, Depth: $max_depth")
    
    # Render
    render(scene, camera, width, height, 
          samples_per_pixel=samples, 
          max_depth=max_depth, 
          output_path="metal_test_render.png")
    
    println("Test complete! Check metal_test_render.png for results.")
end

# Run the test
quick_test()