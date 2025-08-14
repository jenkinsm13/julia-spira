#!/usr/bin/env julia

# Basic SPIRA rendering example
# This demonstrates how to use the SPIRA raytracer

using Pkg
Pkg.activate("..")

using SPIRA

function main()
    println("SPIRA - Basic Rendering Example")
    
    # Set up rendering parameters
    width = 640
    height = 360
    samples_per_pixel = 16
    max_depth = 4
    
    # Create camera
    aspect_ratio = Float32(width / height)
    lookfrom = Point3(Float32(0.0), Float32(1.0), Float32(3.0))
    lookat = Point3(Float32(0.0), Float32(0.0), Float32(0.0))
    vup = Point3(Float32(0.0), Float32(1.0), Float32(0.0))
    camera = Camera(lookfrom, lookat, vup, Float32(40.0), aspect_ratio)
    
    # Create scene
    scene = create_scene()
    
    # Render
    println("Starting render...")
    img = render(scene, camera, width, height, 
                samples_per_pixel=samples_per_pixel, 
                max_depth=max_depth, 
                output_path="../assets/images/basic_render.png")
    
    println("Render complete!")
    return img
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
