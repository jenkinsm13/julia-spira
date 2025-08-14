#!/usr/bin/env julia

# SPIRA - Ultra simple Metal GPU ray tracer
# This version uses Metal.jl's array operations for minimal operations only

using LinearAlgebra
using Random
using Images
using StaticArrays
using FileIO
using Base.Threads

println("SPIRA - Ultra Simple Metal Raytracer")
println("Julia version: ", VERSION)

# Try to load Metal.jl
has_metal = false
try
    using Metal
    global has_metal = true
    println("Metal.jl loaded successfully!")
    dev = Metal.device()
    println("Metal device: $dev")
catch e
    println("Metal.jl not available: $e")
    println("Will use CPU rendering only")
end

# Type aliases for better performance
const Vec3 = SVector{3, Float32}
const Point3 = Vec3
const Color = Vec3

# Constants
const BLACK = Vec3(Float32(0.0), Float32(0.0), Float32(0.0))
const WHITE = Vec3(Float32(1.0), Float32(1.0), Float32(1.0))

# Simple sphere primitive
struct Sphere
    center::Point3
    radius::Float32
    color::Color
end

# Create a simple scene with spheres
function create_scene()
    spheres = [
        # Main sphere (red)
        Sphere(
            Point3(Float32(0.0), Float32(0.0), Float32(-1.0)),
            Float32(0.5),
            Vec3(Float32(0.8), Float32(0.2), Float32(0.2))
        ),
        
        # Ground (green)
        Sphere(
            Point3(Float32(0.0), Float32(-100.5), Float32(-1.0)),
            Float32(100.0),
            Vec3(Float32(0.2), Float32(0.8), Float32(0.2))
        )
    ]
    
    return spheres
end

# Super simple ray-sphere intersection on CPU
function cpu_render(width, height)
    # Initialize image
    img = Array{RGB{Float32}}(undef, height, width)
    
    # Camera setup
    aspect_ratio = width / height
    viewport_height = Float32(2.0)
    viewport_width = Float32(aspect_ratio * viewport_height)
    focal_length = Float32(1.0)
    
    origin = Point3(Float32(0.0), Float32(0.0), Float32(0.0))
    horizontal = Vec3(viewport_width, Float32(0.0), Float32(0.0))
    vertical = Vec3(Float32(0.0), viewport_height, Float32(0.0))
    lower_left = origin - horizontal/Float32(2.0) - vertical/Float32(2.0) - Vec3(Float32(0.0), Float32(0.0), focal_length)
    
    # Get spheres
    spheres = create_scene()
    
    # For each pixel
    for j in 1:height
        for i in 1:width
            # Calculate ray direction
            u = Float32((i - 1) / (width - 1))
            v = Float32((j - 1) / (height - 1))
            dir = normalize(lower_left + u*horizontal + v*vertical - origin)
            
            # Initialize pixel color to background
            color = BLACK
            
            # Trace ray against all spheres
            hit_anything = false
            closest_t = Float32(Inf)
            
            for sphere in spheres
                # Ray-sphere intersection
                oc = origin - sphere.center
                a = 1.0f0  # dir is normalized
                half_b = dot(oc, dir)
                c = dot(oc, oc) - sphere.radius * sphere.radius
                discriminant = half_b * half_b - a * c
                
                if discriminant > 0
                    # Find closest intersection
                    sqrtd = sqrt(discriminant)
                    t = (-half_b - sqrtd) / a
                    
                    if t <= 0.001f0
                        t = (-half_b + sqrtd) / a
                    end
                    
                    if t > 0.001f0 && t < closest_t
                        closest_t = t
                        hit_anything = true
                        
                        # Calculate normal and basic lighting (from above)
                        hit_point = origin + t * dir
                        normal = normalize(hit_point - sphere.center)
                        light_factor = max(0.2f0, dot(normal, Vec3(0.0f0, 1.0f0, 0.0f0)))
                        
                        # Set color
                        color = sphere.color * light_factor
                    end
                end
            end
            
            if !hit_anything
                # Sky gradient for background
                t = 0.5f0 * (dir[2] + 1.0f0)
                color = (1.0f0 - t) * WHITE + t * Vec3(0.5f0, 0.7f0, 1.0f0)
            end
            
            # Set pixel
            img[height-j+1, i] = RGB{Float32}(color[1], color[2], color[3])
        end
    end
    
    return img
end

# Super simple Metal test - just fill a buffer with gradient
function metal_render(width, height)
    if !has_metal
        println("Metal not available, falling back to CPU rendering")
        return cpu_render(width, height)
    end
    
    # Test Metal by just filling a gradient buffer
    println("Using Metal to generate a simple image...")
    
    # Create a buffer on the GPU
    buffer = Metal.MtlArray{Float32}(undef, height, width, 3)
    
    # Fill with a simple gradient
    for j in 1:height
        for i in 1:width
            u = Float32((i - 1) / (width - 1))
            v = Float32((j - 1) / (height - 1))
            
            buffer[j, i, 1] = u          # Red increases from left to right
            buffer[j, i, 2] = v          # Green increases from top to bottom
            buffer[j, i, 3] = 0.5f0      # Blue constant
        end
    end
    
    # Copy buffer back to CPU
    img_data = Array(buffer)
    
    # Create RGB image
    img = Array{RGB{Float32}}(undef, height, width)
    for j in 1:height
        for i in 1:width
            img[j, i] = RGB{Float32}(
                img_data[j, i, 1],
                img_data[j, i, 2],
                img_data[j, i, 3]
            )
        end
    end
    
    return img
end

# Main function
function main()
    width = 640
    height = 360
    
    # Time rendering
    start_time = time()
    
    # Render using Metal or CPU
    if has_metal
        println("Rendering with Metal GPU...")
        img = metal_render(width, height)
    else
        println("Rendering with CPU...")
        img = cpu_render(width, height)
    end
    
    # Report timing
    elapsed = time() - start_time
    println("Render completed in $(round(elapsed, digits=2)) seconds")
    
    # Save the result
    output_file = "metal_simple_render.png"
    save(output_file, img)
    println("Saved render to $output_file")
end

# Run the main function if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end