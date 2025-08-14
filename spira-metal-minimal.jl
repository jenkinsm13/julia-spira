#!/usr/bin/env julia

# SPIRA - Super minimal Metal-based ray tracer
# Just implements single-bounce rendering with minimal complexity

using LinearAlgebra
using Random
using Images
using StaticArrays
using FileIO
using Base.Threads

println("SPIRA - Minimal Metal Raytracer")
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

# Constants - using Float32 constructor as required
const INF = Float32(1e20)
const EPS = Float32(1e-6)
const BLACK = Vec3(Float32(0.0), Float32(0.0), Float32(0.0))
const WHITE = Vec3(Float32(1.0), Float32(1.0), Float32(1.0))

# Ray structure (for CPU rendering)
struct Ray
    origin::Point3
    direction::Vec3
end

# Simple sphere primitive
struct Sphere
    center::Point3
    radius::Float32
    color::Color  # Simplified - just a color instead of material
end

# Create a simple scene with spheres
function create_scene()
    spheres = [
        # Main sphere (red)
        Sphere(
            Point3(Float32(0.0), Float32(0.0), Float32(0.0)),
            Float32(0.5),
            Vec3(Float32(0.8), Float32(0.2), Float32(0.2))
        ),
        
        # Ground (green)
        Sphere(
            Point3(Float32(0.0), Float32(-100.5), Float32(0.0)),
            Float32(100.0),
            Vec3(Float32(0.2), Float32(0.8), Float32(0.2))
        ),
        
        # Right sphere (blue)
        Sphere(
            Point3(Float32(1.0), Float32(0.0), Float32(0.0)),
            Float32(0.5),
            Vec3(Float32(0.2), Float32(0.2), Float32(0.8))
        ),
        
        # Left sphere (white)
        Sphere(
            Point3(Float32(-1.0), Float32(0.0), Float32(0.0)),
            Float32(0.5),
            Vec3(Float32(0.8), Float32(0.8), Float32(0.8))
        )
    ]
    
    return spheres
end

# Prepare camera view parameters
function setup_camera(width::Int, height::Int)
    aspect_ratio = Float32(width / height)
    viewport_height = Float32(2.0)
    viewport_width = aspect_ratio * viewport_height
    focal_length = Float32(1.0)
    
    origin = Point3(Float32(0.0), Float32(0.0), Float32(3.0))
    horizontal = Vec3(viewport_width, Float32(0.0), Float32(0.0))
    vertical = Vec3(Float32(0.0), viewport_height, Float32(0.0))
    lower_left_corner = origin - horizontal/Float32(2.0) - vertical/Float32(2.0) - Vec3(Float32(0.0), Float32(0.0), focal_length)
    
    return (origin, lower_left_corner, horizontal, vertical)
end

# Prepare sphere data for GPU
function prepare_spheres_for_gpu(spheres)
    # Each sphere needs 7 values: center_x, center_y, center_z, radius, color_r, color_g, color_b
    sphere_data = zeros(Float32, 7 * length(spheres))
    
    for (i, sphere) in enumerate(spheres)
        base_idx = (i-1) * 7 + 1
        sphere_data[base_idx] = sphere.center[1]    # center.x
        sphere_data[base_idx+1] = sphere.center[2]  # center.y
        sphere_data[base_idx+2] = sphere.center[3]  # center.z
        sphere_data[base_idx+3] = sphere.radius     # radius
        sphere_data[base_idx+4] = sphere.color[1]   # color.r
        sphere_data[base_idx+5] = sphere.color[2]   # color.g
        sphere_data[base_idx+6] = sphere.color[3]   # color.b
    end
    
    return sphere_data
end

# Quick CPU render - single ray per pixel, no anti-aliasing
function render_cpu(width::Int, height::Int, spheres)
    # Setup camera
    origin, lower_left_corner, horizontal, vertical = setup_camera(width, height)
    
    # Create image buffer
    img = Array{RGB{Float32}}(undef, height, width)
    
    # For each pixel, trace a ray
    for j in 1:height
        for i in 1:width
            # Calculate ray direction
            u = Float32((i - 1) / (width - 1))
            v = Float32((j - 1) / (height - 1))
            ray_dir = normalize(lower_left_corner + u*horizontal + v*vertical - origin)
            
            # Trace ray
            hit_anything = false
            closest_t = INF
            color = BLACK
            
            # Check each sphere
            for sphere in spheres
                # Calculate ray-sphere intersection
                oc = origin - sphere.center
                a = dot(ray_dir, ray_dir)
                half_b = dot(oc, ray_dir)
                c = dot(oc, oc) - sphere.radius * sphere.radius
                discriminant = half_b * half_b - a * c
                
                if discriminant > 0
                    # Calculate closest intersection
                    sqrtd = sqrt(discriminant)
                    t = (-half_b - sqrtd) / a
                    
                    if t < Float32(0.001)
                        t = (-half_b + sqrtd) / a
                    end
                    
                    if t > Float32(0.001) && t < closest_t
                        closest_t = t
                        hit_anything = true
                        
                        # Simple diffuse-like shading
                        hit_point = origin + t * ray_dir
                        normal = normalize(hit_point - sphere.center)
                        
                        # Lighting factor (simple diffuse with light from above)
                        light_dir = normalize(Vec3(Float32(0.0), Float32(1.0), Float32(0.0)))
                        light_factor = max(Float32(0.2), dot(normal, light_dir))
                        
                        # Set color
                        color = sphere.color * light_factor
                    end
                end
            end
            
            if !hit_anything
                # Sky gradient background
                t = Float32(0.5) * (ray_dir[2] + Float32(1.0))
                color = (Float32(1.0) - t) * WHITE + t * Vec3(Float32(0.5), Float32(0.7), Float32(1.0))
            end
            
            # Set pixel color
            img[height-j+1, i] = RGB{Float32}(color...)
        end
        
        # Print progress
        if j % 20 == 0
            println("Rendered $j/$height rows")
        end
    end
    
    return img
end

# Metal kernel for ray tracing
function render_kernel(buffer, width, height, origin_x, origin_y, origin_z,
                       llc_x, llc_y, llc_z, horiz_x, horiz_y, horiz_z,
                       vert_x, vert_y, vert_z, spheres, num_spheres)
    
    # Get current pixel coordinates
    i = thread_position_in_grid_1d() % width + 1
    j = thread_position_in_grid_1d() ÷ width + 1
    
    # Skip if outside image bounds
    if i > width || j > height
        return
    end
    
    # Calculate ray direction
    u = Float32((i - 1) / (width - 1))
    v = Float32((j - 1) / (height - 1))
    
    # Ray origin
    ray_origin_x = origin_x
    ray_origin_y = origin_y
    ray_origin_z = origin_z
    
    # Ray direction components
    dir_x = llc_x + u * horiz_x + v * vert_x - origin_x
    dir_y = llc_y + u * horiz_y + v * vert_y - origin_y
    dir_z = llc_z + u * horiz_z + v * vert_z - origin_z
    
    # Normalize direction
    dir_len = sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)
    dir_x /= dir_len
    dir_y /= dir_len
    dir_z /= dir_len
    
    # Initialize variables for closest hit
    hit_anything = false
    closest_t = Float32(1e20)
    hit_color_r = Float32(0.0)
    hit_color_g = Float32(0.0)
    hit_color_b = Float32(0.0)
    hit_normal_x = Float32(0.0)
    hit_normal_y = Float32(0.0)
    hit_normal_z = Float32(0.0)
    
    # Check each sphere
    for s in 1:num_spheres
        # Extract sphere data
        base_idx = (s-1) * 7 + 1
        sphere_x = spheres[base_idx]
        sphere_y = spheres[base_idx+1]
        sphere_z = spheres[base_idx+2]
        radius = spheres[base_idx+3]
        color_r = spheres[base_idx+4]
        color_g = spheres[base_idx+5]
        color_b = spheres[base_idx+6]
        
        # Ray-sphere intersection
        oc_x = ray_origin_x - sphere_x
        oc_y = ray_origin_y - sphere_y
        oc_z = ray_origin_z - sphere_z
        
        a = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z
        half_b = oc_x*dir_x + oc_y*dir_y + oc_z*dir_z
        c = oc_x*oc_x + oc_y*oc_y + oc_z*oc_z - radius*radius
        discriminant = half_b*half_b - a*c
        
        if discriminant > Float32(0.0)
            # Calculate intersection
            sqrtd = sqrt(discriminant)
            t = (-half_b - sqrtd) / a
            
            if t < Float32(0.001)
                t = (-half_b + sqrtd) / a
            end
            
            if t > Float32(0.001) && t < closest_t
                closest_t = t
                hit_anything = true
                
                # Calculate hit point and normal
                hit_x = ray_origin_x + t * dir_x
                hit_y = ray_origin_y + t * dir_y
                hit_z = ray_origin_z + t * dir_z
                
                hit_normal_x = (hit_x - sphere_x) / radius
                hit_normal_y = (hit_y - sphere_y) / radius
                hit_normal_z = (hit_z - sphere_z) / radius
                
                # Store color
                hit_color_r = color_r
                hit_color_g = color_g
                hit_color_b = color_b
            end
        end
    end
    
    # Determine pixel color
    if hit_anything
        # Simple diffuse-like shading
        light_dir_x = Float32(0.0)
        light_dir_y = Float32(1.0)
        light_dir_z = Float32(0.0)
        
        # Normalize light direction
        light_len = sqrt(light_dir_x*light_dir_x + light_dir_y*light_dir_y + light_dir_z*light_dir_z)
        light_dir_x /= light_len
        light_dir_y /= light_len
        light_dir_z /= light_len
        
        # Calculate lighting factor (dot product with light)
        light_factor = hit_normal_x*light_dir_x + hit_normal_y*light_dir_y + hit_normal_z*light_dir_z
        if light_factor < Float32(0.2)
            light_factor = Float32(0.2)
        end
        
        # Apply light factor to color
        buffer[j, i, 1] = hit_color_r * light_factor
        buffer[j, i, 2] = hit_color_g * light_factor
        buffer[j, i, 3] = hit_color_b * light_factor
    else
        # Sky gradient background
        t = Float32(0.5) * (dir_y + Float32(1.0))
        buffer[j, i, 1] = (Float32(1.0) - t) + t * Float32(0.5)
        buffer[j, i, 2] = (Float32(1.0) - t) + t * Float32(0.7)
        buffer[j, i, 3] = (Float32(1.0) - t) + t * Float32(1.0)
    end
    
    return
end

# A simple Metal approach to render the image
function render_metal(width::Int, height::Int, spheres)
    if !has_metal
        println("Metal not available, falling back to CPU rendering")
        return render_cpu(width, height, spheres)
    end
    
    # Setup camera
    origin, lower_left_corner, horizontal, vertical = setup_camera(width, height)
    
    # Prepare data
    sphere_data = prepare_spheres_for_gpu(spheres)
    sphere_data_gpu = Metal.MtlArray(sphere_data)
    num_spheres = length(spheres)
    
    # Create image buffer
    img_buffer = Metal.zeros(Float32, height, width, 3)
    
    # Run Metal kernel
    println("Running Metal kernel...")
    
    # Launch kernel with 1D grid for simplicity
    total_pixels = width * height
    threads = 256  # Standard thread block size
    groups = cld(total_pixels, threads)
    
    @metal threads=threads groups=groups render_kernel(
        img_buffer, width, height,
        origin[1], origin[2], origin[3],
        lower_left_corner[1], lower_left_corner[2], lower_left_corner[3],
        horizontal[1], horizontal[2], horizontal[3],
        vertical[1], vertical[2], vertical[3],
        sphere_data_gpu, num_spheres
    )
    
    # Wait for kernel completion
    Metal.synchronize()
    
    # Convert GPU buffer to CPU image
    img_cpu = Array(img_buffer)
    
    # Create the final RGB image
    img = Array{RGB{Float32}}(undef, height, width)
    for j in 1:height
        for i in 1:width
            img[j, i] = RGB{Float32}(img_cpu[j, i, 1], img_cpu[j, i, 2], img_cpu[j, i, 3])
        end
    end
    
    return img
end

# Main function
function main()
    width = 480  # Small resolution for quick testing
    height = 270
    
    # Create scene
    spheres = create_scene()
    
    # Time rendering
    start_time = time()
    
    # Choose renderer based on availability
    if has_metal
        println("Rendering with Metal GPU...")
        img = render_metal(width, height, spheres)
    else
        println("Rendering with CPU...")
        img = render_cpu(width, height, spheres)
    end
    
    # Report timing
    elapsed = time() - start_time
    println("Render completed in $(round(elapsed, digits=2)) seconds")
    
    # Save the result
    save("metal_minimal_render.png", img)
    println("Saved render to metal_minimal_render.png")
end

# Run the main function if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end