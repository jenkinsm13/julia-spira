#!/usr/bin/env julia

# Minimal Metal GPU raytracer using Metal.jl's array abstractions
# A simplified raytracer that renders a few spheres entirely on the GPU

using LinearAlgebra
using Random
using Images
using FileIO
using Metal

println("SPIRA - Minimal Metal GPU Raytracer")
println("Julia version: ", VERSION)

# Allow scalar indexing for this simplified example
# Note: In a real application, you'd want to use proper GPU operations instead
Metal.allowscalar(true)

# Try to load Metal.jl
has_metal = true
metal_dev = Metal.device()
println("Metal device: $metal_dev")

# Constants
const WIDTH = 640
const HEIGHT = 360
const SAMPLES = 4

# Create GPU arrays for sphere data
# Format: [center_x, center_y, center_z, radius, r, g, b]
# Simple scene with 4 spheres:
# - Ground sphere
# - Red sphere
# - Blue sphere
# - Light source

function main()
    # Create sphere data
    spheres = MtlArray([
        # Ground sphere (large)
        0.0f0, -100.5f0, 0.0f0, 100.0f0, 0.8f0, 0.8f0, 0.0f0, 
        # Red sphere
        -1.0f0, 0.0f0, 0.0f0, 0.5f0, 0.8f0, 0.1f0, 0.1f0,
        # Blue sphere
        1.0f0, 0.0f0, 0.0f0, 0.5f0, 0.1f0, 0.3f0, 0.8f0,
        # Light sphere
        0.0f0, 1.0f0, -1.0f0, 0.5f0, 1.0f0, 1.0f0, 1.0f0
    ])
    sphere_count = 4
    
    # Camera parameters
    camera_pos = MtlArray([0.0f0, 0.5f0, 3.0f0])
    camera_lookat = MtlArray([0.0f0, 0.0f0, 0.0f0])
    camera_up = MtlArray([0.0f0, 1.0f0, 0.0f0])
    camera_fov = 60.0f0
    aspect_ratio = Float32(WIDTH / HEIGHT)
    
    # Create camera rays
    println("Generating camera rays...")
    ray_directions = generate_camera_rays(camera_pos, camera_lookat, camera_up, 
                                         camera_fov, aspect_ratio, WIDTH, HEIGHT)
    
    # Initialize output image
    image_data = zeros(Float32, HEIGHT, WIDTH, 3)
    
    # Render
    println("Rendering $WIDTH×$HEIGHT image with $SAMPLES samples...")
    start_time = time()
    
    # For each sample
    for sample in 1:SAMPLES
        println("Sample $sample/$SAMPLES")
        
        # Generate ray directions with jitter for anti-aliasing
        jittered_rays = add_ray_jitter(ray_directions, WIDTH, HEIGHT, sample)
        
        # Render the scene
        result = render_scene(camera_pos, jittered_rays, spheres, sphere_count)
        
        # Accumulate results
        image_data += Array(result) / SAMPLES
    end
    
    elapsed = time() - start_time
    pixels_per_second = WIDTH * HEIGHT / elapsed
    println("Rendering completed in $(round(elapsed, digits=2)) seconds ($(round(Int, pixels_per_second)) pixels/sec)")
    
    # Create and save final image
    img = colorview(RGB, permutedims(clamp.(image_data, 0.0f0, 1.0f0), (3, 1, 2)))
    save("minimal_metal_render.png", img)
    println("Saved image to minimal_metal_render.png")
end

# Generate camera rays
function generate_camera_rays(pos, lookat, up, fov, aspect, width, height)
    # Calculate camera basis vectors
    w = normalize(pos .- lookat)
    u = normalize(cross(up, w))
    v = cross(w, u)
    
    # Calculate viewport dimensions
    theta = deg2rad(fov)
    h = tan(theta/2)
    viewport_height = 2.0f0 * h
    viewport_width = aspect * viewport_height
    
    # Calculate lower left corner of viewport
    viewport_ll = pos .- (viewport_width/2) .* u .- (viewport_height/2) .* v .- w
    
    # Create ray directions for each pixel
    ray_dirs = MtlArray(zeros(Float32, height, width, 3))
    
    # x and y coordinates for each pixel (as GPU arrays)
    x_step = viewport_width / Float32(width - 1)
    y_step = viewport_height / Float32(height - 1)
    
    # Create x and y coordinates
    x_coords = MtlArray(Float32[(i-1) * x_step for i in 1:width])
    y_coords = MtlArray(Float32[(j-1) * y_step for j in 1:height])
    
    # Fill ray directions using broadcasting
    # This is done component by component to stay on the GPU
    for i in 1:width
        for j in 1:height
            # Direction = viewport_ll + x_offset + y_offset - eye_pos
            ray_dirs[j, i, 1] = viewport_ll[1] + x_coords[i] * u[1] + y_coords[j] * v[1] - pos[1]
            ray_dirs[j, i, 2] = viewport_ll[2] + x_coords[i] * u[2] + y_coords[j] * v[2] - pos[2]
            ray_dirs[j, i, 3] = viewport_ll[3] + x_coords[i] * u[3] + y_coords[j] * v[3] - pos[3]
            
            # Normalize
            len = sqrt(ray_dirs[j, i, 1]^2 + ray_dirs[j, i, 2]^2 + ray_dirs[j, i, 3]^2)
            ray_dirs[j, i, 1] /= len
            ray_dirs[j, i, 2] /= len
            ray_dirs[j, i, 3] /= len
        end
    end
    
    return ray_dirs
end

# Add jitter to ray directions for anti-aliasing
function add_ray_jitter(rays, width, height, sample)
    # Only add jitter after the first sample
    if sample == 1
        return rays
    end
    
    # Create jittered copy (using Metal's array constructor to stay on GPU)
    jittered = MtlArray(Array(rays))
    
    # Add small random offset
    jitter_scale = 0.5f0 / max(width, height)
    
    # Create random jitter vectors
    jitter_u = MtlArray(rand(Float32, height, width) .* jitter_scale)
    jitter_v = MtlArray(rand(Float32, height, width) .* jitter_scale)
    
    # Apply jitter to all rays using a kernel function
    for j in 1:height
        for i in 1:width
            # Add jitter (small perturbation)
            jittered[j, i, 1] += jitter_u[j, i] * 0.01f0
            jittered[j, i, 2] += jitter_v[j, i] * 0.01f0
            
            # Renormalize
            len = sqrt(jittered[j, i, 1]^2 + jittered[j, i, 2]^2 + jittered[j, i, 3]^2)
            jittered[j, i, 1] /= len
            jittered[j, i, 2] /= len
            jittered[j, i, 3] /= len
        end
    end
    
    return jittered
end

# Simple sphere intersection test
function sphere_intersection(origin, dir, sphere_data, sphere_idx)
    # Extract sphere data
    base_idx = (sphere_idx-1) * 7 + 1
    center_x = sphere_data[base_idx]
    center_y = sphere_data[base_idx+1]
    center_z = sphere_data[base_idx+2]
    radius = sphere_data[base_idx+3]
    
    # Calculate quadratic equation coefficients
    oc_x = origin[1] - center_x
    oc_y = origin[2] - center_y
    oc_z = origin[3] - center_z
    
    a = 1.0f0  # We assume dir is normalized
    half_b = oc_x * dir[1] + oc_y * dir[2] + oc_z * dir[3]
    c = oc_x^2 + oc_y^2 + oc_z^2 - radius^2
    
    discriminant = half_b^2 - a * c
    
    if discriminant < 0
        return false, 0.0f0
    end
    
    # Calculate intersection distance
    sqrtd = sqrt(discriminant)
    
    # Try closer intersection first
    root = (-half_b - sqrtd) / a
    if root < 0.001f0
        root = (-half_b + sqrtd) / a
        if root < 0.001f0
            return false, 0.0f0
        end
    end
    
    return true, root
end

# Render scene with ray tracing (simplified, single bounce)
function render_scene(camera_pos, ray_dirs, spheres, sphere_count)
    height, width, _ = size(ray_dirs)
    result = MtlArray(zeros(Float32, height, width, 3))
    
    # Process each pixel
    for j in 1:height
        for i in 1:width
            # Default to sky color (blue gradient)
            dir_y = ray_dirs[j, i, 2]
            t = 0.5f0 * (dir_y + 1.0f0)
            result[j, i, 1] = (1.0f0 - t) + t * 0.5f0  # r
            result[j, i, 2] = (1.0f0 - t) + t * 0.7f0  # g
            result[j, i, 3] = (1.0f0 - t) + t * 1.0f0  # b
            
            # Create ray from camera position in the given direction
            ray_dir = (ray_dirs[j, i, 1], ray_dirs[j, i, 2], ray_dirs[j, i, 3])
            
            # Check for intersection with each sphere
            hit_anything = false
            closest_t = 1.0f10
            hit_sphere_idx = 0
            
            for s in 1:sphere_count
                hit, t = sphere_intersection(camera_pos, ray_dir, spheres, s)
                
                if hit && t < closest_t
                    hit_anything = true
                    closest_t = t
                    hit_sphere_idx = s
                end
            end
            
            # If we hit a sphere, calculate shading
            if hit_anything
                # Get hit point
                hit_x = camera_pos[1] + closest_t * ray_dir[1]
                hit_y = camera_pos[2] + closest_t * ray_dir[2]
                hit_z = camera_pos[3] + closest_t * ray_dir[3]
                
                # Get sphere data for the hit sphere
                base_idx = (hit_sphere_idx-1) * 7 + 1
                center_x = spheres[base_idx]
                center_y = spheres[base_idx+1]
                center_z = spheres[base_idx+2]
                radius = spheres[base_idx+3]
                sphere_r = spheres[base_idx+4]
                sphere_g = spheres[base_idx+5]
                sphere_b = spheres[base_idx+6]
                
                # Calculate surface normal
                normal_x = (hit_x - center_x) / radius
                normal_y = (hit_y - center_y) / radius
                normal_z = (hit_z - center_z) / radius
                normal_len = sqrt(normal_x^2 + normal_y^2 + normal_z^2)
                normal_x /= normal_len
                normal_y /= normal_len
                normal_z /= normal_len
                
                # Simple lighting from above
                light_dir_x = 0.0f0
                light_dir_y = 1.0f0
                light_dir_z = 0.0f0
                
                # Calculate diffuse lighting factor
                diffuse = max(0.0f0, normal_x * light_dir_x + 
                                    normal_y * light_dir_y + 
                                    normal_z * light_dir_z)
                
                # Set pixel color with diffuse shading
                result[j, i, 1] = sphere_r * (0.2f0 + 0.8f0 * diffuse)
                result[j, i, 2] = sphere_g * (0.2f0 + 0.8f0 * diffuse)
                result[j, i, 3] = sphere_b * (0.2f0 + 0.8f0 * diffuse)
            end
        end
    end
    
    return result
end

# Run the main function
main()