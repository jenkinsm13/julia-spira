#!/usr/bin/env julia

# SPIRA - Metal GPU raytracer using array abstractions
# This version implements a simple ray tracer using Metal.jl's array operations

using LinearAlgebra
using Random
using Images
using StaticArrays
using FileIO
using Base.Threads

println("SPIRA - Metal Raytracer (Array Version)")
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

# Generate primary rays for all pixels
function generate_rays(width::Int, height::Int, camera_params)
    origin, lower_left_corner, horizontal, vertical = camera_params
    
    # Create arrays to hold ray origins and directions for all pixels
    ray_origins = Array{Float32}(undef, width * height, 3)
    ray_directions = Array{Float32}(undef, width * height, 3)
    
    # Set all ray origins to camera origin
    for i in 1:width*height
        ray_origins[i, 1] = origin[1]
        ray_origins[i, 2] = origin[2]
        ray_origins[i, 3] = origin[3]
    end
    
    # Calculate ray directions for each pixel
    idx = 1
    for j in 1:height
        for i in 1:width
            # Calculate UV coordinates
            u = Float32((i - 1) / (width - 1))
            v = Float32((j - 1) / (height - 1))
            
            # Calculate ray direction
            dir = lower_left_corner + u*horizontal + v*vertical - origin
            dir_len = sqrt(dir[1]^2 + dir[2]^2 + dir[3]^2)
            
            # Store normalized direction
            ray_directions[idx, 1] = dir[1] / dir_len
            ray_directions[idx, 2] = dir[2] / dir_len
            ray_directions[idx, 3] = dir[3] / dir_len
            
            idx += 1
        end
    end
    
    return ray_origins, ray_directions
end

# Function to perform sphere intersection tests using CPU (for reference)
function cpu_intersect_spheres(ray_origins, ray_directions, sphere_data, width, height)
    num_rays = size(ray_directions, 1)
    num_spheres = div(length(sphere_data), 7)
    
    # Arrays to store result
    hit_spheres = fill(-1, num_rays)  # -1 = no hit, otherwise index of sphere
    hit_distances = fill(INF, num_rays)
    hit_positions = zeros(Float32, num_rays, 3)
    hit_normals = zeros(Float32, num_rays, 3)
    
    # For each ray
    for ray_idx in 1:num_rays
        ray_origin = @view ray_origins[ray_idx, :]
        ray_dir = @view ray_directions[ray_idx, :]
        
        # For each sphere
        for sphere_idx in 1:num_spheres
            base_idx = (sphere_idx-1) * 7 + 1
            sphere_center = @view sphere_data[base_idx:base_idx+2]
            sphere_radius = sphere_data[base_idx+3]
            
            # Calculate intersection using the quadratic formula
            oc = ray_origin - sphere_center
            a = ray_dir[1]^2 + ray_dir[2]^2 + ray_dir[3]^2
            half_b = oc[1]*ray_dir[1] + oc[2]*ray_dir[2] + oc[3]*ray_dir[3]
            c = oc[1]^2 + oc[2]^2 + oc[3]^2 - sphere_radius^2
            discriminant = half_b*half_b - a*c
            
            if discriminant > 0
                sqrtd = sqrt(discriminant)
                
                # Try both intersection points
                t1 = (-half_b - sqrtd) / a
                t2 = (-half_b + sqrtd) / a
                
                # Use closest valid t
                t = t1 > Float32(0.001) ? t1 : t2
                
                # Update if closer than current hit
                if t > Float32(0.001) && t < hit_distances[ray_idx]
                    hit_spheres[ray_idx] = sphere_idx
                    hit_distances[ray_idx] = t
                    
                    # Calculate hit position
                    hit_positions[ray_idx, 1] = ray_origin[1] + t * ray_dir[1]
                    hit_positions[ray_idx, 2] = ray_origin[2] + t * ray_dir[2]
                    hit_positions[ray_idx, 3] = ray_origin[3] + t * ray_dir[3]
                    
                    # Calculate normal
                    hit_normals[ray_idx, 1] = (hit_positions[ray_idx, 1] - sphere_center[1]) / sphere_radius
                    hit_normals[ray_idx, 2] = (hit_positions[ray_idx, 2] - sphere_center[2]) / sphere_radius
                    hit_normals[ray_idx, 3] = (hit_positions[ray_idx, 3] - sphere_center[3]) / sphere_radius
                    
                    # Normalize normal
                    normal_len = sqrt(hit_normals[ray_idx, 1]^2 + hit_normals[ray_idx, 2]^2 + hit_normals[ray_idx, 3]^2)
                    hit_normals[ray_idx, 1] /= normal_len
                    hit_normals[ray_idx, 2] /= normal_len
                    hit_normals[ray_idx, 3] /= normal_len
                end
            end
        end
    end
    
    return hit_spheres, hit_distances, hit_positions, hit_normals
end

# Function to perform sphere intersection tests using Metal
function metal_intersect_spheres(ray_origins_gpu, ray_directions_gpu, sphere_data_gpu, width, height)
    # We need to process each sphere separately since Metal.jl can't do
    # dynamic iteration of spheres within array operations
    
    Metal.allowscalar(true)  # Allow scalar indexing for simplicity
    
    num_rays = size(ray_directions_gpu, 1)
    num_spheres = div(length(sphere_data_gpu), 7)
    
    # Arrays to store results
    hit_spheres = Metal.MtlArray(fill(Int32(-1), num_rays))
    hit_distances = Metal.MtlArray(fill(INF, num_rays))
    hit_positions = Metal.MtlArray(zeros(Float32, num_rays, 3))
    hit_normals = Metal.MtlArray(zeros(Float32, num_rays, 3))
    
    # Process each sphere
    for sphere_idx in 1:num_spheres
        base_idx = (sphere_idx-1) * 7 + 1
        sphere_center_x = sphere_data_gpu[base_idx]
        sphere_center_y = sphere_data_gpu[base_idx+1]
        sphere_center_z = sphere_data_gpu[base_idx+2]
        sphere_radius = sphere_data_gpu[base_idx+3]
        
        # Calculate oc for all rays
        oc_x = ray_origins_gpu[:, 1] .- sphere_center_x
        oc_y = ray_origins_gpu[:, 2] .- sphere_center_y
        oc_z = ray_origins_gpu[:, 3] .- sphere_center_z
        
        # Calculate quadratic coefficients for all rays
        a = ray_directions_gpu[:, 1].^2 .+ ray_directions_gpu[:, 2].^2 .+ ray_directions_gpu[:, 3].^2
        half_b = oc_x .* ray_directions_gpu[:, 1] .+ oc_y .* ray_directions_gpu[:, 2] .+ oc_z .* ray_directions_gpu[:, 3]
        c = oc_x.^2 .+ oc_y.^2 .+ oc_z.^2 .- sphere_radius^2
        
        # Calculate discriminant
        discriminant = half_b.^2 .- a .* c
        
        # For rays that hit this sphere
        hits = discriminant .> Float32(0.0)
        
        if any(hits)
            # Calculate both intersection points
            sqrtd = sqrt.(discriminant[hits])
            t1 = (-half_b[hits] .- sqrtd) ./ a[hits]
            t2 = (-half_b[hits] .+ sqrtd) ./ a[hits]
            
            # Get indices of rays that hit
            hit_indices = findall(hits)
            
            for (i, ray_idx) in enumerate(hit_indices)
                # Use closest valid t
                t = t1[i] > Float32(0.001) ? t1[i] : t2[i]
                
                # Skip if not valid
                if t <= Float32(0.001)
                    continue
                end
                
                # Update if closer than current hit
                if t < hit_distances[ray_idx]
                    hit_spheres[ray_idx] = Int32(sphere_idx)
                    hit_distances[ray_idx] = t
                    
                    # Calculate hit position
                    hit_positions[ray_idx, 1] = ray_origins_gpu[ray_idx, 1] + t * ray_directions_gpu[ray_idx, 1]
                    hit_positions[ray_idx, 2] = ray_origins_gpu[ray_idx, 2] + t * ray_directions_gpu[ray_idx, 2]
                    hit_positions[ray_idx, 3] = ray_origins_gpu[ray_idx, 3] + t * ray_directions_gpu[ray_idx, 3]
                    
                    # Calculate normal
                    nx = hit_positions[ray_idx, 1] - sphere_center_x
                    ny = hit_positions[ray_idx, 2] - sphere_center_y
                    nz = hit_positions[ray_idx, 3] - sphere_center_z
                    
                    # Normalize normal
                    normal_len = sqrt(nx*nx + ny*ny + nz*nz)
                    hit_normals[ray_idx, 1] = nx / normal_len
                    hit_normals[ray_idx, 2] = ny / normal_len
                    hit_normals[ray_idx, 3] = nz / normal_len
                end
            end
        end
    end
    
    return hit_spheres, hit_distances, hit_positions, hit_normals
end

# Shade the hit points to generate colors
function shade_hits(hit_spheres, hit_normals, ray_directions, sphere_data, width, height)
    num_rays = length(hit_spheres)
    colors = zeros(Float32, num_rays, 3)
    
    # Light direction (from above)
    light_dir = normalize(Vec3(Float32(0.0), Float32(1.0), Float32(0.0)))
    
    for ray_idx in 1:num_rays
        sphere_idx = hit_spheres[ray_idx]
        
        if sphere_idx > 0
            # Get sphere color
            base_idx = (sphere_idx-1) * 7 + 1
            color_r = sphere_data[base_idx+4]
            color_g = sphere_data[base_idx+5]
            color_b = sphere_data[base_idx+6]
            
            # Basic diffuse lighting
            normal = Vec3(hit_normals[ray_idx, 1], hit_normals[ray_idx, 2], hit_normals[ray_idx, 3])
            light_factor = max(Float32(0.2), dot(normal, light_dir))
            
            # Set color with lighting
            colors[ray_idx, 1] = color_r * light_factor
            colors[ray_idx, 2] = color_g * light_factor
            colors[ray_idx, 3] = color_b * light_factor
        else
            # Background color (sky gradient)
            t = Float32(0.5) * (ray_directions[ray_idx, 2] + Float32(1.0))
            colors[ray_idx, 1] = (Float32(1.0) - t) + t * Float32(0.5)  # r
            colors[ray_idx, 2] = (Float32(1.0) - t) + t * Float32(0.7)  # g
            colors[ray_idx, 3] = (Float32(1.0) - t) + t * Float32(1.0)  # b
        end
    end
    
    return colors
end

# Shade the hit points to generate colors (GPU version)
function metal_shade_hits(hit_spheres_gpu, hit_normals_gpu, ray_directions_gpu, sphere_data_gpu, width, height)
    Metal.allowscalar(true)  # Allow scalar indexing for simplicity
    
    num_rays = length(hit_spheres_gpu)
    colors_gpu = Metal.MtlArray(zeros(Float32, num_rays, 3))
    
    # Process each ray
    for ray_idx in 1:num_rays
        sphere_idx = hit_spheres_gpu[ray_idx]
        
        if sphere_idx > 0
            # Get sphere color
            base_idx = (sphere_idx-1) * 7 + 1
            color_r = sphere_data_gpu[base_idx+4]
            color_g = sphere_data_gpu[base_idx+5]
            color_b = sphere_data_gpu[base_idx+6]
            
            # Basic diffuse lighting
            nx = hit_normals_gpu[ray_idx, 1]
            ny = hit_normals_gpu[ray_idx, 2]
            nz = hit_normals_gpu[ray_idx, 3]
            
            # Light direction (from above)
            light_dir_x = Float32(0.0)
            light_dir_y = Float32(1.0)
            light_dir_z = Float32(0.0)
            
            # Dot product for diffuse factor
            light_factor = max(Float32(0.2), nx*light_dir_x + ny*light_dir_y + nz*light_dir_z)
            
            # Set color with lighting
            colors_gpu[ray_idx, 1] = color_r * light_factor
            colors_gpu[ray_idx, 2] = color_g * light_factor
            colors_gpu[ray_idx, 3] = color_b * light_factor
        else
            # Background color (sky gradient)
            t = Float32(0.5) * (ray_directions_gpu[ray_idx, 2] + Float32(1.0))
            colors_gpu[ray_idx, 1] = (Float32(1.0) - t) + t * Float32(0.5)  # r
            colors_gpu[ray_idx, 2] = (Float32(1.0) - t) + t * Float32(0.7)  # g
            colors_gpu[ray_idx, 3] = (Float32(1.0) - t) + t * Float32(1.0)  # b
        end
    end
    
    return colors_gpu
end

# Render function for CPU
function render_cpu(width, height, spheres)
    # Set up camera
    camera_params = setup_camera(width, height)
    
    # Generate primary rays
    ray_origins, ray_directions = generate_rays(width, height, camera_params)
    
    # Prepare sphere data
    sphere_data = prepare_spheres_for_gpu(spheres)
    
    # Perform intersection tests
    hit_spheres, hit_distances, hit_positions, hit_normals = 
        cpu_intersect_spheres(ray_origins, ray_directions, sphere_data, width, height)
    
    # Shade the hit points
    colors = shade_hits(hit_spheres, hit_normals, ray_directions, sphere_data, width, height)
    
    # Create the final image
    img = Array{RGB{Float32}}(undef, height, width)
    idx = 1
    for j in 1:height
        for i in 1:width
            img[height-j+1, i] = RGB{Float32}(
                colors[idx, 1],
                colors[idx, 2],
                colors[idx, 3]
            )
            idx += 1
        end
    end
    
    return img
end

# Render function for Metal
function render_metal(width, height, spheres)
    if !has_metal
        println("Metal not available, falling back to CPU rendering")
        return render_cpu(width, height, spheres)
    end
    
    # Set up camera
    camera_params = setup_camera(width, height)
    
    # Generate primary rays
    ray_origins, ray_directions = generate_rays(width, height, camera_params)
    
    # Prepare sphere data and copy to GPU
    sphere_data = prepare_spheres_for_gpu(spheres)
    
    # Copy data to Metal arrays
    ray_origins_gpu = Metal.MtlArray(ray_origins)
    ray_directions_gpu = Metal.MtlArray(ray_directions)
    sphere_data_gpu = Metal.MtlArray(sphere_data)
    
    # Perform intersection tests on GPU
    hit_spheres_gpu, hit_distances_gpu, hit_positions_gpu, hit_normals_gpu = 
        metal_intersect_spheres(ray_origins_gpu, ray_directions_gpu, sphere_data_gpu, width, height)
    
    # Shade the hit points on GPU
    colors_gpu = metal_shade_hits(hit_spheres_gpu, hit_normals_gpu, ray_directions_gpu, sphere_data_gpu, width, height)
    
    # Copy colors back to CPU
    colors = Array(colors_gpu)
    
    # Create the final image
    img = Array{RGB{Float32}}(undef, height, width)
    idx = 1
    for j in 1:height
        for i in 1:width
            img[height-j+1, i] = RGB{Float32}(
                colors[idx, 1],
                colors[idx, 2],
                colors[idx, 3]
            )
            idx += 1
        end
    end
    
    return img
end

# Main function
function main()
    width = 640  # Default resolution
    height = 360
    
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
    save("metal_array_render.png", img)
    println("Saved render to metal_array_render.png")
end

# Run the main function if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end