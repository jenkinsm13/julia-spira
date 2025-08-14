#!/usr/bin/env julia

# SPIRA - Metal GPU raytracer with optimized array abstractions
# This version properly uses Metal.jl's MtlArray for GPU acceleration

using LinearAlgebra
using Random
using Images
using StaticArrays
using FileIO
using Base.Threads

println("SPIRA - Optimized Metal Raytracer")
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

# Ray structure
struct Ray
    origin::Point3
    direction::Vec3
    
    Ray(origin::Point3, direction::Vec3) = new(origin, normalize(direction))
end

function at(ray::Ray, t::Real)
    return ray.origin + t * ray.direction
end

# Simple sphere primitive
struct Sphere
    center::Point3
    radius::Float32
    material::Int  # Index into materials array
    
    Sphere(center::Point3, radius::Float32, material::Int) = new(center, radius, material)
end

# Material structure
struct Material
    albedo::Color
    emission::Color
    metallic::Float32
    roughness::Float32
    
    Material(albedo::Color; emission::Color=BLACK, metallic::Float32=0.0, roughness::Float32=0.5) = 
        new(albedo, emission, metallic, roughness)
end

# Camera structure
struct Camera
    origin::Point3
    lower_left_corner::Point3
    horizontal::Vec3
    vertical::Vec3
    
    function Camera(lookfrom::Point3, lookat::Point3, vup::Vec3, vfov::Real, aspect_ratio::Real)
        theta = Float32(deg2rad(vfov))
        h = tan(theta/2)
        viewport_height = Float32(2.0) * h
        viewport_width = Float32(aspect_ratio) * viewport_height
        
        w = normalize(lookfrom - lookat)
        u = normalize(cross(vup, w))
        v = cross(w, u)
        
        origin = lookfrom
        horizontal = viewport_width * u
        vertical = viewport_height * v
        lower_left_corner = origin - horizontal/Float32(2.0) - vertical/Float32(2.0) - w
        
        new(origin, lower_left_corner, horizontal, vertical)
    end
end

# Scene structure
struct Scene
    spheres::Vector{Sphere}
    materials::Vector{Material}
end

# Create a simple scene for rendering
function create_scene()
    materials = [
        # Diffuse red
        Material(
            Vec3(Float32(0.7), Float32(0.3), Float32(0.3)),
            emission=BLACK,
            metallic=Float32(0.0),
            roughness=Float32(0.5)
        ),
        
        # Ground
        Material(
            Vec3(Float32(0.5), Float32(0.5), Float32(0.5)),
            emission=BLACK,
            metallic=Float32(0.0),
            roughness=Float32(0.9)
        ),
        
        # Metal
        Material(
            Vec3(Float32(0.8), Float32(0.8), Float32(0.8)),
            emission=BLACK,
            metallic=Float32(1.0),
            roughness=Float32(0.0)
        ),
        
        # Glass-like
        Material(
            Vec3(Float32(0.8), Float32(0.8), Float32(1.0)),
            emission=BLACK,
            metallic=Float32(0.9),
            roughness=Float32(0.0)
        ),
        
        # Light
        Material(
            Vec3(Float32(1.0), Float32(1.0), Float32(1.0)),
            emission=Vec3(Float32(5.0), Float32(5.0), Float32(5.0)),
            metallic=Float32(0.0),
            roughness=Float32(0.0)
        )
    ]
    
    spheres = [
        # Main sphere
        Sphere(
            Point3(Float32(0.0), Float32(0.0), Float32(0.0)),
            Float32(0.5),
            1
        ),
        
        # Ground
        Sphere(
            Point3(Float32(0.0), Float32(-100.5), Float32(0.0)),
            Float32(100.0),
            2
        ),
        
        # Metal sphere
        Sphere(
            Point3(Float32(1.0), Float32(0.0), Float32(0.0)),
            Float32(0.5),
            3
        ),
        
        # Glass-like sphere
        Sphere(
            Point3(Float32(-1.0), Float32(0.0), Float32(0.0)),
            Float32(0.5),
            4
        ),
        
        # Light source
        Sphere(
            Point3(Float32(0.0), Float32(5.0), Float32(0.0)),
            Float32(1.0),
            5
        )
    ]
    
    return Scene(spheres, materials)
end

# Helper function to prepare scene data for GPU
function prepare_scene_data(scene::Scene)
    # Prepare sphere data in flattened array format for GPU
    sphere_data = zeros(Float32, 5 * length(scene.spheres))
    for (i, sphere) in enumerate(scene.spheres)
        idx = (i-1) * 5 + 1
        sphere_data[idx] = sphere.center[1]     # center.x
        sphere_data[idx+1] = sphere.center[2]   # center.y
        sphere_data[idx+2] = sphere.center[3]   # center.z
        sphere_data[idx+3] = sphere.radius      # radius
        sphere_data[idx+4] = Float32(sphere.material)  # material index
    end
    
    # Prepare material data in flattened array format for GPU
    material_data = zeros(Float32, 8 * length(scene.materials))
    for (i, material) in enumerate(scene.materials)
        idx = (i-1) * 8 + 1
        material_data[idx] = material.albedo[1]    # albedo.r
        material_data[idx+1] = material.albedo[2]  # albedo.g
        material_data[idx+2] = material.albedo[3]  # albedo.b
        material_data[idx+3] = material.emission[1]  # emission.r
        material_data[idx+4] = material.emission[2]  # emission.g
        material_data[idx+5] = material.emission[3]  # emission.b
        material_data[idx+6] = material.metallic     # metallic
        material_data[idx+7] = material.roughness    # roughness
    end
    
    return sphere_data, material_data
end

# Generate initial ray data for all pixels
function generate_ray_data(width::Int, height::Int, camera::Camera)
    # Calculate ray origin for each pixel (same for all pixels)
    ray_origins = zeros(Float32, width * height, 3)
    for i in 1:width*height
        ray_origins[i, 1] = camera.origin[1]
        ray_origins[i, 2] = camera.origin[2]
        ray_origins[i, 3] = camera.origin[3]
    end
    
    # Calculate ray directions for each pixel (center of pixel)
    ray_directions = zeros(Float32, width * height, 3)
    idx = 1
    for j in 1:height
        for i in 1:width
            u = Float32((i - 1) / (width - 1))
            v = Float32((j - 1) / (height - 1))
            
            ray_dir = camera.lower_left_corner + u*camera.horizontal + v*camera.vertical - camera.origin
            ray_dir_normalized = normalize(ray_dir)
            
            ray_directions[idx, 1] = ray_dir_normalized[1]
            ray_directions[idx, 2] = ray_dir_normalized[2]
            ray_directions[idx, 3] = ray_dir_normalized[3]
            
            idx += 1
        end
    end
    
    return ray_origins, ray_directions
end

# Apply jitter to ray directions for anti-aliasing
function apply_jitter(width::Int, height::Int, ray_directions::Array{Float32, 2}, camera::Camera)
    jittered_directions = copy(ray_directions)
    
    # Generate jittered u,v coordinates
    jitter_scale = Float32(0.5 / width)  # Scale jitter based on resolution
    
    idx = 1
    for j in 1:height
        for i in 1:width
            # Base u, v coordinates
            u = Float32((i - 1) / (width - 1))
            v = Float32((j - 1) / (height - 1))
            
            # Jittered u, v
            u_jitter = u + (rand(Float32) - Float32(0.5)) * jitter_scale
            v_jitter = v + (rand(Float32) - Float32(0.5)) * jitter_scale
            
            # Calculate jittered ray direction
            ray_dir = camera.lower_left_corner + u_jitter*camera.horizontal + v_jitter*camera.vertical - camera.origin
            ray_dir_normalized = normalize(ray_dir)
            
            jittered_directions[idx, 1] = ray_dir_normalized[1]
            jittered_directions[idx, 2] = ray_dir_normalized[2]
            jittered_directions[idx, 3] = ray_dir_normalized[3]
            
            idx += 1
        end
    end
    
    return jittered_directions
end

# Intersection test for rays and spheres using Metal.jl's array operations
function gpu_ray_sphere_intersection(ray_origins::Metal.MtlArray{Float32, 2}, 
                                     ray_directions::Metal.MtlArray{Float32, 2},
                                     sphere_data::Metal.MtlArray{Float32, 1}, 
                                     num_spheres::Int)
    num_rays = size(ray_origins, 1)
    
    # Arrays to store results, defined in the function's main scope
    hit_results = Metal.MtlArray{Float32}(undef, num_rays, 3)  # [hit?, t, material_idx]
    hit_points = Metal.MtlArray{Float32}(undef, num_rays, 3)   # [x, y, z]
    hit_normals = Metal.MtlArray{Float32}(undef, num_rays, 3)  # [nx, ny, nz]
    
    # Initialize hit results:
    # Column 1: Hit flag (0.0 = no hit, 1.0 = hit)
    # Column 2: Hit distance t (initialized to a large value)
    # Column 3: Material index of the hit sphere (0 if no hit)
    hit_results[:, 1] .= 0.0f0 
    hit_results[:, 2] .= Float32(1e20) 
    hit_results[:, 3] .= 0.0f0

    # Initialize hit points and normals. These will be overwritten on a hit.
    hit_points .= Float32(0.0)
    hit_normals .= Float32(0.0)

    # Enable scalar indexing for operations within this block
    Metal.allowscalar() do
        # Process each sphere
        for s in 1:num_spheres
            # Extract sphere data
            base_idx = (s-1) * 5 + 1
            sphere_center_x = sphere_data[base_idx]
            sphere_center_y = sphere_data[base_idx+1]
            sphere_center_z = sphere_data[base_idx+2]
            sphere_radius = sphere_data[base_idx+3]
            sphere_material = sphere_data[base_idx+4]
            
            # For each ray, calculate intersection
            # oc = ray_origin - sphere_center
            oc_x = ray_origins[:, 1] .- sphere_center_x
            oc_y = ray_origins[:, 2] .- sphere_center_y
            oc_z = ray_origins[:, 3] .- sphere_center_z
            
            # Calculate quadratic terms
            a = ray_directions[:, 1].^2 + ray_directions[:, 2].^2 + ray_directions[:, 3].^2
            half_b = oc_x .* ray_directions[:, 1] + 
                    oc_y .* ray_directions[:, 2] + 
                    oc_z .* ray_directions[:, 3]
            c = oc_x.^2 + oc_y.^2 + oc_z.^2 .- sphere_radius^2
            
            # Calculate discriminant
            discriminant = half_b.^2 .- a .* c
            
            # Create a mask for rays that hit this sphere
            hit_mask = discriminant .> Float32(0.0)
            
            if any(hit_mask)
                # For rays that hit, calculate intersection points
                hit_indices = findall(hit_mask)
                
                # Process these indices
                for idx in hit_indices
                    # Calculate t values
                    sqrtd = sqrt(discriminant[idx])
                    t1 = (-half_b[idx] - sqrtd) / a[idx]
                    t2 = (-half_b[idx] + sqrtd) / a[idx]
                    
                    # Find valid t (closest positive)
                    t = (t1 > Float32(0.001)) ? t1 : t2
                    
                    # Skip if t is not valid or not closer than existing hit
                    if t <= Float32(0.001) || t >= hit_results[idx, 2]
                        continue
                    end
                    
                    # Update hit record
                    hit_results[idx, 1] = Float32(1.0)  # Hit
                    hit_results[idx, 2] = t             # Distance
                    hit_results[idx, 3] = sphere_material  # Material
                    
                    # Calculate hit point: origin + t * direction
                    hit_points[idx, 1] = ray_origins[idx, 1] + t * ray_directions[idx, 1]
                    hit_points[idx, 2] = ray_origins[idx, 2] + t * ray_directions[idx, 2]
                    hit_points[idx, 3] = ray_origins[idx, 3] + t * ray_directions[idx, 3]
                    
                    # Calculate normal: normalize(hit_point - center)
                    nx = hit_points[idx, 1] - sphere_center_x
                    ny = hit_points[idx, 2] - sphere_center_y
                    nz = hit_points[idx, 3] - sphere_center_z
                    
                    # Normalize normal
                    inv_len = Float32(1.0) / sqrt(nx*nx + ny*ny + nz*nz)
                    hit_normals[idx, 1] = nx * inv_len
                    hit_normals[idx, 2] = ny * inv_len
                    hit_normals[idx, 3] = nz * inv_len
                end
            end
        end
    end
    
    return hit_results, hit_points, hit_normals
end

# Generate reflected rays for metallic surfaces
function gpu_reflect_rays(hit_results::Metal.MtlArray{Float32, 2}, 
                         ray_directions::Metal.MtlArray{Float32, 2},
                         hit_normals::Metal.MtlArray{Float32, 2},
                         hit_points::Metal.MtlArray{Float32, 2},
                         material_data::Metal.MtlArray{Float32, 1})
    
    num_rays = size(ray_directions, 1)
    reflected_dirs = copy(ray_directions) # Initialize with original directions
    new_origins = copy(hit_points)        # Initialize with current hit points

    # Enable scalar indexing 
    Metal.allowscalar() do
        # Process rays that hit something
        for i in 1:num_rays
            if hit_results[i, 1] > Float32(0.0)  # If this ray hit something
                # Get material properties
                mat_idx = Int(hit_results[i, 3])
                mat_base_idx = (mat_idx-1) * 8 + 1
                metallic = material_data[mat_base_idx+6]
                roughness = material_data[mat_base_idx+7]
                
                # Get normal and incident direction
                nx = hit_normals[i, 1]
                ny = hit_normals[i, 2]
                nz = hit_normals[i, 3]
                
                dx = ray_directions[i, 1]
                dy = ray_directions[i, 2]
                dz = ray_directions[i, 3]
                
                # Calculate reflection: reflect(d, n) = d - 2*dot(d,n)*n
                dot_prod = dx*nx + dy*ny + dz*nz
                
                if metallic > Float32(0.0)  # Metallic reflection
                    # Basic reflection
                    rx = dx - Float32(2.0) * dot_prod * nx
                    ry = dy - Float32(2.0) * dot_prod * ny
                    rz = dz - Float32(2.0) * dot_prod * nz
                    
                    # Add roughness if needed
                    if roughness > Float32(0.0)
                        # Generate random vector for roughness
                        rnd_x = rand(Float32) - Float32(0.5)
                        rnd_y = rand(Float32) - Float32(0.5)
                        rnd_z = rand(Float32) - Float32(0.5)
                        
                        # Add roughness
                        rx += roughness * rnd_x
                        ry += roughness * rnd_y
                        rz += roughness * rnd_z
                        
                        # Normalize
                        inv_len = Float32(1.0) / sqrt(rx*rx + ry*ry + rz*rz)
                        rx *= inv_len
                        ry *= inv_len
                        rz *= inv_len
                    end
                    
                    # Set new direction
                    reflected_dirs[i, 1] = rx
                    reflected_dirs[i, 2] = ry
                    reflected_dirs[i, 3] = rz
                else  # Diffuse reflection
                    # Random direction in hemisphere
                    rnd_x = rand(Float32) - Float32(0.5)
                    rnd_y = rand(Float32) - Float32(0.5)
                    rnd_z = rand(Float32) - Float32(0.5)
                    
                    # Make sure it's pointing in same hemisphere as normal
                    if (rnd_x*nx + rnd_y*ny + rnd_z*nz) < Float32(0.0)
                        rnd_x = -rnd_x
                        rnd_y = -rnd_y
                        rnd_z = -rnd_z
                    end
                    
                    # Create normalized random direction in hemisphere
                    inv_len = Float32(1.0) / sqrt(rnd_x*rnd_x + rnd_y*rnd_y + rnd_z*rnd_z)
                    reflected_dirs[i, 1] = rnd_x * inv_len
                    reflected_dirs[i, 2] = rnd_y * inv_len
                    reflected_dirs[i, 3] = rnd_z * inv_len
                end
            end
        end
    end
    
    return new_origins, reflected_dirs
end

# Shade rays based on intersection results
function gpu_shade_rays(hit_results::Metal.MtlArray{Float32, 2},
                       ray_directions::Metal.MtlArray{Float32, 2},
                       material_data::Metal.MtlArray{Float32, 1},
                       contribution::Metal.MtlArray{Float32, 1})
    
    num_rays = size(ray_directions, 1)
    colors = Metal.MtlArray{Float32}(undef, num_rays, 3)

    Metal.allowscalar() do
        # Process each ray
        for i in 1:num_rays
            if hit_results[i, 1] > Float32(0.0)  # Ray hit something
                # Get material properties
                mat_idx = Int(hit_results[i, 3])
                mat_base_idx = (mat_idx-1) * 8 + 1
                
                # Get material properties
                albedo_r = material_data[mat_base_idx]
                albedo_g = material_data[mat_base_idx+1]
                albedo_b = material_data[mat_base_idx+2]
                
                emission_r = material_data[mat_base_idx+3]
                emission_g = material_data[mat_base_idx+4]
                emission_b = material_data[mat_base_idx+5]
                
                # Apply contribution factor
                contrib = contribution[i]
                
                # Set color including emission
                colors[i, 1] = albedo_r * contrib + emission_r
                colors[i, 2] = albedo_g * contrib + emission_g
                colors[i, 3] = albedo_b * contrib + emission_b
            else
                # Sky color for rays that don't hit anything
                dir_y = ray_directions[i, 2]
                t = Float32(0.5) * (dir_y + Float32(1.0))
                
                colors[i, 1] = (Float32(1.0) - t) + t * Float32(0.5)  # r
                colors[i, 2] = (Float32(1.0) - t) + t * Float32(0.7)  # g
                colors[i, 3] = (Float32(1.0) - t) + t * Float32(1.0)  # b
            end
        end
    end # Metal.allowscalar()
    
    return colors
end

# Tone mapping function
function gpu_tone_map(colors::Metal.MtlArray{Float32, 2})
    # ACES filmic tone mapping (simplified)
    a = Float32(2.51)
    b = Float32(0.03)
    c = Float32(2.43)
    d = Float32(0.59)
    e = Float32(0.14)
    
    # Apply tone mapping formula to each channel
    r = colors[:, 1]
    g = colors[:, 2]
    b = colors[:, 3]
    
    # ACES tone mapping
    mapped_r = (r .* (a .* r .+ b)) ./ (r .* (c .* r .+ d) .+ e)
    mapped_g = (g .* (a .* g .+ b)) ./ (g .* (c .* g .+ d) .+ e)
    mapped_b = (b .* (a .* b .+ b)) ./ (b .* (c .* b .+ d) .+ e)
    
    # Clamp to [0,1]
    mapped_r = clamp.(mapped_r, Float32(0.0), Float32(1.0))
    mapped_g = clamp.(mapped_g, Float32(0.0), Float32(1.0))
    mapped_b = clamp.(mapped_b, Float32(0.0), Float32(1.0))
    
    # Gamma correction
    mapped_r = sqrt.(mapped_r)
    mapped_g = sqrt.(mapped_g)
    mapped_b = sqrt.(mapped_b)
    
    return Metal.MtlArray(hcat(mapped_r, mapped_g, mapped_b))
end

# Simplified hybrid GPU/CPU path tracer
function render_hybrid_gpu(width::Int, height::Int, scene::Scene, camera::Camera;
                          samples_per_pixel::Int=16, max_depth::Int=4)
    
    if !has_metal
        println("Metal not available, falling back to CPU rendering.")
        return render_with_cpu(width, height, scene, camera, 
                              samples_per_pixel=samples_per_pixel,
                              max_depth=max_depth)
    end
    
    # Prepare scene data
    sphere_data, material_data = prepare_scene_data(scene)
    sphere_data_gpu = Metal.MtlArray(sphere_data)
    material_data_gpu = Metal.MtlArray(material_data)
    
    # Generate initial ray data
    ray_origins_cpu, ray_directions_cpu = generate_ray_data(width, height, camera)
    
    # Create image buffer
    img_buffer = zeros(Float32, height, width, 3)
    
    # Get number of spheres
    num_spheres = length(scene.spheres)
    
    for sample in 1:samples_per_pixel
        # Apply jitter for anti-aliasing (if not first sample)
        if sample > 1
            ray_directions_cpu = apply_jitter(width, height, ray_directions_cpu, camera)
        end
        
        # Copy ray data to GPU
        ray_origins_gpu = Metal.MtlArray(ray_origins_cpu)
        ray_directions_gpu = Metal.MtlArray(ray_directions_cpu)
        
        # Initialize contribution for each ray
        contribution_gpu = Metal.MtlArray(ones(Float32, width * height))
        
        # For each bounce depth
        for depth in 1:max_depth
            # Trace rays against spheres
            hit_results, hit_points, hit_normals = gpu_ray_sphere_intersection(
                ray_origins_gpu, ray_directions_gpu, sphere_data_gpu, num_spheres
            )
            
            # Break if no rays hit anything
            if !any(Array(hit_results[:, 1]) .> Float32(0.0))
                break
            end
            
            # For rays that hit, generate new rays
            ray_origins_gpu, ray_directions_gpu = gpu_reflect_rays(
                hit_results, ray_directions_gpu, hit_normals, hit_points, material_data_gpu
            )
            
            # Apply attenuation (halve contribution for each bounce)
            contribution_gpu .*= Float32(0.5)
            
            # If at max depth, we'll shade these rays
            if depth == max_depth
                # Shade rays and accumulate to image buffer
                colors = gpu_shade_rays(hit_results, ray_directions_gpu, material_data_gpu, contribution_gpu)
                
                # Apply tone mapping
                mapped_colors = gpu_tone_map(colors)
                
                # Accumulate to image buffer
                colors_cpu = Array(mapped_colors)
                
                # Reshape and accumulate to image buffer
                for idx in 1:width*height
                    row = div(idx-1, width) + 1
                    col = mod1(idx, width)
                    
                    img_buffer[height - row + 1, col, 1] += colors_cpu[idx, 1]
                    img_buffer[height - row + 1, col, 2] += colors_cpu[idx, 2]
                    img_buffer[height - row + 1, col, 3] += colors_cpu[idx, 3]
                end
            end
        end
        
        println("Completed sample $sample / $samples_per_pixel")
    end
    
    # Average samples
    img_buffer ./= samples_per_pixel
    
    # Create final RGB image
    img = Array{RGB{Float32}}(undef, height, width)
    for j in 1:height
        for i in 1:width
            img[j,i] = RGB{Float32}(img_buffer[j,i,1], img_buffer[j,i,2], img_buffer[j,i,3])
        end
    end
    
    return img
end

# CPU fallback implementation (same as in spira-metal11.jl)
function render_with_cpu(width, height, scene, camera; samples_per_pixel=16, max_depth=4)
    # Traditional CPU ray tracing implementation
    img = zeros(RGB{Float32}, height, width)
    
    # Helper function to trace a ray
    function trace_ray(ray::Ray, depth::Int)
        if depth <= 0
            return BLACK
        end
        
        hit_anything = false
        closest_t = INF
        hit_normal = BLACK
        hit_material_idx = 0
        
        # Check all spheres for intersection
        for sphere in scene.spheres
            oc = ray.origin - sphere.center
            a = Float32(1.0)  # Direction is normalized
            half_b = dot(oc, ray.direction)
            c = dot(oc, oc) - sphere.radius * sphere.radius
            discriminant = half_b * half_b - a * c
            
            if discriminant > 0
                sqrtd = sqrt(discriminant)
                
                # Try both intersection points
                root = (-half_b - sqrtd) / a
                if root < Float32(0.001)
                    root = (-half_b + sqrtd) / a
                end
                
                if root > Float32(0.001) && root < closest_t
                    closest_t = root
                    hit_anything = true
                    hit_normal = normalize((ray.origin + closest_t * ray.direction) - sphere.center)
                    hit_material_idx = sphere.material
                end
            end
        end
        
        if hit_anything
            hit_point = ray.origin + closest_t * ray.direction
            material = scene.materials[hit_material_idx]
            
            # Emissive material
            if any(material.emission .> 0)
                return material.emission
            end
            
            # Choose between diffuse and metallic reflection
            if rand(Float32) > material.metallic
                # Diffuse reflection
                target = hit_point + hit_normal + normalize(rand(Vec3) - Float32(0.5))
                scattered = Ray(hit_point, normalize(target - hit_point))
                return material.albedo .* trace_ray(scattered, depth - 1) .* Float32(0.5)
            else
                # Metallic reflection
                reflected = ray.direction - Float32(2.0) * dot(ray.direction, hit_normal) * hit_normal
                scattered = Ray(hit_point, normalize(reflected + material.roughness * (rand(Vec3) - Float32(0.5))))
                return material.albedo .* trace_ray(scattered, depth - 1)
            end
        end
        
        # Background sky color
        t = Float32(0.5) * (ray.direction[2] + Float32(1.0))
        return (Float32(1.0) - t) * WHITE + t * Vec3(Float32(0.5), Float32(0.7), Float32(1.0))
    end
    
    # Render with multi-threading
    println("Rendering with CPU ($(nthreads()) threads)...")
    
    @threads for j in 1:height
        if j % 20 == 0
            println("Row $j / $height")
        end
        
        for i in 1:width
            pixel_color = BLACK
            
            # Anti-aliasing with multiple samples
            for s in 1:samples_per_pixel
                u = Float32((i - 1 + rand(Float32)) / (width - 1))
                v = Float32((j - 1 + rand(Float32)) / (height - 1))
                
                ray_dir = camera.lower_left_corner + u*camera.horizontal + v*camera.vertical - camera.origin
                ray = Ray(camera.origin, ray_dir)
                
                pixel_color += trace_ray(ray, max_depth)
            end
            
            # Average samples
            pixel_color = pixel_color / Float32(samples_per_pixel)
            
            # Apply tone mapping and gamma correction
            pixel_color = clamp.(pixel_color, Float32(0.0), Float32(1.0))
            pixel_color = sqrt.(pixel_color)  # Gamma correction
            
            # Set pixel in image
            img[height - j + 1, i] = RGB{Float32}(pixel_color...)
        end
    end
    
    return img
end

# Main rendering function
function render(scene::Scene, camera::Camera, width::Int, height::Int;
              samples_per_pixel::Int=16, max_depth::Int=4,
              output_path::String="metal_optimized_render.png")
    
    # Time the rendering
    start_time = time()
    
    # Render using the hybrid GPU approach
    if has_metal
        println("Rendering with Metal GPU (hybrid approach)...")
        img = render_hybrid_gpu(width, height, scene, camera,
                               samples_per_pixel=samples_per_pixel,
                               max_depth=max_depth)
    else
        println("Rendering with CPU...")
        img = render_with_cpu(width, height, scene, camera,
                             samples_per_pixel=samples_per_pixel,
                             max_depth=max_depth)
    end
    
    # Report timing
    elapsed = time() - start_time
    println("Render completed in $(round(elapsed, digits=2)) seconds")
    
    # Save the image
    save(output_path, img)
    println("Saved render to $output_path")
    
    return img
end

# Main function
function main()
    width = 640  # Use smaller resolution for faster test
    height = 360
    samples = 32
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
    render(scene, camera, width, height, 
          samples_per_pixel=samples, 
          max_depth=max_depth, 
          output_path="metal_optimized_render.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end