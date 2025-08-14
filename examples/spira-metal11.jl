#!/usr/bin/env julia

# SPIRA - Metal GPU raytracer that uses Metal.jl's array abstractions
# This version uses Metal.jl's MtlArray for GPU acceleration

using LinearAlgebra
using Random
using Images
using StaticArrays
using FileIO
using Base.Threads

println("SPIRA - Metal Raytracer with Array Abstractions")
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

# Constants - use Float32 for better GPU compatibility
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

# Ray-sphere intersection for a batch of rays and spheres
function hit_spheres_batch(ray_origins::Metal.MtlArray{Float32, 2}, ray_dirs::Metal.MtlArray{Float32, 2}, 
                          sphere_data::Metal.MtlArray{Float32, 1}, num_spheres::Int)
    
    num_rays = size(ray_origins, 1)
    
    # Initialize hit results (hit?, t, sphere_index)
    hit_results = Metal.MtlArray{Float32}(undef, num_rays, 3)
    hit_results .= Metal.MtlArray([Float32(0.0) INF Float32(0.0)])
    
    # Initialize hit normals
    hit_normals = Metal.MtlArray{Float32}(undef, num_rays, 3)
    hit_normals .= Metal.MtlArray(zeros(Float32, num_rays, 3))
    
    # We need to process spheres on the CPU since scalar indexing isn't allowed with Metal arrays
    # Convert sphere data to CPU for processing
    sphere_data_cpu = Array(sphere_data)

    # For each sphere, check intersections with all rays
    for s in 1:num_spheres
        # Extract sphere data
        idx = (s-1) * 5 + 1
        center_x = sphere_data_cpu[idx]
        center_y = sphere_data_cpu[idx+1]
        center_z = sphere_data_cpu[idx+2]
        radius = sphere_data_cpu[idx+3]
        mat_idx = sphere_data_cpu[idx+4]
        
        # Calculate oc vector (ray_origin - sphere_center) for all rays
        oc_x = ray_origins[:, 1] .- center_x
        oc_y = ray_origins[:, 2] .- center_y
        oc_z = ray_origins[:, 3] .- center_z
        
        # Calculate quadratic coefficients for all rays
        a = ray_dirs[:, 1].^2 .+ ray_dirs[:, 2].^2 .+ ray_dirs[:, 3].^2
        half_b = oc_x .* ray_dirs[:, 1] .+ oc_y .* ray_dirs[:, 2] .+ oc_z .* ray_dirs[:, 3]
        c = oc_x.^2 .+ oc_y.^2 .+ oc_z.^2 .- radius^2
        
        # Calculate discriminant
        discriminant = half_b.^2 .- a .* c
        
        # Find rays that hit this sphere
        hit_mask = discriminant .> Float32(0.0)
        
        if any(hit_mask)
            # Calculate closest intersection for rays that hit
            sqrtd = sqrt.(discriminant[hit_mask])
            
            # Calculate both intersection points
            t1 = (-half_b[hit_mask] .- sqrtd) ./ a[hit_mask]
            t2 = (-half_b[hit_mask] .+ sqrtd) ./ a[hit_mask]
            
            # Find valid intersections
            valid_t1 = t1 .> Float32(0.001)
            valid_t2 = (t1 .<= Float32(0.001)) .& (t2 .> Float32(0.001))
            
            # Use t1 where valid, otherwise use t2 if valid
            t = copy(t1)
            t[.!valid_t1 .& valid_t2] = t2[.!valid_t1 .& valid_t2]
            
            # Find rays where this intersection is closer than previous ones
            closer_mask = t .< hit_results[hit_mask, 2]
            
            # Subset of rays that actually hit this sphere closer than before
            hits = hit_mask .& closer_mask
            
            if any(hits)
                # Update hit results for these rays
                hit_results[hits, 1] .= Float32(1.0)  # Hit
                hit_results[hits, 2] .= t[closer_mask]  # Distance
                hit_results[hits, 3] .= mat_idx  # Material index
                
                # Calculate hit positions and normals
                hit_positions_x = ray_origins[hits, 1] .+ t[closer_mask] .* ray_dirs[hits, 1]
                hit_positions_y = ray_origins[hits, 2] .+ t[closer_mask] .* ray_dirs[hits, 2]
                hit_positions_z = ray_origins[hits, 3] .+ t[closer_mask] .* ray_dirs[hits, 3]
                
                # Calculate normals
                hit_normals[hits, 1] .= hit_positions_x .- center_x
                hit_normals[hits, 2] .= hit_positions_y .- center_y
                hit_normals[hits, 3] .= hit_positions_z .- center_z
                
                # Normalize normals
                normal_lengths = sqrt.(hit_normals[hits, 1].^2 .+ hit_normals[hits, 2].^2 .+ hit_normals[hits, 3].^2)
                hit_normals[hits, 1] .= hit_normals[hits, 1] ./ normal_lengths
                hit_normals[hits, 2] .= hit_normals[hits, 2] ./ normal_lengths
                hit_normals[hits, 3] .= hit_normals[hits, 3] ./ normal_lengths
            end
        end
    end
    
    return hit_results, hit_normals
end

# Tone mapping and gamma correction (vectorized)
function tone_map_batch(colors::Metal.MtlArray{Float32, 2})
    # ACES filmic tone mapping (simplified)
    a = Float32(2.51)
    b = Float32(0.03)
    c = Float32(2.43)
    d = Float32(0.59)
    e = Float32(0.14)
    
    # Apply tone mapping to each color channel
    r = colors[:, 1]
    g = colors[:, 2]
    b = colors[:, 3]
    
    # ACES tone mapping
    r = (r .* (a .* r .+ b)) ./ (r .* (c .* r .+ d) .+ e)
    g = (g .* (a .* g .+ b)) ./ (g .* (c .* g .+ d) .+ e)
    b = (b .* (a .* b .+ b)) ./ (b .* (c .* b .+ d) .+ e)
    
    # Clamp to [0,1]
    r = clamp.(r, Float32(0.0), Float32(1.0))
    g = clamp.(g, Float32(0.0), Float32(1.0))
    b = clamp.(b, Float32(0.0), Float32(1.0))
    
    # Apply gamma correction
    r = sqrt.(r)
    g = sqrt.(g)
    b = sqrt.(b)
    
    return Metal.MtlArray(hcat(r, g, b))
end

# Copy scene data to GPU buffers
function copy_scene_to_gpu(scene::Scene)
    # Copy sphere data to a compact form for Metal.jl
    sphere_data = zeros(Float32, 5 * length(scene.spheres))
    for (i, sphere) in enumerate(scene.spheres)
        idx = (i-1) * 5 + 1
        sphere_data[idx] = sphere.center[1]
        sphere_data[idx+1] = sphere.center[2]
        sphere_data[idx+2] = sphere.center[3]
        sphere_data[idx+3] = sphere.radius
        sphere_data[idx+4] = Float32(sphere.material)
    end
    
    # Copy material data
    material_data = zeros(Float32, 10 * length(scene.materials))
    for (i, material) in enumerate(scene.materials)
        idx = (i-1) * 10 + 1
        material_data[idx] = material.albedo[1]
        material_data[idx+1] = material.albedo[2]
        material_data[idx+2] = material.albedo[3]
        material_data[idx+3] = material.emission[1]
        material_data[idx+4] = material.emission[2]
        material_data[idx+5] = material.emission[3]
        material_data[idx+6] = material.metallic
        material_data[idx+7] = material.roughness
        material_data[idx+8] = Float32(0.0)  # padding
        material_data[idx+9] = Float32(0.0)  # padding
    end
    
    # Create Metal arrays
    if has_metal
        sphere_gpu = Metal.MtlArray(sphere_data)
        material_gpu = Metal.MtlArray(material_data)
        return sphere_gpu, material_gpu, length(scene.spheres)
    else
        return sphere_data, material_data, length(scene.spheres)
    end
end

# Generate rays for a camera
function generate_rays(width, height, camera)
    # Create arrays to hold rays
    rays_origin = fill(camera.origin, height * width)
    rays_dir = Array{Vec3}(undef, height * width)
    
    idx = 1
    for j in 1:height
        for i in 1:width
            # Calculate ray direction
            u = Float32((i - 1) / (width - 1))
            v = Float32((j - 1) / (height - 1))
            
            ray_dir = camera.lower_left_corner + u*camera.horizontal + v*camera.vertical - camera.origin
            rays_dir[idx] = normalize(ray_dir)
            
            idx += 1
        end
    end
    
    # Convert to proper format for GPU
    ray_origins = zeros(Float32, height*width, 3)
    ray_directions = zeros(Float32, height*width, 3)
    
    for i in 1:height*width
        ray_origins[i, 1] = rays_origin[i][1]
        ray_origins[i, 2] = rays_origin[i][2]
        ray_origins[i, 3] = rays_origin[i][3]
        
        ray_directions[i, 1] = rays_dir[i][1]
        ray_directions[i, 2] = rays_dir[i][2]
        ray_directions[i, 3] = rays_dir[i][3]
    end
    
    # Create Metal arrays if available
    if has_metal
        ray_origins_gpu = Metal.MtlArray(ray_origins)
        ray_directions_gpu = Metal.MtlArray(ray_directions)
        return ray_origins_gpu, ray_directions_gpu
    else
        return ray_origins, ray_directions
    end
end

# Simulate basic ray tracing with Metal arrays
function render_with_metal(width, height, scene, camera; samples_per_pixel=16, max_depth=4)
    # Only proceed if Metal is available
    if !has_metal
        println("Metal not available, falling back to CPU rendering.")
        return render_with_cpu(width, height, scene, camera, 
                              samples_per_pixel=samples_per_pixel,
                              max_depth=max_depth)
    end
    
    # Copy scene data to GPU
    sphere_data, material_data, num_spheres = copy_scene_to_gpu(scene)
    
    # Create image buffer
    img_buffer = zeros(Float32, height, width, 3)
    
    # For each sample
    for sample in 1:samples_per_pixel
        # Generate camera rays with jitter
        ray_origins, ray_directions = generate_rays(width, height, camera)
        
        # Add jitter for anti-aliasing
        if sample > 1
            # Add small random offset to each ray for anti-aliasing
            jitter_u = Metal.MtlArray(rand(Float32, height*width) ./ width)
            jitter_v = Metal.MtlArray(rand(Float32, height*width) ./ height)
            
            # Apply jitter to ray directions (simplified)
            # In a full implementation, we'd regenerate the rays with proper jitter
        end
        
        # For each bounce depth
        # This is simplified - a full implementation would trace rays recursively
        # through multiple bounces, handling reflection and refraction
        for depth in 1:1  # Just do primary ray for simplicity
            # Trace rays against spheres
            hit_results, hit_normals = hit_spheres_batch(ray_origins, ray_directions, sphere_data, num_spheres)
            
            # Convert to CPU for processing
            hits_cpu = Array(hit_results)
            normals_cpu = Array(hit_normals)
            
            # Process hits on CPU (in a real implementation, more would be on GPU)
            colors = zeros(Float32, height*width, 3)
            for i in 1:height*width
                if hits_cpu[i, 1] > 0.0  # Ray hit something
                    mat_idx = Int(hits_cpu[i, 3])
                    
                    # Get material properties
                    mat_offset = (mat_idx-1) * 10 + 1
                    albedo = [
                        Array(material_data)[mat_offset],
                        Array(material_data)[mat_offset+1],
                        Array(material_data)[mat_offset+2]
                    ]
                    emission = [
                        Array(material_data)[mat_offset+3],
                        Array(material_data)[mat_offset+4],
                        Array(material_data)[mat_offset+5]
                    ]
                    
                    # For simplicity, just return albedo + emission
                    colors[i, 1] = albedo[1] + emission[1]
                    colors[i, 2] = albedo[2] + emission[2]
                    colors[i, 3] = albedo[3] + emission[3]
                else
                    # Background color (sky gradient)
                    dir_y = Array(ray_directions)[i, 2]
                    t = 0.5f0 * (dir_y + 1.0f0)
                    colors[i, 1] = (1.0f0 - t) + t * 0.5f0  # r
                    colors[i, 2] = (1.0f0 - t) + t * 0.7f0  # g
                    colors[i, 3] = (1.0f0 - t) + t * 1.0f0  # b
                end
            end
            
            # Convert back to Metal array
            colors_gpu = Metal.MtlArray(colors)
            
            # Apply tone mapping
            mapped_colors = tone_map_batch(colors_gpu)
            
            # Accumulate to image buffer
            for i in 1:height*width
                row = div(i-1, width) + 1
                col = mod1(i, width)
                
                img_buffer[height - row + 1, col, 1] += Array(mapped_colors)[i, 1]
                img_buffer[height - row + 1, col, 2] += Array(mapped_colors)[i, 2]
                img_buffer[height - row + 1, col, 3] += Array(mapped_colors)[i, 3]
            end
        end
        
        println("Completed sample $sample / $samples_per_pixel")
    end
    
    # Average samples
    img_buffer = img_buffer ./ samples_per_pixel
    
    # Create the final RGB image
    img = Array{RGB{Float32}}(undef, height, width)
    for j in 1:height
        for i in 1:width
            img[j,i] = RGB{Float32}(img_buffer[j,i,1], img_buffer[j,i,2], img_buffer[j,i,3])
        end
    end
    
    return img
end

# CPU fallback implementation
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
                target = hit_point + hit_normal + normalize(rand(Vec3) - 0.5)
                scattered = Ray(hit_point, normalize(target - hit_point))
                return material.albedo .* trace_ray(scattered, depth - 1) .* Float32(0.5)
            else
                # Metallic reflection
                reflected = ray.direction - 2.0f0 * dot(ray.direction, hit_normal) * hit_normal
                scattered = Ray(hit_point, normalize(reflected + material.roughness * (rand(Vec3) - 0.5)))
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
              output_path::String="minimal_metal_render.png")
    
    # Time the rendering
    start_time = time()
    
    # Try Metal first, fallback to CPU
    if has_metal
        println("Rendering with Metal GPU...")
        img = render_with_metal(width, height, scene, camera,
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
          output_path="minimal_metal_render.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end