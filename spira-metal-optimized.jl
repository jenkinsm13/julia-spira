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

# --- MSL-compatible Julia Struct Definitions ---
# These structs are designed to match the layout of their MSL counterparts
# for direct memory mapping to GPU buffers.

struct Camera_jl
    origin::SVector{3, Float32}          # 12 bytes
    lower_left_corner::SVector{3, Float32} # 12 bytes
    horizontal::SVector{3, Float32}      # 12 bytes
    vertical::SVector{3, Float32}        # 12 bytes
end # Total: 48 bytes (multiple of 16, likely fine)

struct Sphere_jl
    center::SVector{3, Float32} # 12 bytes
    radius::Float32             # 4 bytes
    material_index::UInt32      # 4 bytes
    # MSL float3 in struct array might be padded to 16B, making sphere 16+4+4=24,
    # then padded to next 16B multiple (32B). For now, assume tight packing.
    # Julia sizeof(Sphere_jl) = 20. If issues, add padding:
    # _pad1::UInt32 # 4 bytes
    # _pad2::UInt32 # 4 bytes
    # _pad3::UInt32 # 4 bytes -> total 32
end # Expected size: 20 bytes. May need padding if MSL differs.

struct Material_jl
    albedo::SVector{3, Float32}   # 12 bytes
    emission::SVector{3, Float32} # 12 bytes
    metallic::Float32             # 4 bytes
    roughness::Float32            # 4 bytes
end # Total: 32 bytes (multiple of 16, likely fine)

struct RNGState_jl
    state::UInt32
end # Total: 4 bytes

struct RenderParams_jl
    image_width::UInt32
    image_height::UInt32
    max_depth::UInt32
    current_sample_index::UInt32
    num_spheres::UInt32
    num_materials::UInt32
    # Ensure this struct's size is a multiple of 4, preferably 16 for constant buffers
    # _pad1::UInt32 # if needed
    # _pad2::UInt32 # if needed
end # Total: 6 * 4 = 24 bytes. May need padding to 32 for MSL constant buffer best practice.

# Mutable struct to hold rendering state that needs to persist across GPU operations
mutable struct RenderState
    img_buffer_gpu::Metal.MtlMatrix{Float32}
    rng_states_gpu::Metal.MtlVector{UInt32}
    # Add other variables here if they also prove to be unstable (e.g. width, height, spp)
end

# --- GPU-compatible Random Number Generation ---

# Basic xorshift32 step. Metal.jl should be able to JIT this for the GPU.
@inline function xorshift32_step(state::UInt32)::UInt32
    state = xor(state, state << 13)
    state = xor(state, state >> 17)
    state = xor(state, state << 5)
    return state
end

# Convert UInt32 to Float32 in [0,1).
@inline function gpu_rand_float32_from_state(state::UInt32)::Float32
    # Division by typemax(UInt32) might give 1.0 exactly.
    # To ensure [0,1), can use (state / (typemax(UInt32) + 1.0f0))
    # or simply state * (1.0f0 / Float32(typemax(UInt32)))
    # For simplicity and common practice:
    return Float32(state / typemax(UInt32))
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

# --- End MSL-compatible Julia Struct Definitions ---

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

# --- New GPU Kernel for Initial Ray Generation ---
function gpu_generate_initial_sample_rays_kernel!(
    out_ray_origins_gpu::Metal.MtlMatrix{Float32},    # Output: (width*height, 3)
    out_ray_directions_gpu::Metal.MtlMatrix{Float32}, # Output: (width*height, 3)
    # Camera parameters (passed as individual fields for easier JITting by Metal.jl)
    cam_origin_x::Float32, cam_origin_y::Float32, cam_origin_z::Float32,
    cam_llc_x::Float32, cam_llc_y::Float32, cam_llc_z::Float32,
    cam_horiz_x::Float32, cam_horiz_y::Float32, cam_horiz_z::Float32,
    cam_vert_x::Float32, cam_vert_y::Float32, cam_vert_z::Float32,
    image_width::UInt32,
    image_height::UInt32,
    rng_states_gpu::Metal.MtlVector{UInt32}, # Input/Output: (width*height)
    current_sample_idx::UInt32,
    jitter_scale_factor::Float32
)
    idx = Metal.thread_position_in_grid_1d() # 1-based linear index for the pixel

    # Bounds check (usually not strictly needed if launch grid matches data size)
    if idx > image_width * image_height
        return
    end

    # Retrieve and update RNG state for this pixel/thread
    # Incorporate sample_idx to ensure different random sequences per sample per pixel
    # A simple way is to XOR, or add and then xorshift.
    # Adding sample_idx before the first xorshift effectively changes the seed for this sample.
    state = xorshift32_step(rng_states_gpu[idx] + current_sample_idx) # Initialize state for this sample

    # Generate two random numbers for jitter
    state = xorshift32_step(state)
    rand1 = gpu_rand_float32_from_state(state)
    state = xorshift32_step(state)
    rand2 = gpu_rand_float32_from_state(state)
    
    # Store updated state back
    rng_states_gpu[idx] = state

    # Convert linear idx to 2D pixel column and row (0-based for calculation)
    # Ray generation in the original code implies (u,v) mapping where v=0 is bottom scanline.
    # Here, mimicking the original CPU generate_ray_data:
    # j (height) from 1 to height, i (width) from 1 to width
    # The linear idx corresponds to (j-1)*width + i
    # So, col_1based = mod1(idx, image_width)
    #     row_1based = div(idx-1, image_width) + 1
    # These are 1-based. For u,v calculation (0 to N-1):
    col_0based = (idx - 1) % image_width  # 0 to image_width-1
    row_0based = (idx - 1) ÷ image_width  # 0 to image_height-1

    # Normalized device coordinates (u,v) for pixel center
    # (i-1)/(width-1) and (j-1)/(height-1) in original
    u_center = Float32(col_0based) / Float32(image_width - 1)
    v_center = Float32(row_0based) / Float32(image_height - 1)

    # Apply jitter
    # Jitter scale should be small, e.g., 1.0f0 for full pixel area, 0.5f0 for half pixel.
    # The original CPU jitter used `jitter_scale = Float32(0.5 / width)`.
    # Let's assume jitter_scale_factor = 0.5f0 (meaning half pixel extent).
    # So jitter magnitude is (rand - 0.5) * (1.0 / image_width or 1.0 / image_height) effectively.
    # The prompt `jitter_scale_factor::Float32` implies this is the multiplier for (rand-0.5).
    # The scaling by 1/width or 1/height should be part of the factor if it's fixed.
    # Original jitter: (rand(Float32) - 0.5f0) * (0.5f0 / width_pixels)
    # So here, `jitter_scale_factor` is likely 0.5f0 if we mean to jitter within the pixel.
    # And the 1/width, 1/height is applied.
    pixel_width_uv = 1.0f0 / Float32(image_width -1)
    pixel_height_uv = 1.0f0 / Float32(image_height -1)

    u_jittered = u_center + (rand1 - 0.5f0) * jitter_scale_factor * pixel_width_uv
    v_jittered = v_center + (rand2 - 0.5f0) * jitter_scale_factor * pixel_height_uv
    
    # Calculate ray direction using camera parameters
    dir_x = cam_llc_x + u_jittered * cam_horiz_x + v_jittered * cam_vert_x - cam_origin_x
    dir_y = cam_llc_y + u_jittered * cam_horiz_y + v_jittered * cam_vert_y - cam_origin_y
    dir_z = cam_llc_z + u_jittered * cam_horiz_z + v_jittered * cam_vert_z - cam_origin_z
    
    # Normalize direction
    len_sq = dir_x*dir_x + dir_y*dir_y + dir_z*dir_z
    inv_len = if len_sq > 0.0f0 sqrt(1.0f0 / len_sq) else 0.0f0 end # rsqrt(len_sq)
    
    # Store origin and direction
    out_ray_origins_gpu[idx, 1] = cam_origin_x
    out_ray_origins_gpu[idx, 2] = cam_origin_y
    out_ray_origins_gpu[idx, 3] = cam_origin_z
    
    out_ray_directions_gpu[idx, 1] = dir_x * inv_len
    out_ray_directions_gpu[idx, 2] = dir_y * inv_len
    out_ray_directions_gpu[idx, 3] = dir_z * inv_len
    
    return
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

# Kernel for ray scattering (reflection/diffusion)
function gpu_scatter_kernel!(
    out_new_ray_origins::Metal.MtlMatrix{Float32},     # Output: new origins for scattered rays
    out_new_ray_directions::Metal.MtlMatrix{Float32},  # Output: new directions (pre-filled with current, updated on scatter)
    kernel_hit_results::Metal.MtlMatrix{Float32},        # Input: [hit?, t, material_idx]
    kernel_current_ray_directions::Metal.MtlMatrix{Float32}, # Input: directions before this scatter
    kernel_hit_normals::Metal.MtlMatrix{Float32},        # Input: normals at hit points
    kernel_in_hit_points::Metal.MtlMatrix{Float32},      # Input: geometric hit points (source for new origins)
    kernel_material_data::Metal.MtlVector{Float32},      # Input: flattened material properties
    kernel_rng_states::Metal.MtlVector{UInt32}           # Input/Output: RNG states per ray
)
    idx = Metal.thread_position_in_grid_1d() # Current ray index (1 to num_rays)
    grid_dim = Metal.grid_size_1d()          # Total number of rays

    if idx > grid_dim
        return
    end

    # Set the new origin for this ray from its hit point.
    # If the ray didn't hit (kernel_hit_results[idx,1] == 0), kernel_in_hit_points[idx,:]
    # would be (0,0,0) due to initialization in gpu_ray_sphere_intersection.
    # The corresponding out_new_ray_directions[idx,:] is already pre-filled with
    # kernel_current_ray_directions[idx,:] and will remain unchanged if no hit.
    out_new_ray_origins[idx, 1] = kernel_in_hit_points[idx, 1]
    out_new_ray_origins[idx, 2] = kernel_in_hit_points[idx, 2]
    out_new_ray_origins[idx, 3] = kernel_in_hit_points[idx, 3]

    # Only process rays that hit something
    if kernel_hit_results[idx, 1] > Float32(0.0)
        # Get material properties
        mat_idx = Int(kernel_hit_results[idx, 3])
        mat_base_idx = (mat_idx - 1) * 8 + 1 # Assuming 8 floats per material
        metallic = kernel_material_data[mat_base_idx + 6]
        roughness = kernel_material_data[mat_base_idx + 7]

        # Get normal and incident direction for the current ray
        nx = kernel_hit_normals[idx, 1]
        ny = kernel_hit_normals[idx, 2]
        nz = kernel_hit_normals[idx, 3]

        dx = kernel_current_ray_directions[idx, 1]
        dy = kernel_current_ray_directions[idx, 2]
        dz = kernel_current_ray_directions[idx, 3]

        dot_prod = dx * nx + dy * ny + dz * nz

        # Metallic reflection
        if metallic > Float32(0.0)
            rx = dx - Float32(2.0) * dot_prod * nx
            ry = dy - Float32(2.0) * dot_prod * ny
            rz = dz - Float32(2.0) * dot_prod * nz

            if roughness > Float32(0.0)
                state = kernel_rng_states[idx]
                state = xorshift32_step(state)
                rand_val1 = gpu_rand_float32_from_state(state)
                state = xorshift32_step(state)
                rand_val2 = gpu_rand_float32_from_state(state)
                state = xorshift32_step(state)
                rand_val3 = gpu_rand_float32_from_state(state)
                kernel_rng_states[idx] = state

                rnd_x = rand_val1 - Float32(0.5)
                rnd_y = rand_val2 - Float32(0.5)
                rnd_z = rand_val3 - Float32(0.5)

                norm_len_rnd = sqrt(rnd_x^2 + rnd_y^2 + rnd_z^2)
                if norm_len_rnd > Float32(1e-5)
                    inv_norm_len_rnd = Float32(1.0) / norm_len_rnd
                    rnd_x *= inv_norm_len_rnd
                    rnd_y *= inv_norm_len_rnd
                    rnd_z *= inv_norm_len_rnd
                end

                rx += roughness * rnd_x
                ry += roughness * rnd_y
                rz += roughness * rnd_z

                inv_len_final = Float32(1.0) / sqrt(rx^2 + ry^2 + rz^2)
                rx *= inv_len_final
                ry *= inv_len_final
                rz *= inv_len_final
            end
            out_new_ray_directions[idx, 1] = rx
            out_new_ray_directions[idx, 2] = ry
            out_new_ray_directions[idx, 3] = rz
        else # Diffuse reflection (Lambertian)
            state = kernel_rng_states[idx]
            local local_rnd_x, local_rnd_y, local_rnd_z # Ensure these are distinct per thread
            while true # Rejection sampling for random point in unit sphere
                state = xorshift32_step(state)
                r1 = gpu_rand_float32_from_state(state)
                state = xorshift32_step(state)
                r2 = gpu_rand_float32_from_state(state)
                state = xorshift32_step(state)
                r3 = gpu_rand_float32_from_state(state)

                local_rnd_x = r1 * Float32(2.0) - Float32(1.0)
                local_rnd_y = r2 * Float32(2.0) - Float32(1.0)
                local_rnd_z = r3 * Float32(2.0) - Float32(1.0)
                if (local_rnd_x^2 + local_rnd_y^2 + local_rnd_z^2) <= Float32(1.0)
                    break
                end
            end
            kernel_rng_states[idx] = state

            diffuse_dir_x = nx + local_rnd_x
            diffuse_dir_y = ny + local_rnd_y
            diffuse_dir_z = nz + local_rnd_z

            len_sq_diffuse = diffuse_dir_x^2 + diffuse_dir_y^2 + diffuse_dir_z^2
            if len_sq_diffuse < Float32(1e-5)
                out_new_ray_directions[idx, 1] = nx
                out_new_ray_directions[idx, 2] = ny
                out_new_ray_directions[idx, 3] = nz
            else
                inv_len_diffuse = Float32(1.0) / sqrt(len_sq_diffuse)
                out_new_ray_directions[idx, 1] = diffuse_dir_x * inv_len_diffuse
                out_new_ray_directions[idx, 2] = diffuse_dir_y * inv_len_diffuse
                out_new_ray_directions[idx, 3] = diffuse_dir_z * inv_len_diffuse
            end
        end
    end
    # If kernel_hit_results[idx, 1] == 0.0f0 (no hit for this ray),
    # out_new_ray_directions[idx,:] remains as it was pre-filled (i.e., kernel_current_ray_directions[idx,:]).
    # out_new_ray_origins[idx,:] is already set from kernel_in_hit_points[idx,:].

    return
end

# Wrapper function to launch the gpu_scatter_kernel!
function gpu_reflect_rays(hit_results_param::Metal.MtlArray{Float32, 2}, 
                         current_ray_directions_param::Metal.MtlArray{Float32, 2},
                         hit_normals_param::Metal.MtlArray{Float32, 2},
                         in_hit_points_param::Metal.MtlArray{Float32, 2}, 
                         material_data_param::Metal.MtlArray{Float32, 1},
                         rng_states_gpu_param::Metal.MtlVector{UInt32})
    
    num_rays = size(current_ray_directions_param, 1)

    # Allocate output arrays
    # out_new_ray_origins will be fully written by the kernel.
    out_new_ray_origins_gpu = Metal.MtlArray{Float32}(undef, num_rays, 3)
    # out_new_ray_directions is initialized with current directions, kernel updates if scatter.
    out_new_ray_directions_gpu = copy(current_ray_directions_param)

    if num_rays > 0
        threads_per_group = min(num_rays, 256) # Typical max group size
        num_groups = ceil(Int, num_rays / threads_per_group)

        Metal.@sync @metal threads=num_rays groups=num_groups gpu_scatter_kernel!(
            out_new_ray_origins_gpu,
            out_new_ray_directions_gpu,
            hit_results_param,
            current_ray_directions_param, # Passed as kernel_current_ray_directions
            hit_normals_param,
            in_hit_points_param,          # Passed as kernel_in_hit_points
            material_data_param,
            rng_states_gpu_param
        )
    end
    
    return out_new_ray_origins_gpu, out_new_ray_directions_gpu
end

# Kernel for final image averaging
function average_image_kernel!(
    image_buffer::Metal.MtlMatrix{Float32}, # Shape: (total_pixels, 3)
    num_samples_float::Float32,
    total_pixels::UInt32 # Explicitly pass total_pixels for bounds
)
    idx = Metal.thread_position_in_grid_1d() # 1-based linear index for the pixel

    if idx <= total_pixels # Ensure we are within the bounds of actual pixel data
        image_buffer[idx, 1] /= num_samples_float
        image_buffer[idx, 2] /= num_samples_float
        image_buffer[idx, 3] /= num_samples_float
    end
    return
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

function finalize_image_from_gpu_buffer(
    img_buffer_gpu_arg::Metal.MtlMatrix{Float32}, 
    p_width::Int, 
    p_height::Int, 
    p_samples_per_pixel::Int
)
    # Average samples on the GPU using a kernel
    if (p_width * p_height) > 0 && p_samples_per_pixel > 0
        num_samples_val_float32 = Float32(p_samples_per_pixel)
        # Kernel launch configuration (1D for pixels)
        current_total_pixels = p_width * p_height # Use local calculation for clarity
        threads_avg_kernel = current_total_pixels
        groups_avg_kernel = ceil(Int, current_total_pixels / 256) # Using a common group size

        Metal.@sync @metal threads=threads_avg_kernel groups=groups_avg_kernel average_image_kernel!(
            img_buffer_gpu_arg,
            num_samples_val_float32,
            UInt32(current_total_pixels)
        )
    elseif p_samples_per_pixel == 0 && (p_width * p_height) > 0 # Avoid division by zero
        img_buffer_gpu_arg .= 0.0f0 # Or some other defined state
    end
    
    # Transfer final averaged image data from GPU to CPU
    final_colors_cpu_flat = Array(img_buffer_gpu_arg) # Shape: (width*height, 3)
                                                  # Order: ray (i=1,j=1 bottom-left) is index 1
                                                  # ray (i=width,j=1 bottom-right) is index width
                                                  # ray (i=1,j=height top-left) is index (height-1)*width+1
    
    # Create final RGB image on CPU, reconstructing from flat array
    # The output image `img` should have img[1,1] as top-left pixel.
    img = Array{RGB{Float32}}(undef, p_height, p_width)
    for j_img in 1:p_height  # Output image row: 1=top, height=bottom
        for i_img in 1:p_width # Output image col: 1=left, width=right
            # Corresponding ray generation indices (j_ray=1 is bottom, i_ray=1 is left)
            i_ray = i_img
            j_ray = p_height - j_img + 1 
            
            # Index in the flat array
            flat_idx = (j_ray - 1) * p_width + i_ray
            
            img[j_img, i_img] = RGB{Float32}(
                final_colors_cpu_flat[flat_idx, 1],
                final_colors_cpu_flat[flat_idx, 2],
                final_colors_cpu_flat[flat_idx, 3]
            )
        end
    end
    
    return img
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
    
    # Convert Camera to Camera_jl for GPU
    camera_jl = Camera_jl(camera.origin, camera.lower_left_corner, camera.horizontal, camera.vertical)

    # Initialize RNG states on GPU
    # Seed with random numbers from CPU, or a deterministic sequence.
    rng_states_gpu = Metal.MtlArray(rand(UInt32, width * height))
    # Or for deterministic: rng_states_gpu = Metal.MtlArray(UInt32.((1:(width*height)) .% typemax(UInt32)))
    
    # Create image buffer on the GPU
    img_buffer_gpu_init = Metal.zeros(Float32, width * height, 3) # Use a distinct name for initialization

    # Initialize RenderState
    render_state = RenderState(img_buffer_gpu_init, rng_states_gpu)

    # Pre-allocate GPU arrays for current rays (these are transient per sample/depth)
    current_ray_origins_gpu = Metal.MtlArray{Float32}(undef, width * height, 3)
    current_ray_directions_gpu = Metal.MtlArray{Float32}(undef, width * height, 3)
    
    # Get number of spheres
    num_spheres = length(scene.spheres)
    
    # Determine kernel launch parameters
    total_pixels = width * height
    threads_per_group = 256 # Common choice, can be tuned
    num_groups = ceil(Int, total_pixels / threads_per_group)

    for sample in 1:samples_per_pixel
        # Generate initial rays for this sample directly on GPU
        Metal.@sync @metal threads=total_pixels groups=num_groups gpu_generate_initial_sample_rays_kernel!(
            current_ray_origins_gpu, current_ray_directions_gpu,
            camera_jl.origin[1], camera_jl.origin[2], camera_jl.origin[3],
            camera_jl.lower_left_corner[1], camera_jl.lower_left_corner[2], camera_jl.lower_left_corner[3],
            camera_jl.horizontal[1], camera_jl.horizontal[2], camera_jl.horizontal[3],
            camera_jl.vertical[1], camera_jl.vertical[2], camera_jl.vertical[3],
            UInt32(width), UInt32(height),
            render_state.rng_states_gpu, # Use from RenderState
            UInt32(sample), # current_sample_idx
            0.5f0           # jitter_scale_factor (half pixel extent for jitter)
        )
        
        # Initialize contribution for each ray for this sample
        contribution_gpu = Metal.ones(Float32, width * height) # Reset for each sample path
        
        # Path tracing loop for current sample
        for depth in 1:max_depth
            # Trace rays against spheres
            hit_results, hit_points, hit_normals = gpu_ray_sphere_intersection(
                current_ray_origins_gpu, current_ray_directions_gpu, sphere_data_gpu, num_spheres
            )
            
            # Break if no rays hit anything (check on GPU, transfer only sum)
            if Metal.sum(hit_results[:, 1]) == 0.0f0
                # If nothing was hit by any ray at this depth, remaining paths contribute sky.
                # Shade with current ray directions for sky color.
                # This logic needs to be nuanced: if some rays hit and others miss,
                # the ones that miss should get sky color. gpu_shade_rays handles this.
                # If ALL miss, then all will get sky color from gpu_shade_rays.
                # The break here is if literally zero rays found a next intersection.
                break 
            end
            
            # For rays that hit, generate new rays (reflection/refraction)
            # Note: gpu_reflect_rays updates current_ray_origins_gpu with hit_points
            # and current_ray_directions_gpu with new scatter directions.
            current_ray_origins_gpu, current_ray_directions_gpu = gpu_reflect_rays(
                hit_results, current_ray_directions_gpu, hit_normals, hit_points, material_data_gpu, render_state.rng_states_gpu # Use from RenderState
            )
            
            # Apply attenuation. For rays that missed, their contribution should ideally go to 0
            # or be handled so they don't contribute to surface properties.
            # Current simple model: halve contribution for all active paths.
            # A mask could be used here: contribution_gpu[hit_mask] .*= 0.5f0
            # For now, global attenuation:
            contribution_gpu .*= Float32(0.5) 
            
            # If at max depth, we'll shade these rays
            if depth == max_depth
                # Shade rays and accumulate to image buffer
                # Pass current_ray_directions_gpu for sky color calculation if no hit
                colors = gpu_shade_rays(hit_results, current_ray_directions_gpu, material_data_gpu, contribution_gpu)
                
                # Apply tone mapping
                mapped_colors = gpu_tone_map(colors) # mapped_colors is MtlArray (width*height, 3)
                
                # Accumulate colors directly on the GPU
                render_state.img_buffer_gpu .+= mapped_colors # Use from RenderState
            end
        end
    end
    
    println("Completed sample $sample / $samples_per_pixel")
    
    return finalize_image_from_gpu_buffer(render_state.img_buffer_gpu, width, height, samples_per_pixel) # Use from RenderState
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
        println("Rendering with Metal GPU (GPU-side accumulation)...")
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