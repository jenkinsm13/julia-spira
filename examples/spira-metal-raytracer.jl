#!/usr/bin/env julia

# SPIRA - Optimized Metal GPU raytracer with direct-to-EXR output pipeline
# Implementation with Float32 for GPU compatibility

using LinearAlgebra
using Random
using Images
using StaticArrays
using FileIO
using Base.Threads

# Try to load Metal.jl (for macOS)
global has_metal = false
try
    using Metal
    global has_metal = true
    println("Metal.jl loaded successfully - GPU acceleration enabled")

    device = Metal.device()
    println("Metal device: ", device)
catch e
    println("Metal.jl not available: $e")
    error("GPU acceleration required - Metal.jl is mandatory")
end

# Try to load OpenEXR
has_openexr = false
try
    using OpenEXR
    global has_openexr = true
    println("OpenEXR loaded - EXR export enabled")
catch e
    println("OpenEXR not available: $e")
    error("OpenEXR is required for this renderer")
end

# Type aliases
const Vec3 = SVector{3, Float32}
const Point3 = Vec3
const Color = Vec3
const Mat3 = SMatrix{3, 3, Float32}

# Constants - use Float32 values but without the f0 suffix
const INF = Float32(1e20)
const EPS = Float32(1e-6)
const BLACK = Vec3(0.0, 0.0, 0.0)
const WHITE = Vec3(1.0, 1.0, 1.0)

# Ray structure
struct Ray
    origin::Point3
    direction::Vec3
    inv_direction::Vec3
    
    Ray(origin::Point3, direction::Vec3) = new(origin, normalize(direction), 1.0f0 ./ normalize(direction))
end

function at(ray::Ray, t::Real)
    return ray.origin + t * ray.direction
end

# Triangle primitive with precomputed data for faster intersection
struct Triangle
    v1::Point3
    v2::Point3
    v3::Point3
    e1::Vec3  # Precomputed edges for Möller–Trumbore algorithm
    e2::Vec3
    normal::Vec3  # Precomputed normal
    material_index::Int32 # Index into the scene's material list
    
    function Triangle(v1::Point3, v2::Point3, v3::Point3, material_index::Int=1) # Default to material 1
        e1 = v2 - v1
        e2 = v3 - v1
        normal = normalize(cross(e1, e2))
        new(v1, v2, v3, e1, e2, normal, Int32(material_index))
    end
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
    u::Vec3
    v::Vec3
    w::Vec3
    lens_radius::Float32
    
    function Camera(lookfrom::Point3, lookat::Point3, vup::Vec3, vfov::Real, 
                   aspect_ratio::Real, aperture::Real=0.0, focus_dist::Real=1.0)
        theta = Float32(deg2rad(vfov))
        h = tan(theta/2.0f0) # Ensure Float32 math
        viewport_height = 2.0f0 * h
        viewport_width = Float32(aspect_ratio) * viewport_height
        
        w = normalize(lookfrom - lookat)
        u = normalize(cross(vup, w))
        v = cross(w, u)
        
        origin = lookfrom
        horizontal = focus_dist * viewport_width * u
        vertical = focus_dist * viewport_height * v
        lower_left_corner = origin - horizontal/2.0f0 - vertical/2.0f0 - focus_dist * w # Ensure Float32 math
        lens_radius = Float32(aperture / 2.0f0) # Ensure Float32 math
        
        new(origin, lower_left_corner, horizontal, vertical, u, v, w, lens_radius)
    end
end

# --- BVH and AABB related structures and functions ---

# Axis-Aligned Bounding Box structure
struct AABB
    min_bounds::Point3
    max_bounds::Point3

    AABB() = new(Point3(INF, INF, INF), Point3(-INF, -INF, -INF)) # Empty AABB
    AABB(p::Point3) = new(p, p)
    AABB(p1::Point3, p2::Point3) = new(min.(p1, p2), max.(p1, p2))
end

# Get AABB for a single triangle
function get_triangle_aabb(triangle::Triangle)::AABB
    min_b = min.(triangle.v1, triangle.v2, triangle.v3)
    max_b = max.(triangle.v1, triangle.v2, triangle.v3)
    return AABB(min_b, max_b)
end

# Merge two AABBs
function merge_aabbs(box1::AABB, box2::AABB)::AABB
    min_b = min.(box1.min_bounds, box2.min_bounds)
    max_b = max.(box1.max_bounds, box2.max_bounds)
    return AABB(min_b, max_b)
end

# Efficient Ray-AABB intersection test for GPU (single ray)
@inline function hit_aabb(ray_origin::Point3, ray_inv_direction::Vec3, 
                        aabb_min::Point3, aabb_max::Point3,
                        t_min_input::Float32, t_max_input::Float32)::Tuple{Bool, Float32, Float32}
    
    tx1 = (aabb_min[1] - ray_origin[1]) * ray_inv_direction[1]
    tx2 = (aabb_max[1] - ray_origin[1]) * ray_inv_direction[1]
    tmin_current = min(tx1, tx2)
    tmax_current = max(tx1, tx2)

    ty1 = (aabb_min[2] - ray_origin[2]) * ray_inv_direction[2]
    ty2 = (aabb_max[2] - ray_origin[2]) * ray_inv_direction[2]
    tmin_current = max(tmin_current, min(ty1, ty2))
    tmax_current = min(tmax_current, max(ty1, ty2))

    tz1 = (aabb_min[3] - ray_origin[3]) * ray_inv_direction[3]
    tz2 = (aabb_max[3] - ray_origin[3]) * ray_inv_direction[3]
    tmin_current = max(tmin_current, min(tz1, tz2))
    tmax_current = min(tmax_current, max(tz1, tz2))

    if tmax_current >= tmin_current && tmax_current >= t_min_input && tmin_current <= t_max_input
        return true, max(tmin_current, t_min_input), min(tmax_current, t_max_input)
    else
        return false, INF, -INF
    end
end

# GPU-friendly BVH Node structure
struct BVHNodeGPU
    aabb_min::Point3
    aabb_max::Point3
    # If is_leaf_node_flag == true:
    #   payload1 is tri_offset (into bvh_indices_gpu array for this node's triangles)
    #   payload2 is num_triangles in this leaf
    # If is_leaf_node_flag == false (i.e., it's an inner node):
    #   payload1 is left_child_idx (index into bvh_nodes_gpu array)
    #   payload2 is right_child_idx (index into bvh_nodes_gpu array)
    payload1::Int32
    payload2::Int32
    is_leaf_node_flag::Bool # true for leaf, false for inner node
end

# --- End of BVH and AABB structures ---

# Helper function to reflect a ray - used by all implementations
@inline function reflect(v::Vec3, n::Vec3)
    return v - 2.0f0 * dot(v, n) * n # Ensure Float32 math
end

# Functions to generate random values
function random_in_unit_sphere()
    while true
        p = Vec3(2.0f0*rand(Float32) - 1.0f0, 2.0f0*rand(Float32) - 1.0f0, 2.0f0*rand(Float32) - 1.0f0)
        if dot(p, p) < 1.0f0
            return p
        end
    end
end

function random_unit_vector()
    return normalize(random_in_unit_sphere())
end

# Load OBJ file
function load_obj(filepath::String, material_index::Int; scale::Float32=1.0f0, translation::Point3=Point3(0.0f0, 0.0f0, 0.0f0))
    vertices = Point3[]
    faces = Vector{Int}[]
    
    open(filepath, "r") do file
        for line in eachline(file)
            if startswith(line, "v ")
                parts = split(line)
                x = parse(Float32, parts[2]) * scale + translation[1]
                y = parse(Float32, parts[3]) * scale + translation[2]
                z = parse(Float32, parts[4]) * scale + translation[3]
                push!(vertices, Point3(x, y, z))
            elseif startswith(line, "f ")
                parts = split(line)
                face = Int[]
                for i in 2:length(parts)
                    idx_str = split(parts[i], "/")[1]
                    push!(face, parse(Int, idx_str))
                end
                push!(faces, face)
            end
        end
    end
    
    triangles = Triangle[]
    for face in faces
        if length(face) == 3
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            v3 = vertices[face[3]]
            push!(triangles, Triangle(v1, v2, v3, material_index))
        elseif length(face) > 3
            v1 = vertices[face[1]]
            for i in 3:length(face)
                v2 = vertices[face[i-1]]
                v3 = vertices[face[i]]
                push!(triangles, Triangle(v1, v2, v3, material_index))
            end
        end
    end
    return triangles
end

# Generate a sphere mesh
function generate_sphere(center::Point3, radius::Float32, material_index::Int; subdivisions::Int=3)
    t_ico = Float32((1.0 + sqrt(5.0)) / 2.0) # Icosahedron constant, ensure Float32
    sphere_vertices_init = [
        Point3(-1.0f0, t_ico, 0.0f0), Point3(1.0f0, t_ico, 0.0f0), Point3(-1.0f0, -t_ico, 0.0f0), Point3(1.0f0, -t_ico, 0.0f0),
        Point3(0.0f0, -1.0f0, t_ico), Point3(0.0f0, 1.0f0, t_ico), Point3(0.0f0, -1.0f0, -t_ico), Point3(0.0f0, 1.0f0, -t_ico),
        Point3(t_ico, 0.0f0, -1.0f0), Point3(t_ico, 0.0f0, 1.0f0), Point3(-t_ico, 0.0f0, -1.0f0), Point3(-t_ico, 0.0f0, 1.0f0)
    ]
    
    sphere_vertices = [normalize(v) for v in sphere_vertices_init]
    
    sphere_faces_init = [
        [1, 12, 6], [1, 6, 2], [1, 2, 8], [1, 8, 11], [1, 11, 12],
        [2, 6, 10], [6, 12, 5], [12, 11, 3], [11, 8, 7], [8, 2, 9],
        [4, 10, 5], [4, 5, 3], [4, 3, 7], [4, 7, 9], [4, 9, 10],
        [5, 10, 6], [3, 5, 12], [7, 3, 11], [9, 7, 8], [10, 9, 2]
    ]
    
    current_faces = sphere_faces_init
    for _ in 1:subdivisions
        new_faces = Vector{Vector{Int}}()
        for face_indices in current_faces
            v1_idx, v2_idx, v3_idx = face_indices
            
            v1 = sphere_vertices[v1_idx]
            v2 = sphere_vertices[v2_idx]
            v3 = sphere_vertices[v3_idx]
            
            v12 = normalize((v1 + v2) / 2.0f0)
            v23 = normalize((v2 + v3) / 2.0f0)
            v31 = normalize((v3 + v1) / 2.0f0)
            
            push!(sphere_vertices, v12); v12_new_idx = length(sphere_vertices)
            push!(sphere_vertices, v23); v23_new_idx = length(sphere_vertices)
            push!(sphere_vertices, v31); v31_new_idx = length(sphere_vertices)
            
            push!(new_faces, [v1_idx, v12_new_idx, v31_new_idx])
            push!(new_faces, [v2_idx, v23_new_idx, v12_new_idx])
            push!(new_faces, [v3_idx, v31_new_idx, v23_new_idx])
            push!(new_faces, [v12_new_idx, v23_new_idx, v31_new_idx])
        end
        current_faces = new_faces
    end
    
    triangles_out = Triangle[]
    for face_indices in current_faces
        v1 = sphere_vertices[face_indices[1]] * radius + center
        v2 = sphere_vertices[face_indices[2]] * radius + center
        v3 = sphere_vertices[face_indices[3]] * radius + center
        push!(triangles_out, Triangle(v1, v2, v3, material_index))
    end
    return triangles_out
end

# Ray-triangle intersection
function hit_triangle(ray::Ray, triangle::Triangle, t_min::Real, t_max::Real)
    pvec = cross(ray.direction, triangle.e2)
    det = LinearAlgebra.dot(triangle.e1, pvec)

    if abs(det) < EPS
        return false, INF, Vec3(0.0f0, 0.0f0, 0.0f0)
    end

    inv_det = 1.0f0 / det
    tvec = ray.origin - triangle.v1
    u = LinearAlgebra.dot(tvec, pvec) * inv_det

    if u < 0.0f0 || u > 1.0f0
        return false, INF, Vec3(0.0f0, 0.0f0, 0.0f0)
    end

    qvec = cross(tvec, triangle.e1)
    v = LinearAlgebra.dot(ray.direction, qvec) * inv_det

    if v < 0.0f0 || u + v > 1.0f0
        return false, INF, Vec3(0.0f0, 0.0f0, 0.0f0)
    end

    t = LinearAlgebra.dot(triangle.e2, qvec) * inv_det

    if t < t_min || t > t_max
        return false, INF, Vec3(0.0f0, 0.0f0, 0.0f0)
    end

    return true, Float32(t), triangle.normal
end

# Forward declaration for the recursive helper is not needed if it's defined before its first call by build_and_flatten_bvh_cpu
# Main CPU function to build and flatten the BVH
function build_and_flatten_bvh_cpu(
    scene_triangles::Vector{Triangle},
    max_tris_per_leaf::Int = 4
)::Tuple{Vector{BVHNodeGPU}, Vector{Int32}}

    num_total_triangles = length(scene_triangles)
    if num_total_triangles == 0
        return BVHNodeGPU[], Int32[]
    end

    tri_info = Vector{Tuple{Int32, Point3, AABB}}(undef, num_total_triangles)
    for i in 1:num_total_triangles
        tri = scene_triangles[i]
        centroid = (tri.v1 + tri.v2 + tri.v3) / 3.0f0
        aabb = get_triangle_aabb(tri)
        tri_info[i] = (Int32(i), centroid, aabb)
    end

    output_flat_nodes = BVHNodeGPU[]
    sizehint!(output_flat_nodes, num_total_triangles * 2)
    output_flat_bvh_indices = Int32[]
    sizehint!(output_flat_bvh_indices, num_total_triangles)

    initial_indices_to_process = Int32.(1:num_total_triangles)

    _recursive_build_bvh!(
        output_flat_nodes,
        output_flat_bvh_indices,
        scene_triangles, 
        tri_info,
        initial_indices_to_process,
        max_tris_per_leaf
    )

    return output_flat_nodes, output_flat_bvh_indices
end

# Recursive BVH build helper
function _recursive_build_bvh!(
    output_flat_nodes::Vector{BVHNodeGPU},
    output_flat_bvh_indices::Vector{Int32},
    all_scene_triangles::Vector{Triangle},
    triangle_info::Vector{Tuple{Int32, Point3, AABB}}, 
    indices_to_process::AbstractVector{Int32},
    max_tris_per_leaf::Int
)::Tuple{Int32, Int32}

    num_tris_current_node = length(indices_to_process)
    
    current_aabb = AABB()
    if num_tris_current_node > 0
        current_aabb = triangle_info[indices_to_process[1]][3]
        for i in 2:num_tris_current_node
            tri_idx_in_info = indices_to_process[i]
            current_aabb = merge_aabbs(current_aabb, triangle_info[tri_idx_in_info][3])
        end
    else 
        my_idx = Int32(length(output_flat_nodes) + 1)
        push!(output_flat_nodes, BVHNodeGPU(Point3(0.0f0,0.0f0,0.0f0), Point3(0.0f0,0.0f0,0.0f0), Int32(0), Int32(0), true))
        return my_idx, Int32(1)
    end

    if num_tris_current_node <= max_tris_per_leaf
        my_node_idx = Int32(length(output_flat_nodes) + 1)
        leaf_tri_offset = Int32(length(output_flat_bvh_indices) + 1)
        for i in 1:num_tris_current_node
            original_tri_idx = triangle_info[indices_to_process[i]][1]
            push!(output_flat_bvh_indices, original_tri_idx)
        end
        leaf_node = BVHNodeGPU(
            current_aabb.min_bounds,
            current_aabb.max_bounds,
            leaf_tri_offset,
            Int32(num_tris_current_node),
            true
        )
        push!(output_flat_nodes, leaf_node)
        return my_node_idx, Int32(1)
    end

    my_node_idx_in_flat_list = Int32(length(output_flat_nodes) + 1)
    push!(output_flat_nodes, BVHNodeGPU(Point3(0.0f0,0.0f0,0.0f0), Point3(0.0f0,0.0f0,0.0f0), 0, 0, false))

    diag = current_aabb.max_bounds - current_aabb.min_bounds
    split_axis = 0 
    if diag[2] > diag[1]
        split_axis = 1
    end
    if diag[3] > diag[split_axis+1]
        split_axis = 2
    end
    
    local_indices_to_process = Vector{Int32}(indices_to_process)
    sort!(local_indices_to_process, by = idx -> triangle_info[idx][2][split_axis+1])

    mid_point = div(num_tris_current_node, 2) + 1
    
    first_centroid_val = triangle_info[local_indices_to_process[1]][2][split_axis+1]
    last_centroid_val = triangle_info[local_indices_to_process[end]][2][split_axis+1]
    # median_centroid_val = triangle_info[local_indices_to_process[mid_point > num_tris_current_node ? num_tris_current_node : mid_point]][2][split_axis+1]

    if mid_point <= 1 || mid_point > num_tris_current_node || first_centroid_val == last_centroid_val 
        leaf_tri_offset = Int32(length(output_flat_bvh_indices) + 1)
        for i in 1:num_tris_current_node
            original_tri_idx = triangle_info[local_indices_to_process[i]][1]
            push!(output_flat_bvh_indices, original_tri_idx)
        end
        leaf_node_data = BVHNodeGPU(
            current_aabb.min_bounds,
            current_aabb.max_bounds,
            leaf_tri_offset,
            Int32(num_tris_current_node),
            true
        )
        output_flat_nodes[my_node_idx_in_flat_list] = leaf_node_data
        return my_node_idx_in_flat_list, Int32(1)
    end

    left_child_indices = view(local_indices_to_process, 1:(mid_point-1))
    right_child_indices = view(local_indices_to_process, mid_point:num_tris_current_node)

    left_child_node_idx, num_nodes_left_subtree = _recursive_build_bvh!(
        output_flat_nodes, output_flat_bvh_indices, all_scene_triangles, 
        triangle_info, left_child_indices, max_tris_per_leaf
    )

    right_child_node_idx, num_nodes_right_subtree = _recursive_build_bvh!(
        output_flat_nodes, output_flat_bvh_indices, all_scene_triangles,
        triangle_info, right_child_indices, max_tris_per_leaf
    )
    
    output_flat_nodes[my_node_idx_in_flat_list] = BVHNodeGPU(
        current_aabb.min_bounds,
        current_aabb.max_bounds,
        left_child_node_idx, 
        right_child_node_idx,
        false
    )
    
    total_nodes_in_this_subtree = 1 + num_nodes_left_subtree + num_nodes_right_subtree
    return my_node_idx_in_flat_list, Int32(total_nodes_in_this_subtree)
end

# Metal GPU kernel for HDR rendering

# Helper function for stack push - to be used inside the kernel
@inline function push_to_kernel_stack!(
    main_stack_marray::StaticArrays.MArray{Tuple{64}, Int32, 1, 64}, 
    stack_idx_marray::StaticArrays.MArray{Tuple{1}, Int32, 1, 1}, 
    value_to_push::Int32
)::Bool
    idx_val = stack_idx_marray[1]
    if idx_val < 64 # Max index is 64 (1-based), so if current idx_val is 0 to 63, we can push.
        new_idx_val = idx_val + Int32(1)
        main_stack_marray[new_idx_val] = value_to_push
        stack_idx_marray[1] = new_idx_val
        return true # Pushed successfully
    end
    return false # Stack overflow
end

@inline function pop_from_kernel_stack!(
    main_stack_marray::StaticArrays.MArray{Tuple{64}, Int32, 1, 64},
    stack_idx_marray::StaticArrays.MArray{Tuple{1}, Int32, 1, 1}
)::Tuple{Bool, Int32}
    idx_val = stack_idx_marray[1]
    if idx_val > 0
        value = main_stack_marray[idx_val] # Read from current top
        stack_idx_marray[1] = idx_val - Int32(1) # Decrement stack pointer
        return true, value
    end
    return false, Int32(0) # Stack underflow / empty
end

function metal_raytracer_kernel(red, green, blue, width, height,
                                triangles_buffer::MtlDeviceArray{Triangle,1}, num_triangles_total::Int32,
                                camera::Camera,
                                materials_buffer::MtlDeviceArray{Material,1}, num_materials_total::Int32,
                                bvh_nodes_gpu::MtlDeviceArray{BVHNodeGPU,1},
                                bvh_indices_gpu::MtlDeviceArray{Int32,1})
    # Get thread position (pixel coordinates)
    x_idx = thread_position_in_grid_2d()[1]
    y_idx = thread_position_in_grid_2d()[2]

    # Bounds check
    if x_idx > width || y_idx > height
        return
    end

    # Ray generation
    s_param = Float32(x_idx-1) / Float32(width-1)
    t_param = Float32(height-y_idx) / Float32(height-1) # Flipped y for bottom-to-top t

    kernel_ray_origin = camera.origin
    kernel_ray_direction = camera.lower_left_corner + s_param * camera.horizontal + t_param * camera.vertical - camera.origin
    current_ray = Ray(kernel_ray_origin, kernel_ray_direction) # Ray constructor normalizes direction and computes inv_direction

    # Intersection variables
    hit_anything = false
    closest_t = INF 
    closest_normal_x = 0.0f0
    closest_normal_y = 0.0f0
    closest_normal_z = 0.0f0
    closest_material_idx::Int32 = 0

    # --- BVH Traversal Stack Initialization ---
    travers_stack_bvh_nodes_idx = StaticArrays.MArray{Tuple{64}, Int32, 1, 64}(undef) 
    stack_idx_marray = StaticArrays.MArray{Tuple{1}, Int32, 1, 1}((Int32(0),)) # Stack pointer as 1-element MArray

    # --- BVH Traversal Logic ---
    if length(bvh_nodes_gpu) > 0 # Check if there are any BVH nodes
        # Push root node (index 1)
        push_to_kernel_stack!(travers_stack_bvh_nodes_idx, stack_idx_marray, Int32(1))

        # Main traversal loop
        # Loop while stack_idx_marray[1] > 0
        while stack_idx_marray[1] > 0
            pushed_successfully, current_bvh_node_array_idx = pop_from_kernel_stack!(travers_stack_bvh_nodes_idx, stack_idx_marray)
            
            if !pushed_successfully
                # This should not happen if while condition is stack_idx_marray[1] > 0
                break
            end

            node = bvh_nodes_gpu[current_bvh_node_array_idx]

            aabb_hit, hit_t_min_aabb, hit_t_max_aabb = hit_aabb(
                current_ray.origin, current_ray.inv_direction, 
                node.aabb_min, node.aabb_max, 
                EPS, closest_t
            )

            if aabb_hit # Ray hits the node's AABB and is potentially closer
                if node.is_leaf_node_flag # It's a Leaf Node
                    tri_offset_in_bvh_indices = node.payload1
                    num_tris_in_leaf = node.payload2
                    
                    for i::Int32 in 0:(num_tris_in_leaf - 1)
                        original_triangle_flat_idx = bvh_indices_gpu[tri_offset_in_bvh_indices + i]
                        current_triangle = triangles_buffer[original_triangle_flat_idx]
                        
                        tri_hit_status, t_val_tri, normal_vec_tri = hit_triangle(current_ray, current_triangle, EPS, closest_t)
                        
                        if tri_hit_status && t_val_tri < closest_t
                            hit_anything = true
                            closest_t = t_val_tri
                            closest_normal_x = normal_vec_tri[1]
                            closest_normal_y = normal_vec_tri[2]
                            closest_normal_z = normal_vec_tri[3]
                            closest_material_idx = current_triangle.material_index
                        end
                    end
                else # It's an Inner Node
                    left_child_idx = node.payload1
                    # right_child_idx = node.payload2 

                    # Push left child using helper function
                    push_to_kernel_stack!(travers_stack_bvh_nodes_idx, stack_idx_marray, left_child_idx)
                    
                    # TODO: Add right child push here later if left child works
                    # if right_child_idx != 0
                    #    push_to_kernel_stack!(travers_stack_bvh_nodes_idx, stack_idx_marray, right_child_idx)
                    # end
                end
            end
        end
    end
    # --- End BVH Traversal Logic ---

    # Determine final pixel color based on intersection results
    idx = (height - y_idx) * width + x_idx # Target buffer index

    if hit_anything
        # Use normal for color for now
        final_r = (closest_normal_x + 1.0f0) * 0.5f0
        final_g = (closest_normal_y + 1.0f0) * 0.5f0
        final_b = (closest_normal_z + 1.0f0) * 0.5f0
        intensity = 1.0f0 # Basic intensity for normals
        red[idx] = final_r * intensity
        green[idx] = final_g * intensity
        blue[idx] = final_b * intensity
    else
        # Sky color if no intersection
        ray_dir_y_norm = current_ray.direction[2]
        sky_factor = 0.5f0 * (ray_dir_y_norm + 1.0f0)
        r_sky = (1.0f0 - sky_factor) * 1.0f0 + sky_factor * 0.5f0
        g_sky = (1.0f0 - sky_factor) * 1.0f0 + sky_factor * 0.7f0
        b_sky = (1.0f0 - sky_factor) * 1.0f0 + sky_factor * 1.0f0
        intensity = 1.5f0
        red[idx] = r_sky * intensity
        green[idx] = g_sky * intensity
        blue[idx] = b_sky * intensity
    end

    return
end

# Helper function for dot product in the kernel
function vector_dot(x1, y1, z1, x2, y2, z2)
    return x1 * x2 + y1 * y2 + z1 * z2
end

# DIRECT GPU-to-EXR rendering function - NO CPU RENDERING
function render_metal_exr(triangles::Vector{Triangle}, materials::Vector{Material}, camera::Camera,
                        flat_bvh_nodes::Vector{BVHNodeGPU}, flat_bvh_triangle_indices::Vector{Int32},
                        width::Int, height::Int;
                        samples_per_pixel::Int=16, max_depth::Int=8,
                        output_path::String="render_metal.exr")
    
    println("Rendering with Metal GPU directly to EXR...")
    
    # Create HDR output arrays on GPU - FULL float32 precision
    red_gpu = MtlArray(zeros(Float32, width * height))
    green_gpu = MtlArray(zeros(Float32, width * height))
    blue_gpu = MtlArray(zeros(Float32, width * height))
    
    # Prepare scene data for GPU
    triangles_gpu = MtlArray(triangles) # Convert triangles to MtlArray
    num_triangles = Int32(length(triangles))
    materials_gpu = MtlArray(materials) # Convert materials to MtlArray
    num_materials = Int32(length(materials))
    
    # Prepare BVH data for GPU
    bvh_nodes_gpu = MtlArray(flat_bvh_nodes)
    bvh_indices_gpu = MtlArray(flat_bvh_triangle_indices)

    # Launch kernel with appropriate thread groups
    threads_per_group = (16, 16)  # 256 threads per group
    groups = (div(width + 15, 16), div(height + 15, 16))  # Ceiling division
    
    # Run the HDR kernel - NO tone mapping
    @metal threads=threads_per_group groups=groups metal_raytracer_kernel(red_gpu, green_gpu, blue_gpu, width, height, 
                                                                        triangles_gpu, num_triangles,
                                                                        camera,
                                                                        materials_gpu, num_materials,
                                                                        bvh_nodes_gpu, bvh_indices_gpu)
    
    # Copy results back to CPU
    red_cpu = Array(red_gpu)
    green_cpu = Array(green_gpu)
    blue_cpu = Array(blue_gpu)
    
    # Create OpenEXR buffer directly from GPU data
    hdr_buffer = zeros(Float32, 3, width, height)
    
    println("Preparing HDR data for EXR export...")
    for y in 1:height
        for x in 1:width
            # Direct GPU buffer to EXR buffer transfer - PRESERVING FULL HDR VALUES
            idx = (height - y) * width + x
            
            # Channel data for OpenEXR
            hdr_buffer[1, x, y] = red_cpu[idx]    # R
            hdr_buffer[2, x, y] = green_cpu[idx]  # G
            hdr_buffer[3, x, y] = blue_cpu[idx]   # B
        end
    end
    
    # Write to EXR - DIRECT HDR DATA
    println("Writing EXR file using FileIO and Images.jl...")
    
    # hdr_buffer is currently (Channel, Width, Height) i.e. (3, width, height)
    # For colorview(RGB, data), data needs to be (Channel, Height, Width)
    permuted_hdr_buffer = permutedims(hdr_buffer, (1, 3, 2))
    
    # Create an Images.jl image object (Matrix{RGB{Float32}})
    img = colorview(RGB, permuted_hdr_buffer)
    
    # Save the image using FileIO
    # FileIO will use OpenEXR.jl as the backend for ".exr" files
    FileIO.save(output_path, img)
    
    println("Metal GPU render saved to EXR: $output_path")
    
    return true
end

# Main function
function main()
    # Parameters
    width = 960
    height = 540
    samples = 32
    max_depth = 8
    max_tris_per_leaf_bvh = 4
    
    # Camera setup
    aspect_ratio = width / height
    lookfrom = Point3(0.0, 0.5, 3.0)
    lookat = Point3(0.0, 0.0, 0.0)
    vup = Point3(0.0, 1.0, 0.0)
    fov = 40.0
    aperture = 0.0
    dist_to_focus = norm(lookfrom - lookat)
    
    camera = Camera(lookfrom, lookat, vup, fov, aspect_ratio, aperture, dist_to_focus)
    
    # Create materials
    metal_material = Material(
        Vec3(0.7, 0.6, 0.5),
        metallic=Float32(0.9),
        roughness=Float32(0.1)
    )
    
    ground_material = Material(
        Vec3(0.5, 0.5, 0.5),
        metallic=Float32(0.0),
        roughness=Float32(0.9)
    )
    
    light_material = Material(
        Vec3(0.9, 0.9, 0.9),
        emission=Vec3(10.0, 10.0, 10.0),
        metallic=Float32(0.0),
        roughness=Float32(0.0)
    )
    
    materials = [metal_material, ground_material, light_material]
    
    # Load or generate geometry
    triangles = Triangle[]
    
    # Main object
    if isfile("bunny.obj")
        println("Loading bunny.obj...")
        mesh_triangles = load_obj("bunny.obj", 1, scale=Float32(2.0), translation=Point3(0.0, -0.5, 0.0))
        append!(triangles, mesh_triangles)
    else
        println("Generating sphere mesh...")
        mesh_triangles = generate_sphere(Point3(0.0, 0.0, 0.0), Float32(1.0), 4)
        append!(triangles, mesh_triangles)
    end
    
    # Add ground plane
    ground_y = -1.0
    ground_triangles = [
        Triangle(
            Point3(-10.0, ground_y, -10.0),
            Point3(-10.0, ground_y, 10.0),
            Point3(10.0, ground_y, -10.0)
        ),
        Triangle(
            Point3(10.0, ground_y, -10.0),
            Point3(-10.0, ground_y, 10.0),
            Point3(10.0, ground_y, 10.0)
        )
    ]
    append!(triangles, ground_triangles)
    
    # Add light source
    light_y = 5.0
    light_triangles = [
        Triangle(
            Point3(-1.0, light_y, -1.0),
            Point3(-1.0, light_y, 1.0),
            Point3(1.0, light_y, -1.0)
        ),
        Triangle(
            Point3(1.0, light_y, -1.0),
            Point3(-1.0, light_y, 1.0),
            Point3(1.0, light_y, 1.0)
        )
    ]
    append!(triangles, light_triangles)
    
    # Build BVH
    println("Building BVH...")
    @time flat_bvh_nodes, flat_bvh_triangle_indices = build_and_flatten_bvh_cpu(triangles, max_tris_per_leaf_bvh)
    println("BVH built: $(length(flat_bvh_nodes)) nodes, $(length(flat_bvh_triangle_indices)) triangle indices (should match original triangle count)")

    # Render with direct GPU-to-EXR pipeline
    println("\nStarting Metal GPU direct-to-EXR render...")
    @time render_metal_exr(triangles, materials, camera, 
                          flat_bvh_nodes, flat_bvh_triangle_indices,
                          width, height, 
                          samples_per_pixel=samples, 
                          max_depth=max_depth,
                          output_path="render_metal.exr")
end

# Run main if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end