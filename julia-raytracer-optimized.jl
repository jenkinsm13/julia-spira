# Optimized version of julia-raytracer.jl with mesh-specific BVH

using LinearAlgebra
using Random
using Images      # For image manipulation
using Colors      # For color handling
using FileIO      # For file I/O
using Plots       # For displaying the image
using MeshIO      # For loading OBJ mesh files
using GeometryBasics # For mesh data structures
using Metal       # For GPU acceleration
using OpenEXR     # For EXR file saving

# Basic 3D vector structure
struct Vec3
    x::Float64
    y::Float64
    z::Float64
end

# Vector operations
import Base: +, -, *, /, show
import LinearAlgebra: dot, normalize

+(a::Vec3, b::Vec3) = Vec3(a.x + b.x, a.y + b.y, a.z + b.z)
-(a::Vec3, b::Vec3) = Vec3(a.x - b.x, a.y - b.y, a.z - b.z)
*(a::Vec3, b::Real) = Vec3(a.x * b, a.y * b, a.z * b)
*(b::Real, a::Vec3) = a * b
/(a::Vec3, b::Real) = Vec3(a.x / b, a.y / b, a.z / b)
dot(a::Vec3, b::Vec3) = a.x * b.x + a.y * b.y + a.z * b.z
Base.length(a::Vec3) = sqrt(dot(a, a))
normalize(a::Vec3) = a / length(a)
show(io::IO, v::Vec3) = print(io, "($(v.x), $(v.y), $(v.z))")

# Vector componentwise multiplication for colors
Base.:*(a::Vec3, b::Vec3) = Vec3(a.x * b.x, a.y * b.y, a.z * b.z)

# Cross product
cross(a::Vec3, b::Vec3) = Vec3(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
)

# Ray structure
struct Ray
    origin::Vec3
    direction::Vec3
end

# Function to get point along ray
point_at(ray::Ray, t::Float64) = ray.origin + ray.direction * t

# Material structure
struct Material
    diffuse::Vec3
    emission::Vec3
    specular::Float64
    roughness::Float64
    
    # Constructor with default values
    Material(; diffuse=Vec3(0.8, 0.8, 0.8), emission=Vec3(0.0, 0.0, 0.0),
              specular=0.0, roughness=1.0) = new(diffuse, emission, specular, roughness)
end

# Intersection result structure
struct HitRecord
    t::Float64         # Distance along ray to intersection
    position::Vec3     # Position of intersection
    normal::Vec3       # Surface normal at intersection
    material::Material # Material of the intersected object
    hit::Bool          # Flag indicating if an intersection occurred
end

# AABB (Axis-Aligned Bounding Box) for BVH optimization
struct AABB
    min::Vec3
    max::Vec3
end

# Combine two AABBs
function surrounding_box(box1::AABB, box2::AABB)
    min_point = Vec3(
        min(box1.min.x, box2.min.x),
        min(box1.min.y, box2.min.y),
        min(box1.min.z, box2.min.z)
    )
    
    max_point = Vec3(
        max(box1.max.x, box2.max.x),
        max(box1.max.y, box2.max.y),
        max(box1.max.z, box2.max.z)
    )
    
    return AABB(min_point, max_point)
end

# Ray-AABB intersection test (fast slab method)
function hit_aabb(aabb::AABB, ray::Ray, t_min::Float64, t_max::Float64)
    for a in 1:3
        # Get component for this axis (x, y, or z)
        origin_comp = a == 1 ? ray.origin.x : (a == 2 ? ray.origin.y : ray.origin.z)
        direction_comp = a == 1 ? ray.direction.x : (a == 2 ? ray.direction.y : ray.direction.z)
        min_comp = a == 1 ? aabb.min.x : (a == 2 ? aabb.min.y : aabb.min.z)
        max_comp = a == 1 ? aabb.max.x : (a == 2 ? aabb.max.y : aabb.max.z)
        
        inv_dir = 1.0 / direction_comp
        t0 = (min_comp - origin_comp) * inv_dir
        t1 = (max_comp - origin_comp) * inv_dir
        
        # Sort t values
        if inv_dir < 0.0
            temp = t0
            t0 = t1
            t1 = temp
        end
        
        # Update t_min and t_max
        t_min = t0 > t_min ? t0 : t_min
        t_max = t1 < t_max ? t1 : t_max
        
        if t_max <= t_min
            return false
        end
    end
    
    return true
end

# Base hittable abstract type
abstract type Hittable end

# Method to get bounding box (to be implemented by subtypes)
function bounding_box(object::Hittable)
    error("bounding_box not implemented for $(typeof(object))")
end

# Sphere structure
struct Sphere <: Hittable
    center::Vec3
    radius::Float64
    material::Material
end

# Triangle structure
struct Triangle <: Hittable
    vertices::Vector{Vec3}
    material::Material
    bbox::AABB  # Pre-computed bounding box for efficiency

    # Calculate normal and bounding box during construction
    function Triangle(vertices::Vector{Vec3}, material::Material)
        @assert length(vertices) == 3 "Triangle must have exactly 3 vertices"
        
        # Compute bounding box
        min_point = Vec3(
            min(vertices[1].x, min(vertices[2].x, vertices[3].x)),
            min(vertices[1].y, min(vertices[2].y, vertices[3].y)),
            min(vertices[1].z, min(vertices[2].z, vertices[3].z))
        )
        
        max_point = Vec3(
            max(vertices[1].x, max(vertices[2].x, vertices[3].x)),
            max(vertices[1].y, max(vertices[2].y, vertices[3].y)),
            max(vertices[1].z, max(vertices[2].z, vertices[3].z))
        )
        
        # Add small epsilon to avoid zero-width boxes
        epsilon = 1e-8
        if min_point.x == max_point.x
            min_point = Vec3(min_point.x - epsilon, min_point.y, min_point.z)
            max_point = Vec3(max_point.x + epsilon, max_point.y, max_point.z)
        end
        if min_point.y == max_point.y
            min_point = Vec3(min_point.x, min_point.y - epsilon, min_point.z)
            max_point = Vec3(max_point.x, max_point.y + epsilon, max_point.z)
        end
        if min_point.z == max_point.z
            min_point = Vec3(min_point.x, min_point.y, min_point.z - epsilon)
            max_point = Vec3(max_point.x, max_point.y, max_point.z + epsilon)
        end
        
        bbox = AABB(min_point, max_point)
        
        new(vertices, material, bbox)
    end
end

# Get bounding box for a triangle
function bounding_box(triangle::Triangle)
    return triangle.bbox
end

# Get bounding box for a sphere
function bounding_box(sphere::Sphere)
    min_point = Vec3(
        sphere.center.x - sphere.radius,
        sphere.center.y - sphere.radius,
        sphere.center.z - sphere.radius
    )
    
    max_point = Vec3(
        sphere.center.x + sphere.radius,
        sphere.center.y + sphere.radius,
        sphere.center.z + sphere.radius
    )
    
    return AABB(min_point, max_point)
end

# BVH Node structure for efficient ray-mesh intersection
struct BVHNode <: Hittable
    bbox::AABB
    left::Union{Hittable, Nothing}
    right::Union{Hittable, Nothing}
    
    # Leaf node constructor
    function BVHNode(object::Hittable)
        bbox = bounding_box(object)
        new(bbox, object, nothing)
    end
    
    # Interior node constructor
    function BVHNode(left::Hittable, right::Hittable)
        bbox = surrounding_box(bounding_box(left), bounding_box(right))
        new(bbox, left, right)
    end
end

# Get bounding box for a BVH node
function bounding_box(node::BVHNode)
    return node.bbox
end

# Build a BVH tree from a list of hittable objects
function build_bvh(objects::Vector{Hittable}, start::Int, finish::Int)
    # Choose a random axis to sort on
    axis = rand(1:3)

    # Sort objects based on the chosen axis
    sort_function = axis == 1 ?
        (a, b) -> bounding_box(a).min.x < bounding_box(b).min.x :
        (axis == 2 ?
            (a, b) -> bounding_box(a).min.y < bounding_box(b).min.y :
            (a, b) -> bounding_box(a).min.z < bounding_box(b).min.z)

    object_span = finish - start

    if object_span == 1
        # Leaf node with a single object
        return BVHNode(objects[start])
    elseif object_span == 2
        # Leaf node with two objects
        if sort_function(objects[start], objects[start+1])
            return BVHNode(objects[start], objects[start+1])
        else
            return BVHNode(objects[start+1], objects[start])
        end
    else
        # Interior node with more than two objects
        # Note: In Julia, the end index in a range is inclusive, so we need (start:finish-1)
        sorted_objects = sort(objects[start:finish-1], lt=sort_function)
        objects[start:finish-1] = sorted_objects

        mid = start + floor(Int, object_span / 2)
        left = build_bvh(objects, start, mid)
        right = build_bvh(objects, mid, finish)

        return BVHNode(left, right)
    end
end

# Mesh structure (collection of triangles with shared material)
struct Mesh <: Hittable
    triangles::Vector{Triangle}
    bbox::AABB
    bvh::Union{BVHNode, Nothing}
    
    # Create a mesh from triangles with same material and build BVH
    function Mesh(triangles::Vector{Triangle})
        if length(triangles) == 0
            error("Cannot create mesh with zero triangles")
        end
        
        # Compute bounding box for the whole mesh
        bbox = bounding_box(triangles[1])
        for i in 2:length(triangles)
            bbox = surrounding_box(bbox, bounding_box(triangles[i]))
        end
        
        # Build BVH for the triangles
        hittable_triangles = Hittable[triangle for triangle in triangles]
        bvh = nothing

        # Only build BVH for meshes with more than a few triangles
        if length(triangles) > 4
            # The end index is inclusive in Julia ranges, so we pass the actual length
            bvh = build_bvh(hittable_triangles, 1, length(hittable_triangles))
        end
        
        new(triangles, bbox, bvh)
    end
end

# Get bounding box for a mesh
function bounding_box(mesh::Mesh)
    return mesh.bbox
end

# Triangle normal calculation
function triangle_normal(tri::Triangle)
    edge1 = tri.vertices[2] - tri.vertices[1]
    edge2 = tri.vertices[3] - tri.vertices[1]
    normalize(cross(edge1, edge2))
end

# Ray-Sphere intersection
function hit(sphere::Sphere, ray::Ray, t_min::Float64, t_max::Float64)
    oc = ray.origin - sphere.center
    a = dot(ray.direction, ray.direction)
    b = 2.0 * dot(oc, ray.direction)
    c = dot(oc, oc) - sphere.radius * sphere.radius
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0
        return HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), sphere.material, false)
    end
    
    # Calculate the two intersection points
    sqrtd = sqrt(discriminant)
    root1 = (-b - sqrtd) / (2.0 * a)
    root2 = (-b + sqrtd) / (2.0 * a)
    
    # Check if either intersection point is within the valid range
    if root1 < t_min || root1 > t_max
        root1 = root2
        if root1 < t_min || root1 > t_max
            return HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), sphere.material, false)
        end
    end
    
    t = root1
    point = point_at(ray, t)
    normal = normalize(point - sphere.center)
    
    return HitRecord(t, point, normal, sphere.material, true)
end

# Ray-Triangle intersection (Möller–Trumbore algorithm)
function hit(triangle::Triangle, ray::Ray, t_min::Float64, t_max::Float64)
    # First check bounding box for early rejection
    if !hit_aabb(triangle.bbox, ray, t_min, t_max)
        return HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), triangle.material, false)
    end
    
    v0, v1, v2 = triangle.vertices
    
    # Compute edges
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # Calculate determinant
    h = cross(ray.direction, edge2)
    a = dot(edge1, h)
    
    # Check if ray is parallel to triangle
    if abs(a) < 1e-8
        return HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), triangle.material, false)
    end
    
    f = 1.0 / a
    s = ray.origin - v0
    u = f * dot(s, h)
    
    if u < 0.0 || u > 1.0
        return HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), triangle.material, false)
    end
    
    q = cross(s, edge1)
    v = f * dot(ray.direction, q)
    
    if v < 0.0 || u + v > 1.0
        return HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), triangle.material, false)
    end
    
    # Compute intersection point parameter
    t = f * dot(edge2, q)
    
    if t < t_min || t > t_max
        return HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), triangle.material, false)
    end
    
    point = point_at(ray, t)
    normal = triangle_normal(triangle)
    
    return HitRecord(t, point, normal, triangle.material, true)
end

# Ray-BVHNode intersection
function hit(node::BVHNode, ray::Ray, t_min::Float64, t_max::Float64)
    # First check if ray hits the bounding box
    if !hit_aabb(node.bbox, ray, t_min, t_max)
        return HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), Material(), false)
    end
    
    # This is a leaf node with a single object
    if node.right === nothing
        return hit(node.left, ray, t_min, t_max)
    end
    
    # Check both children
    hit_left = hit(node.left, ray, t_min, t_max)
    hit_right = hit(node.right, ray, t_min, t_max)
    
    # Return the closest hit
    if hit_left.hit && hit_right.hit
        if hit_left.t < hit_right.t
            return hit_left
        else
            return hit_right
        end
    elseif hit_left.hit
        return hit_left
    elseif hit_right.hit
        return hit_right
    else
        return HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), Material(), false)
    end
end

# Ray-Mesh intersection (tests all triangles in the mesh)
function hit(mesh::Mesh, ray::Ray, t_min::Float64, t_max::Float64)
    # First check the mesh's bounding box
    if !hit_aabb(mesh.bbox, ray, t_min, t_max)
        return HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), Material(), false)
    end
    
    # If we have a BVH, use it for faster intersection
    if mesh.bvh !== nothing
        return hit(mesh.bvh, ray, t_min, t_max)
    end
    
    # Otherwise, test each triangle
    closest_so_far = t_max
    hit_anything = false
    result = HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), Material(), false)

    for triangle in mesh.triangles
        temp_rec = hit(triangle, ray, t_min, closest_so_far)
        if temp_rec.hit
            hit_anything = true
            closest_so_far = temp_rec.t
            result = temp_rec
        end
    end
    
    return result
end

# Collection of hittable objects
struct HittableList <: Hittable
    objects::Vector{Hittable}
end

# Get bounding box for a list of objects
function bounding_box(list::HittableList)
    if length(list.objects) == 0
        # Return a default bounding box if the list is empty
        return AABB(Vec3(0, 0, 0), Vec3(0, 0, 0))
    end
    
    # Calculate the bounding box from all objects
    bbox = bounding_box(list.objects[1])
    for i in 2:length(list.objects)
        bbox = surrounding_box(bbox, bounding_box(list.objects[i]))
    end
    
    return bbox
end

# Ray-HittableList intersection
function hit(list::HittableList, ray::Ray, t_min::Float64, t_max::Float64)
    closest_so_far = t_max
    hit_anything = false
    result = HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), Material(), false)

    for object in list.objects
        temp_rec = hit(object, ray, t_min, closest_so_far)
        if temp_rec.hit
            hit_anything = true
            closest_so_far = temp_rec.t
            result = temp_rec
        end
    end

    return result
end

# Simple Bounding Volume Hierarchy structure for the whole scene
struct BoundingVolumeHierarchy <: Hittable
    root::BVHNode
    
    function BoundingVolumeHierarchy(objects::Vector{Hittable})
        if length(objects) == 0
            error("Cannot create BVH with zero objects")
        end
        
        # Build the BVH tree
        root = build_bvh(objects, 1, length(objects) + 1)
        new(root)
    end
end

# Ray-BVH intersection
function hit(bvh::BoundingVolumeHierarchy, ray::Ray, t_min::Float64, t_max::Float64)
    return hit(bvh.root, ray, t_min, t_max)
end

# Camera structure
struct Camera
    position::Vec3
    lower_left_corner::Vec3
    horizontal::Vec3
    vertical::Vec3
    u::Vec3
    v::Vec3
    w::Vec3
    lens_radius::Float64
    
    function Camera(;
        position::Vec3 = Vec3(0, 0, 0),
        look_at::Vec3 = Vec3(0, 0, -1),
        up::Vec3 = Vec3(0, 1, 0),
        fov::Float64 = 90.0,  # Vertical field of view in degrees
        aspect_ratio::Float64 = 16.0/9.0,
        aperture::Float64 = 0.0,
        focus_dist::Float64 = 1.0
    )
        theta = deg2rad(fov)
        h = tan(theta/2)
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height
        
        w = normalize(position - look_at)
        u = normalize(cross(up, w))
        v = cross(w, u)
        
        horizontal = focus_dist * viewport_width * u
        vertical = focus_dist * viewport_height * v
        lower_left_corner = position - horizontal/2 - vertical/2 - focus_dist * w
        
        new(position, lower_left_corner, horizontal, vertical, u, v, w, aperture/2)
    end
end

# Generate a ray from the camera
function get_ray(cam::Camera, s::Float64, t::Float64)
    rd = Vec3(0, 0, 0)  # No defocus blur
    offset = Vec3(0, 0, 0)
    
    origin = cam.position + offset
    direction = normalize(cam.lower_left_corner + s*cam.horizontal + t*cam.vertical - origin)
    
    return Ray(origin, direction)
end

# Random functions for sampling
function random_in_unit_sphere()
    while true
        p = 2.0 * Vec3(rand(), rand(), rand()) - Vec3(1, 1, 1)
        if dot(p, p) < 1.0
            return p
        end
    end
end

function random_unit_vector()
    return normalize(random_in_unit_sphere())
end

# Reflect a vector around a normal
function reflect(v::Vec3, n::Vec3)
    return v - 2 * dot(v, n) * n
end

# Calculate ray color (recursive path tracing)
function ray_color(ray::Ray, world::Hittable, depth::Int)
    # If we've exceeded the ray bounce limit, no more light is gathered
    if depth <= 0
        return Vec3(0, 0, 0)
    end
    
    # Check for intersection with the scene
    rec = hit(world, ray, 0.001, Inf)
    
    if rec.hit
        # Emissive material contribution
        emitted = rec.material.emission
        
        # Calculate scattered ray based on material properties
        if rec.material.specular > 0.0
            # Specular reflection
            reflected = reflect(ray.direction, rec.normal)
            # Add some roughness if needed
            if rec.material.roughness > 0.0
                reflected = reflected + rec.material.roughness * random_in_unit_sphere()
            end
            scattered = Ray(rec.position, normalize(reflected))
            
            # Recursive ray tracing with specular reflection
            specular_color = ray_color(scattered, world, depth-1)
            return emitted + rec.material.specular * specular_color * rec.material.diffuse
        else
            # Diffuse reflection
            target = rec.position + rec.normal + random_in_unit_sphere()
            scattered = Ray(rec.position, normalize(target - rec.position))
            
            # Recursive ray tracing with diffuse reflection
            return emitted + 0.5 * ray_color(scattered, world, depth-1) * rec.material.diffuse
        end
    end
    
    # Background color (simple gradient)
    t = 0.5 * (ray.direction.y + 1.0)
    return (1.0-t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0)
end

# ACEScg color space transformation matrix (for more accurate color reproduction)
function to_acescg(color::Vec3)
    # ACEScg tone mapping parameters
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    
    # Apply ACES tone mapping
    r = clamp((color.x * (a * color.x + b)) / (color.x * (c * color.x + d) + e), 0.0, 1.0)
    g = clamp((color.y * (a * color.y + b)) / (color.y * (c * color.y + d) + e), 0.0, 1.0)
    b = clamp((color.z * (a * color.z + b)) / (color.z * (c * color.z + d) + e), 0.0, 1.0)
    
    return RGB{Float32}(r, g, b)
end

# Render function
function render(world::Hittable, camera::Camera, width::Int, height::Int; samples_per_pixel::Int=50, max_depth::Int=20)
    img = Array{RGB{Float32}}(undef, height, width)
    # Store raw HDR data for EXR output
    hdr_data = Array{Vec3}(undef, height, width)
    
    for j in 1:height
        for i in 1:width
            color = Vec3(0, 0, 0)
            
            # Anti-aliasing with multiple samples per pixel
            for _ in 1:samples_per_pixel
                u = (i - 1 + rand()) / (width - 1)
                v = (j - 1 + rand()) / (height - 1)
                ray = get_ray(camera, u, v)
                color = color + ray_color(ray, world, max_depth)
            end
            
            # Average samples
            color = color / samples_per_pixel
            
            # Store the raw HDR value
            hdr_data[height-j+1, i] = color
            
            # Apply tone mapping for display
            img[height-j+1, i] = to_acescg(color)
        end
        
        # Print progress
        if j % 10 == 0
            println("Rendering progress: $(round(Int, 100 * j / height))%")
        end
    end
    
    return img, hdr_data
end

# Save image as 32-bit EXR with ACEScg color space
function save_exr(hdr_data, filename::String)
    try
        # Convert Vec3 to Array structure for EXR
        width, height = size(hdr_data)
        r_channel = Array{Float32}(undef, height, width)
        g_channel = Array{Float32}(undef, height, width)
        b_channel = Array{Float32}(undef, height, width)
        
        for j in 1:height
            for i in 1:width
                r_channel[j, i] = hdr_data[j, i].x
                g_channel[j, i] = hdr_data[j, i].y
                b_channel[j, i] = hdr_data[j, i].z
            end
        end
        
        # Stack channels into a 3D array
        exr_data = cat(r_channel, g_channel, b_channel, dims=3)
        
        # Save as EXR
        save(filename, colorview(RGB, permutedims(exr_data, (3, 1, 2))))
        println("Saved 32-bit EXR file: $filename")
        return true
    catch e
        println("Error saving EXR file: $e")
        println("Saving as PNG instead...")
        
        # Create RGB image from HDR data
        img = Array{RGB{Float32}}(undef, size(hdr_data)...)
        for j in 1:size(hdr_data, 1)
            for i in 1:size(hdr_data, 2)
                img[j, i] = to_acescg(hdr_data[j, i])
            end
        end
        
        # Save as PNG
        save(replace(filename, ".exr" => ".png"), img)
        return false
    end
end

# Load a mesh from an OBJ file and create a Mesh with BVH
function load_obj_mesh(filename::String, material::Material;
                       scale::Vec3=Vec3(1.0, 1.0, 1.0),
                       rotation::Vec3=Vec3(0.0, 0.0, 0.0),
                       translation::Vec3=Vec3(0.0, 0.0, 0.0),
                       center::Bool=true,
                       normalize_size::Bool=false)
    println("Loading OBJ mesh from: $filename")
    
    # Use MeshIO to load the mesh
    mesh_data = try
        load(filename)
    catch e
        println("Error loading mesh: $e")
        
        # Fallback: parse the OBJ file manually
        vertices = Vector{Vec3}()
        faces = Vector{Vector{Int}}()
        
        open(filename, "r") do file
            for line in eachline(file)
                if startswith(line, "v ")
                    # Parse vertex
                    parts = split(line)
                    x = parse(Float64, parts[2])
                    y = parse(Float64, parts[3])
                    z = parse(Float64, parts[4])
                    push!(vertices, Vec3(x, y, z))
                elseif startswith(line, "f ")
                    # Parse face (supporting triangular faces only for this example)
                    parts = split(line)
                    indices = Int[]
                    
                    # OBJ indices are 1-based, so we can use them directly in Julia
                    for i in 2:length(parts)
                        # Handle different face formats (v, v/vt, v/vt/vn)
                        vertex_index = parse(Int, split(parts[i], "/")[1])
                        push!(indices, vertex_index)
                    end
                    
                    # Only triangles supported for simplicity
                    if length(indices) == 3
                        push!(faces, indices)
                    elseif length(indices) > 3
                        # Triangulate if more than 3 vertices (fan triangulation)
                        for i in 3:length(indices)
                            push!(faces, [indices[1], indices[i-1], indices[i]])
                        end
                    end
                end
            end
        end
        
        # Create triangles
        triangles = Triangle[]
        for face in faces
            # Skip faces with invalid indices
            if any(idx -> idx > length(vertices), face)
                continue
            end
            
            # Create triangle from face vertices
            triangle_vertices = [vertices[face[1]], vertices[face[2]], vertices[face[3]]]
            push!(triangles, Triangle(triangle_vertices, material))
        end
        
        return Mesh(triangles)
    end
    
    # Extract vertices and faces from the loaded mesh
    vertices = Vector{Vec3}()
    triangles = Vector{Triangle}()
    
    # Convert mesh vertices to our Vec3 format
    for point in coordinates(mesh_data)
        push!(vertices, Vec3(Float64(point[1]), Float64(point[2]), Float64(point[3])))
    end
    
    # Center the mesh if requested
    if center
        # Calculate centroid
        centroid = Vec3(0.0, 0.0, 0.0)
        for v in vertices
            centroid = centroid + v
        end
        centroid = centroid / length(vertices)
        
        # Subtract centroid from all vertices
        for i in 1:length(vertices)
            vertices[i] = vertices[i] - centroid
        end
    end
    
    # Normalize size if requested
    if normalize_size
        # Find the maximum distance from the origin
        max_dist = 0.0
        for v in vertices
            dist = length(v)
            if dist > max_dist
                max_dist = dist
            end
        end
        
        # Scale vertices to fit in a unit sphere
        if max_dist > 0.0
            scale_factor = 1.0 / max_dist
            for i in 1:length(vertices)
                vertices[i] = vertices[i] * scale_factor
            end
        end
    end
    
    # Apply rotation (using simple Euler angles for this example)
    if rotation.x != 0 || rotation.y != 0 || rotation.z != 0
        for i in 1:length(vertices)
            v = vertices[i]
            
            # X-axis rotation
            if rotation.x != 0
                theta = deg2rad(rotation.x)
                y = v.y * cos(theta) - v.z * sin(theta)
                z = v.y * sin(theta) + v.z * cos(theta)
                v = Vec3(v.x, y, z)
            end
            
            # Y-axis rotation
            if rotation.y != 0
                theta = deg2rad(rotation.y)
                x = v.x * cos(theta) + v.z * sin(theta)
                z = -v.x * sin(theta) + v.z * cos(theta)
                v = Vec3(x, v.y, z)
            end
            
            # Z-axis rotation
            if rotation.z != 0
                theta = deg2rad(rotation.z)
                x = v.x * cos(theta) - v.y * sin(theta)
                y = v.x * sin(theta) + v.y * cos(theta)
                v = Vec3(x, y, v.z)
            end
            
            vertices[i] = v
        end
    end
    
    # Apply scaling
    if scale.x != 1.0 || scale.y != 1.0 || scale.z != 1.0
        for i in 1:length(vertices)
            vertices[i] = Vec3(
                vertices[i].x * scale.x,
                vertices[i].y * scale.y,
                vertices[i].z * scale.z
            )
        end
    end
    
    # Apply translation
    if translation.x != 0.0 || translation.y != 0.0 || translation.z != 0.0
        for i in 1:length(vertices)
            vertices[i] = vertices[i] + translation
        end
    end
    
    # Convert mesh faces to triangles
    face_count = 0
    
    for face in GeometryBasics.faces(mesh_data)
        # Skip faces with more than 3 vertices (we only support triangles)
        if length(face) > 3
            # Simple triangulation - create a fan of triangles from the first vertex
            for i in 3:length(face)
                if face[1] <= length(vertices) && face[i-1] <= length(vertices) && face[i] <= length(vertices)
                    v1 = vertices[face[1]]
                    v2 = vertices[face[i-1]]
                    v3 = vertices[face[i]]
                    push!(triangles, Triangle([v1, v2, v3], material))
                    face_count += 1
                end
            end
        else
            # Extract the three vertices for this face
            if all(idx -> idx <= length(vertices), face)
                v1 = vertices[face[1]]
                v2 = vertices[face[2]]
                v3 = vertices[face[3]]
                push!(triangles, Triangle([v1, v2, v3], material))
                face_count += 1
            end
        end
    end
    
    println("Loaded OBJ mesh with $(length(vertices)) vertices and $face_count triangular faces")
    
    # Create a mesh object with BVH
    return Mesh(triangles)
end

# Create a simple scene with built-in primitives
function create_default_scene()
    # Example objects
    objects = Hittable[]
    
    # Add a ground plane (represented as a large sphere)
    push!(objects, Sphere(Vec3(0, -100.5, -1), 100, Material(diffuse=Vec3(0.8, 0.8, 0.2))))
    
    # Add a center sphere (red diffuse)
    push!(objects, Sphere(Vec3(0, 0, -1), 0.5, Material(diffuse=Vec3(0.8, 0.2, 0.2))))
    
    # Add a metal sphere (golden)
    push!(objects, Sphere(Vec3(1, 0, -1), 0.5, Material(diffuse=Vec3(0.8, 0.6, 0.2), specular=0.8, roughness=0.3)))
    
    # Add a glass-like sphere
    push!(objects, Sphere(Vec3(-1, 0, -1), 0.5, Material(diffuse=Vec3(0.8, 0.8, 0.8), specular=1.0, roughness=0.0)))
    
    # Add a light source
    push!(objects, Sphere(Vec3(0, 2, 0), 0.5, Material(diffuse=Vec3(0.8, 0.8, 0.8), emission=Vec3(4, 4, 4))))
    
    # Add a triangle (green)
    vertices = [Vec3(-0.5, 0, -2), Vec3(0.5, 0, -2), Vec3(0, 1, -2)]
    push!(objects, Triangle(vertices, Material(diffuse=Vec3(0.2, 0.8, 0.2))))
    
    # Wrap scene in BVH for efficiency
    scene = BoundingVolumeHierarchy(objects)
    
    # Define camera
    camera = Camera(
        position=Vec3(0.0, 1.0, 3.0),
        look_at=Vec3(0.0, 0.0, -1.0),
        up=Vec3(0.0, 1.0, 0.0),
        fov=45.0,
        aspect_ratio=16.0/9.0  # Explicitly set 16:9 aspect ratio to match output dimensions
    )
    
    return scene, camera
end

# Create a scene with a single OBJ mesh
function create_obj_scene()
    # Example objects
    objects = Hittable[]
    
    # Add a ground plane (represented as a large sphere)
    push!(objects, Sphere(Vec3(0, -100.5, -1), 100, Material(diffuse=Vec3(0.8, 0.8, 0.2))))
    
    # Add a light source
    push!(objects, Sphere(Vec3(0, 2, 0), 0.5, Material(diffuse=Vec3(0.8, 0.8, 0.8), emission=Vec3(4, 4, 4))))
    
    # Define a material for the mesh
    mesh_material = Material(diffuse=Vec3(0.7, 0.3, 0.2), specular=0.2, roughness=0.4)
    
    # Path to the OBJ file - replace with your actual file path
    obj_file = expanduser("~/Downloads/Lemon_200k.obj")  # Stanford bunny is a common test model
    
    # Check if the file exists
    if isfile(obj_file)
        println("Loading OBJ mesh from: $obj_file")
        
        # Load the mesh with BVH
        mesh = load_obj_mesh(
            obj_file,
            mesh_material,
            center=true,                        # Center the mesh at origin
            normalize_size=true,                # Normalize to unit size
            scale=Vec3(3.0, 3.0, 3.0),          # Scale to 3x size
            rotation=Vec3(0.0, 90.0, 0.0),      # Rotate 90° around Y axis
            translation=Vec3(0.0, 0.0, -1.0)    # Position at Z=-1
        )
        
        # Add the mesh to our scene
        push!(objects, mesh)
        
        println("Added mesh to the scene")
    else
        # Fallback: Add a simple sphere if OBJ file not found
        println("OBJ file not found, adding a sphere instead")
        push!(objects, Sphere(Vec3(0, 0, -1), 0.5, mesh_material))
    end
    
    # Wrap scene in BVH for efficiency
    scene = BoundingVolumeHierarchy(objects)
    
    # Define camera
    camera = Camera(
        position=Vec3(0.0, 1.0, 3.0),
        look_at=Vec3(0.0, 0.0, -1.0),
        up=Vec3(0.0, 1.0, 0.0),
        fov=45.0,
        aspect_ratio=16.0/9.0
    )
    
    return scene, camera
end

# Create a scene with multiple transformed OBJ meshes
function create_multiple_obj_scene()
    objects = Hittable[]
    
    # Add ground plane
    push!(objects, Sphere(Vec3(0, -100.5, -1), 100, Material(diffuse=Vec3(0.8, 0.8, 0.2))))
    
    # Add lights
    push!(objects, Sphere(Vec3(-2, 3, 2), 0.5, Material(diffuse=Vec3(0.8, 0.8, 0.8), emission=Vec3(5, 5, 5))))
    push!(objects, Sphere(Vec3(2, 2, 1), 0.25, Material(diffuse=Vec3(0.8, 0.6, 0.2), emission=Vec3(3, 2, 1))))
    
    # Path to the OBJ file
    obj_file = expanduser("~/Downloads/Lemon_200k.obj")  # Stanford bunny is a common test model
    
    if isfile(obj_file)
        # Example 1: Load a mesh with metallic material
        metal_material = Material(diffuse=Vec3(0.8, 0.8, 0.9), specular=0.8, roughness=0.1)
        mesh1 = load_obj_mesh(
            obj_file,
            metal_material,
            center=true,                      # Center the mesh
            normalize_size=true,              # Normalize size
            scale=Vec3(1.0, 1.0, 1.0),        # No scaling
            rotation=Vec3(0.0, 0.0, 0.0),     # No rotation
            translation=Vec3(0.0, 0.5, -1.5)  # Push back slightly
        )
        push!(objects, mesh1)
        
        # Example 2: Load the same mesh with diffuse material
        diffuse_material = Material(diffuse=Vec3(0.2, 0.8, 0.3), specular=0.0, roughness=1.0)
        mesh2 = load_obj_mesh(
            obj_file,
            diffuse_material,
            center=true,
            normalize_size=true,
            scale=Vec3(1.0, 1.0, 1.0),           # Larger scaling for a different look
            rotation=Vec3(0.0, 45.0, 0.0),       # Rotate 45° around Y axis
            translation=Vec3(1.8, 0.5, -1.0)     # Position to the right
        )
        push!(objects, mesh2)
        
        # Example 3: Load the same mesh with glass material
        glass_material = Material(diffuse=Vec3(0.9, 0.9, 0.9), specular=1.0, roughness=0.0)
        mesh3 = load_obj_mesh(
            obj_file,
            glass_material,
            center=true,
            normalize_size=true,
            scale=Vec3(1.0, 1.0, 1.0),           # Scale up some
            rotation=Vec3(20.0, -30.0, 0.0),     # Multiple axis rotation
            translation=Vec3(-1.8, 0.6, -1.8)    # Position to the left and slightly up
        )
        push!(objects, mesh3)
    else
        # Fallback with spheres if OBJ file not found
        println("OBJ file not found, using spheres instead")
        
        push!(objects, Sphere(Vec3(0, 0, -1.5), 0.5, Material(diffuse=Vec3(0.8, 0.8, 0.9), specular=0.8, roughness=0.1)))
        push!(objects, Sphere(Vec3(1.2, 0, -1.0), 0.5, Material(diffuse=Vec3(0.2, 0.8, 0.3), specular=0.0, roughness=1.0)))
        push!(objects, Sphere(Vec3(-1.2, 0.1, -0.8), 0.5, Material(diffuse=Vec3(0.9, 0.9, 0.9), specular=1.0, roughness=0.0)))
    end
    
    # Wrap scene in BVH for efficiency
    scene = BoundingVolumeHierarchy(objects)
    
    # Define camera from a slightly higher angle to see all meshes
    camera = Camera(
        position=Vec3(0.0, 1.5, 4.0),
        look_at=Vec3(0.0, 0.0, -1.0),
        up=Vec3(0.0, 1.0, 0.0),
        fov=40.0,
        aspect_ratio=16.0/9.0
    )
    
    return scene, camera
end

# --- Vec3f for GPU (Float32) ---
struct Vec3f
    x::Float32
    y::Float32
    z::Float32
end

# Constructor for Vec3f
Vec3f(x::Real, y::Real, z::Real) = Vec3f(Float32(x), Float32(y), Float32(z))
Vec3f(v::Vec3f) = v # Allow conversion from itself, useful in generic contexts
Vec3f() = Vec3f(0f0, 0f0, 0f0) # Default constructor

# Conversion from Vec3 (Float64) to Vec3f (Float32)
function toVec3f(v_d::Vec3)
    return Vec3f(Float32(v_d.x), Float32(v_d.y), Float32(v_d.z))
end

# Basic operations for Vec3f (mirroring Vec3, but with Float32)
import Base: +, -, *, /, show
import LinearAlgebra: dot, normalize # Ensure these are imported for Vec3f too

+(a::Vec3f, b::Vec3f) = Vec3f(a.x + b.x, a.y + b.y, a.z + b.z)
-(a::Vec3f, b::Vec3f) = Vec3f(a.x - b.x, a.y - b.y, a.z - b.z)
*(a::Vec3f, b::Real) = Vec3f(a.x * Float32(b), a.y * Float32(b), a.z * Float32(b))
*(b::Real, a::Vec3f) = a * Float32(b)
/(a::Vec3f, b::Real) = Vec3f(a.x / Float32(b), a.y / Float32(b), a.z / Float32(b))
dot(a::Vec3f, b::Vec3f) = a.x * b.x + a.y * b.y + a.z * b.z
Base.length(a::Vec3f) = sqrt(dot(a, a))
normalize(a::Vec3f) = isapprox(length(a), 0.0f0) ? Vec3f(0f0,0f0,0f0) : a / length(a)
show(io::IO, v::Vec3f) = print(io, "($(v.x), $(v.y), $(v.z))")

# Vector componentwise multiplication for colors (Float32)
Base.:*(a::Vec3f, b::Vec3f) = Vec3f(a.x * b.x, a.y * b.y, a.z * b.z)

# Cross product (Float32)
cross(a::Vec3f, b::Vec3f) = Vec3f(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
)

# --- GPU Data Structures (using Vec3f and Float32 for Metal compatibility) ---
struct GPUSphere
    center::Vec3f        # 12 bytes
    radius::Float32      # 4 bytes
    material_idx::UInt32 # 4 bytes
    padding::UInt32      # 4 bytes to make struct 24 bytes (good for some alignments, can be adjusted)
                         # Let's make it 32 bytes total for safety: 12+4+4=20. padding=12 bytes.
                         # Vec3f (12) + Float32 (4) + UInt32 (4) = 20 bytes.
                         # To reach 32 bytes (multiple of 16 often good): padding1 (4), padding2 (4), padding3 (4).
    # So, 12 (center) + 4 (radius) + 4 (mat_idx) = 20.
    # Let's adjust padding to make total 32 bytes.
    # _padding1::UInt32 # Placeholder for explicit padding fields if needed.
    # _padding2::UInt32
    # _padding3::UInt32
end

struct GPUTriangle
    v0::Vec3f            # 12 bytes
    v1::Vec3f            # 12 bytes
    v2::Vec3f            # 12 bytes
    material_idx::UInt32 # 4 bytes
    # Total: 3*12 + 4 = 40 bytes. To make it 48 bytes (multiple of 16):
    _padding1::UInt32    # 4 bytes
    _padding2::UInt32    # 4 bytes
end

struct GPUMaterial
    diffuse::Vec3f       # 12 bytes
    emission::Vec3f      # 12 bytes
    specular::Float32    # 4 bytes
    roughness::Float32   # 4 bytes
    # Total: 12 + 12 + 4 + 4 = 32 bytes. (Good alignment, multiple of 16)
end

struct GPUBVHNode
    aabb_min::Vec3f      # 12 bytes
    aabb_max::Vec3f      # 12 bytes
    data1::UInt32        # 4 bytes (left_child_idx or primitive_ref_start_idx)
    data2::UInt32        # 4 bytes (right_child_idx or primitive_ref_count)
    is_leaf::UInt32      # 4 bytes (0 for interior, 1 for leaf)
    # Total so far: 12+12+4+4+4 = 36 bytes.
    # Padding to 48 bytes (multiple of 16):
    _padding1::UInt32    # 4 bytes
    _padding2::UInt32    # 4 bytes
    _padding3::UInt32    # 4 bytes
end

struct GPUPrimitiveReference
    primitive_type::UInt32      # 4 bytes (e.g., 0 for sphere, 1 for triangle)
    index_in_type_array::UInt32 # 4 bytes (Index into spheres or triangles Vector in GPUSceneData)
    # Total 8 bytes (Good alignment)
end

# Container for all data to be sent to the GPU for rendering the scene
# The actual MTLBuffers will be created from these vectors later.
struct GPUSceneData
    nodes::Vector{GPUBVHNode}
    spheres::Vector{GPUSphere}
    triangles::Vector{GPUTriangle}
    materials::Vector{GPUMaterial}
    primitive_refs::Vector{GPUPrimitiveReference} # List referenced by leaf BVH nodes
end

# Camera data structure for the GPU, matches MSL struct layout
struct GPUCameraData
    position::Vec3f
    lower_left_corner::Vec3f
    horizontal::Vec3f
    vertical::Vec3f
    u::Vec3f
    v::Vec3f
    w::Vec3f
    lens_radius::Float32
end

# --- Serialization for GPU --- 

# Helper to get AABB for GPU primitives (which don't store it directly)
function get_primitive_aabb(sphere::GPUSphere)
    min_p = Vec3f(sphere.center.x - sphere.radius, sphere.center.y - sphere.radius, sphere.center.z - sphere.radius)
    max_p = Vec3f(sphere.center.x + sphere.radius, sphere.center.y + sphere.radius, sphere.center.z + sphere.radius)
    return AABB(Vec3(min_p.x,min_p.y,min_p.z), Vec3(max_p.x,max_p.y,max_p.z)) # Convert back to Vec3 for AABB struct for now
end

function get_primitive_aabb(triangle::GPUTriangle)
    min_x = min(triangle.v0.x, triangle.v1.x, triangle.v2.x)
    min_y = min(triangle.v0.y, triangle.v1.y, triangle.v2.y)
    min_z = min(triangle.v0.z, triangle.v1.z, triangle.v2.z)
    max_x = max(triangle.v0.x, triangle.v1.x, triangle.v2.x)
    max_y = max(triangle.v0.y, triangle.v1.y, triangle.v2.y)
    max_z = max(triangle.v0.z, triangle.v1.z, triangle.v2.z)
    return AABB(Vec3(min_x,min_y,min_z), Vec3(max_x,max_y,max_z))
end

# Struct to hold information about each primitive needed for BVH construction.
# This struct will be created for each individual sphere or triangle (including those from meshes).
struct PrimitiveBuildInfo
    # Bounding box of the primitive (using original Float64 Vec3 for precision during BVH build).
    aabb::AABB
    # Centroid of the primitive (using Float32 Vec3f as it's mainly for splitting heuristics).
    centroid::Vec3f
    # This will be the 1-based index into the initial `temp_primitive_refs` array, which is used to fetch the
    # actual GPUPrimitiveReference when populating `final_ordered_primitive_refs` in leaf nodes.
    primitive_ref_final_idx::UInt32
end

function serialize_scene_for_gpu(all_cpu_objects::Vector{Hittable})::GPUSceneData
    # Step 1: Collect materials, and gather initial info for all individual primitives.
    gpu_materials_list = Vector{GPUMaterial}()
    material_to_gpu_idx_map = Dict{Material, UInt32}()

    initial_primitive_details = Vector{NamedTuple{(:type_id, :gpu_typed_idx, :mat_gpu_idx, :aabb, :centroid), 
                                             Tuple{UInt32, UInt32, UInt32, AABB, Vec3f}}}()
    
    gpu_spheres_list = Vector{GPUSphere}()
    gpu_triangles_list = Vector{GPUTriangle}()

    for cpu_obj in all_cpu_objects
        obj_material_cpu = get_object_material(cpu_obj)
        current_mat_gpu_idx::UInt32 = 0
        if haskey(material_to_gpu_idx_map, obj_material_cpu)
            current_mat_gpu_idx = material_to_gpu_idx_map[obj_material_cpu]
        else
            push!(gpu_materials_list, GPUMaterial(toVec3f(obj_material_cpu.diffuse), toVec3f(obj_material_cpu.emission), Float32(obj_material_cpu.specular), Float32(obj_material_cpu.roughness)))
            current_mat_gpu_idx = UInt32(length(gpu_materials_list))
            material_to_gpu_idx_map[obj_material_cpu] = current_mat_gpu_idx
        end

        if cpu_obj isa Sphere
            gpu_sphere = GPUSphere(toVec3f(cpu_obj.center), Float32(cpu_obj.radius), current_mat_gpu_idx, 0)
            push!(gpu_spheres_list, gpu_sphere)
            gpu_typed_idx = UInt32(length(gpu_spheres_list))
            push!(initial_primitive_details, (type_id=UInt32(0), gpu_typed_idx=gpu_typed_idx, mat_gpu_idx=current_mat_gpu_idx, aabb=bounding_box(cpu_obj), centroid=toVec3f(cpu_obj.center)))
        elseif cpu_obj isa Triangle
            gpu_triangle = GPUTriangle(toVec3f(cpu_obj.vertices[1]), toVec3f(cpu_obj.vertices[2]), toVec3f(cpu_obj.vertices[3]), current_mat_gpu_idx, 0, 0)
            push!(gpu_triangles_list, gpu_triangle)
            gpu_typed_idx = UInt32(length(gpu_triangles_list))
            centroid_f = (toVec3f(cpu_obj.vertices[1]) + toVec3f(cpu_obj.vertices[2]) + toVec3f(cpu_obj.vertices[3])) / 3.0f0
            push!(initial_primitive_details, (type_id=UInt32(1), gpu_typed_idx=gpu_typed_idx, mat_gpu_idx=current_mat_gpu_idx, aabb=bounding_box(cpu_obj), centroid=centroid_f))
        elseif cpu_obj isa Mesh
            for tri_in_mesh in cpu_obj.triangles
                gpu_triangle = GPUTriangle(toVec3f(tri_in_mesh.vertices[1]), toVec3f(tri_in_mesh.vertices[2]), toVec3f(tri_in_mesh.vertices[3]), current_mat_gpu_idx, 0, 0)
                push!(gpu_triangles_list, gpu_triangle)
                gpu_typed_idx = UInt32(length(gpu_triangles_list))
                centroid_f = (toVec3f(tri_in_mesh.vertices[1]) + toVec3f(tri_in_mesh.vertices[2]) + toVec3f(tri_in_mesh.vertices[3])) / 3.0f0
                push!(initial_primitive_details, (type_id=UInt32(1), gpu_typed_idx=gpu_typed_idx, mat_gpu_idx=current_mat_gpu_idx, aabb=bounding_box(tri_in_mesh), centroid=centroid_f))
            end
        end
    end

    # Step 2: Prepare PrimitiveBuildInfo list for BVH sorting and create the initial (unsorted) final_gpu_primitive_refs array.
    num_total_primitives = length(initial_primitive_details)
    primitive_build_infos_for_bvh = Vector{PrimitiveBuildInfo}(undef, num_total_primitives)
    # This `temp_primitive_refs` will be reordered later based on the sorted `primitive_build_infos_for_bvh`.
    temp_primitive_refs = Vector{GPUPrimitiveReference}(undef, num_total_primitives)

    for i in 1:num_total_primitives
        detail = initial_primitive_details[i]
        # `primitive_ref_final_idx` here is the index in the *current*, unsorted `temp_primitive_refs` array.
        primitive_build_infos_for_bvh[i] = PrimitiveBuildInfo(detail.aabb, detail.centroid, UInt32(i)) 
        temp_primitive_refs[i] = GPUPrimitiveReference(detail.type_id, detail.gpu_typed_idx)
    end

    # Step 3: Build the flat BVH node array.
    gpu_bvh_nodes_list = Vector{GPUBVHNode}()
    final_ordered_primitive_refs = Vector{GPUPrimitiveReference}() # This will be the truly final, sorted list.

    if num_total_primitives > 0
        # build_flat_bvh_recursive! will sort `primitive_build_infos_for_bvh` in place.
        # It will use the `primitive_ref_final_idx` (which are original indices into `temp_primitive_refs`)
        # to effectively tell us the order.
        build_flat_bvh_recursive!(gpu_bvh_nodes_list, primitive_build_infos_for_bvh, 1, num_total_primitives, temp_primitive_refs, final_ordered_primitive_refs)
        # After this call, `final_ordered_primitive_refs` is populated in the correct order for BVH leaves.
        # Leaf nodes in `gpu_bvh_nodes_list` will have `data1` as the starting index into `final_ordered_primitive_refs`.
    else 
        # Handle empty scene for BVH.
        push!(gpu_bvh_nodes_list, GPUBVHNode(Vec3f(), Vec3f(), 0, 0, 1, 0,0,0)) # Dummy leaf
    end
    
    println("serialize_scene_for_gpu: BVH construction complete. Nodes: ", length(gpu_bvh_nodes_list),". Primitives: ", num_total_primitives)

    return GPUSceneData(
        gpu_bvh_nodes_list,
        gpu_spheres_list,
        gpu_triangles_list,
        gpu_materials_list,
        final_ordered_primitive_refs # Use the sorted list of primitive references
    )
end

# Recursive BVH builder for GPU (populates a flat array of GPUBVHNodes).
function build_flat_bvh_recursive!(
    gpu_nodes_list::Vector{GPUBVHNode},          # Output: list of BVH nodes
    primitive_build_infos::Vector{PrimitiveBuildInfo}, # Input: primitive info, will be sorted in place by this function
    start_idx::Int, 
    end_idx::Int,
    temp_primitive_refs::Vector{GPUPrimitiveReference}, # Input: original, unsorted primitive references
    final_ordered_primitive_refs::Vector{GPUPrimitiveReference} # Output: sorted primitive references for leaves
)::UInt32 # Returns the 1-based index of the created node in gpu_nodes_list

    num_primitives_for_this_node = end_idx - start_idx + 1

    if num_primitives_for_this_node == 0
        push!(gpu_nodes_list, GPUBVHNode(Vec3f(), Vec3f(), 0, 0, 1, 0,0,0))
        return UInt32(length(gpu_nodes_list))
    end

    current_node_aabb = primitive_build_infos[start_idx].aabb
    for i in (start_idx + 1):end_idx
        current_node_aabb = surrounding_box(current_node_aabb, primitive_build_infos[i].aabb)
    end

    this_node_idx_in_array = UInt32(length(gpu_nodes_list) + 1)
    push!(gpu_nodes_list, GPUBVHNode(Vec3f(),Vec3f(),0,0,0,0,0,0)) # Placeholder

    max_primitives_in_leaf = 4
    if num_primitives_for_this_node <= max_primitives_in_leaf
        # This is a leaf node.
        # The primitives for this leaf are in `primitive_build_infos[start_idx:end_idx]` (already sorted for this leaf section).
        # We need to copy their corresponding entries from `temp_primitive_refs` into `final_ordered_primitive_refs`.
        # The `data1` of the leaf node will be the starting 1-based index in `final_ordered_primitive_refs`.
        
        leaf_start_idx_in_final_refs = UInt32(length(final_ordered_primitive_refs) + 1)
        for i in start_idx:end_idx
            original_ref_idx = primitive_build_infos[i].primitive_ref_final_idx # This is index into temp_primitive_refs
            push!(final_ordered_primitive_refs, temp_primitive_refs[original_ref_idx])
        end
        
        gpu_nodes_list[this_node_idx_in_array] = GPUBVHNode(
            toVec3f(current_node_aabb.min),
            toVec3f(current_node_aabb.max),
            leaf_start_idx_in_final_refs, 
            UInt32(num_primitives_for_this_node),
            UInt32(1), # is_leaf = true
            0,0,0      # Padding
        )
        return this_node_idx_in_array
    else
        # Interior node.
        centroids_aabb_min = primitive_build_infos[start_idx].centroid
        centroids_aabb_max = primitive_build_infos[start_idx].centroid
        for i in (start_idx + 1):end_idx
            c = primitive_build_infos[i].centroid
            centroids_aabb_min = Vec3f(min(centroids_aabb_min.x, c.x), min(centroids_aabb_min.y, c.y), min(centroids_aabb_min.z, c.z))
            centroids_aabb_max = Vec3f(max(centroids_aabb_max.x, c.x), max(centroids_aabb_max.y, c.y), max(centroids_aabb_max.z, c.z))
        end

        extent_x = centroids_aabb_max.x - centroids_aabb_min.x
        extent_y = centroids_aabb_max.y - centroids_aabb_min.y
        extent_z = centroids_aabb_max.z - centroids_aabb_min.z
        
        split_axis = 1
        if extent_y > extent_x && extent_y > extent_z
            split_axis = 2
        elseif extent_z > extent_x && extent_z > extent_y
            split_axis = 3
        end

        sort_key_func = if split_axis == 1
            info -> info.centroid.x
        elseif split_axis == 2
            info -> info.centroid.y
        else 
            info -> info.centroid.z
        end
        sort!(view(primitive_build_infos, start_idx:end_idx), by=sort_key_func)
        
        mid_offset = div(num_primitives_for_this_node -1, 2)
        mid_idx_in_slice = start_idx + mid_offset

        left_child_node_idx  = build_flat_bvh_recursive!(gpu_nodes_list, primitive_build_infos, start_idx, mid_idx_in_slice, temp_primitive_refs, final_ordered_primitive_refs)
        right_child_node_idx = build_flat_bvh_recursive!(gpu_nodes_list, primitive_build_infos, mid_idx_in_slice + 1, end_idx, temp_primitive_refs, final_ordered_primitive_refs)
        
        gpu_nodes_list[this_node_idx_in_array] = GPUBVHNode(
            toVec3f(current_node_aabb.min),
            toVec3f(current_node_aabb.max),
            left_child_node_idx,
            right_child_node_idx,
            UInt32(0), # is_leaf = false
            0,0,0      # Padding
        )
        return this_node_idx_in_array
    end
end

# Function to convert CPU camera to GPU-friendly format
function serialize_camera_for_gpu(camera::Camera)::GPUCameraData
    return GPUCameraData(
        toVec3f(camera.position),
        toVec3f(camera.lower_left_corner),
        toVec3f(camera.horizontal),
        toVec3f(camera.vertical),
        toVec3f(camera.u),
        toVec3f(camera.v),
        toVec3f(camera.w),
        Float32(camera.lens_radius) # Ensure lens_radius is Float32
    )
end

# Helper function to robustly get the Material from a Hittable object
function get_object_material(obj::Hittable)::Material
    if obj isa Sphere
        return obj.material
    elseif obj isa Triangle
        return obj.material
    elseif obj isa Mesh
        if isempty(obj.triangles)
            # This case should ideally not happen if meshes are constructed with triangles
            @warn "Mesh object encountered with no triangles. Returning default material."
            return Material() # Default material
        end
        # Assume all triangles in a mesh share the material of the first triangle
        return obj.triangles[1].material
    elseif obj isa BVHNode
        # A BVHNode itself doesn't have a single material. Its constituent primitives do.
        # This function is intended for primitive objects before they are put into a scene-level BVH.
        error("get_object_material should not be called on a BVHNode directly in this context.")
    else
        error("Unsupported Hittable type for material extraction: $(typeof(obj))")
    end
    return Material() # Should be unreachable if all types are handled
end


# --- Metal Rendering Function ---
function render_metal(device::Metal.MTLDevice, world_objects::Vector{Hittable}, camera_cpu::Camera, width::Int, height::Int; 
                      samples_per_pixel::Int=50, max_depth::Int=10)
    
    println("Serializing scene for GPU...")
    gpu_scene_data = serialize_scene_for_gpu(world_objects)
    gpu_camera_data_struct = serialize_camera_for_gpu(camera_cpu)

    println("Setting up Metal device and queue...")
    # device = Metal.current_device() # No longer needed, device is passed in
    if isnothing(device)
        error("No Metal device found. This code requires an Apple Silicon or AMD GPU on macOS.")
    end
    queue = Metal.MTLCommandQueue(device)

    println("Preparing Metal buffers...")
    # Output image buffer (Vec3f per pixel)
    output_image_buffer_size = width * height * sizeof(Vec3f)
    output_image_mtl = Metal.MTLBuffer(device, output_image_buffer_size) # Rely on default options

    # Camera data buffer
    camera_data_mtl = Metal.MTLBuffer(device, [gpu_camera_data_struct], Metal.SharedStorage) # Add Metal.SharedStorage for data

    # Scene data buffers
    nodes_mtl = Metal.MTLBuffer(device, gpu_scene_data.nodes, Metal.SharedStorage) # Add Metal.SharedStorage for data
    num_bvh_nodes_val = UInt32(length(gpu_scene_data.nodes))
    num_bvh_nodes_mtl = Metal.MTLBuffer(device, [num_bvh_nodes_val], Metal.SharedStorage) # Add Metal.SharedStorage for data

    spheres_mtl = Metal.MTLBuffer(device, gpu_scene_data.spheres, Metal.SharedStorage) # Add Metal.SharedStorage for data
    triangles_mtl = Metal.MTLBuffer(device, gpu_scene_data.triangles, Metal.SharedStorage) # Add Metal.SharedStorage for data
    materials_mtl = Metal.MTLBuffer(device, gpu_scene_data.materials, Metal.SharedStorage) # Add Metal.SharedStorage for data
    primitive_refs_mtl = Metal.MTLBuffer(device, gpu_scene_data.primitive_refs, Metal.SharedStorage) # Add Metal.SharedStorage for data

    # Scalar arguments for the kernel
    image_width_val = UInt32(width)
    image_height_val = UInt32(height)
    samples_per_pixel_val = UInt32(samples_per_pixel)
    max_depth_val = UInt32(max_depth)
    random_seed_offset_val = UInt32(rand(1:100000)) # Add randomness per render

    image_width_mtl = Metal.MTLBuffer(device, [image_width_val], Metal.SharedStorage) # Add Metal.SharedStorage for data
    image_height_mtl = Metal.MTLBuffer(device, [image_height_val], Metal.SharedStorage) # Add Metal.SharedStorage for data
    samples_per_pixel_mtl = Metal.MTLBuffer(device, [samples_per_pixel_val], Metal.SharedStorage) # Add Metal.SharedStorage for data
    max_depth_mtl = Metal.MTLBuffer(device, [max_depth_val], Metal.SharedStorage) # Add Metal.SharedStorage for data
    random_seed_offset_mtl = Metal.MTLBuffer(device, [random_seed_offset_val], Metal.SharedStorage) # Add Metal.SharedStorage for data
    
    println("Compiling Metal kernel...")
    options = Metal.MTLCompileOptions()
    library = Metal.MTLLibrary(device, """
#include <metal_stdlib>
#include <metal_math>
#include <metal_compute>

using namespace metal;

// --- Data Structures (matching Julia GPU structs) ---

struct Vec3f {
    float x;
    float y;
    float z;

    Vec3f(float val = 0.0f) : x(val), y(val), z(val) {}
    Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

// Basic vector operations
inline Vec3f operator+(Vec3f a, Vec3f b) { return Vec3f(a.x + b.x, a.y + b.y, a.z + b.z); }
inline Vec3f operator-(Vec3f a, Vec3f b) { return Vec3f(a.x - b.x, a.y - b.y, a.z - b.z); }
inline Vec3f operator-(Vec3f a) { return Vec3f(-a.x, -a.y, -a.z); }
inline Vec3f operator*(Vec3f a, float s) { return Vec3f(a.x * s, a.y * s, a.z * s); }
inline Vec3f operator*(float s, Vec3f a) { return Vec3f(a.x * s, a.y * s, a.z * s); }
inline Vec3f operator/(Vec3f a, float s) { float inv_s = 1.0f / s; return Vec3f(a.x * inv_s, a.y * inv_s, a.z * inv_s); }
inline float dot(Vec3f a, Vec3f b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float length_squared(Vec3f a) { return dot(a, a); }
inline float length(Vec3f a) { return metal::sqrt(length_squared(a)); }
inline Vec3f normalize(Vec3f a) { float l = length(a); return (l < 1e-6f) ? Vec3f(0.0f) : a / l; }
inline Vec3f cross(Vec3f a, Vec3f b) {
    return Vec3f(a.y * b.z - a.z * b.y,
                 a.z * b.x - a.x * b.z,
                 a.x * b.y - a.y * b.x);
}
inline Vec3f reflect(Vec3f v, Vec3f n) { return v - 2.0f * dot(v, n) * n; }
inline Vec3f cmul(Vec3f a, Vec3f b) { return Vec3f(a.x * b.x, a.y * b.y, a.z * b.z); }


struct GPUSphere {
    Vec3f center;
    float radius;
    uint material_idx;   // 1-based from Julia
    uint _padding_match_julia; // To match Julia\'s 24 bytes total
};

struct GPUTriangle {
    Vec3f v0;
    Vec3f v1;
    Vec3f v2;
    uint material_idx;   // 1-based from Julia
    uint _padding1;      // To match Julia\'s 48 bytes total
    uint _padding2;
};

struct GPUMaterial {
    Vec3f diffuse;
    Vec3f emission;
    float specular;
    float roughness;
    // Total 32 bytes
};

struct GPUBVHNode {
    Vec3f aabb_min;
    Vec3f aabb_max;
    uint data1;          // left_child_idx or primitive_ref_start_idx (1-based)
    uint data2;          // right_child_idx or primitive_ref_count
    uint is_leaf;        // 0 for interior, 1 for leaf
    uint _padding1;      // To match Julia\'s 48 bytes total
    uint _padding2;
    uint _padding3;
};

struct GPUPrimitiveReference {
    uint primitive_type;      // 0 for sphere, 1 for triangle
    uint index_in_type_array; // 1-based index from Julia
};

struct Ray {
    Vec3f origin;
    Vec3f direction;
};

struct HitRecord {
    float t;
    Vec3f position;
    Vec3f normal;
    uint material_idx; // 1-based from Julia
    bool hit;
};

struct GPUCameraData {
    Vec3f position;
    Vec3f lower_left_corner;
    Vec3f horizontal;
    Vec3f vertical;
    Vec3f u, v, w;
    float lens_radius;
    // Total 88 bytes
};

// --- Random Number Generation ---
struct RNGState {
    uint seed;
};

inline float random_float(thread RNGState& state) {
    state.seed = (1664525u * state.seed + 1013904223u);
    return float(state.seed & 0x00FFFFFFu) * (1.0f / 16777216.0f); // [0,1)
}

inline Vec3f random_in_unit_sphere(thread RNGState& state) {
    while (true) {
        Vec3f p = Vec3f(random_float(state), random_float(state), random_float(state)) * 2.0f - Vec3f(1.0f);
        if (length_squared(p) < 1.0f) return p;
    }
}

// --- Intersection Functions ---

inline bool hit_aabb(Vec3f aabb_min, Vec3f aabb_max, thread const Ray& r, float t_min, thread float& t_max_ref) { // Pass t_max by reference
    for (int i = 0; i < 3; ++i) {
        float invD = 1.0f / (i == 0 ? r.direction.x : (i == 1 ? r.direction.y : r.direction.z));
        float origin_comp = (i == 0 ? r.origin.x : (i == 1 ? r.origin.y : r.origin.z));
        float min_comp = (i == 0 ? aabb_min.x : (i == 1 ? aabb_min.y : aabb_min.z));
        float max_comp = (i == 0 ? aabb_max.x : (i == 1 ? aabb_max.y : aabb_max.z));

        float t0 = (min_comp - origin_comp) * invD;
        float t1 = (max_comp - origin_comp) * invD;

        if (invD < 0.0f) metal::swap(t0, t1);
        
        t_min = metal::max(t0, t_min);
        t_max_ref = metal::min(t1, t_max_ref);
        
        if (t_max_ref <= t_min) return false;
    }
    return true;
}

HitRecord hit_sphere(device const GPUSphere& sphere, thread const Ray& r, float t_min, float t_max) {
    HitRecord rec;
    rec.hit = false;
    Vec3f oc = r.origin - sphere.center;
    float a = length_squared(r.direction);
    float half_b = dot(oc, r.direction);
    float c = length_squared(oc) - sphere.radius * sphere.radius;
    float discriminant = half_b * half_b - a * c;

    if (discriminant < 0) return rec;
    
    float sqrtd = metal::sqrt(discriminant);
    float root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) return rec;
    }

    rec.t = root;
    rec.position = r.origin + r.direction * rec.t;
    rec.normal = normalize(rec.position - sphere.center);
    rec.material_idx = sphere.material_idx;
    rec.hit = true;
    return rec;
}

HitRecord hit_triangle(device const GPUTriangle& tri, thread const Ray& r, float t_min, float t_max) {
    HitRecord rec;
    rec.hit = false;

    Vec3f edge1 = tri.v1 - tri.v0;
    Vec3f edge2 = tri.v2 - tri.v0;
    Vec3f h = cross(r.direction, edge2);
    float det = dot(edge1, h);

    if (metal::abs(det) < 1e-8f) return rec;

    float inv_det = 1.0f / det;
    Vec3f s = r.origin - tri.v0;
    float u = inv_det * dot(s, h);

    if (u < 0.0f || u > 1.0f) return rec;

    Vec3f q = cross(s, edge1);
    float v = inv_det * dot(r.direction, q);

    if (v < 0.0f || u + v > 1.0f) return rec;

    float t_intersect = inv_det * dot(edge2, q);

    if (t_intersect > t_min && t_intersect < t_max) {
        rec.t = t_intersect;
        rec.position = r.origin + r.direction * rec.t;
        rec.normal = normalize(cross(edge1, edge2));
        rec.material_idx = tri.material_idx;
        rec.hit = true;
        return rec;
    }
    return rec;
}

// --- BVH Traversal ---
HitRecord hit_bvh(
    thread const Ray& r, float t_min, 
    device const GPUBVHNode* nodes,
    uint num_bvh_nodes,
    device const GPUSphere* spheres,
    device const GPUTriangle* triangles,
    device const GPUPrimitiveReference* primitive_refs)
{
    HitRecord closest_hit_rec;
    closest_hit_rec.hit = false;
    float closest_t = FLT_MAX; // t_max for intersections

    uint stack[64]; 
    int stack_ptr = 0;
    if (num_bvh_nodes == 0) return closest_hit_rec; // Empty BVH
    stack[stack_ptr++] = 1; // Start with root node (node 1)

    while (stack_ptr > 0) {
        uint node_idx_1based = stack[--stack_ptr];
        
        if (node_idx_1based == 0 || node_idx_1based > num_bvh_nodes) continue;

        device const GPUBVHNode& node = nodes[node_idx_1based - 1];

        float current_node_t_max = closest_t;
        if (!hit_aabb(node.aabb_min, node.aabb_max, r, t_min, current_node_t_max)) {
            continue;
        }

        if (node.is_leaf == 1) {
            uint prim_ref_start_idx_1based = node.data1;
            uint prim_count = node.data2;
            for (uint i = 0; i < prim_count; ++i) {
                uint current_prim_ref_idx_1based = prim_ref_start_idx_1based + i;
                device const GPUPrimitiveReference& prim_ref = primitive_refs[current_prim_ref_idx_1based - 1];

                HitRecord temp_rec;
                temp_rec.hit = false;

                if (prim_ref.primitive_type == 0) { // Sphere
                    temp_rec = hit_sphere(spheres[prim_ref.index_in_type_array - 1], r, t_min, closest_t);
                } else { // Triangle (type == 1)
                    temp_rec = hit_triangle(triangles[prim_ref.index_in_type_array - 1], r, t_min, closest_t);
                }

                if (temp_rec.hit && temp_rec.t < closest_t) {
                    closest_hit_rec = temp_rec;
                    closest_t = temp_rec.t;
                }
            }
        } else { // Interior node
            if (stack_ptr < 62) { 
                uint left_child_1based = node.data1;
                uint right_child_1based = node.data2;
                
                if (right_child_1based > 0 && right_child_1based <= num_bvh_nodes) {
                     stack[stack_ptr++] = right_child_1based;
                }
                if (left_child_1based > 0 && left_child_1based <= num_bvh_nodes) {
                    stack[stack_ptr++] = left_child_1based;
                }
            }
        }
    }
    return closest_hit_rec;
}

// --- Main Tracing Logic ---
Vec3f trace_ray_gpu(
    thread Ray& r,
    uint max_depth_val,
    thread RNGState& rng,
    device const GPUBVHNode* nodes,
    uint num_bvh_nodes,
    device const GPUSphere* spheres,
    device const GPUTriangle* triangles,
    device const GPUMaterial* materials,
    device const GPUPrimitiveReference* primitive_refs)
{
    Vec3f accumulated_color(0.0f);
    Vec3f current_attenuation(1.0f);

    for (uint depth_iter = 0; depth_iter < max_depth_val; ++depth_iter) {
        HitRecord rec = hit_bvh(r, 0.001f, nodes, num_bvh_nodes, spheres, triangles, primitive_refs);

        if (rec.hit) {
            device const GPUMaterial& mat = materials[rec.material_idx - 1];
            
            accumulated_color = accumulated_color + cmul(current_attenuation, mat.emission);

            if (mat.specular > 0.0f) {
                Vec3f reflected_dir = reflect(normalize(r.direction), rec.normal);
                if (mat.roughness > 0.0f) {
                    reflected_dir = normalize(reflected_dir + mat.roughness * random_in_unit_sphere(rng));
                }
                r.origin = rec.position;
                r.direction = reflected_dir;
                current_attenuation = cmul(current_attenuation, mat.diffuse) * mat.specular;
            } else { 
                Vec3f scatter_direction = normalize(rec.normal + random_in_unit_sphere(rng));
                if (length_squared(scatter_direction) < 1e-6f) {
                    scatter_direction = rec.normal;
                }
                r.origin = rec.position;
                r.direction = scatter_direction;
                current_attenuation = cmul(current_attenuation, mat.diffuse) * 0.5f;
            }
            
            if (max(current_attenuation.x, max(current_attenuation.y, current_attenuation.z)) < 0.01f && depth_iter > 3) {
                 break;
            }

        } else {
            float t_bg = 0.5f * (normalize(r.direction).y + 1.0f);
            Vec3f bg_color = (1.0f - t_bg) * Vec3f(1.0f, 1.0f, 1.0f) + t_bg * Vec3f(0.5f, 0.7f, 1.0f);
            accumulated_color = accumulated_color + cmul(current_attenuation, bg_color);
            break;
        }
    }
    return accumulated_color;
}

// --- Kernel Function ---
kernel void ray_color_kernel(
    device Vec3f* output_image [[buffer(0)]],
    device const GPUCameraData& camera_data [[buffer(1)]],
    device const GPUBVHNode* nodes [[buffer(2)]],
    device const uint& num_bvh_nodes [[buffer(3)]], 
    device const GPUSphere* spheres [[buffer(4)]],
    device const GPUTriangle* triangles [[buffer(5)]],
    device const GPUMaterial* materials [[buffer(6)]],
    device const GPUPrimitiveReference* primitive_refs [[buffer(7)]],
    device const uint& image_width [[buffer(8)]],
    device const uint& image_height [[buffer(9)]],
    device const uint& samples_per_pixel [[buffer(10)]],
    device const uint& max_depth [[buffer(11)]],
    device const uint& random_seed_offset [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]]) 
{
    if (gid.x >= image_width || gid.y >= image_height) {
        return;
    }

    RNGState rng_state;
    rng_state.seed = gid.x + gid.y * image_width + random_seed_offset + samples_per_pixel;

    Vec3f total_color(0.0f);

    for (uint s = 0; s < samples_per_pixel; ++s) {
        rng_state.seed += s * 31;

        float u_coord = (float(gid.x) + random_float(rng_state)) / float(image_width -1);
        float v_coord = (float(gid.y) + random_float(rng_state)) / float(image_height - 1);

        Ray r;
        r.origin = camera_data.position; 
        r.direction = normalize(camera_data.lower_left_corner +
                                u_coord * camera_data.horizontal +
                                v_coord * camera_data.vertical -
                                r.origin);
        
        total_color = total_color + trace_ray_gpu(r, max_depth, rng_state, nodes, num_bvh_nodes, spheres, triangles, materials, primitive_refs);
    }

    total_color = total_color / float(samples_per_pixel);

    uint linear_idx = gid.y * image_width + gid.x;
    output_image[linear_idx] = total_color;
}

""", options) # End of embedded MSL string
    if isnothing(library)
        error("Failed to compile Metal library. Check MSL_KERNEL_STRING for errors.")
    end
    kernel_function = Metal.MTLFunction(library, "ray_color_kernel")
    pipeline_state = Metal.MTLComputePipelineState(device, kernel_function)

    println("Encoding and dispatching Metal kernel...")
    command_buffer = Metal.MTLCommandBuffer(queue)
    encoder = Metal.MTLComputeCommandEncoder(command_buffer)
    Metal.set_compute_pipeline_state!(encoder, pipeline_state)

    Metal.set_buffer!(encoder, output_image_mtl, 0, 0)
    Metal.set_buffer!(encoder, camera_data_mtl, 0, 1)
    Metal.set_buffer!(encoder, nodes_mtl, 0, 2)
    Metal.set_buffer!(encoder, num_bvh_nodes_mtl,0, 3)
    Metal.set_buffer!(encoder, spheres_mtl, 0, 4)
    Metal.set_buffer!(encoder, triangles_mtl, 0, 5)
    Metal.set_buffer!(encoder, materials_mtl, 0, 6)
    Metal.set_buffer!(encoder, primitive_refs_mtl, 0, 7)
    Metal.set_buffer!(encoder, image_width_mtl, 0, 8)
    Metal.set_buffer!(encoder, image_height_mtl, 0, 9)
    Metal.set_buffer!(encoder, samples_per_pixel_mtl, 0, 10)
    Metal.set_buffer!(encoder, max_depth_mtl, 0, 11)
    Metal.set_buffer!(encoder, random_seed_offset_mtl, 0, 12)
    
    # Calculate grid and threadgroup sizes
    # MaxTotalThreadsPerThreadgroup is typically 1024 for Apple GPUs
    # Let's use 16x16 = 256 threads per group, or 8x8=64 for smaller tasks
    # Choose threadgroup size (e.g., 16x16, or check device limits)
    w = Metal.max_total_threads_per_threadgroup(pipeline_state)
    # A common 2D configuration that fits 256, 512, or 1024 limits:
    threads_per_group_x = 16
    threads_per_group_y = 16
    if threads_per_group_x * threads_per_group_y > w
        threads_per_group_x = 8
        threads_per_group_y = 8
    end

    threadgroup_size = Metal.MTLSize(threads_per_group_x, threads_per_group_y, 1)
    grid_size = Metal.MTLSize(ceil(Int, width / threads_per_group_x), ceil(Int, height / threads_per_group_y), 1)
    
    Metal.dispatch_threads!(encoder, grid_size, threadgroup_size)
    Metal.end_encoding!(encoder)
    Metal.commit!(command_buffer)
    Metal.wait_until_completed!(command_buffer)

    println("Metal kernel execution complete. Retrieving image...")
    # Retrieve data from output_image_mtl
    # The buffer contains Vec3f, convert to Array{RGB{Float32}} for display/saving
    # Or Array{Vec3} (Float64) for hdr_data
    raw_hdr_output = Metal.download(output_image_mtl, Vector{Vec3f}, width * height)
    
    # Reshape and convert to desired output formats
    hdr_data_cpu = Array{Vec3}(undef, height, width) # For EXR saving (Float64 Vec3)
    img_display = Array{RGB{Float32}}(undef, height, width) # For display (tonemapped)
    
    for j_idx in 1:height # j_idx is 1-based Julia index for rows
        for i_idx in 1:width # i_idx is 1-based Julia index for columns
            # Metal output is linear array, (0,0) is often top-left in GPU memory if not specified otherwise
            # The kernel writes with (height - 1 - j) to flip Y for typical image coords.
            # So, reading linearly should match this.
            linear_idx = (j_idx - 1) * width + i_idx
            vec3f_val = raw_hdr_output[linear_idx]
            
            # Store for EXR (convert Vec3f to Vec3 Float64)
            hdr_data_cpu[j_idx, i_idx] = Vec3(Float64(vec3f_val.x), Float64(vec3f_val.y), Float64(vec3f_val.z))
            # Apply tone mapping for display (using existing to_acescg that takes Vec3)
            img_display[j_idx, i_idx] = to_acescg(hdr_data_cpu[j_idx, i_idx])
        end
    end

    println("Rendering complete.")
    return img_display, hdr_data_cpu
end

# Modified render_example to use Metal renderer
function render_example(; device::Union{Nothing, Metal.MTLDevice}, width=1280, height=720, samples=100, max_depth=10, interactive=true, output_file="optimized_render_metal.exr", scene_type="default", renderer="metal")
    # Create the appropriate scene (CPU objects first)
    println("Creating $scene_type scene for $renderer renderer...")
    world_hittables = Hittable[] # This will be the list of top-level objects
    local camera_setup::Camera # Ensure camera_setup is defined

    if scene_type == "obj"
        # For obj_scene, it returns scene (BVH) and camera.
        # We need the raw objects list that went into the BVH.
        # Temporarily, let's reconstruct the objects list or modify create_obj_scene.
        # For simplicity, let's assume the structure allows access to original objects.
        # This requires modifying create_obj_scene. For now, let's assume it does.

        # Let's redefine create_obj_scene slightly for this purpose or assume a common pattern.
        # For now, this is a simplification: we'll assume the structure allows access to original objects. THIS NEEDS REFACTORING of scene creation.
        # A temporary kludge for the provided file structure:
        # The create_..._scene functions build `objects` then wrap in `BoundingVolumeHierarchy`.
        # Let's just use `create_default_scene` for now, which is easier to reason about its `objects` list.
        # And assume `create_default_scene` is modified to return `objects, camera`

        if scene_type == "default"
            world_hittables, camera_setup = create_default_scene_modified_for_metal_demo()
        elseif scene_type == "obj"
            world_hittables, camera_setup = create_obj_scene_modified_for_metal_demo()
        elseif scene_type == "multiple_obj"
            world_hittables, camera_setup = create_multiple_obj_scene_modified_for_metal_demo()
        else
            world_hittables, camera_setup = create_default_scene_modified_for_metal_demo()
        end

    else # default or multiple_obj. Assume they can be modified.
        if scene_type == "default"
            world_hittables, camera_setup = create_default_scene_modified_for_metal_demo()
        elseif scene_type == "multiple_obj"
            world_hittables, camera_setup = create_multiple_obj_scene_modified_for_metal_demo()
        else # Fallback
            world_hittables, camera_setup = create_default_scene_modified_for_metal_demo()
        end
    end

    image = nothing
    hdr_data = nothing

    if renderer == "metal"
        println("Rendering with Metal: $samples samples per pixel...")
        # Ensure Metal.jl is functional before calling
        try
            if isnothing(device)
                error("Metal device not available to render_metal.")
            end
            # Metal.current_device() # Check if Metal is available - No longer needed, pass device
            image, hdr_data = render_metal(device, world_hittables, camera_setup, width, height, samples_per_pixel=samples, max_depth=max_depth)
        catch e
            println("Metal rendering failed: $e")
            println("Falling back to CPU renderer.")
            # Fallback to CPU rendering
            scene_cpu_bvh = BoundingVolumeHierarchy(world_hittables) # Build CPU BVH for CPU renderer
            image, hdr_data = render(scene_cpu_bvh, camera_setup, width, height, samples_per_pixel=samples, max_depth=max_depth) # Original render call
        end
    else # CPU renderer (original)
        println("Rendering with CPU: $samples samples per pixel...")
        scene_cpu_bvh = BoundingVolumeHierarchy(world_hittables) # Ensure CPU BVH is built
        image, hdr_data = render(scene_cpu_bvh, camera_setup, width, height, samples_per_pixel=samples, max_depth=max_depth)
    end
    
    # Save as EXR (ACEScg color space, 32-bit)
    if !interactive || output_file != ""
        println("Saving to $output_file...")
        save_exr(hdr_data, output_file)
    end
    
    # Display the image if interactive mode
    if interactive && !isnothing(image)
        println("Displaying rendered image...")
        # Ensure Plots.jl is happy with the image format
        if image isa Array{RGB{Float32}}
            display(plot(image, size=(width, height)))
        else
            println("Cannot display image: format not RGB{Float32}.")
        end
    end
    
    return image, hdr_data
end

# Dummy modified scene functions for the demo. User needs to implement these properly.
function create_default_scene_modified_for_metal_demo()
    objects = Hittable[]
    push!(objects, Sphere(Vec3(0, -100.5, -1), 100, Material(diffuse=Vec3(0.8, 0.8, 0.2))))
    push!(objects, Sphere(Vec3(0, 0, -1), 0.5, Material(diffuse=Vec3(0.8, 0.2, 0.2))))
    push!(objects, Sphere(Vec3(1, 0, -1), 0.5, Material(diffuse=Vec3(0.8, 0.6, 0.2), specular=0.8, roughness=0.3)))
    push!(objects, Sphere(Vec3(-1, 0, -1), 0.5, Material(diffuse=Vec3(0.8, 0.8, 0.8), specular=1.0, roughness=0.0)))
    push!(objects, Sphere(Vec3(0, 2, 0), 0.5, Material(diffuse=Vec3(0.8, 0.8, 0.8), emission=Vec3(4, 4, 4))))
    vertices = [Vec3(-0.5, 0, -2), Vec3(0.5, 0, -2), Vec3(0, 1, -2)]
    push!(objects, Triangle(vertices, Material(diffuse=Vec3(0.2, 0.8, 0.2))))
    camera = Camera(position=Vec3(0.0, 1.0, 3.0), look_at=Vec3(0.0, 0.0, -1.0), up=Vec3(0.0, 1.0, 0.0), fov=45.0, aspect_ratio=16.0/9.0)
    return objects, camera
end

function create_obj_scene_modified_for_metal_demo()
    objects = Hittable[]
    push!(objects, Sphere(Vec3(0, -100.5, -1), 100, Material(diffuse=Vec3(0.8, 0.8, 0.2))))
    push!(objects, Sphere(Vec3(0, 2, 0), 0.5, Material(diffuse=Vec3(0.8, 0.8, 0.8), emission=Vec3(4, 4, 4))))
    mesh_material = Material(diffuse=Vec3(0.7, 0.3, 0.2), specular=0.2, roughness=0.4)
    obj_file = expanduser("~/Downloads/Lemon_200k.obj")
    if isfile(obj_file)
        mesh = load_obj_mesh(obj_file, mesh_material, center=true, normalize_size=true, scale=Vec3(3.0,3.0,3.0), rotation=Vec3(0.0,90.0,0.0), translation=Vec3(0.0,0.0,-1.0))
        push!(objects, mesh)
    else
        push!(objects, Sphere(Vec3(0,0,-1), 0.5, mesh_material))
    end
    camera = Camera(position=Vec3(0.0, 1.0, 3.0), look_at=Vec3(0.0, 0.0, -1.0), up=Vec3(0.0, 1.0, 0.0), fov=45.0, aspect_ratio=16.0/9.0)
    return objects, camera
end

function create_multiple_obj_scene_modified_for_metal_demo()
    objects = Hittable[]
    push!(objects, Sphere(Vec3(0, -100.5, -1), 100, Material(diffuse=Vec3(0.8, 0.8, 0.2))))
    push!(objects, Sphere(Vec3(-2, 3, 2), 0.5, Material(diffuse=Vec3(0.8,0.8,0.8), emission=Vec3(5,5,5))))
    obj_file = expanduser("~/Downloads/Lemon_200k.obj")
    if isfile(obj_file)
        mesh1 = load_obj_mesh(obj_file, Material(diffuse=Vec3(0.8,0.8,0.9), specular=0.8, roughness=0.1), center=true, normalize_size=true, translation=Vec3(0.0,0.5,-1.5))
        push!(objects, mesh1)
        mesh2 = load_obj_mesh(obj_file, Material(diffuse=Vec3(0.2,0.8,0.3)), center=true, normalize_size=true, rotation=Vec3(0.0,45.0,0.0), translation=Vec3(1.8,0.5,-1.0))
        push!(objects, mesh2)
    else
        push!(objects, Sphere(Vec3(0,0,-1.5),0.5,Material(diffuse=Vec3(0.8,0.8,0.9), specular=0.8, roughness=0.1)))
    end
    camera = Camera(position=Vec3(0.0, 1.5, 4.0), look_at=Vec3(0.0,0.0,-1.0), fov=40.0, aspect_ratio=16.0/9.0)
    return objects, camera
end

# Main function to run the renderer
function main()
    # Initialize Metal device early if possible, or ensure it's done in render_metal
    # Metal.device() # Can be called to check availability
    local metal_device::Union{Nothing, Metal.MTLDevice} = nothing
    if Sys.isapple()
        try
            metal_device = Metal.device() # Get device using Metal.device()
            if isnothing(metal_device)
                println("Warning: Metal.device() returned nothing. Metal rendering will likely fail.")
            else
                # Try to get name, but don't error out if Metal.name fails
                try
                    println("Metal device found: ", Metal.name(metal_device))
                catch e
                    println("Could not query Metal device name (non-fatal): $e. Using device object directly.")
                end
            end
        catch e
            println("Could not initialize Metal device: $e. Metal rendering will fail.")
            # Potentially exit or ensure CPU fallback is triggered robustly
            # For now, we'll let it proceed and fail in render_metal if device is truly unusable
        end
    else
        println("Metal is only available on Apple systems.")
    end

    interactive = false # Disable interactive plotting to clean console
    output_file = "optimized_render_metal.exr"
    width = 640 
    height = 360
    samples = 50 # Lower for faster Metal testing initially
    max_depth = 10 # Define max_depth for the rendering
    
    scene_type = "default" # "default", "obj", or "multiple_obj"
    # scene_type = "obj"
    # scene_type = "multiple_obj"
    
    render_example(
        device=metal_device, # Pass the device
        width=width,
        height=height,
        samples=samples,
        max_depth=max_depth, # Pass max_depth to render_example
        interactive=interactive,
        output_file=output_file,
        scene_type=scene_type,
        renderer="metal" # Change to "cpu" to use original renderer
    )
end

# Run the main function if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    # Make sure Metal.jl can find a device if we are on macOS
    if Sys.isapple()
        try
            if Metal.current_device() === nothing
                println("Warning: No Metal device found by Metal.jl. Metal rendering will fail.")
            else
                println("Metal device found: ", Metal.name(Metal.current_device()))
            end
        catch e
            println("Could not query Metal device: $e. Metal rendering might fail.")
        end
    end
    main()
end

