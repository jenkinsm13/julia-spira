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

# Render the scene
function render_example(; width=1280, height=720, samples=100, interactive=true, output_file="optimized_render.exr", scene_type="default")
    # Create the appropriate scene
    println("Creating $scene_type scene...")
    if scene_type == "obj"
        scene, camera = create_obj_scene()
    elseif scene_type == "multiple_obj"
        scene, camera = create_multiple_obj_scene()
    else
        scene, camera = create_default_scene()
    end
    
    println("Rendering with $samples samples per pixel...")
    image, hdr_data = render(scene, camera, width, height, samples_per_pixel=samples, max_depth=10)
    
    # Save as EXR (ACEScg color space, 32-bit)
    if !interactive || output_file != ""
        println("Saving to $output_file...")
        save_exr(hdr_data, output_file)
    end
    
    # Display the image if interactive mode
    if interactive
        println("Displaying rendered image...")
        display(plot(image, size=(width, height)))
    end
    
    return image, hdr_data
end

# Main function to run the renderer
function main()
    # Parse command line arguments here if needed
    interactive = true  # Set to false for headless rendering
    output_file = "optimized_render.exr"
    width = 640         # Lower resolution for faster rendering
    height = 360        # 16:9 aspect ratio
    samples = 50        # Reduced sample count for faster rendering
    
    # Scene type: "default", "obj", or "multiple_obj"
    scene_type = "multiple_obj"  # Try rendering with optimized OBJ mesh
    
    # Run the renderer
    render_example(
        width=width,
        height=height,
        samples=samples,
        interactive=interactive,
        output_file=output_file,
        scene_type=scene_type
    )
end

# Run the main function if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# --- GPU Data Structures & Serialization ---
# These structs are intended to be memory-layout compatible with MSL structs.
# They will be populated by serialization functions and copied to Metal buffers.
# Note: Vec3 uses Float64; for GPU, Float32 is often preferred. Conversion or
# matching MSL types (e.g., using `double` in MSL) will be necessary.

struct GPUSphere
    center::Vec3
    radius::Float32       # Explicitly Float32 for GPU
    material_idx::UInt32
end

struct GPUTriangle
    v0::Vec3
    v1::Vec3
    v2::Vec3
    material_idx::UInt32
end

struct GPUMaterial
    diffuse::Vec3         # Assuming Vec3 remains Float64 for now
    emission::Vec3
    specular::Float32     # Explicitly Float32
    roughness::Float32    # Explicitly Float32
end

struct GPUBVHNode
    aabb_min::Vec3
    aabb_max::Vec3
    # If is_leaf == 1:
    #   data1 is primitive_ref_start_idx (index into an array of GPUPrimitiveReference)
    #   data2 is primitive_ref_count
    # If is_leaf == 0 (interior node):
    #   data1 is left_child_node_idx (index into the GPUBVHNode array)
    #   data2 is right_child_node_idx (index into the GPUBVHNode array)
    data1::UInt32
    data2::UInt32
    is_leaf::UInt32 # 0 for interior, 1 for leaf
    padding::UInt32 # For alignment, making struct size a multiple of 16 bytes if Vec3 is 24 bytes.
                    # (2 * 24) + (4 * 4) = 48 + 16 = 64 bytes.
end

# Reference to a primitive in one of the typed arrays (GPUSphere, GPUTriangle)
struct GPUPrimitiveReference
    primitive_type::UInt32 # e.g., 0 for sphere, 1 for triangle
    index_in_type_array::UInt32 # Index into spheres or triangles Vector in GPUSceneData
end

# Container for all data to be sent to the GPU for rendering the scene
struct GPUSceneData
    nodes::Vector{GPUBVHNode}
    spheres::Vector{GPUSphere}
    triangles::Vector{GPUTriangle}
    materials::Vector{GPUMaterial}
    primitive_refs::Vector{GPUPrimitiveReference} # List referenced by leaf BVH nodes
end

# Camera data structure for the GPU
struct GPUCameraData
    position::Vec3
    lower_left_corner::Vec3
    horizontal::Vec3
    vertical::Vec3
    u::Vec3
    v::Vec3
    w::Vec3
    lens_radius::Float32      # Explicitly Float32
end

# Placeholder function to convert CPU scene representation to GPU-friendly format
function serialize_scene_for_gpu(scene_bvh_root::BVHNode, all_hittables::Vector{Hittable})
    # This function will be responsible for:
    # 1. Iterating through `all_hittables` to identify unique materials, spheres, and triangles
    #    (including those within Meshes, which need to be "flattened" into the global lists).
    # 2. Populating `GPUMaterial`, `GPUSphere`, `GPUTriangle` vectors.
    # 3. Creating `GPUPrimitiveReference` vector that maps generic primitive references to
    #    their specific locations in the typed GPU primitive vectors.
    # 4. Traversing the `scene_bvh_root` (CPU BVHNode tree) and building the
    #    `Vector{GPUBVHNode}` (flattened BVH). Leaf nodes in `GPUBVHNode` will use
    #    `data1` and `data2` to point to a range of indices in the `primitive_refs` vector.
    #    Interior `GPUBVHNode`s will use `data1` and `data2` to point to their child nodes
    #    within the `nodes` vector itself.

    println("serialize_scene_for_gpu: NOT YET IMPLEMENTED")
    # Return dummy/empty data for now
    return GPUSceneData(
        Vector{GPUBVHNode}(),
        Vector{GPUSphere}(),
        Vector{GPUTriangle}(),
        Vector{GPUMaterial}(),
        Vector{GPUPrimitiveReference}()
    )
end

# Function to convert CPU camera to GPU-friendly format
function serialize_camera_for_gpu(camera::Camera)::GPUCameraData
    return GPUCameraData(
        camera.position,
        camera.lower_left_corner,
        camera.horizontal,
        camera.vertical,
        camera.u,
        camera.v,
        camera.w,
        Float32(camera.lens_radius) # Ensure lens_radius is Float32
    )
end

# Build a BVH tree from a list of hittable objects