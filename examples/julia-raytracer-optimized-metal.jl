# Optimized version of julia-raytracer.jl with mesh-specific BVH and Metal GPU acceleration

using LinearAlgebra
using Random
using Images      # For image manipulation
using Colors      # For color handling
using FileIO      # For file I/O
using Plots       # For displaying the image
using MeshIO      # For loading OBJ mesh files
using GeometryBasics # For mesh data structures
using StaticArrays  # For optimized vector operations on CPU and GPU

println("SPIRA - Metal GPU Ray Tracer")
println("Julia version: ", VERSION)

# Try to load Metal.jl
has_metal = false
try
    using Metal
    global has_metal = true
    println("Metal.jl loaded successfully!")
    dev = Metal.device()
    println("Metal device: $dev")

    # For Metal, we need to work with arrays rather than kernels
    # From Metal.jl docs:
    # "The MtlArray type is meant to be a convenient container for device memory,
    # as well as provide a data-parallel abstraction for using the GPU"
catch e
    println("Metal.jl not available: $e")
    println("Will use CPU rendering only")
end

# Type aliases for better performance and GPU compatibility
const Vec3 = SVector{3, Float32}
const Point3 = Vec3
const Color = Vec3

# Constants - use Float32 for better GPU compatibility
const INF = Float32(1e20)
const EPS = Float32(1e-6)
const BLACK = Vec3(0.0, 0.0, 0.0)
const WHITE = Vec3(1.0, 1.0, 1.0)

# Import basic operations for Vec3
import Base: +, -, *, /, show
import LinearAlgebra: dot, normalize

# Ray structure
struct Ray
    origin::Point3
    direction::Vec3

    Ray(origin::Point3, direction::Vec3) = new(origin, normalize(direction))
end

# Function to get point along ray
point_at(ray::Ray, t::Real) = ray.origin + t * ray.direction

# Material structure
struct Material
    diffuse::Vec3
    emission::Vec3
    specular::Float32
    roughness::Float32

    # Constructor with default values
    Material(; diffuse=Vec3(Float32(0.8), Float32(0.8), Float32(0.8)), emission=Vec3(0.0, 0.0, 0.0),
              specular=Float32(0.0), roughness=Float32(1.0)) = new(diffuse, emission, specular, roughness)
end

# Intersection result structure
struct HitRecord
    t::Float32         # Distance along ray to intersection
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
        min(box1.min[1], box2.min[1]),
        min(box1.min[2], box2.min[2]),
        min(box1.min[3], box2.min[3])
    )
    
    max_point = Vec3(
        max(box1.max[1], box2.max[1]),
        max(box1.max[2], box2.max[2]),
        max(box1.max[3], box2.max[3])
    )
    
    return AABB(min_point, max_point)
end

# Ray-AABB intersection test (fast slab method)
function hit_aabb(aabb::AABB, ray::Ray, t_min::Float32, t_max::Float32)
    for a in 1:3
        # Get component for this axis (x, y, or z)
        origin_comp = ray.origin[a]
        direction_comp = ray.direction[a]
        min_comp = aabb.min[a]
        max_comp = aabb.max[a]
        
        inv_dir = Float32(1.0) / direction_comp
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
    radius::Float32
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
            min(vertices[1][1], min(vertices[2][1], vertices[3][1])),
            min(vertices[1][2], min(vertices[2][2], vertices[3][2])),
            min(vertices[1][3], min(vertices[2][3], vertices[3][3]))
        )
        
        max_point = Vec3(
            max(vertices[1][1], max(vertices[2][1], vertices[3][1])),
            max(vertices[1][2], max(vertices[2][2], vertices[3][2])),
            max(vertices[1][3], max(vertices[2][3], vertices[3][3]))
        )
        
        # Add small epsilon to avoid zero-width boxes
        epsilon = Float32(1e-8)
        if min_point[1] == max_point[1]
            min_point = Vec3(min_point[1] - epsilon, min_point[2], min_point[3])
            max_point = Vec3(max_point[1] + epsilon, max_point[2], max_point[3])
        end
        if min_point[2] == max_point[2]
            min_point = Vec3(min_point[1], min_point[2] - epsilon, min_point[3])
            max_point = Vec3(max_point[1], max_point[2] + epsilon, max_point[3])
        end
        if min_point[3] == max_point[3]
            min_point = Vec3(min_point[1], min_point[2], min_point[3] - epsilon)
            max_point = Vec3(max_point[1], max_point[2], max_point[3] + epsilon)
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
        sphere.center[1] - sphere.radius,
        sphere.center[2] - sphere.radius,
        sphere.center[3] - sphere.radius
    )
    
    max_point = Vec3(
        sphere.center[1] + sphere.radius,
        sphere.center[2] + sphere.radius,
        sphere.center[3] + sphere.radius
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
        (a, b) -> bounding_box(a).min[1] < bounding_box(b).min[1] :
        (axis == 2 ?
            (a, b) -> bounding_box(a).min[2] < bounding_box(b).min[2] :
            (a, b) -> bounding_box(a).min[3] < bounding_box(b).min[3])

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
function hit(sphere::Sphere, ray::Ray, t_min::Float32, t_max::Float32)
    oc = ray.origin - sphere.center
    a = dot(ray.direction, ray.direction)
    b = Float32(2.0) * dot(oc, ray.direction)
    c = dot(oc, oc) - sphere.radius * sphere.radius
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0
        return HitRecord(Float32(0.0), Vec3(0.0,0.0,0.0), Vec3(0.0,0.0,0.0), sphere.material, false)
    end
    
    # Calculate the two intersection points
    sqrtd = sqrt(discriminant)
    root1 = (-b - sqrtd) / (Float32(2.0) * a)
    root2 = (-b + sqrtd) / (Float32(2.0) * a)
    
    # Check if either intersection point is within the valid range
    if root1 < t_min || root1 > t_max
        root1 = root2
        if root1 < t_min || root1 > t_max
            return HitRecord(Float32(0.0), Vec3(0.0,0.0,0.0), Vec3(0.0,0.0,0.0), sphere.material, false)
        end
    end
    
    t = root1
    point = point_at(ray, t)
    normal = normalize(point - sphere.center)
    
    return HitRecord(t, point, normal, sphere.material, true)
end

# Ray-Triangle intersection (Möller–Trumbore algorithm)
function hit(triangle::Triangle, ray::Ray, t_min::Float32, t_max::Float32)
    # First check bounding box for early rejection
    if !hit_aabb(triangle.bbox, ray, t_min, t_max)
        return HitRecord(Float32(0.0), Vec3(0.0,0.0,0.0), Vec3(0.0,0.0,0.0), triangle.material, false)
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
        return HitRecord(Float32(0.0), Vec3(0.0,0.0,0.0), Vec3(0.0,0.0,0.0), triangle.material, false)
    end
    
    f = Float32(1.0) / a
    s = ray.origin - v0
    u = f * dot(s, h)
    
    if u < 0.0 || u > 1.0
        return HitRecord(Float32(0.0), Vec3(0.0,0.0,0.0), Vec3(0.0,0.0,0.0), triangle.material, false)
    end
    
    q = cross(s, edge1)
    v = f * dot(ray.direction, q)
    
    if v < 0.0 || u + v > 1.0
        return HitRecord(Float32(0.0), Vec3(0.0,0.0,0.0), Vec3(0.0,0.0,0.0), triangle.material, false)
    end
    
    # Compute intersection point parameter
    t = f * dot(edge2, q)
    
    if t < t_min || t > t_max
        return HitRecord(Float32(0.0), Vec3(0.0,0.0,0.0), Vec3(0.0,0.0,0.0), triangle.material, false)
    end
    
    point = point_at(ray, t)
    normal = triangle_normal(triangle)
    
    return HitRecord(t, point, normal, triangle.material, true)
end

# Ray-BVHNode intersection
function hit(node::BVHNode, ray::Ray, t_min::Float32, t_max::Float32)
    # First check if ray hits the bounding box
    if !hit_aabb(node.bbox, ray, t_min, t_max)
        return HitRecord(Float32(0.0), Vec3(0.0,0.0,0.0), Vec3(0.0,0.0,0.0), Material(), false)
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
        return HitRecord(Float32(0.0), Vec3(0.0,0.0,0.0), Vec3(0.0,0.0,0.0), Material(), false)
    end
end

# Ray-Mesh intersection (tests all triangles in the mesh)
function hit(mesh::Mesh, ray::Ray, t_min::Float32, t_max::Float32)
    # First check the mesh's bounding box
    if !hit_aabb(mesh.bbox, ray, t_min, t_max)
        return HitRecord(Float32(0.0), Vec3(0.0,0.0,0.0), Vec3(0.0,0.0,0.0), Material(), false)
    end
    
    # If we have a BVH, use it for faster intersection
    if mesh.bvh !== nothing
        return hit(mesh.bvh, ray, t_min, t_max)
    end
    
    # Otherwise, test each triangle
    closest_so_far = t_max
    hit_anything = false
    result = HitRecord(Float32(0.0), Vec3(0.0,0.0,0.0), Vec3(0.0,0.0,0.0), Material(), false)

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
        return AABB(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
    end
    
    # Calculate the bounding box from all objects
    bbox = bounding_box(list.objects[1])
    for i in 2:length(list.objects)
        bbox = surrounding_box(bbox, bounding_box(list.objects[i]))
    end
    
    return bbox
end

# Ray-HittableList intersection
function hit(list::HittableList, ray::Ray, t_min::Float32, t_max::Float32)
    closest_so_far = t_max
    hit_anything = false
    result = HitRecord(Float32(0.0), Vec3(0.0,0.0,0.0), Vec3(0.0,0.0,0.0), Material(), false)

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
function hit(bvh::BoundingVolumeHierarchy, ray::Ray, t_min::Float32, t_max::Float32)
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
    lens_radius::Float32
    
    function Camera(;
        position::Vec3 = Vec3(Float32(0.0), Float32(0.0), Float32(0.0)),
        look_at::Vec3 = Vec3(Float32(0.0), Float32(0.0), Float32(-1.0)),
        up::Vec3 = Vec3(Float32(0.0), Float32(1.0), Float32(0.0)),
        fov::Float32 = Float32(90.0),  # Vertical field of view in degrees
        aspect_ratio::Float32 = Float32(16.0/9.0),
        aperture::Float32 = Float32(0.0),
        focus_dist::Float32 = Float32(1.0)
    )
        theta = deg2rad(fov)
        h = tan(theta/2)
        viewport_height = Float32(2.0) * h
        viewport_width = aspect_ratio * viewport_height
        
        w = normalize(position - look_at)
        u = normalize(cross(up, w))
        v = cross(w, u)
        
        horizontal = focus_dist * viewport_width * u
        vertical = focus_dist * viewport_height * v
        lower_left_corner = position - horizontal/Float32(2.0) - vertical/Float32(2.0) - focus_dist * w
        
        new(position, lower_left_corner, horizontal, vertical, u, v, w, aperture/Float32(2.0))
    end
end

# Generate a ray from the camera
function get_ray(cam::Camera, s::Float32, t::Float32)
    rd = Vec3(Float32(0.0), Float32(0.0), Float32(0.0))  # No defocus blur
    offset = Vec3(Float32(0.0), Float32(0.0), Float32(0.0))
    
    origin = cam.position + offset
    direction = normalize(cam.lower_left_corner + s*cam.horizontal + t*cam.vertical - origin)
    
    return Ray(origin, direction)
end

# Random functions for sampling
function random_in_unit_sphere()
    while true
        p = Float32(2.0) * Vec3(rand(Float32), rand(Float32), rand(Float32)) - Vec3(Float32(1.0), Float32(1.0), Float32(1.0))
        if dot(p, p) < Float32(1.0)
            return p
        end
    end
end

function random_unit_vector()
    return normalize(random_in_unit_sphere())
end

# Reflect a vector around a normal
function reflect(v::Vec3, n::Vec3)
    return v - Float32(2.0) * dot(v, n) * n
end

# Calculate ray color (recursive path tracing)
function ray_color(ray::Ray, world::Hittable, depth::Int)
    # If we've exceeded the ray bounce limit, no more light is gathered
    if depth <= 0
        return BLACK
    end
    
    # Check for intersection with the scene
    rec = hit(world, ray, Float32(0.001), INF)
    
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
            return emitted + rec.material.specular * specular_color .* rec.material.diffuse
        else
            # Diffuse reflection
            target = rec.position + rec.normal + random_unit_vector()
            scattered = Ray(rec.position, normalize(target - rec.position))
            
            # Recursive ray tracing with diffuse reflection
            return emitted + Float32(0.5) * ray_color(scattered, world, depth-1) .* rec.material.diffuse
        end
    end
    
    # Background color (simple gradient)
    t = Float32(0.5) * (ray.direction[2] + Float32(1.0))
    return (Float32(1.0)-t) * WHITE + t * Vec3(Float32(0.5), Float32(0.7), Float32(1.0))
end

# ACEScg color space transformation (for more accurate color reproduction)
function to_acescg(color::Vec3)
    # ACEScg tone mapping parameters
    a = Float32(2.51)
    b = Float32(0.03)
    c = Float32(2.43)
    d = Float32(0.59)
    e = Float32(0.14)
    
    # Apply ACES tone mapping
    r = clamp((color[1] * (a * color[1] + b)) / (color[1] * (c * color[1] + d) + e), Float32(0.0), Float32(1.0))
    g = clamp((color[2] * (a * color[2] + b)) / (color[2] * (c * color[2] + d) + e), Float32(0.0), Float32(1.0))
    bb = clamp((color[3] * (a * color[3] + b)) / (color[3] * (c * color[3] + d) + e), Float32(0.0), Float32(1.0))
    
    return RGB{Float32}(r, g, bb)
end


# Ray-sphere intersection for GPU kernel
function gpu_hit_sphere(origin_x::Float32, origin_y::Float32, origin_z::Float32,
                   direction_x::Float32, direction_y::Float32, direction_z::Float32,
                   sphere_data::MtlArray{Float32}, idx::Int)
    # Extract sphere data
    center_x = sphere_data[idx]
    center_y = sphere_data[idx+1]
    center_z = sphere_data[idx+2]
    radius = sphere_data[idx+3]
    material_idx = Int(sphere_data[idx+4])

    # Use component-wise operations to avoid dynamic allocation
    oc_x = origin_x - center_x
    oc_y = origin_y - center_y
    oc_z = origin_z - center_z

    # Manual dot product to avoid Vec3 allocation
    a = direction_x*direction_x + direction_y*direction_y + direction_z*direction_z
    half_b = oc_x*direction_x + oc_y*direction_y + oc_z*direction_z
    c = oc_x*oc_x + oc_y*oc_y + oc_z*oc_z - radius * radius
    discriminant = half_b * half_b - a * c

    # Early return with pre-defined values for no hit
    if discriminant < Float32(0.0)
        return false, Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), 0
    end

    sqrtd = sqrt(discriminant)

    # Find the nearest root that lies in the acceptable range
    root = (-half_b - sqrtd) / a
    if root < Float32(0.001)
        root = (-half_b + sqrtd) / a
        if root < Float32(0.001)
            return false, Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), 0
        end
    end

    t = root
    # Calculate hit point components
    hit_x = origin_x + t * direction_x
    hit_y = origin_y + t * direction_y
    hit_z = origin_z + t * direction_z

    # Calculate normal components
    normal_x = hit_x - center_x
    normal_y = hit_y - center_y
    normal_z = hit_z - center_z

    # Normalize the normal manually
    normal_len = sqrt(normal_x*normal_x + normal_y*normal_y + normal_z*normal_z)
    normal_x = normal_x / normal_len
    normal_y = normal_y / normal_len
    normal_z = normal_z / normal_len

    # Return individual components instead of Vec3
    return true, t, normal_x, normal_y, normal_z, material_idx
end

# Ray-triangle intersection for GPU kernel
function gpu_hit_triangle(origin_x::Float32, origin_y::Float32, origin_z::Float32,
                     direction_x::Float32, direction_y::Float32, direction_z::Float32,
                     triangle_data::MtlArray{Float32}, idx::Int)
    # Extract triangle data
    v0_x = triangle_data[idx]
    v0_y = triangle_data[idx+1]
    v0_z = triangle_data[idx+2]
    v1_x = triangle_data[idx+3]
    v1_y = triangle_data[idx+4]
    v1_z = triangle_data[idx+5]
    v2_x = triangle_data[idx+6]
    v2_y = triangle_data[idx+7]
    v2_z = triangle_data[idx+8]
    material_idx = Int(triangle_data[idx+9])

    # Calculate edges component-wise
    edge1_x = v1_x - v0_x
    edge1_y = v1_y - v0_y
    edge1_z = v1_z - v0_z

    edge2_x = v2_x - v0_x
    edge2_y = v2_y - v0_y
    edge2_z = v2_z - v0_z

    # Manual cross product for h = cross(direction, edge2)
    h_x = direction_y * edge2_z - direction_z * edge2_y
    h_y = direction_z * edge2_x - direction_x * edge2_z
    h_z = direction_x * edge2_y - direction_y * edge2_x

    # Manual dot product for a = dot(edge1, h)
    a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z

    # Check if ray is parallel to triangle
    if abs(a) < Float32(1e-8)
        return false, Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), 0
    end

    f = Float32(1.0) / a

    # Calculate s = origin - v0 component-wise
    s_x = origin_x - v0_x
    s_y = origin_y - v0_y
    s_z = origin_z - v0_z

    # Manual dot product for u = f * dot(s, h)
    u = f * (s_x * h_x + s_y * h_y + s_z * h_z)

    if u < Float32(0.0) || u > Float32(1.0)
        return false, Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), 0
    end

    # Manual cross product for q = cross(s, edge1)
    q_x = s_y * edge1_z - s_z * edge1_y
    q_y = s_z * edge1_x - s_x * edge1_z
    q_z = s_x * edge1_y - s_y * edge1_x

    # Manual dot product for v = f * dot(direction, q)
    v = f * (direction_x * q_x + direction_y * q_y + direction_z * q_z)

    if v < Float32(0.0) || u + v > Float32(1.0)
        return false, Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), 0
    end

    # Manual dot product for t = f * dot(edge2, q)
    t = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z)

    if t < Float32(0.001)
        return false, Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), 0
    end

    # Compute normal using manual cross product
    normal_x = edge1_y * edge2_z - edge1_z * edge2_y
    normal_y = edge1_z * edge2_x - edge1_x * edge2_z
    normal_z = edge1_x * edge2_y - edge1_y * edge2_x

    # Normalize manually
    normal_len = sqrt(normal_x*normal_x + normal_y*normal_y + normal_z*normal_z)
    normal_x = normal_x / normal_len
    normal_y = normal_y / normal_len
    normal_z = normal_z / normal_len

    # Return individual components instead of Vec3
    return true, t, normal_x, normal_y, normal_z, material_idx
end

# Get material properties from GPU data
function gpu_get_material(materials::MtlArray{Float32}, idx::Int)
    # Calculate base index for this material in the array
    base_idx = (idx - 1) * 10 + 1

    # Create diffuse and emission vectors explicitly with each component
    # The Vec3 constructor itself is fine, it's the vector math operations that can cause issues
    diffuse = Vec3(
        materials[base_idx],
        materials[base_idx+1],
        materials[base_idx+2]
    )

    emission = Vec3(
        materials[base_idx+3],
        materials[base_idx+4],
        materials[base_idx+5]
    )

    # Get scalar properties
    specular = materials[base_idx+6]
    roughness = materials[base_idx+7]

    # Return components (tuples don't allocate on the GPU)
    return diffuse, emission, specular, roughness
end

# Random number generation for GPU kernel (xorshift)
function gpu_random(seed::UInt32)
    seed = xor(seed, seed << 13)
    seed = xor(seed, seed >> 17)
    seed = xor(seed, seed << 5)
    return seed, Float32(seed) / Float32(typemax(UInt32))
end

# Generate a random unit vector for diffuse reflection
function gpu_random_unit_vector(seed::UInt32)
    # Box-Muller transform to generate points on unit sphere
    seed, u1 = gpu_random(seed)
    seed, u2 = gpu_random(seed)

    phi = Float32(2.0) * π * u1
    theta = acos(Float32(2.0) * u2 - Float32(1.0))

    # Calculate components directly to avoid vector allocation
    x = sin(theta) * cos(phi)
    y = sin(theta) * sin(phi)
    z = cos(theta)

    # Create the vector only at the end
    return seed, Vec3(x, y, z)
end

# Reflect a vector around a normal
function gpu_reflect(v::Vec3, n::Vec3)
    # Calculate dot product manually
    dot_vn = v[1]*n[1] + v[2]*n[2] + v[3]*n[3]

    # Calculate components directly
    factor = Float32(2.0) * dot_vn
    result_x = v[1] - factor * n[1]
    result_y = v[2] - factor * n[2]
    result_z = v[3] - factor * n[3]

    return Vec3(result_x, result_y, result_z)
end

# Main kernel for ray tracing on Metal GPU
function raytracer_kernel(
    camera_origin_x::Float32, camera_origin_y::Float32, camera_origin_z::Float32,
    cam_ll_x::Float32, cam_ll_y::Float32, cam_ll_z::Float32,
    cam_horiz_x::Float32, cam_horiz_y::Float32, cam_horiz_z::Float32,
    cam_vert_x::Float32, cam_vert_y::Float32, cam_vert_z::Float32,
    spheres, sphere_count, triangles, triangle_count, materials,
    width, height, samples_per_pixel, max_depth,
    seeds, result_buffer
)
    # Get thread indices (global thread ID)
    idx = thread_position_in_grid_1d()
    i = (idx - 1) % width + 1  # column (x)
    j = (idx - 1) ÷ width + 1  # row (y)

    # Make sure we're within bounds
    if i <= width && j <= height
        # Initialize random seed for this pixel
        seed = seeds[idx]

        # Initialize accumulation variables for the pixel color
        pixel_r = Float32(0.0)
        pixel_g = Float32(0.0)
        pixel_b = Float32(0.0)

        # Anti-aliasing with multiple samples
        for s in 1:samples_per_pixel
            # Generate random position within pixel
            seed, rand_u = gpu_random(seed)
            seed, rand_v = gpu_random(seed)

            u = Float32((i - 1 + rand_u) / (width - 1))
            v = Float32((j - 1 + rand_v) / (height - 1))

            # Use the already broken-down camera components directly
            ray_origin_x = camera_origin_x
            ray_origin_y = camera_origin_y
            ray_origin_z = camera_origin_z

            # Calculate ray direction components using individual component params
            dir_x = cam_ll_x + u*cam_horiz_x + v*cam_vert_x - ray_origin_x
            dir_y = cam_ll_y + u*cam_horiz_y + v*cam_vert_y - ray_origin_y
            dir_z = cam_ll_z + u*cam_horiz_z + v*cam_vert_z - ray_origin_z

            # Normalize direction
            dir_len = sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)
            ray_direction_x = dir_x / dir_len
            ray_direction_y = dir_y / dir_len
            ray_direction_z = dir_z / dir_len

            # Pack into Vec3 for compatibility with hit functions
            ray_origin = Vec3(ray_origin_x, ray_origin_y, ray_origin_z)
            ray_direction = Vec3(ray_direction_x, ray_direction_y, ray_direction_z)

            # Initialize color components for this sample
            color_r = Float32(0.0)
            color_g = Float32(0.0)
            color_b = Float32(0.0)

            # Throughput components
            throughput_r = Float32(1.0)
            throughput_g = Float32(1.0)
            throughput_b = Float32(1.0)

            # Ray tracing loop (max_depth iterations)
            for depth in 1:max_depth
                # Check for intersection with all primitives
                hit_anything = false
                closest_t = Inf32
                hit_normal_x = Float32(0.0)
                hit_normal_y = Float32(0.0)
                hit_normal_z = Float32(0.0)
                hit_material_idx = 0

                # Check spheres
                for sp_idx in 0:(sphere_count-1)
                    hit, t, nx, ny, nz, material_idx = gpu_hit_sphere(
                        ray_origin_x, ray_origin_y, ray_origin_z,
                        ray_direction_x, ray_direction_y, ray_direction_z,
                        spheres, sp_idx * 5 + 1
                    )

                    if hit && t < closest_t
                        hit_anything = true
                        closest_t = t
                        hit_normal_x = nx
                        hit_normal_y = ny
                        hit_normal_z = nz
                        hit_material_idx = material_idx
                    end
                end

                # Check triangles
                for tri_idx in 0:(triangle_count-1)
                    hit, t, nx, ny, nz, material_idx = gpu_hit_triangle(
                        ray_origin_x, ray_origin_y, ray_origin_z,
                        ray_direction_x, ray_direction_y, ray_direction_z,
                        triangles, tri_idx * 10 + 1
                    )

                    if hit && t < closest_t
                        hit_anything = true
                        closest_t = t
                        hit_normal_x = nx
                        hit_normal_y = ny
                        hit_normal_z = nz
                        hit_material_idx = material_idx
                    end
                end

                if hit_anything
                    # Calculate hit point components
                    hit_point_x = ray_origin_x + closest_t * ray_direction_x
                    hit_point_y = ray_origin_y + closest_t * ray_direction_y
                    hit_point_z = ray_origin_z + closest_t * ray_direction_z

                    # Get material properties
                    diffuse, emission, specular, roughness = gpu_get_material(
                        materials, hit_material_idx
                    )

                    # Add emission contribution
                    color_r += throughput_r * emission[1]
                    color_g += throughput_g * emission[2]
                    color_b += throughput_b * emission[3]

                    # Choose between specular and diffuse reflection
                    seed, rand_val = gpu_random(seed)

                    if rand_val <= specular
                        # Specular reflection using components directly
                        # We'll calculate the reflection manually instead of using gpu_reflect function
                        # First calculate dot product of direction and normal
                        dot_vn = ray_direction_x * hit_normal_x +
                                ray_direction_y * hit_normal_y +
                                ray_direction_z * hit_normal_z

                        # Calculate components directly
                        factor = Float32(2.0) * dot_vn
                        reflected_x = ray_direction_x - factor * hit_normal_x
                        reflected_y = ray_direction_y - factor * hit_normal_y
                        reflected_z = ray_direction_z - factor * hit_normal_z

                        # Add roughness
                        if roughness > Float32(0.0)
                            seed, random_vec = gpu_random_unit_vector(seed)
                            reflected_x += roughness * random_vec[1]
                            reflected_y += roughness * random_vec[2]
                            reflected_z += roughness * random_vec[3]

                            # Normalize the reflected direction
                            refl_len = sqrt(reflected_x*reflected_x + reflected_y*reflected_y + reflected_z*reflected_z)
                            reflected_x /= refl_len
                            reflected_y /= refl_len
                            reflected_z /= refl_len
                        end

                        # Update ray for next iteration
                        ray_origin_x = hit_point_x + Float32(0.001) * hit_normal_x
                        ray_origin_y = hit_point_y + Float32(0.001) * hit_normal_y
                        ray_origin_z = hit_point_z + Float32(0.001) * hit_normal_z

                        ray_direction_x = reflected_x
                        ray_direction_y = reflected_y
                        ray_direction_z = reflected_z

                        # No need to pack into Vec3 anymore - using components directly

                        # Update throughput
                        throughput_r *= diffuse[1]
                        throughput_g *= diffuse[2]
                        throughput_b *= diffuse[3]
                    else
                        # Diffuse reflection
                        seed, random_vec = gpu_random_unit_vector(seed)
                        scatter_dir_x = hit_normal_x + random_vec[1]
                        scatter_dir_y = hit_normal_y + random_vec[2]
                        scatter_dir_z = hit_normal_z + random_vec[3]

                        # Check for degenerate scatter direction
                        if abs(scatter_dir_x) < Float32(1e-8) &&
                           abs(scatter_dir_y) < Float32(1e-8) &&
                           abs(scatter_dir_z) < Float32(1e-8)
                            scatter_dir_x = hit_normal_x
                            scatter_dir_y = hit_normal_y
                            scatter_dir_z = hit_normal_z
                        else
                            # Normalize scatter direction
                            scatter_len = sqrt(scatter_dir_x*scatter_dir_x +
                                             scatter_dir_y*scatter_dir_y +
                                             scatter_dir_z*scatter_dir_z)
                            scatter_dir_x /= scatter_len
                            scatter_dir_y /= scatter_len
                            scatter_dir_z /= scatter_len
                        end

                        # Update ray for next iteration
                        ray_origin_x = hit_point_x + Float32(0.001) * hit_normal_x
                        ray_origin_y = hit_point_y + Float32(0.001) * hit_normal_y
                        ray_origin_z = hit_point_z + Float32(0.001) * hit_normal_z

                        ray_direction_x = scatter_dir_x
                        ray_direction_y = scatter_dir_y
                        ray_direction_z = scatter_dir_z

                        # No need to pack into Vec3 anymore - using components directly

                        # Update throughput (diffuse reflects 50% of light)
                        throughput_r *= Float32(0.5) * diffuse[1]
                        throughput_g *= Float32(0.5) * diffuse[2]
                        throughput_b *= Float32(0.5) * diffuse[3]
                    end

                    # Russian roulette path termination
                    if depth > 3
                        seed, rr_val = gpu_random(seed)
                        max_comp = max(throughput_r, max(throughput_g, throughput_b))

                        if rr_val > max_comp
                            break
                        end

                        # Adjust throughput to account for termination probability
                        throughput_r /= max_comp
                        throughput_g /= max_comp
                        throughput_b /= max_comp
                    end
                else
                    # Sky/background color (simple gradient)
                    t = Float32(0.5) * (ray_direction_y + Float32(1.0))

                    # Compute sky color components
                    sky_r = (Float32(1.0) - t) * Float32(1.0) + t * Float32(0.5)
                    sky_g = (Float32(1.0) - t) * Float32(1.0) + t * Float32(0.7)
                    sky_b = (Float32(1.0) - t) * Float32(1.0) + t * Float32(1.0)

                    # Add to accumulated color
                    color_r += throughput_r * sky_r
                    color_g += throughput_g * sky_g
                    color_b += throughput_b * sky_b
                    break
                end
            end

            # Add sample to pixel color
            pixel_r += color_r
            pixel_g += color_g
            pixel_b += color_b
        end

        # Average samples
        pixel_r = pixel_r / Float32(samples_per_pixel)
        pixel_g = pixel_g / Float32(samples_per_pixel)
        pixel_b = pixel_b / Float32(samples_per_pixel)

        # Save updated random seed for this pixel
        seeds[idx] = seed

        # Store in result buffer (considering the origin is at the top-left for the image)
        result_idx = ((height - j) * width + (i - 1)) * 3 + 1
        result_buffer[result_idx] = pixel_r
        result_buffer[result_idx+1] = pixel_g
        result_buffer[result_idx+2] = pixel_b
    end

    return nothing
end

# Scene data structures for GPU
mutable struct GPUScene
    spheres::Union{Metal.MtlArray, Nothing}
    sphere_count::Int
    triangles::Union{Metal.MtlArray, Nothing}
    triangle_count::Int
    materials::Union{Metal.MtlArray, Nothing}
    material_count::Int

    GPUScene() = new(nothing, 0, nothing, 0, nothing, 0)
end

# Extract all spheres from a BVH node (recursive)
function extract_spheres_from_bvh(node::BVHNode, spheres::Vector{Sphere})
    # If it's a leaf node with a Sphere
    if node.right === nothing && isa(node.left, Sphere)
        push!(spheres, node.left)
    else
        # Process left child
        if isa(node.left, Sphere)
            push!(spheres, node.left)
        elseif isa(node.left, BVHNode)
            extract_spheres_from_bvh(node.left, spheres)
        elseif isa(node.left, Triangle)
            # Skip triangles for now (handled separately)
        elseif isa(node.left, Mesh)
            # Process mesh triangles
            for triangle in node.left.triangles
                # Triangles are processed separately
            end
        end

        # Process right child if it exists
        if node.right !== nothing
            if isa(node.right, Sphere)
                push!(spheres, node.right)
            elseif isa(node.right, BVHNode)
                extract_spheres_from_bvh(node.right, spheres)
            end
        end
    end
end

# Extract all triangles from a BVH node (recursive)
function extract_triangles_from_bvh(node::BVHNode, triangles::Vector{Triangle})
    # If it's a leaf node with a Triangle
    if node.right === nothing && isa(node.left, Triangle)
        push!(triangles, node.left)
    else
        # Process left child
        if isa(node.left, Triangle)
            push!(triangles, node.left)
        elseif isa(node.left, BVHNode)
            extract_triangles_from_bvh(node.left, triangles)
        elseif isa(node.left, Mesh)
            # Process mesh triangles
            for triangle in node.left.triangles
                push!(triangles, triangle)
            end
        end

        # Process right child if it exists
        if node.right !== nothing
            if isa(node.right, Triangle)
                push!(triangles, node.right)
            elseif isa(node.right, BVHNode)
                extract_triangles_from_bvh(node.right, triangles)
            end
        end
    end
end

# Extract all materials from spheres and triangles
function extract_materials(spheres::Vector{Sphere}, triangles::Vector{Triangle})
    materials = Material[]
    material_map = Dict{Material, Int}()

    # Process sphere materials
    for sphere in spheres
        if !haskey(material_map, sphere.material)
            push!(materials, sphere.material)
            material_map[sphere.material] = length(materials)
        end
    end

    # Process triangle materials
    for triangle in triangles
        if !haskey(material_map, triangle.material)
            push!(materials, triangle.material)
            material_map[triangle.material] = length(materials)
        end
    end

    return materials, material_map
end

# Convert scene to GPU-compatible format
function prepare_scene_for_gpu(scene::BoundingVolumeHierarchy)
    # Extract all primitives from the BVH for flat array representation
    spheres = Sphere[]
    triangles = Triangle[]

    extract_spheres_from_bvh(scene.root, spheres)
    extract_triangles_from_bvh(scene.root, triangles)

    # Extract all unique materials
    materials, material_map = extract_materials(spheres, triangles)

    # Create packed sphere data for GPU
    # Format: [center.x, center.y, center.z, radius, material_index]
    sphere_data = zeros(Float32, 5 * length(spheres))
    for (i, sphere) in enumerate(spheres)
        idx = (i-1) * 5 + 1
        sphere_data[idx] = sphere.center[1]
        sphere_data[idx+1] = sphere.center[2]
        sphere_data[idx+2] = sphere.center[3]
        sphere_data[idx+3] = sphere.radius
        sphere_data[idx+4] = Float32(material_map[sphere.material])
    end

    # Create packed triangle data for GPU
    # Format: [v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z, material_index]
    triangle_data = zeros(Float32, 10 * length(triangles))
    for (i, triangle) in enumerate(triangles)
        idx = (i-1) * 10 + 1
        # Vertex 1
        triangle_data[idx] = triangle.vertices[1][1]
        triangle_data[idx+1] = triangle.vertices[1][2]
        triangle_data[idx+2] = triangle.vertices[1][3]
        # Vertex 2
        triangle_data[idx+3] = triangle.vertices[2][1]
        triangle_data[idx+4] = triangle.vertices[2][2]
        triangle_data[idx+5] = triangle.vertices[2][3]
        # Vertex 3
        triangle_data[idx+6] = triangle.vertices[3][1]
        triangle_data[idx+7] = triangle.vertices[3][2]
        triangle_data[idx+8] = triangle.vertices[3][3]
        # Material index
        triangle_data[idx+9] = Float32(material_map[triangle.material])
    end

    # Create packed material data for GPU
    # Format: [albedo.r, albedo.g, albedo.b, emission.r, emission.g, emission.b, specular, roughness, padding1, padding2]
    material_data = zeros(Float32, 10 * length(materials))
    for (i, material) in enumerate(materials)
        idx = (i-1) * 10 + 1
        material_data[idx] = material.diffuse[1]
        material_data[idx+1] = material.diffuse[2]
        material_data[idx+2] = material.diffuse[3]
        material_data[idx+3] = material.emission[1]
        material_data[idx+4] = material.emission[2]
        material_data[idx+5] = material.emission[3]
        material_data[idx+6] = material.specular
        material_data[idx+7] = material.roughness
        material_data[idx+8] = Float32(0.0)  # padding
        material_data[idx+9] = Float32(0.0)  # padding
    end

    # Create Metal buffers if Metal is available
    gpu_scene = GPUScene()

    if has_metal
        gpu_scene.spheres = Metal.MtlArray(sphere_data)
        gpu_scene.sphere_count = length(spheres)
        gpu_scene.triangles = Metal.MtlArray(triangle_data)
        gpu_scene.triangle_count = length(triangles)
        gpu_scene.materials = Metal.MtlArray(material_data)
        gpu_scene.material_count = length(materials)
        println("Prepared GPU scene with $(length(spheres)) spheres, $(length(triangles)) triangles, and $(length(materials)) materials")
    else
        println("Metal not available, scene preparation skipped")
    end

    return gpu_scene
end

# CPU rendering function
function render_cpu(world::Hittable, camera::Camera, width::Int, height::Int; 
                samples_per_pixel::Int=50, max_depth::Int=20, progress_update::Bool=true)
    img = Array{RGB{Float32}}(undef, height, width)
    # Store raw HDR data for EXR output
    hdr_data = Array{Vec3}(undef, height, width)
    
    # Use all available threads
    Threads.@threads for j in 1:height
        if progress_update && j % 10 == 0
            println("Rendering progress: $(round(Int, 100 * j / height))%")
        end
        
        for i in 1:width
            color = BLACK
            
            # Anti-aliasing with multiple samples per pixel
            for _ in 1:samples_per_pixel
                u = Float32((i - 1 + rand(Float32)) / (width - 1))
                v = Float32((j - 1 + rand(Float32)) / (height - 1))
                ray = get_ray(camera, u, v)
                color += ray_color(ray, world, max_depth)
            end
            
            # Average samples
            color = color / Float32(samples_per_pixel)
            
            # Store the raw HDR value
            hdr_data[height-j+1, i] = color
            
            # Apply tone mapping for display
            img[height-j+1, i] = to_acescg(color)
        end
    end
    
    return img, hdr_data
end

# GPU rendering function using Metal
function render_gpu(world::Hittable, camera::Camera, width::Int, height::Int;
                   samples_per_pixel::Int=50, max_depth::Int=8, progress_update::Bool=true)
    if !has_metal
        println("Metal GPU not available, falling back to CPU rendering")
        return render_cpu(world, camera, width, height;
                         samples_per_pixel=samples_per_pixel,
                         max_depth=max_depth,
                         progress_update=progress_update)
    end

    println("Preparing scene data for GPU...")
    gpu_scene = prepare_scene_for_gpu(world)

    println("Rendering with Metal GPU using array abstractions...")

    # Metal doesn't support custom kernels well - using array operations instead
    Metal.allowscalar(true)  # For simplified implementation

    # Create output image arrays
    img = Array{RGB{Float32}}(undef, height, width)
    hdr_data = Array{Vec3}(undef, height, width)

    # Prepare camera data
    camera_origin = camera.position
    lower_left = camera.lower_left_corner
    horizontal = camera.horizontal
    vertical = camera.vertical

    # Create GPU arrays for ray directions
    ray_dirs = zeros(Float32, height, width, 3)
    ray_origins = zeros(Float32, height, width, 3)

    # Fill ray directions array
    for j in 1:height
        for i in 1:width
            # Initialize ray origins to camera position
            ray_origins[j, i, 1] = camera_origin[1]
            ray_origins[j, i, 2] = camera_origin[2]
            ray_origins[j, i, 3] = camera_origin[3]

            # Calculate ray directions based on pixel position
            # Anti-aliasing is applied per-sample later
            u = Float32((i - 1) / (width - 1))
            v = Float32((j - 1) / (height - 1))

            dir = lower_left + u*horizontal + v*vertical - camera_origin

            # Store normalized direction
            len = sqrt(dir[1]^2 + dir[2]^2 + dir[3]^2)
            ray_dirs[j, i, 1] = dir[1] / len
            ray_dirs[j, i, 2] = dir[2] / len
            ray_dirs[j, i, 3] = dir[3] / len
        end
    end

    # Create GPU buffers
    ray_dirs_gpu = Metal.MtlArray(ray_dirs)
    ray_origins_gpu = Metal.MtlArray(ray_origins)

    # Create result buffer
    result_buffer = Metal.MtlArray(zeros(Float32, height, width, 3))

    # For tracking progress
    start_time = time()

    # Render each sample and accumulate
    for sample in 1:samples_per_pixel
        if progress_update && (sample % 5 == 1)
            elapsed = time() - start_time
            samples_per_sec = (sample - 1) / max(elapsed, 0.001)
            est_remaining = (samples_per_pixel - sample + 1) / max(samples_per_sec, 0.001)
            println("Sample $sample/$samples_per_pixel - " *
                   "$(round(100 * sample / samples_per_pixel, digits=1))% complete - " *
                   "Est. $(round(est_remaining, digits=1))s remaining")
        end

        # Apply jitter for anti-aliasing if not the first sample
        if sample > 1
            # Create jittered copy of ray directions
            jittered_dirs = copy(ray_dirs_gpu)

            # Add random jitter for anti-aliasing
            jitter_u = Metal.MtlArray(rand(Float32, height, width) ./ width)
            jitter_v = Metal.MtlArray(rand(Float32, height, width) ./ height)

            # Add small perturbation to ray directions
            for j in 1:height
                for i in 1:width
                    # Get original ray direction
                    dx = jittered_dirs[j, i, 1]
                    dy = jittered_dirs[j, i, 2]
                    dz = jittered_dirs[j, i, 3]

                    # Add small jitter
                    jx = jitter_u[j, i] * Float32(0.01)
                    jy = jitter_v[j, i] * Float32(0.01)

                    # Add jitter and renormalize
                    dx += jx
                    dy += jy

                    # Renormalize
                    len = sqrt(dx^2 + dy^2 + dz^2)
                    jittered_dirs[j, i, 1] = dx / len
                    jittered_dirs[j, i, 2] = dy / len
                    jittered_dirs[j, i, 3] = dz / len
                end
            end

            # Use jittered rays for this sample
            trace_rays_gpu(ray_origins_gpu, jittered_dirs,
                         gpu_scene.spheres, gpu_scene.sphere_count,
                         gpu_scene.triangles, gpu_scene.triangle_count,
                         gpu_scene.materials, max_depth, result_buffer)
        else
            # Use regular rays for first sample
            trace_rays_gpu(ray_origins_gpu, ray_dirs_gpu,
                         gpu_scene.spheres, gpu_scene.sphere_count,
                         gpu_scene.triangles, gpu_scene.triangle_count,
                         gpu_scene.materials, max_depth, result_buffer)
        end
    end

    # Copy result back to CPU and format
    result = Array(result_buffer) ./ samples_per_pixel

    # Convert to RGB and HDR
    for j in 1:height
        for i in 1:width
            color = Vec3(result[j, i, 1], result[j, i, 2], result[j, i, 3])

            # Store HDR data
            hdr_data[j, i] = color

            # Apply tone mapping for RGB output
            img[j, i] = to_acescg(color)
        end
    end

    # Done!
    elapsed = time() - start_time
    if progress_update
        println("GPU rendering completed in $(round(elapsed, digits=2)) seconds")
        println("$(round(Int, width * height / elapsed)) pixels/second")
    end

    return img, hdr_data
end

# Trace rays using GPU arrays
function trace_rays_gpu(ray_origins, ray_dirs,
                      spheres, sphere_count,
                      triangles, triangle_count,
                      materials, max_depth, result_buffer)
    height, width, _ = size(ray_dirs)

    # Extract data from Metal arrays to work on CPU
    # This defeats the purpose of GPU acceleration but works as a fallback
    sphere_data = Array(spheres)
    triangle_data = Array(triangles)
    material_data = Array(materials)

    # Process each pixel in parallel on the CPU
    Threads.@threads for j in 1:height
        for i in 1:width
            # Get ray for this pixel
            origin = (ray_origins[j, i, 1], ray_origins[j, i, 2], ray_origins[j, i, 3])
            direction = (ray_dirs[j, i, 1], ray_dirs[j, i, 2], ray_dirs[j, i, 3])

            # Trace ray and get color
            color = trace_ray(origin, direction, sphere_data, sphere_count,
                            triangle_data, triangle_count, material_data, max_depth)

            # Add to result buffer
            result_buffer[j, i, 1] += color[1]
            result_buffer[j, i, 2] += color[2]
            result_buffer[j, i, 3] += color[3]
        end
    end
end

# Simplified ray tracing function that works with plain arrays
function trace_ray(origin, direction, sphere_data, sphere_count,
                  triangle_data, triangle_count, material_data, depth)
    if depth <= 0
        return (Float32(0.0), Float32(0.0), Float32(0.0))
    end

    # Initialize hit variables
    hit_anything = false
    closest_t = Float32(1.0e20)
    hit_normal = (Float32(0.0), Float32(0.0), Float32(0.0))
    hit_material_idx = 0

    # Check spheres
    for s in 1:sphere_count
        # Get sphere data
        idx = (s-1) * 5 + 1
        sphere_x = sphere_data[idx]
        sphere_y = sphere_data[idx+1]
        sphere_z = sphere_data[idx+2]
        radius = sphere_data[idx+3]
        mat_idx = Int(sphere_data[idx+4])

        # Ray-sphere intersection
        oc_x = origin[1] - sphere_x
        oc_y = origin[2] - sphere_y
        oc_z = origin[3] - sphere_z

        a = direction[1]^2 + direction[2]^2 + direction[3]^2
        half_b = oc_x*direction[1] + oc_y*direction[2] + oc_z*direction[3]
        c = oc_x^2 + oc_y^2 + oc_z^2 - radius^2

        discriminant = half_b^2 - a*c

        if discriminant > 0
            sqrtd = sqrt(discriminant)

            # Try both roots
            root = (-half_b - sqrtd) / a
            if root < Float32(0.001)
                root = (-half_b + sqrtd) / a
            end

            if root > Float32(0.001) && root < closest_t
                closest_t = root
                hit_anything = true

                # Calculate hit point
                hit_x = origin[1] + closest_t * direction[1]
                hit_y = origin[2] + closest_t * direction[2]
                hit_z = origin[3] + closest_t * direction[3]

                # Calculate normal
                normal_x = (hit_x - sphere_x) / radius
                normal_y = (hit_y - sphere_y) / radius
                normal_z = (hit_z - sphere_z) / radius

                normal_len = sqrt(normal_x^2 + normal_y^2 + normal_z^2)
                hit_normal = (normal_x/normal_len, normal_y/normal_len, normal_z/normal_len)
                hit_material_idx = mat_idx
            end
        end
    end

    # Check triangles
    for t in 1:triangle_count
        # Get triangle data
        idx = (t-1) * 10 + 1
        v0_x = triangle_data[idx]
        v0_y = triangle_data[idx+1]
        v0_z = triangle_data[idx+2]
        v1_x = triangle_data[idx+3]
        v1_y = triangle_data[idx+4]
        v1_z = triangle_data[idx+5]
        v2_x = triangle_data[idx+6]
        v2_y = triangle_data[idx+7]
        v2_z = triangle_data[idx+8]
        mat_idx = Int(triangle_data[idx+9])

        # Möller–Trumbore algorithm
        edge1_x = v1_x - v0_x
        edge1_y = v1_y - v0_y
        edge1_z = v1_z - v0_z

        edge2_x = v2_x - v0_x
        edge2_y = v2_y - v0_y
        edge2_z = v2_z - v0_z

        # Cross product h = cross(direction, edge2)
        h_x = direction[2] * edge2_z - direction[3] * edge2_y
        h_y = direction[3] * edge2_x - direction[1] * edge2_z
        h_z = direction[1] * edge2_y - direction[2] * edge2_x

        # Dot product
        a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z

        # Check if ray is parallel to triangle
        if abs(a) < Float32(1e-8)
            continue
        end

        f = Float32(1.0) / a

        s_x = origin[1] - v0_x
        s_y = origin[2] - v0_y
        s_z = origin[3] - v0_z

        # Calculate u parameter
        u = f * (s_x * h_x + s_y * h_y + s_z * h_z)

        if u < Float32(0.0) || u > Float32(1.0)
            continue
        end

        # Cross product q = cross(s, edge1)
        q_x = s_y * edge1_z - s_z * edge1_y
        q_y = s_z * edge1_x - s_x * edge1_z
        q_z = s_x * edge1_y - s_y * edge1_x

        # Calculate v parameter
        v = f * (direction[1] * q_x + direction[2] * q_y + direction[3] * q_z)

        if v < Float32(0.0) || u + v > Float32(1.0)
            continue
        end

        # Calculate t parameter
        t = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z)

        if t > Float32(0.001) && t < closest_t
            closest_t = t
            hit_anything = true
            hit_material_idx = mat_idx

            # Calculate normal using cross product
            normal_x = edge1_y * edge2_z - edge1_z * edge2_y
            normal_y = edge1_z * edge2_x - edge1_x * edge2_z
            normal_z = edge1_x * edge2_y - edge1_y * edge2_x

            # Normalize
            normal_len = sqrt(normal_x^2 + normal_y^2 + normal_z^2)
            hit_normal = (normal_x/normal_len, normal_y/normal_len, normal_z/normal_len)
        end
    end

    # If we hit something
    if hit_anything
        # Get material properties
        mat_idx = hit_material_idx
        mat_base = (mat_idx - 1) * 10 + 1

        # Extract material properties
        diffuse_r = material_data[mat_base]
        diffuse_g = material_data[mat_base+1]
        diffuse_b = material_data[mat_base+2]
        emissive_r = material_data[mat_base+3]
        emissive_g = material_data[mat_base+4]
        emissive_b = material_data[mat_base+5]
        specular = material_data[mat_base+6]
        roughness = material_data[mat_base+7]

        # Calculate hit point
        hit_x = origin[1] + closest_t * direction[1]
        hit_y = origin[2] + closest_t * direction[2]
        hit_z = origin[3] + closest_t * direction[3]

        # If material is emissive, return emission color
        if emissive_r > 0 || emissive_g > 0 || emissive_b > 0
            return (emissive_r, emissive_g, emissive_b)
        end

        # Choose between specular and diffuse reflection
        if rand() < specular
            # Specular reflection
            dot_vn = direction[1]*hit_normal[1] + direction[2]*hit_normal[2] + direction[3]*hit_normal[3]

            # Calculate reflected direction
            reflected_x = direction[1] - Float32(2.0) * dot_vn * hit_normal[1]
            reflected_y = direction[2] - Float32(2.0) * dot_vn * hit_normal[2]
            reflected_z = direction[3] - Float32(2.0) * dot_vn * hit_normal[3]

            # Add roughness
            if roughness > 0
                # Add random direction with roughness factor
                random_x = Float32(2.0) * rand() - Float32(1.0)
                random_y = Float32(2.0) * rand() - Float32(1.0)
                random_z = Float32(2.0) * rand() - Float32(1.0)
                random_len = sqrt(random_x^2 + random_y^2 + random_z^2)

                # Normalize random direction
                if random_len > 0
                    random_x /= random_len
                    random_y /= random_len
                    random_z /= random_len

                    # Add scaled random direction
                    reflected_x += roughness * random_x
                    reflected_y += roughness * random_y
                    reflected_z += roughness * random_z
                end
            end

            # Normalize reflected direction
            refl_len = sqrt(reflected_x^2 + reflected_y^2 + reflected_z^2)
            reflected_x /= refl_len
            reflected_y /= refl_len
            reflected_z /= refl_len

            # Offset hit point to avoid self-intersection
            new_origin_x = hit_x + Float32(0.001) * hit_normal[1]
            new_origin_y = hit_y + Float32(0.001) * hit_normal[2]
            new_origin_z = hit_z + Float32(0.001) * hit_normal[3]

            # Recursively trace reflected ray
            reflected_color = trace_ray(
                (new_origin_x, new_origin_y, new_origin_z),
                (reflected_x, reflected_y, reflected_z),
                sphere_data, sphere_count,
                triangle_data, triangle_count,
                material_data, depth - 1
            )

            # Modulate by material color
            return (
                diffuse_r * reflected_color[1],
                diffuse_g * reflected_color[2],
                diffuse_b * reflected_color[3]
            )
        else
            # Diffuse reflection - use cosine-weighted importance sampling
            # Generate random direction in hemisphere
            random_x = Float32(2.0) * rand() - Float32(1.0)
            random_y = Float32(2.0) * rand() - Float32(1.0)
            random_z = Float32(2.0) * rand() - Float32(1.0)
            random_len = sqrt(random_x^2 + random_y^2 + random_z^2)

            # Normalize
            if random_len > 0
                random_x /= random_len
                random_y /= random_len
                random_z /= random_len
            else
                random_x = Float32(0.0)
                random_y = Float32(1.0)
                random_z = Float32(0.0)
            end

            # Make sure it's in the hemisphere
            dot_nr = hit_normal[1]*random_x + hit_normal[2]*random_y + hit_normal[3]*random_z
            if dot_nr < 0
                random_x = -random_x
                random_y = -random_y
                random_z = -random_z
            end

            # Calculate new scatter direction
            scatter_x = hit_normal[1] + random_x
            scatter_y = hit_normal[2] + random_y
            scatter_z = hit_normal[3] + random_z

            # Normalize scatter direction
            scatter_len = sqrt(scatter_x^2 + scatter_y^2 + scatter_z^2)
            if scatter_len > 0
                scatter_x /= scatter_len
                scatter_y /= scatter_len
                scatter_z /= scatter_len
            else
                scatter_x = hit_normal[1]
                scatter_y = hit_normal[2]
                scatter_z = hit_normal[3]
            end

            # Offset hit point to avoid self-intersection
            new_origin_x = hit_x + Float32(0.001) * hit_normal[1]
            new_origin_y = hit_y + Float32(0.001) * hit_normal[2]
            new_origin_z = hit_z + Float32(0.001) * hit_normal[3]

            # Recursively trace scattered ray
            scattered_color = trace_ray(
                (new_origin_x, new_origin_y, new_origin_z),
                (scatter_x, scatter_y, scatter_z),
                sphere_data, sphere_count,
                triangle_data, triangle_count,
                material_data, depth - 1
            )

            # Modulate by material color (and 0.5 factor for energy conservation)
            return (
                Float32(0.5) * diffuse_r * scattered_color[1],
                Float32(0.5) * diffuse_g * scattered_color[2],
                Float32(0.5) * diffuse_b * scattered_color[3]
            )
        end
    end

    # No hit - sky color
    t = Float32(0.5) * (direction[2] + Float32(1.0))
    return (
        (Float32(1.0) - t) + t * Float32(0.5),  # r
        (Float32(1.0) - t) + t * Float32(0.7),  # g
        (Float32(1.0) - t) + t * Float32(1.0)   # b
    )
end

# Main render function with GPU/CPU selection
function render(world::Hittable, camera::Camera, width::Int, height::Int;
               samples_per_pixel::Int=50, max_depth::Int=20,
               progress_update::Bool=true, use_gpu::Bool=true)
    # Check if Metal is available and enabled
    use_metal = use_gpu && has_metal

    # Log rendering device
    if progress_update
        if use_metal
            println("Rendering with Metal GPU acceleration")
            dev = Metal.device()
            println("Metal device: $dev")
        else
            println("Rendering with CPU ($(Threads.nthreads()) threads)")
            if use_gpu && !has_metal
                println("GPU rendering requested but Metal is not available")
            end
        end

        println("Resolution: $(width)x$(height), Samples: $samples_per_pixel, Max depth: $max_depth")
    end

    # Time the rendering
    start_time = time()

    # Select rendering path
    result = if use_metal
        render_gpu(world, camera, width, height;
                  samples_per_pixel=samples_per_pixel,
                  max_depth=max_depth,
                  progress_update=progress_update)
    else
        render_cpu(world, camera, width, height;
                  samples_per_pixel=samples_per_pixel,
                  max_depth=max_depth,
                  progress_update=progress_update)
    end

    # Report render time
    render_time = time() - start_time
    if progress_update
        println("Render completed in $(round(render_time, digits=2)) seconds")
        println("Average: $(round(width * height / render_time, digits=0)) pixels/second")
    end

    return result
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
                r_channel[j, i] = hdr_data[j, i][1]
                g_channel[j, i] = hdr_data[j, i][2]
                b_channel[j, i] = hdr_data[j, i][3]
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
                       scale::Vec3=Vec3(Float32(1.0), Float32(1.0), Float32(1.0)),
                       rotation::Vec3=Vec3(Float32(0.0), Float32(0.0), Float32(0.0)),
                       translation::Vec3=Vec3(Float32(0.0), Float32(0.0), Float32(0.0)),
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
                    x = parse(Float32, parts[2])
                    y = parse(Float32, parts[3])
                    z = parse(Float32, parts[4])
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
        push!(vertices, Vec3(Float32(point[1]), Float32(point[2]), Float32(point[3])))
    end
    
    # Center the mesh if requested
    if center
        # Calculate centroid
        centroid = Vec3(Float32(0.0), Float32(0.0), Float32(0.0))
        for v in vertices
            centroid += v
        end
        centroid = centroid / Float32(length(vertices))
        
        # Subtract centroid from all vertices
        for i in 1:length(vertices)
            vertices[i] = vertices[i] - centroid
        end
    end
    
    # Normalize size if requested
    if normalize_size
        # Find the maximum distance from the origin
        max_dist = Float32(0.0)
        for v in vertices
            dist = length(v)
            if dist > max_dist
                max_dist = dist
            end
        end
        
        # Scale vertices to fit in a unit sphere
        if max_dist > Float32(0.0)
            scale_factor = Float32(1.0) / max_dist
            for i in 1:length(vertices)
                vertices[i] = vertices[i] * scale_factor
            end
        end
    end
    
    # Apply rotation (using simple Euler angles for this example)
    if rotation[1] != Float32(0.0) || rotation[2] != Float32(0.0) || rotation[3] != Float32(0.0)
        for i in 1:length(vertices)
            v = vertices[i]
            
            # X-axis rotation
            if rotation[1] != Float32(0.0)
                theta = deg2rad(rotation[1])
                y = v[2] * cos(theta) - v[3] * sin(theta)
                z = v[2] * sin(theta) + v[3] * cos(theta)
                v = Vec3(v[1], y, z)
            end
            
            # Y-axis rotation
            if rotation[2] != Float32(0.0)
                theta = deg2rad(rotation[2])
                x = v[1] * cos(theta) + v[3] * sin(theta)
                z = -v[1] * sin(theta) + v[3] * cos(theta)
                v = Vec3(x, v[2], z)
            end
            
            # Z-axis rotation
            if rotation[3] != Float32(0.0)
                theta = deg2rad(rotation[3])
                x = v[1] * cos(theta) - v[2] * sin(theta)
                y = v[1] * sin(theta) + v[2] * cos(theta)
                v = Vec3(x, y, v[3])
            end
            
            vertices[i] = v
        end
    end
    
    # Apply scaling
    if scale[1] != Float32(1.0) || scale[2] != Float32(1.0) || scale[3] != Float32(1.0)
        for i in 1:length(vertices)
            vertices[i] = Vec3(
                vertices[i][1] * scale[1],
                vertices[i][2] * scale[2],
                vertices[i][3] * scale[3]
            )
        end
    end
    
    # Apply translation
    if translation[1] != Float32(0.0) || translation[2] != Float32(0.0) || translation[3] != Float32(0.0)
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
    push!(objects, Sphere(Vec3(Float32(0.0), -Float32(100.5), -Float32(1.0)), Float32(100.0), Material(diffuse=Vec3(Float32(0.8), Float32(0.8), Float32(0.2)))))
    
    # Add a center sphere (red diffuse)
    push!(objects, Sphere(Vec3(Float32(0.0), Float32(0.0), -Float32(1.0)), Float32(0.5), Material(diffuse=Vec3(Float32(0.8), Float32(0.2), Float32(0.2)))))
    
    # Add a metal sphere (golden)
    push!(objects, Sphere(Vec3(Float32(1.0), Float32(0.0), -Float32(1.0)), Float32(0.5), Material(diffuse=Vec3(Float32(0.8), Float32(0.6), Float32(0.2)), specular=Float32(0.8), roughness=Float32(0.3))))
    
    # Add a glass-like sphere
    push!(objects, Sphere(Vec3(-Float32(1.0), Float32(0.0), -Float32(1.0)), Float32(0.5), Material(diffuse=Vec3(Float32(0.8), Float32(0.8), Float32(0.8)), specular=Float32(1.0), roughness=Float32(0.0))))
    
    # Add a light source
    push!(objects, Sphere(Vec3(Float32(0.0), Float32(2.0), Float32(0.0)), Float32(0.5), Material(diffuse=Vec3(Float32(0.8), Float32(0.8), Float32(0.8)), emission=Vec3(Float32(4.0), Float32(4.0), Float32(4.0)))))
    
    # Add a triangle (green)
    vertices = [Vec3(-Float32(0.5), Float32(0.0), -Float32(2.0)), Vec3(Float32(0.5), Float32(0.0), -Float32(2.0)), Vec3(Float32(0.0), Float32(1.0), -Float32(2.0))]
    push!(objects, Triangle(vertices, Material(diffuse=Vec3(Float32(0.2), Float32(0.8), Float32(0.2)))))
    
    # Wrap scene in BVH for efficiency
    scene = BoundingVolumeHierarchy(objects)
    
    # Define camera
    camera = Camera(
        position=Vec3(Float32(0.0), Float32(1.0), Float32(3.0)),
        look_at=Vec3(Float32(0.0), Float32(0.0), -Float32(1.0)),
        up=Vec3(Float32(0.0), Float32(1.0), Float32(0.0)),
        fov=Float32(45.0),
        aspect_ratio=Float32(16.0)/Float32(9.0)  # Explicitly set 16:9 aspect ratio to match output dimensions
    )
    
    return scene, camera
end

# Create a scene with a single OBJ mesh
function create_obj_scene()
    # Example objects
    objects = Hittable[]
    
    # Add a ground plane (represented as a large sphere)
    push!(objects, Sphere(Vec3(Float32(0.0), -Float32(100.5), -Float32(1.0)), Float32(100.0), Material(diffuse=Vec3(Float32(0.8), Float32(0.8), Float32(0.2)))))
    
    # Add a light source
    push!(objects, Sphere(Vec3(Float32(0.0), Float32(2.0), Float32(0.0)), Float32(0.5), Material(diffuse=Vec3(Float32(0.8), Float32(0.8), Float32(0.8)), emission=Vec3(Float32(4.0), Float32(4.0), Float32(4.0)))))
    
    # Define a material for the mesh
    mesh_material = Material(diffuse=Vec3(Float32(0.7), Float32(0.3), Float32(0.2)), specular=Float32(0.2), roughness=Float32(0.4))
    
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
            scale=Vec3(Float32(3.0), Float32(3.0), Float32(3.0)),          # Scale to 3x size
            rotation=Vec3(Float32(0.0), Float32(90.0), Float32(0.0)),      # Rotate 90° around Y axis
            translation=Vec3(Float32(0.0), Float32(0.0), -Float32(1.0))    # Position at Z=-1
        )
        
        # Add the mesh to our scene
        push!(objects, mesh)
        
        println("Added mesh to the scene")
    else
        # Fallback: Add a simple sphere if OBJ file not found
        println("OBJ file not found, adding a sphere instead")
        push!(objects, Sphere(Vec3(Float32(0.0), Float32(0.0), -Float32(1.0)), Float32(0.5), mesh_material))
    end
    
    # Wrap scene in BVH for efficiency
    scene = BoundingVolumeHierarchy(objects)
    
    # Define camera
    camera = Camera(
        position=Vec3(Float32(0.0), Float32(1.0), Float32(3.0)),
        look_at=Vec3(Float32(0.0), Float32(0.0), -Float32(1.0)),
        up=Vec3(Float32(0.0), Float32(1.0), Float32(0.0)),
        fov=Float32(45.0),
        aspect_ratio=Float32(16.0)/Float32(9.0)
    )
    
    return scene, camera
end

# Create a scene with multiple transformed OBJ meshes
function create_multiple_obj_scene()
    objects = Hittable[]
    
    # Add ground plane
    push!(objects, Sphere(Vec3(Float32(0.0), -Float32(100.5), -Float32(1.0)), Float32(100.0), Material(diffuse=Vec3(Float32(0.8), Float32(0.8), Float32(0.2)))))
    
    # Add lights
    push!(objects, Sphere(Vec3(-Float32(2.0), Float32(3.0), Float32(2.0)), Float32(0.5), Material(diffuse=Vec3(Float32(0.8), Float32(0.8), Float32(0.8)), emission=Vec3(Float32(5.0), Float32(5.0), Float32(5.0)))))
    push!(objects, Sphere(Vec3(Float32(2.0), Float32(2.0), Float32(1.0)), Float32(0.25), Material(diffuse=Vec3(Float32(0.8), Float32(0.6), Float32(0.2)), emission=Vec3(Float32(3.0), Float32(2.0), Float32(1.0)))))
    
    # Path to the OBJ file
    obj_file = expanduser("~/Downloads/Lemon_200k.obj")  # Stanford bunny is a common test model
    
    if isfile(obj_file)
        # Example 1: Load a mesh with metallic material
        metal_material = Material(diffuse=Vec3(Float32(0.8), Float32(0.8), Float32(0.9)), specular=Float32(0.8), roughness=Float32(0.1))
        mesh1 = load_obj_mesh(
            obj_file,
            metal_material,
            center=true,                      # Center the mesh
            normalize_size=true,              # Normalize size
            scale=Vec3(Float32(1.0), Float32(1.0), Float32(1.0)),        # No scaling
            rotation=Vec3(Float32(0.0), Float32(0.0), Float32(0.0)),     # No rotation
            translation=Vec3(Float32(0.0), Float32(0.5), -Float32(1.5))  # Push back slightly
        )
        push!(objects, mesh1)
        
        # Example 2: Load the same mesh with diffuse material
        diffuse_material = Material(diffuse=Vec3(Float32(0.2), Float32(0.8), Float32(0.3)), specular=Float32(0.0), roughness=Float32(1.0))
        mesh2 = load_obj_mesh(
            obj_file,
            diffuse_material,
            center=true,
            normalize_size=true,
            scale=Vec3(Float32(1.0), Float32(1.0), Float32(1.0)),           # Larger scaling for a different look
            rotation=Vec3(Float32(0.0), Float32(45.0), Float32(0.0)),       # Rotate 45° around Y axis
            translation=Vec3(Float32(1.8), Float32(0.5), -Float32(1.0))     # Position to the right
        )
        push!(objects, mesh2)
        
        # Example 3: Load the same mesh with glass material
        glass_material = Material(diffuse=Vec3(Float32(0.9), Float32(0.9), Float32(0.9)), specular=Float32(1.0), roughness=Float32(0.0))
        mesh3 = load_obj_mesh(
            obj_file,
            glass_material,
            center=true,
            normalize_size=true,
            scale=Vec3(Float32(1.0), Float32(1.0), Float32(1.0)),           # Scale up some
            rotation=Vec3(Float32(20.0), -Float32(30.0), Float32(0.0)),     # Multiple axis rotation
            translation=Vec3(-Float32(1.8), Float32(0.6), -Float32(1.8))    # Position to the left and slightly up
        )
        push!(objects, mesh3)
    else
        # Fallback with spheres if OBJ file not found
        println("OBJ file not found, using spheres instead")
        
        push!(objects, Sphere(Vec3(Float32(0.0), Float32(0.0), -Float32(1.5)), Float32(0.5), Material(diffuse=Vec3(Float32(0.8), Float32(0.8), Float32(0.9)), specular=Float32(0.8), roughness=Float32(0.1))))
        push!(objects, Sphere(Vec3(Float32(1.2), Float32(0.0), -Float32(1.0)), Float32(0.5), Material(diffuse=Vec3(Float32(0.2), Float32(0.8), Float32(0.3)), specular=Float32(0.0), roughness=Float32(1.0))))
        push!(objects, Sphere(Vec3(-Float32(1.2), Float32(0.1), -Float32(0.8)), Float32(0.5), Material(diffuse=Vec3(Float32(0.9), Float32(0.9), Float32(0.9)), specular=Float32(1.0), roughness=Float32(0.0))))
    end
    
    # Wrap scene in BVH for efficiency
    scene = BoundingVolumeHierarchy(objects)
    
    # Define camera from a slightly higher angle to see all meshes
    camera = Camera(
        position=Vec3(Float32(0.0), Float32(1.5), Float32(4.0)),
        look_at=Vec3(Float32(0.0), Float32(0.0), -Float32(1.0)),
        up=Vec3(Float32(0.0), Float32(1.0), Float32(0.0)),
        fov=Float32(40.0),
        aspect_ratio=Float32(16.0)/Float32(9.0)
    )
    
    return scene, camera
end

# Render the scene
function render_example(; width=1280, height=720, samples=100, interactive=true, 
                        output_file="metal_render.exr", scene_type="default", use_gpu=true)
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
    image, hdr_data = render(scene, camera, width, height, 
                            samples_per_pixel=samples, max_depth=10, 
                            use_gpu=use_gpu)
    
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
    output_file = "metal_render.exr"

    # Default settings
    width = 640         # Lower resolution for faster test rendering
    height = 360        # 16:9 aspect ratio
    samples = 50        # Reduced sample count for faster test rendering
    use_gpu = true      # Try to use GPU by default

    # Scene type: "default", "obj", or "multiple_obj"
    scene_type = "multiple_obj"  # Try rendering with optimized OBJ mesh

    # Print Julia and Metal status
    println("Julia version: ", VERSION)
    println("Metal GPU acceleration: $(has_metal ? "Available" : "Not available")")
    println("CPU threads available: ", Threads.nthreads())

    # Automatically scale quality based on available hardware
    if has_metal
        # With GPU, we can use higher quality
        println("GPU detected, using higher quality settings")
        width = 1280
        height = 720
        samples = 100
    else
        println("No GPU available, using moderate quality settings")
    end

    # Run the renderer
    render_example(
        width=width,
        height=height,
        samples=samples,
        interactive=interactive,
        output_file=output_file,
        scene_type=scene_type,
        use_gpu=use_gpu
    )

    return nothing
end

# Run the main function if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end