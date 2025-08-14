# Geometry definitions and ray intersection routines

using LinearAlgebra
using StaticArrays
using GeometryBasics
using MeshIO
using FileIO

# -----------------------------------------------------------------------------
# Geometry Primitives
# -----------------------------------------------------------------------------

# Sphere primitive
struct Sphere <: Hittable
    center::Point3f
    radius::Float32
    material_id::Int32
    
    # Constructor with validation
    function Sphere(center::Point3f, radius::Float32, material_id::Int32)
        @assert radius > 0 "Sphere radius must be positive"
        new(center, radius, material_id)
    end
end

# Triangle primitive
struct Triangle <: Hittable
    vertices::Vector{Point3f}  # Three vertices
    normal::Norm3f             # Precomputed face normal
    material_id::Int32
    bbox::AABB                 # Pre-computed bounding box
    
    # Constructor that calculates normal and bounding box
    function Triangle(vertices::Vector{Point3f}, material_id::Int32)
        @assert length(vertices) == 3 "Triangle must have exactly 3 vertices"
        
        # Calculate face normal
        edge1 = vertices[2] - vertices[1]
        edge2 = vertices[3] - vertices[1]
        normal = normalize(cross(edge1, edge2))
        
        # Compute bounding box
        min_point = Point3f(
            min(vertices[1][1], min(vertices[2][1], vertices[3][1])),
            min(vertices[1][2], min(vertices[2][2], vertices[3][2])),
            min(vertices[1][3], min(vertices[2][3], vertices[3][3]))
        )
        
        max_point = Point3f(
            max(vertices[1][1], max(vertices[2][1], vertices[3][1])),
            max(vertices[1][2], max(vertices[2][2], vertices[3][2])),
            max(vertices[1][3], max(vertices[2][3], vertices[3][3]))
        )
        
        # Add small epsilon to avoid zero-width boxes
        epsilon = Float32(1e-8)
        for i in 1:3
            if min_point[i] == max_point[i]
                min_point = setindex(min_point, min_point[i] - epsilon, i)
                max_point = setindex(max_point, max_point[i] + epsilon, i)
            end
        end
        
        bbox = AABB(min_point, max_point)
        
        new(vertices, normal, material_id, bbox)
    end
end

# Mesh structure (collection of triangles)
struct Mesh <: Hittable
    triangles::Vector{Triangle}
    bbox::AABB
    material_id::Int32
    
    # Constructor that computes the bounding box
    function Mesh(triangles::Vector{Triangle}, material_id::Int32)
        if isempty(triangles)
            error("Cannot create mesh with zero triangles")
        end
        
        # Compute bounding box for the whole mesh
        bbox = triangles[1].bbox
        for i in 2:length(triangles)
            bbox = surrounding_box(bbox, triangles[i].bbox)
        end
        
        new(triangles, bbox, material_id)
    end
end

# Simple BVH node structure
struct BVHNode <: Hittable
    bbox::AABB
    left::Union{Hittable, Nothing}
    right::Union{Hittable, Nothing}
    
    # Leaf node constructor
    function BVHNode(object::Hittable)
        new(bounding_box(object), object, nothing)
    end
    
    # Interior node constructor
    function BVHNode(left::Hittable, right::Hittable)
        bbox = surrounding_box(bounding_box(left), bounding_box(right))
        new(bbox, left, right)
    end
end

# Collection of hittable objects
struct HittableList <: Hittable
    objects::Vector{Hittable}
end

# -----------------------------------------------------------------------------
# Bounding Box Calculation
# -----------------------------------------------------------------------------

# Generic fallback for bounding_box
function bounding_box(object::Hittable)
    error("bounding_box not implemented for $(typeof(object))")
end

# Bounding box for a triangle
bounding_box(triangle::Triangle) = triangle.bbox

# Bounding box for a sphere
function bounding_box(sphere::Sphere)
    min_point = Point3f(
        sphere.center[1] - sphere.radius,
        sphere.center[2] - sphere.radius,
        sphere.center[3] - sphere.radius
    )
    
    max_point = Point3f(
        sphere.center[1] + sphere.radius,
        sphere.center[2] + sphere.radius,
        sphere.center[3] + sphere.radius
    )
    
    return AABB(min_point, max_point)
end

# Bounding box for a BVH node
bounding_box(node::BVHNode) = node.bbox

# Bounding box for a mesh
bounding_box(mesh::Mesh) = mesh.bbox

# Bounding box for a list of objects
function bounding_box(list::HittableList)
    if isempty(list.objects)
        # Return a default bounding box if the list is empty
        return AABB(Point3f(0, 0, 0), Point3f(0, 0, 0))
    end
    
    # Calculate the bounding box from all objects
    bbox = bounding_box(list.objects[1])
    for i in 2:length(list.objects)
        bbox = surrounding_box(bbox, bounding_box(list.objects[i]))
    end
    
    return bbox
end

# -----------------------------------------------------------------------------
# Ray Intersection Methods
# -----------------------------------------------------------------------------

# Ray-Sphere intersection
function hit(sphere::Sphere, ray::Ray, t_min::Float32, t_max::Float32)::HitRecord
    oc = ray.origin - sphere.center
    a = dot(ray.direction, ray.direction)
    b = Float32(2.0) * dot(oc, ray.direction)
    c = dot(oc, oc) - sphere.radius * sphere.radius
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0
        return HitRecord()
    end
    
    # Calculate the two intersection points
    sqrtd = sqrt(discriminant)
    root1 = (-b - sqrtd) / (Float32(2.0) * a)
    root2 = (-b + sqrtd) / (Float32(2.0) * a)
    
    # Check if either intersection point is within the valid range
    if root1 < t_min || root1 > t_max
        root1 = root2
        if root1 < t_min || root1 > t_max
            return HitRecord()
        end
    end
    
    t = root1
    point = point_at(ray, t)
    normal = normalize(point - sphere.center)
    
    # Calculate UV coordinates for the sphere
    phi = atan(normal[3], normal[1])
    theta = asin(clamp(normal[2], Float32(-1.0), Float32(1.0)))
    u = Float32(1.0) - (phi + π) / (Float32(2.0) * π)
    v = (theta + π / Float32(2.0)) / π
    uv = SVector{2, Float32}(u, v)
    
    # Outgoing direction is the negative of the ray direction
    wo = -ray.direction
    
    return HitRecord(t, point, normal, sphere.material_id, uv, wo, true)
end

# Ray-Triangle intersection (Möller–Trumbore algorithm)
function hit(triangle::Triangle, ray::Ray, t_min::Float32, t_max::Float32)::HitRecord
    # First check bounding box for early rejection
    if !hit_aabb(triangle.bbox, ray, t_min, t_max)
        return HitRecord()
    end
    
    v0, v1, v2 = triangle.vertices
    
    # Compute edges
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # Calculate determinant
    h = cross(ray.direction, edge2)
    a = dot(edge1, h)
    
    # Check if ray is parallel to triangle
    if abs(a) < Float32(1e-8)
        return HitRecord()
    end
    
    f = Float32(1.0) / a
    s = ray.origin - v0
    u = f * dot(s, h)
    
    if u < Float32(0.0) || u > Float32(1.0)
        return HitRecord()
    end
    
    q = cross(s, edge1)
    v = f * dot(ray.direction, q)
    
    if v < Float32(0.0) || u + v > Float32(1.0)
        return HitRecord()
    end
    
    # Compute intersection point parameter
    t = f * dot(edge2, q)
    
    if t < t_min || t > t_max
        return HitRecord()
    end
    
    point = point_at(ray, t)
    uv = SVector{2, Float32}(u, v)
    wo = -ray.direction
    
    return HitRecord(t, point, triangle.normal, triangle.material_id, uv, wo, true)
end

# Ray-Mesh intersection - tests all triangles
function hit(mesh::Mesh, ray::Ray, t_min::Float32, t_max::Float32)::HitRecord
    # First check the mesh's bounding box
    if !hit_aabb(mesh.bbox, ray, t_min, t_max)
        return HitRecord()
    end
    
    closest_so_far = t_max
    hit_anything = false
    result = HitRecord()
    
    for triangle in mesh.triangles
        temp_rec = hit(triangle, ray, t_min, closest_so_far)
        if temp_rec.hit_flag == 1
            hit_anything = true
            closest_so_far = temp_rec.t
            result = temp_rec
        end
    end
    
    # If we hit anything, ensure we use the mesh's material ID, not the triangles'
    if hit_anything
        result = HitRecord(
            result.t, 
            result.position, 
            result.normal, 
            mesh.material_id,
            result.uv,
            result.wo,
            true
        )
    end
    
    return result
end

# Ray-BVHNode intersection
function hit(node::BVHNode, ray::Ray, t_min::Float32, t_max::Float32)::HitRecord
    # First check if ray hits the bounding box
    if !hit_aabb(node.bbox, ray, t_min, t_max)
        return HitRecord()
    end
    
    # This is a leaf node with a single object
    if node.right === nothing
        return hit(node.left, ray, t_min, t_max)
    end
    
    # Check both children
    hit_left = hit(node.left, ray, t_min, t_max)
    hit_right = hit(node.right, ray, t_min, t_max)
    
    # Return the closest hit
    if hit_left.hit_flag == 1 && hit_right.hit_flag == 1
        if hit_left.t < hit_right.t
            return hit_left
        else
            return hit_right
        end
    elseif hit_left.hit_flag == 1
        return hit_left
    elseif hit_right.hit_flag == 1
        return hit_right
    else
        return HitRecord()
    end
end

# Ray-HittableList intersection
function hit(list::HittableList, ray::Ray, t_min::Float32, t_max::Float32)::HitRecord
    closest_so_far = t_max
    hit_anything = false
    result = HitRecord()
    
    for object in list.objects
        temp_rec = hit(object, ray, t_min, closest_so_far)
        
        if temp_rec.hit_flag == 1
            hit_anything = true
            closest_so_far = temp_rec.t
            result = temp_rec
        end
    end
    
    return result
end

# -----------------------------------------------------------------------------
# BVH Construction
# -----------------------------------------------------------------------------

"""
    build_bvh(objects::Vector{Hittable}, start::Int, finish::Int) -> BVHNode

Build a Bounding Volume Hierarchy from a list of objects.
`start` and `finish` define a range of indices (inclusive/exclusive) in the objects array.
"""
function build_bvh(objects::Vector{Hittable}, start::Int, finish::Int)
    # Choose a random axis to sort on
    axis = rand(1:3)
    
    # Sort objects based on the chosen axis
    if axis == 1
        sort_function = (a, b) -> bounding_box(a).min[1] < bounding_box(b).min[1]
    elseif axis == 2
        sort_function = (a, b) -> bounding_box(a).min[2] < bounding_box(b).min[2]
    else
        sort_function = (a, b) -> bounding_box(a).min[3] < bounding_box(b).min[3]
    end
    
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
        # In Julia, the end index in a range is inclusive, so we need start:finish-1
        sorted_objects = sort(objects[start:finish-1], lt=sort_function)
        objects[start:finish-1] = sorted_objects
        
        mid = start + floor(Int, object_span / 2)
        left = build_bvh(objects, start, mid)
        right = build_bvh(objects, mid, finish)
        
        bbox = surrounding_box(bounding_box(left), bounding_box(right))
        return BVHNode(left, right, bbox)
    end
end

# -----------------------------------------------------------------------------
# OBJ File Loading
# -----------------------------------------------------------------------------

"""
    load_mesh_from_obj(filename::String, material_id::Int32=Int32(0), 
                      scale::Point3f=Point3f(1.0, 1.0, 1.0),
                      rotation::Point3f=Point3f(0.0, 0.0, 0.0),
                      translation::Point3f=Point3f(0.0, 0.0, 0.0),
                      center::Bool=true,
                      normalize_size::Bool=false)

Load an OBJ file and create a Mesh object. Returns the Mesh.
"""
function load_mesh_from_obj(filename::String, material_id::Int32=Int32(0);
                          scale::Point3f=Point3f(Float32(1.0), Float32(1.0), Float32(1.0)),
                          rotation::Point3f=Point3f(Float32(0.0), Float32(0.0), Float32(0.0)),
                          translation::Point3f=Point3f(Float32(0.0), Float32(0.0), Float32(0.0)),
                          center::Bool=true,
                          normalize_size::Bool=false)
    
    println("Loading OBJ mesh from: $filename")
    
    # Use GeometryBasics and MeshIO to load the mesh
    try
        mesh_data = load(filename)
        
        # Extract vertices
        vertices = Vector{Point3f}()
        for point in coordinates(mesh_data)
            push!(vertices, Point3f(Float32(point[1]), Float32(point[2]), Float32(point[3])))
        end
        
        # Center the mesh if requested
        if center
            # Calculate centroid
            centroid = Point3f(Float32(0.0), Float32(0.0), Float32(0.0))
            for v in vertices
                centroid += v
            end
            centroid /= length(vertices)
            
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
                dist = norm(v)
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
        
        # Apply rotation (using simple Euler angles)
        if any(x -> x != Float32(0.0), rotation)
            for i in 1:length(vertices)
                v = vertices[i]
                
                # X-axis rotation
                if rotation[1] != Float32(0.0)
                    theta = deg2rad(rotation[1])
                    y = v[2] * cos(theta) - v[3] * sin(theta)
                    z = v[2] * sin(theta) + v[3] * cos(theta)
                    v = Point3f(v[1], y, z)
                end
                
                # Y-axis rotation
                if rotation[2] != Float32(0.0)
                    theta = deg2rad(rotation[2])
                    x = v[1] * cos(theta) + v[3] * sin(theta)
                    z = -v[1] * sin(theta) + v[3] * cos(theta)
                    v = Point3f(x, v[2], z)
                end
                
                # Z-axis rotation
                if rotation[3] != Float32(0.0)
                    theta = deg2rad(rotation[3])
                    x = v[1] * cos(theta) - v[2] * sin(theta)
                    y = v[1] * sin(theta) + v[2] * cos(theta)
                    v = Point3f(x, y, v[3])
                end
                
                vertices[i] = v
            end
        end
        
        # Apply scaling
        if any(x -> x != Float32(1.0), scale)
            for i in 1:length(vertices)
                vertices[i] = Point3f(
                    vertices[i][1] * scale[1],
                    vertices[i][2] * scale[2],
                    vertices[i][3] * scale[3]
                )
            end
        end
        
        # Apply translation
        if any(x -> x != Float32(0.0), translation)
            for i in 1:length(vertices)
                vertices[i] = vertices[i] + translation
            end
        end
        
        # Create triangles from mesh faces
        triangles = Triangle[]
        face_count = 0
        
        for face in GeometryBasics.faces(mesh_data)
            # Skip faces with more than 3 vertices
            if length(face) > 3
                # Simple triangulation - create a fan of triangles
                for i in 3:length(face)
                    if face[1] <= length(vertices) && face[i-1] <= length(vertices) && face[i] <= length(vertices)
                        v1 = vertices[face[1]]
                        v2 = vertices[face[i-1]]
                        v3 = vertices[face[i]]
                        push!(triangles, Triangle([v1, v2, v3], material_id))
                        face_count += 1
                    end
                end
            else
                # Standard triangle
                if all(idx -> idx <= length(vertices), face)
                    v1 = vertices[face[1]]
                    v2 = vertices[face[2]]
                    v3 = vertices[face[3]]
                    push!(triangles, Triangle([v1, v2, v3], material_id))
                    face_count += 1
                end
            end
        end
        
        println("Loaded OBJ mesh with $(length(vertices)) vertices and $face_count triangular faces")
        
        # Create a mesh object
        return Mesh(triangles, material_id)
        
    catch e
        println("Error loading mesh from OBJ: $e")
        rethrow(e)
    end
end

# --- Plane Intersection ---

function bounding_box(plane::Plane) # Satisfy Hittable interface
    # Infinite plane doesn't have a meaningful AABB for BVH subdivision.
    # Return a very large box, or handle plane intersection separately outside BVH.
    # For now, returning a large box. This might need refinement based on usage.
    max_val = Float32(1.0e10)
    return AABB(Point3f(-max_val, -max_val, -max_val), Point3f(max_val, max_val, max_val))
end

function hit(plane::Plane, ray::Ray, t_min::Float32, t_max::Float32)::HitRecord
    denom = dot(ray.direction, plane.normal)

    # Check if ray is parallel to the plane
    parallel = abs(denom) < Float32(1e-6)

    t = Float32(Inf)
    if !parallel
        t = -(dot(ray.origin, plane.normal) + plane.d) / denom
    end

    if t <= t_min || t >= t_max || parallel
        return HitRecord() # Return a miss
    end

    # Valid hit
    hit_position = point_at(ray, t)
    hit_normal = plane.normal # Plane normal is constant
    wo = -ray.direction

    # Checkerboard pattern calculation
    checker_val = floor(hit_position[1] * plane.checker_scale) + floor(hit_position[3] * plane.checker_scale)
    checker_idx = mod(abs(round(Int32, checker_val)), Int32(2))
    
    material_id = (checker_idx == 0) ? plane.material_id_a : plane.material_id_b

    # UV Calculation (Planar XZ)
    uv_scale = Float32(1.0) # Matches Python
    u_coord = mod(hit_position[1] * uv_scale, Float32(1.0))
    v_coord = mod(hit_position[2] * uv_scale, Float32(1.0)) # Python used position[2] (Z) for v. Correcting. Julia Y is up, XZ is ground.
                                                  # Python XZ plane means u=pos.x, v=pos.z.
                                                  # If Julia uses Y-up, XZ is ground, so u=pos.x, v=pos.z is correct.
                                                  # The original translation to Julia likely kept python's Z-up implicit indexing.
                                                  # Python: position[0] (X), position[2] (Z)
                                                  # Julia: hit_position[1] (X), hit_position[3] (Z) for XZ plane.
    v_coord = mod(hit_position[3] * uv_scale, Float32(1.0)) # Corrected: use Z for V on XZ plane.

    uv = UV2f(u_coord, v_coord)

    return HitRecord(t, hit_position, hit_normal, material_id, uv, wo, true)
end