using LinearAlgebra
using Random
using Images      # For image manipulation
using Colors      # For color handling
using FileIO      # For file I/O
using Plots       # For displaying the image
using MeshIO      # For loading OBJ mesh files
using GeometryBasics # For mesh data structures

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
function Base.:*(a::Vec3, b::Vec3)
    Vec3(a.x * b.x, a.y * b.y, a.z * b.z)
end

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

# Base hittable abstract type
abstract type Hittable end

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

    # Calculate normal during construction
    function Triangle(vertices::Vector{Vec3}, material::Material)
        @assert length(vertices) == 3 "Triangle must have exactly 3 vertices"
        new(vertices, material)
    end
end

# Mesh structure (collection of triangles with shared material)
struct Mesh <: Hittable
    triangles::Vector{Triangle}

    # Create a mesh from triangles with same material
    function Mesh(triangles::Vector{Triangle})
        new(triangles)
    end
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

# Collection of hittable objects
struct HittableList <: Hittable
    objects::Vector{Hittable}
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

# Ray-Mesh intersection (tests all triangles in the mesh)
function hit(mesh::Mesh, ray::Ray, t_min::Float64, t_max::Float64)
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

# Simple Bounding Volume Hierarchy structure
struct BoundingVolumeHierarchy <: Hittable
    objects::Vector{Hittable}
    
    function BoundingVolumeHierarchy(objects::Vector{Hittable})
        # In a full implementation, this would build a tree for efficient intersection
        # For simplicity, we just store the list as-is
        new(objects)
    end
end

# Ray-BVH intersection
function hit(bvh::BoundingVolumeHierarchy, ray::Ray, t_min::Float64, t_max::Float64)
    # This is essentially the same as a HittableList for our simplified implementation
    closest_so_far = t_max
    hit_anything = false
    result = HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), Material(), false)
    
    for object in bvh.objects
        temp_rec = hit(object, ray, t_min, closest_so_far)
        if temp_rec.hit
            hit_anything = true
            closest_so_far = temp_rec.t
            result = temp_rec
        end
    end
    
    return result
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

# Function to load OBJ mesh and transform it (scale, rotate, translate)
function load_obj_mesh(filename::String, material::Material;
                       scale::Vec3=Vec3(1.0, 1.0, 1.0),
                       rotation::Vec3=Vec3(0.0, 0.0, 0.0),
                       translation::Vec3=Vec3(0.0, 0.0, 0.0),
                       center::Bool=true,
                       normalize_size::Bool=false)
    # Parse the OBJ file
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

    # Center the mesh if requested (before scaling and rotation)
    if center || normalize_size
        # Calculate bounding box to find center and size
        min_coords = Vec3(Inf, Inf, Inf)
        max_coords = Vec3(-Inf, -Inf, -Inf)

        for v in vertices
            min_coords = Vec3(min(min_coords.x, v.x), min(min_coords.y, v.y), min(min_coords.z, v.z))
            max_coords = Vec3(max(max_coords.x, v.x), max(max_coords.y, v.y), max(max_coords.z, v.z))
        end

        # Find mesh center and size
        center_point = (min_coords + max_coords) / 2.0
        mesh_size = max_coords - min_coords
        max_dimension = max(mesh_size.x, max(mesh_size.y, mesh_size.z))

        # Center the mesh if requested
        if center
            for i in 1:length(vertices)
                vertices[i] = vertices[i] - center_point
            end
        end

        # Normalize size if requested (scale to fit in a unit cube)
        if normalize_size && max_dimension > 0
            normalized_scale = 1.0 / max_dimension
            for i in 1:length(vertices)
                vertices[i] = vertices[i] * normalized_scale
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

    # Create triangles and add to scene
    triangles = Hittable[]
    for face in faces
        # Create triangle from face vertices
        triangle_vertices = [vertices[face[1]], vertices[face[2]], vertices[face[3]]]
        push!(triangles, Triangle(triangle_vertices, material))
    end

    return triangles
end

# Create a simple scene with a mesh
function create_scene()
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

# Example function to create a scene with an OBJ mesh
function create_scene_with_obj()
    # Example objects
    objects = Hittable[]

    # Add a ground plane (represented as a large sphere)
    push!(objects, Sphere(Vec3(0, -100.5, -1), 100, Material(diffuse=Vec3(0.8, 0.8, 0.2))))

    # Add a light source
    push!(objects, Sphere(Vec3(0, 2, 0), 0.5, Material(diffuse=Vec3(0.8, 0.8, 0.8), emission=Vec3(4, 4, 4))))

    # Define a material for the mesh
    mesh_material = Material(diffuse=Vec3(0.7, 0.3, 0.2), specular=0.2, roughness=0.4)

    # Path to the OBJ file - replace with your actual file path
    # Try to find a common location for sample 3D models
    obj_file = expanduser("~/Downloads/bunny.obj")  # Stanford bunny is a common test model

    # Check if the file exists
    if isfile(obj_file)
        println("Loading OBJ mesh from: $obj_file")

        # Load and transform the mesh:
        # 1. Center it at the origin
        # 2. Normalize its size to fit in a 1x1x1 box
        # 3. Scale it to desired size (e.g., 0.5 units tall)
        # 4. Rotate it 90 degrees around the Y axis
        # 5. Translate it to desired position (e.g., Y=0 so it sits on the ground plane)
        mesh_triangles = load_obj_mesh(
            obj_file,
            mesh_material,
            center=true,                        # Center the mesh at origin
            normalize_size=true,                # Normalize to unit size
            scale=Vec3(0.5, 0.5, 0.5),          # Scale to half size
            rotation=Vec3(0.0, 90.0, 0.0),      # Rotate 90° around Y axis
            translation=Vec3(0.0, 0.0, -1.0)    # Position at Z=-1
        )

        # Add all triangles from the mesh to our scene
        for triangle in mesh_triangles
            push!(objects, triangle)
        end

        println("Added $(length(mesh_triangles)) triangles to the scene")
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

# Render the scene
function render_example(; width=1280, height=720, samples=200, interactive=true, output_file="render.exr", scene=nothing, camera=nothing)
    # If no scene/camera provided, create the default scene
    if scene === nothing || camera === nothing
        println("No scene provided, creating default scene...")
        scene, camera = create_scene()
    end

    println("Rendering with $samples samples per pixel...")
    image, hdr_data = render(scene, camera, width, height, samples_per_pixel=samples, max_depth=25)
    
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

# Create a scene with multiple transformed OBJ meshes
function create_scene_with_multiple_meshes()
    objects = Hittable[]

    # Add ground plane
    push!(objects, Sphere(Vec3(0, -100.5, -1), 100, Material(diffuse=Vec3(0.8, 0.8, 0.2))))

    # Add lights
    push!(objects, Sphere(Vec3(-2, 3, 2), 0.5, Material(diffuse=Vec3(0.8, 0.8, 0.8), emission=Vec3(5, 5, 5))))
    push!(objects, Sphere(Vec3(2, 2, 1), 0.25, Material(diffuse=Vec3(0.8, 0.6, 0.2), emission=Vec3(3, 2, 1))))

    # Example 1: Load a mesh and place it at the center
    # Create a metallic material for the mesh
    metal_material = Material(diffuse=Vec3(0.8, 0.8, 0.9), specular=0.8, roughness=0.1)
    obj_file_1 = expanduser("~/Downloads/bunny.obj")  # Stanford bunny is a common test model

    if isfile(obj_file_1)
        mesh1_triangles = load_obj_mesh(
            obj_file_1,
            metal_material,
            center=true,                      # Center the mesh
            normalize_size=true,              # Normalize size
            scale=Vec3(0.4, 0.4, 0.4),        # Scale down
            rotation=Vec3(0.0, 0.0, 0.0),     # No rotation
            translation=Vec3(0.0, 0.0, -1.5)  # Push back slightly
        )

        # Add all mesh triangles to the scene
        for triangle in mesh1_triangles
            push!(objects, triangle)
        end
        println("Added model 1 with $(length(mesh1_triangles)) triangles")
    end

    # Example 2: Load the same or another mesh but with different transformation
    diffuse_material = Material(diffuse=Vec3(0.2, 0.8, 0.3), specular=0.0, roughness=1.0)
    obj_file_2 = obj_file_1  # Reuse the same model with different parameters

    if isfile(obj_file_2)
        mesh2_triangles = load_obj_mesh(
            obj_file_2,
            diffuse_material,
            center=true,
            normalize_size=true,
            scale=Vec3(0.3, 0.5, 0.3),           # Non-uniform scaling for a different look
            rotation=Vec3(0.0, 45.0, 0.0),       # Rotate 45° around Y axis
            translation=Vec3(1.2, 0.0, -1.0)     # Position to the right
        )

        for triangle in mesh2_triangles
            push!(objects, triangle)
        end
        println("Added model 2 with $(length(mesh2_triangles)) triangles")
    end

    # Example 3: One more with different parameters
    glass_material = Material(diffuse=Vec3(0.9, 0.9, 0.9), specular=1.0, roughness=0.0)
    obj_file_3 = obj_file_1  # Reuse the same model with glass material

    if isfile(obj_file_3)
        mesh3_triangles = load_obj_mesh(
            obj_file_3,
            glass_material,
            center=true,
            normalize_size=true,
            scale=Vec3(0.25, 0.25, 0.25),        # Scale down more
            rotation=Vec3(20.0, -30.0, 0.0),     # Multiple axis rotation
            translation=Vec3(-1.2, 0.1, -0.8)    # Position to the left and slightly up
        )

        for triangle in mesh3_triangles
            push!(objects, triangle)
        end
        println("Added model 3 with $(length(mesh3_triangles)) triangles")
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

# Main function to run the renderer
function main()
    # Parse command line arguments here if needed
    interactive = true  # Set to false for headless rendering
    output_file = "render.exr"
    width = 640         # Wider for proper aspect ratio
    height = 360         # Standard 16:9 aspect ratio
    samples = 50        # Reduced from 400 to balance quality and rendering time

    # Choose which type of scene to render:
    scene_type = "obj"  # Options: "default", "obj", "multiple_obj"

    # Run the renderer with the selected scene type
    if scene_type == "obj"
        # Render a scene with a single OBJ model
        println("Rendering scene with a single OBJ model...")
        scene, camera = create_scene_with_obj()
    elseif scene_type == "multiple_obj"
        # Render a scene with multiple OBJ models
        println("Rendering scene with multiple OBJ models...")
        scene, camera = create_scene_with_multiple_meshes()
    else
        # Default scene with primitives
        println("Rendering default scene...")
        scene, camera = create_scene()
    end

    # Render the selected scene
    render_example(
        width=width,
        height=height,
        samples=samples,
        interactive=interactive,
        output_file=output_file,
        scene=scene,
        camera=camera
    )
end

# Run the main function
main()
