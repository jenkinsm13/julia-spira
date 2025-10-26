
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
    img_buffer_gpu::AbstractMatrix{Float32}
    rng_states_gpu::AbstractVector{UInt32}
    # Add other variables here if they also prove to be unstable (e.g. width, height, spp)
end
