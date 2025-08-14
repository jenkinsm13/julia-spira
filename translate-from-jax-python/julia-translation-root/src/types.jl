# Types for the SpectralRenderer

# Required packages
using LinearAlgebra
using StaticArrays

# Core data types for rendering system

# -----------------------------------------------------------------------------
# Core data types for rendering system
# -----------------------------------------------------------------------------

# Use StaticArrays for better performance with small vectors/matrices
const Vec3f = SVector{3, Float32}
const Point3f = Vec3f 
const Norm3f = Vec3f  # Type alias for normal vectors
const UV2f = SVector{2, Float32}  # UV coordinates

# Constants for spectral rendering
const N_WAVELENGTHS = 8
const MIN_WAVELENGTH_NM = Float32(400.0)
const MAX_WAVELENGTH_NM = Float32(700.0)
const WAVELENGTHS_NM = range(MIN_WAVELENGTH_NM, MAX_WAVELENGTH_NM, length=N_WAVELENGTHS)
const DELTA_WAVELENGTH_NM = (MAX_WAVELENGTH_NM - MIN_WAVELENGTH_NM) / (N_WAVELENGTHS - 1)

# Type Aliases for Physical Quantities
const Spectrum = SVector{N_WAVELENGTHS, Float32}
const MuellerMatrix = SMatrix{4, 4, Float32, 16}
const StokesVector = SVector{4, Float32}

# Mueller-Stokes Constants
const IDENTITY_MUELLER = SMatrix{4,4,Float32,16}(Float32(1.0),Float32(0.0),Float32(0.0),Float32(0.0),
                                                 Float32(0.0),Float32(1.0),Float32(0.0),Float32(0.0),
                                                 Float32(0.0),Float32(0.0),Float32(1.0),Float32(0.0),
                                                 Float32(0.0),Float32(0.0),Float32(0.0),Float32(1.0))

const DEPOLARIZER_MUELLER = SMatrix{4,4,Float32,16}(Float32(1.0),Float32(0.0),Float32(0.0),Float32(0.0),
                                                    Float32(0.0),Float32(0.0),Float32(0.0),Float32(0.0),
                                                    Float32(0.0),Float32(0.0),Float32(0.0),Float32(0.0),
                                                    Float32(0.0),Float32(0.0),Float32(0.0),Float32(0.0))

# Create spectral versions of Mueller matrices
const IDENTITY_MUELLER_SPECTRAL = [IDENTITY_MUELLER for _ in 1:N_WAVELENGTHS]
const DEPOLARIZER_MUELLER_SPECTRAL = [DEPOLARIZER_MUELLER for _ in 1:N_WAVELENGTHS]

# Ray structure with origin and direction
struct Ray
    origin::Point3f
    direction::Vec3f
    stokes_vector::SVector{N_WAVELENGTHS, StokesVector}
    tmin::Float32
    tmax::Float32

    # Basic inner constructor (for internal use by outer constructors)
    Ray(origin_val::Point3f, direction_val::Vec3f, stokes_val::SVector{N_WAVELENGTHS, StokesVector}, tmin_val::Float32, tmax_val::Float32) =
        new(origin_val, direction_val, stokes_val, tmin_val, tmax_val)
end

# Outer constructor with default stokes_vector and optional tmin/tmax
function Ray(origin::Point3f, direction::Vec3f, tmin::Float32=Float32(1.0e-4), tmax::Float32=Float32(1.0e10))
    norm_dir = normalize(direction)
    # Default Stokes: unpolarized, S0=1 for all wavelengths
    default_stokes = SVector{N_WAVELENGTHS}(ntuple(i -> StokesVector(Float32(1.0), Float32(0.0), Float32(0.0), Float32(0.0)), Val(N_WAVELENGTHS)))
    # Call the basic inner constructor
    return Ray(origin, norm_dir, default_stokes, tmin, tmax)
end

# Function to get a point along a ray
point_at(ray::Ray, t::Float32) = ray.origin + ray.direction * t

# Material structure
struct Material
    # Base material properties
    diffuse::Spectrum  # Diffuse reflectance spectrum
    emission::Spectrum  # Emission spectrum
    specular::Float32  # Specular reflection coefficient
    roughness::Float32  # Surface roughness
    
    # Polarization properties
    mueller_matrix::Vector{MuellerMatrix}  # Wavelength-dependent Mueller matrix
    
    # Optional texture data
    diffuse_texture::Union{Nothing, Array{Float32, 3}}  # H x W x N_WAVELENGTHS
    normal_map::Union{Nothing, Array{Float32, 3}}  # H x W x 3
    roughness_map::Union{Nothing, Array{Float32, 2}}  # H x W
    specular_map::Union{Nothing, Array{Float32, 2}}  # H x W
end

# Constructor with default values
Material(;
    diffuse::Spectrum = Spectrum(fill(Float32(0.5), N_WAVELENGTHS)),
    emission::Spectrum = Spectrum(fill(Float32(0.0), N_WAVELENGTHS)),
    specular::Float32 = Float32(0.0),
    roughness::Float32 = Float32(1.0),
    mueller_matrix::Union{Nothing, Vector{MuellerMatrix}} = nothing
) = Material(
    diffuse,
    emission,
    specular,
    roughness,
    mueller_matrix === nothing ? IDENTITY_MUELLER_SPECTRAL : mueller_matrix,
    nothing,  # diffuse_texture
    nothing,  # normal_map
    nothing,  # roughness_map
    nothing   # specular_map
)

# Intersection result structure
struct HitRecord
    t::Float32                # Distance along ray to intersection
    position::Point3f         # Position of intersection
    normal::Norm3f            # Surface normal at intersection (normalized)
    material_id::Int32        # Material ID for the intersected object
    uv::UV2f                  # UV coordinates for texture mapping
    wo::Vec3f                 # Outgoing direction (viewing direction, normalized)
    hit_flag::Int32           # Changed from hit::Bool - Flag indicating if intersection occurred (0 for false, 1 for true)
    
    # Default constructor for misses
    function HitRecord()
        return new(
            Float32(Inf), 
            Point3f(0, 0, 0), 
            Norm3f(0, 0, 1), 
            Int32(-1), 
            UV2f(0, 0),
            Vec3f(0, 0, 1),
            Int32(0) # hit_flag (false)
        )
    end
    
    # Constructor for hits
    function HitRecord(
            t::Float32, 
            position::Point3f, 
            normal::Vec3f, 
            material_id::Int32,
            uv::UV2f,
            wo::Vec3f,
            hit_status::Bool=true) # Parameter name changed for clarity
        return new(t, position, normalize(normal), material_id, uv, normalize(wo), hit_status ? Int32(1) : Int32(0)) # hit_flag
    end
end

# Simple light structures
struct DirectionalLight
    direction::Vec3f  # Direction TO the light (normalized)
    spd::Spectrum     # Spectral power distribution
    
    # Constructor to ensure the direction is normalized
    DirectionalLight(direction::Vec3f, spd::Spectrum) = new(normalize(direction), spd)
end

struct PointLight
    position::Point3f  # Position in 3D space
    spd::Spectrum      # Spectral power distribution
end

# Axis-Aligned Bounding Box for BVH optimization
struct AABB
    min::Point3f
    max::Point3f
end

# Function to combine two AABBs
function surrounding_box(box1::AABB, box2::AABB)
    min_point = Point3f(
        min(box1.min[1], box2.min[1]),
        min(box1.min[2], box2.min[2]),
        min(box1.min[3], box2.min[3])
    )
    
    max_point = Point3f(
        max(box1.max[1], box2.max[1]),
        max(box1.max[2], box2.max[2]),
        max(box1.max[3], box2.max[3])
    )
    
    return AABB(min_point, max_point)
end

# Abstract type for scene geometry
abstract type Hittable end

# Ray-AABB intersection test (fast slab method)
function hit_aabb(aabb::AABB, ray::Ray, t_min::Float32, t_max::Float32)
    # For each axis
    for a in 1:3
        # Get component for this axis (x, y, or z)
        origin_comp = ray.origin[a]
        direction_comp = ray.direction[a]
        min_comp = aabb.min[a]
        max_comp = aabb.max[a]
        
        # Handle division by zero
        inv_dir = abs(direction_comp) > Float32(1.0e-8) ? Float32(1.0) / direction_comp : Float32(1.0e8) * sign(direction_comp)
        t0 = (min_comp - origin_comp) * inv_dir
        t1 = (max_comp - origin_comp) * inv_dir
        
        # Sort t values
        if inv_dir < Float32(0.0)
            t0, t1 = t1, t0
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

# --- Plane Primitive ---
struct Plane <: Hittable
    normal::Vec3f
    d::Float32 # Offset from origin: dot(P, normal) + d = 0
    material_id_a::Int32
    material_id_b::Int32
    checker_scale::Float32
end

# Scene definition
# struct Scene ... # If we need a more complex scene structure later

# For AD, we might need to define EnzymeRules for custom types if not auto-handled.

# End of types.jl 