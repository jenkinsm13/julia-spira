using LinearAlgebra
using Random
using Images
using Colors
using FileIO
using Metal
using ObjectiveC.Foundation

# Vector and Matrix Types
struct Vec3
    x::Float64
    y::Float64
    z::Float64
end

+(a::Vec3, b::Vec3) = Vec3(Base.:+(a.x, b.x), Base.:+(a.y, b.y), Base.:+(a.z, b.z))
-(a::Vec3, b::Vec3) = Vec3(Base.:-(a.x, b.x), Base.:-(a.y, b.y), Base.:-(a.z, b.z))
*(a::Vec3, b::Real) = Vec3(Base.:*(a.x, b), Base.:*(a.y, b), Base.:*(a.z, b))
*(b::Real, a::Vec3) = a * b
*(a::Vec3, b::Vec3) = Vec3(Base.:*(a.x, b.x), Base.:*(a.y, b.y), Base.:*(a.z, b.z))  # Component-wise for colors
/(a::Vec3, b::Real) = Vec3(Base.:/(a.x, b), Base.:/(a.y, b), Base.:/(a.z, b))
dot(a::Vec3, b::Vec3) = Base.:+(Base.:+(Base.:*(a.x, b.x), Base.:*(a.y, b.y)), Base.:*(a.z, b.z))
Base.length(v::Vec3) = sqrt(dot(v, v))
normalize(v::Vec3) = v / length(v)
cross(a::Vec3, b::Vec3) = Vec3(
    Base.:-(Base.:*(a.y, b.z), Base.:*(a.z, b.y)),
    Base.:-(Base.:*(a.z, b.x), Base.:*(a.x, b.z)),
    Base.:-(Base.:*(a.x, b.y), Base.:*(a.y, b.x))
)

struct Mat3x3
    m::Matrix{Float64}
    Mat3x3() = new(zeros(3, 3))
    Mat3x3(m::Matrix{Float64}) = new(m)
end

*(A::Mat3x3, v::Vec3) = Vec3(
    dot(Vec3(A.m[1,1], A.m[1,2], A.m[1,3]), v),
    dot(Vec3(A.m[2,1], A.m[2,2], A.m[2,3]), v),
    dot(Vec3(A.m[3,1], A.m[3,2], A.m[3,3]), v)
)
*(A::Mat3x3, B::Mat3x3) = Mat3x3(A.m * B.m)

# GPU-compatible Vec3f for data transfer
struct Vec3f_GPU
    x::Float32
    y::Float32
    z::Float32
end
Base.isbits(::Type{Vec3f_GPU}) = true # Ensure it's a bits type for Metal buffers

# GPU-compatible Sphere for data transfer
struct Sphere_GPU
    center::Vec3f_GPU
    radius::Float32
    material_id::UInt32
end
Base.isbits(::Type{Sphere_GPU}) = true

# GPU-compatible OpenPBRMaterial for data transfer (simplified)
struct OpenPBRMaterial_GPU
    base_color::Vec3f_GPU
    base_metalness::Float32
    specular_roughness::Float32
    specular_ior::Float32
    emission_color::Vec3f_GPU
    emission_luminance::Float32
end
Base.isbits(::Type{OpenPBRMaterial_GPU}) = true

# GPU-compatible Camera for data transfer
struct Camera_GPU
    origin::Vec3f_GPU
    lower_left_corner::Vec3f_GPU
    horizontal::Vec3f_GPU
    vertical::Vec3f_GPU
    u::Vec3f_GPU # Orthonormal basis x
    v::Vec3f_GPU # Orthonormal basis y
    w::Vec3f_GPU # Orthonormal basis z (view direction)
    lens_radius::Float32
end
Base.isbits(::Type{Camera_GPU}) = true

# Spectral and Polarimetric Types
struct SpectralPower
    values::Vector{Float64}  # Power at each wavelength
end

struct StokesVector
    s::Vector{Float64}  # [S0, S1, S2, S3]
    StokesVector(s::Vector{Float64}) = new(length(s) == 4 ? s : error("Stokes vector must have 4 components"))
end

struct MuellerMatrix
    m::Mat3x3
end

*(M::MuellerMatrix, S::StokesVector) = StokesVector([dot(M.m.m[i, :], S.s) for i in 1:4])

# Ray and Geometry
struct Ray
    origin::Vec3
    direction::Vec3
end

point_at(ray::Ray, t::Float64) = ray.origin + ray.direction * t

struct AABB
    min::Vec3
    max::Vec3
end

surrounding_box(box1::AABB, box2::AABB) = AABB(
    Vec3(min(box1.min.x, box2.min.x), min(box1.min.y, box2.min.y), min(box1.min.z, box2.min.z)),
    Vec3(max(box1.max.x, box2.max.x), max(box1.max.y, box2.max.y), max(box1.max.z, box2.max.z))
)

function hit_aabb(aabb::AABB, ray::Ray, t_min::Float64, t_max::Float64)
    for a in 1:3
        origin_comp = [ray.origin.x, ray.origin.y, ray.origin.z][a]
        direction_comp = [ray.direction.x, ray.direction.y, ray.direction.z][a]
        min_comp = [aabb.min.x, aabb.min.y, aabb.min.z][a]
        max_comp = [aabb.max.x, aabb.max.y, aabb.max.z][a]
        
        inv_dir = 1.0 / direction_comp
        t0 = (min_comp - origin_comp) * inv_dir
        t1 = (max_comp - origin_comp) * inv_dir
        
        if inv_dir < 0
            t0, t1 = t1, t0
        end
        
        t_min = max(t0, t_min)
        t_max = min(t1, t_max)
        
        if t_max <= t_min
            return false
        end
    end
    return true
end

# OpenPBR Material
struct OpenPBRMaterial
    base_weight::Float64
    base_color::Vec3
    base_metalness::Float64
    base_diffuse_roughness::Float64
    specular_weight::Float64
    specular_color::Vec3
    specular_roughness::Float64
    specular_roughness_anisotropy::Float64
    specular_ior::Float64
    transmission_weight::Float64
    transmission_color::Vec3
    transmission_depth::Float64
    transmission_scatter::Vec3
    transmission_scatter_anisotropy::Float64
    transmission_dispersion_scale::Float64
    transmission_dispersion_abbe_number::Float64
    subsurface_weight::Float64
    subsurface_color::Vec3
    subsurface_radius::Float64
    subsurface_radius_scale::Vec3
    subsurface_scatter_anisotropy::Float64
    coat_weight::Float64
    coat_color::Vec3
    coat_roughness::Float64
    coat_roughness_anisotropy::Float64
    coat_ior::Float64
    coat_darkening::Float64
    fuzz_weight::Float64
    fuzz_color::Vec3
    fuzz_roughness::Float64
    emission_luminance::Float64
    emission_color::Vec3
    thin_film_weight::Float64
    thin_film_thickness::Float64
    thin_film_ior::Float64
    geometry_opacity::Float64
    geometry_thin_walled::Bool
end

# Hit Record
struct HitRecord
    t::Float64
    position::Vec3
    normal::Vec3
    material::OpenPBRMaterial
    hit::Bool
    wavelength::Float64
    stokes::StokesVector
end

# Hittable Types
abstract type Hittable end

struct Sphere <: Hittable
    center::Vec3
    radius::Float64
    material::OpenPBRMaterial
end

bounding_box(s::Sphere) = AABB(
    s.center - Vec3(s.radius, s.radius, s.radius),
    s.center + Vec3(s.radius, s.radius, s.radius)
)

struct Triangle <: Hittable
    vertices::Vector{Vec3}
    material::OpenPBRMaterial
    bbox::AABB
end

function Triangle(vertices::Vector{Vec3}, material::OpenPBRMaterial)
    min_p = Vec3(minimum(v.x for v in vertices), minimum(v.y for v in vertices), minimum(v.z for v in vertices))
    max_p = Vec3(maximum(v.x for v in vertices), maximum(v.y for v in vertices), maximum(v.z for v in vertices))
    epsilon = 1e-8
    min_p -= Vec3(epsilon, epsilon, epsilon)
    max_p += Vec3(epsilon, epsilon, epsilon)
    Triangle(vertices, material, AABB(min_p, max_p))
end

bounding_box(t::Triangle) = t.bbox

struct BVHNode <: Hittable
    bbox::AABB
    left::Union{Hittable, Nothing}
    right::Union{Hittable, Nothing}
end

function BVHNode(objects::Vector{Hittable}, start::Int, finish::Int)
    axis = rand(1:3)
    sort_func = (a, b) -> bounding_box(a).min[axis] < bounding_box(b).min[axis]
    span = finish - start
    if span == 1
        return BVHNode(bounding_box(objects[start]), objects[start], nothing)
    elseif span == 2
        left, right = sort_func(objects[start], objects[start+1]) ? (objects[start], objects[start+1]) : (objects[start+1], objects[start])
        return BVHNode(surrounding_box(bounding_box(left), bounding_box(right)), left, right)
    else
        sorted = sort(objects[start:finish-1], lt=sort_func)
        mid = start + div(span, 2)
        left = BVHNode(sorted, start, mid)
        right = BVHNode(sorted, mid, finish)
        return BVHNode(surrounding_box(bounding_box(left), bounding_box(right)), left, right)
    end
end

bounding_box(node::BVHNode) = node.bbox

# Intersection Functions
function hit(s::Sphere, ray::Ray, t_min::Float64, t_max::Float64, wavelength::Float64, stokes::StokesVector)
    oc = ray.origin - s.center
    a = dot(ray.direction, ray.direction)
    b = 2.0 * dot(oc, ray.direction)
    c = dot(oc, oc) - s.radius^2
    disc = b^2 - 4 * a * c
    if disc < 0
        return HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), s.material, false, wavelength, stokes)
    end
    t = (-b - sqrt(disc)) / (2.0 * a)
    if t < t_min || t > t_max
        t = (-b + sqrt(disc)) / (2.0 * a)
        if t < t_min || t > t_max
            return HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), s.material, false, wavelength, stokes)
        end
    end
    p = point_at(ray, t)
    n = normalize(p - s.center)
    return HitRecord(t, p, n, s.material, true, wavelength, stokes)
end

function hit(t::Triangle, ray::Ray, t_min::Float64, t_max::Float64, wavelength::Float64, stokes::StokesVector)
    if !hit_aabb(t.bbox, ray, t_min, t_max)
        return HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), t.material, false, wavelength, stokes)
    end
    v0, v1, v2 = t.vertices
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = cross(ray.direction, edge2)
    a = dot(edge1, h)
    if abs(a) < 1e-8
        return HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), t.material, false, wavelength, stokes)
    end
    f = 1.0 / a
    s = ray.origin - v0
    u = f * dot(s, h)
    if u < 0 || u > 1
        return HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), t.material, false, wavelength, stokes)
    end
    q = cross(s, edge1)
    v = f * dot(ray.direction, q)
    if v < 0 || u + v > 1
        return HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), t.material, false, wavelength, stokes)
    end
    t = f * dot(edge2, q)
    if t < t_min || t > t_max
        return HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), t.material, false, wavelength, stokes)
    end
    p = point_at(ray, t)
    n = normalize(cross(edge1, edge2))
    return HitRecord(t, p, n, t.material, true, wavelength, stokes)
end

function hit(node::BVHNode, ray::Ray, t_min::Float64, t_max::Float64, wavelength::Float64, stokes::StokesVector)
    if !hit_aabb(node.bbox, ray, t_min, t_max)
        return HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0), OpenPBRMaterial(0.0, Vec3(0,0,0), 0.0, 0.0, 0.0, Vec3(0,0,0), 0.0, 0.0, 0.0, 0.0, Vec3(0,0,0), 0.0, Vec3(0,0,0), 0.0, 0.0, 0.0, 0.0, Vec3(0,0,0), 0.0, Vec3(0,0,0), 0.0, 0.0, Vec3(0,0,0), 0.0, 0.0, 0.0, 0.0, 0.0, Vec3(0,0,0), 0.0, 0.0, Vec3(0,0,0), 0.0, 0.0, 0.0, 1.0, false), false, wavelength, stokes)
    end
    if isnothing(node.right)
        return hit(node.left, ray, t_min, t_max, wavelength, stokes)
    end
    left_hit = hit(node.left, ray, t_min, t_max, wavelength, stokes)
    right_hit = hit(node.right, ray, t_min, left_hit.hit ? left_hit.t : t_max, wavelength, stokes)
    return right_hit.hit ? right_hit : left_hit
end

# Utility Functions
random_in_unit_sphere(rng) = normalize(Vec3(randn(rng), randn(rng), randn(rng)))

function create_coordinate_system(n::Vec3)
    t = abs(n.x) > abs(n.y) ? Vec3(-n.z, 0, n.x) / sqrt(n.x^2 + n.z^2) : Vec3(0, n.z, -n.y) / sqrt(n.y^2 + n.z^2)
    b = cross(n, t)
    return t, b, n
end

function local_to_world(v::Vec3, t::Vec3, b::Vec3, n::Vec3)
    return v.x * t + v.y * b + v.z * n
end

# Spectral and Polarimetric Functions
const WAVELENGTHS = collect(380.0:10.0:780.0)  # nm

function fresnel_dielectric(cos_theta, eta)
    sin_theta = sqrt(1 - cos_theta^2)
    sin_theta_t = sin_theta / eta
    if sin_theta_t >= 1
        return 1.0
    end
    cos_theta_t = sqrt(1 - sin_theta_t^2)
    r_par = (eta * cos_theta - cos_theta_t) / (eta * cos_theta + cos_theta_t)
    r_perp = (cos_theta - eta * cos_theta_t) / (cos_theta + eta * cos_theta_t)
    return (r_par^2 + r_perp^2) / 2
end

function fresnel_metal(cos_theta, F0::Vec3, specular_color::Vec3)
    F_schlick = F0 + (Vec3(1,1,1) - F0) * (1 - cos_theta)^5
    mu_bar = 1/7
    F_mu_bar = specular_color * (F0 + (Vec3(1,1,1) - F0) * (1 - mu_bar)^5)
    correction = (cos_theta * (1 - cos_theta)^6) / (mu_bar * (1 - mu_bar)^6) * (F_schlick - F_mu_bar)
    return F_schlick - correction
end

function sample_ggx(n::Vec3, roughness::Float64, rng)
    alpha = roughness^2
    xi1, xi2 = rand(rng), rand(rng)
    phi = 2 * π * xi2
    cos_theta = sqrt((1 - xi1) / (xi1 * (alpha^2 - 1) + 1))
    sin_theta = sqrt(1 - cos_theta^2)
    h_local = Vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta)
    t, b, _ = create_coordinate_system(n)
    return normalize(local_to_world(h_local, t, b, n))
end

function ggx_D(h::Vec3, n::Vec3, alpha::Float64)
    cos_theta = dot(h, n)
    if cos_theta <= 0
        return 0.0
    end
    tan_theta_sq = (1 - cos_theta^2) / cos_theta^2
    denom = π * alpha^2 * (1 + tan_theta_sq / (alpha^2))^2
    return alpha^2 / denom
end

function smith_G(wi::Vec3, wo::Vec3, n::Vec3, alpha::Float64)
    cos_theta_i = dot(wi, n)
    cos_theta_o = dot(wo, n)
    if cos_theta_i <= 0 || cos_theta_o <= 0
        return 0.0
    end
    tan_theta_i = sqrt(1 - cos_theta_i^2) / cos_theta_i
    tan_theta_o = sqrt(1 - cos_theta_o^2) / cos_theta_o
    lambda_i = 0.5 * (sqrt(1 + alpha^2 * tan_theta_i^2) - 1)
    lambda_o = 0.5 * (sqrt(1 + alpha^2 * tan_theta_o^2) - 1)
    return 1 / (1 + lambda_i + lambda_o)
end

function mueller_reflection(cos_theta)
    s = sqrt(1 - cos_theta^2)
    M = Mat3x3([
        1.0  0.0  0.0;
        0.0  1.0  0.0;
        0.0  0.0 -1.0
    ])
    return MuellerMatrix(M)
end

# Scattering Function
function scatter(ray::Ray, rec::HitRecord, rng)
    wi = -ray.direction
    n = rec.normal
    cos_theta_i = dot(wi, n)
    if cos_theta_i < 0
        n = -n
        cos_theta_i = -cos_theta_i
    end
    
    mat = rec.material
    t, b, _ = create_coordinate_system(n)
    
    # Simplified BSDF evaluation (full OpenPBR to be implemented)
    if mat.base_metalness > 0.5
        h = sample_ggx(n, mat.specular_roughness, rng)
        wo = reflect(wi, h)
        if dot(wo, n) > 0
            F = fresnel_metal(cos_theta_i, mat.base_color * mat.base_weight, mat.specular_color)
            D = ggx_D(h, n, mat.specular_roughness^2)
            G = smith_G(wi, wo, n, mat.specular_roughness^2)
            atten = mat.specular_weight * F * D * G / (4 * cos_theta_i * dot(wo, n))
            M = mueller_reflection(cos_theta_i)
            return true, Ray(rec.position, wo), SpectralPower(fill(atten, length(WAVELENGTHS))), M * rec.stokes
        end
    else
        F = fresnel_dielectric(cos_theta_i, mat.specular_ior)
        if rand(rng) < F
            h = sample_ggx(n, mat.specular_roughness, rng)
            wo = reflect(wi, h)
            if dot(wo, n) > 0
                atten = F
                M = mueller_reflection(cos_theta_i)
                return true, Ray(rec.position, wo), SpectralPower(fill(atten, length(WAVELENGTHS))), M * rec.stokes
            end
        else
            xi1, xi2 = rand(rng), rand(rng)
            cos_theta = sqrt(xi1)
            sin_theta = sqrt(1 - cos_theta^2)
            phi = 2 * π * xi2
            local_dir = Vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta)
            wo = local_to_world(local_dir, t, b, n)
            atten = mat.base_color * (1 - F) / π
            M = MuellerMatrix(Mat3x3(zeros(3,3)))  # Depolarizing for diffuse
            return true, Ray(rec.position, wo), SpectralPower(fill(atten.x, length(WAVELENGTHS))), M * rec.stokes
        end
    end
    return false, Ray(Vec3(0,0,0), Vec3(0,0,0)), SpectralPower(zeros(length(WAVELENGTHS))), rec.stokes
end

# Path Tracing
function ray_color(ray::Ray, world::Hittable, depth::Int, rng, wavelength::Float64, stokes::StokesVector)
    if depth <= 0
        return SpectralPower(zeros(length(WAVELENGTHS)))
    end
    rec = hit(world, ray, 0.001, Inf, wavelength, stokes)
    if rec.hit
        emitted = SpectralPower(fill(rec.material.emission_luminance * rec.material.emission_color.x, length(WAVELENGTHS)))
        scattered, new_ray, atten, new_stokes = scatter(ray, rec, rng)
        if scattered
            return emitted + atten * ray_color(new_ray, world, depth-1, rng, wavelength, new_stokes)
        end
        return emitted
    end
    t = 0.5 * (ray.direction.y + 1.0)
    return SpectralPower(fill((1.0 - t) * 1.0 + t * 0.5, length(WAVELENGTHS)))  # Simplified background
end

# Metal GPU Rendering
const METAL_KERNEL = """
#include <metal_stdlib>
#include <metal_math>
using namespace metal;

// --- Vec3f and Ray Structs (slight review/ensure compatibility) ---
struct Vec3f {
    float x, y, z;
    Vec3f(float x=0.0f, float y=0.0f, float z=0.0f) : x(x), y(y), z(z) {}
    Vec3f operator+(Vec3f v) { return Vec3f(x+v.x, y+v.y, z+v.z); }
    Vec3f operator-(Vec3f v) { return Vec3f(x-v.x, y-v.y, z-v.z); }
    Vec3f operator*(float s) { return Vec3f(x*s, y*s, z*s); }
    Vec3f operator*(Vec3f v) { return Vec3f(x*v.x, y*v.y, z*v.z); } // Component-wise
    Vec3f operator/(float s) { return Vec3f(x/s, y/s, z/s); }
    float dot(Vec3f v) { return x*v.x + y*v.y + z*v.z; }
};

inline Vec3f operator*(float s, Vec3f v) { return v * s; }

inline float length_squared(Vec3f v) { return v.x*v.x + v.y*v.y + v.z*v.z; }
inline float length(Vec3f v) { return sqrt(length_squared(v)); }

inline Vec3f normalize(Vec3f v) {
    float l = length(v);
    if (l < 1e-6f) {
        return Vec3f(0.0f, 0.0f, 0.0f);
    }
    return v * (1.0f / l);
}

inline Vec3f cross(Vec3f a, Vec3f b) {
    return Vec3f(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

inline Vec3f reflect(Vec3f v, Vec3f n) {
    return v - 2.0f * dot(v, n) * n;
}

struct Ray {
    Vec3f origin, direction;
    Ray(Vec3f o, Vec3f d) : origin(o), direction(d) {}
};

inline Vec3f point_at(Ray r, float t) {
    return r.origin + r.direction * t;
}

// --- New GPU Structs ---
struct OpenPBRMaterial_GPU {
    Vec3f base_color;
    float base_metalness; // Simplified: 0 for dielectric, 1 for metal
    float specular_roughness;
    float specular_ior; // Index of Refraction for dielectrics
    Vec3f emission_color;
    float emission_luminance;
    // Add other relevant OpenPBR params as needed, keeping it simple for now
};

struct Sphere_GPU {
    Vec3f center;
    float radius;
    uint material_id; // Index into a materials buffer
};

struct HitRecord_GPU {
    float t;
    Vec3f position;
    Vec3f normal;
    uint material_id;
    bool front_face; // To handle normal orientation

    inline void set_face_normal(Ray r, Vec3f outward_normal) {
        front_face = dot(r.direction, outward_normal) < 0.0f;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

struct Camera_GPU {
    Vec3f origin;
    Vec3f lower_left_corner;
    Vec3f horizontal;
    Vec3f vertical;
    Vec3f u, v, w; // Orthonormal basis
    float lens_radius;
};


// --- Random Number Generation (already present, ensure it's used correctly) ---
float random_float(thread uint& state) {
    state = state * 747796405u + 2891336453u;
    uint result = ((state >> ((state >> 28) + 4)) ^ state) * 277803737u;
    result = (result >> 22) ^ result;
    return float(result) / 4294967295.0f;
}

Vec3f random_vec3(thread uint& state) {
    return Vec3f(random_float(state), random_float(state), random_float(state));
}

Vec3f random_vec3(float min_val, float max_val, thread uint& state) {
    return Vec3f(min_val + (max_val - min_val) * random_float(state),
                 min_val + (max_val - min_val) * random_float(state),
                 min_val + (max_val - min_val) * random_float(state));
}

Vec3f random_in_unit_sphere(thread uint& state) {
    while (true) {
        Vec3f p = random_vec3(-1.0f, 1.0f, state);
        if (length_squared(p) < 1.0f) return p;
    }
}

Vec3f random_unit_vector(thread uint& state) {
    return normalize(random_in_unit_sphere(state));
}

Vec3f random_in_hemisphere(Vec3f normal, thread uint& state) {
    Vec3f in_unit_sphere = random_in_unit_sphere(state);
    if (dot(in_unit_sphere, normal) > 0.0f) // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

// --- GGX related (simplified for now, full implementation is complex) ---
// For a more complete PBR, GGX sampling and distribution would be here.
// Keeping scatter logic simpler for this pass.

// --- Intersection Functions ---
bool hit_sphere_gpu(constant Sphere_GPU& sphere, Ray r, float t_min, float t_max, thread HitRecord_GPU& rec) {
    Vec3f oc = r.origin - sphere.center;
    float a = length_squared(r.direction);
    float half_b = dot(oc, r.direction);
    float c = length_squared(oc) - sphere.radius * sphere.radius;
    float discriminant = half_b * half_b - a * c;

    if (discriminant < 0.0f) return false;
    float sqrtd = sqrt(discriminant);

    float root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) return false;
    }

    rec.t = root;
    rec.position = point_at(r, rec.t);
    Vec3f outward_normal = (rec.position - sphere.center) / sphere.radius;
    rec.set_face_normal(r, outward_normal);
    rec.material_id = sphere.material_id;
    return true;
}

// Scene intersection: iterates through all spheres
// In a real engine, this would use a BVH or other acceleration structure
bool hit_scene_gpu(device Sphere_GPU* spheres, uint num_spheres, Ray r, float t_min, float t_max, thread HitRecord_GPU& rec) {
    HitRecord_GPU temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (uint i = 0; i < num_spheres; ++i) {
        if (hit_sphere_gpu(spheres[i], r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}


// --- Scattering Logic (Simplified) ---
// Returns true if scattered, false if absorbed.
// `attenuation` is the color/energy multiplied.
// `scattered_ray` is the new ray.
bool scatter_gpu(Ray r_in, thread HitRecord_GPU& rec, device OpenPBRMaterial_GPU* materials,
                 thread Vec3f& attenuation, thread Ray& scattered_ray, thread uint& rng_state)
{
    OpenPBRMaterial_GPU mat = materials[rec.material_id];
    attenuation = mat.base_color; // Default attenuation

    if (mat.base_metalness > 0.5f) { // Metal
        Vec3f reflected = reflect(normalize(r_in.direction), rec.normal);
        // Add fuzziness based on roughness
        scattered_ray = Ray(rec.position, reflected + mat.specular_roughness * random_in_unit_sphere(rng_state));
        attenuation = mat.base_color; // Metals use base_color for specular
        return (dot(scattered_ray.direction, rec.normal) > 0.0f); // Check for grazing angles
    } else { // Dielectric
        float refraction_ratio = rec.front_face ? (1.0f / mat.specular_ior) : mat.specular_ior;
        Vec3f unit_direction = normalize(r_in.direction);

        // Schlick's approximation for reflectance
        float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0f);
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
        float R0 = (1.0f - refraction_ratio) / (1.0f + refraction_ratio);
        R0 = R0 * R0;
        float reflectance = R0 + (1.0f - R0) * pow((1.0f - cos_theta), 5.0f);

        if (cannot_refract || reflectance > random_float(rng_state)) { // Reflection
             Vec3f reflected = reflect(unit_direction, rec.normal);
             scattered_ray = Ray(rec.position, reflected);
        } else { // Refraction (simplified, no roughness for refraction yet)
            Vec3f perp = refraction_ratio * (unit_direction + cos_theta * rec.normal);
            Vec3f parallel = -sqrt(fabs(1.0f - length_squared(perp))) * rec.normal;
            scattered_ray = Ray(rec.position, perp + parallel);
        }
        attenuation = Vec3f(1.0f, 1.0f, 1.0f); // Dielectrics are non-colored specularly
        return true;
    }
    return false; // Should be handled by logic above
}


// --- Ray Color (Iterative Path Tracing) ---
Vec3f ray_color_gpu(Ray r,
                    device Sphere_GPU* spheres, uint num_spheres,
                    device OpenPBRMaterial_GPU* materials,
                    int max_depth,
                    thread uint& rng_state)
{
    Vec3f accumulated_color(0.0f, 0.0f, 0.0f);
    Vec3f current_attenuation(1.0f, 1.0f, 1.0f);
    Ray current_ray = r;

    for (int depth = 0; depth < max_depth; ++depth) {
        HitRecord_GPU rec;
        if (hit_scene_gpu(spheres, num_spheres, current_ray, 0.001f, FLT_MAX, rec)) {
            OpenPBRMaterial_GPU mat = materials[rec.material_id];
            Vec3f emitted = mat.emission_luminance * mat.emission_color;
            accumulated_color = accumulated_color + emitted * current_attenuation;

            Ray scattered_ray_temp(Vec3f(0,0,0), Vec3f(0,0,0)); // Placeholder
            Vec3f attenuation_temp;
            if (scatter_gpu(current_ray, rec, materials, attenuation_temp, scattered_ray_temp, rng_state)) {
                current_attenuation = current_attenuation * attenuation_temp;
                current_ray = scattered_ray_temp;
                if (length_squared(current_attenuation) < 0.001f) { // Russian roulette / absorption
                     break;
                }
            } else { // Absorbed or hit nothing that scatters
                break;
            }
        } else { // Ray hit background/sky
            // Simple sky: gradient based on y-direction
            Vec3f unit_direction = normalize(current_ray.direction);
            float t = 0.5f * (unit_direction.y + 1.0f);
            Vec3f sky_color = (1.0f - t) * Vec3f(1.0f, 1.0f, 1.0f) + t * Vec3f(0.5f, 0.7f, 1.0f);
            accumulated_color = accumulated_color + sky_color * current_attenuation;
            break;
        }
    }
    return accumulated_color;
}


// --- Kernel Entry Point ---
kernel void trace_ray_gpu(
    device Vec3f* output_colors [[buffer(0)]],
    device Sphere_GPU* spheres [[buffer(1)]],
    device OpenPBRMaterial_GPU* materials [[buffer(2)]],
    constant Camera_GPU& cam [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]], // Global ID (pixel coordinates)
    uint image_width [[buffer(4)]],
    uint image_height [[buffer(5)]],
    uint num_samples [[buffer(6)]],
    uint max_depth [[buffer(7)]],
    uint num_spheres_arg [[buffer(8)]] // Actual number of spheres
) {
    uint rng_seed = gid.x + gid.y * image_width + num_samples; // Simple seed
    thread uint rng_state = rng_seed;

    Vec3f final_color(0.0f, 0.0f, 0.0f);
    for (uint s = 0; s < num_samples; ++s) {
        // Anti-aliasing: re-calculate u,v for each sample
        float u_sample = (float(gid.x) + random_float(rng_state)) / float(image_width);
        float v_sample = (float(gid.y) + random_float(rng_state)) / float(image_height);
        
        // Get ray from camera model
        Vec3f rd_offset = cam.lens_radius * random_in_unit_sphere(rng_state); // For depth of field
        Vec3f offset = cam.u * rd_offset.x + cam.v * rd_offset.y;
        Ray r(cam.origin + offset, normalize(cam.lower_left_corner + u_sample * cam.horizontal + v_sample * cam.vertical - cam.origin - offset));

        final_color = final_color + ray_color_gpu(r, spheres, num_spheres_arg, materials, int(max_depth), rng_state);
    }
    
    uint flat_idx = gid.y * image_width + gid.x;
    output_colors[flat_idx] = final_color / float(num_samples);
    
    // Gamma correction (simple approx)
    output_colors[flat_idx].x = sqrt(output_colors[flat_idx].x);
    output_colors[flat_idx].y = sqrt(output_colors[flat_idx].y);
    output_colors[flat_idx].z = sqrt(output_colors[flat_idx].z);

    // Ensure values are clamped [0,1] - though sqrt might handle some negatives if color was <0
    output_colors[flat_idx].x = fmax(0.0f, fmin(1.0f, output_colors[flat_idx].x));
    output_colors[flat_idx].y = fmax(0.0f, fmin(1.0f, output_colors[flat_idx].y));
    output_colors[flat_idx].z = fmax(0.0f, fmin(1.0f, output_colors[flat_idx].z));
}
"""

# Helper function to convert Vec3 to Vec3f_GPU
function to_vec3f_gpu(v::Vec3)
    return Vec3f_GPU(Float32(v.x), Float32(v.y), Float32(v.z))
end

function setup_camera(lookfrom::Vec3, lookat::Vec3, vup::Vec3, vfov_deg::Float64, aspect_ratio::Float64, aperture::Float64, focus_dist::Float64)
    theta = Base.:/(Base.:*(vfov_deg, pi), 180.0) # More explicit deg2rad
    h = tan(Base.:/(theta, 2.0))
    viewport_height = Base.:*(2.0, h)
    viewport_width = Base.:*(aspect_ratio, viewport_height)

    w = normalize(lookfrom - lookat) # normalize itself uses Base qualified ops now
    u_vec = normalize(cross(vup, w)) # Renamed to u_vec to avoid conflict with Camera_GPU.u
    v_vec = cross(w, u_vec)      # Renamed to v_vec

    origin_gpu = to_vec3f_gpu(lookfrom)
    horizontal_gpu = to_vec3f_gpu( Base.:*(focus_dist, viewport_width) * u_vec ) # Corrected: Base.* for scalar * scalar, then normal * for scalar * Vec3
    vertical_gpu = to_vec3f_gpu( Base.:*(focus_dist, viewport_height) * v_vec ) # Corrected
    
    # Breaking down the lower_left_corner calculation for clarity and qualification
    term1_scalar = Base.:*(focus_dist, Base.:*(viewport_width, Base.:/(1.0, 2.0)))
    term1 = term1_scalar * u_vec # Corrected
    term2_scalar = Base.:*(focus_dist, Base.:*(viewport_height, Base.:/(1.0, 2.0)))
    term2 = term2_scalar * v_vec # Corrected
    term3 = focus_dist * w       # Corrected (Real * Vec3)
    lower_left_corner_val = lookfrom - term1 - term2 - term3 # Vec3 ops are already Base qualified internally
    lower_left_corner_gpu = to_vec3f_gpu(lower_left_corner_val)
    
    return Camera_GPU(
        origin_gpu,
        lower_left_corner_gpu,
        horizontal_gpu,
        vertical_gpu,
        to_vec3f_gpu(u_vec),
        to_vec3f_gpu(v_vec),
        to_vec3f_gpu(w),
        Base.:/(Float32(aperture), Float32(2.0)) # Qualified and ensured Float32 for consistency
    )
end

function render_gpu(scene::Vector{Hittable}, width::Int, height::Int, samples_per_pixel::Int, max_depth_render::Int, camera_gpu::Camera_GPU)
    println("Starting render_gpu with full path tracing logic...")
    device = Metal.device()
    if isnothing(device)
        println("No Metal device found, falling back to CPU")
        return render_cpu(scene, width, height, samples_per_pixel)
    end
    
    println("Metal device found. Compiling kernel...")
    lib = nothing
    try
        lib = Metal.MTLLibrary(device, METAL_KERNEL)
    catch e
        println("Metal kernel compilation failed. Error details:")
        if isa(e, Foundation.NSErrorInstance)
            println("  Localized Description: ", e.localizedDescription)
            println("  Error Domain: ", e.domain)
            println("  Error Code: ", e.code)
            println("  User Info: ", e.userInfo)
        else
            println("  Error was not an NSErrorInstance: ", typeof(e))
            showerror(stdout, e)
            println()
        end
        println("Falling back to CPU rendering.")
        return render_cpu(scene, width, height, samples_per_pixel)
    end
    println("Metal kernel compilation successful.")

    kernel_function = Metal.MTLFunction(lib, "trace_ray_gpu")
    if isnothing(kernel_function)
        println("Failed to get kernel function. Falling back to CPU.")
        return render_cpu(scene, width, height, samples_per_pixel)
    end
    println("Kernel function 'trace_ray_gpu' obtained.")

    pipeline_state = Metal.MTLComputePipelineState(device, kernel_function)
    if isnothing(pipeline_state)
        println("Failed to create compute pipeline state. Falling back to CPU.")
        return render_cpu(scene, width, height, samples_per_pixel)
    end
    println("Compute pipeline state created.")

    num_pixels = width * height

    # Prepare scene data for GPU
    gpu_spheres = Sphere_GPU[]
    gpu_materials = OpenPBRMaterial_GPU[]
    mat_map = Dict{OpenPBRMaterial, UInt32}() # To map Julia material objects to an index
    material_idx_counter::UInt32 = 0

    for hittable_obj in scene
        if isa(hittable_obj, Sphere)
            obj_material = hittable_obj.material
            mat_id::UInt32 = 0
            if haskey(mat_map, obj_material)
                mat_id = mat_map[obj_material]
            else
                mat_id = material_idx_counter
                mat_map[obj_material] = mat_id
                # Convert Julia OpenPBRMaterial to OpenPBRMaterial_GPU (simplified version)
                gpu_mat = OpenPBRMaterial_GPU(
                    to_vec3f_gpu(obj_material.base_color),
                    Float32(obj_material.base_metalness),
                    Float32(obj_material.specular_roughness),
                    Float32(obj_material.specular_ior),
                    to_vec3f_gpu(obj_material.emission_color),
                    Float32(obj_material.emission_luminance)
                )
                push!(gpu_materials, gpu_mat)
                material_idx_counter += 1
            end
            push!(gpu_spheres, Sphere_GPU(to_vec3f_gpu(hittable_obj.center), Float32(hittable_obj.radius), mat_id))
        else
            println("Warning: Skipping non-Sphere object in scene for GPU rendering.")
        end
    end
    
    if isempty(gpu_spheres)
        println("No spheres to render on GPU. Returning black image.")
        return zeros(RGB{Float64}, height, width)
    end
    if isempty(gpu_materials) # Should not happen if there are spheres with materials
        println("Warning: No materials for GPU. This might cause issues.")
        # Add a default material to prevent crashes if kernel expects at least one
        push!(gpu_materials, OpenPBRMaterial_GPU(Vec3f_GPU(0.5,0.5,0.5),0.0,0.5,1.5,Vec3f_GPU(0,0,0),0.0))
    end

    println("Scene data prepared: $(length(gpu_spheres)) spheres, $(length(gpu_materials)) materials.")

    # Create Metal Buffers
    output_colors_buffer = Metal.MTLBuffer(Vec3f_GPU, device, num_pixels, Metal.MTLResourceStorageModeManaged)
    spheres_buffer = Metal.MTLBuffer(device, gpu_spheres, Metal.MTLResourceStorageModeManaged)
    materials_buffer = Metal.MTLBuffer(device, gpu_materials, Metal.MTLResourceStorageModeManaged)
    
    # Camera buffer (single struct)
    camera_buffer = Metal.MTLBuffer(device, [camera_gpu], Metal.MTLResourceStorageModeManaged) # Pass as an array of one

    # Scalar parameters need to be in buffers too
    image_width_buffer = Metal.MTLBuffer(device, [UInt32(width)], Metal.MTLResourceStorageModeManaged)
    image_height_buffer = Metal.MTLBuffer(device, [UInt32(height)], Metal.MTLResourceStorageModeManaged)
    num_samples_buffer = Metal.MTLBuffer(device, [UInt32(samples_per_pixel)], Metal.MTLResourceStorageModeManaged)
    max_depth_buffer = Metal.MTLBuffer(device, [UInt32(max_depth_render)], Metal.MTLResourceStorageModeManaged)
    num_spheres_buffer = Metal.MTLBuffer(device, [UInt32(length(gpu_spheres))], Metal.MTLResourceStorageModeManaged)

    println("Metal buffers created and populated.")

    command_queue = Metal.MTLCommandQueue(device)
    command_buffer = Metal.MTLCommandBuffer(command_queue)
    compute_encoder = Metal.MTLComputeCommandEncoder(command_buffer)

    Metal.set_compute_pipeline_state!(compute_encoder, pipeline_state)
    Metal.set_buffer!(compute_encoder, output_colors_buffer, 0, 0)
    Metal.set_buffer!(compute_encoder, spheres_buffer, 0, 1)
    Metal.set_buffer!(compute_encoder, materials_buffer, 0, 2)
    Metal.set_buffer!(compute_encoder, camera_buffer, 0, 3)
    Metal.set_buffer!(compute_encoder, image_width_buffer, 0, 4)
    Metal.set_buffer!(compute_encoder, image_height_buffer, 0, 5)
    Metal.set_buffer!(compute_encoder, num_samples_buffer, 0, 6)
    Metal.set_buffer!(compute_encoder, max_depth_buffer, 0, 7)
    Metal.set_buffer!(compute_encoder, num_spheres_buffer, 0, 8)

    # Dispatch threads in a 2D grid
    threads_per_grid = MTLSize(width, height, 1)
    
    # Determine appropriate threadgroup size (e.g. 16x16 or from pipeline_state)
    # pipeline_state.maxTotalThreadsPerThreadgroup
    # pipeline_state.threadExecutionWidth
    # For simplicity, using a fixed common size. MaxTotalThreadsPerThreadgroup can be very large.
    # A good 2D size depends on threadExecutionWidth, often 16x16 or 32x8 etc.
    tg_width = 16 # pipeline_state.threadExecutionWidth might be a good guide, often 32 or 64
    tg_height = 16 # Ensure tg_width * tg_height <= pipeline_state.maxTotalThreadsPerThreadgroup
    
    # Cap threadgroup dimensions by maxTotalThreadsPerThreadgroup if fixed size is too large
    if tg_width * tg_height > pipeline_state.maxTotalThreadsPerThreadgroup && pipeline_state.maxTotalThreadsPerThreadgroup > 0
        # Simple scaling, might not be optimal
        scale_factor = sqrt(pipeline_state.maxTotalThreadsPerThreadgroup / (tg_width * tg_height))
        tg_width = max(1, Int(floor(tg_width * scale_factor)))
        tg_height = max(1, Int(floor(tg_height * scale_factor)))
    elseif pipeline_state.maxTotalThreadsPerThreadgroup == 0 # Should not happen with valid pipeline
        println("Warning: maxTotalThreadsPerThreadgroup is 0. Using 16x16 threadgroup.")
        tg_width = 16; tg_height = 16;
    end

    threads_per_threadgroup = MTLSize(tg_width, tg_height, 1)
    
    println("Dispatching 2D threads: grid=($width, $height), group=($tg_width, $tg_height)")
    Metal.dispatch_threads!(compute_encoder, threads_per_grid, threads_per_threadgroup)

    Metal.end_encoding!(compute_encoder)
    Metal.commit!(command_buffer)
    println("Command buffer committed. Waiting for completion...")
    Metal.wait_until_completed!(command_buffer)
    println("GPU execution completed.")

    gpu_colors_ptr = Metal.contents(output_colors_buffer)
    gpu_colors_array = unsafe_wrap(Vector{Vec3f_GPU}, gpu_colors_ptr, num_pixels)

    img = zeros(RGB{Float64}, height, width)
    for r_idx in 1:height # Julia is 1-indexed, iterates columns first for 2D array access
        for c_idx in 1:width
            # Kernel used gid.y for row, gid.x for column
            # Flat index in kernel was gid.y * image_width + gid.x
            # So, gpu_colors_array index should be (r_idx-1)*width + (c_idx-1) + 1 for 1-based Julia
            flat_idx_julia = (r_idx - 1) * width + c_idx
            if flat_idx_julia <= num_pixels
                 gpu_color = gpu_colors_array[flat_idx_julia]
                 # Image in Julia typically has y-axis inverted if (0,0) is top-left for gid
                 # Metal gid.(0,0) is top-left. Julia Images.jl (0,0) is top-left, access is img[y,x]
                 img[r_idx, c_idx] = RGB{Float64}(gpu_color.x, gpu_color.y, gpu_color.z)
            end
        end
    end
    
    println("Image data retrieved from GPU and processed. Exiting render_gpu.")
    return img
end

# CPU Rendering
function render_cpu(scene::Vector{Hittable}, width::Int, height::Int, samples::Int)
    rng = RandomDevice()
    img = zeros(RGB{Float64}, height, width)
    bvh = BVHNode(scene, 1, length(scene) + 1)
    
    for j in 1:height, i in 1:width
        spectral_color = zeros(length(WAVELENGTHS))
        for s in 1:samples
            u = (i + rand(rng)) / width
            v = (j + rand(rng)) / height
            ray = Ray(Vec3(0,0,10), normalize(Vec3(u-0.5, v-0.5, -1)))
            wavelength = WAVELENGTHS[rand(rng, 1:length(WAVELENGTHS))]
            stokes = StokesVector([1.0, 0.0, 0.0, 0.0])  # Unpolarized
            spd = ray_color(ray, bvh, 50, rng, wavelength, stokes)
            spectral_color += spd.values
        end
        spectral_color /= samples
        # Convert spectral to RGB (simplified)
        rgb = RGB(mean(spectral_color[1:20]), mean(spectral_color[21:40]), mean(spectral_color[41:end]))
        img[height-j+1, i] = clamp.(rgb, 0, 1)
    end
    return img
end

# Main Function
function main()
    println("Main function started.")
    img_width, img_height, samples_per_pixel_main, max_depth_main = 800, 600, 100, 50
    # img_width, img_height, samples_per_pixel_main, max_depth_main = 200, 150, 20, 10 # Smaller for faster testing

    # Camera setup
    lookfrom = Vec3(0,1,10) # Adjusted camera position for a better view of a sphere at (0,0,-1)
    lookat = Vec3(0,0,0)
    vup = Vec3(0,1,0)
    vfov = 20.0
    aspect_ratio = Base.:/(Float64(img_width), img_height)
    aperture = 0.1 # Small aperture for some depth of field
    focus_dist = length(lookfrom - Vec3(0,0,-1)) # Focus on the sphere

    cam_gpu = setup_camera(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist)

    # Scene definition
    scene = Hittable[
        Sphere(Vec3(0,0,-1), 0.5, OpenPBRMaterial( # A central diffuse sphere
            1.0, Vec3(0.7,0.3,0.3), 0.0, 0.0, 1.0, Vec3(1,1,1), 0.0, 0.0, 1.5,
            0.0, Vec3(1,1,1), 0.0, Vec3(0,0,0), 0.0, 0.0, 20.0,
            0.0, Vec3(0.8,0.8,0.8), 1.0, Vec3(1,0.5,0.25), 0.0,
            0.0, Vec3(1,1,1), 0.0, 0.0, 1.6, 1.0,
            0.0, Vec3(1,1,1), 0.5,
            0.0, Vec3(0,0,0),
            0.0, 0.5, 1.4,
            1.0, false
        )),
        Sphere(Vec3(0,-100.5,-1), 100.0, OpenPBRMaterial( # A large ground sphere
            1.0, Vec3(0.5,0.5,0.5), 0.0, 0.0, 1.0, Vec3(1,1,1), 0.0, 0.0, 1.5,
            0.0, Vec3(1,1,1), 0.0, Vec3(0,0,0), 0.0, 0.0, 20.0,
            0.0, Vec3(0.8,0.8,0.8), 1.0, Vec3(1,0.5,0.25), 0.0,
            0.0, Vec3(1,1,1), 0.0, 0.0, 1.6, 1.0,
            0.0, Vec3(1,1,1), 0.5,
            0.0, Vec3(0,0,0),
            0.0, 0.5, 1.4,
            1.0, false
        ))
    ]
    println("Scene created with $(length(scene)) objects. Calling render_gpu...")
    img = render_gpu(scene, img_width, img_height, samples_per_pixel_main, max_depth_main, cam_gpu)
    println("render_gpu finished. Saving image...")
    save("output.png", img)
    println("Image saved as output.png. Main function finished.")
end

main()