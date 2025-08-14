#include &lt;metal_stdlib&gt;
using namespace metal;

// Constants
constant float PI = 3.14159265359f;
constant float INF = 1e20f;
constant float EPSILON = 0.0001f;

// Struct Definitions
struct Camera_msl {
    float3 origin;
    float3 lower_left_corner;
    float3 horizontal;
    float3 vertical;
};

struct Ray_msl {
    float3 origin;
    float3 direction; // Should always be normalized
};

struct Sphere_msl {
    float3 center;
    float radius;
    uint material_index; // 0-based index into materials array
};

struct Material_msl {
    float3 albedo;
    float3 emission;
    float metallic;
    float roughness;
};

struct RNGState_msl {
    uint state;
};

struct RenderParams_msl {
    uint image_width;
    uint image_height;
    uint max_depth;
    uint current_sample_index;
    uint num_spheres;
    uint num_materials;
};

// --- Helper MSL Functions ---

// Simple LCG random number generator
// Returns a float in [0, 1)
float random_uniform(thread RNGState_msl&amp; rng_state) {
    // LCG parameters (Numerical Recipes)
    uint M = 1664525;
    uint C = 1013904223;
    rng_state.state = rng_state.state * M + C;
    return float(rng_state.state &amp; 0x00FFFFFF) / float(0x01000000); // Use lower 24 bits for ~uniformity
}

// Generate a random vector uniformly inside a unit sphere
float3 random_unit_vector(thread RNGState_msl&amp; rng_state) {
    while (true) {
        float3 p = float3(random_uniform(rng_state) * 2.0f - 1.0f,
                          random_uniform(rng_state) * 2.0f - 1.0f,
                          random_uniform(rng_state) * 2.0f - 1.0f);
        if (length_squared(p) &lt; 1.0f) {
            return normalize(p); // Normalize to get a point on the sphere surface, or just return p for inside
        }
    }
}

// Generate a cosine-weighted random direction in the hemisphere defined by 'normal'
float3 random_hemisphere_direction_cosine_weighted(thread RNGState_msl&amp; rng_state, float3 normal) {
    float r1 = random_uniform(rng_state);
    float r2 = random_uniform(rng_state);

    float phi = 2.0f * PI * r1;
    float cos_theta = sqrt(1.0f - r2); // Incorrect: this is uniform, not cosine weighted for r2. Corrected: sqrt(r2) for cos_theta, sin_theta from that.
                                      // Let's use a simpler approach: uniform sphere point + ensure it's in hemisphere
    
    // A common way: Generate point on unit disk, project to hemisphere
    float x = cos(phi) * sqrt(r2); // sqrt(r2) for uniform distribution on disk
    float y = sin(phi) * sqrt(r2);
    float z = sqrt(max(0.0f, 1.0f - x*x - y*y)); // z based on x,y on disk

    // Create orthonormal basis (U,V,W) with W = normal
    float3 w_axis = normal;
    float3 u_axis_temp = (abs(w_axis.x) > 0.1f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f));
    float3 u_axis = normalize(cross(u_axis_temp, w_axis));
    float3 v_axis = cross(w_axis, u_axis);

    return normalize(x * u_axis + y * v_axis + z * w_axis);
}

// Standard reflection function
float3 reflect(float3 incident, float3 normal) {
    return incident - 2.0f * dot(incident, normal) * normal;
}

// Ray-Sphere Intersection
// Returns smallest positive t, or INF if no hit
// Also outputs normal at hit point
struct HitRecord {
    float t;
    float3 normal;
    uint material_idx;
};

HitRecord intersect_sphere(Ray_msl r, Sphere_msl s) {
    HitRecord rec;
    rec.t = INF;

    float3 oc = r.origin - s.center;
    float a = dot(r.direction, r.direction); // Should be 1.0 if direction is normalized
    float half_b = dot(oc, r.direction);
    float c = dot(oc, oc) - s.radius * s.radius;
    float discriminant = half_b * half_b - a * c;

    if (discriminant &gt; 0.0f) {
        float root = (-half_b - sqrt(discriminant)) / a;
        if (root &gt; EPSILON) {
            rec.t = root;
            rec.normal = normalize((r.origin + rec.t * r.direction) - s.center);
            rec.material_idx = s.material_index;
            return rec;
        }
        root = (-half_b + sqrt(discriminant)) / a;
        if (root &gt; EPSILON) {
            rec.t = root;
            rec.normal = normalize((r.origin + rec.t * r.direction) - s.center);
            rec.material_idx = s.material_index;
            return rec;
        }
    }
    return rec;
}


// --- Main Path Tracing Kernel ---
kernel void path_trace(device const Sphere_msl* spheres [[buffer(0)]],
                       device const Material_msl* materials [[buffer(1)]],
                       device const Camera_msl* camera [[buffer(2)]],
                       device RNGState_msl* rng_states [[buffer(3)]],
                       device float3* output_hdr_image [[buffer(4)]],
                       constant RenderParams_msl&amp; params [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]])
{
    uint pixel_idx = gid.y * params.image_width + gid.x;

    // Bounds check for threads outside image (if grid is larger)
    if (gid.x &gt;= params.image_width || gid.y &gt;= params.image_height) {
        return;
    }

    thread RNGState_msl rng_state = rng_states[pixel_idx];

    // --- Ray Generation ---
    // Jittered UV coordinates (assuming (0,0) is top-left, positive y is down)
    // If Metal default (0,0) is bottom-left, gid.y might need inversion for v_jittered.
    // Let's assume standard image UV (0,0) top-left for now.
    float u_jittered = (float(gid.x) + random_uniform(rng_state)) / float(params.image_width);
    float v_jittered = (float(gid.y) + random_uniform(rng_state)) / float(params.image_height); 
                                   // If (0,0) is bottom-left: (float(params.image_height - 1 - gid.y) + random_uniform(rng_state)) / float(params.image_height);

    Ray_msl current_ray;
    current_ray.origin = camera[0].origin;
    current_ray.direction = normalize(camera[0].lower_left_corner +
                                    u_jittered * camera[0].horizontal +
                                    v_jittered * camera[0].vertical -
                                    camera[0].origin);

    // --- Path Tracing Loop ---
    float3 accumulated_color = float3(0.0f);
    float3 path_throughput = float3(1.0f);

    for (uint depth = 0; depth &lt; params.max_depth; ++depth) {
        float closest_t = INF;
        int hit_sphere_idx = -1; // Using int for -1 sentinel
        float3 hit_normal_temp = float3(0.0f);

        // --- Intersection ---
        for (uint s = 0; s &lt; params.num_spheres; ++s) {
            HitRecord temp_rec = intersect_sphere(current_ray, spheres[s]);
            if (temp_rec.t &lt; closest_t) {
                closest_t = temp_rec.t;
                hit_sphere_idx = int(s);
                hit_normal_temp = temp_rec.normal;
            }
        }

        // --- If No Hit ---
        if (hit_sphere_idx == -1) {
            // Simple sky color (blue gradient based on ray.direction.y)
            float t_sky = 0.5f * (current_ray.direction.y + 1.0f); // -1 to 1 -&gt; 0 to 1
            float3 sky_color = (1.0f - t_sky) * float3(1.0f, 1.0f, 1.0f) + t_sky * float3(0.5f, 0.7f, 1.0f);
            accumulated_color += path_throughput * sky_color;
            break; // End path
        }

        // --- If Hit ---
        Sphere_msl hit_sphere = spheres[hit_sphere_idx];
        Material_msl hit_material = materials[hit_sphere.material_index];
        float3 hit_point = current_ray.origin + closest_t * current_ray.direction;
        float3 hit_normal = hit_normal_temp;

        // Ensure normal faces outwards (towards the ray origin)
        if (dot(current_ray.direction, hit_normal) &gt; 0.0f) {
            hit_normal = -hit_normal;
        }

        // --- Emission ---
        accumulated_color += path_throughput * hit_material.emission;

        // --- Material Scattering ---
        float3 scatter_origin = hit_point + hit_normal * EPSILON;
        float3 scattered_direction;
        float3 attenuation_factor = hit_material.albedo;

        if (random_uniform(rng_state) &lt; hit_material.metallic) { // Metal
            scattered_direction = reflect(current_ray.direction, hit_normal);
            if (hit_material.roughness &gt; 0.0f) {
                scattered_direction = normalize(scattered_direction + hit_material.roughness * random_unit_vector(rng_state));
            }
            // Metals tint reflection: attenuation_factor is already albedo.
        } else { // Dielectric / Diffuse
            scattered_direction = random_hemisphere_direction_cosine_weighted(rng_state, hit_normal);
            // Attenuation for diffuse is albedo.
        }
        
        current_ray.origin = scatter_origin;
        current_ray.direction = scattered_direction; // Must be normalized if not already by helper
        path_throughput *= attenuation_factor;


        // --- Russian Roulette (Basic) ---
        if (depth &gt; 3) {
            float p_continue = max(path_throughput.x, max(path_throughput.y, path_throughput.z));
            p_continue = min(p_continue, 0.95f); // Cap probability
            if (random_uniform(rng_state) &gt; p_continue) {
                break; // Terminate path
            }
            path_throughput /= p_continue; // Adjust throughput for unbiasedness
        }

        // Early exit if throughput is too low
        if (max(path_throughput.x, max(path_throughput.y, path_throughput.z)) &lt; 0.01f) {
            break;
        }
    }

    // --- Store Result ---
    // The host will manage accumulation by re-reading output_hdr_image, adding new sample, then writing back,
    // or by clearing output_hdr_image for the first sample.
    // For simplicity if current_sample_index == 0, we overwrite. Otherwise, we add.
    // This means the host MUST ensure output_hdr_image is zeroed before the first sample dispatch
    // if params.current_sample_index will always be non-zero after the first.
    // A more robust way is for the kernel to *always* add, and the host to zero the buffer.
    // OR, a common pattern: output_pixel_color = accumulated_color / total_samples_per_pixel;
    // then atomic_add to a buffer.
    // Given the prompt: "accumulate it into the HDR output buffer on the GPU".
    // So, the kernel adds its contribution. The host must zero the buffer initially.
    
    // If the output_hdr_image is initialized to 0 by the host before the first sample:
    output_hdr_image[pixel_idx] += accumulated_color;


    // --- Update RNG State for next sample/frame ---
    rng_states[pixel_idx] = rng_state;
}

