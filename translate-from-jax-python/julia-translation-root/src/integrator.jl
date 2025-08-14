# Integrator for spectral rendering with differentiability support

using LinearAlgebra
using StaticArrays
using Random
using Enzyme
using EnzymeCore
using EnzymeCore: EnzymeRules
import Random: MersenneTwister # Ensure MersenneTwister is explicitly available for the rule
EnzymeCore.EnzymeRules.inactive(::typeof(rand), ::Random.MersenneTwister, ::Type{Float32}, args...) = nothing
using Base.Threads: @threads, nthreads, Atomic, atomic_add! # Explicitly what we need
using Printf # For @printf

# Constants
const RAY_EPSILON = Float32(1e-4)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

"""
    texture_lookup_bilinear(texture::Array{Float32}, u::Float32, v::Float32)

Perform bilinear interpolation on a texture.
Texture dimensions should be (height, width, channels).
u, v are in [0, 1] range, with (0,0) at the top-left.
"""
function texture_lookup_bilinear(texture::Array{Float32}, u::Float32, v::Float32)
    h, w, c = size(texture)
    
    # Convert normalized coords to pixel coords
    x = u * w - Float32(0.5)
    y = v * h - Float32(0.5)
    
    # Get integer pixel coords and fractional parts
    x0 = floor(Int, x)
    y0 = floor(Int, y)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Fractional parts for interpolation weights
    wx = x - x0
    wy = y - y0
    
    # Clamp coordinates to texture bounds
    x0 = clamp(x0, 1, w)
    y0 = clamp(y0, 1, h)
    x1 = clamp(x1, 1, w)
    y1 = clamp(y1, 1, h)
    
    # Sample the four neighboring pixels
    p00 = texture[y0, x0, :]
    p10 = texture[y0, x1, :]
    p01 = texture[y1, x0, :]
    p11 = texture[y1, x1, :]
    
    # Bilinear interpolation
    # Interpolate along x
    interp_x0 = (Float32(1.0) - wx) * p00 + wx * p10
    interp_x1 = (Float32(1.0) - wx) * p01 + wx * p11
    # Interpolate along y
    result = (Float32(1.0) - wy) * interp_x0 + wy * interp_x1
    
    return result
end

"""
    texture_lookup_udim_bilinear(udim_texture_tiles::Dict{Int, Array{Float32}}, uv::SVector{2, Float32}, default_spd::Vector{Float32})

Perform UDIM texture lookup with bilinear interpolation.
"""
function texture_lookup_udim_bilinear(udim_texture_tiles::Dict{Int, Array{Float32}}, uv::SVector{2, Float32}, default_spd::Vector{Float32})
    u, v = uv
    
    # Calculate target UDIM tile index and fractional UVs
    tile_u_idx = floor(Int, u)
    tile_v_idx = floor(Int, v)
    tile_u_idx = max(0, tile_u_idx)
    tile_v_idx = max(0, tile_v_idx)
    tile_index = 1001 + tile_u_idx + 10 * tile_v_idx
    
    frac_u = u - tile_u_idx
    frac_v = v - tile_v_idx
    
    # Check if the tile exists in our dictionary
    if haskey(udim_texture_tiles, tile_index)
        return texture_lookup_bilinear(udim_texture_tiles[tile_index], frac_u, frac_v)
    else
        # Return default SPD if tile not found
        return default_spd
    end
end

# -----------------------------------------------------------------------------
# Path Tracing Functions
# -----------------------------------------------------------------------------

"""
    direct_lighting(hit_point::Point3f, normal::Norm3f, material::Material, 
                   wo::Vec3f, # Outgoing direction (towards camera/previous path segment)
                   directional_lights::Vector{DirectionalLight}, 
                   point_lights::Vector{PointLight}, 
                   world_objects::HittableList, # For shadow rays (objects)
                   ground_plane::Plane)       # For shadow rays (ground)

Calculate direct lighting contribution (as Stokes vectors) from all lights at a hit point.
"""
function direct_lighting(hit_point::Point3f, normal::Norm3f, material::Material, 
                        wo::Vec3f,
                        directional_lights::Vector{DirectionalLight},
                        point_lights::Vector{PointLight},
                        world_objects::HittableList, 
                        ground_plane::Plane)

    accumulated_direct_stokes_spectral = [StokesVector(Float32(0.0),Float32(0.0),Float32(0.0),Float32(0.0)) for _ in 1:N_WAVELENGTHS]
    shadow_world_list = Hittable[world_objects, ground_plane]
    shadow_world = HittableList(shadow_world_list)

    # Directional Lights
    for light in directional_lights
        light_dir_wi = light.direction # Direction *to* the light source from the point
        cos_theta_i = max(Float32(0.0), dot(normal, light_dir_wi))

        if cos_theta_i > Float32(1e-6)
            shadow_origin = hit_point + normal * RAY_EPSILON
            shadow_ray = Ray(shadow_origin, light_dir_wi, RAY_EPSILON, Float32(1.0e10))
            shadow_hit_record = hit(shadow_world, shadow_ray, RAY_EPSILON, Float32(1.0e10))

            if shadow_hit_record.hit_flag == 0 # If not in shadow
                # Initial light Stokes vector (unpolarized)
                light_initial_stokes_spectral = [StokesVector(light.spd[i], Float32(0.0),Float32(0.0),Float32(0.0)) for i in 1:N_WAVELENGTHS]

                # BSDF Mueller matrix for this light interaction.
                # For Lambertian, M_bsdf = (material.diffuse/pi) * material.mueller_matrix (if material.mueller_matrix is normalized depolarizer)
                # Or, if material.mueller_matrix *includes* albedo in S00, then just use it.
                # Let's assume material.mueller_matrix is the polarizing part, and diffuse is energy.
                # The BSDF is (diffuse/pi). The Mueller matrix is material.mueller_matrix.
                for i in 1:N_WAVELENGTHS
                    # M_interaction = (material.diffuse[i] / Float32(π)) * material.mueller_matrix[i] 
                    # This is more aligned with typical BRDF * Mueller matrix formulation.
                    # Where material.mueller_matrix[i] could be DEPOLARIZER_MUELLER.
                    # The material.mueller_matrix is already wavelength-dependent.
                    # If material.mueller_matrix[i] is defined as (albedo_pol_effect * PolMatrix), then diffuse isn't needed again.
                    # Given our setup, material.mueller_matrix is DEPOLARIZER for diffuse materials.
                    # The energy term is material.diffuse[i] / pi.
                    energy_term = material.diffuse[i] / Float32(π)
                    effective_mueller_for_bsdf = material.mueller_matrix[i] .* energy_term # Scale matrix by scalar

                    reflected_stokes = effective_mueller_for_bsdf * light_initial_stokes_spectral[i]
                    accumulated_direct_stokes_spectral[i] += reflected_stokes * cos_theta_i
                end
            end
        end
    end

    # Point Lights
    for light in point_lights
        to_light_vec = light.position - hit_point
        distance_squared = dot(to_light_vec, to_light_vec)
        distance = sqrt(distance_squared)
        light_dir_wi = normalize_safe(to_light_vec)
        cos_theta_i = max(Float32(0.0), dot(normal, light_dir_wi))

        if cos_theta_i > Float32(1e-6)
            shadow_origin = hit_point + normal * RAY_EPSILON
            # Shadow ray tmax is distance to light
            shadow_ray = Ray(shadow_origin, light_dir_wi, RAY_EPSILON, Float32(distance - RAY_EPSILON)) 
            shadow_hit_record = hit(shadow_world, shadow_ray, RAY_EPSILON, Float32(distance - RAY_EPSILON))

            if shadow_hit_record.hit_flag == 0 # If not in shadow
                falloff = Float32(1.0) / max(distance_squared, Float32(1e-6))
                light_intensity_at_point = light.spd .* falloff
                light_initial_stokes_spectral = [StokesVector(light_intensity_at_point[i], Float32(0.0),Float32(0.0),Float32(0.0)) for i in 1:N_WAVELENGTHS]

                for i in 1:N_WAVELENGTHS
                    energy_term = material.diffuse[i] / Float32(π)
                    effective_mueller_for_bsdf = material.mueller_matrix[i] .* energy_term
                    
                    reflected_stokes = effective_mueller_for_bsdf * light_initial_stokes_spectral[i]
                    accumulated_direct_stokes_spectral[i] += reflected_stokes * cos_theta_i
                end
            end
        end
    end

    return accumulated_direct_stokes_spectral
end

"""
    sample_bsdf(material::Material, normal::Vec3f, wo::Vec3f, rng::AbstractRNG)

Sample a direction from the BSDF and return the sampled direction, PDF, 
the BSDF's Mueller matrix for the interaction, and the cosine term.
Assumes Lambertian reflection for now, using the material's diffuse properties and Mueller matrix.
"""
function sample_bsdf(material::Material, normal::Vec3f, wo::Vec3f, rng::AbstractRNG)
    # Forcing diffuse path (Lambertian reflection) to match Python's indirect bounces.
    # Sample direction using cosine-weighted hemisphere sampling.
    direction_wi = random_cosine_direction(normal, rng)
    
    # PDF for cosine-weighted hemisphere sampling is cos(theta) / pi.
    # Ensure normal and direction_wi are normalized for dot product.
    cos_theta_wi = max(Float32(0.0), dot(normal, direction_wi))
    pdf = cos_theta_wi / Float32(π)
    
    if pdf < Float32(1e-6) # Avoid division by zero or tiny PDFs
        # Return a zero Mueller matrix or handle degenerate case
        # A zero Mueller matrix will effectively terminate the path segment for this wavelength.
        zero_mueller_spectral = [SMatrix{4,4,Float32}(zeros(Float32,4,4)) for _ in 1:N_WAVELENGTHS]
        return direction_wi, Float32(1e-6), zero_mueller_spectral, cos_theta_wi
    end

    # For a Lambertian surface, the BSDF is material.diffuse / pi.
    # The Mueller matrix for an ideal Lambertian diffuse surface that also depolarizes is:
    # (material.diffuse[i] / pi) * DEPOLARIZER_MUELLER
    # However, the material.mueller_matrix field *already* should represent the interaction.
    # If material.mueller_matrix is DEPOLARIZER_MUELLER_SPECTRAL, then we scale it by (diffuse_albedo/pi).
    # This interpretation means material.mueller_matrix is the *polarizing* part, and diffuse is the *color/energy* part.
    
    # Let's assume material.mueller_matrix ALREADY accounts for the albedo scaling for its S00 component implicitly
    # OR that it represents the change in polarization state, and energy is handled separately.
    # For simplicity, let's say the provided material.mueller_matrix is what should be used directly.
    # The scaling by (albedo/pi) * (cos_theta/pdf) is the overall throughput scaling factor.

    # The returned Mueller matrix should be the one intrinsic to the BSDF interaction (material property).
    bsdf_interaction_mueller_spectral = material.mueller_matrix 
    # This assumes material.mueller_matrix is set up to be, e.g., DEPOLARIZER_MUELLER_SPECTRAL for diffuse.
    # The energy scaling (diffuse_albedo/pi) will be part of the weight in trace_path.

    return direction_wi, pdf, bsdf_interaction_mueller_spectral, cos_theta_wi
end

"""
    trace_path(initial_ray::Ray, world_objects::HittableList, ground_plane::Plane,
              directional_lights::Vector{DirectionalLight}, point_lights::Vector{PointLight},
              materials::Vector{Material}, background_spd::Spectrum,
              max_depth::Int, rng::AbstractRNG,
              max_depth_override::Union{Nothing,Int}=nothing)

Trace a path through the scene using polarimetric path tracing.
Returns the accumulated Stokes vector (per wavelength) for this path.
"""
function trace_path(initial_ray::Ray, world_objects::HittableList, ground_plane::Plane,
                   directional_lights::Vector{DirectionalLight}, point_lights::Vector{PointLight},
                   materials::Vector{Material}, background_spd::Spectrum,
                   max_depth::Int, rng::AbstractRNG,
                   max_depth_override::Union{Nothing,Int}=nothing)
    
    # Initialize accumulated Stokes vector for this path (light reaching camera)
    accumulated_stokes_spectral = [StokesVector(Float32(0.0),Float32(0.0),Float32(0.0),Float32(0.0)) for _ in 1:N_WAVELENGTHS]
    
    # Initialize path Mueller matrix (cumulative transformation from camera to current point)
    path_mueller_spectral = [IDENTITY_MUELLER for _ in 1:N_WAVELENGTHS]
    
    current_ray = initial_ray
    path_completed_max_depth_actively = true

    # Use override if provided
    current_max_depth = isnothing(max_depth_override) ? max_depth : max_depth_override

    for depth in 0:current_max_depth-1 # Use current_max_depth
        obj_hit_record = hit(world_objects, current_ray, RAY_EPSILON, Float32(1.0e10)) 
        plane_hit_record = hit(ground_plane, current_ray, RAY_EPSILON, Float32(1.0e10)) 
        final_hit_record = HitRecord() # Default miss

        if obj_hit_record.hit_flag == 1 && plane_hit_record.hit_flag == 1
            final_hit_record = obj_hit_record.t < plane_hit_record.t ? obj_hit_record : plane_hit_record
        elseif obj_hit_record.hit_flag == 1
            final_hit_record = obj_hit_record
        elseif plane_hit_record.hit_flag == 1
            final_hit_record = plane_hit_record
        end
        
        if final_hit_record.hit_flag == 0
            # Ray missed. Add background contribution, transformed by the path Mueller matrix.
            # Assuming background is unpolarized.
            for i in 1:N_WAVELENGTHS
                background_stokes_unpolarized = StokesVector(background_spd[i], Float32(0.0),Float32(0.0),Float32(0.0))
                accumulated_stokes_spectral[i] += path_mueller_spectral[i] * background_stokes_unpolarized
            end
            path_completed_max_depth_actively = false
            break
        end
        
        mat_id = final_hit_record.material_id
        if mat_id < 1 || mat_id > length(materials)
            path_completed_max_depth_actively = false
            break 
        end
        material = materials[mat_id]
        
        # Add emission from the hit surface, transformed by the path Mueller matrix.
        # Assuming emission is unpolarized.
        if sum(material.emission) > 1e-5 # Check if there is any emission
            for i in 1:N_WAVELENGTHS
                emission_stokes_unpolarized = StokesVector(material.emission[i], Float32(0.0),Float32(0.0),Float32(0.0))
                accumulated_stokes_spectral[i] += path_mueller_spectral[i] * emission_stokes_unpolarized
            end
        end
        
        # Direct lighting contribution (Next Event Estimation)
        # wo is direction from hit point to previous vertex (or camera)
        wo_from_hit = normalize(-current_ray.direction)
        direct_light_contrib_stokes = direct_lighting(final_hit_record.position, final_hit_record.normal, material,
                                                     wo_from_hit, directional_lights, point_lights, 
                                                     world_objects, ground_plane)
        for i in 1:N_WAVELENGTHS
            # Path Mueller matrix transforms the light that would be scattered *towards the camera* by direct illumination.
            accumulated_stokes_spectral[i] += path_mueller_spectral[i] * direct_light_contrib_stokes[i]
        end

        # BSDF Sampling for Indirect Lighting ( seuraava pomppu )
        # wi is new direction from hit point, wo_from_hit is old incoming from POV of BSDF
        wi_new_direction, pdf_bsdf, bsdf_mueller_interaction_spectral, cos_theta_wi = 
            sample_bsdf(material, final_hit_record.normal, wo_from_hit, rng)
        
        if pdf_bsdf < Float32(1e-7) || cos_theta_wi < Float32(1e-7)
            path_completed_max_depth_actively = false
            break
        end
        
        # Update path Mueller matrix for the next segment
        # PathMueller_new = (BSDF_Mueller * albedo/pi * cos/pdf) * PathMueller_old
        # Here, sample_bsdf returns the BSDF_Mueller (material.mueller_matrix).
        # The energy scaling is material.diffuse[i]/pi * cos_theta_wi / pdf_bsdf.
        # Since pdf = cos_theta_wi / pi for Lambertian, this simplifies to material.diffuse[i].
        
        for i in 1:N_WAVELENGTHS
            # bsdf_mueller_interaction_spectral[i] is material.mueller_matrix[i] (e.g., DEPOLARIZER_MUELLER)
            # The weight for Lambertian is (material.diffuse[i]/pi) * cos_theta_wi / pdf.
            # If pdf = cos_theta_wi/pi, then weight is material.diffuse[i].
            lambertian_weight = material.diffuse[i] # pdf cancels cos_theta_wi/pi
            
            # Effective Mueller for this bounce (polarization change * energy scaling)
            bounce_transform_mueller = bsdf_mueller_interaction_spectral[i] .* lambertian_weight
            
            path_mueller_spectral[i] = bounce_transform_mueller * path_mueller_spectral[i]
        end
        
        # Russian Roulette
        # Use S00 component (intensity throughput) of the path Mueller matrix average
        # A non-zero path_mueller_spectral[i][1,1] means the path can still transmit light intensity.
        path_s00_avg = sum(path_mueller_spectral[i][1,1] for i in 1:N_WAVELENGTHS) / N_WAVELENGTHS
        
        if path_s00_avg < Float32(1e-4) # If throughput is too low
            # Absorption probability for Russian Roulette
            # q = max(0.05, 1.0 - path_s00_avg) # Example, can be tuned
            # For simplicity, if average S00 is low, just terminate, or use a fixed probability. Python used sum(throughput) < 1e-4.
            # For Mueller matrices, S00 is the total intensity transmittance for unpolarized light.
             path_completed_max_depth_actively = false
             break
        end
        # Optional: Compensate throughput if surviving RR: path_mueller_spectral ./= (1.0 - q) if RR used and survived.
        # For now, simple termination if S00 is low.

        current_ray = Ray(final_hit_record.position + final_hit_record.normal * RAY_EPSILON, wi_new_direction, Float32(1.0e-4), Float32(1.0e10))
        # The stokes_vector of this new ray is not directly used by this path_mueller accumulation method.
        nothing # Added to potentially help parsing before the loop end
    end # End of path tracing loop for depth

    # If path completed by reaching max_depth (not by miss or RR absorption earlier)
    # and the last ray would have escaped to infinity, it might hit background.
    # This case is complex: the last segment of path_mueller_spectral would transform background light.
    # However, the loop already handles background if final_hit_record.hit_flag == 0.
    # If path_completed_max_depth_actively is true, it means we did all bounces.
    # The last bounce updated path_mueller_spectral. If that last ray were to hit background,
    # it would be: path_mueller_spectral_at_max_depth * background_stokes.
    # This is implicitly handled if the next iteration (max_depth+1) would miss.
    # The current structure adds emission/direct at the current hit, then updates path_mueller for *next* segment.
    # So, if loop finishes due to depth, the accumulated_stokes is complete for contributions up to max_depth-1 bounces
    # and interactions at the vertex of depth max_depth-1.
    
    return accumulated_stokes_spectral
end

"""
    render_pixel(x::Int, y::Int, width::Int, height::Int, camera::Camera, 
                 world_objects::HittableList, ground_plane::Plane,
                 directional_lights::Vector{DirectionalLight}, point_lights::Vector{PointLight},
                 materials::Vector{Material}, background_spd::Spectrum,
                 samples_per_pixel::Int, max_depth::Int, rng::AbstractRNG,
                 ad_mode::Bool=false)

Render a single pixel with multi-sample anti-aliasing, returning the average Stokes vector.
"""
function render_pixel(x::Int, y::Int, width::Int, height::Int, camera::Camera, 
                     world_objects::HittableList, ground_plane::Plane,
                     directional_lights::Vector{DirectionalLight}, point_lights::Vector{PointLight},
                     materials::Vector{Material}, background_spd::Spectrum,
                     samples_per_pixel::Int, max_depth::Int, rng::AbstractRNG,
                     ad_mode::Bool=false)
    
    # Initialize accumulated Stokes vector for this pixel (sum over samples)
    accumulated_pixel_stokes_spectral = [StokesVector(Float32(0.0),Float32(0.0),Float32(0.0),Float32(0.0)) for _ in 1:N_WAVELENGTHS]
    
    # Define wrapper function that Enzyme won't differentiate through
    const_rand_f32 = (rng) -> rand(rng, Float32)
    
    for _ in 1:samples_per_pixel
        # Use direct rand calls - Enzyme will handle these within autodiff context appropriately
        # given our Const annotations in the ∇render_enzyme function
        u_rand = const_rand_f32(rng)
        v_rand = const_rand_f32(rng)
        
        u = (x - 1 + u_rand) / (width - 1)
        v = (y - 1 + v_rand) / (height - 1)
        ray = generate_ray(camera, u, v, rng)
        
        # Determine max_depth_override for trace_path based on ad_mode
        trace_max_depth_override = ad_mode ? 1 : nothing

        # Trace path, returns accumulated Stokes vector for the path
        path_stokes_spectral = trace_path(ray, world_objects, ground_plane,
                                           directional_lights, point_lights, 
                                           materials, background_spd, max_depth, rng,
                                           trace_max_depth_override) # Pass override
        
        # Accumulate Stokes vectors
        accumulated_pixel_stokes_spectral = accumulated_pixel_stokes_spectral .+ path_stokes_spectral
    end
    
    # Average Stokes vector over samples
    avg_stokes_spectral = accumulated_pixel_stokes_spectral ./ Float32(samples_per_pixel)
    
    # Apply exposure based on S0 (intensity) component before returning full Stokes
    # Exposure calculation remains the same, but applied to the Stokes vector. Usually only S0 is scaled.
    exposure_factor = (camera.shutter_speed * (camera.iso / Float32(100.0))) / camera.f_number^2
    
    # Apply exposure to all components? Or just S0? Common practice is to scale S0.
    # Let's scale only S0 (index 1) for physical accuracy of intensity scaling.
    exposed_stokes_spectral = [
        StokesVector(stokes[1] * exposure_factor, stokes[2], stokes[3], stokes[4]) 
        for stokes in avg_stokes_spectral
    ]
    # If scaling all: exposed_stokes_spectral = avg_stokes_spectral .* exposure_factor

    return exposed_stokes_spectral # Return Vector{StokesVector}
end

"""
    render_image(camera::Camera, world_objects::HittableList, ground_plane::Plane,
                directional_lights::Vector{DirectionalLight}, point_lights::Vector{PointLight},
                materials::Vector{Material}, background_spd::Spectrum,
                width::Int, height::Int, samples_per_pixel::Int, max_depth::Int, 
                rng::AbstractRNG = Random.GLOBAL_RNG; # Add rng argument
                progress_update::Bool=true,
                force_single_thread::Bool=false) # New argument

Render a complete image using polarimetric path tracing.
Returns an image buffer containing Stokes vectors per pixel [height, width, N_WAVELENGTHS, 4].
"""
function render_image(camera::Camera, 
                     world_objects::HittableList, ground_plane::Plane,
                     directional_lights::Vector{DirectionalLight}, point_lights::Vector{PointLight},
                     materials::Vector{Material}, background_spd::Spectrum,
                     width::Int, height::Int, samples_per_pixel::Int, max_depth::Int, 
                     rng::AbstractRNG = Random.GLOBAL_RNG; # Add rng argument
                     progress_update::Bool=true,
                     force_single_thread::Bool=false) # New argument
    
    pixel_stokes_array = Array{Vector{StokesVector}}(undef, height, width)
    total_pixels = width * height
    pixels_rendered_atomic = Atomic{Int}(0) # Atomic counter for pixels

    if progress_update && !force_single_thread # Only print nthreads if actually threading
        println("Rendering $total_pixels pixels using $(nthreads()) threads...")
    elseif progress_update && force_single_thread
        println("Rendering $total_pixels pixels using 1 thread (AD active)...")
    end

    if !force_single_thread
        Threads.@threads for k in 1:(width * height)
            # Convert flat index k to 2D indices (j for row, i for col)
            row_j = div(k - 1, width) + 1 
            col_i = mod(k - 1, width) + 1

            # Create RNG for this pixel, seeded with the flat pixel index for determinism
            rng_for_pixel = MersenneTwister(abs(hash(k)))

            pixel_stokes_array[row_j, col_i] = render_pixel(
                col_i, row_j, width, height, camera, # Pass col_i, row_j
                world_objects, ground_plane,
                directional_lights, point_lights,
                materials, background_spd,
                samples_per_pixel, max_depth, rng_for_pixel,
                force_single_thread # Pass force_single_thread as ad_mode
            )

            if progress_update
                current_pixels_done = atomic_add!(pixels_rendered_atomic, 1) + 1
                
                print_interval = max(1, div(total_pixels, 200)) 
                if mod(current_pixels_done, print_interval) == 0 || current_pixels_done == total_pixels
                    percentage_done = (current_pixels_done / total_pixels) * 100.0
                    @printf("\rRendering: %.2f%% complete (%d/%d pixels)", percentage_done, current_pixels_done, total_pixels)
                end
            end
        end # End Threads.@threads
    else # force_single_thread is true
        for k in 1:(width * height) # Plain single-threaded loop
            row_j = div(k - 1, width) + 1 
            col_i = mod(k - 1, width) + 1
            rng_for_pixel = MersenneTwister(abs(hash(k))) # Still use per-pixel RNG for consistency
            pixel_stokes_array[row_j, col_i] = render_pixel(
                col_i, row_j, width, height, camera,
                world_objects, ground_plane,
                directional_lights, point_lights,
                materials, background_spd,
                samples_per_pixel, max_depth, rng_for_pixel,
                force_single_thread # Pass force_single_thread as ad_mode
            )
            if progress_update
                current_pixels_done = atomic_add!(pixels_rendered_atomic, 1) + 1 
                print_interval = max(1, div(total_pixels, 200))
                if mod(current_pixels_done, print_interval) == 0 || current_pixels_done == total_pixels
                    percentage_done = (current_pixels_done / total_pixels) * 100.0
                    @printf("\rRendering: %.2f%% complete (%d/%d pixels)", percentage_done, current_pixels_done, total_pixels)
                end
            end
        end # End single-threaded loop
    end

    if progress_update # Ensure 100% is printed and newline
        @printf("\rRendering: 100.00%% complete (%d/%d pixels)\n", total_pixels, total_pixels)
        if !force_single_thread
             println("Rendering complete.") 
        else
             println("Single-threaded rendering pass complete (AD active).")
        end
    end

    # Convert the Array{Vector{StokesVector}} into the final 4D array [h, w, wavelengths, 4]
    final_stokes_image = Array{Float32}(undef, height, width, N_WAVELENGTHS, 4)
    for j in 1:height
        for i in 1:width
            stokes_spectral_vector = pixel_stokes_array[j, i]
            for k in 1:N_WAVELENGTHS
                # Assign SVector components to the slice
                final_stokes_image[j, i, k, 1] = stokes_spectral_vector[k][1] # S0
                final_stokes_image[j, i, k, 2] = stokes_spectral_vector[k][2] # S1
                final_stokes_image[j, i, k, 3] = stokes_spectral_vector[k][3] # S2
                final_stokes_image[j, i, k, 4] = stokes_spectral_vector[k][4] # S3
            end
        end
    end

    return final_stokes_image
end

# -----------------------------------------------------------------------------
# Differentiable Rendering Functions
# -----------------------------------------------------------------------------

"""
    ∇render_enzyme(camera::Camera, 
                   world::Union{HittableList, BVHNode}, # Allow BVH
                   ground_plane::Plane,
                   directional_lights::Vector{DirectionalLight},
                   point_lights::Vector{PointLight},
                   materials::Vector{Material},
                   background_spd::Spectrum,
                   width::Int, height::Int,
                   samples_per_pixel::Int, max_depth::Int,
                   target_image::Array{Float32}) # Target intensity [h, w, wavelengths]

Compute gradient of render output (intensity S0) with respect to material parameters using Enzyme.
This is a simplified version for demonstration purposes.
"""
function ∇render_enzyme(camera::Camera, 
                       world::Union{HittableList, BVHNode}, # Allow BVH
                       ground_plane::Plane,
                       directional_lights::Vector{DirectionalLight},
                       point_lights::Vector{PointLight},
                       materials::Vector{Material},
                       background_spd::Spectrum,
                       width::Int, height::Int,
                       samples_per_pixel::Int, max_depth::Int,
                       target_image::Array{Float32}) # Target intensity [h, w, wavelengths]

    # Define loss function based on MSE between rendered S0 and target intensity
    function loss_fn(
        materials_in::Vector{Material}, # Active argument
        # Const arguments
        _camera::Camera, 
        _world::Union{HittableList, BVHNode}, 
        _ground_plane::Plane,
        _directional_lights::Vector{DirectionalLight}, 
        _point_lights::Vector{PointLight}, 
        _background_spd::Spectrum,
        _width::Int, 
        _height::Int, 
        _samples_per_pixel::Int, 
        _max_depth::Int,
        _target_image::Array{Float32}
    )
        println("DEBUG: Entered loss_fn. About to call render_image (single-threaded for AD).")
        # Render image with current materials, returns full Stokes [h, w, wavelengths, 4]
        rendered_stokes = render_image(_camera, 
                                       _world, # Pass world (can be HittableList or BVH)
                                       _ground_plane,
                                       _directional_lights, _point_lights,
                                       materials_in, _background_spd, _width, _height,
                                       _samples_per_pixel, _max_depth, progress_update=false, # Maybe turn off progress for AD passes
                                       force_single_thread=true) # <--- PASS true HERE

        println("DEBUG: render_image completed in loss_fn. Calculating loss.")
        # Extract S0 component (intensity) for comparison
        rendered_s0_intensity = rendered_stokes[:, :, :, 1]
        
        # Ensure target image dimensions match rendered S0
        if size(rendered_s0_intensity) != size(_target_image)
             error("Target image dimensions mismatch: Target $(size(_target_image)), Rendered S0 $(size(rendered_s0_intensity))")
        end

        # Compute MSE loss on intensity
        diff = rendered_s0_intensity .- _target_image
        loss = sum(diff.^2) / length(_target_image) # Normalize by total number of elements
        return loss
    end

    # Compute gradient of loss with respect to materials
    # Need to handle the structure of materials (Vector{Material})
    # Enzyme should handle differentiating through the Material struct fields.
    
    # Prepare duplicated structures for Enzyme
    # d_materials = deepcopy(materials) # Primal values - not needed if passing materials directly
    d_materials_shadow = Enzyme.make_zero(materials) # Gradient storage

    # Perform autodiff
    # We need gradient w.r.t fields of materials[1], assuming only that is optimized.
    # This requires careful handling of Enzyme's interface for struct differentiation.
    # A simpler approach might be to pass only the optimizable parameters.
    
    # Let's assume loss_fn takes the full materials vector and we get gradient for all.
    # If only optimizing materials[1].diffuse: 
    # Adapt loss_fn signature or extract gradient afterwards.

    # Using Duplicated for the whole vector of materials
    println("DEBUG: About to call Enzyme.autodiff in ∇render_enzyme.")
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse), 
        loss_fn, 
        Duplicated(materials, d_materials_shadow), # Active input
        Const(camera),
        Const(world),
        Const(ground_plane),
        Const(directional_lights),
        Const(point_lights),
        Const(background_spd),
        Const(width),
        Const(height),
        Const(samples_per_pixel),
        Const(max_depth),
        Const(target_image) # Pass target_image as Const
    )
    println("DEBUG: Enzyme.autodiff call finished in ∇render_enzyme.")
    
    # The d_materials_shadow is now populated with the gradients.
    return d_materials_shadow # Return gradient structure matching materials vector
end