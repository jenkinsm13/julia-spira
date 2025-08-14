module SpectralRenderer

using LinearAlgebra
using StaticArrays
using Images
using FileIO
using Colors
using Random
using Enzyme
using ArgParse
using GeometryBasics
using MeshIO

# Include submodules in correct order with proper includes
include("types.jl")
include("utils.jl")
include("spd_library.jl")
include("geometry.jl")
include("camera.jl")
include("integrator.jl")
include("bayer.jl") # Include the Bayer module

# Re-export types and functions from submodules
export Vec3f, Point3f, Norm3f, UV2f, Ray, point_at
export Material, HitRecord, Hittable
export DirectionalLight, PointLight
export N_WAVELENGTHS, MIN_WAVELENGTH_NM, MAX_WAVELENGTH_NM
export WAVELENGTHS_NM, DELTA_WAVELENGTH_NM, Spectrum
export AABB, surrounding_box, hit_aabb

# Export utility functions
export spectral_to_xyz, xyz_to_rgb, linear_rgb_to_srgb, rgb_to_spd_d65_approx
export normalize_safe, reflect, random_in_unit_sphere, random_unit_vector
export random_cosine_direction, intersect_ray_triangle_moller_trumbore

# Export SPD library
export create_spectrum_profile

# Export geometry types and functions
export Sphere, Triangle, Mesh, BVHNode, HittableList
export hit, bounding_box, build_bvh, load_mesh_from_obj

# Export camera functions
export Camera, generate_ray, generate_rays, apply_brown_distortion

# Export integrator functions
export render_image, render_pixel, trace_path
export direct_lighting, sample_bsdf, texture_lookup_bilinear
export ∇render_enzyme

# Main rendering functions

"""
    render_image_forward(width::Int, height::Int, samples_per_pixel::Int, max_depth::Int, obj_path::String)

Render an image using forward path tracing.
"""
function render_image_forward(;width::Int=512, height::Int=512,
                             samples_per_pixel::Int=64, max_depth::Int=5,
                             obj_path::Union{String, Nothing}=nothing,
                             focus_distance_arg::Union{Float32, Nothing}=nothing,
                             output_bayer_tiff::Bool = false)

    println("Rendering $width x $height image at $samples_per_pixel spp (Polarimetric) with focus: $focus_distance_arg...")

    # Set up the scene - matched exactly to Python implementation
    # Define materials
    materials = [
        # Material 1: Mesh material (fluorescent_yellow, treated as Lambertian)
        Material(
            diffuse=create_spectrum_profile("vegetation", Float32(1.0)), 
            emission=Spectrum(fill(Float32(0.0), N_WAVELENGTHS)), # Ensure Spectrum type
            specular=Float32(0.0),
            roughness=Float32(1.0),
            mueller_matrix=DEPOLARIZER_MUELLER_SPECTRAL # Explicitly depolarizing
        ),
        # Material 2: Checker A - Gray
        Material(
            diffuse=create_spectrum_profile("white_paint", Float32(0.18)),
            emission=Spectrum(fill(Float32(0.0), N_WAVELENGTHS)),
            specular=Float32(0.0),
            roughness=Float32(1.0),
            mueller_matrix=DEPOLARIZER_MUELLER_SPECTRAL
        ),
        # Material #3: Checker B - White
        Material(
            diffuse=create_spectrum_profile("white_paint", Float32(0.9)),
            emission=Spectrum(fill(Float32(0.0), N_WAVELENGTHS)),
            specular=Float32(0.0),
            roughness=Float32(1.0),
            mueller_matrix=DEPOLARIZER_MUELLER_SPECTRAL
        )
    ]

    # Define light sources
    directional_lights = [
        DirectionalLight(
            normalize(Vec3f(Float32(1.0), Float32(1.5), Float32(0.8))),  # Python negates [-1, -1.5, -0.8], so this matches \'to light\'
            create_spectrum_profile("daylight_noon", Float32(1.0)), # Match Python SPD scale
        ),
        DirectionalLight(
            normalize(Vec3f(Float32(-10.0), Float32(5.5), Float32(-0.5))), # Python negates [10, -5.5, 0.5]
            create_spectrum_profile("blackbody_emitter", Float32(3200.0)), # Match Python temp
        )
    ]

    point_lights = [
        PointLight(
            Vec3f(Float32(2.0), Float32(3.0), Float32(2.0)),
            create_spectrum_profile("laser_red", Float32(150.0)), # Match Python SPD scale
        )
    ]

    # Define background
    background_spd = create_spectrum_profile("daylight_noon", Float32(1.2))

    # Create geometry
    objects = Hittable[]

    # Remove ground plane (as a large sphere)
    # push!(objects, Sphere(
    #     Point3f(0.0f0, -100.5f0, 0.0f0),
    #     Float32(100.0),
    #     Int32(1)  # Gray material
    # ))

    # Define the ground plane (matches Python Y=0 plane)
    ground_plane = Plane(
        normalize_safe(Vec3f(Float32(0.0), Float32(1.0), Float32(0.0))), # Normal (Y-up)
        Float32(0.0),                                      # d (offset from origin, so Y=0)
        Int32(2),                                   # 1-based index -> materials[2] (Gray)
        Int32(3),                                   # 1-based index -> materials[3] (White)
        Float32(2.0)                                       # checker_scale (matches Python)
    )

    # Add light source sphere (if it's supposed to be there - Python doesn't have it)
    # For now, let's keep it to see its effect, but this is a discrepancy with Python forward_render.py
    # Removing this to match Python scene which has no explicit emissive sphere object.
    # push!(objects, Sphere(
    #     Point3f(0.0f0, 2.0f0, 0.0f0),
    #     Float32(0.5),
    #     Int32(3)  # Emissive material
    # ))

    # Load mesh if provided
    if obj_path !== nothing && isfile(obj_path)
        println("Loading mesh from $obj_path")
        mesh = load_mesh_from_obj(
            obj_path,
            Int32(1),  # 1-based index -> materials[1] (fluorescent_yellow)
            scale=Point3f(Float32(1.0), Float32(1.0), Float32(1.0)),
            rotation=Point3f(Float32(0.0), Float32(0.0), Float32(0.0)),
            translation=Point3f(Float32(0.0), Float32(1.0), Float32(0.0)),
            center=true,
            normalize_size=true
        )
        push!(objects, mesh)
    else
        # If no OBJ path, do not add a default sphere, to match Python (empty mesh_data)
        println("No OBJ file provided or found. Proceeding without mesh object.")
        # push!(objects, Sphere(
        #     Point3f(0.0f0, 1.0f0, 0.0f0),
        #     Float32(1.0),
        #     Int32(1)  # Fluorescent yellow material (if used, should be 1)
        # ))
    end

    # Create a hittable list for other objects (e.g., mesh, other spheres)
    world_objects = HittableList(objects)

    # Define camera - match Python version with brighter exposure
    # Calculate focal length for about 50-degree FOV
    focal_length = Float32(1671.1020503131649) # Match Python
    center_x = Float32(width / 2.0)
    center_y = Float32(height / 2.0)
    camera = Camera(
        # Intrinsics
        fx=focal_length,
        fy=focal_length, # Assuming square pixels fx=fy=f
        cx=center_x,
        cy=center_y,

        # Extrinsics - match Python camera position
        eye=Point3f(Float32(-2.5), Float32(3.0), Float32(4.5)), # Match Python
        center=Point3f(Float32(0.0), Float32(0.0), Float32(0.0)), # Changed: Point directly at mesh origin
        up=Vec3f(Float32(1.0), Float32(0.0), Float32(0.0)),    # Changed: Use X-axis as UP for 90-deg clockwise roll

        # Lens distortion to match Python
        k1=Float32(0.0050715816041100002),
        k2=Float32(-0.0024127882483701408),
        k3=Float32(-0.0012653936614391772),
        p1=Float32(-0.0005047349357100193),
        p2=Float32(0.00094860125385438975),

        # Physical Exposure - match Python
        f_number=Float32(16.0),
        shutter_speed=Float32(1/10),
        iso=Float32(200.0),
        focus_distance=focus_distance_arg # MODIFIED: Pass focus_distance_arg
    )

    # Render the image
    rendered_stokes_image = render_image(
        camera, world_objects, ground_plane, # Pass ground_plane separately
        directional_lights, point_lights,
        materials, background_spd, width, height,
        samples_per_pixel, max_depth, progress_update=true
    )

    # --- Bayer Pattern Output (Optional) ---
    if output_bayer_tiff
        println("Generating Bayer pattern TIFF from S0...")
        height, width, num_wavelengths, _ = size(rendered_stokes_image)
        bayer_image_raw = zeros(Float32, height, width)

        # Process each pixel
        for r in 1:height
            for c in 1:width
                # Extract S0 spectrum for the pixel
                pixel_s0_spectrum = Spectrum(rendered_stokes_image[r, c, :, 1])
                
                # Determine Bayer color for this pixel
                bayer_color = get_bayer_color(r, c) # row, col ordering
                
                # Apply the corresponding ARRI filter and integrate
                bayer_image_raw[r, c] = apply_bayer_filter(pixel_s0_spectrum, bayer_color)
            end
        end

        # Save the raw Bayer data as a grayscale Float32 TIFF
        # Images.Gray automatically handles creating a grayscale image view
        # Flip vertically to match the final RGB output convention (OpenGL)
        bayer_image_raw_flipped = permutedims(bayer_image_raw, (2, 1)) # Flip like RGB
        bayer_tiff_path = "output_bayer_pattern.tif"
        try
            save(bayer_tiff_path, Images.Gray.(bayer_image_raw_flipped))
            println("Bayer pattern TIFF saved to $bayer_tiff_path")
        catch e
            println("Error saving Bayer TIFF: $e")
            println("Ensure you have necessary image saving packages installed (e.g., TIFF.jl via FileIO).")
        end
        # Continue with standard RGB processing below...
    end
    # ----------------------------------------

    # Extract S0 component (spectral intensity) for RGB conversion
    # S0 is the first component of the Stokes vector (index 1)
    rendered_spectral_intensity = rendered_stokes_image[:, :, :, 1] # Shape [h, w, wavelengths]

    # Convert spectral intensity to RGB
    println("Converting S0 (intensity) image to RGB...")
    rendered_xyz = spectral_to_xyz(rendered_spectral_intensity)
    rendered_rgb = xyz_to_rgb(rendered_xyz, XYZ_TO_SRGB_MATRIX)

    # Flip image vertically (OpenGL convention)
    rendered_rgb = permutedims(rendered_rgb, (2, 1, 3))

    # Apply sRGB gamma correction
    rendered_srgb = linear_rgb_to_srgb(rendered_rgb)

    # Clip values to valid range
    rendered_ldr = clamp.(rendered_srgb, 0.0f0, 1.0f0)

    # Create image
    img = colorview(RGB, permutedims(rendered_ldr, (3, 1, 2)))

    # Save image
    output_path = "output_render_spectral_S0.png"
    save(output_path, img)

    # Also save ACEScg EXR (HDR format)
    output_exr_path = "output_render_spectral_acescg.exr"
    # Convert XYZ to Linear ACEScg using the correct matrix
    image_linear_acescg = xyz_to_rgb(rendered_xyz, XYZ_TO_ACESCG_MATRIX)
    # We need to handle EXR saving separately

    println("S0 (Intensity) Image saved to $output_path")

    # Return the rendered data
    return rendered_stokes_image, img
end

"""
    render_image_inverse(target_spectral::Array{Float32, 3}, max_iterations::Int=50, obj_path::String)

Optimize material parameters to match a target image.
"""
function render_image_inverse(target_spectral::Array{Float32, 3};
                             max_iterations::Int=50,
                             obj_path::Union{String, Nothing}=nothing,
                             width::Int=256, height::Int=256,
                             samples_per_pixel::Int=4, max_depth::Int=3)

    println("Performing inverse rendering with $max_iterations iterations...")

    # Set up scene similar to forward rendering but with simplified settings
    # for faster optimization

    # Define initial material to optimize (fluorescent yellow)
    materials = [
        # Material 0: Starting point for optimization (will be modified)
        Material(
            diffuse=create_spectrum_profile("flat", Float32(0.5)),  # Start with flat gray
            emission=Spectrum(fill(Float32(0.0), N_WAVELENGTHS)),
            specular=Float32(0.1), # Non-zero specular implies non-Lambertian
            roughness=Float32(0.2),
            # For a potentially specular material, IDENTITY is a placeholder.
            # A real specular material would have a Fresnel-based Mueller matrix.
            # For optimization, we might optimize elements of this matrix or its parameters.
            # Let's start with IDENTITY, assuming the optimized material could be anything.
            mueller_matrix=IDENTITY_MUELLER_SPECTRAL
        ),
        # Material 1: Checker A - Gray (fixed)
        Material(
            diffuse=create_spectrum_profile("white_paint", Float32(0.18)),
            emission=Spectrum(fill(Float32(0.0), N_WAVELENGTHS)),
            specular=Float32(0.0),
            roughness=Float32(1.0),
            mueller_matrix=DEPOLARIZER_MUELLER_SPECTRAL
        ),
        # Material 2: Checker B - White (fixed)
        Material(
            diffuse=create_spectrum_profile("white_paint", Float32(0.9)),
            emission=Spectrum(fill(Float32(0.0), N_WAVELENGTHS)),
            specular=Float32(0.0),
            roughness=Float32(1.0),
            mueller_matrix=DEPOLARIZER_MUELLER_SPECTRAL
        )
    ]

    # Define light sources (simplified for optimization)
    directional_lights = [
        DirectionalLight(
            normalize(Vec3f(Float32(-1.0), Float32(-1.0), Float32(-1.0))),
            create_spectrum_profile("led_white_cool", Float32(2.0))
        )
    ]

    point_lights = PointLight[]  # Simplify by removing point lights for optimization

    # Define background
    background_spd = create_spectrum_profile("flat", Float32(0.1))  # Simple flat background

    # Create geometry
    objects = Hittable[]

    # Remove ground plane (as a large sphere)
    # push!(objects, Sphere(
    #     Point3f(0.0f0, -100.5f0, 0.0f0),
    #     Float32(100.0),
    #     Int32(1)  # Gray material
    # ))

    # Define the ground plane (matches Python Y=0 plane)
    ground_plane = Plane(
        normalize_safe(Vec3f(Float32(0.0), Float32(1.0), Float32(0.0))), # Normal (Y-up)
        Float32(0.0),                                      # d (offset from origin, so Y=0)
        Int32(2),                                   # 1-based index -> materials[2] (Gray - fixed)
        Int32(3),                                   # 1-based index -> materials[3] (White - fixed)
        Float32(2.0)                                       # checker_scale (matches Python)
    )

    # Add light source sphere (if it's supposed to be there - Python doesn't have it)
    # For now, let's keep it to see its effect, but this is a discrepancy with Python forward_render.py
    # Removing this to match Python scene which has no explicit emissive sphere object.
    # push!(objects, Sphere(
    #     Point3f(0.0f0, 2.0f0, 0.0f0),
    #     Float32(0.5),
    #     Int32(3)  # Emissive material
    # ))

    # Load mesh or use sphere for optimization
    if obj_path !== nothing && isfile(obj_path)
        println("Loading mesh from $obj_path for optimization")
        mesh = load_mesh_from_obj(
            obj_path,
            Int32(1),  # 1-based index -> materials[1] (to be optimized)
            scale=Point3f(Float32(1.0), Float32(1.0), Float32(1.0)),
            translation=Point3f(Float32(0.0), Float32(1.0), Float32(0.0)),
            center=true,
            normalize_size=true
        )
        push!(objects, mesh)
    else
        # Add a default sphere if no mesh provided
        push!(objects, Sphere(
            Point3f(Float32(0.0), Float32(1.0), Float32(0.0)),
            Float32(1.0),
            Int32(1)  # 1-based index -> materials[1] (to be optimized)
        ))
    end

    # Create a hittable list
    world = HittableList(objects)

    # Define simplified camera
    camera = Camera(
        # Intrinsics (simplified)
        fx=Float32(width),
        fy=Float32(width),
        cx=Float32(width / 2),
        cy=Float32(height / 2),

        # Extrinsics
        eye=Point3f(Float32(-2.0), Float32(2.0), Float32(4.0)),
        center=Point3f(Float32(0.0), Float32(1.0), Float32(0.0)),
        up=Vec3f(Float32(0.0), Float32(1.0), Float32(0.0)),

        # No distortion for optimization
        k1=Float32(0.0),
        k2=Float32(0.0),
        k3=Float32(0.0),
        p1=Float32(0.0),
        p2=Float32(0.0),

        # Simple exposure
        f_number=Float32(8.0),
        shutter_speed=Float32(1/100),
        iso=Float32(100.0)
    )

    # Optimization loop using Enzyme for gradient computation
    println("Starting optimization...")

    for iter in 1:max_iterations
        println("Iteration $iter/$max_iterations")

        # Compute gradient of the loss with respect to material parameters
        gradient = ∇render_enzyme(
            camera, world, ground_plane, directional_lights, point_lights,
            materials, background_spd, width, height,
            samples_per_pixel, max_depth, target_spectral
        )

        # Simple gradient descent step
        learning_rate = Float32(0.01)

        # Update only the diffuse color of material 0
        materials[1] = Material(
            diffuse=materials[1].diffuse .- learning_rate .* gradient[1].diffuse,
            emission=materials[1].emission,
            specular=materials[1].specular,
            roughness=materials[1].roughness,
            mueller_matrix=materials[1].mueller_matrix
        )

        # Clamp diffuse values to valid range
        materials[1] = Material(
            diffuse=clamp.(materials[1].diffuse, 0.0f0, 1.0f0),
            emission=materials[1].emission,
            specular=materials[1].specular,
            roughness=materials[1].roughness,
            mueller_matrix=materials[1].mueller_matrix
        )

        # Render intermediate result
        if iter % 10 == 0 || iter == max_iterations
            # Render with current parameters (low samples for speed)
            intermediate_stokes, _ = render_image_forward(
                width=width, height=height,
                samples_per_pixel=samples_per_pixel,
                max_depth=max_depth,
                obj_path=obj_path
            )

            # Extract S0 component (spectral intensity) for RGB conversion
            intermediate_s0 = intermediate_stokes[:,:,:,1]
            output_path = "opt_result_iter_$(iter)_S0.png"
            intermediate_xyz = spectral_to_xyz(intermediate_s0)
            intermediate_rgb = xyz_to_rgb(intermediate_xyz, XYZ_TO_SRGB_MATRIX)
            intermediate_rgb = permutedims(intermediate_rgb, (2, 1, 3))
            intermediate_srgb = linear_rgb_to_srgb(intermediate_rgb)
            intermediate_ldr = clamp.(intermediate_srgb, 0.0f0, 1.0f0)
            img = colorview(RGB, permutedims(intermediate_ldr, (3, 1, 2)))
            save(output_path, img)
            println("Saved intermediate S0 result to $output_path")
        end
    end

    println("Optimization complete")

    # Return optimized materials
    return materials
end

# Export main rendering functions
export render_image_forward, render_image_inverse

end # module