# Forward rendering with array of objects arranged in a grid
# Translated from Python/JAX implementation

# Include the main module
include("src/SpectralRenderer.jl")

# Import everything from the module
using .SpectralRenderer
using ArgParse
using LinearAlgebra
using Random
using Dates

function main()
    # --- Argument Parsing ---
    parser = ArgParseSettings(description="Minimal Julia Renderer - Forward Rendering Array")
    
    @add_arg_table parser begin
        "--width"
            help = "Image width"
            arg_type = Int
            default = 1024
        "--height"
            help = "Image height"
            arg_type = Int
            default = 1024
        "--spp"
            help = "Samples per pixel (reduced default for array scene)"
            arg_type = Int
            default = 16
        "--max-depth"
            help = "Maximum path tracing depth"
            arg_type = Int
            default = 5
        "--sphere-obj-path"
            help = "Path to the template sphere OBJ file"
            arg_type = String
            default = "test_sphere.obj"
    end
    
    args = parse_args(parser)
    
    width = args["width"]
    height = args["height"]
    samples_per_pixel = args["spp"]
    max_depth = args["max-depth"]
    sphere_obj_template_path = args["sphere-obj-path"]
    
    # --- Check if template mesh exists ---
    if !isfile(sphere_obj_template_path)
        println("Error: Template sphere OBJ file not found at $sphere_obj_template_path. Exiting.")
        return
    end

    println("Using sphere mesh template from $sphere_obj_template_path (for reference only)")

    # --- Array Setup ---
    num_spheres_x = 3
    num_spheres_z = 3
    total_spheres = num_spheres_x * num_spheres_z
    sphere_spacing = Float32(3.0)  # Spacing between sphere centers
    sphere_radius = Float32(1.0)   # Assuming sphere has radius 1.0, centered at origin
    
    # SPD names for the spheres
    spd_names = [
        "red", "green", "blue", 
        "gold", "silver", "copper",
        "fluorescent_pink", "fluorescent_green", "fluorescent_yellow"
    ]
    
    # Repeat SPDs if not enough unique ones
    if total_spheres > length(spd_names)
        spd_names = vcat(fill(spd_names, cld(total_spheres, length(spd_names)))...)[1:total_spheres]
    end
    
    # Combined array for all spheres
    combined_triangles = Triangle[]
    sphere_world_positions = Point3f[]
    
    # Process each sphere
    for i in 0:(total_spheres-1)
        instance_x_idx = i % num_spheres_x
        instance_z_idx = i ÷ num_spheres_x
        
        # Calculate sphere position
        pos_x = (instance_x_idx - (num_spheres_x - 1) / Float32(2.0)) * sphere_spacing
        pos_y = sphere_radius  # Place base of sphere on Y=0 plane
        pos_z = (instance_z_idx - (num_spheres_z - 1) / Float32(2.0)) * sphere_spacing
        sphere_pos_offset = Point3f(pos_x, pos_y, pos_z)
        push!(sphere_world_positions, sphere_pos_offset)
        
        # Create a mesh for this sphere with the correct position
        for triangle in template_triangles
            # Get vertices and transform them
            transformed_vertices = [
                triangle.vertices[1] + sphere_pos_offset,
                triangle.vertices[2] + sphere_pos_offset,
                triangle.vertices[3] + sphere_pos_offset
            ]
            
            # Create a new triangle with transformed vertices
            new_triangle = Triangle(transformed_vertices, Int32(0)) # material_id 0
            push!(combined_triangles, new_triangle)
        end
        
        # Create material for this sphere
        material_color = create_spectrum_profile(spd_names[i+1], Float32(1.0))
    end
    
    # Create the combined mesh
    combined_mesh = Mesh(combined_triangles, Int32(0))
    
    # Fixed SPDs for plane materials
    spd_checker_a = create_spectrum_profile("white_paint", Float32(0.18))
    spd_checker_b = create_spectrum_profile("white_paint", Float32(0.9))
    
    # Define plane parameters
    plane_normal = Vec3f(Float32(0.0), Float32(1.0), Float32(0.0))
    plane_d = Float32(0.0)  # Place plane at Y=0
    plane_checker_scale = sphere_spacing  # Scale checkers relative to sphere spacing
    
    # Background SPD
    background_spd = create_spectrum_profile("daylight_sunset", Float32(0.5))  # Dimmer background
    
    # --- Lights Definition ---
    println("Defining light sources...")
    spd_light1 = create_spectrum_profile("led_white_cool", Float32(10.0))  # Brighter light
    spd_light2 = create_spectrum_profile("blackbody_emitter", Float32(4500.0))  # Different temp
    spd_point_light = create_spectrum_profile("laser_blue", Float32(200.0))

    directional_lights = [
        DirectionalLight(
            normalize(Vec3f(Float32(-0.5), Float32(-1.0), Float32(-0.7))),
            spd_light1
        ),
        DirectionalLight(
            normalize(Vec3f(Float32(0.8), Float32(-0.8), Float32(0.5))),
            spd_light2
        )
    ]

    # Move point light to better illuminate array
    point_light_position = Vec3f(Float32(0.0), sphere_radius + num_spheres_z * sphere_spacing * Float32(0.75), Float32(0.0))
    point_lights = [
        PointLight(
            point_light_position,
            spd_point_light
        )
    ]
    
    # --- Camera Adjustment ---
    # First we need a list of all sphere positions
    sphere_world_positions = []

    for i in 0:total_spheres-1
        instance_x_idx = i % num_spheres_x
        instance_z_idx = i ÷ num_spheres_x

        # Calculate sphere position
        pos_x = (instance_x_idx - (num_spheres_x - 1) / Float32(2.0)) * sphere_spacing
        pos_y = sphere_radius # Place base of sphere on Y=0 plane
        pos_z = (instance_z_idx - (num_spheres_z - 1) / Float32(2.0)) * sphere_spacing

        push!(sphere_world_positions, Point3f(pos_x, pos_y, pos_z))
    end

    # Calculate scene bounds to frame all spheres
    min_coords = Point3f(
        minimum([p[1] for p in sphere_world_positions]) - sphere_radius,
        minimum([p[2] for p in sphere_world_positions]) - sphere_radius,
        minimum([p[3] for p in sphere_world_positions]) - sphere_radius
    )

    max_coords = Point3f(
        maximum([p[1] for p in sphere_world_positions]) + sphere_radius,
        maximum([p[2] for p in sphere_world_positions]) + sphere_radius,
        maximum([p[3] for p in sphere_world_positions]) + sphere_radius
    )

    scene_center = (min_coords + max_coords) / Float32(2.0)
    scene_extent = maximum(max_coords - min_coords)

    # Basic camera setup: look at scene_center, pull back based on extent
    cam_eye_pos = scene_center + Vec3f(scene_extent * Float32(0.6), scene_extent * Float32(0.7), scene_extent * Float32(1.5))  # Angled view
    cam_look_at = scene_center
    cam_up_dir = Vec3f(Float32(0.0), Float32(1.0), Float32(0.0))

    # Approximate 60 degree FoV
    focal_length_new = Float32((width / 2.0) / tan(deg2rad(60 / 2.0)))

    center_x = Float32(width / 2.0)
    center_y = Float32(height / 2.0)
    println("Centering principal point: Overriding cx, cy to ($center_x, $center_y)")
    
    camera = Camera(
        # Intrinsics
        fx=focal_length_new,
        fy=focal_length_new,
        cx=center_x,
        cy=center_y,
        
        # Extrinsics
        eye=cam_eye_pos,
        center=cam_look_at,
        up=cam_up_dir,
        
        # No distortion for simplicity
        k1=Float32(0.0),
        k2=Float32(0.0),
        k3=Float32(0.0),
        p1=Float32(0.0),
        p2=Float32(0.0),
        
        # Adjusted exposure for potentially brighter scene
        f_number=Float32(5.6),
        shutter_speed=Float32(1.0/125.0), # Explicitly Float32 for division
        iso=Float32(100.0)
    )
    
    # First create all materials
    # Create materials for the scene
    materials = [
        # Material 1: Checker A - Gray
        Material(
            diffuse=spd_checker_a,
            emission=fill(Float32(0.0), N_WAVELENGTHS),
            specular=Float32(0.0),
            roughness=Float32(1.0)
        ),
        # Material 2: Checker B - White
        Material(
            diffuse=spd_checker_b,
            emission=fill(Float32(0.0), N_WAVELENGTHS),
            specular=Float32(0.0),
            roughness=Float32(1.0)
        )
    ]

    # Add materials for each sphere
    for i in 1:total_spheres
        spd_for_sphere = create_spectrum_profile(spd_names[i], Float32(1.0))
        sphere_material = Material(
            diffuse=spd_for_sphere,
            emission=fill(Float32(0.0), N_WAVELENGTHS),
            specular=Float32(0.1),
            roughness=Float32(0.2)
        )
        push!(materials, sphere_material)
    end

    # Now create scene objects
    objects = Hittable[]

    # Add ground plane (as a large sphere)
    push!(objects, Sphere(
        Point3f(0.0f0, -100.5f0, 0.0f0),
        Float32(100.0),
        Int32(1)  # Material 1 - Gray checker
    ))

    # Create individual spheres
    for i in 1:total_spheres
        sphere_idx = i - 1  # 0-based indexing to match Python
        instance_x_idx = sphere_idx % num_spheres_x
        instance_z_idx = sphere_idx ÷ num_spheres_x

        # Calculate sphere position
        pos_x = (instance_x_idx - (num_spheres_x - 1) / Float32(2.0)) * sphere_spacing
        pos_y = sphere_radius # Place base of sphere on Y=0 plane
        pos_z = (instance_z_idx - (num_spheres_z - 1) / Float32(2.0)) * sphere_spacing

        sphere_position = Point3f(pos_x, pos_y, pos_z)

        # Add sphere with reference to its material
        # Material indices: 1,2 = checker materials, 3+ = sphere materials
        material_id = Int32(2 + i)  # Start from material index 3 (1-based in Julia)

        push!(objects, Sphere(
            sphere_position,
            sphere_radius,
            material_id
        ))
    end
    
    # Create a hittable list
    world = HittableList(objects)
    
    # --- Rendering ---
    # Set the random seed to 0 to match Python version
    rng = MersenneTwister(0)
    
    start_time = now()
    println("Rendering $(width)x$(height) image at $(samples_per_pixel) spp (Spectral)... Camera: f/$(camera.f_number), $(camera.shutter_speed)s, ISO $(camera.iso)")
    println("Combined mesh has $(length(combined_triangles)) triangles.")
    
    # Create materials for spheres - in the real implementation these would be UDIM textures
    for i in 1:total_spheres
        material_color = create_spectrum_profile(spd_names[i], Float32(1.0))
        # In a full UDIM implementation, we would be setting these in the texture tiles
    end
    
    # Render the image
    rendered_spectral = render_image(
        camera, world, directional_lights, point_lights,
        materials, background_spd, width, height,
        samples_per_pixel, max_depth, progress_update=true
    )
    
    end_time = now()
    render_time = Dates.value(end_time - start_time) / 1000.0  # milliseconds to seconds
    println("Rendering finished in $(round(render_time, digits=2)) seconds.")
    
    # --- Post-processing: Convert Spectral to RGB ---
    println("Converting spectral image to RGB...")
    
    # Convert spectral to XYZ
    image_xyz = spectral_to_xyz(rendered_spectral)
    
    # Convert XYZ to Linear sRGB
    image_linear_srgb = xyz_to_rgb(image_xyz, XYZ_TO_SRGB_MATRIX)
    
    # Flip the image vertically (along height axis)
    image_linear_srgb = permutedims(image_linear_srgb, (2, 1, 3))
    
    # 1. Save PNG (Apply sRGB gamma, Clamp LDR)
    println("Saving LDR image (PNG) with sRGB gamma...")
    image_srgb_gamma = linear_rgb_to_srgb(image_linear_srgb)
    rendered_ldr = clamp.(image_srgb_gamma, Float32(0.0), Float32(1.0))
    
    # Create image
    img = colorview(RGB, permutedims(rendered_ldr, (3, 1, 2)))
    
    # Save to PNG
    output_png_path = "output_render_array_spectral.png"
    save(output_png_path, img)
    println("PNG saved to $output_png_path")
    
    # 2. Save EXR (HDR Float16 with ACEScg Primaries)
    println("Saving HDR image (EXR) with ACEScg primaries...")
    output_exr_path = "output_render_array_spectral_acescg.exr"
    
    # Convert XYZ to Linear ACEScg using the correct matrix
    image_linear_acescg = xyz_to_rgb(image_xyz, XYZ_TO_ACESCG_MATRIX)
    image_linear_acescg = permutedims(image_linear_acescg, (2, 1, 3))
    
    # Save EXR using FileIO (assuming there's a handler for EXR)
    try
        save(output_exr_path, colorview(RGB, permutedims(image_linear_acescg, (3, 1, 2))))
        println("EXR saved to $output_exr_path")
    catch e
        println("Error saving EXR: $e")
        println("Ensure you have EXR support in FileIO/ImageIO.")
    end
end

# Run main function
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end