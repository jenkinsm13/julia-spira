# Main entry point for spectral renderer with Enzyme-based automatic differentiation
# Translated from Python/JAX implementation

# Include the main module
# Corrected path to the module
include("julia-translation-root/src/SpectralRenderer.jl")

# Import everything from the module
using .SpectralRenderer
using ArgParse
using Images
using FileIO # For saving images
using ImageIO # For EXR saving

# -----------------------------------------------------------------------------
# Command Line Argument Parsing
# -----------------------------------------------------------------------------
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--width", "-W"
            help = "Image width"
            arg_type = Int
            default = 1024
        "--height", "-H"
            help = "Image height"
            arg_type = Int
            default = 1024
        "--spp", "-s"
            help = "Samples per pixel"
            arg_type = Int
            default = 64 # Default for forward, inverse will use its own
        "--max-depth", "-d"
            help = "Maximum path tracing depth"
            arg_type = Int
            default = 5 # Default for forward, inverse will use its own
        "--obj-path", "-o"
            help = "Path to OBJ file for the main mesh"
            arg_type = String
            default = "julia-translation-root/test_sphere.obj"
        "--mode", "-m"
            help = "Rendering mode: forward or inverse"
            arg_type = String
            default = "forward"
        "--target-exr", "-t"
            help = "Path to target EXR image for optimization (optional for inverse mode, will generate if not provided)"
            arg_type = String
            default = nothing
        "--steps", "-i"
            help = "Number of optimization steps (for inverse mode)"
            arg_type = Int
            default = 50
        "--focus-distance", "-f"
            help = "Camera focus distance in scene units (forward mode)"
            arg_type = Float32
            default = nothing
        "--bayer-tiff"
            help = "Output a raw Bayer pattern TIFF image (BGGR layout, ARRI CFA) when in forward mode"
            action = :store_true
    end

    return parse_args(s)
end

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
function main()
    args = parse_commandline()

    if args["mode"] == "forward"
        println("Executing spectral_renderer.jl in forward mode with arguments: ", args)
        render_image_forward(
            width=args["width"],
            height=args["height"],
            samples_per_pixel=args["spp"],
            max_depth=args["max-depth"],
            obj_path=args["obj-path"],
            focus_distance_arg=args["focus-distance"],
            output_bayer_tiff=args["bayer-tiff"]
        )
    elseif args["mode"] == "inverse"
        println("Executing spectral_renderer.jl in inverse mode with arguments: ", args)
        inv_width = args["width"]
        inv_height = args["height"]
        inv_spp = 4 
        inv_max_depth = 3
        inv_obj_path = args["obj-path"]
        num_opt_steps = args["steps"]

        target_spectral::Array{Float32, 3}

        if args["target-exr"] === nothing
            println("No target EXR provided by user. Generating one with a forward pass...")
            initial_opt_material = Material( 
                diffuse=create_spectrum_profile("flat", 0.5f0),
                emission=Spectrum(fill(Float32(0.0), N_WAVELENGTHS)),
                specular=Float32(0.1),
                roughness=Float32(0.2),
                mueller_matrix=IDENTITY_MUELLER_SPECTRAL
            )
            target_gen_materials = [
                initial_opt_material, 
                Material(diffuse=create_spectrum_profile("white_paint", 0.18f0), mueller_matrix=DEPOLARIZER_MUELLER_SPECTRAL),
                Material(diffuse=create_spectrum_profile("white_paint", 0.9f0), mueller_matrix=DEPOLARIZER_MUELLER_SPECTRAL)
            ]
            target_gen_directional_lights = [DirectionalLight(normalize(Vec3f(-1.0f0, -1.0f0, -1.0f0)), create_spectrum_profile("led_white_cool", 2.0f0))]
            target_gen_point_lights = PointLight[]
            target_gen_background_spd = create_spectrum_profile("flat", 0.1f0)
            target_gen_objects = Hittable[]
            if inv_obj_path !== nothing && isfile(inv_obj_path)
                mesh = load_mesh_from_obj(inv_obj_path, Int32(1), scale=Point3f(1.0f0, 1.0f0, 1.0f0), translation=Point3f(0.0f0, 1.0f0, 0.0f0), center=true, normalize_size=true)
                push!(target_gen_objects, mesh)
            else 
                println("No OBJ for target gen, using default sphere.")
                push!(target_gen_objects, Sphere(Point3f(0.0f0, 1.0f0, 0.0f0), Float32(1.0), Int32(1)))
            end
            target_gen_world = HittableList(target_gen_objects)
            target_gen_ground_plane = Plane(normalize_safe(Vec3f(0.0f0, 1.0f0, 0.0f0)), 0.0f0, Int32(2), Int32(3), 2.0f0)
            target_gen_camera = Camera(fx=Float32(inv_width), fy=Float32(inv_width), cx=Float32(inv_width / 2), cy=Float32(inv_height / 2), eye=Point3f(-2.0f0, 2.0f0, 4.0f0), center=Point3f(0.0f0, 1.0f0, 0.0f0), up=Vec3f(0.0f0, 1.0f0, 0.0f0), k1=0.0f0, k2=0.0f0, k3=0.0f0, p1=0.0f0, p2=0.0f0, f_number=8.0f0, shutter_speed=1/100.0f0, iso=100.0f0)
            println("Generating reference image (spectral S0) for inverse rendering test...")
            reference_stokes_image = render_image(target_gen_camera, target_gen_world, target_gen_ground_plane, target_gen_directional_lights, target_gen_point_lights, target_gen_materials, target_gen_background_spd, inv_width, inv_height, inv_spp, inv_max_depth, progress_update=true)
            target_spectral = reference_stokes_image[:, :, :, 1]
            println("Using internally generated S0 spectral data as target for inverse rendering.")
            reference_exr_path = "temp_generated_target.exr"
            try
                temp_target_xyz = spectral_to_xyz(target_spectral)
                temp_target_rgb_linear = xyz_to_rgb(temp_target_xyz, XYZ_TO_SRGB_MATRIX)
                img_rgb_f32_for_save = Array{RGB{Float32}}(undef, inv_height, inv_width)
                for r_idx in 1:inv_height, c_idx in 1:inv_width
                    img_rgb_f32_for_save[r_idx, c_idx] = RGB{Float32}(temp_target_rgb_linear[r_idx, c_idx, 1], temp_target_rgb_linear[r_idx, c_idx, 2], temp_target_rgb_linear[r_idx, c_idx, 3])
                end
                save(reference_exr_path, img_rgb_f32_for_save)
                println("Saved RGB version of generated target to $reference_exr_path for inspection.")
            catch e
                println("Error saving temporary generated target EXR: $e")
            end
        else 
            println("Using provided target EXR: ", args["target-exr"])
            loaded_target_rgb_image = load(args["target-exr"])
            target_rgb_float_array::Array{Float32, 3}
            if eltype(loaded_target_rgb_image) <: RGB{<:Normed}
                cview = channelview(loaded_target_rgb_image)
                target_rgb_float_array = Float32.(permutedims(cview, (2,3,1)))
            elseif eltype(loaded_target_rgb_image) <: RGB{Float32}
                cview = channelview(loaded_target_rgb_image)
                target_rgb_float_array = permutedims(cview, (2,3,1))
            else
                println("Warning: Loaded target is not RGB type. Attempting direct conversion.")
                numeric_array = rawview(channelview(loaded_target_rgb_image))
                if ndims(numeric_array) == 3 && size(numeric_array,1) == 3
                     target_rgb_float_array = Float32.(permutedims(numeric_array, (2,3,1)))
                elseif ndims(numeric_array) == 3 && size(numeric_array,3) == 3
                     target_rgb_float_array = Float32.(numeric_array)
                else
                    println("Error: Could not interpret loaded target format.")
                    return
                end
            end
            if size(target_rgb_float_array, 1) != inv_height || size(target_rgb_float_array, 2) != inv_width
                println("Warning: Resizing target from $(size(target_rgb_float_array,2))x$(size(target_rgb_float_array,1)) to $(inv_width)x$(inv_height)")
                resized_target_rgb_array = imresize(target_rgb_float_array, (inv_height, inv_width))
            else
                resized_target_rgb_array = target_rgb_float_array
            end
            target_spectral = rgb_to_spd_d65_approx(resized_target_rgb_array)
            println("Converted provided target EXR to spectral.")
        end

        println("Starting inverse rendering process...")
        optimized_materials = render_image_inverse(target_spectral, max_iterations=num_opt_steps, obj_path=inv_obj_path, width=inv_width, height=inv_height, samples_per_pixel=inv_spp, max_depth=inv_max_depth)
        println("Optimized material parameters after $num_opt_steps steps:")
        if !isempty(optimized_materials)
            println("Diffuse (first 5 wavelengths): ", optimized_materials[1].diffuse[1:min(5,end)])
            println("Specular: ", optimized_materials[1].specular)
            println("Roughness: ", optimized_materials[1].roughness)
        else
            println("Optimization did not return any materials.")
        end
    else
        println("Error: Unknown mode '$(args["mode"])'")
        println("Supported modes: forward, inverse")
    end
end

# Run the main function if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 