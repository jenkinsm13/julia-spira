# Main entry point for spectral renderer with Enzyme-based automatic differentiation
# Translated from Python/JAX implementation

# Include the main module
include("src/SpectralRenderer.jl")

# Import everything from the module
using .SpectralRenderer
using ArgParse
using Images

# -----------------------------------------------------------------------------
# Main Function
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
            default = 64
        "--max-depth", "-d"
            help = "Maximum path tracing depth"
            arg_type = Int
            default = 5
        "--obj-path", "-o"
            help = "Path to OBJ file for the main mesh"
            arg_type = String
            default = "julia-translation-root/test_sphere.obj"
        "--mode", "-m"
            help = "Rendering mode: forward or inverse"
            arg_type = String
            default = "forward"
        "--target-exr", "-t"
            help = "Path to target EXR image for optimization (required for inverse mode)"
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
        # Check if target image is provided
        if args["target-exr"] === nothing
            println("Error: Target EXR image is required for inverse rendering")
            println("Please provide a target image with --target-exr")
            return
        end

        # Load target image
        target_image = load(args["target-exr"])

        # Convert target_image (e.g., Array{RGB{Float16},2}) to an HxWx3 Array{Float32}
        # 1. Convert pixel component type from Float16 to Float32, resulting in e.g. Array{RGB{Float32},2}
        target_image_rgb_f32 = RGB{Float32}.(target_image)
        # 2. Convert array of RGB structs to a 3D numeric array (e.g., 3 x Height x Width)
        target_channels_first = channelview(target_image_rgb_f32)
        # 3. Permute dimensions to (Height x Width x Channels)
        target = permutedims(target_channels_first, (2, 3, 1))

        # Resize if dimensions don't match
        if size(target, 1) != args["height"] || size(target, 2) != args["width"]
            println("Warning: Resizing target image to $(args["width"]) x $(args["height"])")
            target = imresize(target, (args["height"], args["width"]))
        end

        # Convert RGB to spectral (approximate)
        target_spectral = rgb_to_spd_d65_approx(target)

        # Run inverse rendering
        optimized_materials = render_image_inverse(
            target_spectral,
            max_iterations=args["steps"],
            obj_path=args["obj-path"],
            width=args["width"],
            height=args["height"]
        )

        println("Optimized material parameters:")
        println("Diffuse: ", optimized_materials[1].diffuse)
        println("Specular: ", optimized_materials[1].specular)
        println("Roughness: ", optimized_materials[1].roughness)
    else
        println("Error: Unknown mode '$(args["mode"])'")
        println("Supported modes: forward, inverse")
    end
end

# Run the main function if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end