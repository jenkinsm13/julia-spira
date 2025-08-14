# Forward rendering with a single mesh or sphere
# Translated from Python/JAX implementation

# Include the main module
include("src/SpectralRenderer.jl")

# Import everything from the module
using .SpectralRenderer
using ArgParse
using Random
using Dates

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--width", "-W"
            help = "Image width"
            arg_type = Int
            default = 512
        "--height", "-H"
            help = "Image height"
            arg_type = Int
            default = 512
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
            default = nothing
        "--focus-distance", "-f"
            help = "Camera focus distance in scene units"
            arg_type = Float32
            default = nothing
        "--bayer-tiff"
            help = "Output a raw Bayer pattern TIFF image (BGGR layout, ARRI CFA)"
            action = :store_true # Makes it a boolean flag
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    println("Executing forward_render.jl with arguments: ", args)

    render_image_forward(
        width=args["width"],
        height=args["height"],
        samples_per_pixel=args["spp"],
        max_depth=args["max-depth"],
        obj_path=args["obj-path"],
        focus_distance_arg=args["focus-distance"],
        output_bayer_tiff=args["bayer-tiff"]
    )
end

# Run main function
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end