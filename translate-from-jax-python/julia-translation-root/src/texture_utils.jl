module TextureUtils

# Import specific types and constants from the Main.SpectralRenderer module
using Main.SpectralRenderer: UV2f, N_WAVELENGTHS, Spectrum

using Images
using FileIO
using StaticArrays
using LinearAlgebra
using Colors # For RGB, XYZ types

# These consts are now expected to be available from the main module (SpectralRenderer)
# which includes types.jl before texture_utils.jl
# const UV2f = SVector{2, Float32}
# const Spectrum = Vector{Float32} # Placeholder, ideally SVector{N, Float32}
# const N_WAVELENGTHS = 31 # Placeholder, should come from a central config

# Global texture cache to avoid reloading files
# Key: filepath (String), Value: Loaded texture data (Array)
# The cache stores the raw Colorant matrix from FileIO.load
const TEXTURE_CACHE = Dict{String, Matrix{<:Colorant}}()

function clear_texture_cache!()
    empty!(TEXTURE_CACHE)
    println("Texture cache cleared.")
end

"""
    load_image_cached(filepath::String)::Union{Nothing, Matrix{<:Colorant}}

Loads an image using Images.jl and caches it.
Returns Nothing if loading fails.
"""
function load_image_cached(filepath::String)::Union{Nothing, Matrix{<:Colorant}}
    if haskey(TEXTURE_CACHE, filepath)
        # println("Texture cache hit for: $filepath")
        return TEXTURE_CACHE[filepath]
    end

    # First, check if the file actually exists
    if !isfile(filepath)
        println(stderr, "Texture file not found: $filepath")
        return nothing
    end

    # If file exists, attempt to load it
    try
        # println("Loading texture from disk: $filepath")
        img = FileIO.load(filepath)

        if !(typeof(img) <: AbstractMatrix{<:Colorant})
            println(stderr, "Warning: Loaded file $filepath is not a recognized image matrix. Type: $(typeof(img))")
            return nothing
        end
        
        TEXTURE_CACHE[filepath] = img
        return img
    catch e
        # This catch block now handles errors for files that exist but failed to load/process correctly
        println(stderr, "Error processing texture file $filepath: $e")
        return nothing
    end
end


"""
    load_texture_as_spectral(filepath::String)::Union{Nothing, Array{Float32,3}}

Loads an image and converts it to a spectral representation (H, W, N_WAVELENGTHS).
If the image is RGB, it's converted to XYZ then to a placeholder spectral format.
"""
function load_texture_as_spectral(filepath::String; N_WAVELENGTHS_PARAM::Int = N_WAVELENGTHS)::Union{Nothing, Array{Float32,3}}
    # This function now relies on N_WAVELENGTHS being available in its scope.
    # It's better to pass it as an argument if it can vary or for explicitness.
    # For now, assuming N_WAVELENGTHS from SpectralRenderer context is visible.

    img_colorant = load_image_cached(filepath)
    if img_colorant === nothing
        return nothing
    end

    height, width = size(img_colorant)
    img_rgb_f32 = convert.(RGB{Float32}, img_colorant) # Ensure RGB{Float32}

    spectral_image = zeros(Float32, height, width, N_WAVELENGTHS_PARAM)

    for r in 1:height
        for c in 1:width
            pixel_rgb = img_rgb_f32[r, c]
            pixel_xyz = convert(XYZ, pixel_rgb)
            
            # Placeholder XYZ to Spectral (as before)
            # THIS IS A VERY CRUDE PLACEHOLDER.
            chunk_size = N_WAVELENGTHS_PARAM ÷ 3
            if chunk_size == 0 # Handle cases where N_WAVELENGTHS_PARAM < 3
                # Default to grayscale or some other sensible fallback
                avg_intensity = (Float32(pixel_rgb.r) + Float32(pixel_rgb.g) + Float32(pixel_rgb.b)) / Float32(3.0)
                spectral_image[r, c, :] .= avg_intensity
            else
                last_idx_r = chunk_size
                last_idx_g = 2*chunk_size
                
                spectral_image[r, c, 1:last_idx_r] .= Float32(pixel_rgb.r)
                spectral_image[r, c, (last_idx_r+1):last_idx_g] .= Float32(pixel_rgb.g)
                # Remaining wavelengths get blue or average
                if N_WAVELENGTHS_PARAM > last_idx_g
                    spectral_image[r, c, (last_idx_g+1):N_WAVELENGTHS_PARAM] .= Float32(pixel_rgb.b)
                end
                # Ensure all wavelengths are covered if N_WAVELENGTHS_PARAM is not a multiple of 3
                # This logic can be refined, e.g. by ensuring the last band gets all remaining blue
                # or by a more graceful distribution.
            end
        end
    end
    return spectral_image
end

"""
    load_texture_as_scalar(filepath::String)::Union{Nothing, Array{Float32,2}}

Loads an image and converts it to a single-channel (grayscale) Float32 array (H, W).
"""
function load_texture_as_scalar(filepath::String)::Union{Nothing, Array{Float32,2}}
    img_colorant = load_image_cached(filepath)
    if img_colorant === nothing
        return nothing
    end
    img_gray_f32 = convert.(Gray{Float32}, img_colorant)
    return Float32[p.val for p in img_gray_f32]
end

"""
    load_texture_as_normalmap(filepath::String)::Union{Nothing, Array{Float32,3}}

Loads an image intended as a tangent-space normal map (H, W, 3).
Values are remapped from [0,1] to [-1,1] for X,Y and Z is assumed to be [0,1] or also remapped.
"""
function load_texture_as_normalmap(filepath::String)::Union{Nothing, Array{Float32,3}}
    img_colorant = load_image_cached(filepath)
    if img_colorant === nothing
        return nothing
    end

    height, width = size(img_colorant)
    normal_map = zeros(Float32, height, width, 3)
    img_rgb_f32 = convert.(RGB{Float32}, img_colorant)

    for r in 1:height
        for c in 1:width
            pixel_rgb = img_rgb_f32[r, c]
            nx = Float32(pixel_rgb.r) * Float32(2.0) - Float32(1.0)
            ny = Float32(pixel_rgb.g) * Float32(2.0) - Float32(1.0)
            # Common OpenGL convention: Z is in [0,1] range, derived from map_bump which is often [0,1] Gray
            # If normal map is in object space or a different tangent space, this needs adjustment.
            # For tangent space where Z is up (common for tools like Substance), B channel is often 1.0.
            # If map stores Z as [0,1], then nz = Float32(pixel_rgb.b) is fine.
            # If map stores Z packed as [-1,1] (like X,Y) then nz = Float32(pixel_rgb.b) * Float32(2.0) - Float32(1.0).
            # Let's assume packed Z for now, but this is a common point of confusion.
            nz = Float32(pixel_rgb.b) * Float32(2.0) - Float32(1.0) 
            # Or, recompute Z: nz = sqrt(max(Float32(0.0), Float32(1.0) - nx*nx - ny*ny)). This assumes X,Y correct.
            normal_map[r, c, 1] = nx
            normal_map[r, c, 2] = ny
            normal_map[r, c, 3] = nz
        end
    end
    return normal_map
end


"""
    sample_texture_bilinear(texture_data::AbstractArray{T, 3}, uv::UV2f) where T<:AbstractFloat

Performs bilinear interpolation on a 3D texture array (e.g., spectral image HxWxChannels).
UV coordinates are expected to be in [0,1] range, mapping to texture array indices.
Returns a Vector{T} of channel values.
"""
function sample_texture_bilinear(texture_data::AbstractArray{T, 3}, uv::UV2f)::Vector{T} where T<:AbstractFloat
    h, w, channels = size(texture_data)
    if h == 0 || w == 0; return zeros(T, channels); end # Handle empty texture

    # Texture coordinates (continuous, 0-indexed)
    tx = uv[1] * (w - 1)
    ty = uv[2] * (h - 1)

    # Integer part of coordinates (0-indexed)
    x0 = floor(Int, tx)
    y0 = floor(Int, ty)

    # Fractional part for interpolation weights
    fx = tx - x0
    fy = ty - y0

    # Clamp coordinates for 1-based array access, ensuring they are within [1, w] or [1, h]
    x0_cl = clamp(x0 + 1, 1, w) # From 0-indexed floor to 1-based index
    y0_cl = clamp(y0 + 1, 1, h)
    x1_cl = clamp(x0 + 2, 1, w) # Next pixel, also 1-based
    y1_cl = clamp(y0 + 2, 1, h)

    val = zeros(T, channels)
    for ch in 1:channels
        c00 = texture_data[y0_cl, x0_cl, ch]
        c10 = texture_data[y0_cl, x1_cl, ch]
        c01 = texture_data[y1_cl, x0_cl, ch]
        c11 = texture_data[y1_cl, x1_cl, ch]

        val[ch] = c00 * (1-fx) * (1-fy) +
                  c10 * fx     * (1-fy) +
                  c01 * (1-fx) * fy +
                  c11 * fx     * fy
    end
    return val
end

"""
    sample_texture_bilinear(texture_data::AbstractArray{T, 2}, uv::UV2f) where T<:AbstractFloat

Performs bilinear interpolation on a 2D texture array (e.g., scalar image HxW).
UV coordinates are expected to be in [0,1] range.
Returns a scalar value of type T.
"""
function sample_texture_bilinear(texture_data::AbstractArray{T, 2}, uv::UV2f)::T where T<:AbstractFloat
    h, w = size(texture_data)
    if h == 0 || w == 0; return zero(T); end # Handle empty texture

    tx = uv[1] * (w - 1)
    ty = uv[2] * (h - 1)

    x0 = floor(Int, tx)
    y0 = floor(Int, ty)

    fx = tx - x0
    fy = ty - y0

    x0_cl = clamp(x0 + 1, 1, w)
    y0_cl = clamp(y0 + 1, 1, h)
    x1_cl = clamp(x0 + 2, 1, w)
    y1_cl = clamp(y0 + 2, 1, h)
    
    c00 = texture_data[y0_cl, x0_cl]
    c10 = texture_data[y0_cl, x1_cl]
    c01 = texture_data[y1_cl, x0_cl]
    c11 = texture_data[y1_cl, x1_cl]

    return c00 * (1-fx) * (1-fy) +
           c10 * fx     * (1-fy) +
           c01 * (1-fx) * fy +
           c11 * fx     * fy
end


# TODO: Add UDIM sampling logic, potentially moving/refining from integrator.jl
# function texture_lookup_udim_bilinear(...)
# Needs access to Material.udim_albedo_tiles, Material.udim_default_albedo_spd etc.

end # module TextureUtils 