using StaticArrays

# Bayer Pattern (Top-left 2x2 cell: B G / G R for 1-indexed row, col)
# (1,1) B  (1,2) G
# (2,1) G  (2,2) R

@enum BayerColor R G B

"""
    get_bayer_color(row::Int, col::Int)::BayerColor

Determines the Bayer filter color for a given pixel coordinate (1-indexed).
The pattern is BGGR:
B G
G R
"""
function get_bayer_color(row::Int, col::Int)::BayerColor
    if isodd(row)
        if isodd(col)
            return B  # e.g., (1,1)
        else
            return G  # e.g., (1,2)
        end
    else # iseven(row)
        if isodd(col)
            return G  # e.g., (2,1)
        else
            return R  # e.g., (2,2)
        end
    end
end

# Interpolate ARRI CFA sensitivities to the project's WAVELENGTHS_NM
# collect() converts SVectors from Utils to Vectors for interpolate_array.
const ARRI_R_RESPONSE_SPECTRAL = Spectrum(interpolate_array(WAVELENGTHS_NM, collect(ARRI_ALEXA_WAVELENGTHS), collect(ARRI_ALEXA_RED_SENSITIVITIES)))
const ARRI_G_RESPONSE_SPECTRAL = Spectrum(interpolate_array(WAVELENGTHS_NM, collect(ARRI_ALEXA_WAVELENGTHS), collect(ARRI_ALEXA_GREEN_SENSITIVITIES)))
const ARRI_B_RESPONSE_SPECTRAL = Spectrum(interpolate_array(WAVELENGTHS_NM, collect(ARRI_ALEXA_WAVELENGTHS), collect(ARRI_ALEXA_BLUE_SENSITIVITIES)))

"""
    apply_bayer_filter(pixel_spectrum::Spectrum, bayer_color::BayerColor)::Float32

Applies the appropriate ARRI spectral filter to a pixel's spectrum (S0 component)
and returns the integrated response.
"""
function apply_bayer_filter(pixel_spectrum::Spectrum, bayer_color::BayerColor)::Float32
    if length(pixel_spectrum.samples) != N_WAVELENGTHS
        error("Pixel spectrum length ($(length(pixel_spectrum.samples))) does not match N_WAVELENGTHS ($N_WAVELENGTHS)")
    end

    local filter_response_curve::Spectrum

    if bayer_color == R
        filter_response_curve = ARRI_R_RESPONSE_SPECTRAL
    elseif bayer_color == G
        filter_response_curve = ARRI_G_RESPONSE_SPECTRAL
    elseif bayer_color == B
        filter_response_curve = ARRI_B_RESPONSE_SPECTRAL
    # No else needed due to @enum type safety
    end
    
    # ARRI sensitivity data can sometimes be negative (e.g., blue channel at long wavelengths)
    # or have values > 1. We should use them as is for physical accuracy but clamp final result.
    # The multiplication by DELTA_WAVELENGTH_NM performs the integration over the spectrum.
    integrated_response = sum(pixel_spectrum.samples .* filter_response_curve.samples) * DELTA_WAVELENGTH_NM
    
    # Sensor response cannot be negative.
    return max(Float32(0.0), integrated_response)
end

export BayerColor, get_bayer_color, apply_bayer_filter
export ARRI_R_RESPONSE_SPECTRAL, ARRI_G_RESPONSE_SPECTRAL, ARRI_B_RESPONSE_SPECTRAL

# end # module Bayer # REMOVE THIS LINE 