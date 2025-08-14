# Utilities for spectral rendering and math operations

using LinearAlgebra
using StaticArrays
using Images
using Colors
using Enzyme
using EnzymeCore
using EnzymeCore.EnzymeRules
using Random

import .EnzymeRules: inactive

# Custom rule to mark rand(rng, Float32) as inactive for Enzyme
# This tells Enzyme to treat its output as a constant during differentiation
EnzymeRules.inactive(::typeof(rand), ::Random.MersenneTwister, ::Type{Float32}, ::Vararg{Any}) = nothing
EnzymeRules.inactive(::typeof(rand), ::Random.TaskLocalRNG, ::Type{Float32}, ::Vararg{Any}) = nothing
EnzymeRules.inactive(::typeof(Base.rand), ::Random.MersenneTwister, ::Type{Float32}, ::Vararg{Any}) = nothing
EnzymeRules.inactive(::typeof(Base.rand), ::Random.TaskLocalRNG, ::Type{Float32}, ::Vararg{Any}) = nothing
# If rand is called without an explicit RNG (using global_rng)
EnzymeRules.inactive(::typeof(rand), ::Type{Float32}, ::Vararg{Any}) = nothing
EnzymeRules.inactive(::typeof(Base.rand), ::Type{Float32}, ::Vararg{Any}) = nothing

# -----------------------------------------------------------------------------
# CIE Color Matching Functions
# -----------------------------------------------------------------------------

# CIE 1931 2-degree Standard Observer data
# Wavelengths: 380 to 780 nm at 5 nm steps (81 samples)
const CIE_1931_5NM_SAMPLE_WAVELENGTHS = Float32[
    380.0, 385.0, 390.0, 395.0, 400.0, 405.0, 410.0, 415.0, 420.0, 425.0, 430.0, 435.0, 
    440.0, 445.0, 450.0, 455.0, 460.0, 465.0, 470.0, 475.0, 480.0, 485.0, 490.0, 495.0, 
    500.0, 505.0, 510.0, 515.0, 520.0, 525.0, 530.0, 535.0, 540.0, 545.0, 550.0, 555.0, 
    560.0, 565.0, 570.0, 575.0, 580.0, 585.0, 590.0, 595.0, 600.0, 605.0, 610.0, 615.0, 
    620.0, 625.0, 630.0, 635.0, 640.0, 645.0, 650.0, 655.0, 660.0, 665.0, 670.0, 675.0, 
    680.0, 685.0, 690.0, 695.0, 700.0, 705.0, 710.0, 715.0, 720.0, 725.0, 730.0, 735.0, 
    740.0, 745.0, 750.0, 755.0, 760.0, 765.0, 770.0, 775.0, 780.0 
]

# Corrected full 81x3 matrix with CIE XYZ color matching functions
const CIE_1931_5NM_SAMPLE_VALUES = Float32[
    0.0014 0.0000 0.0065;  # 380 nm
    0.0022 0.0001 0.0105; 
    0.0042 0.0001 0.0201; 
    0.0076 0.0002 0.0362; 
    0.0143 0.0004 0.0679;  # 400 nm
    0.0232 0.0006 0.1102; 
    0.0435 0.0012 0.2074; 
    0.0776 0.0022 0.3713; 
    0.1344 0.0040 0.6456;  # 420 nm
    0.2148 0.0073 1.0391; 
    0.2839 0.0116 1.3856; 
    0.3285 0.0168 1.6230; 
    0.3483 0.0230 1.7471;  # 440 nm
    0.3481 0.0298 1.7826; 
    0.3362 0.0380 1.7721; 
    0.3187 0.0480 1.7441; 
    0.2908 0.0600 1.6692;  # 460 nm
    0.2511 0.0739 1.5281; 
    0.1954 0.0910 1.2876; 
    0.1421 0.1126 1.0419; 
    0.0956 0.1390 0.8130;  # 480 nm
    0.0580 0.1693 0.6162; 
    0.0320 0.2080 0.4652; 
    0.0147 0.2586 0.3533; 
    0.0049 0.3230 0.2720;  # 500 nm
    0.0024 0.4073 0.2123; 
    0.0093 0.5030 0.1582; 
    0.0291 0.6082 0.1117; 
    0.0633 0.7100 0.0782;  # 520 nm
    0.1096 0.7932 0.0573; 
    0.1655 0.8620 0.0422; 
    0.2257 0.9149 0.0298; 
    0.2904 0.9540 0.0203;  # 540 nm
    0.3597 0.9803 0.0134; 
    0.4334 0.9950 0.0087; 
    0.5121 1.0000 0.0057; 
    0.5945 0.9950 0.0039;  # 560 nm
    0.6784 0.9786 0.0027; 
    0.7621 0.9520 0.0021; 
    0.8425 0.9154 0.0018; 
    0.9163 0.8700 0.0017;  # 580 nm
    0.9786 0.8163 0.0014; 
    1.0263 0.7570 0.0011; 
    1.0567 0.6949 0.0008; 
    1.0622 0.6310 0.0006;  # 600 nm
    1.0456 0.5668 0.0003; 
    1.0026 0.5030 0.0002; 
    0.9384 0.4412 0.0001; 
    0.8544 0.3810 0.0001;  # 620 nm
    0.7514 0.3210 0.0000; 
    0.6424 0.2650 0.0000; 
    0.5419 0.2170 0.0000; 
    0.4479 0.1750 0.0000;  # 640 nm
    0.3608 0.1382 0.0000; 
    0.2835 0.1070 0.0000; 
    0.2187 0.0816 0.0000; 
    0.1649 0.0610 0.0000;  # 660 nm
    0.1212 0.0446 0.0000; 
    0.0874 0.0320 0.0000; 
    0.0636 0.0232 0.0000; 
    0.0468 0.0170 0.0000;  # 680 nm
    0.0329 0.0119 0.0000; 
    0.0227 0.0082 0.0000; 
    0.0158 0.0057 0.0000; 
    0.0114 0.0041 0.0000;  # 700 nm
    0.0081 0.0029 0.0000; 
    0.0058 0.0021 0.0000; 
    0.0041 0.0015 0.0000; 
    0.0029 0.0010 0.0000;  # 720 nm
    0.0020 0.0007 0.0000; 
    0.0014 0.0005 0.0000; 
    0.0010 0.0004 0.0000; 
    0.0007 0.0002 0.0000;  # 740 nm
    0.0005 0.0002 0.0000; 
    0.0003 0.0001 0.0000; 
    0.0002 0.0001 0.0000; 
    0.0002 0.0001 0.0000;  # 760 nm
    0.0001 0.0000 0.0000; 
    0.0001 0.0000 0.0000; 
    0.0001 0.0000 0.0000; 
    0.0000 0.0000 0.0000   # 780 nm
]

# Interpolate the CMFs to our WAVELENGTHS_NM
function interpolate_array(x_new, x_old, y_old)
    y_new = zeros(Float32, length(x_new))
    for i in 1:length(x_new)
        # Find where x_new[i] would fit in x_old
        if x_new[i] <= x_old[1]
            y_new[i] = y_old[1]
        elseif x_new[i] >= x_old[end]
            y_new[i] = y_old[end]
        else
            # Find indices for interpolation
            idx = 1
            while idx < length(x_old) && x_old[idx+1] < x_new[i]
                idx += 1
            end
            
            # Linear interpolation
            t = (x_new[i] - x_old[idx]) / (x_old[idx+1] - x_old[idx])
            y_new[i] = y_old[idx] * (1 - t) + y_old[idx+1] * t
        end
    end
    return y_new
end

# Extract and interpolate each CIE function
const CIE_X = interpolate_array(WAVELENGTHS_NM, CIE_1931_5NM_SAMPLE_WAVELENGTHS, CIE_1931_5NM_SAMPLE_VALUES[:, 1])
const CIE_Y = interpolate_array(WAVELENGTHS_NM, CIE_1931_5NM_SAMPLE_WAVELENGTHS, CIE_1931_5NM_SAMPLE_VALUES[:, 2])
const CIE_Z = interpolate_array(WAVELENGTHS_NM, CIE_1931_5NM_SAMPLE_WAVELENGTHS, CIE_1931_5NM_SAMPLE_VALUES[:, 3])

# Shape: (N_WAVELENGTHS, 3)
const CIE_XYZ_MATCHING_FUNCTIONS = hcat(CIE_X, CIE_Y, CIE_Z)

# -----------------------------------------------------------------------------
# ARRI Alexa CFA Data
# -----------------------------------------------------------------------------

const ARRI_ALEXA_WAVELENGTHS = SVector{81, Float32}(
    Float32(380.0), Float32(385.0), Float32(390.0), Float32(395.0), Float32(400.0), Float32(405.0), Float32(410.0), Float32(415.0), Float32(420.0), Float32(425.0),
    Float32(430.0), Float32(435.0), Float32(440.0), Float32(445.0), Float32(450.0), Float32(455.0), Float32(460.0), Float32(465.0), Float32(470.0), Float32(475.0),
    Float32(480.0), Float32(485.0), Float32(490.0), Float32(495.0), Float32(500.0), Float32(505.0), Float32(510.0), Float32(515.0), Float32(520.0), Float32(525.0),
    Float32(530.0), Float32(535.0), Float32(540.0), Float32(545.0), Float32(550.0), Float32(555.0), Float32(560.0), Float32(565.0), Float32(570.0), Float32(575.0),
    Float32(580.0), Float32(585.0), Float32(590.0), Float32(595.0), Float32(600.0), Float32(605.0), Float32(610.0), Float32(615.0), Float32(620.0), Float32(625.0),
    Float32(630.0), Float32(635.0), Float32(640.0), Float32(645.0), Float32(650.0), Float32(655.0), Float32(660.0), Float32(665.0), Float32(670.0), Float32(675.0),
    Float32(680.0), Float32(685.0), Float32(690.0), Float32(695.0), Float32(700.0), Float32(705.0), Float32(710.0), Float32(715.0), Float32(720.0), Float32(725.0),
    Float32(730.0), Float32(735.0), Float32(740.0), Float32(745.0), Float32(750.0), Float32(755.0), Float32(760.0), Float32(765.0), Float32(770.0), Float32(775.0),
    Float32(780.0)
)

const ARRI_ALEXA_RED_SENSITIVITIES = SVector{81, Float32}(
    Float32(0.000644), Float32(0.000763), Float32(0.000902), Float32(0.001067), Float32(0.001261), Float32(0.001487), Float32(0.001748), Float32(0.002047), Float32(0.002385), Float32(0.002765),
    Float32(0.003188), Float32(0.003657), Float32(0.004173), Float32(0.004738), Float32(0.005354), Float32(0.006022), Float32(0.006744), Float32(0.007522), Float32(0.008357), Float32(0.009251),
    Float32(0.010206), Float32(0.011223), Float32(0.012304), Float32(0.01345), Float32(0.014662), Float32(0.015942), Float32(0.017292), Float32(0.018713), Float32(0.020206), Float32(0.021773),
    Float32(0.023415), Float32(0.025134), Float32(0.02693), Float32(0.028804), Float32(0.030758), Float32(0.032793), Float32(0.03491), Float32(0.03711), Float32(0.039394), Float32(0.041763),
    Float32(0.044218), Float32(0.04676), Float32(0.049389), Float32(0.052105), Float32(0.054908), Float32(0.057798), Float32(0.060775), Float32(0.063838), Float32(0.066987), Float32(0.07022),
    Float32(0.073537), Float32(0.076936), Float32(0.080416), Float32(0.083975), Float32(0.087612), Float32(0.091323), Float32(0.095106), Float32(0.098957), Float32(0.102873), Float32(0.106849),
    Float32(0.110881), Float32(0.114964), Float32(0.119092), Float32(0.123259), Float32(0.127457), Float32(0.131677), Float32(0.135911), Float32(0.140149), Float32(0.144382), Float32(0.148599),
    Float32(0.152791), Float32(0.156946), Float32(0.161054), Float32(0.165099), Float32(0.169068), Float32(0.172947), Float32(0.17672), Float32(0.180371), Float32(0.183881), Float32(0.187231),
    Float32(0.190723)
)

const ARRI_ALEXA_GREEN_SENSITIVITIES = SVector{81, Float32}(
    Float32(0.005604), Float32(0.007637), Float32(0.010577), Float32(0.014669), Float32(0.020268), Float32(0.028269), Float32(0.040129), Float32(0.057823), Float32(0.083488), Float32(0.120036),
    Float32(0.170999), Float32(0.239762), Float32(0.330113), Float32(0.444974), Float32(0.586567), Float32(0.753397), Float32(0.942995), Float32(1.149094), Float32(1.365976), Float32(1.581801),
    Float32(1.786737), Float32(1.969697), Float32(2.123336), Float32(2.244142), Float32(2.332868), Float32(2.392533), Float32(2.426513), Float32(2.439999), Float32(2.436911), Float32(2.419265),
    Float32(2.386830), Float32(2.338351), Float32(2.272334), Float32(2.191483), Float32(2.093576), Float32(1.984077), Float32(1.865257), Float32(1.738163), Float32(1.607048), Float32(1.472950),
    Float32(1.339062), Float32(1.207238), Float32(1.078365), Float32(0.955463), Float32(0.839048), Float32(0.729420), Float32(0.629723), Float32(0.539267), Float32(0.458577), Float32(0.387290),
    Float32(0.325124), Float32(0.271479), Float32(0.226019), Float32(0.187846), Float32(0.156251), Float32(0.130496), Float32(0.109720), Float32(0.093155), Float32(0.080125), Float32(0.069992),
    Float32(0.062194), Float32(0.056288), Float32(0.051855), Float32(0.048577), Float32(0.046199), Float32(0.044546), Float32(0.043441), Float32(0.042754), Float32(0.042373), Float32(0.042219),
    Float32(0.042224), Float32(0.042337), Float32(0.042522), Float32(0.042750), Float32(0.042996), Float32(0.043248), Float32(0.043496), Float32(0.043729), Float32(0.043941), Float32(0.044128),
    Float32(0.044299)
)

const ARRI_ALEXA_BLUE_SENSITIVITIES = SVector{81, Float32}(
    Float32(0.007782), Float32(0.009992), Float32(0.012735), Float32(0.016147), Float32(0.020394), Float32(0.025674), Float32(0.032222), Float32(0.040311), Float32(0.050257), Float32(0.062426),
    Float32(0.077255), Float32(0.095257), Float32(0.117022), Float32(0.143222), Float32(0.17461), Float32(0.212003), Float32(0.256284), Float32(0.308403), Float32(0.369376), Float32(0.440215),
    Float32(0.521904), Float32(0.615398), Float32(0.721595), Float32(0.841278), Float32(0.975096), Float32(1.123572), Float32(1.286976), Float32(1.465378), Float32(1.658598), Float32(1.866205),
    Float32(2.087492), Float32(2.321478), Float32(2.566915), Float32(2.82232), Float32(3.086027), Float32(3.356186), Float32(3.630819), Float32(3.907763), Float32(4.184748), Float32(4.45948),
    Float32(4.729637), Float32(4.992896), Float32(5.246938), Float32(5.489485), Float32(5.718295), Float32(5.931193), Float32(6.126092), Float32(6.300985), Float32(6.45394), Float32(6.583109),
    Float32(6.686745), Float32(6.763198), Float32(6.810982), Float32(6.828797), Float32(6.815532), Float32(6.770186), Float32(6.691864), Float32(6.579786), Float32(6.433267), Float32(6.25173),
    Float32(6.034712), Float32(5.781835), Float32(5.492896), Float32(5.167835), Float32(4.806745), Float32(4.410003), Float32(3.978218), Float32(3.512222), Float32(3.012982), Float32(2.481645),
    Float32(1.920035), Float32(1.330386), Float32(0.715444), Float32(0.079586), Float32(-0.579786), Float32(-1.260482), Float32(-1.959943), Float32(-2.675654), Float32(-3.404943), Float32(-4.145156),
    Float32(-4.553889)
)

function _interpolate_arri_cfa_data(wavelength::Float32, cfa_wavelengths::SVector{81, Float32}, cfa_sensitivities::SVector{81, Float32})::Float32
    if wavelength < cfa_wavelengths[1] || wavelength > cfa_wavelengths[end]
        return Float32(0.0)
    end

    idx_upper = findfirst(wl -> wl >= wavelength, cfa_wavelengths)
    
    if idx_upper === nothing
        return Float32(0.0)
    end
    if cfa_wavelengths[idx_upper] == wavelength
        return cfa_sensitivities[idx_upper]
    end
    if idx_upper == 1
         return Float32(0.0) 
    end

    idx_lower = idx_upper - 1
    
    wl_lower = cfa_wavelengths[idx_lower]
    wl_upper = cfa_wavelengths[idx_upper]
    sens_lower = cfa_sensitivities[idx_lower]
    sens_upper = cfa_sensitivities[idx_upper]

    t = (wavelength - wl_lower) / (wl_upper - wl_lower)
    return sens_lower + t * (sens_upper - sens_lower)
end

# -----------------------------------------------------------------------------
# CIE Standard Illuminant D65
# -----------------------------------------------------------------------------

# Original sampled data at 5nm intervals
const D65_5NM_SAMPLE_WAVELENGTHS = CIE_1931_5NM_SAMPLE_WAVELENGTHS

const D65_5NM_SAMPLE_VALUES = Float32[
    44.80, 49.40, 54.00, 57.70, 61.40, 65.00, 84.00, 94.80, 100.50, 101.40, 100.80, 110.00,
    117.00, 119.40, 119.00, 119.60, 120.20, 117.80, 115.30, 115.70, 116.10, 111.70, 107.20, 106.70,
    106.10, 105.00, 103.80, 104.60, 105.30, 103.80, 102.20, 102.80, 103.30, 102.10, 100.80, 99.80,
    100.00, 99.50, 98.90, 98.40, 97.80, 95.50, 93.10, 92.10, 91.00, 89.70, 88.30, 88.00,
    87.60, 84.80, 81.90, 83.60, 85.20, 82.30, 79.30, 80.40, 81.40, 78.80, 76.10, 74.30,
    72.40, 68.10, 63.70, 66.20, 68.60, 66.30, 63.90, 61.80, 59.60, 59.30, 58.90, 53.40,
    47.80, 50.70, 53.50, 50.60, 47.60, 47.30, 46.90, 46.90, 46.90
]

# Interpolate D65 to our WAVELENGTHS_NM
const ILLUMINANT_D65_RELATIVE = interpolate_array(WAVELENGTHS_NM, D65_5NM_SAMPLE_WAVELENGTHS, D65_5NM_SAMPLE_VALUES)

# Calculate the normalization factor k = 1 / (sum(D65_rel * y_bar * delta_lambda))
# For D65, this should result in Y=100 for a perfect reflector (S=1)
const N_D65 = sum(ILLUMINANT_D65_RELATIVE .* CIE_Y) * DELTA_WAVELENGTH_NM
const SPECTRAL_NORMALIZATION_K = Float32(100.0) / N_D65

# Normalized D65 SPD
const ILLUMINANT_D65_NORM = ILLUMINANT_D65_RELATIVE

# -----------------------------------------------------------------------------
# Color Space Conversion Matrices
# -----------------------------------------------------------------------------

# XYZ to Linear sRGB (D65 white point)
const XYZ_TO_SRGB_MATRIX = Float32[
     3.24096994 -1.53738318 -0.49861076;
    -0.96924364  1.8759675   0.04155506;
     0.05563008 -0.20397706  1.05697151
]

# XYZ to Linear ACEScg (AP1 primaries, ACES D60 white point ~ D65)
const XYZ_TO_ACESCG_MATRIX = Float32[
     1.6410233797 -0.3248032942 -0.2364246952;
    -0.6636628587  1.6153315917  0.0167563477;
     0.0117218943 -0.0082844420  0.9883948585
]

# ACEScg to XYZ Matrix (Inverse)
const ACESCG_TO_XYZ_MATRIX = inv(XYZ_TO_ACESCG_MATRIX)

# -----------------------------------------------------------------------------
# Basic Math Utilities
# -----------------------------------------------------------------------------

# Vector math using LinearAlgebra and StaticArrays
dot(v1::Vec3f, v2::Vec3f) = LinearAlgebra.dot(v1, v2)

# Normalize with safe handling of zero vectors
function normalize_safe(v::Vec3f)
    len = norm(v)
    if len > Float32(1.0e-8)
        return v / len
    else
        return Vec3f(0, 0, 1) # Default up direction
    end
end

# Reflect a vector around a normal
reflect(v::Vec3f, n::Vec3f) = v - 2 * dot(v, n) * n

# -----------------------------------------------------------------------------
# Spectral Conversion Utilities
# -----------------------------------------------------------------------------

"""
    spectral_to_xyz(spd::Array{Float32})

Convert a spectral power distribution to CIE XYZ (D65 illuminant).
Input spd shape: (..., N_WAVELENGTHS)
Output xyz shape: (..., 3)
"""
function spectral_to_xyz(spd::Array{Float32})
    # Reshape spd to 2D for easier processing if needed
    orig_dims = size(spd)
    if length(orig_dims) > 1
        # Collapse all dimensions except the last (spectral) dimension
        spd_2d = reshape(spd, prod(orig_dims[1:end-1]), N_WAVELENGTHS)
    else
        # Already a single spectral distribution
        spd_2d = reshape(spd, 1, N_WAVELENGTHS)
    end
    
    # Preallocate result
    num_spectra = size(spd_2d, 1)
    xyz = zeros(Float32, num_spectra, 3)
    
    # For each spectrum, compute XYZ
    for i in 1:num_spectra
        for j in 1:3 # X, Y, Z components
            # Calculate XYZ by integrating product of SPD, illuminant and CMF
            xyz[i, j] = sum(spd_2d[i, :] .* ILLUMINANT_D65_NORM .* CIE_XYZ_MATCHING_FUNCTIONS[:, j]) * 
                        SPECTRAL_NORMALIZATION_K * DELTA_WAVELENGTH_NM
        end
    end
    
    # Reshape back to original dimensions, replacing spectral dim with 3 (XYZ)
    if length(orig_dims) > 1
        return reshape(xyz, orig_dims[1:end-1]..., 3)
    else
        # For single spectrum, return a flat array of length 3
        return xyz[1, :]
    end
end

"""
    xyz_to_rgb(xyz::Array{Float32}, color_space_matrix::Matrix{Float32})

Convert CIE XYZ to a linear RGB color space using the provided matrix.
Input xyz shape: (..., 3)
Output rgb shape: (..., 3)
"""
function xyz_to_rgb(xyz::Array{Float32}, color_space_matrix::Matrix{Float32})
    # Reshape xyz to 2D for easier processing if needed
    orig_dims = size(xyz)
    if length(orig_dims) > 1
        # Collapse all dimensions except the last (XYZ) dimension
        xyz_2d = reshape(xyz, prod(orig_dims[1:end-1]), 3)
    else
        # Already a single XYZ value
        xyz_2d = reshape(xyz, 1, 3)
    end
    
    # Apply color space transformation matrix
    rgb_2d = xyz_2d * transpose(color_space_matrix)
    
    # Reshape back to original dimensions
    if length(orig_dims) > 1
        return reshape(rgb_2d, orig_dims)
    else
        # For single xyz value, return a flat array of length 3
        return rgb_2d[1, :]
    end
end

"""
    linear_rgb_to_srgb(rgb_linear::Array{Float32})

Apply sRGB gamma correction to linear RGB values.
Input and output have the same shape.
"""
function linear_rgb_to_srgb(rgb_linear::Array{Float32})
    # Apply sRGB transfer function
    a = Float32(0.055)
    
    # Broadcasting version to handle arrays
    return map(rgb_linear) do x
        if x <= Float32(0.0031308)
            return Float32(12.92) * x
        else
            return (Float32(1.0) + a) * x^(Float32(1.0/2.4)) - a
        end
    end
end

"""
    rgb_to_spd_d65_approx(rgb::Array{Float32})

Approximate a spectrum from RGB values using D65 illuminant.
This is a basic conversion - for visualization purposes, not physically accurate.
Input rgb shape: (..., 3)
Output spd shape: (..., N_WAVELENGTHS)
"""
function rgb_to_spd_d65_approx(rgb::Array{Float32})
    # Reshape rgb to 2D for easier processing if needed
    orig_dims = size(rgb)
    if length(orig_dims) > 1
        # Collapse all dimensions except the last (RGB) dimension
        rgb_2d = reshape(rgb, prod(orig_dims[1:end-1]), 3)
    else
        # Already a single RGB value
        rgb_2d = reshape(rgb, 1, 3)
    end
    
    # First convert RGB to XYZ
    xyz_2d = rgb_2d * transpose(inv(XYZ_TO_SRGB_MATRIX))
    
    # Preallocate result
    num_colors = size(rgb_2d, 1)
    spd = zeros(Float32, num_colors, N_WAVELENGTHS)
    
    # For each color, compute an approximate SPD
    for i in 1:num_colors
        # Approximate SPD by distributing XYZ across wavelengths
        # This is a very basic approach using the CIE matching functions
        for λ in 1:N_WAVELENGTHS
            # Weight by matching functions
            x_contrib = xyz_2d[i, 1] * CIE_X[λ] / sum(CIE_X)
            y_contrib = xyz_2d[i, 2] * CIE_Y[λ] / sum(CIE_Y)
            z_contrib = xyz_2d[i, 3] * CIE_Z[λ] / sum(CIE_Z)
            
            # Simple additive approximation
            spd[i, λ] = (x_contrib + y_contrib + z_contrib)
        end
        
        # Normalize to ensure valid SPD
        if maximum(spd[i, :]) > 0
            spd[i, :] = spd[i, :] / maximum(spd[i, :])
        end
    end
    
    # Reshape back to original dimensions, replacing RGB dim with N_WAVELENGTHS
    if length(orig_dims) > 1
        new_dims = (orig_dims[1:end-1]..., N_WAVELENGTHS)
        return reshape(spd, new_dims)
    else
        # For single RGB value, return a flat array of length N_WAVELENGTHS
        return spd[1, :]
    end
end

# -----------------------------------------------------------------------------
# Sampling Utilities
# -----------------------------------------------------------------------------

"""
    random_in_unit_sphere(rng::AbstractRNG=Random.GLOBAL_RNG)

Generate a random point inside a unit sphere using rejection sampling.
"""
function random_in_unit_sphere(rng::AbstractRNG=Random.GLOBAL_RNG)
    while true
        # Generate random numbers directly
        rx = rand(rng, Float32)
        ry = rand(rng, Float32)
        rz = rand(rng, Float32)
        p = Float32(2.0) * Vec3f(rx, ry, rz) - Vec3f(Float32(1.0), Float32(1.0), Float32(1.0))
        if dot(p, p) < Float32(1.0)
            return p
        end
    end
end

"""
    random_unit_vector(rng::AbstractRNG=Random.GLOBAL_RNG)

Generate a random unit vector (point on unit sphere).
"""
function random_unit_vector(rng::AbstractRNG=Random.GLOBAL_RNG)
    normalize(random_in_unit_sphere(rng))
end

"""
    random_cosine_direction(normal::Vec3f, rng::AbstractRNG=Random.GLOBAL_RNG)

Generate a random direction weighted by cosine from a normal vector.
Used for diffuse reflection sampling.
"""
function random_cosine_direction(normal::Vec3f, rng::AbstractRNG=Random.GLOBAL_RNG)
    # Create a coordinate system (tangent space)
    if abs(normal[1]) > Float32(0.9)
        tangent = normalize(Vec3f(Float32(0.0), Float32(1.0), Float32(0.0)) × normal)
    else
        tangent = normalize(Vec3f(Float32(1.0), Float32(0.0), Float32(0.0)) × normal)
    end
    bitangent = normal × tangent

    # Sample a cosine-weighted direction in tangent space
    # Random numbers are treated as constants by Enzyme due to the inactive rule
    r1 = rand(rng, Float32)
    r2 = rand(rng, Float32)

    phi = Float32(2.0) * π * r1
    cos_theta = sqrt(r2)
    sin_theta = sqrt(Float32(1.0) - r2)

    # Convert to Cartesian coordinates
    x = cos(phi) * sin_theta
    y = sin(phi) * sin_theta
    z = cos_theta

    # Transform to world space
    return tangent * x + bitangent * y + normal * z
end

# -----------------------------------------------------------------------------
# Ray-Triangle Intersection
# -----------------------------------------------------------------------------

"""
    intersect_ray_triangle_moller_trumbore(ray_origin, ray_direction, v0, v1, v2; 
                                          t_min=1.0e-8, t_max=Inf,
                                          backface_culling=false)

Computes ray-triangle intersection using the Möller-Trumbore algorithm.
Returns (t, u, v, valid_hit)
"""
function intersect_ray_triangle_moller_trumbore(
    ray_origin::Vec3f, ray_direction::Vec3f, 
    v0::Vec3f, v1::Vec3f, v2::Vec3f;
    t_min::Float32=Float32(1.0e-8), t_max::Float32=Float32(Inf),
    backface_culling::Bool=false
)
    # Compute edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # Calculate determinant
    pvec = ray_direction × edge2
    det = dot(edge1, pvec)
    
    # Check if ray is parallel to triangle
    if backface_culling
        # Only consider front-facing triangles
        if det < Float32(1.0e-8)
            return Float32(Inf), Float32(0.0), Float32(0.0), false
        end
    else
        # Consider both front and back faces
        if abs(det) < Float32(1.0e-8)
            return Float32(Inf), Float32(0.0), Float32(0.0), false
        end
    end
    
    inv_det = Float32(1.0) / det
    
    # Calculate barycentric coordinates
    tvec = ray_origin - v0
    u = dot(tvec, pvec) * inv_det
    
    # Check if barycentric u is outside range [0,1]
    if u < Float32(0.0) || u > Float32(1.0)
        return Float32(Inf), u, Float32(0.0), false
    end
    
    qvec = tvec × edge1
    v = dot(ray_direction, qvec) * inv_det
    
    # Check if barycentric v is outside range [0,1] or u+v > 1
    if v < Float32(0.0) || u + v > Float32(1.0)
        return Float32(Inf), u, v, false
    end
    
    # Calculate t (intersection distance)
    t = dot(edge2, qvec) * inv_det
    
    # Check if intersection is within valid range
    if t < t_min || t > t_max
        return Float32(Inf), u, v, false
    end
    
    return t, u, v, true
end