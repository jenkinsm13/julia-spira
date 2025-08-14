import jax
import jax.numpy as jnp
from jax import jit, lax
# Import types and spectral constants from types.py
from .types import Spectrum, N_WAVELENGTHS, WAVELENGTHS_NM, DELTA_WAVELENGTH_NM 
from .spd_library import create_spectrum_profile
# <<< ADD functools import >>>
from functools import partial
# <<< END ADD >>>
# <<< ADD typing import >>>
from typing import Tuple
# <<< END ADD >>>
import os # Added
import numpy as np # Added
import pywavefront # Added

# --- Spectral Rendering Configuration --- REMOVED
# N_WAVELENGTHS = 8
# MIN_WAVELENGTH_NM = 400.0
# MAX_WAVELENGTH_NM = 700.0
# WAVELENGTHS_NM = jnp.linspace(MIN_WAVELENGTH_NM, MAX_WAVELENGTH_NM, N_WAVELENGTHS)
# DELTA_WAVELENGTH_NM = (MAX_WAVELENGTH_NM - MIN_WAVELENGTH_NM) / (N_WAVELENGTHS - 1) if N_WAVELENGTHS > 1 else 1.0

# --- CIE 1931 2-degree Standard Observer Color Matching Functions ---
# Data sourced from CIE via CVRL (http://cvrl.ioo.ucl.ac.uk/database/text/cmfs/ciexyz31_5.htm)
# Wavelengths: 380 to 780 nm at 5 nm steps (81 samples)

CIE_1931_5NM_SAMPLE_WAVELENGTHS = jnp.array([
    380., 385., 390., 395., 400., 405., 410., 415., 420., 425., 430., 435., 
    440., 445., 450., 455., 460., 465., 470., 475., 480., 485., 490., 495., 
    500., 505., 510., 515., 520., 525., 530., 535., 540., 545., 550., 555., 
    560., 565., 570., 575., 580., 585., 590., 595., 600., 605., 610., 615., 
    620., 625., 630., 635., 640., 645., 650., 655., 660., 665., 670., 675., 
    680., 685., 690., 695., 700., 705., 710., 715., 720., 725., 730., 735., 
    740., 745., 750., 755., 760., 765., 770., 775., 780. 
])

# Corrected full 81x3 array
CIE_1931_5NM_SAMPLE_VALUES = jnp.array([
    [0.0014, 0.0000, 0.0065], # 380 nm
    [0.0022, 0.0001, 0.0105], 
    [0.0042, 0.0001, 0.0201], 
    [0.0076, 0.0002, 0.0362], 
    [0.0143, 0.0004, 0.0679], # 400 nm
    [0.0232, 0.0006, 0.1102], 
    [0.0435, 0.0012, 0.2074], 
    [0.0776, 0.0022, 0.3713], 
    [0.1344, 0.0040, 0.6456], # 420 nm
    [0.2148, 0.0073, 1.0391], 
    [0.2839, 0.0116, 1.3856], 
    [0.3285, 0.0168, 1.6230], 
    [0.3483, 0.0230, 1.7471], # 440 nm
    [0.3481, 0.0298, 1.7826], 
    [0.3362, 0.0380, 1.7721], 
    [0.3187, 0.0480, 1.7441], 
    [0.2908, 0.0600, 1.6692], # 460 nm
    [0.2511, 0.0739, 1.5281], 
    [0.1954, 0.0910, 1.2876], 
    [0.1421, 0.1126, 1.0419], 
    [0.0956, 0.1390, 0.8130], # 480 nm
    [0.0580, 0.1693, 0.6162], 
    [0.0320, 0.2080, 0.4652], 
    [0.0147, 0.2586, 0.3533], 
    [0.0049, 0.3230, 0.2720], # 500 nm
    [0.0024, 0.4073, 0.2123], 
    [0.0093, 0.5030, 0.1582], 
    [0.0291, 0.6082, 0.1117], 
    [0.0633, 0.7100, 0.0782], # 520 nm
    [0.1096, 0.7932, 0.0573], 
    [0.1655, 0.8620, 0.0422], 
    [0.2257, 0.9149, 0.0298], 
    [0.2904, 0.9540, 0.0203], # 540 nm
    [0.3597, 0.9803, 0.0134], 
    [0.4334, 0.9950, 0.0087], 
    [0.5121, 1.0000, 0.0057], 
    [0.5945, 0.9950, 0.0039], # 560 nm
    [0.6784, 0.9786, 0.0027], 
    [0.7621, 0.9520, 0.0021], 
    [0.8425, 0.9154, 0.0018], 
    [0.9163, 0.8700, 0.0017], # 580 nm
    [0.9786, 0.8163, 0.0014], 
    [1.0263, 0.7570, 0.0011], 
    [1.0567, 0.6949, 0.0008], 
    [1.0622, 0.6310, 0.0006], # 600 nm
    [1.0456, 0.5668, 0.0003], 
    [1.0026, 0.5030, 0.0002], 
    [0.9384, 0.4412, 0.0001], 
    [0.8544, 0.3810, 0.0001], # 620 nm
    [0.7514, 0.3210, 0.0000], 
    [0.6424, 0.2650, 0.0000], 
    [0.5419, 0.2170, 0.0000], 
    [0.4479, 0.1750, 0.0000], # 640 nm
    [0.3608, 0.1382, 0.0000], 
    [0.2835, 0.1070, 0.0000], 
    [0.2187, 0.0816, 0.0000], 
    [0.1649, 0.0610, 0.0000], # 660 nm
    [0.1212, 0.0446, 0.0000], 
    [0.0874, 0.0320, 0.0000], 
    [0.0636, 0.0232, 0.0000], 
    [0.0468, 0.0170, 0.0000], # 680 nm
    [0.0329, 0.0119, 0.0000], 
    [0.0227, 0.0082, 0.0000], 
    [0.0158, 0.0057, 0.0000], 
    [0.0114, 0.0041, 0.0000], # 700 nm
    [0.0081, 0.0029, 0.0000], 
    [0.0058, 0.0021, 0.0000], 
    [0.0041, 0.0015, 0.0000], 
    [0.0029, 0.0010, 0.0000], # 720 nm
    [0.0020, 0.0007, 0.0000], 
    [0.0014, 0.0005, 0.0000], 
    [0.0010, 0.0004, 0.0000], 
    [0.0007, 0.0002, 0.0000], # 740 nm
    [0.0005, 0.0002, 0.0000], 
    [0.0003, 0.0001, 0.0000], 
    [0.0002, 0.0001, 0.0000], 
    [0.0002, 0.0001, 0.0000], # 760 nm
    [0.0001, 0.0000, 0.0000], 
    [0.0001, 0.0000, 0.0000], 
    [0.0001, 0.0000, 0.0000], 
    [0.0000, 0.0000, 0.0000]  # 780 nm
])

# Interpolate the CMFs to our WAVELENGTHS_NM
CIE_X = jnp.interp(WAVELENGTHS_NM, CIE_1931_5NM_SAMPLE_WAVELENGTHS, CIE_1931_5NM_SAMPLE_VALUES[:, 0])
CIE_Y = jnp.interp(WAVELENGTHS_NM, CIE_1931_5NM_SAMPLE_WAVELENGTHS, CIE_1931_5NM_SAMPLE_VALUES[:, 1])
CIE_Z = jnp.interp(WAVELENGTHS_NM, CIE_1931_5NM_SAMPLE_WAVELENGTHS, CIE_1931_5NM_SAMPLE_VALUES[:, 2])
CIE_XYZ_MATCHING_FUNCTIONS = jnp.stack([CIE_X, CIE_Y, CIE_Z], axis=-1) # Shape: (N_WAVELENGTHS, 3)

# --- CIE Standard Illuminant D65 ---
# Data sourced from CIE via CVRL (derived from ISO/CIE 11664-2:2022)
# Original data typically 300-830nm at 1nm or 5nm steps.
# We need to interpolate this data to our specific WAVELENGTHS_NM.

# Original sampled data (example subset at 5nm intervals)
# Wavelength (nm), Relative SPD
D65_5NM_SAMPLE_WAVELENGTHS = jnp.array([
    380.0, 385.0, 390.0, 395.0, 400.0, 405.0, 410.0, 415.0, 420.0, 425.0, 430.0, 435.0, 
    440.0, 445.0, 450.0, 455.0, 460.0, 465.0, 470.0, 475.0, 480.0, 485.0, 490.0, 495.0, 
    500.0, 505.0, 510.0, 515.0, 520.0, 525.0, 530.0, 535.0, 540.0, 545.0, 550.0, 555.0, 
    560.0, 565.0, 570.0, 575.0, 580.0, 585.0, 590.0, 595.0, 600.0, 605.0, 610.0, 615.0, 
    620.0, 625.0, 630.0, 635.0, 640.0, 645.0, 650.0, 655.0, 660.0, 665.0, 670.0, 675.0, 
    680.0, 685.0, 690.0, 695.0, 700.0, 705.0, 710.0, 715.0, 720.0, 725.0, 730.0, 735.0, 
    740.0, 745.0, 750.0, 755.0, 760.0, 765.0, 770.0, 775.0, 780.0
])
D65_5NM_SAMPLE_VALUES = jnp.array([
    44.80, 49.40, 54.00, 57.70, 61.40, 65.00, 84.00, 94.80, 100.50, 101.40, 100.80, 110.00,
    117.00, 119.40, 119.00, 119.60, 120.20, 117.80, 115.30, 115.70, 116.10, 111.70, 107.20, 106.70,
    106.10, 105.00, 103.80, 104.60, 105.30, 103.80, 102.20, 102.80, 103.30, 102.10, 100.80, 99.80,
    100.00, 99.50, 98.90, 98.40, 97.80, 95.50, 93.10, 92.10, 91.00, 89.70, 88.30, 88.00,
    87.60, 84.80, 81.90, 83.60, 85.20, 82.30, 79.30, 80.40, 81.40, 78.80, 76.10, 74.30,
    72.40, 68.10, 63.70, 66.20, 68.60, 66.30, 63.90, 61.80, 59.60, 59.30, 58.90, 53.40,
    47.80, 50.70, 53.50, 50.60, 47.60, 47.30, 46.90, 46.90, 46.90 # Added missing values for 775nm, 780nm
])
# Normalize D65 so Y=100. Standard data is usually relative, normalized to 100 at 560nm.
# We need the absolute scaling for spectral_to_xyz normalization.
# The standard normalization constant N is sum(I * y_bar * delta_lambda)
# For D65, this should result in Y=100 for a perfect reflector (S=1).

# Interpolate D65 to our WAVELENGTHS_NM
ILLUMINANT_D65_RELATIVE = jnp.interp(WAVELENGTHS_NM, D65_5NM_SAMPLE_WAVELENGTHS, D65_5NM_SAMPLE_VALUES)

# Calculate the normalization factor k = 1 / (sum(D65_rel * y_bar * delta_lambda))
# Note: Using relative D65 values here as per standard practice.
# The integral is approximated by a sum: sum(I_rel * y_bar) * delta_lambda
N_D65 = jnp.sum(ILLUMINANT_D65_RELATIVE * CIE_Y) * DELTA_WAVELENGTH_NM
SPECTRAL_NORMALIZATION_K = 100.0 / N_D65 # Scale factor to ensure Y=100 for perfect reflector under D65

# Normalized D65 SPD to be used in spectral_to_xyz calculation
# This effectively pre-multiplies the illuminant by K / N
ILLUMINANT_D65_NORM = ILLUMINANT_D65_RELATIVE # For lighting calcs, use relative SPD
# Store the normalization constant separately for spectral_to_xyz
# No, let's compute XYZ directly with normalization included.

# --- Color Space Conversion Matrices ---
# Bradford Cone Transformation Matrix from D65 to D50 (used for ICC profiles etc., not directly needed here)
# Chromatic adaptation might be needed for conversions between spaces with different white points.

# XYZ to Linear sRGB (D65 white point)
XYZ_TO_SRGB_MATRIX = jnp.array([
    [ 3.24096994, -1.53738318, -0.49861076],
    [-0.96924364,  1.8759675 ,  0.04155506],
    [ 0.05563008, -0.20397706,  1.05697151]
])

# XYZ to Linear ACEScg (AP1 primaries, ACES D60 white point ~ D65)
# Requires chromatic adaptation if source XYZ is D65. Using direct matrix for simplicity assuming ~D60.
# Source: https://github.com/colour-science/colour/blob/develop/colour/models/dataset/aces.py#L166
# Note: Different sources give slightly different matrices depending on CAT and precision.
XYZ_TO_ACESCG_MATRIX = jnp.array([
    [ 1.6410233797, -0.3248032942, -0.2364246952],
    [-0.6636628587,  1.6153315917,  0.0167563477],
    [ 0.0117218943, -0.0082844420,  0.9883948585]
])

# --- ADD ACEScg to XYZ Matrix (Inverse) ---
ACESCG_TO_XYZ_MATRIX = jnp.linalg.inv(XYZ_TO_ACESCG_MATRIX)
# --- END ADD ---

# --- Basic Math Utilities ---
# (normalize, reflect - keep as is)

# --- Spectral Conversion Utilities ---

@jit
def spectral_to_xyz(spd: jnp.ndarray) -> jnp.ndarray:
    """Convert a spectral power distribution to CIE XYZ (D65 illuminant)."""
    # Assumes spd is the spectral reflectance/transmittance/emission, shape (..., N_WAVELENGTHS)
    # Formula: X = k * sum( S(lambda) * I(lambda) * x_bar(lambda) * delta_lambda )
    # Where k = 100 / sum( I(lambda) * y_bar(lambda) * delta_lambda )
    # We use the relative D65 illuminant and pre-calculated k.

    # Calculate term inside summation: Reflectance * Illuminant * CMF
    # spd[..., None] broadcasts reflectance to match CMF shape (..., N_WAVELENGTHS, 3)
    integrand = spd[..., None] * ILLUMINANT_D65_RELATIVE[:, None] * CIE_XYZ_MATCHING_FUNCTIONS
    
    # Sum over wavelength dimension
    xyz_sum = jnp.sum(integrand, axis=-2) # Sum over N_WAVELENGTHS axis
    
    # Apply normalization factor and interval width
    xyz = xyz_sum * SPECTRAL_NORMALIZATION_K * DELTA_WAVELENGTH_NM
    return xyz

@jit
def xyz_to_rgb(xyz: jnp.ndarray, color_space_matrix: jnp.ndarray) -> jnp.ndarray:
    """Convert CIE XYZ to a linear RGB color space using a matrix."""
    rgb_linear = jnp.einsum('...k,lk->...l', xyz, color_space_matrix)
    return rgb_linear

@jit
def linear_rgb_to_srgb(rgb_linear: jnp.ndarray) -> jnp.ndarray:
    """Apply sRGB gamma correction."""
    a = 0.055
    return jnp.where(
        rgb_linear <= 0.0031308,
        12.92 * rgb_linear,
        (1.0 + a) * jnp.power(jnp.maximum(rgb_linear, 1e-9), 1.0 / 2.4) - a # Increased epsilon slightly
    )

# --- Polarization Utilities ---
# (IDENTITY_MUELLER_SPECTRAL, DEPOLARIZER_MUELLER_SPECTRAL - keep as is for now)
IDENTITY_MUELLER = jnp.eye(4)
IDENTITY_MUELLER_SPECTRAL = jnp.tile(IDENTITY_MUELLER[None, :, :], (N_WAVELENGTHS, 1, 1))

DEPOLARIZER_MUELLER = jnp.zeros((4, 4)).at[0, 0].set(1.0)
DEPOLARIZER_MUELLER_SPECTRAL = jnp.tile(DEPOLARIZER_MUELLER[None, :, :], (N_WAVELENGTHS, 1, 1))

# --- Vector Utilities ---
def dot(v1, v2):
    return jnp.sum(v1 * v2, axis=-1)

def normalize(v):
    """Normalize a vector."""
    # Add epsilon to avoid division by zero for zero-length vectors
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.maximum(norm, 1e-6) 

def reflect(v, n):
    """Reflect vector v around normal n."""
    return v - 2 * dot(v, n)[..., None] * n 

# --- JAX Ray-Triangle Intersection --- 
@partial(jax.jit, static_argnames=('backface_culling',))
def intersect_ray_triangle_moller_trumbore(
    ray_origin: jnp.ndarray, ray_direction: jnp.ndarray, 
    v0: jnp.ndarray, v1: jnp.ndarray, v2: jnp.ndarray, 
    t_min: float = 1e-6, t_max: float = jnp.inf,
    backface_culling: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes ray-triangle intersection using the Moller-Trumbore algorithm.
    Args:
        ray_origin (3,): Ray origin.
        ray_direction (3,): Ray direction (normalized).
        v0, v1, v2 (3,): Triangle vertices.
        t_min, t_max: Min/max valid intersection distance.
        backface_culling: If True, ignore hits where ray hits back face.
    Returns:
        Tuple: (t, u, v, valid_hit)
            t (float): Intersection distance (inf if no hit).
            u (float): Barycentric coordinate (relative to v1).
            v (float): Barycentric coordinate (relative to v2).
            valid_hit (bool): True if a valid intersection occurred within [t_min, t_max].
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = jnp.cross(ray_direction, edge2)
    det = jnp.dot(edge1, pvec)

    # If determinant is near zero, ray lies in plane of triangle or is parallel
    # Handle potential backface culling
    if backface_culling:
        valid_det = det > 1e-6
    else:
        valid_det = jnp.abs(det) > 1e-6

    # Calculate results assuming valid determinant, otherwise return defaults
    def calculate_hit(_): # Accept one (unused) argument for lax.cond
        inv_det = 1.0 / det
        tvec = ray_origin - v0
        u = jnp.dot(tvec, pvec) * inv_det
        qvec = jnp.cross(tvec, edge1)
        v = jnp.dot(ray_direction, qvec) * inv_det
        t = jnp.dot(edge2, qvec) * inv_det

        # Check barycentric coordinates and distance range
        bary_valid = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
        t_valid = (t >= t_min) & (t <= t_max)
        hit_valid_internal = bary_valid & t_valid

        # If hit is not valid, return t=inf
        t_final = jnp.where(hit_valid_internal, t, jnp.inf)
        return t_final, u, v, hit_valid_internal

    # Use lax.cond to handle determinant check without Python branching
    t, u, v, valid_hit = lax.cond(
        valid_det,
        calculate_hit, # Execute if determinant is valid
        lambda _: (jnp.inf, 0.0, 0.0, False), # Return defaults if determinant is invalid (accept dummy arg)
        operand=None # Operand is passed to whichever function is chosen
    )

    return t, u, v, valid_hit
# ------------------------------------

# --- Coordinate System & Vector Operations ---
# ... (existing vector functions like dot, normalize etc.) ...

# Small epsilon for safe math operations
EPSILON = 1e-6 

@jax.jit
def calculate_sphere_uv(hit_point_world, sphere_center):
    """
    Calculates UV coordinates for a point on a unit sphere centered at the origin
    using spherical mapping. Assumes Z is up for the texture mapping convention.
    Input point is assumed to be already normalized (on the unit sphere surface).

    Args:
        hit_point_world: (3,) array, the point on the sphere surface in world space.
        sphere_center: (3,) array, the center of the sphere in world space.

    Returns:
        (u, v): (2,) array, the UV coordinates in the range [0, 1].
    """
    # Calculate vector from center to hit point and normalize
    # Assuming hit_point_world is already on the sphere surface relative to center simplifies things
    # For generality, calculate relative point and normalize it
    relative_hit_point = hit_point_world - sphere_center
    # Use a small epsilon to prevent division by zero if hit is exactly center (though unlikely for surface point)
    normalized_hit = normalize(relative_hit_point)

    x, y, z = normalized_hit[0], normalized_hit[1], normalized_hit[2]

    # Calculate spherical coordinates (phi: azimuth, theta: inclination/polar angle)
    # Phi = atan2(y, x) -> maps to [-pi, pi]
    # Theta = acos(z) -> maps to [0, pi]
    
    phi = jnp.arctan2(y, x)  # Azimuthal angle
    
    # Ensure z is within [-1, 1] for acos due to potential floating point inaccuracies
    safe_z = jnp.clip(z, -1.0, 1.0) 
    theta = jnp.arccos(safe_z) # Polar angle from +Z axis

    # Map spherical coordinates to UV [0, 1]
    # u = phi / (2 * pi) + 0.5  => Maps [-pi, pi] to [0, 1]
    # v = theta / pi            => Maps [0, pi] to [0, 1] (v=0 at +Z pole, v=1 at -Z pole)
    
    u = phi / (2.0 * jnp.pi) + 0.5
    # Invert v so v=0 is bottom pole (-Z), v=1 is top pole (+Z), matching common texture map origins.
    v = 1.0 - (theta / jnp.pi)

    # JAX handles modulo correctly for potential edge cases near phi = +/- pi mapping to u=0/1
    # Ensure u is in [0, 1), often handled by lookup function\'s wrapping/clamping
    # For safety, apply modulo 1.0? Might interfere with UDIM logic later if not careful.
    # u = u % 1.0 # Let's omit this for now and rely on lookup handling.
    
    # v should naturally be in [0, 1]

    return jnp.array([u, v])

@jax.jit
def sphere_normal(point_on_surface: jnp.ndarray, sphere_center: jnp.ndarray) -> jnp.ndarray:
    """Calculates the outward-facing normal vector for a point on a sphere surface."""
    # Vector from center to point
    normal_vec = point_on_surface - sphere_center
    return normalize(normal_vec)

@jax.jit
def sample_sphere_surface(rng_key: jax.random.PRNGKey, sphere_center: jnp.ndarray, sphere_radius: float) -> jnp.ndarray:
    """
    Uniformly samples a point on the surface of a sphere.

    Args:
        rng_key: JAX PRNG key.
        sphere_center: (3,) array, center of the sphere.
        sphere_radius: Float, radius of the sphere.

    Returns:
        (3,) array, a point on the sphere surface.
    """
    key1, key2, key3 = jax.random.split(rng_key, 3)
    
    # Sample using Marsaglia's method for uniform sphere surface sampling
    # Generate points in a cube until one is inside the unit sphere
    # Then normalize to project onto the surface.
    
    # This loop is problematic for JIT. Alternative: Use normal distribution method.
    # x, y, z = jax.random.normal(key1, (3,))
    # norm = jnp.linalg.norm(jnp.array([x, y, z]))
    # point_on_unit_sphere = jnp.array([x, y, z]) / jnp.maximum(norm, EPSILON)

    # Use spherical coordinates method (simpler math for uniform surface)
    u = jax.random.uniform(key1, minval=0.0, maxval=1.0) # Azimuthal angle component
    v = jax.random.uniform(key2, minval=0.0, maxval=1.0) # Polar angle component

    phi = 2.0 * jnp.pi * u
    cos_theta = 1.0 - 2.0 * v # Uniformly distributed cos(theta) in [-1, 1]
    sin_theta = jnp.sqrt(jnp.maximum(1.0 - cos_theta**2, 0.0)) # Ensure non-negative under sqrt

    x = sin_theta * jnp.cos(phi)
    y = sin_theta * jnp.sin(phi)
    z = cos_theta

    point_on_unit_sphere = jnp.array([x, y, z])

    # Scale by radius and translate to center
    point_on_surface = sphere_center + sphere_radius * point_on_unit_sphere
    
    return point_on_surface

# Placeholder: Define a default SPD (e.g., flat grey) if a UDIM tile is missing
# This should ideally match the number of wavelengths used elsewhere
DEFAULT_SPD = jnp.full((N_WAVELENGTHS,), 0.1) # Dark grey default

from jax.scipy.ndimage import map_coordinates
from functools import partial
from typing import Dict, Tuple, List # For type hints

# Refactored texture_lookup_udim_bilinear to avoid recursion error during JIT
def texture_lookup_udim_bilinear(
    # Conceptually static, but passed dynamically due to outer JIT
    udim_keys_sorted: Tuple[int, ...],
    udim_shapes: Tuple[Tuple[int, int, int], ...],

    # Dynamic data
    udim_texture_tiles: Tuple[jnp.ndarray, ...], 
    uv: jnp.ndarray, 
    default_spd: jnp.ndarray
) -> jnp.ndarray:
    """
    Performs differentiable bilinear texture lookup with UDIM support.
    Refactored to avoid JIT recursion errors by selecting data before interpolation.
    """
    # Add guard for empty UDIM data
    if not udim_keys_sorted or not udim_texture_tiles:
        return default_spd

    u, v = uv[0], uv[1]

    # Calculate target UDIM tile index and fractional UVs
    tile_u_idx = jnp.floor(u).astype(jnp.int32)
    tile_v_idx = jnp.floor(v).astype(jnp.int32)
    tile_u_idx = jnp.maximum(0, tile_u_idx) 
    tile_v_idx = jnp.maximum(0, tile_v_idx)
    tile_index = 1001 + tile_u_idx + 10 * tile_v_idx 

    frac_u = u - jnp.floor(u)
    frac_v = v - jnp.floor(v)

    # --- JIT-Compatible Lookup using pre-selection --- 
    present_tile_indices_arr = jnp.array(udim_keys_sorted)
    comparison = (tile_index == present_tile_indices_arr)
    match_found = jnp.any(comparison)
    # Find the index in the static sorted list, default to 0 if not found (will be ignored)
    match_index = jnp.where(match_found, jnp.argmax(comparison), 0)

    # Precompute heights and widths (minus 1, clamped)
    heights_m1 = jnp.array([jnp.maximum(1, s[0] - 1) for s in udim_shapes])
    widths_m1 = jnp.array([jnp.maximum(1, s[1] - 1) for s in udim_shapes])

    # --- Calculate interpolated SPD using the correctly selected tile --- 
    # Select the actual tile and its shape based on match_index.
    # This dynamic indexing into tuples of JAX arrays works because JAX can trace
    # the selection when the array shapes within the tuple are consistent, 
    # or when it can unroll if the tuple is small and contents are static shapes.
    # For this specific case, all UDIM tiles (2x2 SPDs) have the same shape.
    # OLD LINES:
    # actual_texture_tile = udim_texture_tiles[match_index]
    # actual_h_m1 = heights_m1[match_index]
    # actual_w_m1 = widths_m1[match_index]

    # NEW LINES for JIT compatibility:
    # Create branches for lax.switch to select the correct texture tile
    # udim_texture_tiles is a Python tuple of JAX arrays.
    # Only create branches if udim_texture_tiles is not empty, otherwise lax.switch might error.
    if not udim_texture_tiles:
        # This case should ideally be caught by the initial guard, but as a fallback:
        return default_spd
        
    tile_branches = [lambda i=i: udim_texture_tiles[i] for i in range(len(udim_texture_tiles))]
    actual_texture_tile = lax.switch(match_index, tile_branches)

    # heights_m1 and widths_m1 are already JAX arrays, so direct indexing is JIT-compatible.
    actual_h_m1 = heights_m1[match_index]
    actual_w_m1 = widths_m1[match_index]
    
    coords_y = frac_v * actual_h_m1
    coords_x = frac_u * actual_w_m1
    coords_yx = jnp.array([coords_y, coords_x])
    
    # Reshape the selected tile for vmap over channels
    texture_reshaped_actual = jnp.moveaxis(actual_texture_tile, -1, 0) # (C, H, W)

    # Define channel lookup for the selected tile
    def lookup_channel_actual(channel_data): 
         return map_coordinates(channel_data, coords_yx, order=1, mode='nearest') 

    # Perform bilinear interpolation on the selected tile
    interpolated_spd_actual = jax.vmap(lookup_channel_actual)(texture_reshaped_actual)

    # --- Select result based on match_found using lax.where ---
    # If match_found is true, use the interpolated SPD from the correct tile, otherwise use default_spd.
    selected_spd = jnp.where(match_found, interpolated_spd_actual, default_spd)

    return selected_spd

# --- Sampling Utilities ---

@jax.jit
def sample_cosine_weighted_hemisphere(normal: jnp.ndarray, rng_key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample a direction from a cosine-weighted hemisphere oriented around the normal.

    Args:
        normal: The surface normal vector (shape 3,).
        rng_key: JAX PRNG key.

    Returns:
        A tuple containing:
            - sampled_direction: The sampled direction vector (shape 3,).
            - pdf: The probability density function of the sample (cosine(theta) / pi).
    """
    # Generate two uniform random numbers
    key1, key2 = jax.random.split(rng_key)
    u1 = jax.random.uniform(key1)
    u2 = jax.random.uniform(key2)

    # Map uniform random numbers to spherical coordinates (phi, theta)
    # for cosine-weighted hemisphere sampling
    phi = 2.0 * jnp.pi * u1
    # cos_theta = sqrt(1 - u2) -> Incorrect, should be cos_theta = sqrt(u2)
    # Correct mapping: cos(theta)^2 = u2 => cos(theta) = sqrt(u2)
    cos_theta = jnp.sqrt(u2)
    sin_theta = jnp.sqrt(1.0 - cos_theta*cos_theta) # sin = sqrt(1 - cos^2)

    # Convert spherical coordinates to Cartesian coordinates in a local frame
    # where the z-axis aligns with the normal
    local_x = sin_theta * jnp.cos(phi)
    local_y = sin_theta * jnp.sin(phi)
    local_z = cos_theta
    local_dir = jnp.array([local_x, local_y, local_z])

    # Create an orthonormal basis (ONB) around the normal
    # Use a robust method to create tangent and bitangent
    up_vector = jnp.where(jnp.abs(normal[2]) < 0.999, jnp.array([0.0, 0.0, 1.0]), jnp.array([1.0, 0.0, 0.0]))
    tangent = normalize(jnp.cross(up_vector, normal))
    bitangent = jnp.cross(normal, tangent)

    # Transform the local direction to world coordinates using the ONB
    # world_dir = tangent * local_x + bitangent * local_y + normal * local_z
    onb_matrix = jnp.stack([tangent, bitangent, normal], axis=-1) # Columns form the basis
    sampled_direction = jnp.dot(onb_matrix, local_dir)
    
    # The PDF for cosine-weighted hemisphere sampling is cos(theta) / pi
    # cos(theta) is the dot product of the sampled direction and the normal
    # (or simply local_z if the normal is the z-axis)
    # Ensure pdf is not zero if direction is valid
    pdf = jnp.maximum(cos_theta / jnp.pi, 1e-9) 

    return normalize(sampled_direction), pdf

# --- Spectral Data & Conversions --- 

# --- Mesh Data Loading ---
def load_mesh_data(obj_path):
    """
    Loads mesh data (vertices, faces, UVs, normals) from an OBJ file using pywavefront.
    Relies on pywavefront's material processing to get interleaved data.
    Returns a dictionary containing the data as JAX arrays.
    """
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    try:
        print(f"Loading mesh from {obj_path}... (Using interleaved format)")
        # Enable create_materials=True to get per-material vertex data
        scene = pywavefront.Wavefront(obj_path, collect_faces=True, create_materials=True, parse=True)

        all_vertices = []
        all_uvs = []
        all_normals = [] # Store normals if available
        all_faces = []
        vertex_offset = 0

        print(f"Processing {len(scene.mesh_list)} meshes in OBJ...")
        for mesh_item in scene.mesh_list: # Renamed mesh to mesh_item to avoid conflict
            print(f"  Processing mesh with {len(mesh_item.materials)} materials...")
            for material in mesh_item.materials:
                vertex_data = np.array(material.vertices, dtype=np.float32)
                vertex_format = material.vertex_format
                stride = 0
                if 'T' in vertex_format: stride += 2 # T2F
                if 'C' in vertex_format: stride += 3 # C3F (not common)
                if 'N' in vertex_format: stride += 3 # N3F
                if 'V' in vertex_format: stride += 3 # V3F
                if stride == 0: raise ValueError(f"Could not determine vertex stride from format: {vertex_format}")

                if len(vertex_data) % stride != 0:
                    raise ValueError(f"Vertex data length ({len(vertex_data)}) not divisible by calculated stride ({stride}) for format {vertex_format}")
                num_vertices_in_material = len(vertex_data) // stride
                
                # print(f"    Material vertex format: {vertex_format}, Stride: {stride}, Num Verts: {num_vertices_in_material}") # Optional: too verbose

                current_offset = 0
                vt_indices, vn_indices, v_indices = None, None, None
                if 'T' in vertex_format:
                    vt_indices = slice(current_offset, current_offset + 2)
                    current_offset += 2
                if 'C' in vertex_format:
                    current_offset += 3
                if 'N' in vertex_format:
                    vn_indices = slice(current_offset, current_offset + 3)
                    current_offset += 3
                if 'V' in vertex_format:
                    v_indices = slice(current_offset, current_offset + 3)
                    current_offset += 3
                
                reshaped_data = vertex_data.reshape(-1, stride)

                if v_indices is None: continue
                verts = reshaped_data[:, v_indices]
                all_vertices.append(verts)

                if vt_indices is not None:
                    uvs = reshaped_data[:, vt_indices]
                    all_uvs.append(uvs)
                else:
                    all_uvs.append(np.zeros((num_vertices_in_material, 2), dtype=np.float32))
                
                if vn_indices is not None:
                    norms = reshaped_data[:, vn_indices]
                    all_normals.append(norms)
                else:
                     all_normals.append(np.zeros((num_vertices_in_material, 3), dtype=np.float32))

                material_faces = np.arange(vertex_offset, vertex_offset + num_vertices_in_material).reshape(-1, 3)
                all_faces.append(material_faces)
                vertex_offset += num_vertices_in_material

        if vertex_offset == 0:
            # Return empty arrays if no valid data, instead of raising error, to allow scenes without meshes
            print(f"Warning: No valid vertex data found in OBJ file: {obj_path}. Proceeding with empty mesh.")
            return {
                'vertices': jnp.array([]).reshape(0, 3),
                'faces': jnp.array([]).reshape(0, 3),
                'uvs': jnp.array([]).reshape(0, 2),
                'normals': jnp.array([]).reshape(0, 3)
            }

        final_vertices = np.concatenate(all_vertices, axis=0)
        final_uvs = np.concatenate(all_uvs, axis=0)
        final_normals = np.concatenate(all_normals, axis=0)
        final_faces = np.concatenate(all_faces, axis=0)

        print(f"Mesh loaded: {final_vertices.shape[0]} vertices (in faces array), {final_faces.shape[0]} faces.")
        print(f"           UVs shape: {final_uvs.shape}")
        print(f"           Normals shape: {final_normals.shape}")

        return {
            'vertices': jnp.array(final_vertices),
            'faces': jnp.array(final_faces),
            'uvs': jnp.array(final_uvs),
            'normals': jnp.array(final_normals),
        }
        
    except Exception as e:
        print(f"Error loading OBJ file '{obj_path}': {e}")
        import traceback
        traceback.print_exc()
        # Return empty arrays on error as well, to allow rendering to proceed
        print(f"Returning empty mesh data due to error.")
        return {
            'vertices': jnp.array([]).reshape(0, 3),
            'faces': jnp.array([]).reshape(0, 3),
            'uvs': jnp.array([]).reshape(0, 2),
            'normals': jnp.array([]).reshape(0, 3)
        }
# -------------------------