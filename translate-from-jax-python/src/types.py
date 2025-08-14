import jax.numpy as jnp
from flax import struct
# Import field from dataclasses
from dataclasses import field
# from .utils import N_WAVELENGTHS # Import spectral dimension size

# --- Spectral Configuration (Centralized) ---
N_WAVELENGTHS = 31 # e.g., 400nm to 700nm in 10nm steps
MIN_WAVELENGTH_NM = 400.0
MAX_WAVELENGTH_NM = 700.0
WAVELENGTHS_NM = jnp.linspace(MIN_WAVELENGTH_NM, MAX_WAVELENGTH_NM, N_WAVELENGTHS)
DELTA_WAVELENGTH_NM = (MAX_WAVELENGTH_NM - MIN_WAVELENGTH_NM) / (N_WAVELENGTHS - 1) if N_WAVELENGTHS > 1 else 1.0

# --- Type Aliases for Clarity ---
# Use type aliases for JAX arrays representing physical quantities
Spectrum = jnp.ndarray # Represents spectral data (..., N_WAVELENGTHS)
MuellerMatrix = jnp.ndarray # Represents a 4x4 Mueller matrix (..., N_WAVELENGTHS, 4, 4)
StokesVector = jnp.ndarray # Represents a 4-element Stokes vector (..., N_WAVELENGTHS, 4)

# Remove outdated struct definitions
# @struct.dataclass
# class ArrayContainer:
#     data: jnp.ndarray

# @struct.dataclass
# class Spectrum:
#     spd: jnp.ndarray # Spectral Power Distribution, shape (..., N_WAVELENGTHS)

# @struct.dataclass
# class MuellerMatrix:
#     matrix: jnp.ndarray # Shape (..., N_WAVELENGTHS, 4, 4)

@struct.dataclass
class Ray:
    origin: jnp.ndarray          # Shape (..., 3)
    direction: jnp.ndarray       # Shape (..., 3)
    # Spectral Stokes vector representing polarization state and intensity
    stokes_vector: StokesVector = None # Use type alias
    tmin: float = 1e-4
    tmax: float = jnp.inf

@struct.dataclass
class HitRecord:
    t: float
    position: jnp.ndarray
    normal: jnp.ndarray
    material_id: int
    uv: jnp.ndarray = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.float32))
    wo: jnp.ndarray = field(default_factory=lambda: jnp.zeros(3, dtype=jnp.float32))

# --- Light Data Structures ---
# Using flax.struct for automatic JAX PyTree registration

@struct.dataclass
class DirectionalLight:
    direction: jnp.ndarray # Direction vector *TO* the light source (normalized) - CONVENTION CHANGE
    spd: Spectrum         # Spectral power distribution (intensity/color) at unit distance

@struct.dataclass
class PointLight:
    position: jnp.ndarray # Position in 3D space
    spd: Spectrum         # Spectral power distribution (intensity/color) at unit distance
    # Add falloff later if needed (e.g., constant, linear, quadratic factors)

# Could add SpotLight, AreaLight etc. later 