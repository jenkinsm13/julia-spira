import jax
import jax.numpy as jnp
from flax import struct
from dataclasses import field
from functools import partial

from .types import Ray
from .utils import normalize, dot, N_WAVELENGTHS

@struct.dataclass
class Camera:
    # Intrinsics
    fx: float = 1000.0  # Focal length in x (pixels)
    fy: float = 1000.0  # Focal length in y (pixels)
    cx: float = 128.0   # Principal point x (pixels)
    cy: float = 128.0   # Principal point y (pixels)

    # Extrinsics (Position & Orientation)
    eye: jnp.ndarray = field(default_factory=lambda: jnp.array([-2.0, 2.5, 4.0]))
    center: jnp.ndarray = field(default_factory=lambda: jnp.array([0.0, 1.0, 0.0]))
    up: jnp.ndarray = field(default_factory=lambda: jnp.array([0.0, 1.0, 0.0]))

    # Distortion Coefficients (Brown-Conrady)
    k1: float = 0.0     # Radial distortion coeff 1
    k2: float = 0.0     # Radial distortion coeff 2
    k3: float = 0.0     # Radial distortion coeff 3
    p1: float = 0.0     # Tangential distortion coeff 1
    p2: float = 0.0     # Tangential distortion coeff 2

    # Physical Exposure Parameters
    f_number: float = 8.0  # Aperture f-stop
    shutter_speed: float = 1/100.0 # Exposure time in seconds
    iso: float = 100.0     # Sensor sensitivity

    # Precompute basis vectors on initialization
    def __post_init__(self):
        # Using object.__setattr__ to bypass frozen=True during init
        w = normalize(self.eye - self.center)
        u = normalize(jnp.cross(self.up, w))
        v = jnp.cross(w, u)
        object.__setattr__(self, '_u', u)
        object.__setattr__(self, '_v', v)
        object.__setattr__(self, '_w', w)

    @partial(jax.jit, static_argnames=['width', 'height'])
    def generate_rays(self, width: int, height: int, rng_key: jax.random.PRNGKey) -> Ray:
        """Generate rays for each pixel, applying distortion and initializing Stokes vector."""
        # Create pixel grid coordinates (image plane, origin at top-left)
        x, y = jnp.meshgrid(jnp.arange(width), jnp.arange(height))
        
        # <<< Apply Jitter to Pixel Coordinates >>>
        # Generate random offsets within [-0.5, 0.5) for each pixel
        jitter_key, rng_key_rest = jax.random.split(rng_key) # Use a separate key for jitter
        dx = jax.random.uniform(jitter_key, shape=x.shape, minval=-0.5, maxval=0.5)
        # Need a new key for dy
        dy_key, rng_key_rest = jax.random.split(rng_key_rest)
        dy = jax.random.uniform(dy_key, shape=y.shape, minval=-0.5, maxval=0.5)
        
        # Add jitter to integer coordinates
        x_j = x + dx
        y_j = y + dy
        # <<< END Apply Jitter >>>
        
        # Shift to center origin and normalize by focal length
        # Using jittered coordinates now
        # Pixel coords (x_p, y_p) relative to principal point
        x_p = (x_j - self.cx) / self.fx
        y_p = (y_j - self.cy) / self.fy

        # Apply radial and tangential distortion (Brown-Conrady model)
        # Use the apply_brown_distortion helper function for clarity
        x_d, y_d = apply_brown_distortion(x_p, y_p, self.k1, self.k2, self.k3, self.p1, self.p2)
        # r2 = x_p**2 + y_p**2
        # r4 = r2**2
        # r6 = r2**3
        # radial_dist = (1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6)
        # x_d = x_p * radial_dist + (2 * self.p1 * x_p * y_p + self.p2 * (r2 + 2 * x_p**2))
        # y_d = y_p * radial_dist + (self.p1 * (r2 + 2 * y_p**2) + 2 * self.p2 * x_p * y_p)

        # Calculate ray directions in camera space
        # Negative z because camera looks down -z axis
        ray_dir_cam = jnp.stack([x_d, y_d, -jnp.ones_like(x_d)], axis=-1)

        # Transform directions from camera space to world space
        # ray_dir_world = ray_dir_cam @ jnp.stack([self._u, self._v, self._w], axis=-1) # Incorrect mat mul
        # Correct transformation: R * v_cam = v_world, where R = [u, v, w]^T if u,v,w form the rows
        # If u, v, w are columns of rotation matrix from world to camera, then transpose is camera to world
        cam_to_world_matrix = jnp.stack([self._u, self._v, self._w], axis=1)
        
        # <<< Normalize the direction *before* transforming to world space >>>
        # Apply jitter in world space? Or camera space?
        # Let's add jitter to camera-space direction for now.
        # ray_dir_cam_jittered = ray_dir_cam + scaled_jitter # Additive jitter might not be ideal
        ray_dir_cam_norm = normalize(ray_dir_cam) # Use the non-jittered camera direction here
        # <<< END Normalize >>>

        # Transform normalized direction to world space
        # einsum for batch matrix vector product: ij,b...j->b...i
        ray_dir_world = jnp.einsum('ij,...j->...i', cam_to_world_matrix, ray_dir_cam_norm)

        # Initialize Stokes vector for unpolarized light (S0=1, S1=S2=S3=0)
        # Shape: (height, width, N_WAVELENGTHS, 4)
        initial_stokes = jnp.zeros((height, width, N_WAVELENGTHS, 4)).at[..., 0].set(1.0)

        # Origin is the camera eye position for all rays
        ray_origin = jnp.tile(self.eye[None, None, :], (height, width, 1))

        return Ray(
            origin=ray_origin,
            direction=ray_dir_world,
            stokes_vector=initial_stokes
        )

@jax.jit
def apply_brown_distortion(xp, yp, k1, k2, k3, p1, p2):
    """Applies Brown-Conrady distortion (incl k3) to normalized coords."""
    r2 = xp**2 + yp**2
    r4 = r2**2
    r6 = r2**3 # Or r4*r2
    radial_factor = 1.0 + k1 * r2 + k2 * r4 + k3 * r6 # Added k3 term

    xd_radial = xp * radial_factor
    yd_radial = yp * radial_factor

    dx_tangential = 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp**2)
    dy_tangential = p1 * (r2 + 2.0 * yp**2) + 2.0 * p2 * xp * yp

    xd = xd_radial + dx_tangential
    yd = yd_radial + dy_tangential

    return xd, yd 