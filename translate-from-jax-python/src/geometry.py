import jax
import jax.numpy as jnp
from flax import struct
from functools import partial

from .types import Ray, HitRecord
from .utils import normalize, dot

# --- Bilinear Sampling Helper ---
@jax.jit
def bilinear_sample_2d(texture: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Performs bilinear sampling on a 2D texture (H, W, C). u, v are in [0, 1]."""
    h, w, c = texture.shape
    
    # Convert normalized coords to pixel coords
    x = u * w - 0.5
    y = v * h - 0.5
    
    # Get integer pixel coords and fractional parts
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Fractional parts for interpolation weights
    wx = x - x0
    wy = y - y0
    
    # Clamp coordinates to texture bounds
    x0 = jnp.clip(x0, 0, w - 1)
    y0 = jnp.clip(y0, 0, h - 1)
    x1 = jnp.clip(x1, 0, w - 1)
    y1 = jnp.clip(y1, 0, h - 1)
    
    # Sample the four neighboring pixels
    p00 = texture[y0, x0]
    p10 = texture[y0, x1]
    p01 = texture[y1, x0]
    p11 = texture[y1, x1]
    
    # Bilinear interpolation
    # Interpolate along x
    interp_x0 = (1.0 - wx)[..., None] * p00 + wx[..., None] * p10
    interp_x1 = (1.0 - wx)[..., None] * p01 + wx[..., None] * p11
    # Interpolate along y
    result = (1.0 - wy)[..., None] * interp_x0 + wy[..., None] * interp_x1
    
    return result

# Keep Sphere just for convenient definition before stacking
@struct.dataclass
class Sphere:
    center: jnp.ndarray  # Shape (3,)
    radius: jnp.ndarray  # Scalar
    material_id: int     # Scalar integer

@jax.jit
def _intersect_sphere_analytic_simplified(center, radius, material_id, ray: Ray) -> HitRecord:
    """Helper: Calculates single ray-sphere intersection. Assumes single ray input for vmap."""
    # Assumes ray.origin and ray.direction are shape (3,)
    # Assumes center is (3,), radius is scalar ()
    oc = ray.origin - center
    a = dot(ray.direction, ray.direction) # Should be > 0 if direction is normalized
    b = 2.0 * dot(oc, ray.direction)
    c = dot(oc, oc) - radius * radius
    discriminant = b * b - 4 * a * c

    # Find the nearest valid intersection distance t
    sqrt_discriminant = jnp.sqrt(jnp.maximum(0, discriminant)) # Avoid NaN
    # Add small epsilon to denominator to avoid division by zero if ray.direction is zero vector (though unlikely)
    safe_a = jnp.maximum(a, 1e-8)
    t = jnp.where(
        discriminant >= 0,
        (-b - sqrt_discriminant) / (2.0 * safe_a),
        jnp.inf
    )

    # Check if t is within bounds (ray.tmin, ray.tmax are scalar)
    hit_valid = (t > ray.tmin) & (t < ray.tmax)
    t_final = jnp.where(hit_valid, t, jnp.inf) # Scalar t

    # Calculate hit position and normal
    position = ray.origin + t_final * ray.direction # (3,) + () * (3,) -> (3,)
    # Ensure normal calculation is safe even if t_final is inf (position becomes inf)
    # Normal should only be valid if hit_valid is true.
    normal_unnormalized = position - center
    normal = normalize(normal_unnormalized) # Normalizes direction, handles potential inf/nan safely if normalize does

    # --- Add UV Calculation (Spherical Coordinates) ---
    # Map position relative to center onto unit sphere
    p_relative = normalize(position - center)
    phi = jnp.arctan2(p_relative[2], p_relative[0]) # Angle around Y axis (azimuthal)
    theta = jnp.arcsin(p_relative[1])              # Angle from XZ plane (polar)
    # Convert angles to UV coordinates [0, 1]
    u = 1.0 - (phi + jnp.pi) / (2.0 * jnp.pi)
    v = (theta + jnp.pi / 2.0) / jnp.pi
    uv = jnp.stack([u, v])
    # --- End UV Calculation ---

    # --- Add Outgoing Direction (wo) ---
    wo = -ray.direction
    # --- End Outgoing Direction ---

    # Material ID is scalar, return -1 if not hit
    final_material_id = jnp.where(hit_valid, material_id, -1) # Scalar id

    # Populate all fields, including default zeros/None for invalid hits
    return HitRecord(
        t=t_final,
        position=jnp.where(hit_valid, position, jnp.zeros_like(position)),
        normal=jnp.where(hit_valid, normal, jnp.zeros_like(normal)),
        material_id=final_material_id,
        uv=jnp.where(hit_valid, uv, jnp.zeros_like(uv)), # Add UVs
        wo=jnp.where(hit_valid, wo, jnp.zeros_like(wo))  # Add wo
    )

# JIT-compilable function to intersect multiple spheres using scan
@partial(jax.jit, static_argnames=())
def intersect_scene_geometry(sphere_centers, sphere_radii, sphere_material_ids, ray: Ray) -> HitRecord:
    """Finds the closest hit by scanning over sphere geometry arrays."""

    def scan_body(carry_hit, sphere_params):
        center, radius, mat_id = sphere_params
        # Use the new simplified function
        current_hit = _intersect_sphere_analytic_simplified(center, radius, mat_id, ray)

        # Find where the current object hit is closer than the carry hit
        is_closer = current_hit.t < carry_hit.t # Scalar comparison

        # Select the fields from the closer hit using simple jnp.where
        # jnp.where broadcasts the scalar `is_closer` correctly
        next_hit = jax.tree.map(
            lambda c, h: jnp.where(is_closer, h, c),
            carry_hit, current_hit
        )
        return next_hit, None # Return updated closest hit, no scan output needed

    # Initialize closest hit record - shape determined by single ray input
    # batch_dims will be () when called from render_pixel (via vmap)
    batch_dims = ray.origin.shape[:-1] # Should be () for single ray (3,)[:-1]
    init_hit = HitRecord(
        t=jnp.full(batch_dims, jnp.inf, dtype=jnp.float32),           # shape ()
        position=jnp.zeros(batch_dims + (3,), dtype=jnp.float32),    # shape (3,)
        normal=jnp.zeros(batch_dims + (3,), dtype=jnp.float32),      # shape (3,)
        material_id=jnp.full(batch_dims, -1, dtype=jnp.int32),        # shape ()
        uv=jnp.zeros(batch_dims + (2,), dtype=jnp.float32),           # Add default UV (shape (2,))
        wo=jnp.zeros(batch_dims + (3,), dtype=jnp.float32)            # Add default WO (shape (3,))
    )

    # Combine sphere parameters for scanning
    scan_params = (sphere_centers, sphere_radii, sphere_material_ids)

    # Run the scan
    closest_hit_final, _ = jax.lax.scan(
        scan_body,
        init_hit,
        scan_params
    )

    return closest_hit_final

# --- Plane Intersection ---

@partial(jax.jit, static_argnums=())
def intersect_plane(
    plane_normal, plane_d, # Base plane definition
    material_id_a, material_id_b, checker_scale, # Material/pattern params
    ray: Ray
) -> HitRecord:
    """Calculates single ray-plane intersection. Returns HitRecord with checker material ID."""
    # Plane equation: dot(P - plane_point, plane_normal) = 0
    # Assuming plane passes through origin, plane_point is 0, so dot(P, plane_normal) = d
    # Here, we use d = -dot(plane_point, plane_normal). If plane passes through origin, d=0.
    # A point P is on the plane if dot(P, plane_normal) + plane_d = 0.
    # Substitute P = ray.origin + t * ray.direction:
    # dot(ray.origin + t * ray.direction, plane_normal) + plane_d = 0
    # dot(ray.origin, plane_normal) + t * dot(ray.direction, plane_normal) + plane_d = 0
    # t = -(dot(ray.origin, plane_normal) + plane_d) / dot(ray.direction, plane_normal)

    denom = dot(ray.direction, plane_normal)

    # Check if ray is parallel to the plane (allow for small epsilon)
    # If denom is close to zero, no intersection or infinite intersections
    parallel = jnp.abs(denom) < 1e-6

    t = jnp.where(
        ~parallel,
        -(dot(ray.origin, plane_normal) + plane_d) / denom,
        jnp.inf
    )

    # Check if t is within bounds
    hit_valid = (t > ray.tmin) & (t < ray.tmax) & (~parallel)
    t_final = jnp.where(hit_valid, t, jnp.inf)

    # Calculate hit position
    position = ray.origin + t_final * ray.direction

    # --- Normal Perturbation --- REMOVED
    # (Removed texture coord calc, sampling, perturbation)
    final_normal = plane_normal # Always return the base plane normal

    # --- Checkerboard pattern calculation (remains the same) ---
    checker_val = (jnp.floor(position[0] * checker_scale) + jnp.floor(position[2] * checker_scale))
    checker_idx = jnp.mod(jnp.abs(checker_val).astype(jnp.int32), 2)
    material_id = jnp.where(checker_idx == 0, material_id_a, material_id_b)
    final_material_id = jnp.where(hit_valid, material_id, -1)

    # --- Add UV Calculation (Planar XZ) ---
    # Simple planar mapping: use world XZ coordinates, potentially scaled/offset
    uv_scale = 1.0 # Adjust as needed
    u = jnp.mod(position[0] * uv_scale, 1.0)
    v = jnp.mod(position[2] * uv_scale, 1.0)
    uv = jnp.stack([u, v])
    # --- End UV Calculation ---

    # --- Add Outgoing Direction (wo) ---
    wo = -ray.direction
    # --- End Outgoing Direction ---

    return HitRecord(
        t=t_final,
        position=jnp.where(hit_valid, position, jnp.zeros_like(position)),
        # Use the base plane normal if hit is valid
        normal=jnp.where(hit_valid, final_normal, jnp.zeros_like(final_normal)), 
        material_id=final_material_id,
        uv=jnp.where(hit_valid, uv, jnp.zeros_like(uv)), # Add UVs
        wo=jnp.where(hit_valid, wo, jnp.zeros_like(wo))  # Add wo
    ) 