import jax
import jax.numpy as jnp
from flax import struct
from typing import List, Tuple

# Scene is now primarily a container for passing static data if needed,
# but the core rendering logic will operate on the arrays directly.

@struct.dataclass
class SceneData:
    # Geometry represented as arrays
    sphere_centers: jnp.ndarray     # Shape (num_spheres, 3)
    sphere_radii: jnp.ndarray       # Shape (num_spheres,)
    sphere_material_ids: jnp.ndarray # Shape (num_spheres,)

    # Material parameters as arrays
    material_albedos: jnp.ndarray   # Shape (num_materials, 3)

    # Other scene properties
    background_color: jnp.ndarray   # Shape (3,)