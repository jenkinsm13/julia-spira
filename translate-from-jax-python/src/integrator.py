import jax
import jax.numpy as jnp
import jax.random
from jax import lax, remat, vmap
from functools import partial
from scipy.ndimage import map_coordinates # Added import
from typing import Tuple, NamedTuple, List, Union

# Import spectral types and utils
from .types import Ray, HitRecord, Spectrum, MuellerMatrix 
from .geometry import intersect_scene_geometry, intersect_plane # Import the plane intersection function
from .utils import dot, normalize, N_WAVELENGTHS, WAVELENGTHS_NM, DEFAULT_SPD # Added DEFAULT_SPD
from .utils import IDENTITY_MUELLER_SPECTRAL, DEPOLARIZER_MUELLER_SPECTRAL # Basic Mueller matrices
from .utils import ILLUMINANT_D65_RELATIVE # Changed from ILLUMINANT_D65
from .utils import intersect_ray_triangle_moller_trumbore
from .camera import Camera # For type hinting
# Import the new utility functions
from .utils import calculate_sphere_uv, texture_lookup_udim_bilinear, sample_cosine_weighted_hemisphere, sphere_normal # Added sphere_normal
# Import new Light Types
from .types import DirectionalLight, PointLight

# Define the arguments for the plane globally for cleaner passing
# Removed - these will be passed into render_image now if needed for consistency
# plane_norm = jnp.array([0.0, 1.0, 0.0])
# plane_d = 0.0 # Plane passes through origin
# checker_mat_id_a = 1 # Index for first checker color in material_spds
# checker_mat_id_b = 2 # Index for second checker color
# checker_scale = 2.0 # Controls the size of the checkers

# Small epsilon for ray offsetting
RAY_EPSILON = 1e-4

# --- ADD Mesh Intersection Helper ---
def intersect_mesh_bruteforce(ray: Ray, mesh_data: dict) -> HitRecord:
    """
    Performs brute-force ray-mesh intersection against all triangles.
    Returns the closest valid HitRecord.
    Assumes mesh material_id is 0.
    """
    vertices = mesh_data.get('vertices')
    faces = mesh_data.get('faces')
    uvs = mesh_data.get('uvs')
    normals = mesh_data.get('normals')
    has_mesh = vertices is not None and faces is not None and faces.shape[0] > 0
    has_uvs = uvs is not None
    has_normals = normals is not None

    # --- Defaults for No Hit --- 
    default_t = jnp.inf
    default_pos = jnp.zeros(3)
    default_norm = jnp.array([0.0, 0.0, 1.0])
    default_uv = jnp.array([0.0, 0.0])
    default_bary = jnp.array([1.0, 0.0, 0.0])
    default_face_idx = -1
    default_material_id = -1 # Material ID if no hit
    # -------------------------

    def intersect_logic():
        # Ensure faces is not empty before proceeding.
        # This acts as a safeguard, especially during JIT tracing.
        if faces.shape[0] == 0:
            return HitRecord(
                t=default_t, 
                position=default_pos, 
                normal=default_norm, 
                uv=default_uv, 
                material_id=default_material_id
            )

        num_faces = faces.shape[0]

        # Function to process a single triangle
        def intersect_single_triangle(face_index):
            v_idx0, v_idx1, v_idx2 = faces[face_index]
            v0, v1, v2 = vertices[v_idx0], vertices[v_idx1], vertices[v_idx2]
            
            # Use ray tmin/tmax for valid intersection range
            t, u, v, valid_hit = intersect_ray_triangle_moller_trumbore(
                ray.origin, ray.direction, v0, v1, v2, t_max=ray.tmax
            )
            # Check if hit is within valid t range (after tmin)
            valid_hit_in_range = valid_hit & (t > ray.tmin)
            # Store barycentric coords (w, u, v)
            bary = jnp.array([1.0 - u - v, u, v])
            return t, bary, valid_hit_in_range, face_index

        # Map intersection over all faces
        all_ts, all_barys, all_valid_hits, all_indices = jax.vmap(intersect_single_triangle)(jnp.arange(num_faces))

        # Find the closest valid hit
        masked_ts = jnp.where(all_valid_hits, all_ts, jnp.inf)
        closest_idx_in_map = jnp.argmin(masked_ts)
        closest_t = masked_ts[closest_idx_in_map]
        final_valid_hit = closest_t < jnp.inf

        # Get info for the closest hit (if any)
        def get_closest_hit_info():
             bary = all_barys[closest_idx_in_map]
             f_idx = all_indices[closest_idx_in_map]
             return bary, f_idx, closest_t

        closest_bary, closest_face_idx, final_t = lax.cond(
            final_valid_hit,
            get_closest_hit_info,
            lambda: (default_bary, default_face_idx, default_t) 
        )

        # --- Calculate final world position, normal, and UV --- 
        hit_pos_world = ray.origin + ray.direction * final_t

        def calculate_final_attrs():
            v_idx0, v_idx1, v_idx2 = faces[closest_face_idx]
            w, u, v = closest_bary[0], closest_bary[1], closest_bary[2]
            # Interpolate Normals if available
            if has_normals:
                n0, n1, n2 = normals[v_idx0], normals[v_idx1], normals[v_idx2]
                interp_normal = w*n0 + u*n1 + v*n2
                normal = normalize(interp_normal)
            else: # Fallback: Calculate flat face normal
                v0, v1, v2 = vertices[v_idx0], vertices[v_idx1], vertices[v_idx2]
                normal = normalize(jnp.cross(v1 - v0, v2 - v0))
            # Interpolate UVs if available
            if has_uvs:
                uv0, uv1, uv2 = uvs[v_idx0], uvs[v_idx1], uvs[v_idx2]
                uv = w*uv0 + u*uv1 + v*uv2
            else: 
                uv = default_uv
            return normal, uv

        final_normal, final_uv = lax.cond(
            final_valid_hit,
            calculate_final_attrs, 
            lambda: (default_norm, default_uv)
        )
        
        # Assign material ID 0 if hit, otherwise -1
        final_material_id = jnp.where(final_valid_hit, 0, default_material_id)

        return HitRecord(
            t=final_t,
            position=hit_pos_world,
            normal=final_normal,
            uv=final_uv,
            material_id=final_material_id
        )

    # Only run intersection if mesh exists
    hit_record = lax.cond(
        has_mesh,
        intersect_logic,
        lambda: HitRecord(t=default_t, position=default_pos, normal=default_norm, uv=default_uv, material_id=default_material_id)
    )
    return hit_record
# --- END Mesh Intersection Helper ---

# --- Path Tracing Loop State ---
class PathState(NamedTuple):
    ray: Ray
    depth: int
    rng_key: jax.random.PRNGKey
    radiance: jnp.ndarray
    throughput: jnp.ndarray
    active: bool

# --- Path Tracing Logic (Iterative version) ---
def trace_ray(
    # --- Static Args ---
    max_depth: int,
    udim_keys_sorted: Tuple[int, ...],
    udim_shapes: Tuple[Tuple[int, int, int], ...],
    plane_d: float, 
    plane_mat_id_a: int, 
    plane_mat_id_b: int, 
    plane_checker_scale: float,
    # --- Dynamic Args ---
    plane_normal: jnp.ndarray, 
    mesh_data: dict,
    udim_texture_tiles: Tuple[jnp.ndarray, ...],
    fixed_material_spds: jnp.ndarray,
    background_spd: jnp.ndarray,
    default_spd: jnp.ndarray,
    # <<< Use Stacked Light Structs >>>
    stacked_directional_lights: DirectionalLight,
    stacked_point_lights: PointLight,
    # <<< END Use Stacked >>>
    # --- Initial Ray State ---
    ray: Ray,
    rng_key: jax.random.PRNGKey
) -> jnp.ndarray:
    """Traces a single ray iteratively using path tracing with mesh/plane intersection."""
    
    # Initialize the path state
    init_state = PathState(
        ray=ray,
        depth=0,
        rng_key=rng_key,
        radiance=jnp.zeros(N_WAVELENGTHS),
        throughput=jnp.ones(N_WAVELENGTHS),
        active=True # Start active
    )
    
    # --- Define the body function for lax.scan ---
    @remat # <<< RE-ENABLED Gradient Checkpointing
    def scan_body(state, _): # state is the carry, _ is unused input from xs
        """Performs one step of path tracing. Input: current state. Output: (new_state, None)."""

        # Define the core logic of a single bounce (similar to old body_fun)
        def active_bounce_logic(current_state):
            current_ray = current_state.ray
            
            # --- Intersect with Mesh and Plane ---
            mesh_hit = intersect_mesh_bruteforce(current_ray, mesh_data)
            plane_hit = intersect_plane(plane_normal, plane_d, plane_mat_id_a, plane_mat_id_b, plane_checker_scale, current_ray)
            
            # --- Select Closest Hit (Mesh or Plane) ---
            hit_mesh = mesh_hit.t < jnp.inf
            hit_plane = plane_hit.t < jnp.inf
            hit_anything = hit_mesh | hit_plane
            
            # Determine which is closer if both are hit
            closer_mesh = hit_mesh & (~hit_plane | (mesh_hit.t < plane_hit.t))
            
            # Select the closest hit using tree_map
            closest_hit = jax.tree.map(
                lambda m, p: jnp.where(closer_mesh, m, p),
                mesh_hit, plane_hit
            )
            # Ensure valid t and material_id if nothing hit
            closest_hit = closest_hit.replace(
                material_id=jnp.where(hit_anything, closest_hit.material_id, -1),
                t=jnp.where(hit_anything, closest_hit.t, jnp.inf)
            )
            # --- End Hit Selection ---
            
            # Define actions for miss and hit
            def handle_miss(miss_state):
                # Path terminates, add background contribution scaled by current throughput
                # Background assumed to be infinitely far
                background_contribution = miss_state.throughput * background_spd
                final_radiance = miss_state.radiance + background_contribution
                # Return state with accumulated radiance and marked inactive
                return miss_state._replace(radiance=final_radiance, active=False)

            def handle_hit(hit_state, hit_rec):
                hit_pos = hit_rec.position
                
                # --- Calculate Material Color at Hit Point ---
                # Use hit_rec.uv for mesh (mat_id=0) or use fixed plane color
                mesh_color_uv = texture_lookup_udim_bilinear(
                    udim_keys_sorted=udim_keys_sorted,
                    udim_shapes=udim_shapes,
                    udim_texture_tiles=udim_texture_tiles,
                    uv=hit_rec.uv, 
                    default_spd=default_spd
                )
                # Select fixed material for plane based on ID (offset by 1, clamped)
                idx = jnp.maximum(0, hit_rec.material_id - 1)
                safe_idx = jnp.minimum(idx, fixed_material_spds.shape[0] - 1)
                plane_color = fixed_material_spds[safe_idx]
                
                is_mesh = hit_rec.material_id == 0 # Check if the hit was the mesh
                material_color = jnp.where(is_mesh, mesh_color_uv, plane_color)
                # --- End Material Color Calculation ---

                # --- Emitted Light --- (Assuming non-emissive surfaces for now)
                emitted_light = jnp.zeros_like(hit_state.radiance)

                # --- Direct Light Calculation (NEE) ---
                direct_light_contribution = jnp.zeros_like(material_color)
                
                # --- Define Light Contribution Functions (Specific Types) ---
                
                # Contribution from a single Directional Light
                def calculate_directional_light_contribution(dir_light):
                    light_dir_to = dir_light.direction
                    light_intensity_at_hit = dir_light.spd
                    shadow_ray_tmax = jnp.inf
                    
                    cos_term = jnp.maximum(0.0, dot(hit_rec.normal, light_dir_to))
                    contribution = jnp.zeros_like(material_color)
                    
                    def check_visibility_directional():
                        shadow_ray_origin = hit_pos + hit_rec.normal * RAY_EPSILON
                        shadow_ray = Ray(
                            origin=shadow_ray_origin, direction=light_dir_to,
                            stokes_vector=current_ray.stokes_vector, tmin=RAY_EPSILON,
                            tmax=shadow_ray_tmax - RAY_EPSILON 
                        )
                        # --- Shadow Ray Intersection (Mesh & Plane) --- 
                        shadow_mesh_hit = intersect_mesh_bruteforce(shadow_ray, mesh_data)
                        shadow_plane_hit = intersect_plane(plane_normal, plane_d, plane_mat_id_a, plane_mat_id_b, plane_checker_scale, shadow_ray)
                        is_occluded = (shadow_mesh_hit.t < jnp.inf) | (shadow_plane_hit.t < jnp.inf)
                        # --- End Shadow Ray Intersection ---
                        calculated_contribution = (material_color / jnp.pi) * light_intensity_at_hit * cos_term
                        return jnp.where(is_occluded, jnp.zeros_like(material_color), calculated_contribution)

                    contribution = lax.cond(cos_term > 1e-6, lambda _: check_visibility_directional(), lambda _: jnp.zeros_like(material_color), operand=None)
                    return contribution

                # Contribution from a single Point Light
                def calculate_point_light_contribution(point_light):
                    vec_to_light = point_light.position - hit_pos
                    dist_sq = jnp.sum(vec_to_light**2)
                    dist = jnp.sqrt(dist_sq)
                    light_dir_to = vec_to_light / jnp.maximum(dist, 1e-6)
                    light_intensity_at_hit = point_light.spd / jnp.maximum(dist_sq, 1e-6)
                    shadow_ray_tmax = dist # Max t for shadow ray is distance to light
                    
                    cos_term = jnp.maximum(0.0, dot(hit_rec.normal, light_dir_to))
                    contribution = jnp.zeros_like(material_color)

                    def check_visibility_point():
                        shadow_ray_origin = hit_pos + hit_rec.normal * RAY_EPSILON
                        shadow_ray = Ray(
                            origin=shadow_ray_origin, direction=light_dir_to,
                            stokes_vector=current_ray.stokes_vector, tmin=RAY_EPSILON,
                            tmax=shadow_ray_tmax - RAY_EPSILON # Use tmax to avoid hitting beyond light
                        )
                        # --- Shadow Ray Intersection (Mesh & Plane) --- 
                        shadow_mesh_hit = intersect_mesh_bruteforce(shadow_ray, mesh_data)
                        shadow_plane_hit = intersect_plane(plane_normal, plane_d, plane_mat_id_a, plane_mat_id_b, plane_checker_scale, shadow_ray)
                        is_occluded = (shadow_mesh_hit.t < jnp.inf) | (shadow_plane_hit.t < jnp.inf)
                        # --- End Shadow Ray Intersection ---
                        calculated_contribution = (material_color / jnp.pi) * light_intensity_at_hit * cos_term
                        return jnp.where(is_occluded, jnp.zeros_like(material_color), calculated_contribution)

                    contribution = lax.cond(cos_term > 1e-6, lambda _: check_visibility_point(), lambda _: jnp.zeros_like(material_color), operand=None)
                    return contribution
                # --- End of contribution functions ---
                
                # --- Iterate through lights lists using vmap --- 
                vmapped_dir_calc = jax.vmap(calculate_directional_light_contribution)
                dir_light_contributions = vmapped_dir_calc(stacked_directional_lights)
                dir_light_sum = jnp.sum(dir_light_contributions, axis=0) # Sum contributions
                
                vmapped_point_calc = jax.vmap(calculate_point_light_contribution)
                point_light_contributions = vmapped_point_calc(stacked_point_lights)
                point_light_sum = jnp.sum(point_light_contributions, axis=0) # Sum contributions
                
                direct_light_contribution = dir_light_sum + point_light_sum
                # --- End Vmap Iterate --- 

                # --- Accumulate Radiance for this Bounce ---
                radiance_added_this_bounce = emitted_light + direct_light_contribution
                new_accumulated_radiance = hit_state.radiance + hit_state.throughput * radiance_added_this_bounce

                # --- Indirect Light Path Continuation ---
                bounce_key, next_key_for_state = jax.random.split(hit_state.rng_key)
                
                next_dir, pdf = sample_cosine_weighted_hemisphere(hit_rec.normal, bounce_key)
                
                next_origin = hit_pos + hit_rec.normal * RAY_EPSILON
                next_ray = Ray(
                    origin=next_origin,
                    direction=next_dir,
                    stokes_vector=current_ray.stokes_vector, # Propagate Stokes
                    tmin=RAY_EPSILON,
                    tmax=jnp.inf
                )

                brdf_term = material_color # Simplified for Lambertian/Cosine sampling
                new_next_throughput = hit_state.throughput * brdf_term 
                
                # --- Russian Roulette (Optional but Recommended) ---
                # Simple check: terminate if throughput is very low (or based on bounce depth)
                throughput_magnitude = jnp.mean(new_next_throughput) # Example metric
                terminate_rr = throughput_magnitude < 1e-4
                # Alternative: Implement proper RR later
                
                # Path terminates if RR terminates OR if depth limit reached in *next* iteration
                active_for_next = True & ~terminate_rr # Path continues based on RR

                # Return the updated state for the next iteration
                return PathState(
                    ray=next_ray,
                    depth=hit_state.depth + 1,
                    rng_key=next_key_for_state, 
                    radiance=new_accumulated_radiance,
                    throughput=new_next_throughput,
                    active=active_for_next # Set based on RR / next depth check implicit in scan length
                )

            # --- Choose branch based on hit ---
            bounce_next_state = lax.cond(
                hit_anything,
                lambda s: handle_hit(s, closest_hit), # Pass state and hit_rec
                lambda s: handle_miss(s),             # Pass state
                operand=current_state # Pass the current state to the lambda functions
            )
            return bounce_next_state
            
        # --- Main Scan Body Logic --- 
        # If the path is inactive, just return the current state
        # Otherwise, execute the bounce logic
        next_state = lax.cond(
            state.active,
            active_bounce_logic, # Function to call if active
            lambda s: s,         # Function to call if inactive (identity)
            operand=state        # Argument to pass to the chosen function
        )
        
        # Return the next state as the carry, and None for the per-iteration output
        return next_state, None 
    
    # --- Execute the path tracing using lax.scan ---
    # scan iterates scan_body max_depth times.
    # init_state is the initial carry state.
    # xs is None because we don't have per-iteration inputs.
    final_state, _ = lax.scan(scan_body, init_state, xs=None, length=max_depth)
    
    # Return the final accumulated radiance
    # Need to handle the case where the path became inactive *exactly* on the last
    # iteration. The radiance might not include the final background hit.
    # Re-check for miss on the final state's ray if it's still marked active.
    final_radiance = lax.cond(
        final_state.active, # If still active after max_depth bounces
        lambda fs: fs.radiance + fs.throughput * background_spd, # Assume it missed and add background
        lambda fs: fs.radiance, # Otherwise, radiance is already final
        operand=final_state
    )

    return final_radiance

# --- Vmap Definitions --- (Adjusted for mesh_data)
def render_pixel(
    # --- Static Args ---
    max_depth: int,
    udim_keys_sorted: Tuple[int, ...],
    udim_shapes: Tuple[Tuple[int, int, int], ...],
    plane_d: float, 
    plane_mat_id_a: int, 
    plane_mat_id_b: int, 
    plane_checker_scale: float,
    # --- Dynamic Args ---
    mesh_data: dict,
    udim_texture_tiles: Tuple[jnp.ndarray, ...],
    fixed_material_spds: jnp.ndarray,
    background_spd: jnp.ndarray,
    default_spd: jnp.ndarray,
    plane_normal: jnp.ndarray,
    # <<< Use Stacked Light Structs >>>
    stacked_directional_lights: DirectionalLight,
    stacked_point_lights: PointLight,
    # <<< END Use Stacked >>>
    # --- Initial Ray/Key ---
    ray: Ray, 
    rng_key: jax.random.PRNGKey
) -> jnp.ndarray: 
    """Render a single pixel using path tracing with mesh support."""
    
    # Call the iterative trace_ray function
    spectral_radiance = trace_ray(
        # Static args
        max_depth, udim_keys_sorted, udim_shapes, 
        plane_d, plane_mat_id_a, plane_mat_id_b, plane_checker_scale,
        # Dynamic args
        plane_normal,
        mesh_data,
        udim_texture_tiles, fixed_material_spds, background_spd, default_spd,
        # <<< Pass stacked light structs >>>
        stacked_directional_lights,
        stacked_point_lights,
        # <<< END Pass >>>
        # Initial ray and key
        ray, rng_key
    )
    
    # Path tracing returns spectral radiance. 
    pixel_stokes = jnp.zeros((N_WAVELENGTHS, 4)).at[:, 0].set(spectral_radiance)
    return pixel_stokes

# --- Vmap Definitions --- 
ray_in_axes = Ray(origin=0, direction=0, stokes_vector=0, tmin=None, tmax=None)
pixel_key_in_axes = 0 # Map over pixels

# Axes for mapping render_pixel across pixels
pixel_axes = (
    # Static Args (Broadcast)
    None,  # max_depth 
    None,  # udim_keys_sorted
    None,  # udim_shapes
    None,  # plane_d
    None,  # plane_mat_id_a
    None,  # plane_mat_id_b
    None,  # plane_checker_scale
    # Dynamic Args (Broadcast)
    None,  # mesh_data
    None,  # udim_texture_tiles
    None,  # fixed_material_spds
    None,  # background_spd
    None,  # default_spd
    None,  # plane_normal
    # <<< Stacked Lights (Broadcast) >>>
    None,  # stacked_directional_lights
    None,  # stacked_point_lights
    # <<< END Stacked >>>
    # Mapped Args
    ray_in_axes,        # Ray (map axis 0 of row data)
    pixel_key_in_axes   # rng_key (map axis 0 of row data)
)

# Vmap render_pixel over a row (width)
render_row = jax.vmap(render_pixel, in_axes=pixel_axes) 

# Need to adjust the outer vmap as well
ray_in_axes_image = Ray(origin=0, direction=0, stokes_vector=0, tmin=None, tmax=None)
pixel_key_in_axes_image = 0

# Axes for mapping the row renderer over the image height
row_pixel_axes = (
    # Static/Dynamic Args (Broadcast across rows)
    None,  # max_depth
    None,  # udim_keys_sorted
    None,  # udim_shapes
    None,  # plane_d
    None,  # plane_mat_id_a
    None,  # plane_mat_id_b
    None,  # plane_checker_scale
    None,  # mesh_data
    None,  # udim_texture_tiles
    None,  # fixed_material_spds
    None,  # background_spd
    None,  # default_spd
    None,  # plane_normal
    # <<< Stacked Lights (Broadcast) >>>
    None,  # stacked_directional_lights
    None,  # stacked_point_lights
    # <<< END Stacked >>>
    # Mapped Args (Sliced per row from image arrays)
    ray_in_axes_image,        # Pass rays[h, :, :]
    pixel_key_in_axes_image   # Pass keys[h, :, :]
)

# Map the row renderer across the height dimension
render_image_pixels_uncompiled = jax.vmap(render_row, in_axes=row_pixel_axes)

# Define static arguments for the *entire* image pixel rendering
# Static args remain the same as mesh_data is dynamic
_render_image_pixels_static_argnames = (
    'max_depth', 'udim_keys_sorted', 'udim_shapes',
    'plane_d', 'plane_mat_id_a', 'plane_mat_id_b', 'plane_checker_scale'
)
render_image_pixels_jit = partial(jax.jit, static_argnames=_render_image_pixels_static_argnames)(
    render_image_pixels_uncompiled
)

# --- Main render_image function --- 
# Static args remain the same
_render_image_static_argnames = (
    'width', 'height', 'samples_per_pixel', 'max_depth',
    'cam_fx', 'cam_fy', 'cam_cx', 'cam_cy',
    'cam_k1', 'cam_k2', 'cam_k3', 'p1', 'p2',
    'cam_f_number', 'cam_shutter_speed', 'cam_iso',
    'udim_keys_sorted', 'udim_shapes', 
    'plane_d', 'plane_mat_id_a', 'plane_mat_id_b', 'plane_checker_scale'
)

@partial(jax.jit, static_argnames=_render_image_static_argnames)
def render_image(
    # --- Static Args ---
    width: int, height: int, samples_per_pixel: int, max_depth: int,
    cam_fx: float, cam_fy: float, cam_cx: float, cam_cy: float,
    cam_k1: float, cam_k2: float, cam_k3: float, p1: float, p2: float,
    cam_f_number: float, cam_shutter_speed: float, cam_iso: float,
    udim_keys_sorted: Tuple[int, ...],
    udim_shapes: Tuple[Tuple[int, int, int], ...],
    plane_d: float, plane_mat_id_a: int, plane_mat_id_b: int, plane_checker_scale: float,
    # --- Dynamic Args ---
    mesh_data: dict,
    udim_texture_tiles: Tuple[jnp.ndarray, ...], 
    fixed_material_spds: jnp.ndarray, 
    background_spd: jnp.ndarray, 
    default_spd: jnp.ndarray,
    plane_normal: jnp.ndarray,
    # <<< Use Stacked Light Structs >>>
    stacked_directional_lights: DirectionalLight,
    stacked_point_lights: PointLight,
    # <<< END Use Stacked >>>
    cam_eye: jnp.ndarray, 
    cam_center: jnp.ndarray, 
    cam_up: jnp.ndarray, 
    rng_key: jax.random.PRNGKey 
) -> jnp.ndarray:
    """Render the entire image using path tracing with mesh support and supersampling."""

    # Function to render one sample for the entire image
    @remat
    def render_sample(carry_key):
        sample_key, pixel_rng_key = jax.random.split(carry_key)
        # Create temporary Camera object (as before)
        temp_camera = Camera(
            fx=cam_fx, fy=cam_fy, cx=cam_cx, cy=cam_cy,
            eye=cam_eye, center=cam_center, up=cam_up,
            k1=cam_k1, k2=cam_k2, k3=cam_k3, p1=p1, p2=p2, 
            f_number=cam_f_number, shutter_speed=cam_shutter_speed, iso=cam_iso
        )
        rays = temp_camera.generate_rays(width, height, sample_key) 
        pixel_keys = jax.random.split(pixel_rng_key, width * height).reshape(height, width, 2)

        # Call the vmapped pixel renderer
        image_sample_stokes = render_image_pixels_jit(
            # Broadcasted Args (Static/Dynamic)
            max_depth, udim_keys_sorted, udim_shapes, 
            plane_d, plane_mat_id_a, plane_mat_id_b, plane_checker_scale,
            mesh_data,
            udim_texture_tiles, fixed_material_spds, background_spd, default_spd,
            plane_normal,
            # <<< Pass stacked light structs >>>
            stacked_directional_lights,
            stacked_point_lights,
            # <<< END Pass >>>
            # Mapped Args (Rays and Keys)
            rays, pixel_keys
        )
        return image_sample_stokes # Corrected: return only the Stokes data for the sample

    # Sum Stokes vectors from all samples
    initial_carry_sum_key = (jnp.zeros((height, width, N_WAVELENGTHS, 4)), rng_key)
    
    # Define the body for the scan over samples
    # @remat # Optional: checkpoint the per-sample rendering if memory is an issue
    def scan_body_samples(carry_sum_key, _):
        current_sum_stokes, current_rng_key = carry_sum_key
        sample_render_key, next_rng_key = jax.random.split(current_rng_key)
        
        one_sample_stokes = render_sample(sample_render_key)
        
        updated_sum_stokes = current_sum_stokes + one_sample_stokes
        return (updated_sum_stokes, next_rng_key), None

    # Perform the scan over samples_per_pixel
    (final_sum_stokes, _), _ = lax.scan(
        scan_body_samples, 
        initial_carry_sum_key, 
        xs=None, 
        length=samples_per_pixel
    )

    # Averaging the Stokes vectors
    averaged_stokes = final_sum_stokes / samples_per_pixel

    # Apply camera exposure to the S0 component (intensity)
    # Ensure f_number is not zero to avoid division by zero; add a small epsilon.
    safe_f_number_sq = cam_f_number**2 + 1e-6 # Add epsilon for stability
    exposure_factor = (cam_shutter_speed * (cam_iso / 100.0)) / safe_f_number_sq
    
    exposed_S0 = averaged_stokes[..., 0] * exposure_factor
    final_exposed_stokes = averaged_stokes.at[..., 0].set(exposed_S0)

    return final_exposed_stokes
    # return averaged_stokes # Old return before exposure

# --- ADD Direct Illumination Renderer ---

# Removed JIT decorator to simplify nested compilation
# @partial(jax.jit, static_argnames=(
#     'udim_keys_sorted', 'udim_shapes',
#     'plane_d', 'plane_mat_id_a', 'plane_mat_id_b', 'plane_checker_scale',
#     'fixed_material_spds'
# ))
def render_direct_illumination(
    # --- Surface Point Info (Dynamic) ---
    hit_pos: jnp.ndarray,       
    hit_normal: jnp.ndarray,    
    hit_uv: jnp.ndarray,        
    view_dir: jnp.ndarray,      
    # --- Texture/Material Info (Dynamic) ---
    udim_texture_tiles: Tuple[jnp.ndarray, ...], 
    default_spd: jnp.ndarray,
    params_template: dict, 
    # --- Scene Geometry Info (Dynamic & Static) ---
    plane_normal: jnp.ndarray, # Keep for plane shadow checks
    mesh_data: dict,           # Pass loaded mesh data
    # Static parts passed for JIT
    udim_keys_sorted: Tuple[int, ...],
    udim_shapes: Tuple[Tuple[int, int, int], ...],
    plane_d: float,
    plane_mat_id_a: int,
    plane_mat_id_b: int,
    plane_checker_scale: float,
    fixed_material_spds: jnp.ndarray, 
    # --- Light Sources (Dynamic) ---
    stacked_directional_lights: DirectionalLight,
    stacked_point_lights: PointLight
) -> Spectrum:
    """
    Computes direct illumination (Lambertian) + shadows for a single surface point.
    Uses brute-force mesh intersection for shadows.
    """
    # --- Access mesh data for shadow intersections --- 
    vertices = mesh_data.get('vertices')
    faces = mesh_data.get('faces')
    has_mesh = vertices is not None and faces is not None
    
    # --- Remove Sphere/Plane Shadow Geometry --- 
    # sphere_radii = jnp.array([1.0]) 
    # sphere_center_shadow = sphere_center 
    # sphere_material_ids = jnp.array([0], dtype=jnp.int32) 

    # 1. Look up Albedo (Spectral Reflectance) from texture
    spectral_albedo = texture_lookup_udim_bilinear(
        udim_keys_sorted=udim_keys_sorted,
        udim_shapes=udim_shapes,
        udim_texture_tiles=udim_texture_tiles,
        uv=hit_uv,
        default_spd=default_spd
    )
    # Ensure albedo is non-negative
    spectral_albedo = jnp.maximum(0.0, spectral_albedo)

    # 2. Initialize Radiance
    direct_radiance = jnp.zeros(N_WAVELENGTHS)

    # --- Shared function for mesh shadow check --- 
    def check_mesh_occlusion(shadow_origin, shadow_dir, t_max=jnp.inf):
        if not has_mesh:
            return False # No mesh to occlude
            
        num_faces = faces.shape[0]
        
        def intersect_shadow_triangle(face_index):
            v_idx0, v_idx1, v_idx2 = faces[face_index]
            v0, v1, v2 = vertices[v_idx0], vertices[v_idx1], vertices[v_idx2]
            # Use backface culling potentially? For shadows, maybe not needed.
            t, _, _, valid_hit = intersect_ray_triangle_moller_trumbore(
                shadow_origin, shadow_dir, v0, v1, v2, t_max=t_max, backface_culling=False
            )
            return valid_hit & (t < jnp.inf) # Return True if valid hit closer than t_max
            
        # Check all faces - vmap and check if *any* hit is found
        any_hit = jnp.any(jax.vmap(intersect_shadow_triangle)(jnp.arange(num_faces)))
        return any_hit
    # -----------------------------------------

    # 3. Loop through Directional Lights
    def directional_light_body(i, current_radiance):
        light = jax.tree.map(lambda x: x[i], stacked_directional_lights)
        light_dir_to = light.direction 
        light_intensity = light.spd
        shadow_ray_origin = hit_pos + hit_normal * RAY_EPSILON

        # --- Mesh Shadow Check --- 
        mesh_occluded = check_mesh_occlusion(shadow_ray_origin, light_dir_to)
        # --- End Mesh Shadow Check --- 
        
        # --- Check Plane Shadow (Optional, keep if plane exists) ---
        shadow_ray_plane = Ray(origin=shadow_ray_origin, direction=light_dir_to, tmin=RAY_EPSILON, tmax=jnp.inf)
        shadow_plane_hit = intersect_plane(plane_normal, plane_d, plane_mat_id_a, plane_mat_id_b, plane_checker_scale, shadow_ray_plane)
        plane_occluded = shadow_plane_hit.t < jnp.inf
        # --- End Plane Shadow Check --- 

        is_occluded = mesh_occluded | plane_occluded # Occluded if hit mesh OR plane
        
        cos_term = jnp.maximum(0.0, dot(hit_normal, light_dir_to))
        contribution = (spectral_albedo / jnp.pi) * light_intensity * cos_term
        return current_radiance + jnp.where(is_occluded | (cos_term <= 1e-6), 0.0, contribution)

    num_dir_lights = stacked_directional_lights.direction.shape[0]
    direct_radiance = lax.fori_loop(0, num_dir_lights, directional_light_body, direct_radiance)

    # 4. Loop through Point Lights
    def point_light_body(i, current_radiance):
        light = jax.tree.map(lambda x: x[i], stacked_point_lights)
        vec_to_light = light.position - hit_pos
        dist_sq = jnp.sum(vec_to_light**2)
        dist = jnp.sqrt(dist_sq)
        light_dir_to = vec_to_light / jnp.maximum(dist, 1e-6)
        light_intensity = light.spd / jnp.maximum(dist_sq, 1e-6)
        shadow_ray_tmax = dist - RAY_EPSILON # Max distance for shadow ray

        shadow_ray_origin = hit_pos + hit_normal * RAY_EPSILON

        # --- Mesh Shadow Check --- 
        mesh_occluded = check_mesh_occlusion(shadow_ray_origin, light_dir_to, t_max=shadow_ray_tmax)
        # --- End Mesh Shadow Check --- 

        # --- Check Plane Shadow --- 
        shadow_ray_plane = Ray(origin=shadow_ray_origin, direction=light_dir_to, tmin=RAY_EPSILON, tmax=shadow_ray_tmax)
        shadow_plane_hit = intersect_plane(plane_normal, plane_d, plane_mat_id_a, plane_mat_id_b, plane_checker_scale, shadow_ray_plane)
        plane_occluded = shadow_plane_hit.t < jnp.inf
        # --- End Plane Shadow Check --- 

        is_occluded = mesh_occluded | plane_occluded

        cos_term = jnp.maximum(0.0, dot(hit_normal, light_dir_to))
        contribution = (spectral_albedo / jnp.pi) * light_intensity * cos_term
        return current_radiance + jnp.where(is_occluded | (cos_term <= 1e-6), 0.0, contribution)

    num_point_lights = stacked_point_lights.position.shape[0]
    direct_radiance = lax.fori_loop(0, num_point_lights, point_light_body, direct_radiance)

    # TODO: Add emission term if needed later
    # TODO: Replace Lambertian with more complex BRDF if needed

    return direct_radiance