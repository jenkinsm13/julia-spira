import jax
import jax.numpy as jnp
# <<< ADD JVP/VJP/CG Imports >>>
from jax import jvp, vjp
from jax.scipy.sparse.linalg import cg
# <<< END ADD >>>
# <<< ADD lax import >>>
from jax import lax
# <<< END ADD >>>

# <<< ADD DEVICE CHECK START >>>
print("JAX Devices:", jax.devices())
# <<< ADD DEVICE CHECK END >>>

import numpy as np
from PIL import Image
import optax
import time
from tqdm import tqdm
from functools import partial
import os
import OpenEXR
import Imath
import argparse
from typing import Tuple, NamedTuple, Optional
import copy
# <<< Import Light Structs >>>
from src.types import DirectionalLight, PointLight, Spectrum # Import Spectrum type alias
from typing import List # For list type hint
# <<< END Import >>>

# <<< ADD pywavefront import >>>
import pywavefront
# <<< END ADD >>>

# Import necessary components
from src.types import Ray, HitRecord, Spectrum, DirectionalLight
from src.geometry import Sphere # Keep for definition
from src.camera import Camera # Renamed from PinholeCamera
from src.integrator import render_image, trace_ray, render_direct_illumination # Added trace_ray, render_direct_illumination import
# Import spectral utilities and forward render helper
from src.utils import (
    N_WAVELENGTHS, WAVELENGTHS_NM, 
    spectral_to_xyz, xyz_to_rgb, linear_rgb_to_srgb, 
    XYZ_TO_SRGB_MATRIX, XYZ_TO_ACESCG_MATRIX,
    ACESCG_TO_XYZ_MATRIX,
    create_spectrum_profile, # Import the helper
    normalize, # <-- IMPORT normalize
    dot, # <-- IMPORT dot
    intersect_ray_triangle_moller_trumbore # <-- IMPORT Triangle Intersection
)
# Import newly added functions (or define them here if edit failed)
# from src.integrator import render_direct_illumination # Import the new renderer
# Attempt to import from utils, will define locally if not found
from src.utils import sample_sphere_surface, sphere_normal, calculate_sphere_uv

# Define default plane parameters (not optimized here)
plane_norm = jnp.array([0.0, 1.0, 0.0])
plane_d = 0.0 # Match forward_render.py: Plane at y=0
plane_mat_id_a = 1 # Use fixed ground material index
plane_mat_id_b = 2 # Use different material for checkerboard pattern (match forward_render.py)
plane_checker_scale = 2.0
DEFAULT_SAMPLES_PER_PIXEL = 16 

# Function to load EXR files with ACEScg color space support
def load_exr_acescg(exr_path):
    """
    Load an EXR file with ACEScg primaries.
    Returns a JAX array with shape (height, width, 3) in linear RGB space.
    """
    if not os.path.exists(exr_path):
        raise FileNotFoundError(f"EXR file not found: {exr_path}")
    
    try:
        exr_file = OpenEXR.InputFile(exr_path)
        
        # Get image dimensions
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # Check for ACEScg chromaticities in header
        has_acescg = False
        if 'chromaticities' in header:
            chroma = header['chromaticities']
            # ACEScg primaries (approximate check)
            acescg_r = (0.713, 0.293)
            acescg_g = (0.165, 0.830)
            acescg_b = (0.128, -0.044)
            
            r_close = abs(chroma.red.x - acescg_r[0]) < 0.01 and abs(chroma.red.y - acescg_r[1]) < 0.01
            g_close = abs(chroma.green.x - acescg_g[0]) < 0.01 and abs(chroma.green.y - acescg_g[1]) < 0.01
            b_close = abs(chroma.blue.x - acescg_b[0]) < 0.01 and abs(chroma.blue.y - acescg_b[1]) < 0.01
            
            has_acescg = r_close and g_close and b_close
        
        print(f"EXR loaded: {width}x{height}, ACEScg primaries: {'Yes' if has_acescg else 'No/Unknown'}")
        if not has_acescg:
            print("Warning: EXR file may not use ACEScg primaries, but will be treated as such")
        
        # Get pixel type (prefer FLOAT, fallback to HALF)
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        try:
            r_str = exr_file.channel('R', pixel_type)
        except:
            pixel_type = Imath.PixelType(Imath.PixelType.HALF)
            print(f"Using HALF pixel type instead of FLOAT")
        
        # Read color channels
        r_str = exr_file.channel('R', pixel_type)
        g_str = exr_file.channel('G', pixel_type)
        b_str = exr_file.channel('B', pixel_type)
        
        # Convert to numpy arrays
        if pixel_type == Imath.PixelType(Imath.PixelType.FLOAT):
            dtype = np.float32
        else:
            dtype = np.float16
        
        r = np.frombuffer(r_str, dtype=dtype).reshape(height, width)
        g = np.frombuffer(g_str, dtype=dtype).reshape(height, width)
        b = np.frombuffer(b_str, dtype=dtype).reshape(height, width)
        
        # Stack into RGB image
        rgb = np.stack([r, g, b], axis=-1)
        
        # Convert to JAX array
        return jnp.array(rgb)
        
    except Exception as e:
        print(f"Error loading EXR: {e}")
        raise

# Function to save EXR files with ACEScg color space support
def save_exr_acescg(exr_path, image_data_jax, pixel_type_enum=Imath.PixelType.HALF):
    """
    Save a JAX array (H, W, 3) as an EXR file with ACEScg primaries.
    Defaults to HALF precision (float16).
    Args:
        exr_path: Path to save the EXR file.
        image_data_jax: JAX array containing linear RGB image data.
        pixel_type_enum: Imath.PixelType enum value (e.g., Imath.PixelType.HALF or Imath.PixelType.FLOAT). Defaults to HALF.
    """
    try:
        # Determine NumPy dtype based on pixel_type_enum
        if pixel_type_enum == Imath.PixelType.HALF:
            numpy_dtype = np.float16
        elif pixel_type_enum == Imath.PixelType.FLOAT:
            numpy_dtype = np.float32
        else:
            raise ValueError(f"Unsupported pixel type enum: {pixel_type_enum}")

        # Convert JAX array to NumPy array with the correct dtype
        image_data_np = np.asarray(image_data_jax).astype(numpy_dtype)
        height, width, _ = image_data_np.shape

        # Split into R, G, B channels
        r_channel = image_data_np[:, :, 0]
        g_channel = image_data_np[:, :, 1]
        b_channel = image_data_np[:, :, 2]

        # Prepare EXR header
        header = OpenEXR.Header(width, height)
        
        # Define ACEScg primaries and white point (D60 for ACES)
        acescg_chromaticities = Imath.Chromaticities(
            Imath.V2f(0.713, 0.293),  # Red
            Imath.V2f(0.165, 0.830),  # Green
            Imath.V2f(0.128, -0.044), # Blue
            Imath.V2f(0.32168, 0.33767) # White point (D60)
        )
        header['chromaticities'] = acescg_chromaticities
        
        # Create the channel type description based on the enum value
        channel_definition = Imath.Channel(Imath.PixelType(pixel_type_enum))
        header['channels'] = {
            'R': channel_definition,
            'G': channel_definition,
            'B': channel_definition
        }
        header['compression'] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION) # Use efficient compression

        # Create EXR file
        exr_file = OpenEXR.OutputFile(exr_path, header)

        # Write pixel data (already converted to correct dtype)
        r_bytes = r_channel.tobytes()
        g_bytes = g_channel.tobytes()
        b_bytes = b_channel.tobytes()

        exr_file.writePixels({'R': r_bytes, 'G': g_bytes, 'B': b_bytes})
        exr_file.close()
        print(f"Image saved to {exr_path} with ACEScg primaries (Precision: {numpy_dtype.__name__}).")

    except Exception as e:
        print(f"Error saving EXR: {e}")
        raise

# --- REINSTATED Loss function --- 
# Loss function: Mean Squared Error between rendered and target image
# Updated to compare batches of values (e.g., from surface points)
def loss_fn(rendered_image, target_image, mask_patch=None, start_y=0, start_x=0, patch_size=None):
    """Calculates MSE loss, optionally over a patch and using a mask."""
    if patch_size is not None:
        # Rendered image IS the patch when patch args are used in render_with_params
        rendered_patch = rendered_image 
        
        # Slice the target image to get the corresponding patch
        # Ensure patch doesn't go out of bounds (basic clipping)
        max_y = target_image.shape[0] - patch_size
        max_x = target_image.shape[1] - patch_size
        start_y = jnp.clip(start_y, 0, max_y)
        start_x = jnp.clip(start_x, 0, max_x)
        target_patch = jax.lax.dynamic_slice(
            target_image, (start_y, start_x, 0), (patch_size, patch_size, 3)
        )
        
        # Calculate squared error
        squared_error = (rendered_patch - target_patch) ** 2
        
        # Apply mask if provided (mask_patch should be the sliced mask)
        if mask_patch is not None:
            # Ensure mask_patch has the same H, W dimensions as image patches
            # Assume mask_patch is (patch_size, patch_size) or (patch_size, patch_size, 1)
            # Broadcast mask to match the shape of squared_error (H, W, C)
            mask_broadcast = jnp.expand_dims(mask_patch, axis=-1) if mask_patch.ndim == 2 else mask_patch
            # Apply mask and calculate mean over masked pixels
            masked_squared_error = squared_error * mask_broadcast
            # Calculate mean only over the masked area (where mask > 0)
            # Add epsilon to avoid division by zero if mask is all zeros
            mask_sum = jnp.sum(mask_broadcast) + 1e-8 
            loss = jnp.sum(masked_squared_error) / mask_sum
        else:
            # No mask, calculate standard MSE over the patch
            loss = jnp.mean(squared_error)
            
        return loss
    else:
        # Full image loss (masking could be added here too if needed)
        squared_error = (rendered_image - target_image) ** 2
        if mask_patch is not None: # Assume mask_patch is full mask here
            mask_broadcast = jnp.expand_dims(mask_patch, axis=-1) if mask_patch.ndim == 2 else mask_patch
            masked_squared_error = squared_error * mask_broadcast
            mask_sum = jnp.sum(mask_broadcast) + 1e-8
            loss = jnp.sum(masked_squared_error) / mask_sum
        else:
            loss = jnp.mean(squared_error)
            
        return loss

# Define static args for the *pixel* objective JIT
# Renamed from _surface_objective_static_argnames
_pixel_objective_static_argnames = (
    'width', 'height', # Image dimensions
    # Placeholder for mesh structure info if needed statically
    'plane_d', 'plane_mat_id_a', 'plane_mat_id_b', 'plane_checker_scale',
    'num_samples_per_step' # Number of pixels sampled per step
)

# --- Mesh Data Loading --- 
def load_mesh_data(obj_path):
    """
    Loads mesh data (vertices, faces, UVs) from an OBJ file using pywavefront.
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
        for mesh in scene.mesh_list:
            print(f"  Processing mesh with {len(mesh.materials)} materials...")
            for material in mesh.materials:
                # material.vertices is the interleaved data: [T1N1V1 T1N1V2 T1N1V3 T2N2V1 T2N2V2 T2N2V3 ...]
                # Format depends on what's in the OBJ (e.g., V, V/T, V//N, V/T/N)
                vertex_data = np.array(material.vertices, dtype=np.float32)
                vertex_format = material.vertex_format
                # --- Correct Stride Calculation --- 
                # Calculate stride based on format components (T2F, N3F, V3F etc.)
                stride = 0
                if 'T' in vertex_format: stride += 2 # T2F
                if 'C' in vertex_format: stride += 3 # C3F (not common)
                if 'N' in vertex_format: stride += 3 # N3F
                if 'V' in vertex_format: stride += 3 # V3F
                if stride == 0: raise ValueError(f"Could not determine vertex stride from format: {vertex_format}")
                # --- End Correct Stride Calculation ---
                
                # Ensure data length is multiple of stride
                if len(vertex_data) % stride != 0:
                    raise ValueError(f"Vertex data length ({len(vertex_data)}) not divisible by calculated stride ({stride}) for format {vertex_format}")
                num_vertices_in_material = len(vertex_data) // stride
                
                print(f"    Material vertex format: {vertex_format}, Stride: {stride}, Num Verts: {num_vertices_in_material}")

                # Extract vertices, UVs, normals based on format
                # Indices for different components within the stride
                current_offset = 0
                vt_indices, vn_indices, v_indices = None, None, None
                if 'T' in vertex_format: 
                    vt_indices = slice(current_offset, current_offset + 2)
                    current_offset += 2
                if 'C' in vertex_format: # Add color handling if needed later
                    # c_indices = slice(current_offset, current_offset + 3)
                    current_offset += 3 
                if 'N' in vertex_format: 
                    vn_indices = slice(current_offset, current_offset + 3)
                    current_offset += 3
                if 'V' in vertex_format: 
                    v_indices = slice(current_offset, current_offset + 3)
                    current_offset += 3
                
                reshaped_data = vertex_data.reshape(-1, stride)

                # Extract Vertices (mandatory)
                if v_indices is None: continue # Skip if no vertex data somehow
                verts = reshaped_data[:, v_indices]
                all_vertices.append(verts)

                # Extract UVs if present
                if vt_indices is not None:
                    uvs = reshaped_data[:, vt_indices]
                    all_uvs.append(uvs)
                else:
                    all_uvs.append(np.zeros((num_vertices_in_material, 2), dtype=np.float32))
                
                # Extract Normals if present
                if vn_indices is not None:
                    norms = reshaped_data[:, vn_indices]
                    all_normals.append(norms)
                else:
                     all_normals.append(np.zeros((num_vertices_in_material, 3), dtype=np.float32))

                # Create simple triangle faces for this chunk of vertices
                # Assumes the interleaved data is already per-triangle vertex
                material_faces = np.arange(vertex_offset, vertex_offset + num_vertices_in_material).reshape(-1, 3)
                all_faces.append(material_faces)
                vertex_offset += num_vertices_in_material

        if vertex_offset == 0:
            raise ValueError("No valid vertex data found in the OBJ file.")

        # Concatenate all parts
        final_vertices = np.concatenate(all_vertices, axis=0)
        final_uvs = np.concatenate(all_uvs, axis=0)
        final_normals = np.concatenate(all_normals, axis=0)
        final_faces = np.concatenate(all_faces, axis=0)

        print(f"Mesh loaded: {final_vertices.shape[0]} unique vertices (in faces), {final_faces.shape[0]} faces.")
        print(f"           UVs shape: {final_uvs.shape}")
        print(f"           Normals shape: {final_normals.shape}")

        # Note: This structure gives unique vertices *per face corner*. 
        # `final_vertices`, `final_uvs`, `final_normals` have the same length.
        # `final_faces` just contains indices like [[0,1,2], [3,4,5], ...]
        return {
            'vertices': jnp.array(final_vertices),
            'faces': jnp.array(final_faces), # Simple sequential indices now
            'uvs': jnp.array(final_uvs),
            'normals': jnp.array(final_normals), # Store normals too
            # No need for face_uv_indices as data is directly indexed by face indices
        }
        
    except Exception as e:
        print(f"Error loading OBJ file with interleaved format: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        raise
# -------------------------

# Renamed from placeholder_unproject_and_intersect
def unproject_and_intersect_mesh_bruteforce(px, py, camera, mesh_data, width: int, height: int):
    """ 
    Uses JAX ray-triangle intersection against all mesh triangles (brute force).
    Interpolates attributes (UVs, Normals) from the direct vertex data.
    Returns: hit_pos (3,), hit_normal (3,), hit_uv (2,), is_valid (bool)
    WARNING: Still VERY SLOW for complex meshes.
    """
    # 1. Generate Ray (Pinhole Model)
    # --- Apply Inverse Distortion --- 
    # Convert pixel coordinates to normalized image plane coordinates
    x_n = (px - camera.cx) / camera.fx
    y_n = (py - camera.cy) / camera.fy

    # Apply Brown-Conrady distortion model (radial + tangential)
    # Note: This applies the *forward* distortion. An iterative method is
    # usually needed for the true inverse. This is an approximation for consistency.
    r2 = x_n**2 + y_n**2
    r4 = r2**2
    r6 = r2**3
    radial_dist = (1 + camera.k1 * r2 + camera.k2 * r4 + camera.k3 * r6)
    x_distorted = x_n * radial_dist + (2 * camera.p1 * x_n * y_n + camera.p2 * (r2 + 2 * x_n**2))
    y_distorted = y_n * radial_dist + (camera.p1 * (r2 + 2 * y_n**2) + 2 * camera.p2 * x_n * y_n)

    # Use distorted normalized coordinates to form camera-space direction
    view_dir_cam = normalize(jnp.array([x_distorted, y_distorted, -1.0]))
    # --- End Apply Inverse Distortion ---

    # --- Calculate Camera Transformation Matrix --- 
    # Forward direction (from eye to center)
    forward = normalize(camera.center - camera.eye)
    # Right direction (cross product of up and forward)
    # Normalize the provided up vector first
    cam_up_norm = normalize(camera.up)
    right = normalize(jnp.cross(forward, cam_up_norm))
    # Recalculate actual up vector (orthogonal to right and forward)
    up_actual = jnp.cross(right, forward)
    
    # Construct rotation part of camera-to-world matrix (transposed view matrix rotation)
    cam_to_world_rot = jnp.stack([right, up_actual, -forward], axis=-1)
    
    # Inverse of view matrix (Camera-to-World matrix)
    # world_matrix = jnp.eye(4)
    # world_matrix = world_matrix.at[:3, :3].set(cam_to_world_rot)
    # world_matrix = world_matrix.at[:3, 3].set(camera.eye)
    # --- End Calculate --- 

    # Transform view direction from camera space (-Z forward) to world space
    view_dir_world = cam_to_world_rot @ view_dir_cam 
    ray_origin = camera.eye
    ray_direction = normalize(view_dir_world)

    # --- Ray-Mesh Intersection (Test ALL Triangles using direct vertex data) --- 
    # Access the potentially large vertex/uv/normal arrays
    vertices = mesh_data.get('vertices') # Shape (N*3, 3) where N is num faces
    faces = mesh_data.get('faces')       # Shape (N, 3) -> [[0,1,2], [3,4,5], ...]
    uvs = mesh_data.get('uvs')           # Shape (N*3, 2)
    normals = mesh_data.get('normals')   # Shape (N*3, 3)
    has_uvs = uvs is not None
    has_normals = normals is not None

    closest_t = jnp.inf
    closest_face_idx = -1
    closest_bary = jnp.array([1.0, 0.0, 0.0])
    final_valid_hit = False

    if vertices is not None and faces is not None and faces.shape[0] > 0:
        num_faces = faces.shape[0]
        
        # Function to process a single triangle
        def intersect_single_triangle(face_index):
            # Get indices for the three vertices of this face
            v_idx0, v_idx1, v_idx2 = faces[face_index] 
            # Get the actual vertex positions using these indices
            v0 = vertices[v_idx0]
            v1 = vertices[v_idx1]
            v2 = vertices[v_idx2]
            
            t, u, v, valid_hit = intersect_ray_triangle_moller_trumbore(
                ray_origin, ray_direction, v0, v1, v2
            )
            return t, jnp.array([1.0 - u - v, u, v]), valid_hit, face_index

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
             f_idx = all_indices[closest_idx_in_map] # This is the actual face index
             return bary, f_idx
        closest_bary, closest_face_idx = lax.cond(
            final_valid_hit,
            get_closest_hit_info,
            lambda: (jnp.array([1.0, 0.0, 0.0]), -1) 
        )

    # --- Calculate final world position, normal, and UV --- 
    hit_pos_world = ray_origin + ray_direction * closest_t
    hit_normal_world_default = jnp.array([0.0, 0.0, 0.0]) 
    hit_uv_interpolated_default = jnp.array([0.0, 0.0])

    # Calculate actual normal and UV if there was a valid hit by interpolating
    def calculate_final_attrs(_): # Accept one (unused) argument for lax.cond
        # Get the indices of the 3 vertices for the hit face
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
            uv = jnp.array([0.0, 0.0])
        return normal, uv

    hit_normal_world, hit_uv_interpolated = lax.cond(
        final_valid_hit,
        calculate_final_attrs, 
        lambda defaults: defaults, 
        (hit_normal_world_default, hit_uv_interpolated_default) 
    )

    # Removed sphere fallback - script now requires valid mesh
    # Ensure defaults if mesh intersection itself failed
    if not (vertices is not None and faces is not None and faces.shape[0] > 0):
         final_valid_hit = False
         hit_pos_world = jnp.zeros(3) 
         hit_normal_world = hit_normal_world_default
         hit_uv_interpolated = hit_uv_interpolated_default

    return hit_pos_world, hit_normal_world, hit_uv_interpolated, final_valid_hit
# ----------------------------------------

# JIT the objective function based on sampled pixels
@partial(jax.jit, static_argnames=_pixel_objective_static_argnames)
def objective_fn_pixel(
     params_flat, 
    # --- Dynamic Args ---
    # Sampled pixel data
    sampled_pixels_xy: jnp.ndarray,    # (N, 2) integer pixel coordinates [x, y]
    # --- FIX: Target Spectral data corresponding to these pixels ---
    target_spds_at_pixels: jnp.ndarray, # (N, N_WAVELENGTHS)
    # --- END FIX ---
    # Pixel mask values (0 or 1)
    mask_values_at_pixels: jnp.ndarray,# (N,)
    # Other dynamic scene elements
    params_template, camera, mesh_data, # Pass mesh data
    fixed_material_spds, background_spd, plane_norm,
    stacked_dir_lights_scene, stacked_point_lights_scene,
    # --- Static Primitive Args (passed last) ---
    width, height, 
    plane_d, plane_mat_id_a, plane_mat_id_b, plane_checker_scale,
    num_samples_per_step # Static number of samples
):
    """Computes loss between directly rendered pixels (spectral) and target spectral values."""
    current_params_dict = unflatten_params(params_flat, params_template)
    
    # Pre-process UDIM textures needed by render_direct_illumination
    # TODO: Handle multiple UDIMs properly if params_template contains them
    udim_keys_sorted = tuple(sorted(current_params_dict.keys()))
    udim_texture_tiles = tuple(current_params_dict[k] for k in udim_keys_sorted)
    udim_shapes = tuple(t.shape for t in udim_texture_tiles)
    default_spd = jnp.full((N_WAVELENGTHS,), 0.1) # Fallback default

    # Define function to unproject, render (spectral), and check validity for a single pixel
    def process_single_pixel(pixel_xy):
        px, py = pixel_xy[0], pixel_xy[1]

        # 1. Unproject pixel
        hit_pos, hit_normal, hit_uv, is_valid_hit = unproject_and_intersect_mesh_bruteforce(
            px, py, camera, mesh_data, width, height 
        )

        # 2. Render direct illumination (spectral) at the hit point IF valid
        def render_valid_hit():
            view_dir = normalize(camera.eye - hit_pos)
            spectral_radiance = render_direct_illumination(
                hit_pos=hit_pos,
                hit_normal=hit_normal,
                hit_uv=hit_uv,
                view_dir=view_dir,
                udim_texture_tiles=udim_texture_tiles,
                default_spd=default_spd,
                params_template=params_template, 
                plane_normal=plane_norm,
                udim_keys_sorted=udim_keys_sorted,
                udim_shapes=udim_shapes,
                plane_d=plane_d,
                plane_mat_id_a=plane_mat_id_a,
                plane_mat_id_b=plane_mat_id_b,
                plane_checker_scale=plane_checker_scale,
                fixed_material_spds=fixed_material_spds, 
                stacked_directional_lights=stacked_dir_lights_scene,
                stacked_point_lights=stacked_point_lights_scene,
                mesh_data=mesh_data
            )
            return spectral_radiance

        rendered_spd = jax.lax.cond(
            is_valid_hit,
            render_valid_hit, 
            lambda: jnp.zeros(N_WAVELENGTHS), # Return zero spectrum on miss
        )
        # --- DEBUG: Return hit_uv along with rendered_spd and validity --- 
        return rendered_spd, is_valid_hit, hit_uv
        # --- END DEBUG ---

    # Vmap over the sampled pixels
    # --- DEBUG: Capture hit_uv_batch --- 
    rendered_spds_batch, valid_hit_flags, hit_uv_batch = jax.vmap(process_single_pixel)(sampled_pixels_xy)
    # --- END DEBUG ---

    # --- FIX: Calculate loss in Spectral space --- 
    squared_error = (rendered_spds_batch - target_spds_at_pixels) ** 2
    
    # Combine validity flags: must be a valid hit AND not masked out
    combined_mask = valid_hit_flags * mask_values_at_pixels # Element-wise multiplication
    
    # Apply combined mask (broadcast to spectral channels)
    masked_squared_error = squared_error * combined_mask[:, None] # Expand mask shape from (N,) to (N, 1)
    
    # Calculate mean loss over valid, unmasked pixels and spectral channels
    valid_pixel_count = jnp.sum(combined_mask) + 1e-8 # Count valid pixels
    # Sum error over spectral channels first, then average over valid pixels
    loss_per_pixel = jnp.sum(masked_squared_error, axis=-1) # Sum over spectral dimension
    total_loss = jnp.sum(loss_per_pixel) # Sum loss across valid pixels
    loss = total_loss / valid_pixel_count # Average over number of valid pixels
    # --- END FIX ---

    # --- DEBUG: Return hit_uv_batch as aux data --- 
    return loss, hit_uv_batch # Return loss and UVs for inspection
    # --- END DEBUG ---

# Value and Grad function for the pixel objective
# --- DEBUG: Update value_and_grad to handle aux output --- 
value_and_grad_fn_pixel = jax.value_and_grad(objective_fn_pixel, has_aux=True)
# --- END DEBUG ---

# JIT the value_and_grad function
value_and_grad_fn_pixel_jit = partial(jax.jit, static_argnames=_pixel_objective_static_argnames)(value_and_grad_fn_pixel)

def flatten_params(params_dict):
    """
    Flatten a dictionary of {UDIM_ID: texture_array} into a 1D array.
    """
    flat_params = []
    # Sort keys for consistent order
    for udim_id in sorted(params_dict.keys()):
        texture_array = params_dict[udim_id]
        if isinstance(texture_array, (np.ndarray, jnp.ndarray)):
            flat_params.extend(texture_array.flatten())
        else:
            # Handle potential scalar parameters if ever needed, though textures are arrays
            flat_params.append(texture_array) 
            
    return jnp.array(flat_params)

def unflatten_params(flat_params, template_dict):
    """
    Reconstitute a dictionary of {UDIM_ID: texture_array} from a flat array
    using the template_dict for keys and shapes.
    """
    result = {}
    idx = 0
    # Use the sorted keys from the template for structure
    for udim_id in sorted(template_dict.keys()):
        template_array = template_dict[udim_id]
        if isinstance(template_array, (np.ndarray, jnp.ndarray)):
            shape = template_array.shape
            size = np.prod(shape)
            # Ensure flat_params has enough elements left
            if idx + size > flat_params.shape[0]:
                 raise ValueError(f"Not enough elements in flat_params to unflatten UDIM {udim_id} with size {size}. Index: {idx}, Flat size: {flat_params.shape[0]}")
            result[udim_id] = flat_params[idx:idx+size].reshape(shape)
            idx += size
        else:
            # Handle potential scalar parameters
            if idx >= flat_params.shape[0]:
                 raise ValueError(f"Not enough elements in flat_params to unflatten scalar for UDIM {udim_id}. Index: {idx}, Flat size: {flat_params.shape[0]}")
            result[udim_id] = flat_params[idx]
            idx += 1
            
    # Optional: Check if all elements from flat_params were used
    if idx != flat_params.shape[0]:
        print(f"Warning: {flat_params.shape[0] - idx} elements remaining in flat_params after unflattening.")
        
    return result

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Inverse rendering optimization")
    parser.add_argument("--target-exr", type=str, help="Path to target EXR image for optimization (optional)")
    parser.add_argument("--obj-path", type=str, help="Path to the 3D object file (e.g., .obj)") # Added OBJ path arg
    parser.add_argument("--width", type=int, default=1024, help="Rendering width (smaller default for speed)")
    parser.add_argument("--height", type=int, default=1024, help="Rendering height (smaller default for speed)")
    parser.add_argument("--steps", type=int, default=50, help="Number of optimization steps (smaller default for speed)")
    parser.add_argument("--mask-path", type=str, help="Path to mask image (optional, grayscale/binary)") # Added mask arg
    args = parser.parse_args()
    
    # --- Shared Scene Setup ---
    width = args.width
    height = args.height
    rng_key = jax.random.PRNGKey(42)

    # --- Load Actual Mesh Data --- 
    if args.obj_path:
        try:
            mesh_data = load_mesh_data(args.obj_path)
        except Exception as e:
            print(f"Failed to load mesh: {e}. Exiting.")
            return
    else:
        print("Error: --obj-path is required for mesh-based optimization. Exiting.")
        # Or fallback to sphere placeholder if desired:
        # print("Warning: No --obj-path specified. Using sphere placeholder logic.")
        # mesh_data = {'center': jnp.array([0.0, 1.2, 0.0])} 
        return
    # -----------------------------

    # --- Load Mask (Optional) --- (Keep existing logic)
    mask_image = None
    if args.mask_path:
        if os.path.exists(args.mask_path):
            try:
                print(f"Loading mask image from: {args.mask_path}")
                mask_pil = Image.open(args.mask_path).convert('L') # Load as grayscale
                if mask_pil.width != width or mask_pil.height != height:
                    print(f"Warning: Mask dimensions ({mask_pil.width}x{mask_pil.height}) don't match rendering resolution ({width}x{height}). Resizing mask...")
                    mask_pil = mask_pil.resize((width, height), Image.NEAREST)
                mask_np = np.array(mask_pil).astype(np.float32) / 255.0
                mask_image = jnp.array(mask_np) # Shape (H, W)
                # Ensure mask is binary (0 or 1) for multiplication logic
                mask_image = (mask_image > 0.5).astype(jnp.float32)
                print("Mask loaded and binarized successfully.")
            except Exception as e:
                print(f"Error loading mask image: {e}. Continuing without mask.")
        else:
            print(f"Warning: Mask file not found at {args.mask_path}. Continuing without mask.")
    # If no mask provided, create a dummy mask of all ones
    if mask_image is None:
        print("No mask provided, using a full image mask (all ones).")
        mask_image = jnp.ones((height, width), dtype=jnp.float32)
    # --------------------------

    # Define Geometry using Sphere class then stack into arrays - REMOVE THIS
    # sphere = Sphere(center=jnp.array([0.0, 1.2, 0.0]), radius=jnp.array(1.0), material_id=0)
    # sphere_center = sphere.center # Now part of mesh_data_placeholder if needed

    # Fixed Camera (Keep existing definition)
    # ... (camera setup remains the same) ...
    focal_length = 1671.1020503131649
    k1 = 0.0050715816041100002
    k2 = -0.0024127882483701408
    k3 = -0.0012653936614391772
    p1 = -0.0005047349357100193
    p2 = 0.00094860125385438975
    center_x = width / 2.0
    center_y = height / 2.0
    camera = Camera(
        fx=focal_length, fy=focal_length, cx=center_x, cy=center_y,
        eye=jnp.array([-2.5, 3.0, 4.5]), 
        center=jnp.array([0.0, 1.0, 0.0]), # Look towards origin area
        up=jnp.array([0.0, 1.0, 0.0]),
        k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
        f_number=11.0, shutter_speed=1/500.0, iso=400.0
    )

    # --- Define Target & Fixed Materials (Spectrally) --- (Keep existing)
    # ... (fixed_material_spds, background_spd setup remains the same) ...
    fixed_spd_checker_a = create_spectrum_profile("white_paint", 0.18)
    fixed_spd_checker_b = create_spectrum_profile("white_paint", 0.9)
    fixed_material_spds = jnp.stack([fixed_spd_checker_a, fixed_spd_checker_b])
    background_spd = create_spectrum_profile("daylight_sunset", 1.2)
    target_spd_sphere_synthetic = create_spectrum_profile("fluorescent_pink", 1.0)

    # --- ADD Lights Definition (Matching forward_render.py) --- (Keep existing)
    # ... (light definitions and stacking remain the same) ...
    print("Defining light sources...")
    light1_dir_from = normalize(jnp.array([-1.0, -1.5, -0.8]))
    light1_spd = create_spectrum_profile("led_white_cool", 1.0) 
    light2_dir_from = normalize(jnp.array([10.0, -5.5, 0.5]))
    light2_spd = create_spectrum_profile("blackbody_emitter", 3200)
    light3_pos = jnp.array([2.0, 3.0, 2.0])
    light3_spd = create_spectrum_profile("laser_blue", 150.0)
    stacked_dir_lights_target = DirectionalLight(
        direction=jnp.stack([normalize(-light1_dir_from), normalize(-light2_dir_from)]),
        spd=jnp.stack([light1_spd, light2_spd])
    )
    stacked_point_lights_target = PointLight(
        position=jnp.stack([light3_pos]),
        spd=jnp.stack([light3_spd])
    )
    stacked_dir_lights_scene = stacked_dir_lights_target # Use same lights for scene/target gen
    stacked_point_lights_scene = stacked_point_lights_target

    # --- Generate or Load Target Image --- (Keep existing logic, ensure render uses sphere_center)
    SYNTHETIC_TARGET_FILENAME = "target_render_synthetic.exr"
    if args.target_exr:
        # ... (loading logic remains same) ...
        print(f"Loading target EXR image from: {args.target_exr}")
        target_image = load_exr_acescg(args.target_exr)
        if target_image.shape[0] != height or target_image.shape[1] != width:
            print(f"Warning: Target EXR dimensions ({target_image.shape[1]}x{target_image.shape[0]}) " 
                 f"don't match rendering resolution ({width}x{height})")
            print("Resizing target image to match rendering resolution...")
            target_np = np.array(target_image)
            img_pil = Image.fromarray((np.clip(target_np, 0, 1) * 255).astype(np.uint8), 'RGB')
            img_pil = img_pil.resize((width, height), Image.LANCZOS)
            target_np = np.array(img_pil).astype(np.float32) / 255.0
            target_image = jnp.array(target_np)
        target_image_np = np.asarray(target_image)
        img_target = Image.fromarray((np.clip(target_image_np, 0, 1) * 255).astype(np.uint8), 'RGB')
        img_target.save("target_exr_reference.png")
        print("Target EXR image loaded and saved as target_exr_reference.png for inspection.")
        # --- FIX: Store the *spectral* data for lookup, not ACEScg ---
        target_image_for_lookup = target_image 
        # target_image = jnp.array(xyz_to_rgb(target_xyz_save, XYZ_TO_ACESCG_MATRIX)) # OLD: Stored ACEScg
        # --- END FIX ---
    else:
        # --- GENERATE TARGET USING DIRECT ILLUMINATION --- 
        print("Generating synthetic target image using DIRECT ILLUMINATION...")
        tex_height_synth = 64
        tex_width_synth = 64
        print(f"(Using {tex_width_synth}x{tex_height_synth} target texture: {target_spd_sphere_synthetic[:3]}...)")
        synthetic_tile_1001 = jnp.tile(target_spd_sphere_synthetic, (tex_height_synth, tex_width_synth, 1))
        # Create the dictionary for the target textures
        synthetic_udim_textures = { 1001: synthetic_tile_1001 }
        # Prepare params needed by render_direct_illumination
        udim_keys_sorted_target = tuple(sorted(synthetic_udim_textures.keys()))
        udim_shapes_target = tuple(v.shape for k, v in synthetic_udim_textures.items())
        udim_texture_tiles_target = tuple(v for k, v in synthetic_udim_textures.items())
        default_spd_target = jnp.full((N_WAVELENGTHS,), 0.1) # Use a default SPD
        params_template_target = { # Dummy template needed by direct renderer call
            'udim_texture_tiles': synthetic_udim_textures,
            'default_spd': default_spd_target
        } 

        # Function to render a single pixel using direct illumination
        def render_pixel_direct(px, py):
            # 1. Unproject pixel
            hit_pos, hit_normal, hit_uv, is_valid_hit = unproject_and_intersect_mesh_bruteforce(
                px, py, camera, mesh_data, width, height
            )
            # 2. Render direct illumination if hit
            def render_valid_hit_direct():
                view_dir = normalize(camera.eye - hit_pos)
                spectral_radiance = render_direct_illumination(
                    hit_pos=hit_pos,
                    hit_normal=hit_normal,
                    hit_uv=hit_uv,
                    view_dir=view_dir,
                    udim_texture_tiles=udim_texture_tiles_target,
                    default_spd=default_spd_target,
                    params_template=params_template_target,
                    plane_normal=plane_norm,
                    mesh_data=mesh_data,
                    udim_keys_sorted=udim_keys_sorted_target,
                    udim_shapes=udim_shapes_target,
                    plane_d=plane_d,
                    plane_mat_id_a=plane_mat_id_a,
                    plane_mat_id_b=plane_mat_id_b,
                    plane_checker_scale=plane_checker_scale,
                    fixed_material_spds=fixed_material_spds, 
                    stacked_directional_lights=stacked_dir_lights_target,
                    stacked_point_lights=stacked_point_lights_target
                )
                return spectral_radiance
            
            rendered_spd = jax.lax.cond(
                is_valid_hit,
                render_valid_hit_direct,
                lambda: background_spd, # Return background if miss
            )
            return rendered_spd

        # --- FIX: Render target row by row to avoid OOM --- 
        print(f"Rendering target image row by row ({height} rows of {width} pixels)...")
        target_spectral_rows = []
        # Use tqdm for progress indication if available, otherwise simple loop
        try:
            from tqdm import tqdm
            row_iterator = tqdm(range(height), desc="Generating Target Rows")
        except ImportError:
            row_iterator = range(height)
            print("tqdm not installed, progress bar unavailable for target generation.")

        for py_val in row_iterator:
            px_row = jnp.arange(width) # x coordinates for the current row
            py_row = jnp.full((width,), py_val) # y coordinate for the current row
            # Vmap render_pixel_direct over the width of the row
            vmapped_render_row_direct = jax.vmap(render_pixel_direct, in_axes=(0, 0))
            rendered_row_spd = vmapped_render_row_direct(px_row, py_row)
            target_spectral_rows.append(rendered_row_spd)
            # Optional: block_until_ready here can help manage memory/CPU usage more granularly
            # rendered_row_spd.block_until_ready()
            
        # Stack the rendered rows into the full image
        target_spectral_intensity = jnp.stack(target_spectral_rows, axis=0) # Shape: (H, W, N_WAVELENGTHS)
        target_spectral_intensity.block_until_ready() # Wait for final computation
        # --- END FIX: Render Target Row by Row ---

        print("Direct illumination target image generated (Spectral).")
        # --- END Generate Target Direct --- 

        # --- Save Target Previews (using the direct result) ---
        target_xyz_save = spectral_to_xyz(target_spectral_intensity)
        target_linear_acescg_save = xyz_to_rgb(target_xyz_save, XYZ_TO_ACESCG_MATRIX)
        # Don't flip for EXR saving
        save_exr_acescg(SYNTHETIC_TARGET_FILENAME, target_linear_acescg_save)
        print(f"Synthetic target saved to {SYNTHETIC_TARGET_FILENAME}")
        # Flip for PNG preview saving
        target_linear_srgb_png = xyz_to_rgb(target_xyz_save, XYZ_TO_SRGB_MATRIX)
        target_linear_srgb_png_flipped = np.flipud(np.asarray(target_linear_srgb_png))
        target_gamma_corrected_png = linear_rgb_to_srgb(target_linear_srgb_png_flipped)
        target_ldr_png = np.clip(target_gamma_corrected_png, 0.0, 1.0)
        img_target_uint8_png = (target_ldr_png * 255).astype(np.uint8)
        img_target_png = Image.fromarray(img_target_uint8_png, 'RGB')
        img_target_png.save("target_render_synthetic_preview.png")
        print("Synthetic target preview saved to target_render_synthetic_preview.png")
        # --- End Save Target Previews ---

        # Set the image for lookup
        target_image_for_lookup = target_spectral_intensity 

    # --- FIX: Ensure target_image_for_lookup is defined and has spectral data ---
    # If loaded from EXR, convert ACEScg to Spectral
    if args.target_exr:
        print("Converting loaded ACEScg target to Spectral...")
        # Need to import rgb_to_spd_d65_approx or similar
        from src.utils import rgb_to_spd_d65_approx 
        target_image_for_lookup = rgb_to_spd_d65_approx(target_image) # target_image holds loaded ACEScg
    # Ensure it's a JAX array
    target_image_jax = jnp.asarray(target_image_for_lookup) 
    print(f"Target image prepared for lookup (Spectral). Shape: {target_image_jax.shape}")
    # --- END FIX ---

    # --- Optimization Setup --- (Keep existing texture init)
    tex_height = 64 
    tex_width = 64
    print(f"Initializing texture map ({tex_width}x{tex_height}) for optimization...")
    initial_tile_1001_spd = create_spectrum_profile("flat", 0.5) 
    initial_tile_1001 = jnp.tile(initial_tile_1001_spd, (tex_height, tex_width, 1))
    params = { 1001: initial_tile_1001 } 
    
    # --- Adam Optimizer Setup --- (Keep existing)
    print("Using Adam optimizer with pixel gradients...")
    learning_rate = 1e-2 
    optimizer = optax.adam(learning_rate)
    # Initialize with flattened params for Adam
    params_flat = flatten_params(params) 
    opt_state = optimizer.init(params_flat) 
    params_template = params # Keep template for unflattening grads

    # --- Start Adam Loop (Pixel Sampling) ---
    print(f"Starting Adam optimization for {args.steps} steps...")

    # Parameters for Pixel Sampling
    num_samples_per_step = 4096 # Number of pixels per step (tuneable)

    loop_key = rng_key 

    for step in tqdm(range(args.steps), desc="Optimizing (Adam Pixels)"): 
        
        # --- Sample Pixel Coordinates (px, py) --- 
        loop_key, sample_key_x, sample_key_y = jax.random.split(loop_key, 3)
        # Sample integer coordinates within image bounds
        sampled_px = jax.random.randint(sample_key_x, (num_samples_per_step,), 0, width)
        sampled_py = jax.random.randint(sample_key_y, (num_samples_per_step,), 0, height)
        # Stack into (N, 2) array [x, y]
        sampled_pixels_xy = jnp.stack([sampled_px, sampled_py], axis=-1)

        # --- FIX: Look up Target SPD and Mask values for sampled pixels --- 
        # Use the correctly prepared spectral target_image_jax
        target_spds_at_pixels = target_image_jax[sampled_py, sampled_px] 
        mask_values_at_pixels = mask_image[sampled_py, sampled_px]
        # --- END FIX ---

        # --- Compute Loss and Gradient for the Sampled Pixels --- 
        # --- DEBUG: Update call to handle aux output (hit_uv_batch) --- 
        (loss_value, hit_uv_batch), grads_flat = value_and_grad_fn_pixel_jit(
        # loss_value, grads_flat = value_and_grad_fn_pixel_jit( # Old call
        # --- END DEBUG ---
            params_flat,
            # Pass dynamic pixel data
            sampled_pixels_xy,
            # Pass dynamic target SPD data
            target_spds_at_pixels,
            mask_values_at_pixels,
            # Pass dynamic scene args
            params_template, camera, mesh_data, # Pass loaded mesh data
            fixed_material_spds, background_spd, plane_norm,
            stacked_dir_lights_scene, stacked_point_lights_scene,
            # Pass static primitive args
            width, height, 
            plane_d, plane_mat_id_a, plane_mat_id_b, plane_checker_scale,
            num_samples_per_step # Pass static num_samples
        )
        loss_value.block_until_ready()
        grads_flat.block_until_ready()
        # --- DEBUG: Block and print UV stats for the first few steps --- 
        if step < 5: 
            hit_uv_batch.block_until_ready() # Ensure UV data is computed
            # Filter out UVs from invalid hits (e.g., where hit_uv might be default [0,0])
            # Note: We need valid_hit_flags here. Let's recompute aux without grad.
            # (Alternatively, modify objective_fn_pixel to return more aux data)
            
            # Simplified approach: Just print stats of all returned UVs for now
            # This might include default UVs if intersection failed, giving misleading min [0,0]
            print(f"  Step {step+1} UV Stats (Batch - N={hit_uv_batch.shape[0]}):")
            print(f"    U Min: {jnp.min(hit_uv_batch[:, 0]):.4f}, Max: {jnp.max(hit_uv_batch[:, 0]):.4f}, Mean: {jnp.mean(hit_uv_batch[:, 0]):.4f}")
            print(f"    V Min: {jnp.min(hit_uv_batch[:, 1]):.4f}, Max: {jnp.max(hit_uv_batch[:, 1]):.4f}, Mean: {jnp.mean(hit_uv_batch[:, 1]):.4f}")
        # --- END DEBUG ---

        # --- Safeguard Gradients --- (Keep existing)
        grads_flat = jnp.where(jnp.isnan(grads_flat) | jnp.isinf(grads_flat), jnp.zeros_like(grads_flat), grads_flat)

        # --- Gradient Debug Prints --- (Keep existing)
        grad_norm = jnp.linalg.norm(grads_flat)
        grad_max = jnp.max(jnp.abs(grads_flat))
        grad_mean_abs = jnp.mean(jnp.abs(grads_flat))
        print(f"    Grad Norm: {grad_norm:.6e}, Max Abs: {grad_max:.6e}, Mean Abs: {grad_mean_abs:.6e}")

        print(f"Step {step+1}/{args.steps}: Pixel Loss = {loss_value:.6f}")

        if jnp.isnan(loss_value) or jnp.isinf(loss_value):
            print(f"    !!! WARNING: Loss is NaN/Inf at step {step + 1}. Stopping optimization. !!!")
            break 
            
        # --- Adam Update --- (Operate on flat params)
        updates, opt_state = optimizer.update(grads_flat, opt_state, params_flat)
        params_flat = optax.apply_updates(params_flat, updates)
        
        # --- Apply Constraints (Clipping) on flat params before unflattening for preview --- 
        # Need to know structure/bounds if params are mixed types
        # Assuming params_flat is only texture data for now:
        params_flat = jnp.clip(params_flat, 0.0, 2.0) # Clip texture values

        # --- Periodic Preview Rendering --- 
        if (step + 1) % 10 == 0:
            # Unflatten current params for rendering
            params_current = unflatten_params(params_flat, params_template)
            print(f"Saving intermediate result at iteration {step+1}")
            intermediate_key = jax.random.PRNGKey(42 + step)
            
            opt_udim_keys = tuple(sorted(params_current.keys()))
            opt_udim_shapes = tuple(params_current[k].shape for k in opt_udim_keys)
            opt_udim_tiles = tuple(params_current[k] for k in opt_udim_keys)
            opt_default_spd = jnp.full((N_WAVELENGTHS,), 0.1)

            print(f"Rendering preview using mesh: {args.obj_path}") # DEBUG

            intermediate_stokes = render_image(
                width=width, height=height, samples_per_pixel=DEFAULT_SAMPLES_PER_PIXEL, max_depth=10,
                # Corrected camera parameters
                cam_fx=camera.fx, cam_fy=camera.fy, cam_cx=camera.cx, cam_cy=camera.cy,
                cam_k1=camera.k1, cam_k2=camera.k2, cam_k3=camera.k3, p1=camera.p1, p2=camera.p2,
                cam_f_number=camera.f_number, cam_shutter_speed=camera.shutter_speed, cam_iso=camera.iso,
                udim_keys_sorted=opt_udim_keys,
                udim_shapes=opt_udim_shapes,
                plane_d=plane_d, plane_mat_id_a=plane_mat_id_a, plane_mat_id_b=plane_mat_id_b, plane_checker_scale=plane_checker_scale,
                mesh_data=mesh_data,
                udim_texture_tiles=opt_udim_tiles, 
                fixed_material_spds=fixed_material_spds,
                background_spd=background_spd,
                default_spd=opt_default_spd,
                plane_normal=plane_norm,
                stacked_directional_lights=stacked_dir_lights_scene,
                stacked_point_lights=stacked_point_lights_scene,
                cam_eye=camera.eye,
                cam_center=camera.center,
                cam_up=camera.up,
                rng_key=intermediate_key
            )
            intermediate_stokes.block_until_ready()
            intermediate_spectral = intermediate_stokes[..., 0]
            intermediate_xyz = spectral_to_xyz(intermediate_spectral)
            intermediate_linear_srgb = xyz_to_rgb(intermediate_xyz, XYZ_TO_SRGB_MATRIX)
            intermediate_linear_srgb_np = np.asarray(intermediate_linear_srgb)
            intermediate_linear_srgb_np = np.flipud(intermediate_linear_srgb_np)
            intermediate_gamma_corrected = linear_rgb_to_srgb(intermediate_linear_srgb_np)
            intermediate_ldr = np.clip(intermediate_gamma_corrected, 0.0, 1.0)
            img_intermediate_uint8 = (intermediate_ldr * 255).astype(np.uint8)
            img_intermediate = Image.fromarray(img_intermediate_uint8, 'RGB')
            img_intermediate.save(f"opt_result_iter_{step+1}.png")
            print(f"Intermediate result saved to opt_result_iter_{step+1}.png")

    # --- End Adam Loop --- 
    
    print("Optimization complete.") 

    # Unflatten final parameters
    params_final = unflatten_params(params_flat, params_template)

    # Save final optimized image (Using integrator.render_image) - Keep existing
    # ... (Final render and PNG saving logic remains the same, using params_final)
    print("Rendering final image...")
    final_key = jax.random.PRNGKey(999)
    final_udim_keys = tuple(sorted(params_final.keys()))
    final_udim_shapes = tuple(params_final[k].shape for k in final_udim_keys)
    final_udim_tiles = tuple(params_final[k] for k in final_udim_keys)
    final_default_spd = jnp.full((N_WAVELENGTHS,), 0.1)
    print(f"Rendering final image using mesh: {args.obj_path}") # DEBUG

    final_stokes = render_image(
         width=width, height=height, samples_per_pixel=DEFAULT_SAMPLES_PER_PIXEL, max_depth=10, 
         # Corrected camera parameters
         cam_fx=camera.fx, cam_fy=camera.fy, cam_cx=camera.cx, cam_cy=camera.cy,
         cam_k1=camera.k1, cam_k2=camera.k2, cam_k3=camera.k3, p1=camera.p1, p2=camera.p2,
         cam_f_number=camera.f_number, cam_shutter_speed=camera.shutter_speed, cam_iso=camera.iso,
         udim_keys_sorted=final_udim_keys,
         udim_shapes=final_udim_shapes,
         plane_d=plane_d, plane_mat_id_a=plane_mat_id_a, plane_mat_id_b=plane_mat_id_b, plane_checker_scale=plane_checker_scale,
         mesh_data=mesh_data,
         udim_texture_tiles=final_udim_tiles, # Use final optimized textures
         fixed_material_spds=fixed_material_spds,
         background_spd=background_spd,
         default_spd=final_default_spd,
         plane_normal=plane_norm,
         stacked_directional_lights=stacked_dir_lights_scene, 
         stacked_point_lights=stacked_point_lights_scene,   
         cam_eye=camera.eye,
         cam_center=camera.center,
         cam_up=camera.up,
         rng_key=final_key
    )
    final_stokes.block_until_ready()
    final_spectral = final_stokes[..., 0]
    final_xyz = spectral_to_xyz(final_spectral)
    final_linear_srgb = xyz_to_rgb(final_xyz, XYZ_TO_SRGB_MATRIX)
    final_linear_srgb_np = np.asarray(final_linear_srgb)
    final_linear_srgb_np = np.flipud(final_linear_srgb_np)
    final_gamma_corrected = linear_rgb_to_srgb(final_linear_srgb_np)
    final_render_ldr = np.clip(final_gamma_corrected, 0.0, 1.0)
    img_final_uint8 = (final_render_ldr * 255).astype(np.uint8)
    img_final = Image.fromarray(img_final_uint8, 'RGB')
    img_final.save("inverse_result_final.png")
    print("Final result saved to inverse_result_final.png")

    # Save final optimized texture as EXR (Keep existing)
    # ... (Texture saving logic remains the same, using params_final)
    final_texture_tile = params_final[1001]
    tex_h, tex_w, _ = final_texture_tile.shape
    texture_xyz = spectral_to_xyz(final_texture_tile)
    texture_linear_acescg = xyz_to_rgb(texture_xyz, XYZ_TO_ACESCG_MATRIX)
    save_exr_acescg("optimized_texture_1001.exr", texture_linear_acescg, pixel_type_enum=Imath.PixelType.FLOAT)
    print("Final optimized texture saved to optimized_texture_1001.exr")


if __name__ == "__main__":
    main() 