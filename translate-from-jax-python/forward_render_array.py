import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import time
# from tqdm import tqdm # Can remove if not needed elsewhere
import OpenEXR
import Imath
import array # For converting data to bytes
import jax.random
from src.types import Ray, Spectrum, MuellerMatrix # Assume Spectrum is just (N_WAVELENGTHS,) for now
from src.utils import (
    WAVELENGTHS_NM, N_WAVELENGTHS, spectral_to_xyz, xyz_to_rgb, 
    linear_rgb_to_srgb, XYZ_TO_SRGB_MATRIX, XYZ_TO_ACESCG_MATRIX,
    create_spectrum_profile, normalize, load_mesh_data
)
from src.integrator import render_image
from src.camera import Camera
import imageio.v3 as iio # Use imageio v3 for better EXR support
from tqdm import tqdm
import os
import argparse # Added
# from perlin_noise import PerlinNoise # Removed noise import

# Import necessary components from the src directory
# Use absolute imports from src
from src.types import Ray, HitRecord # HitRecord might not be directly needed here
from src.geometry import Sphere # Keep Sphere for definition
# from src.camera import Camera # Updated import name - ALREADY IMPORTED
# from src.integrator import render_image # ALREADY IMPORTED
# Import spectral utilities
# from src.utils import (\ # ALREADY IMPORTED
#     N_WAVELENGTHS, WAVELENGTHS_NM, \
#     spectral_to_xyz, xyz_to_rgb, linear_rgb_to_srgb, \
#     XYZ_TO_SRGB_MATRIX,
#     XYZ_TO_ACESCG_MATRIX, # Import ACEScg matrix
#     create_spectrum_profile # Import the helper
# )
# --- ADD Light Type Import ---
from src.types import DirectionalLight, PointLight # Import Spectrum type alias
from typing import List # For list type hint
# --- ADD normalize import ---
# from src.utils import normalize # ALREADY IMPORTED
# -------------------------

# --- Material Definition Helper ---

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Minimal JAX Renderer - Forward Rendering Array") # Updated description
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--spp', type=int, default=16, help='Samples per pixel (reduced default for array scene)') # Adjusted default spp
    parser.add_argument('--max-depth', type=int, default=5, help='Maximum path tracing depth')
    parser.add_argument('--sphere-obj-path', type=str, default="test_sphere.obj", help='Path to the template sphere OBJ file') # New arg for template
    # Add other arguments as needed (e.g., output file path)
    args = parser.parse_args()

    width = args.width
    height = args.height
    samples_per_pixel = args.spp
    max_depth = args.max_depth
    sphere_obj_template_path = args.sphere_obj_path

    # --- Load Template Mesh Data ---
    if not os.path.exists(sphere_obj_template_path):
        print(f"Error: Template sphere OBJ file not found at {sphere_obj_template_path}. Exiting.")
        return
    
    mesh_data_template = load_mesh_data(sphere_obj_template_path)
    if mesh_data_template['vertices'].shape[0] == 0:
        print(f"Error: Template sphere OBJ file {sphere_obj_template_path} loaded an empty mesh. Exiting.")
        return

    # --- Array Setup ---
    num_spheres_x = 3
    num_spheres_z = 3
    total_spheres = num_spheres_x * num_spheres_z
    sphere_spacing = 3.0  # Spacing between sphere centers
    sphere_radius = 1.0 # Assuming template sphere has radius 1.0, centered at origin. Adjust if not.
    
    # Start building the combined mesh and UDIM textures
    combined_vertices_list = []
    combined_faces_list = []
    combined_uvs_list = []
    combined_normals_list = []
    udim_textures_dict = {}
    vertex_offset = 0

    spd_names = [
        "red", "green", "blue", 
        "gold", "silver", "copper",
        "fluorescent_pink", "fluorescent_green", "fluorescent_yellow"
    ]
    if total_spheres > len(spd_names): # Repeat SPDs if not enough unique ones
        spd_names = (spd_names * (total_spheres // len(spd_names) + 1))[:total_spheres]

    sphere_world_positions = []

    for i in range(total_spheres):
        instance_x_idx = i % num_spheres_x
        instance_z_idx = i // num_spheres_x

        # Calculate sphere position
        pos_x = (instance_x_idx - (num_spheres_x - 1) / 2.0) * sphere_spacing
        pos_y = sphere_radius # Place base of sphere on Y=0 plane, or adjust as desired
        pos_z = (instance_z_idx - (num_spheres_z - 1) / 2.0) * sphere_spacing
        sphere_pos_offset = jnp.array([pos_x, pos_y, pos_z])
        sphere_world_positions.append(sphere_pos_offset)

        # 1. Vertices
        transformed_vertices = mesh_data_template['vertices'] + sphere_pos_offset
        combined_vertices_list.append(transformed_vertices)

        # 2. Faces (offset indices)
        transformed_faces = mesh_data_template['faces'] + vertex_offset
        combined_faces_list.append(transformed_faces)
        
        # 3. Normals (normals are not affected by translation)
        combined_normals_list.append(mesh_data_template['normals'])

        # 4. UVs and UDIM Texture Tile
        udim_u_offset = instance_x_idx
        udim_v_offset = instance_z_idx
        
        transformed_uvs = mesh_data_template['uvs'] + jnp.array([udim_u_offset, udim_v_offset], dtype=jnp.float32)
        combined_uvs_list.append(transformed_uvs)
        
        udim_tile_key = 1001 + udim_u_offset + 10 * udim_v_offset
        
        spd_for_tile = create_spectrum_profile(spd_names[i], 1.0)
        # Create a small (e.g., 2x2) texture tile for this SPD
        udim_tile_texture = jnp.tile(spd_for_tile, (2, 2, 1)) 
        udim_textures_dict[udim_tile_key] = udim_tile_texture
        
        vertex_offset += mesh_data_template['vertices'].shape[0]

    # Concatenate all mesh data
    mesh_data = {
        'vertices': jnp.concatenate(combined_vertices_list, axis=0),
        'faces': jnp.concatenate(combined_faces_list, axis=0),
        'uvs': jnp.concatenate(combined_uvs_list, axis=0),
        'normals': jnp.concatenate(combined_normals_list, axis=0)
    }
    
    # Fixed SPDs for plane materials (mat_id 1 and 2) - keep as is
    spd_checker_a = create_spectrum_profile("white_paint", 0.18)
    spd_checker_b = create_spectrum_profile("white_paint", 0.9)
    fixed_material_spds = jnp.stack([spd_checker_a, spd_checker_b])

    # Define Plane Geometry & Parameters - keep as is
    plane_normal = jnp.array([0.0, 1.0, 0.0])
    plane_d = 0.0 # Place plane at Y=0
    plane_mat_id_a = 1
    plane_mat_id_b = 2
    plane_checker_scale = sphere_spacing # Scale checkers relative to sphere spacing

    # Background Spectrum - keep as is
    background_spd = create_spectrum_profile("daylight_sunset", 0.5) # Dimmer background

    # Lights Definition - keep as is or adjust
    print("Defining light sources...")
    spd_light1 = create_spectrum_profile("led_white_cool", 10.0) # Brighter light
    spd_light2 = create_spectrum_profile("blackbody_emitter", 4500) # Different temp
    spd_point_light = create_spectrum_profile("laser_blue", 200.0)
    
    light1 = DirectionalLight(
        direction=normalize(-jnp.array([-0.5, -1.0, -0.7])), 
        spd=spd_light1
    )
    light2 = DirectionalLight(
        direction=normalize(-jnp.array([0.8, -0.8, 0.5])), 
        spd=spd_light2
    )
    light3 = PointLight( # Move point light to better illuminate array
        position=jnp.array([0.0, sphere_radius + num_spheres_z * sphere_spacing * 0.75, 0.0]), 
        spd=spd_point_light
    )
    stacked_dir_lights = DirectionalLight(
        direction=jnp.stack([light1.direction, light2.direction]),
        spd=jnp.stack([light1.spd, light2.spd])
    )
    stacked_point_lights = PointLight(
        position=jnp.stack([light3.position]),
        spd=jnp.stack([light3.spd])
    )

    # --- Camera Adjustment ---
    # Calculate scene bounds to frame all spheres
    all_sphere_positions_np = np.array([p.tolist() for p in sphere_world_positions])
    min_coords = np.min(all_sphere_positions_np, axis=0) - sphere_radius
    max_coords = np.max(all_sphere_positions_np, axis=0) + sphere_radius
    
    scene_center = (min_coords + max_coords) / 2.0
    scene_extent = np.max(max_coords - min_coords)
    
    # Basic camera setup: look at scene_center, pull back based on extent
    cam_eye_pos = scene_center + jnp.array([scene_extent * 0.6, scene_extent * 0.7, scene_extent * 1.5]) # Angled view
    cam_look_at = jnp.array(scene_center)
    cam_up_dir = jnp.array([0.0, 1.0, 0.0])

    # Estimate FoV needed or set fixed focal length and adjust distance
    # For a fixed focal length, distance determines the view.
    # Let's use a moderately wide lens perspective.
    # The original camera had focal_length = 1671 for 1024px width.
    # This implies a FoV of 2 * atan((1024/2)/1671) approx 34 degrees.
    # To fit 'scene_extent', distance D ~ (scene_extent/2) / tan(FOV/2)
    # distance_factor = 1.5 # Pull back further
    # cam_distance = (scene_extent / (2 * jnp.tan(jnp.radians(34 / 2.0)))) * distance_factor
    # cam_eye_vec = normalize(jnp.array([0.5, 0.4, 1.0])) # Example view direction
    # cam_eye_pos = cam_look_at - cam_eye_vec * cam_distance
    
    focal_length_new = (width / 2.0) / jnp.tan(jnp.radians(60 / 2.0)) # Approx 60 deg FoV

    center_x = width / 2.0
    center_y = height / 2.0
    print(f"Centering principal point: Overriding cx, cy to ({center_x}, {center_y})")
    
    camera = Camera(
        fx=focal_length_new, fy=focal_length_new, # Adjusted focal length for wider view
        cx=center_x, cy=center_y,
        eye=cam_eye_pos,
        center=cam_look_at,
        up=cam_up_dir,
        k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0, # No distortion for simplicity
        f_number=5.6, shutter_speed=1/125, iso=100.0 # Adjusted exposure for potentially brighter scene
    )

    # --- Rendering ---
    rng_key = jax.random.PRNGKey(0)
    
    # --- Pre-process UDIM Textures for JIT --- 
    if udim_textures_dict:
        udim_keys_sorted = tuple(sorted(udim_textures_dict.keys()))
        udim_shapes = tuple(udim_textures_dict[key].shape for key in udim_keys_sorted)
        udim_texture_tiles = tuple(udim_textures_dict[key] for key in udim_keys_sorted)
        if udim_shapes and udim_shapes[0][-1] != N_WAVELENGTHS:
            raise ValueError(f"Texture channel count ({udim_shapes[0][-1]}) doesn't match N_WAVELENGTHS ({N_WAVELENGTHS})")
        default_spd_val = jnp.full((N_WAVELENGTHS,), 0.1) 
    else:
        udim_keys_sorted = tuple()
        udim_shapes = tuple()
        udim_texture_tiles = tuple()
        default_spd_val = jnp.full((N_WAVELENGTHS,), 0.1)
        print("Warning: No UDIM textures provided for the mesh material.")
        
    print(f"Rendering {width}x{height} image at {samples_per_pixel} spp (Spectral)... Camera: f/{camera.f_number:.1f}, {camera.shutter_speed:.4f}s, ISO {camera.iso:.1f}")
    print(f"Combined mesh has {mesh_data['vertices'].shape[0]} vertices and {mesh_data['faces'].shape[0]} faces.")
    print(f"Using {len(udim_textures_dict)} UDIM tiles for sphere materials.")
    print("Compiling spectral renderer (JIT)...")
    start_time = time.time()

    # Convert JAX arrays to Python floats for static JIT args if necessary
    static_cam_fx = float(camera.fx)
    static_cam_fy = float(camera.fy)
    static_cam_cx = float(camera.cx)
    static_cam_cy = float(camera.cy)
    static_cam_k1 = float(camera.k1)
    static_cam_k2 = float(camera.k2)
    static_cam_k3 = float(camera.k3)
    static_p1 = float(camera.p1)
    static_p2 = float(camera.p2)
    static_f_number = float(camera.f_number)
    static_shutter_speed = float(camera.shutter_speed)
    static_iso = float(camera.iso)
    
    rendered_image_stokes = render_image(
        # Static Render Params
        width=width, height=height, samples_per_pixel=samples_per_pixel, max_depth=max_depth,
        # Static Camera Params
        cam_fx=static_cam_fx, cam_fy=static_cam_fy, cam_cx=static_cam_cx, cam_cy=static_cam_cy,
        cam_k1=static_cam_k1, cam_k2=static_cam_k2, cam_k3=static_cam_k3, p1=static_p1, p2=static_p2,
        cam_f_number=static_f_number, cam_shutter_speed=static_shutter_speed, cam_iso=static_iso,
        # Static UDIM Structure
        udim_keys_sorted=udim_keys_sorted,
        udim_shapes=udim_shapes,
        # Static Plane Params (ensure these are all static and correctly ordered)
        plane_d=plane_d, 
        plane_mat_id_a=plane_mat_id_a, 
        plane_mat_id_b=plane_mat_id_b, 
        plane_checker_scale=plane_checker_scale,
        # Dynamic Scene/Material Args (ensure plane_normal is here)
        mesh_data=mesh_data, 
        udim_texture_tiles=udim_texture_tiles,
        fixed_material_spds=fixed_material_spds,
        background_spd=background_spd,
        default_spd=default_spd_val, 
        plane_normal=plane_normal, # Moved to dynamic args section
        # --- Pass Lights --- (New List format)
        # <<< Pass Stacked Structs >>>
        stacked_directional_lights=stacked_dir_lights,
        stacked_point_lights=stacked_point_lights,
        # directional_lights=directional_lights,
        # point_lights=point_lights,
        # <<< END Pass Stacked Structs >>>
        # -----------------
        # Dynamic Camera Pose Args
        cam_eye=camera.eye,
        cam_center=camera.center,
        cam_up=camera.up,
        # Dynamic RNG Key
        rng_key=rng_key
    )
    rendered_image_stokes.block_until_ready() 
    end_time = time.time()
    print(f"Rendering finished in {end_time - start_time:.2f} seconds.")

    # --- Post-processing: Convert Spectral Stokes to RGB ---
    print("Converting spectral Stokes image to RGB...")
    spectral_intensity = rendered_image_stokes[..., 0]
    image_xyz = spectral_to_xyz(spectral_intensity)
    image_linear_srgb = xyz_to_rgb(image_xyz, XYZ_TO_SRGB_MATRIX)

    # --- Save Images ---
    rendered_image_linear_srgb_np = np.asarray(image_linear_srgb)
    rendered_image_linear_srgb_np = np.flipud(rendered_image_linear_srgb_np)

    # 1. Save PNG
    print("Saving LDR image (PNG) with sRGB gamma...")
    image_srgb_gamma = linear_rgb_to_srgb(rendered_image_linear_srgb_np)
    rendered_image_ldr = np.clip(image_srgb_gamma, 0.0, 1.0)
    image_uint8 = (rendered_image_ldr * 255).astype(np.uint8)
    img = Image.fromarray(image_uint8, 'RGB')
    output_png_path = "output_render_array_spectral.png" 
    img.save(output_png_path)
    print(f"PNG saved to {output_png_path}")

    # 2. Save EXR
    print("Saving HDR image (EXR) with ACEScg primaries...")
    output_exr_path = "output_render_array_spectral_acescg.exr"
    try:
        image_linear_acescg = xyz_to_rgb(image_xyz, XYZ_TO_ACESCG_MATRIX)
        image_linear_acescg_np = np.asarray(image_linear_acescg)
        image_linear_acescg_np = np.flipud(image_linear_acescg_np)
        image_f16 = image_linear_acescg_np.astype(np.float16)
        
        exr_height, exr_width, _ = image_f16.shape # Use actual exr_height, exr_width
        header = OpenEXR.Header(exr_width, exr_height)
        acescg_chromaticities = Imath.Chromaticities(
            Imath.V2f(0.713, 0.293), Imath.V2f(0.165, 0.830),
            Imath.V2f(0.128, -0.044), Imath.V2f(0.32168, 0.33767)
        )
        header['chromaticities'] = acescg_chromaticities
        header['acesImageContainerFlag'] = 1
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header['channels'] = {'R': half_chan, 'G': half_chan, 'B': half_chan}
        
        r_bytes = image_f16[:, :, 0].tobytes()
        g_bytes = image_f16[:, :, 1].tobytes()
        b_bytes = image_f16[:, :, 2].tobytes()

        exr_file = OpenEXR.OutputFile(output_exr_path, header)
        exr_file.writePixels({'R': r_bytes, 'G': g_bytes, 'B': b_bytes})
        exr_file.close()
        print(f"EXR saved to {output_exr_path}")

    except ImportError:
         print("Error saving EXR: OpenEXR or Imath module not found.")
    except Exception as e:
        print(f"Error saving EXR: {e}")

if __name__ == "__main__":
    main() 