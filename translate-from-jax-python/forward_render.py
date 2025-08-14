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
from src.camera import Camera # Updated import name
from src.integrator import render_image
# Import spectral utilities
from src.utils import (\
    N_WAVELENGTHS, WAVELENGTHS_NM, \
    spectral_to_xyz, xyz_to_rgb, linear_rgb_to_srgb, \
    XYZ_TO_SRGB_MATRIX,
    XYZ_TO_ACESCG_MATRIX, # Import ACEScg matrix
    create_spectrum_profile # Import the helper
)
# --- ADD Light Type Import ---
from src.types import DirectionalLight, PointLight # Import Spectrum type alias
from typing import List # For list type hint
# --- ADD normalize import ---
from src.utils import normalize
# -------------------------

# --- Material Definition Helper ---

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Minimal JAX Renderer - Forward Rendering")
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--spp', type=int, default=64, help='Samples per pixel')
    parser.add_argument('--max-depth', type=int, default=5, help='Maximum path tracing depth')
    parser.add_argument('--obj-path', type=str, default=None, help='Path to OBJ file for the main mesh')
    # Add other arguments as needed (e.g., output file path)
    args = parser.parse_args()

    width = args.width
    height = args.height
    samples_per_pixel = args.spp
    max_depth = args.max_depth
    obj_path = args.obj_path
    # exposure_scale = 0.05 / 1024.0 # REMOVED - Handled by physical camera params

    # --- Noise Setup --- REMOVED
    # noise_texture_size = 256 
    # noise_octaves = 4
    # noise_world_scale = 10.0 
    # noise_strength = 1.3 
    # print(f"Generating {noise_texture_size}x{noise_texture_size} Perlin noise gradient texture...")
    # noise_gen = PerlinNoise(octaves=noise_octaves)
    # noise_height_map = np.zeros((noise_texture_size, noise_texture_size))
    # for i in range(noise_texture_size):
    #     for j in range(noise_texture_size):
    #         noise_height_map[i, j] = noise_gen([i / noise_texture_size, j / noise_texture_size])
    # dy_noise, dx_noise = np.gradient(noise_height_map)
    # noise_gradient_texture_np = np.stack([dx_noise, dy_noise], axis=-1)
    # noise_gradient_texture = jnp.array(noise_gradient_texture_np)
    # print("Noise texture generated.")

    # --- Load Mesh Data ---
    if obj_path and os.path.exists(obj_path):
        print(f"Object path provided: {obj_path}")
        mesh_data = load_mesh_data(obj_path)
    else:
        if obj_path:
            print(f"Warning: OBJ file not found at {obj_path}. Proceeding without mesh or with default.")
        else:
            print("No OBJ file provided. Proceeding without mesh or with default.")
        # Create a default empty mesh if no obj is provided or found
        mesh_data = {
            'vertices': jnp.array([]).reshape(0, 3),
            'faces': jnp.array([]).reshape(0, 3),
            'uvs': jnp.array([]).reshape(0, 2),
            'normals': jnp.array([]).reshape(0, 3)
        }
        # For now, we'll still define the sphere params for material/texture setup,
        # but it won't be directly rendered unless mesh_data is populated with it.
        # This part might need refactoring if the sphere is to be a default scene object.

    # Define extremely bright materials for better visibility during debugging
    spd_sphere = create_spectrum_profile("fluorescent_yellow", 1.0) 
    spd_checker_a = create_spectrum_profile("white_paint", 0.18)  # Gray checkerboard
    spd_checker_b = create_spectrum_profile("white_paint", 0.9)  # White checkerboard
    
    # --- Create UDIM texture dictionary and fixed SPDs --- 
    # Assume mesh (mat_id=0) uses the spd_sphere, plane uses checkers
    # Define texture size (can be small for forward render, not optimized)
    fwd_tex_height = 256
    fwd_tex_width = 256
    # The sphere_tile is intended for material_id 0, which is the mesh material ID
    # If a mesh is loaded, its material ID is 0.
    mesh_material_tile = jnp.tile(spd_sphere, (fwd_tex_height, fwd_tex_width, 1))
    # udim_textures for material_id 0 (the mesh)
    udim_textures = { 1001: mesh_material_tile } 

    # Fixed SPDs for plane materials (mat_id 1 and 2)
    fixed_material_spds = jnp.stack([spd_checker_a, spd_checker_b])
    # --- --- --- 

    # Define Sphere Geometry (only one sphere now) - This is now for fallback or specific material definition
    # The actual geometry rendered as "sphere" will come from mesh_data if an OBJ is loaded.
    # If no OBJ, mesh_data is empty, and no sphere-like mesh will be rendered from here.
    # sphere = Sphere(center=jnp.array([0.0, 1.2, 0.0]), radius=jnp.array(1.0), material_id=0) # material_id 0 is for mesh
    # sphere_center = sphere.center # NO LONGER USED by render_image directly
    # Sphere data arrays (only one sphere) - NO LONGER USED BY RENDER_IMAGE
    # sphere_centers = sphere.center[None, :] # Add batch dim
    # sphere_radii = sphere.radius[None]      # Add batch dim
    # sphere_material_ids = jnp.array([sphere.material_id], dtype=jnp.int32)

    # Define Plane Geometry & Parameters
    plane_normal = jnp.array([0.0, 1.0, 0.0])
    plane_d = 0.0 # Plane at y=0
    plane_mat_id_a = 1 # Index for albedo_checker_a
    plane_mat_id_b = 2 # Index for albedo_checker_b
    plane_checker_scale = 2.0 # Size of checkers

    # Background Spectrum (using blue profile)
    # background_spd = create_spectrum_profile("flat", 0.15) # Dark flat grey background
    background_spd = create_spectrum_profile("daylight_sunset", 1.2) # Sky background

    # --- ADD Lights Definition --- (Using New Structs)
    print("Defining light sources...")
    
    # Define SPDs first
    spd_light1 = create_spectrum_profile("led_white_cool", 1.0)
    spd_light2 = create_spectrum_profile("blackbody_emitter", 3200)
    spd_point_light = create_spectrum_profile("laser_blue", 150.0) # SPD for point light
    
    # Light 1: Directional Light
    # Convention: direction is *TO* the light
    light1 = DirectionalLight(
        direction=normalize(-jnp.array([-1.0, -1.5, -0.8])), # Negate the 'from' direction
        spd=spd_light1
    )
    
    # Light 2: Directional Light
    light2 = DirectionalLight(
        direction=normalize(-jnp.array([10.0, -5.5, 0.5])), # Negate the 'from' direction
        spd=spd_light2
    )
    
    # Light 3: Point Light (Example)
    light3 = PointLight(
        position=jnp.array([2.0, 3.0, 2.0]), 
        spd=spd_point_light
    )
    
    # Create list of lights
    # <<< Stack light attributes >>>
    # Stack directional lights
    stacked_dir_lights = DirectionalLight(
        direction=jnp.stack([light1.direction, light2.direction]),
        spd=jnp.stack([light1.spd, light2.spd])
    )
    
    # Stack point lights (only one in this case)
    stacked_point_lights = PointLight(
        position=jnp.stack([light3.position]), # Still needs stacking for consistent shape
        spd=jnp.stack([light3.spd])
    )
    # <<< END Stack >>>
    # directional_lights = [light1, light2]
    # point_lights = [light3]
    # light_directions = jnp.stack([light1_dir_from, light2_dir_from])
    # light_spds = jnp.stack([light1_spd, light2_spd])
    # --------------------------

    # Camera Definition with Specific Intrinsics and Distortion
    focal_length = 1671.1020503131649
    center_x = width / 2.0
    center_y = height / 2.0
    print(f"Centering principal point: Overriding cx, cy to ({center_x}, {center_y})")
    camera = Camera( # Updated class name
        # Intrinsics (fx/fy from params, cx/cy overridden for centering)
        fx = focal_length,
        fy = focal_length, # Assuming square pixels fx=fy=f
        cx = center_x,    # Overridden: Image center x
        cy = center_y,    # Overridden: Image center y
        # Extrinsics - nice angle for final render
        eye=jnp.array([-2.5, 3.0, 4.5]),
        center=jnp.array([0.0, 1.0, 0.0]), # Still look at sphere
        up=jnp.array([0.0, 1.0, 0.0]),
        # Distortion Coefficients (keep provided params)
        k1 = 0.0050715816041100002,
        k2 = -0.0024127882483701408,
        k3 = -0.0012653936614391772,
        p1 = -0.0005047349357100193,
        p2 = 0.00094860125385438975,
        # Physical Exposure - good settings
        f_number=5.6,
        shutter_speed=1/100,
        iso=640.0
    )

    # --- Rendering ---
    rng_key = jax.random.PRNGKey(0)
    start_time = time.time()

    # --- Pre-process UDIM Textures for JIT --- 
    if udim_textures: # Changed from sphere_udim_textures
        udim_keys_sorted = tuple(sorted(udim_textures.keys()))
        udim_shapes = tuple(udim_textures[key].shape for key in udim_keys_sorted)
        udim_texture_tiles = tuple(udim_textures[key] for key in udim_keys_sorted)
        # Check if shapes match wavelengths
        if udim_shapes and udim_shapes[0][-1] != N_WAVELENGTHS:
            raise ValueError(f"Texture channel count ({udim_shapes[0][-1]}) doesn't match N_WAVELENGTHS ({N_WAVELENGTHS})")
        default_spd = jnp.full((N_WAVELENGTHS,), 0.1) # Dark grey default
    else:
        udim_keys_sorted = tuple()
        udim_shapes = tuple()
        udim_texture_tiles = tuple()
        default_spd = jnp.full((N_WAVELENGTHS,), 0.1)
        print("Warning: No UDIM textures provided for the mesh material.")
        
    # --- Compile and Render --- 
    # Pass Camera params unpacked
    # Pass UDIM data pre-processed
    print(f"Rendering {width}x{height} image at {samples_per_pixel} spp (Spectral)... Camera: f/{camera.f_number:.1f}, {camera.shutter_speed:.4f}s, ISO {camera.iso:.1f}")
    print("Compiling spectral renderer (JIT)...")
    start_time = time.time()
    # --- Call Updated render_image --- 
    rendered_image_stokes = render_image(
        # Static Render Params
        width=width, height=height, samples_per_pixel=samples_per_pixel, max_depth=max_depth,
        # Static Camera Params
        cam_fx=camera.fx, cam_fy=camera.fy, cam_cx=camera.cx, cam_cy=camera.cy,
        cam_k1=camera.k1, cam_k2=camera.k2, cam_k3=camera.k3, p1=camera.p1, p2=camera.p2,
        cam_f_number=camera.f_number, cam_shutter_speed=camera.shutter_speed, cam_iso=camera.iso,
        # Static UDIM Structure
        udim_keys_sorted=udim_keys_sorted,
        udim_shapes=udim_shapes,
        # Static Plane Params
        plane_normal=plane_normal, plane_d=plane_d, 
        plane_mat_id_a=plane_mat_id_a, plane_mat_id_b=plane_mat_id_b, 
        plane_checker_scale=plane_checker_scale,
        # Dynamic Scene/Material Args
        mesh_data=mesh_data, # Added mesh_data
        # sphere_center=sphere_center, # REMOVED
        udim_texture_tiles=udim_texture_tiles,
        fixed_material_spds=fixed_material_spds,
        background_spd=background_spd,
        default_spd=default_spd,
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
    rendered_image_stokes.block_until_ready() # Wait for GPU completion
    end_time = time.time()
    print(f"Rendering finished in {end_time - start_time:.2f} seconds.")

    # --- Post-processing: Convert Spectral Stokes to RGB ---
    print("Converting spectral Stokes image to RGB...")
    # Extract spectral intensity S0 (shape: H, W, N_WAVELENGTHS)
    spectral_intensity = rendered_image_stokes[..., 0]

    # Convert spectral intensity to CIE XYZ
    # spectral_to_xyz expects shape (..., N_WAVELENGTHS)
    image_xyz = spectral_to_xyz(spectral_intensity) # Shape: (H, W, 3)

    # Convert XYZ to Linear sRGB
    image_linear_srgb = xyz_to_rgb(image_xyz, XYZ_TO_SRGB_MATRIX) # Shape: (H, W, 3)

    # --- Save Images ---
    # Convert to NumPy array (float32)
    rendered_image_linear_srgb_np = np.asarray(image_linear_srgb)

    # Flip the image vertically (along height axis)
    rendered_image_linear_srgb_np = np.flipud(rendered_image_linear_srgb_np)

    # 1. Save PNG (Apply sRGB gamma, Clamp LDR)
    print("Saving LDR image (PNG) with sRGB gamma...")
    image_srgb_gamma = linear_rgb_to_srgb(rendered_image_linear_srgb_np)
    rendered_image_ldr = np.clip(image_srgb_gamma, 0.0, 1.0)
    image_uint8 = (rendered_image_ldr * 255).astype(np.uint8)
    img = Image.fromarray(image_uint8, 'RGB')
    output_png_path = "output_render_spectral.png" # New name
    img.save(output_png_path)
    print(f"PNG saved to {output_png_path}")

    # 2. Save EXR (HDR Float16 with ACEScg Primaries)
    print("Saving HDR image (EXR) with ACEScg primaries...")
    output_exr_path = "output_render_spectral_acescg.exr" # New name
    try:
        # Convert XYZ to Linear ACEScg using the correct matrix
        image_linear_acescg = xyz_to_rgb(image_xyz, XYZ_TO_ACESCG_MATRIX)
        image_linear_acescg_np = np.asarray(image_linear_acescg)
        image_linear_acescg_np = np.flipud(image_linear_acescg_np) # Flip ACEScg too
        
        # Convert to float16 (HALF)
        image_f16 = image_linear_acescg_np.astype(np.float16)

        # Get dimensions
        height, width, _ = image_f16.shape

        # Prepare Header
        header = OpenEXR.Header(width, height)

        # Define ACEScg (AP1) primaries and white point
        acescg_chromaticities = Imath.Chromaticities(
            Imath.V2f(0.713, 0.293),  # Red
            Imath.V2f(0.165, 0.830),  # Green
            Imath.V2f(0.128, -0.044), # Blue
            Imath.V2f(0.32168, 0.33767) # White (ACES D60 sim)
        )
        header['chromaticities'] = acescg_chromaticities
        # Optional: Indicate ACES container standard compliance
        header['acesImageContainerFlag'] = 1

        # Define Channels (R, G, B as HALF)
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header['channels'] = {'R': half_chan, 'G': half_chan, 'B': half_chan}

        # Prepare Pixel Data (convert channels to bytes)
        # The OpenEXR library should handle float16 conversion based on header type
        r_bytes = image_f16[:, :, 0].tobytes()
        g_bytes = image_f16[:, :, 1].tobytes()
        b_bytes = image_f16[:, :, 2].tobytes()

        # Write EXR file
        exr_file = OpenEXR.OutputFile(output_exr_path, header)
        exr_file.writePixels({'R': r_bytes, 'G': g_bytes, 'B': b_bytes})
        exr_file.close()
        print(f"EXR saved to {output_exr_path}")

    except ImportError:
         print("Error saving EXR: OpenEXR or Imath module not found.")
         print("Make sure you have installed it (e.g., pip install openexr)")
    except Exception as e:
        print(f"Error saving EXR: {e}")

if __name__ == "__main__":
    main() 