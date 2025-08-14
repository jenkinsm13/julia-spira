import jax
import jax.numpy as jnp
import pytest # Using pytest framework
import sys
import os

# Add src directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import functions to test from utils
from utils import (
    calculate_sphere_uv, 
    texture_lookup_udim_bilinear, 
    DEFAULT_SPD, # Import default SPD for comparison
    N_WAVELENGTHS, # Import for creating test textures
    normalize # Import if needed for test setup
)

# --- Tests for calculate_sphere_uv ---

def test_sphere_uv_north_pole():
    """Test UV calculation at the North Pole (0, 0, 1)."""
    sphere_center = jnp.array([0.0, 0.0, 0.0])
    hit_point = jnp.array([0.0, 0.0, 1.0]) # Point on unit sphere
    # Expected: v should be 0, u is arbitrary (often convention maps to 0.5)
    expected_uv = jnp.array([0.5, 0.0]) 
    calculated_uv = calculate_sphere_uv(hit_point, sphere_center)
    assert jnp.allclose(calculated_uv, expected_uv, atol=1e-6)

def test_sphere_uv_south_pole():
    """Test UV calculation at the South Pole (0, 0, -1)."""
    sphere_center = jnp.array([0.0, 0.0, 0.0])
    hit_point = jnp.array([0.0, 0.0, -1.0]) # Point on unit sphere
    # Expected: v should be 1, u is arbitrary (often convention maps to 0.5)
    expected_uv = jnp.array([0.5, 1.0])
    calculated_uv = calculate_sphere_uv(hit_point, sphere_center)
    assert jnp.allclose(calculated_uv, expected_uv, atol=1e-6)

def test_sphere_uv_equator_points():
    """Test UV calculation at points on the equator."""
    sphere_center = jnp.array([0.0, 0.0, 0.0])
    # Point +X (1, 0, 0): phi=0 -> u=0.5, theta=pi/2 -> v=0.5
    p_x_pos = jnp.array([1.0, 0.0, 0.0])
    uv_x_pos = calculate_sphere_uv(p_x_pos, sphere_center)
    assert jnp.allclose(uv_x_pos, jnp.array([0.5, 0.5]), atol=1e-6)

    # Point +Y (0, 1, 0): phi=pi/2 -> u=0.75, theta=pi/2 -> v=0.5
    p_y_pos = jnp.array([0.0, 1.0, 0.0])
    uv_y_pos = calculate_sphere_uv(p_y_pos, sphere_center)
    assert jnp.allclose(uv_y_pos, jnp.array([0.75, 0.5]), atol=1e-6)

    # Point -X (-1, 0, 0): phi=pi -> u=1.0 (or 0.0), theta=pi/2 -> v=0.5
    p_x_neg = jnp.array([-1.0, 0.0, 0.0])
    uv_x_neg = calculate_sphere_uv(p_x_neg, sphere_center)
    # arctan2(-0.0, -1.0) -> pi. u = pi/(2pi)+0.5 = 1.0. 
    # Depending on lookup wrap/clamp, 1.0 might behave like 0.0. Test expects 1.0 for now.
    assert jnp.allclose(uv_x_neg, jnp.array([1.0, 0.5]), atol=1e-6) 

    # Point -Y (0, -1, 0): phi=-pi/2 -> u=0.25, theta=pi/2 -> v=0.5
    p_y_neg = jnp.array([0.0, -1.0, 0.0])
    uv_y_neg = calculate_sphere_uv(p_y_neg, sphere_center)
    assert jnp.allclose(uv_y_neg, jnp.array([0.25, 0.5]), atol=1e-6)
    
def test_sphere_uv_offset_center():
    """Test UV calculation with an offset sphere center."""
    sphere_center = jnp.array([1.0, 2.0, 3.0])
    # North pole relative to center
    hit_point = jnp.array([1.0, 2.0, 4.0]) 
    expected_uv = jnp.array([0.5, 0.0]) 
    calculated_uv = calculate_sphere_uv(hit_point, sphere_center)
    assert jnp.allclose(calculated_uv, expected_uv, atol=1e-6)


# --- Tests for texture_lookup_udim_bilinear ---

# Fixture to create a sample UDIM texture dictionary
@pytest.fixture
def sample_udim_textures():
    tex_height, tex_width = 4, 4 # Small texture for easy testing
    # Create tile 1001: Gradient from black to white (all channels same)
    y_vals = jnp.linspace(0, 1, tex_height)
    x_vals = jnp.linspace(0, 1, tex_width)
    yy, xx = jnp.meshgrid(y_vals, x_vals, indexing='ij')
    tile_1001_grey = (yy + xx) / 2.0 # Simple gradient
    tile_1001 = jnp.repeat(tile_1001_grey[:, :, None], N_WAVELENGTHS, axis=2)

    # Create tile 1002: Constant red color (e.g., SPD = [1, 0, ..., 0])
    tile_1002_spd = jnp.zeros((N_WAVELENGTHS,)).at[0].set(0.8) # Example red
    tile_1002 = jnp.tile(tile_1002_spd, (tex_height, tex_width, 1))
    
    return {
        1001: tile_1001,
        1002: tile_1002
    }

def test_udim_lookup_center_tile1(sample_udim_textures):
    """Test lookup at the center of UDIM tile 1001."""
    uv = jnp.array([0.5, 0.5]) # Center of tile 1001
    
    # Expected value at center of the gradient texture (0.5, 0.5) -> (0.5+0.5)/2 = 0.5
    expected_spd_val = 0.5
    expected_spd = jnp.full((N_WAVELENGTHS,), expected_spd_val)
    
    calculated_spd = texture_lookup_udim_bilinear(sample_udim_textures, uv, DEFAULT_SPD)
    assert jnp.allclose(calculated_spd, expected_spd, atol=1e-6)

def test_udim_lookup_corner_tile1(sample_udim_textures):
    """Test lookup at a corner (0,0) of UDIM tile 1001."""
    uv = jnp.array([0.0, 0.0]) # Corner of tile 1001
    
    # Expected value at corner (0,0) -> (0+0)/2 = 0.0
    expected_spd_val = 0.0
    expected_spd = jnp.full((N_WAVELENGTHS,), expected_spd_val)
    
    calculated_spd = texture_lookup_udim_bilinear(sample_udim_textures, uv, DEFAULT_SPD)
    assert jnp.allclose(calculated_spd, expected_spd, atol=1e-6)

def test_udim_lookup_interpolated_tile1(sample_udim_textures):
    """Test interpolated lookup within UDIM tile 1001."""
    # UV corresponds roughly to pixel (1.5, 1.5) in a 4x4 grid (indices [0,3])
    uv = jnp.array([0.5, 0.5]) # Corresponds to center pixel coords (1.5, 1.5) when mapped
    # Let's test slightly off-center, e.g., u=0.6, v=0.4
    uv_interp = jnp.array([0.6, 0.4])
    
    # Expected value for gradient (v+u)/2 at (0.4, 0.6) -> (0.4+0.6)/2 = 0.5
    # Bilinear interpolation should give this exact value for this simple case
    expected_spd_val = 0.5 
    expected_spd = jnp.full((N_WAVELENGTHS,), expected_spd_val)

    calculated_spd = texture_lookup_udim_bilinear(sample_udim_textures, uv_interp, DEFAULT_SPD)
    assert jnp.allclose(calculated_spd, expected_spd, atol=1e-6)


def test_udim_lookup_tile2(sample_udim_textures):
    """Test lookup within UDIM tile 1002."""
    uv = jnp.array([1.5, 0.5]) # Center of tile 1002 (u=[1,2), v=[0,1))
    
    # Expected value is the constant red SPD from tile 1002
    expected_spd = jnp.zeros((N_WAVELENGTHS,)).at[0].set(0.8)
    
    calculated_spd = texture_lookup_udim_bilinear(sample_udim_textures, uv, DEFAULT_SPD)
    assert jnp.allclose(calculated_spd, expected_spd, atol=1e-6)

def test_udim_lookup_missing_tile(sample_udim_textures):
    """Test lookup targeting a non-existent UDIM tile (e.g., 1011)."""
    uv = jnp.array([0.5, 1.5]) # Targets tile 1011 (u=[0,1), v=[1,2))
    
    # Expected value is the default SPD
    expected_spd = DEFAULT_SPD
    
    calculated_spd = texture_lookup_udim_bilinear(sample_udim_textures, uv, DEFAULT_SPD)
    assert jnp.allclose(calculated_spd, expected_spd, atol=1e-6)

# TODO: Add tests for edge cases, different boundary modes if needed. 