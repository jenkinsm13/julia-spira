# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains a minimal differentiable renderer implemented in JAX. The renderer is capable of both forward rendering (generating images from a scene) and inverse rendering (parameter estimation).

### Key Features

- Forward rendering of 3D scenes (meshes, spheres, planes)
- Inverse rendering for parameter optimization
- Spectral rendering with physically-based light transport
- Path tracing with importance sampling
- Differentiable workflows using JAX's automatic differentiation
- Support for mesh loading, texturing, and various light sources

## Code Architecture

The codebase is organized into the following components:

1. **Entry Points**:
   - `forward_render.py` - Generates images from a scene
   - `forward_render_array.py` - Renders scenes with multiple objects arranged in an array
   - `inverse_render.py` - Parameter optimization through gradient descent

2. **Core Rendering Components** (in `src/`):
   - `types.py` - Core data structures (rays, spectra, hit records, etc.)
   - `camera.py` - Camera model with physical parameters
   - `geometry.py` - Shape primitives and intersection routines
   - `integrator.py` - Light transport algorithms
   - `utils.py` - Helper functions for spectral conversion, math utilities
   - `spd_library.py` - Spectral Power Distribution library for materials/lights
   - `scene.py` - Scene definition and management

## Development Environment

### Requirements

This project requires Python and specific dependencies listed in `requirements.txt`:

```
jax
jaxlib
numpy
pillow
optax
tqdm
perlin-noise
pywavefront
```

### Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Common Commands

### Running Forward Rendering

Basic rendering with default settings:
```bash
python forward_render.py
```

With custom settings:
```bash
python forward_render.py --width 512 --height 512 --spp 64 --max-depth 5 --obj-path path/to/mesh.obj
```

Rendering scenes with multiple objects:
```bash
python forward_render_array.py --width 512 --height 512 --spp 16 --sphere-obj-path test_sphere.obj
```

### Running Inverse Rendering

Basic parameter optimization:
```bash
python inverse_render.py --obj-path test_sphere.obj --steps 50
```

With a target EXR image:
```bash
python inverse_render.py --target-exr path/to/target.exr --obj-path test_sphere.obj --steps 50
```

### Running Tests

Run all tests:
```bash
python -m pytest tests/
```

Run specific test file:
```bash
python -m pytest tests/test_utils.py
```

## Spectral Rendering

The renderer uses a spectral rendering approach for physically-based light simulation:

- Wavelengths are uniformly sampled between 400nm and 700nm
- The number of wavelength samples is configurable (default: 31)
- Spectral Power Distributions (SPDs) define material reflectance, emission properties
- CIE color matching functions convert spectral data to XYZ color space
- Final images are saved in both sRGB (PNG) and ACEScg (EXR) color spaces

## Physics Simulation

The renderer implements physically-based light transport including:

- Direct illumination from multiple light types
- Global illumination via path tracing
- Russian Roulette path termination
- Multiple Importance Sampling for combining BRDF and light sampling
- Support for emissive materials
- Physically-based camera exposure model

## Extending the Renderer

When adding new features or materials:

1. For new materials or light sources, extend the appropriate classes in `types.py`
2. For new spectral profiles, add them to `spd_library.py`
3. For new rendering algorithms, modify or extend the functions in `integrator.py`
4. For new shape primitives, add intersection functions in `geometry.py`