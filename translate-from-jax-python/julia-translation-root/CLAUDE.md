# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains a Julia translation of a differentiable spectral renderer originally implemented in Python/JAX. The renderer is capable of both forward rendering (generating images from a scene) and inverse rendering (parameter optimization through automatic differentiation).

## Key Features

- Forward rendering of 3D scenes (meshes, spheres)
- Inverse rendering for parameter optimization using Enzyme for automatic differentiation
- Spectral rendering with physically-based light transport
- Path tracing with importance sampling
- Support for mesh loading, various materials, and light sources
- HDR image output with proper tone mapping

## Code Structure

The codebase is organized into the following components:

1. **Entry Points**:
   - `spectral_renderer.jl` - Main entry point with command-line interface
   - `forward_render.jl` - Simplified script for forward rendering
   - `forward_render_array.jl` - Renders scenes with multiple objects

2. **Core Module** (`src/SpectralRenderer.jl`):
   - Exports all necessary types and functions
   - Provides high-level rendering functions

3. **Core Rendering Components** (in `src/`):
   - `types.jl` - Core data structures (rays, spectra, hit records, etc.)
   - `camera.jl` - Camera model with physical parameters
   - `geometry.jl` - Shape primitives and intersection routines
   - `integrator.jl` - Light transport algorithms
   - `utils.jl` - Helper functions for spectral conversion, math utilities
   - `spd_library.jl` - Spectral Power Distribution library for materials/lights

## Common Commands

### Running Forward Rendering

Basic rendering with default settings:
```bash
julia spectral_renderer.jl
```

With custom settings:
```bash
julia spectral_renderer.jl --width 512 --height 512 --spp 64 --max-depth 5 --obj-path path/to/mesh.obj
```

### Running Inverse Rendering

Basic parameter optimization using a target EXR image:
```bash
julia spectral_renderer.jl --mode inverse --target-exr target.exr --obj-path path/to/mesh.obj --steps 50
```

### Using Individual Scripts

For forward rendering only:
```bash
julia forward_render.jl --width 512 --height 512 --spp 64
```

For rendering scenes with multiple objects:
```bash
julia forward_render_array.jl --width 512 --height 512 --spp 16
```

## Project Dependencies

The project relies on these Julia packages:
- `LinearAlgebra`: Vector math operations
- `StaticArrays`: Performance-optimized vector operations
- `Images` and `Colors`: Image manipulation and color space conversions
- `FileIO`: File I/O operations
- `GeometryBasics` and `MeshIO`: OBJ file loading
- `Enzyme`: Automatic differentiation for inverse rendering
- `ArgParse`: Command-line argument parsing
- `OpenEXR`: EXR file format support

## Julia-Specific Guidelines

1. **Float32 Syntax**: Always use the longer `Float32(0.0)` syntax instead of the shorter `0f0` syntax.

2. **StaticArrays Usage**: Use `SVector` for small fixed-size vectors throughout the codebase for performance.

3. **Thread Safety**: Avoid global mutable state for potential future multi-threading support.

4. **Type Stability**: Maintain type stability in performance-critical functions.

## Extending the Renderer

1. To add new materials:
   - Add new spectral profiles in `spd_library.jl`
   - Update material constructors or BSDF functions in `integrator.jl`

2. To add new geometry primitives:
   - Implement the required `hit` and `bounding_box` methods in `geometry.jl`
   - Follow the `Hittable` interface

3. To modify render settings:
   - Adjust camera parameters in render calls
   - Modify lighting setup in the renderer scripts