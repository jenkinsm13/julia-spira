# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains a Julia-based path tracer/ray tracer implementation called SPIRA (Spatially Partitioned, Iterative, Ray-traced Algorithm). The codebase implements a complete 3D rendering system capable of generating photo-realistic images with advanced features including mesh rendering, multi-threading, spatial partitioning, and automatic differentiation support.

## Available Versions

- **julia-raytracer.jl**: Original basic implementation with sphere primitives
- **julia-raytracer-optimized.jl**: Version with mesh loading and simple BVH acceleration
- **julia-raytracer-optimized2.jl**: Advanced version with octree and SAH-optimized BVH
- **julia-raytracer-threaded.jl**: Fully multi-threaded implementation with Enzyme differentiability support
- **julia-raytracer-gpu.jl**: GPU-accelerated implementation with cross-platform support (Metal/AMDGPU/CUDA)

## Installation & Setup

Before running certain versions of the raytracer, you may need to install dependencies:

```bash
# Install GUI dependencies (required for spira-gui.jl)
julia install-gui-deps.jl

# Install GPU dependencies (required for GPU acceleration)
julia install-gpu-deps.jl
```

## Running the Code

```bash
# Run original ray tracer
julia julia-raytracer.jl

# Run optimized mesh version
julia julia-raytracer-optimized.jl

# Run with CPU threading (quick preview)
julia render-threaded.jl

# Run with GPU acceleration (Metal on macOS, AMDGPU or CUDA elsewhere)
julia render-gpu.jl

# Run with simplified GPU/CPU hybrid renderer (more reliable)
julia render-gpu-simple.jl

# macOS only: Metal implementations
julia working-metal.jl       # Self-testing Metal compatibility script
julia metal-raytracer.jl     # Metal raytracer (if Metal.jl works)

# Guaranteed to work everywhere
julia clean-raytracer.jl     # Clean CPU raytracer using Float64
julia fast-raytracer.jl      # Optimized Float32 raytracer (better performance, GPU compatible)

# Run the GUI interface
julia spira-gui.jl

# Run the simplified GUI (better compatibility)
julia spira-gui-simple.jl
```

## Code Structure

The ray tracer follows an advanced path tracing architecture:

1. **Vector Math**: Optimized 3D vector operations using StaticArrays
2. **Ray Tracing**: Definition of rays and fast intersection calculations
3. **Materials**: Material properties including metallic workflow and emission
4. **Geometry**: Supports mesh geometry with triangle primitives
5. **Acceleration Structures**:
   - BVH (Bounding Volume Hierarchy) with Surface Area Heuristic
   - Octree spatial partitioning
6. **Camera**: View definition and ray generation with depth of field
7. **Rendering**: Multi-threaded Monte Carlo path tracing
8. **Image Output**: Advanced ACEScg tone mapping for better color reproduction
9. **Differentiability**: Compatible with Enzyme automatic differentiation

## Dependencies

The code requires these Julia packages:

- LinearAlgebra: For vector operations
- Random: For stochastic sampling
- Images: For image manipulation
- StaticArrays: For performance-optimized vector math
- FileIO: For file I/O
- GeometryBasics and MeshIO: For OBJ file loading
- Base.Threads: For multi-threading
- Enzyme: For automatic differentiation (inverse rendering)

### GPU Dependencies (Optional)

For GPU acceleration, at least one of the following is required:

- Metal.jl: For macOS GPU acceleration via Metal API
- AMDGPU.jl: For AMD GPU acceleration
- CUDA.jl: For NVIDIA GPU acceleration

## Common Tasks

### Modifying the Scene

To modify the rendered scene:
- Load different OBJ files using the `load_mesh` function
- Apply transformations (scale, rotation, translation) to positioned meshes
- Create custom materials with the `Material` constructor
- Use the GUI to interactively load and position meshes

### Using the GUI

The SPIRA GUI provides a complete interface for working with the raytracer:

1. **Scene Management**
   - Add/remove meshes from OBJ files
   - Transform meshes (scale, rotate, translate) interactively
   - Edit materials with real-time color preview

2. **Camera Controls**
   - Position the camera and look-at point
   - Adjust field of view
   - See real-time updates in the preview

3. **Render Settings**
   - Choose resolution presets (360p to 4K)
   - Set sample count for quality control
   - Adjust ray bounce depth for realism

4. **Output Options**
   - Render to PNG with tone mapping
   - Render to EXR (HDR) for professional workflows

### Performance Optimization

Performance optimizations include:
- Multi-threaded rendering (automatically uses all available threads)
- BVH with Surface Area Heuristic for optimal traversal
- Octree spatial partitioning for large scenes
- StaticArrays for cache-friendly vector operations
- Fast AABB-ray intersection tests with precomputed inverse direction

### Extending the Renderer

To add new features:
1. Add new primitive types by implementing appropriate `hit` functions
2. Implement additional material models in the `scatter` function
3. Extend differentiable rendering by expanding the `∇render_enzyme` function

## Debugging Tips

- Use `render-threaded.jl` for faster preview renders
- Adjust the `width`, `height`, and `samples_per_pixel` parameters for quality/speed tradeoffs
- Enable `progress_update=true` in render calls to see rendering progress

## Metal.jl GPU Acceleration Memories

### Array Abstraction
The easiest way to work with Metal.jl is by using its array abstraction. The MtlArray type is both meant to be a convenient container for device memory, as well as provide a data-parallel abstraction for using the GPU without writing your own kernels:

```julia
julia> a = MtlArray([1])
1-element MtlArray{Int64, 1}:
 1

julia> a .+ 1
1-element MtlArray{Int64, 1}:
 2
```

### Kernel Programming
These array abstractions are implemented using Metal kernels written in Julia. These kernels follow a similar programming style to Julia's other GPU back-ends, and deviate from Metal C kernel conventions:

```julia
julia> function vadd(a, b, c)
           i = thread_position_in_grid_1d()
           c[i] = a[i] + b[i]
           return
       end
vadd (generic function with 1 method)

julia> a = MtlArray([1,1,1,1]); b = MtlArray([2,2,2,2]); c = similar(a);

julia> @metal threads=2 groups=2 vadd(a, b, c)

julia> Array(c)
4-element Vector{Int64}:
 3
 3
 3
 3
```

### Metal API Wrapper
The functionality is made possible by interfacing with Metal libraries through ObjectiveC.jl. Low-level objects and functions are provided in the MTL submodule:

```julia
julia> dev = Metal.MTL.devices()[1]
<AGXG13XDevice: 0x14c17f200>
    name = Apple M1 Pro

julia> dev.name
NSString("Apple M1 Pro")
```