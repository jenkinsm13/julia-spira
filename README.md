# Julia SPIRA - Metal GPU Raytracer

A high-performance ray tracing renderer implemented in Julia with Metal GPU acceleration for macOS.

## Features

- **Metal GPU Acceleration**: Leverages Apple's Metal framework for high-performance ray tracing
- **Julia Implementation**: Written in Julia for scientific computing and GPU programming
- **Multiple Renderers**: Includes various implementations from simple to optimized
- **Spectral Rendering**: Support for spectral rendering with SPD libraries
- **Cross-Platform**: Python and Julia implementations available

## Project Structure

```
julia-spira/
├── src/                    # Main source code
│   ├── SPIRA.jl           # Main module
│   ├── spira-metal-optimized.jl  # Optimized Metal renderer
│   └── spira_path_trace_kernel.metal  # Metal shader
├── examples/               # Example implementations
│   ├── basic_render.jl    # Simple usage example
│   ├── spira-metal-*.jl   # Various Metal implementations
│   └── julia-raytracer*.jl # CPU-based implementations
├── tests/                  # Test files
├── docs/                   # Documentation
├── assets/                 # Assets (images, 3D models)
│   └── images/            # Rendered images
├── translate-from-jax-python/  # Python implementations
└── Project.toml           # Julia package configuration
```

## Requirements

- macOS with Metal support
- Julia 1.8+
- Metal.jl package

## Quick Start

### As a Julia Package

```julia
using Pkg
Pkg.add("https://github.com/jenkinsm13/julia-spira.git")
using SPIRA

# Create scene and camera
scene = create_scene()
camera = Camera(Point3(0.0f0, 1.0f0, 3.0f0), 
                Point3(0.0f0, 0.0f0, 0.0f0), 
                Point3(0.0f0, 1.0f0, 0.0f0), 
                40.0f0, 1.0f0)

# Render
img = render(scene, camera, 640, 360, 
            samples_per_pixel=16, 
            max_depth=4)
```

### From Source

```bash
git clone https://github.com/jenkinsm13/julia-spira.git
cd julia-spira
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia examples/basic_render.jl
```

## Examples

- `examples/basic_render.jl` - Simple usage example
- `examples/spira-metal-minimal.jl` - Minimal Metal implementation
- `examples/spira-metal-optimized.jl` - Optimized Metal renderer
- `examples/spira-metal-raytracer.jl` - Full-featured raytracer

## Performance

The Metal GPU implementation provides significant speedup over CPU rendering:
- **GPU**: ~10-50x faster than CPU for complex scenes
- **Memory efficient**: Uses Metal.jl's optimized array operations
- **Scalable**: Performance scales with GPU capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [Metal.jl](https://github.com/JuliaGPU/Metal.jl)
- Inspired by Peter Shirley's "Ray Tracing in One Weekend"
- Python translation work from JAX-based implementations
