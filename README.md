# Julia SPIRA - Metal GPU Raytracer

A high-performance ray tracing renderer implemented in Julia with Metal GPU acceleration for macOS.

## Features

- **Metal GPU Acceleration**: Leverages Apple's Metal framework for high-performance ray tracing
- **Julia Implementation**: Written in Julia for scientific computing and GPU programming
- **Multiple Renderers**: Includes various implementations from simple to optimized
- **Spectral Rendering**: Support for spectral rendering with SPD libraries
- **Cross-Platform**: Python and Julia implementations available

## Project Structure

- `spira-metal-*.jl` - Metal GPU implementations
- `julia-raytracer*.jl` - CPU-based Julia implementations
- `translate-from-jax-python/` - Python implementations and translation work
- `*.metal` - Metal shader files for GPU kernels

## Requirements

- macOS with Metal support
- Julia 1.8+
- Metal.jl package

## Quick Start

```julia
using Metal
include("spira-metal-optimized.jl")
# Run your render
```

## Examples

- `spira-metal-minimal.jl` - Minimal Metal implementation
- `spira-metal-optimized.jl` - Optimized Metal renderer
- `spira-metal-raytracer.jl` - Full-featured raytracer

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
