#!/usr/bin/env julia

# Ultra minimal Metal test script
# Just creates a simple gradient directly on GPU

using Images
using FileIO
using Metal

println("Testing basic Metal functionality...")

try
    # Check if Metal is available
    dev = Metal.device()
    println("Metal device: $dev")
    
    # Set dimensions - make them small for testing
    width = 200
    height = 100
    
    # Create a simple array directly on GPU
    # First on CPU, then transfer to GPU
    cpu_arr = zeros(Float32, height, width)
    for j in 1:height
        for i in 1:width
            cpu_arr[j, i] = Float32(i) / Float32(width)
        end
    end
    
    # Transfer to GPU
    gpu_arr = MtlArray(cpu_arr)
    println("Created Metal array of size $height x $width")
    
    # Do a simple GPU operation (multiply by 2)
    result = 2.0f0 .* gpu_arr
    
    # Copy back to CPU
    cpu_result = Array(result)
    
    # Create RGB image
    img = Array{RGB{Float32}}(undef, height, width)
    for j in 1:height
        for i in 1:width
            # Red channel gradient
            img[j, i] = RGB{Float32}(cpu_result[j, i], 0.0f0, 0.0f0)
        end
    end
    
    # Save the image to verify everything works
    save("minimal_metal_render.png", img)
    println("Metal test completed successfully! Check minimal_metal_render.png")
    
catch e
    println("Metal test failed: $e")
    println("Backtrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end