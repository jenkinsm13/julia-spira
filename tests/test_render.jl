using SPIRA
using Test
using Images
using StaticArrays

@testset "Rendering" begin

    @testset "CPU Rendering" begin
        width = 10
        height = 10
        samples_per_pixel = 1
        max_depth = 1

        scene = create_scene()
        aspect_ratio = Float32(width / height)
        lookfrom = Point3(Float32(0.0), Float32(1.0), Float32(3.0))
        lookat = Point3(Float32(0.0), Float32(0.0), Float32(0.0))
        vup = Point3(Float32(0.0), Float32(1.0), Float32(0.0))
        camera = Camera(lookfrom, lookat, vup, Float32(40.0), aspect_ratio)

        img = render_with_cpu(width, height, scene, camera, samples_per_pixel=samples_per_pixel, max_depth=max_depth)
        @test size(img) == (height, width)
    end

    @testset "GPU Rendering" begin
        # This is a placeholder test. A full GPU test would require a GPU environment.
        # We will just check if the function exists.
        @test isdefined(SPIRA, :render_hybrid_gpu)
    end

end
