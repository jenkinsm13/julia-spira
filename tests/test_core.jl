using SPIRA
using Test
using StaticArrays

@testset "Core Components" begin

    @testset "Ray" begin
        origin = Point3(0, 0, 0)
        direction = Vec3(1, 0, 0)
        ray = Ray(origin, direction)
        @test ray.origin == origin
        @test ray.direction == normalize(direction)
        @test at(ray, 2.0) == Point3(2, 0, 0)
    end

    @testset "Sphere" begin
        center = Point3(0, 0, 0)
        radius = 1.0f0
        material = 1
        sphere = Sphere(center, radius, material)
        @test sphere.center == center
        @test sphere.radius == radius
        @test sphere.material == material
    end

    @testset "Material" begin
        albedo = Color(0.8, 0.8, 0.8)
        emission = Color(1.0, 0.0, 0.0)
        metallic = 1.0f0
        roughness = 0.1f0
        material = Material(albedo, emission=emission, metallic=metallic, roughness=roughness)
        @test material.albedo == albedo
        @test material.emission == emission
        @test material.metallic == metallic
        @test material.roughness == roughness
    end

    @testset "Camera" begin
        lookfrom = Point3(0, 0, 1)
        lookat = Point3(0, 0, 0)
        vup = Vec3(0, 1, 0)
        vfov = 90.0
        aspect_ratio = 16.0f0 / 9.0f0
        cam = Camera(lookfrom, lookat, vup, vfov, aspect_ratio)
        @test cam.origin == lookfrom
        @test cam.lower_left_corner ≈ Point3(-1.7777, -1.0, 0.0) atol=1e-4
    end

    @testset "Scene" begin
        scene = create_scene()
        @test length(scene.spheres) == 5
        @test length(scene.materials) == 5
    end

end
