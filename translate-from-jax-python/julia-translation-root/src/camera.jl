# Camera implementation for spectral renderer

using LinearAlgebra
using StaticArrays
using Random

"""
Camera with physical parameters for realistic rendering.
Supports distortion, depth of field, and physical camera parameters.
"""
struct Camera
    # Intrinsics
    fx::Float32  # Focal length in x (pixels)
    fy::Float32  # Focal length in y (pixels)
    cx::Float32  # Principal point x (pixels)
    cy::Float32  # Principal point y (pixels)
    
    # Extrinsics (Position & Orientation)
    eye::Point3f      # Camera position
    center::Point3f   # Look-at point
    up::Vec3f         # Up vector
    
    # Camera basis vectors (precomputed)
    u::Vec3f   # Right vector
    v::Vec3f   # Up vector
    w::Vec3f   # Forward vector (negated)
    
    # Distortion Coefficients (Brown-Conrady)
    k1::Float32  # Radial distortion coeff 1
    k2::Float32  # Radial distortion coeff 2
    k3::Float32  # Radial distortion coeff 3
    p1::Float32  # Tangential distortion coeff 1
    p2::Float32  # Tangential distortion coeff 2
    
    # Physical Exposure Parameters
    f_number::Float32       # Aperture f-stop
    shutter_speed::Float32  # Exposure time in seconds
    iso::Float32            # Sensor sensitivity
    lens_radius::Float32    # Derived from f_number
    
    # Depth of Field Parameters
    focus_distance::Float32 # Distance to the plane in perfect focus
    
    # Constructor
    function Camera(;
        # Intrinsics
        fx::Float32 = Float32(1000.0),
        fy::Float32 = Float32(1000.0),
        cx::Float32 = Float32(128.0),
        cy::Float32 = Float32(128.0),
        
        # Extrinsics
        eye::Point3f = Point3f(Float32(-2.0), Float32(2.5), Float32(4.0)),
        center::Point3f = Point3f(Float32(0.0), Float32(1.0), Float32(0.0)),
        up::Vec3f = Vec3f(Float32(0.0), Float32(1.0), Float32(0.0)),
        
        # Distortion
        k1::Float32 = Float32(0.0),
        k2::Float32 = Float32(0.0),
        k3::Float32 = Float32(0.0),
        p1::Float32 = Float32(0.0),
        p2::Float32 = Float32(0.0),
        
        # Physical parameters
        f_number::Float32 = Float32(16.0),
        shutter_speed::Float32 = Float32(1/100.0),
        iso::Float32 = Float32(100.0),

        # Focus distance (optional, defaults to distance to look-at point)
        focus_distance::Union{Float32, Nothing} = nothing
    )
        # Compute camera basis vectors
        _w = normalize(eye - center)  # Forward direction (negated)
        _u = normalize(cross(up, _w))  # Right direction
        _v = cross(_w, _u)              # Up direction
        _up_normalized = normalize(up)
        
        # --- Unified & Physically Correct Depth of Field Calculation ---
        pixel_size_mm = Float32(0.00435)  # Assuming fixed pixel size for now
        focal_length_mm = fx * pixel_size_mm 
        _lens_radius = focal_length_mm / (Float32(2.0) * f_number)

        # Determine focus distance
        _focus_distance = if focus_distance === nothing
            norm(eye - center) # Default to look-at point distance
        else
            Float32(focus_distance)
        end
        # --- End Unified & Correct DOF Calculation ---
        
        new(fx, fy, cx, cy, eye, center, _up_normalized,
            _u, _v, _w, k1, k2, k3, p1, p2,
            f_number, shutter_speed, iso, _lens_radius, _focus_distance)
    end
end

"""
    apply_brown_distortion(xp, yp, k1, k2, k3, p1, p2) -> (xd, yd)

Apply Brown-Conrady distortion model to normalized image coordinates.
Takes undistorted normalized coordinates (xp, yp) and returns distorted coordinates (xd, yd).
"""
function apply_brown_distortion(xp::Float32, yp::Float32, k1::Float32, k2::Float32, k3::Float32, p1::Float32, p2::Float32)
    r2 = xp^2 + yp^2
    r4 = r2^2
    r6 = r2 * r4
    
    # Radial distortion
    radial_factor = Float32(1.0) + k1 * r2 + k2 * r4 + k3 * r6
    
    xd_radial = xp * radial_factor
    yd_radial = yp * radial_factor
    
    # Tangential distortion
    dx_tangential = Float32(2.0) * p1 * xp * yp + p2 * (r2 + Float32(2.0) * xp^2)
    dy_tangential = p1 * (r2 + Float32(2.0) * yp^2) + Float32(2.0) * p2 * xp * yp
    
    # Combined distortion
    xd = xd_radial + dx_tangential
    yd = yd_radial + dy_tangential
    
    return xd, yd
end

"""
    generate_ray(camera::Camera, s::Float32, t::Float32, rng::AbstractRNG=Random.GLOBAL_RNG)

Generate a ray from the camera through the pixel at normalized coordinates (s, t).
Handles distortion and depth of field using the camera's focus_distance.
"""
function generate_ray(camera::Camera, s::Float32, t::Float32, rng::AbstractRNG=Random.GLOBAL_RNG)
    # Convert normalized image coordinates [0,1] to pixel coordinates
    x = s * camera.cx * Float32(2.0)
    y = t * camera.cy * Float32(2.0)
    
    # Shift to center origin and normalize by focal length
    x_p = (x - camera.cx) / camera.fx
    y_p = (y - camera.cy) / camera.fy
    
    # Apply distortion
    x_d, y_d = apply_brown_distortion(x_p, y_p, camera.k1, camera.k2, camera.k3, camera.p1, camera.p2)
    
    # Calculate pinhole ray direction in camera space and transform to world space
    ray_dir_cam = Vec3f(x_d, y_d, Float32(-1.0))
    ray_dir_cam_norm = normalize(ray_dir_cam)
    cam_to_world = hcat(camera.u, camera.v, camera.w) # Note: Check basis vector calculation if issues arise
    pinhole_ray_dir_world = cam_to_world * ray_dir_cam_norm
    
    # Default origin and direction (pinhole)
    origin = camera.eye
    ray_dir_world = pinhole_ray_dir_world

    # Apply depth of field if lens radius > 0
    if camera.lens_radius > Float32(1e-6) # Use a small threshold
        # Sample a point on the lens using the camera's basis vectors u, v
        rd = camera.lens_radius * random_in_unit_disk(rng)
        offset = camera.u * rd[1] + camera.v * rd[2]
        origin = camera.eye + offset # Ray starts from the offset point on the lens
        
        # Calculate the point on the focal plane this pinhole ray intersects
        focus_point = camera.eye + pinhole_ray_dir_world * camera.focus_distance # Use camera.focus_distance
        
        # The final ray direction goes from the point on the lens towards the focus point
        ray_dir_world = normalize(focus_point - origin)
    end
    
    return Ray(origin, ray_dir_world)
end

"""
    generate_rays(camera::Camera, width::Int, height::Int, rng::AbstractRNG=Random.GLOBAL_RNG)

Generate rays for a grid of pixels of size (width, height).
Returns a 2D array of rays.
"""
function generate_rays(camera::Camera, width::Int, height::Int, rng::AbstractRNG=Random.GLOBAL_RNG)
    rays = Array{Ray}(undef, height, width)
    
    for j in 1:height
        for i in 1:width
            # Add random jitter for anti-aliasing
            s = (i - 1 + rand(rng, Float32)) / (width - 1)
            t = (j - 1 + rand(rng, Float32)) / (height - 1)
            rays[j, i] = generate_ray(camera, s, t, rng)
        end
    end
    
    return rays
end

"""
    random_in_unit_disk(rng::AbstractRNG=Random.GLOBAL_RNG)

Generate a random point in the unit disk for lens sampling.
"""
function random_in_unit_disk(rng::AbstractRNG=Random.GLOBAL_RNG)
    while true
        p = Float32(2.0) * Vec3f(rand(rng, Float32), rand(rng, Float32), Float32(0.0)) - Vec3f(Float32(1.0), Float32(1.0), Float32(1.0))
        if p[1]^2 + p[2]^2 < Float32(1.0)
            return p
        end
    end
end