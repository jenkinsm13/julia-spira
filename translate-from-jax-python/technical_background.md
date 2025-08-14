# Technical Background: JAX Renderer based on Unified Sphere Framework

**Version:** 0.1
**Date:** 2024-08-01
**Author:** AI Assistant (based on user specification)

## 1. Introduction

This document outlines the technical background and theory of operation for a differentiable rendering system being developed in JAX. The primary goal is to implement a renderer capable of simulating light transport according to a novel, fundamental physical framework proposed by the user.

The user's framework posits that reality can be understood through mechanisms involving nested spheres, chronon flow, mass-induced distortions, and emergent particle excitations. This renderer aims to directly simulate these principles.

The core implementation strategy leverages JAX's strengths in automatic differentiation (`jax.grad`) and accelerated computation (`jax.jit`, `jax.vmap`, `jax.lax.scan`). It achieves this by modeling all scene interactions at a fundamental level as operations on sphere primitives, whose properties are derived directly from the state variables of the user's physical framework.

This document serves as a guide for AI agents and developers collaborating on this project, ensuring consistency and understanding of the underlying principles and implementation choices.

## 2. Conceptual Framework Overview (User-Defined Physics - TBD)

The rendering logic will be driven by a specific physical framework provided by the user. Key concepts mentioned include:

*   **Nested Spheres:** A fundamental structure underlying space/reality.
*   **Chronons:** Quantized units of time, potentially flowing between spheres.
*   **Distortion:** Mass or other factors causing distortions in the spherical structure.
*   **Particle Excitations:** Particles emerging from interactions or excitations within this structure.

**Rendering Relevance (Hypothesized - Requires User Math):**

The interaction of light with matter within this framework needs to be mathematically defined. This definition will replace traditional BSDFs/BSSRDFs. We anticipate the framework's mathematics will dictate:

*   **Absorption:** How much light energy is lost upon interaction, potentially related to chronon density or specific excitation states.
*   **Scattering:** The probability distribution of outgoing light directions based on incoming direction, sphere properties (derived from local distortion?), and potentially chronon flow dynamics.
*   **Emission:** Whether interactions or specific framework states lead to light emission.
*   **Refraction/Transmission:** How light propagates *through* the conceptual spheres or between nested levels.

**Note:** This section is currently high-level. **The specific mathematical formulae derived from the user's framework are required** to implement the core light interaction logic.

## 3. Core Implementation Strategy: Sphere-Based Rendering

Based on the user's framework and the capabilities of JAX, the chosen strategy is to represent all points of interaction within the scene as sphere primitives.

*   **Rationale:** This aligns with the framework's apparent spherical motifs and allows leveraging highly optimized array processing in JAX.
*   **Scene Elements:**
    *   **Point Clouds:** Directly map points to spheres (center, radius, framework parameters).
    *   **Surfaces (Meshes, etc.):** Conceptually, intersections with surfaces are treated as interactions with a local sphere primitive whose properties (radius, distortion parameters, etc.) are determined by the surface properties and the framework's state at that location.
*   **Implementation:** The scene geometry is represented not as a list of objects, but as stacked JAX arrays containing the parameters for all sphere primitives.

## 4. JAX Implementation Details

The `minimal_jax_renderer/` directory provides the foundational codebase.

*   **Scene Representation (Arrays):**
    *   Geometry is defined by JAX arrays: `sphere_centers` (N, 3), `sphere_radii` (N,), `sphere_material_ids` (N,).
    *   "Material" properties (currently just `material_albedos`) are also stored in arrays, indexed by `sphere_material_ids`. **This `material_albedos` array will be replaced or augmented by arrays holding the parameters derived from the user's framework.**
    *   See `minimal_jax_renderer/forward_render.py` and `inverse_render.py` for how these arrays are constructed and passed.

*   **Intersection (`minimal_jax_renderer/src/geometry.py`):**
    *   The `intersect_scene_geometry` function calculates the closest ray-sphere intersection.
    *   It takes the geometry arrays and a `Ray` as input.
    *   It uses `jax.lax.scan` to efficiently iterate over the sphere arrays, making it JIT-compilable and scalable.
    *   It returns a `HitRecord` containing intersection details (t, position, normal, material_id).

*   **Integration (`minimal_jax_renderer/src/integrator.py`):**
    *   The `render_pixel` function orchestrates the rendering of a single pixel ray.
    *   It calls `intersect_scene_geometry` to find hits.
    *   **Crucially, the light interaction logic (currently placeholder Lambertian shading within `get_color`) must be replaced.** This replacement function will compute the outgoing radiance based on the hit information and the framework-derived parameters associated with the hit sphere.
    *   The `render_image` function uses `jax.vmap` to parallelize `render_pixel` across all image pixels and is JIT-compiled for performance.

*   **Differentiation (`minimal_jax_renderer/inverse_render.py`):**
    *   Leverages `jax.value_and_grad` to automatically compute gradients of the rendering process with respect to specified input parameters (e.g., the framework parameters associated with the spheres).
    *   Enables inverse rendering tasks like parameter estimation using optimizers like `optax`. The current example optimizes sphere albedo.

## 5. Bridging Framework and Implementation

The central development task is to implement the mathematical light interaction model derived from the user's framework within the JAX integrator.

*   **Parameter Mapping:**
    *   **Input:** User defines the state variables of their framework at any given point/sphere (e.g., `distortion_tensor`, `chronon_density`, `excitation_level`).
    *   **Storage:** These variables need to be stored in JAX arrays, indexed by the `sphere_material_ids` (similar to how `material_albedos` is currently used). The `SceneData` structure in `src/scene.py` will need modification.
    *   **Access:** The `render_pixel` function (specifically, the light interaction part) will fetch these parameters based on the `hit.material_id`.

*   **Light Interaction Function (LIF):**
    *   A new JIT-compatible function needs to be created, e.g., `calculate_scattering(...)`.
    *   **Required Inputs:**
        *   `incoming_direction`: Vec3 (normalized)
        *   `hit_position`: Point3
        *   `hit_normal`: Vec3 (normalized)
        *   `framework_params`: Pytree or dictionary containing the framework state variables for the hit sphere (e.g., fetched from the parameter arrays).
        *   `rng_key`: For any stochastic sampling.
    *   **Required Outputs (Path Tracing Style):**
        *   `outgoing_direction`: Vec3 (sampled based on framework physics)
        *   `bsdf_value`: Color3 or scalar representing throughput modification (analogous to `BSDF * cos(theta) / pdf`).
        *   `pdf`: Probability density function value for the sampled `outgoing_direction`.
        *   `(Alternative)` If not path tracing, it might directly compute the color contribution from a light source.
    *   **Implementation:** This function will contain the core JAX implementation of the user's framework mathematics related to light interaction.

*   **Potential Complexity:**
    *   **Inter-Sphere Interactions:** If the framework dictates that a sphere's state or light interaction depends on neighboring spheres, the integrator will need modification. This could involve neighbor lookups (potentially complex in JAX) or multi-pass rendering techniques. Start with the assumption of independent sphere interactions unless specified otherwise.
    *   **Volumetric Effects:** If the framework involves interaction *within* the sphere volume or between nested spheres, volumetric rendering techniques adapted to the framework's math will be needed.

## 6. Development Workflow

1.  **Current State:** Minimal JAX renderer performing efficient sphere intersection and placeholder direct lighting. Differentiable w.r.t. material parameters.
2.  **User Input:** Obtain the specific mathematical formulae defining light interaction (absorption, scattering, emission) from the user's framework.
3.  **Parameter Definition:** Define the structure (e.g., update `SceneData`) to hold the necessary framework parameters in JAX arrays.
4.  **Implement LIF:** Code the `calculate_scattering` (or equivalent) function in `integrator.py` using JAX based on the provided math.
5.  **Integrator Refinement:** Update `render_pixel` to use the new LIF. Adapt for path tracing (recursive calls or scan loop) or other integration techniques as appropriate.
6.  **Testing & Validation:** Create test scenes and validation metrics. Utilize `inverse_render.py` to test differentiability and parameter recovery.
7.  **Extension:** Incorporate advanced features (inter-sphere interactions, volumetrics) if required by the framework.

## 7. Conclusion

This project aims to create a unique, differentiable renderer in JAX, grounded in a novel physical framework. By representing scene interactions as sphere primitives and leveraging JAX's array processing and autodiff, we can efficiently simulate the framework's principles. The immediate next step requires the mathematical specification of the framework's light interaction model to replace the placeholder logic in the current integrator.