import jax
import jax.numpy as jnp
from jax import jit
from .types import Spectrum, N_WAVELENGTHS, WAVELENGTHS_NM # Assuming types.py is in the same directory or accessible


# --- Material Definition Helper ---
def create_spectrum_profile(profile_type: str, value: float = None) -> Spectrum:
    """Helper to create simple spectral profiles."""

    if profile_type == "flat":
        # Flat spectrum across all wavelengths
        return jnp.full((N_WAVELENGTHS,), value if value is not None else 0.8)
    
    elif profile_type == "red":
        # Sigmoid transition: low reflectance at short WL, high at long WL
        # Shift center wavelength further into red and increase steepness for saturation
        center_wl = 630.0  # Transition wavelength (nm) - Moved from 590nm
        steepness = 0.1    # Controls sharpness of transition - Increased from 0.05
        min_reflectance = 0.02 # Optionally lower the minimum slightly
        max_reflectance = value if value is not None else 0.9
        
        # Sigmoid function: 1 / (1 + exp(-k * (x - x0)))
        scaled_wavelengths = (WAVELENGTHS_NM - center_wl) * steepness
        sigmoid_vals = 1.0 / (1.0 + jnp.exp(-scaled_wavelengths))
        
        spd = min_reflectance + (max_reflectance - min_reflectance) * sigmoid_vals
        return spd
    
    elif profile_type == "green":
         # Gaussian shape centered in the green region
         center_wl = 540.0 # Center wavelength (nm)
         sigma = 40.0      # Width of the peak (nm)
         peak_value = value if value is not None else 0.8
         trough_value = 0.05 # Base reflectance outside the peak
         
         spd = trough_value + (peak_value - trough_value) * jnp.exp(-0.5 * ((WAVELENGTHS_NM - center_wl) / sigma)**2)
         return spd
    
    elif profile_type == "blue":
        # Inverse Sigmoid transition: high reflectance at short WL, low at long WL
        center_wl = 490.0  # Transition wavelength (nm)
        steepness = 0.05 # Controls sharpness of transition
        min_reflectance = 0.05
        max_reflectance = value if value is not None else 0.8

        scaled_wavelengths = (WAVELENGTHS_NM - center_wl) * steepness
        sigmoid_vals = 1.0 / (1.0 + jnp.exp(-scaled_wavelengths))

        # Invert the sigmoid shape
        spd = min_reflectance + (max_reflectance - min_reflectance) * (1.0 - sigmoid_vals)
        return spd
    
    elif profile_type == "gold":
        # Metallic gold spectral profile with IR reflectance
        base_reflectance = value if value is not None else 0.92
        wl_peak = 700.0  # Strong reflectance in near-IR
        wl_width = 200.0
        spd = base_reflectance * jnp.exp(-0.5 * ((WAVELENGTHS_NM - wl_peak) / wl_width)**2)
        spd += 0.3 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 450.0) / 50.0)**2)  # Blue absorption
        return jnp.clip(spd, 0.0, 1.0)

    elif profile_type == "copper":
        # Copper metallic reflectance with characteristic orange-red response
        main_peak = 0.8 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 620.0) / 80.0)**2)
        secondary_peak = 0.4 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 450.0) / 30.0)**2)
        spd = (value if value is not None else 1.0) * (main_peak + secondary_peak)
        return jnp.clip(spd, 0.0, 0.95)

    elif profile_type == "silver":
        # Neutral metallic reflectance with enhanced blue response
        base = jnp.full_like(WAVELENGTHS_NM, 0.95 * (value if value else 1.0))
        blue_enhance = 0.1 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 450.0) / 40.0)**2)
        return jnp.clip(base + blue_enhance, 0.0, 1.0)
    
    elif profile_type == "skin_medium":
        # Physically-based human skin spectral profile for MEDIUM skin tone.
        # Based on combining dermal scattering, hemoglobin absorption, and melanin absorption.
        # 'value' scales the overall reflectance intensity (default 1.0).

        # --- Define Base Dermal/Epidermal Scattering Reflectance ---
        # Sigmoid curve representing general scattering increase towards red.
        base_center_wl = 550.0 # Wavelength around which the reflectance transitions
        base_steepness = 0.015 # Controls how quickly the reflectance rises
        min_reflectance = 0.15 # Reflectance at short wavelengths (blue/violet)
        max_reflectance = 0.65 # Reflectance at long wavelengths (red/NIR)

        scaled_wavelengths_base = (WAVELENGTHS_NM - base_center_wl) * base_steepness
        base_spd = min_reflectance + (max_reflectance - min_reflectance) * (1.0 / (1.0 + jnp.exp(-scaled_wavelengths_base)))

        # --- Define Hemoglobin Absorption Features (Gaussian dips) ---
        # Represents absorption by oxygenated hemoglobin in blood vessels.
        # Soret band (strong absorption peak in blue/violet)
        soret_peak = 415.0
        soret_width = 25.0 # Width of the absorption band
        soret_depth = 0.04 # Depth of the absorption dip (relative to base)

        # Q-bands (weaker absorption peaks in green/yellow)
        q1_peak = 542.0
        q1_width = 18.0
        q1_depth = 0.02

        q2_peak = 577.0
        q2_width = 18.0
        q2_depth = 0.025

        hb_absorption = (
            soret_depth * jnp.exp(-0.5 * ((WAVELENGTHS_NM - soret_peak) / soret_width)**2) +
            q1_depth * jnp.exp(-0.5 * ((WAVELENGTHS_NM - q1_peak) / q1_width)**2) +
            q2_depth * jnp.exp(-0.5 * ((WAVELENGTHS_NM - q2_peak) / q2_width)**2)
        )

        # --- Define Melanin Absorption Feature ---
        # Represents absorption by melanin pigment, stronger at shorter wavelengths.
        # Modeled as an inverse sigmoid (decreasing absorption with increasing wavelength).
        melanin_center_wl = 450.0 # Wavelength around which melanin absorption drops off
        melanin_steepness = 0.02  # Controls the sharpness of the drop-off
        melanin_max_absorption = 0.45 # Maximum absorption level (at short wavelengths) for medium skin

        scaled_wavelengths_melanin = (WAVELENGTHS_NM - melanin_center_wl) * melanin_steepness
        # Inverse sigmoid: 1 - sigmoid
        melanin_absorption_factor = 1.0 - (1.0 / (1.0 + jnp.exp(-scaled_wavelengths_melanin)))
        melanin_absorption = melanin_max_absorption * melanin_absorption_factor

        # --- Combine Components ---
        # Start with base scattering and subtract absorption features.
        spd = base_spd - hb_absorption - melanin_absorption

        # --- Apply Scaling Factor ---
        scale_factor = value if value is not None else 1.0
        spd = spd * scale_factor

        # --- Clip to Plausible Range ---
        # Ensure reflectance stays between 0 and 1 (or slightly lower max for realism).
        return jnp.clip(spd, 0.0, 0.95) # Clip max to avoid overly bright skin

    elif profile_type == "skin_dark":
        # Physically-based human skin spectral profile for DARK skin tone.
        # Similar structure to medium skin, but with increased melanin absorption.
        # 'value' scales the overall reflectance intensity (default 1.0).

        # --- Define Base Dermal/Epidermal Scattering Reflectance ---
        # Slightly lower overall reflectance compared to medium skin.
        base_center_wl = 560.0
        base_steepness = 0.016
        min_reflectance = 0.08 # Lower min reflectance
        max_reflectance = 0.55 # Lower max reflectance

        scaled_wavelengths_base = (WAVELENGTHS_NM - base_center_wl) * base_steepness
        base_spd = min_reflectance + (max_reflectance - min_reflectance) * (1.0 / (1.0 + jnp.exp(-scaled_wavelengths_base)))

        # --- Define Hemoglobin Absorption Features (Gaussian dips) ---
        # Parameters from the original 'skin' profile
        soret_peak = 415.0
        soret_width = 25.0
        soret_depth = 0.04 # Original depth

        q1_peak = 542.0
        q1_width = 18.0
        q1_depth = 0.02 # Original depth

        q2_peak = 577.0
        q2_width = 18.0
        q2_depth = 0.025 # Original depth

        hb_absorption = (
            soret_depth * jnp.exp(-0.5 * ((WAVELENGTHS_NM - soret_peak) / soret_width)**2) +
            q1_depth * jnp.exp(-0.5 * ((WAVELENGTHS_NM - q1_peak) / q1_width)**2) +
            q2_depth * jnp.exp(-0.5 * ((WAVELENGTHS_NM - q2_peak) / q2_width)**2)
        )

        # --- Define Melanin Absorption Feature ---
        # Parameters from the original 'skin' profile - high absorption
        melanin_center_wl = 460.0
        melanin_steepness = 0.025
        melanin_max_absorption = 0.80 # High melanin absorption

        scaled_wavelengths_melanin = (WAVELENGTHS_NM - melanin_center_wl) * melanin_steepness
        melanin_absorption_factor = 1.0 - (1.0 / (1.0 + jnp.exp(-scaled_wavelengths_melanin)))
        melanin_absorption = melanin_max_absorption * melanin_absorption_factor

        # --- Combine Components ---
        spd = base_spd - hb_absorption - melanin_absorption

        # --- Apply Scaling Factor ---
        scale_factor = value if value is not None else 1.0
        spd = spd * scale_factor

        # --- Clip to Plausible Range ---
        # Clip slightly lower max for dark skin to prevent unnatural brightness when scaled.
        return jnp.clip(spd, 0.0, 0.95)

    elif profile_type == "skin_fair":
        # Physically-based human skin spectral profile for FAIR skin.
        # Adjusted parameters for lower melanin but increased hemoglobin influence
        # relative to the previous 'fair' attempt, aiming for a more pink/peach tone.
        # 'value' scales the overall reflectance intensity (default 1.0).

        # --- Define Base Dermal/Epidermal Scattering Reflectance ---
        # Increased baseline reflectance for a paler appearance.
        base_center_wl = 550.0
        base_steepness = 0.018
        min_reflectance = 0.35  # Increased minimum reflectance
        max_reflectance = 0.75  # Increased maximum reflectance

        scaled_wavelengths_base = (WAVELENGTHS_NM - base_center_wl) * base_steepness
        base_spd = min_reflectance + (max_reflectance - min_reflectance) * (1.0 / (1.0 + jnp.exp(-scaled_wavelengths_base)))

        # --- Define Hemoglobin Absorption Features (Gaussian dips) ---
        # Increased depth compared to previous 'fair' attempt to add more redness/saturation,
        # closer to 'medium'/'dark' levels, as fair skin still has significant blood influence.
        soret_peak = 415.0
        soret_width = 25.0
        soret_depth = 0.035     # Slightly increased depth vs previous fair (0.03)

        q1_peak = 542.0
        q1_width = 18.0
        q1_depth = 0.02         # Increased depth vs previous fair (0.015), matches medium/dark

        q2_peak = 577.0
        q2_width = 18.0
        q2_depth = 0.025        # Increased depth vs previous fair (0.02), matches medium/dark

        hb_absorption = (
            soret_depth * jnp.exp(-0.5 * ((WAVELENGTHS_NM - soret_peak) / soret_width)**2) +
            q1_depth * jnp.exp(-0.5 * ((WAVELENGTHS_NM - q1_peak) / q1_width)**2) +
            q2_depth * jnp.exp(-0.5 * ((WAVELENGTHS_NM - q2_peak) / q2_width)**2)
        )

        # --- Define Melanin Absorption Feature ---
        # Significantly reduced melanin absorption for fair skin (kept from previous 'fair' attempt).
        melanin_center_wl = 480.0 # Adjusted center
        melanin_steepness = 0.020 # Adjusted steepness
        melanin_max_absorption = 0.20 # Significantly reduced absorption

        scaled_wavelengths_melanin = (WAVELENGTHS_NM - melanin_center_wl) * melanin_steepness
        melanin_absorption_factor = 1.0 - (1.0 / (1.0 + jnp.exp(-scaled_wavelengths_melanin)))
        melanin_absorption = melanin_max_absorption * melanin_absorption_factor

        # --- Combine Components ---
        # Subtract absorption from base reflectance
        spd = base_spd - hb_absorption - melanin_absorption

        # --- Apply Scaling Factor ---
        scale_factor = value if value is not None else 1.0
        spd = spd * scale_factor

        # --- Clip to Plausible Range ---
        # Ensure reflectance stays between 0 and 1
        return jnp.clip(spd, 0.0, 1.0) # Clip max to 1.0 for fair skin

    elif profile_type == "vegetation":
        # Vegetation with chlorophyll absorption and NIR reflectance
        green_peak = 0.8 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 550.0) / 40.0)**2)
        red_absorption = 0.6 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 680.0) / 10.0)**2)
        nir_reflectance = 0.9 * jnp.where(WAVELENGTHS_NM > 700.0, 1.0, 0.0)
        spd = (green_peak - red_absorption + nir_reflectance) * (value if value else 1.0)
        return jnp.clip(spd, 0.0, 1.0)

    elif profile_type == "water":
        # Water with wavelength-dependent absorption
        absorption_coeff = jnp.where(WAVELENGTHS_NM < 500.0, 0.2, 0.8)
        depth_factor = value if value is not None else 5.0
        spd = 0.98 * jnp.exp(-absorption_coeff * depth_factor * WAVELENGTHS_NM / 1000.0)
        return spd

    # --- Senior Engineer Review: Neon Profiles ---
    # Rationale: Previous neon models had issues:
    # 1. Incorrect absorption bands (e.g., pink absorbing green).
    # 2. Disconnected absorption and emission (emission wasn't driven by absorbed energy).
    # Refactored Model:
    # - `base_reflection`: Non-fluorescent color. Low reflectance.
    # - `absorption_spec`: Fraction of light absorbed (0-1) in UV/Blue bands.
    # - `emission_shape`: Normalized spectrum shape of emitted light (Stokes-shifted).
    # - `emission_scale`: Factor combining absorption strength and quantum yield efficiency. Scales `emission_shape`.
    # - `spd = base_reflection * (1 - absorption_spec) + emission_scale * emission_shape`
    #   - Absorption reduces base reflection in absorption bands.
    #   - Emission is added, scaled by how much energy is potentially absorbed.
    # - `quantum_yield_factor`: Tunable parameter representing fluorescence efficiency.
    # - Clipping allows HDR values (>1.0) for bright emission.

    elif profile_type == "fluorescent_pink":
        # Fluorescent neon pink: Absorbs UV/Blue, emits Pink (Red + Blue).
        # Goal: Appear pinkish normally, very bright "hot pink" under UV/Blue.
        # SR_REVISION_v2: Reduced green component in base reflection by lowering base level
        # and narrowing peaks. Increased quantum yield significantly for stronger fluorescence.

        # --- Base Reflection (Reduced green, sharper magenta/pink base) ---
        # Lowered base level and narrowed peaks to reduce overlap in green wavelengths.
        # Adjusted peak heights slightly to maintain some base color.
        base_reflect_level = 0.04 
        # Red component peak (narrower)
        base_reflect_peak_red = 0.30 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 620.0) / 50.0)**2) # Width 80->50, Height 0.25->0.30
        # Blue/Violet component peak (narrower)
        base_reflect_peak_blue = 0.15 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 440.0) / 35.0)**2) # Width 50->35, Height 0.10->0.15
        base_reflection = base_reflect_level + base_reflect_peak_red + base_reflect_peak_blue
        base_reflection = jnp.clip(base_reflection, 0.0, 1.0) # Keep base reflection physical

        # --- Absorption (Strong UV/Violet + moderate Blue absorption) ---
        # Defines the fraction of light absorbed in these bands, reducing reflection
        # and providing the energy potential for fluorescence. (Parameters unchanged)
        uv_abs_depth = 1.5 # Strong UV absorption
        uv_abs_center = 385.0
        uv_abs_width = 15.0
        uv_absorption_frac = uv_abs_depth * jnp.exp(-0.5 * ((WAVELENGTHS_NM - uv_abs_center) / uv_abs_width)**2)

        blue_abs_depth = 0.6 # Moderate blue absorption
        blue_abs_center = 460.0
        blue_abs_width = 40.0
        blue_absorption_frac = blue_abs_depth * jnp.exp(-0.5 * ((WAVELENGTHS_NM - blue_abs_center) / blue_abs_width)**2)

        # Combine absorption bands, ensuring it represents the fraction of light absorbed [0, 1]
        absorption_spec = uv_absorption_frac + blue_absorption_frac

        # Total potential absorption strength used for scaling emission brightness
        # This represents the integrated potential to absorb energy across the bands.
        total_abs_potential = uv_abs_depth + blue_abs_depth # Remains ~1.55

        # --- Emission (Bimodal: Strong Red + weaker Blue for Pink appearance, Stokes-shifted) ---
        # Shape of the light emitted due to fluorescence. (Parameters unchanged)
        # Strong Red peak
        emission_center_red = 640.0 # Deep red
        emission_width_red = 30.0
        emission_peak_red = 1.0 # Relative height reference
        emission_shape_red = emission_peak_red * jnp.exp(-0.5 * ((WAVELENGTHS_NM - emission_center_red) / emission_width_red)**2)

        # Weaker Blue/Violet peak to create the "pink" sensation
        emission_center_blue = 445.0 # Blue/Violet
        emission_width_blue = 25.0
        emission_peak_blue = 0.35 # Relative height (TUNE this for desired pink hue)
        emission_shape_blue = emission_peak_blue * jnp.exp(-0.5 * ((WAVELENGTHS_NM - emission_center_blue) / emission_width_blue)**2)

        # Combined emission shape (unnormalized)
        emission_shape = emission_shape_red + emission_shape_blue

        # --- Fluorescence Scaling ---
        # How brightly the material fluoresces based on absorbed energy potential.
        # quantum_yield_factor: Tunable parameter representing fluorescence efficiency.
        # emission_scale: Factor combining absorption strength and quantum yield efficiency. Scales `emission_shape`.
        # SENIOR REVIEW: Further increased quantum yield to enhance the UV->Pink effect,
        # making the fluorescence much more dominant when excited by UV/Blue light.
        quantum_yield_factor = 1.5 # INCREASED AGAIN (from 150.0) for much stronger fluorescence effect
        # Scale emission brightness based on total absorption potential and efficiency
        emission_scale = total_abs_potential * quantum_yield_factor
        # (Previous scale was ~1.55 * 150.0 = 232.5, New scale is ~1.55 * 250.0 = 387.5)

        # --- Combine Components ---
        # Final SPD = Reflected part + Emitted part
        # Reflected part: Base color minus the light absorbed for fluorescence
        reflected_light = base_reflection * (1.0 - absorption_spec)
        # Emitted part: Scaled emission shape (constant addition in this model)
        emitted_light = emission_scale * emission_shape
        spd = reflected_light + emitted_light

        # --- Apply User Scaling & Clip ---
        # Allow external control over brightness and clip to a high value for HDR.
        scale_factor = value if value is not None else 1.0
        spd = spd * scale_factor
        # Increase upper clip significantly to allow very bright "hot pink" emission
        # Using jnp.maximum ensures non-negativity before clipping upper bound.
        # Adjusted clip to accommodate potentially higher emission scale
        # SR_REVISION_v2: Increased clip further due to higher quantum yield.
        return jnp.minimum(jnp.maximum(spd, 0.0), 40.0) # Allow very bright HDR emission (was 25.0)
    
    elif profile_type == "fluorescent_green":
        # SR_REVISION_v5: Completely revised neon green fluorescence parameters
        # Fluorescent green: Strongly absorbs UV, emits bright green.

        # --- Base Reflection (Subtle green base) ---
        base_reflect_level = 0.05  # Reduced base reflection to emphasize fluorescence
        base_reflect_peak_green = 0.03 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 540.0) / 80.0)**2)
        base_reflection = base_reflect_level + base_reflect_peak_green

        # --- Absorption (Strong UV absorption, moderate blue) ---
        # Define absorption peaks and widths
        uv_abs_center = 385.0  # Centered on common UV-A wavelength
        uv_abs_width = 25.0    # Narrower for more concentrated absorption
        uv_abs_depth = 5.0    # Dramatically increased UV absorption

        blue_abs_center = 440.0
        blue_abs_width = 15.0
        blue_abs_depth = 1.5   # Increased blue absorption

        # Calculate Gaussian absorption profiles
        uv_absorption_frac = uv_abs_depth * jnp.exp(-0.5 * ((WAVELENGTHS_NM - uv_abs_center) / uv_abs_width)**2)
        blue_absorption_frac = blue_abs_depth * jnp.exp(-0.5 * ((WAVELENGTHS_NM - blue_abs_center) / blue_abs_width)**2)

        # Combine absorption spectra, clipping at 1.0
        absorption_spec = uv_absorption_frac + blue_absorption_frac

        # Total potential absorption strength used for scaling emission brightness
        total_abs_potential = uv_abs_depth + blue_abs_depth  # ~36.5

        # --- Emission (Bright Green emission, Stokes-shifted) ---
        # Shape of the light emitted due to fluorescence.
        emission_center_green = 532.0  # Bright green wavelength
        emission_width_green = 10.0    # Narrower for more saturated green
        emission_peak_green = 1.0      # Relative height reference
        emission_shape = emission_peak_green * jnp.exp(-0.5 * ((WAVELENGTHS_NM - emission_center_green) / emission_width_green)**2)

        # --- Fluorescence Scaling ---
        # How brightly the material fluoresces based on absorbed energy potential.
        quantum_yield_factor = 0.7  # High quantum yield for efficient conversion
        # Scale emission brightness based on total absorption potential and efficiency
        emission_scale = total_abs_potential * quantum_yield_factor  # ~36.5 * 8 = 292

        # --- Combine Components ---
        # Final SPD = Reflected part + Emitted part
        reflected_light = base_reflection * (1.0 - absorption_spec)
        emitted_light = emission_scale * emission_shape
        spd = reflected_light + emitted_light

        # --- Apply User Scaling & Clip ---
        # Allow external control over brightness and clip to a high value for HDR.
        scale_factor = value if value is not None else 1.0
        spd = spd * scale_factor
        return jnp.minimum(jnp.maximum(spd, 0.0), 60.0)  # Increased clip limit for brighter emission

    elif profile_type == "fluorescent_yellow":
        # SR_REVISION_v4: Restored unique parameters for neon yellow, maintaining structure.
        # Fluorescent yellow: Absorbs UV & Blue, emits Yellow.

        # --- Base Reflection (Subtle yellow base) ---
        base_reflect_level = 0.04
        base_reflect_peak_yellow = 0.08 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 580.0) / 100.0)**2)
        base_reflection = base_reflect_level + base_reflect_peak_yellow

        # --- Absorption (UV and Blue bands) ---
        # Define absorption peaks and widths
        uv_abs_center = 385.0
        uv_abs_width = 55.0
        uv_abs_depth = 5.0 # How much UV light is absorbed

        blue_abs_center = 470.0 # Shifted slightly higher than green
        blue_abs_width = 30.0
        blue_abs_depth = 0.7 # How much blue light is absorbed

        # Calculate Gaussian absorption profiles
        uv_absorption_frac = uv_abs_depth * jnp.exp(-0.5 * ((WAVELENGTHS_NM - uv_abs_center) / uv_abs_width)**2)
        blue_absorption_frac = blue_abs_depth * jnp.exp(-0.5 * ((WAVELENGTHS_NM - blue_abs_center) / blue_abs_width)**2)

        # Combine absorption spectra
        absorption_spec = uv_absorption_frac + blue_absorption_frac

        # Total potential absorption strength used for scaling emission brightness
        total_abs_potential = uv_abs_depth + blue_abs_depth # ~1.5

        # --- Emission (Yellow emission, Stokes-shifted) ---
        # Shape of the light emitted due to fluorescence.
        emission_center_yellow = 590.0
        emission_width_yellow = 25.0 # Slightly wider for yellow
        emission_peak_yellow = 1.0 # Relative height reference
        emission_shape = emission_peak_yellow * jnp.exp(-0.5 * ((WAVELENGTHS_NM - emission_center_yellow) / emission_width_yellow)**2)

        # --- Fluorescence Scaling ---
        # How brightly the material fluoresces based on absorbed energy potential.
        # SR_REVISION_v3: Increased quantum yield significantly for strong fluorescence.
        quantum_yield_factor = 0.7 # Tunable efficiency/brightness (was 1.7)
        # Scale emission brightness based on total absorption potential and efficiency
        emission_scale = total_abs_potential * quantum_yield_factor # ~1.5 * 150 = 225

        # --- Combine Components ---
        # Final SPD = Reflected part + Emitted part
        reflected_light = base_reflection * (1.0 - absorption_spec)
        emitted_light = emission_scale * emission_shape
        spd = reflected_light + emitted_light

        # --- Apply User Scaling & Clip ---
        # Allow external control over brightness and clip to a high value for HDR.
        scale_factor = value if value is not None else 1.0
        spd = spd * scale_factor
        # SR_REVISION_v3: Increased clip to accommodate higher emission scale.
        return jnp.minimum(jnp.maximum(spd, 0.0), 40.0) # Allow very bright HDR emission (was 2.6)

    elif profile_type == "fluorescent_orange":
        # SR_REVISION_v4: Restored unique parameters for neon orange, maintaining structure.
        # Fluorescent orange: Absorbs UV & Blue, emits Orange/Red.

        # --- Base Reflection (Subtle orange/reddish base) ---
        base_reflect_level = 0.04
        base_reflect_peak_orange = 0.08 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 600.0) / 120.0)**2)
        base_reflection = base_reflect_level + base_reflect_peak_orange

        # --- Absorption (UV and Blue bands) ---
        # Define absorption peaks and widths
        uv_abs_center = 385.0
        uv_abs_width = 55.0
        uv_abs_depth = 15.0 # Stronger UV absorption

        blue_abs_center = 460.0
        blue_abs_width = 40.0
        blue_abs_depth = 0.9 # Strong blue absorption

        # Calculate Gaussian absorption profiles
        uv_absorption_frac = uv_abs_depth * jnp.exp(-0.5 * ((WAVELENGTHS_NM - uv_abs_center) / uv_abs_width)**2)
        blue_absorption_frac = blue_abs_depth * jnp.exp(-0.5 * ((WAVELENGTHS_NM - blue_abs_center) / blue_abs_width)**2)

        # Combine absorption spectra, clipping at 1.0
        absorption_spec = uv_absorption_frac + blue_absorption_frac
        
        # Total potential absorption strength used for scaling emission brightness
        total_abs_potential = uv_abs_depth + blue_abs_depth # ~1.7

        # --- Emission (Orange/Red emission, Stokes-shifted) ---
        # Shape of the light emitted due to fluorescence.
        emission_center_orange = 660.0
        emission_width_orange = 35.0
        emission_peak_orange = 1.0 # Relative height reference
        emission_shape = emission_peak_orange * jnp.exp(-0.5 * ((WAVELENGTHS_NM - emission_center_orange) / emission_width_orange)**2)

        # --- Fluorescence Scaling ---
        # How brightly the material fluoresces based on absorbed energy potential.
        # SR_REVISION_v3: Increased quantum yield significantly for strong fluorescence.
        quantum_yield_factor = 0.8 # Tunable efficiency/brightness (was 1.5)
        # Scale emission brightness based on total absorption potential and efficiency
        emission_scale = total_abs_potential * quantum_yield_factor # ~1.7 * 150 = 255

        # --- Combine Components ---
        # Final SPD = Reflected part + Emitted part
        reflected_light = base_reflection * (1.0 - absorption_spec)
        emitted_light = emission_scale * emission_shape
        spd = reflected_light + emitted_light

        # --- Apply User Scaling & Clip ---
        # Allow external control over brightness and clip to a high value for HDR.
        # SR_REVISION_v3: Standardized default scale factor.
        scale_factor = value if value is not None else 1.0 # (was 1.1 default)
        spd = spd * scale_factor
        # SR_REVISION_v3: Increased clip to accommodate higher emission scale.
        return jnp.minimum(jnp.maximum(spd, 0.0), 40.0) # Allow very bright HDR emission (was 2.6)

    elif profile_type == "color_shift":
        # Fluorescent neon pink with green absorption and Stokes shift
        # Absorption in green spectrum (500-600nm) with Gaussian dip
        green_absorption = 0.7 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 550.0) / 40.0)**2)
        
        # Stokes-shifted emission: absorbed green light re-emitted as longer wavelength red
        # Primary emission at 630nm with narrower bandwidth
        stokes_red_peak = 2.2 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 640.0) / 10.0)**2)
        # Secondary emission in near-IR showing fluorescence efficiency
        stokes_ir_peak = 0.8 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 720.0) / 50.0)**2)
        
        # Base reflection in blue spectrum with anti-Stokes component
        blue_reflection = 1.1 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 460.0) / 25.0)**2)
        
        # Combine components with fluorescence intensity scaling
        spd = (blue_reflection + stokes_red_peak + stokes_ir_peak - green_absorption) * (value if value else 1.2)
        return jnp.minimum(jnp.maximum(spd, 0.0), 1.5)  # Allow oversaturation up to 1.5 for HDR
    
    elif profile_type == "rainbow_dispersive":
        # Create a more balanced rainbow spectrum with better color distribution
        
        # Base rainbow pattern using sine waves with different frequencies
        # Adjusted to create more uniform intensity across all wavelengths
        red_region = jnp.exp(-0.5 * ((WAVELENGTHS_NM - 650) / 40.0)**2) * 1.0
        green_region = jnp.exp(-0.5 * ((WAVELENGTHS_NM - 550) / 30.0)**2) * 0.9
        blue_region = jnp.exp(-0.5 * ((WAVELENGTHS_NM - 450) / 35.0)**2) * 1.1
        
        # Add violet and yellow to complete the spectrum
        violet_region = jnp.exp(-0.5 * ((WAVELENGTHS_NM - 400) / 25.0)**2) * 0.8
        yellow_region = jnp.exp(-0.5 * ((WAVELENGTHS_NM - 580) / 20.0)**2) * 0.85
        
        # Combine the regions to form a complete rainbow
        rainbow_base = red_region + green_region + blue_region + violet_region + yellow_region
        
        # Add interference patterns for a dispersive effect
        theta = jnp.radians(WAVELENGTHS_NM * 0.2)
        interference = (jnp.sin(theta * 2.0)**2 + jnp.cos(theta * 3.5)**2) * 0.3
        
        # Combine base rainbow with interference patterns
        spd = rainbow_base + interference
        
        # Apply user scaling
        scale = value if value is not None else 0.8
        spd = spd * scale
        
        # Ensure proper range with balanced clipping
        return jnp.clip(spd, 0.0, 1.5)

    elif profile_type == "blackbody_emitter":
        # Thermal emitter using Planckian radiation approximation.
        # 'value' is the intended temperature in Kelvin.
        # If 'value' is None, defaults to 3500K.
        # If 'value' is provided but below 1500K, it's clamped to 1500K
        # to ensure some visible emission, as very low temperatures
        # produce negligible visible light.
        default_temp = 3500.0
        min_visible_temp = 1500.0 # Clamp temperature to at least this value

        temp = value if value is not None else default_temp
        # Clamp the temperature to the minimum visible threshold
        temp = jnp.maximum(temp, min_visible_temp)

        C2 = 1.4388e7 # Planck's second constant in nm*K
        # Add epsilon to wavelength to avoid potential division by zero at wavelength=0 (though unlikely with WAVELENGTHS_NM)
        wavelengths_safe = WAVELENGTHS_NM + 1e-9
        x = C2 / (wavelengths_safe * temp)

        # Calculate spectral power distribution proportional to Planck's law B_lambda.
        # Increased scaling factor significantly (e.g., 1e18) as raw Planck values
        # can be very small depending on units (nm vs m) and might be too dim
        # relative to other scene elements or camera exposure. Adjust as needed.
        # Using jnp.expm1(x) for numerical stability: exp(x) - 1.
        # Add epsilon to denominator to prevent division by zero/NaNs.
        scaling_factor = 1#e18 # Significantly increased scaling
        denominator = (wavelengths_safe**5 * jnp.expm1(x)) + 1e-30 # Add epsilon
        spd = scaling_factor / denominator # Scaled spectral power

        # Ensure non-negative results. Return the calculated spectral power directly.
        # Normalization was removed previously as it discards intensity.
        return jnp.maximum(spd, 0.0)

    elif profile_type == "white_paint":
        # Broad-spectrum white paint with slight warm tint
        base = 0.85 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 580.0) / 300.0)**2)
        spd = base + 0.1 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 450.0) / 100.0)**2)
        return jnp.clip(spd * (value if value else 1.0), 0.0, 0.95)
    
    elif profile_type == "aurora_borealis":
        # Simulates the northern lights with ionized gas emission lines
        # Oxygen (green/red) and nitrogen (blue/purple) transitions
        intensity_scale = value if value is not None else 2.5
        spd = jnp.zeros_like(WAVELENGTHS_NM)
        
        # Oxygen emission lines (green and red)
        spd += 1.2 * jnp.exp(-0.5*((WAVELENGTHS_NM - 557.7)/0.3)**2)  # Strong green line
        spd += 0.6 * jnp.exp(-0.5*((WAVELENGTHS_NM - 630.0)/0.4)**2)  # Red line
        spd += 0.4 * jnp.exp(-0.5*((WAVELENGTHS_NM - 636.4)/0.4)**2)  # Secondary red
        
        # Nitrogen emission lines (blue/purple)
        spd += 0.9 * jnp.exp(-0.5*((WAVELENGTHS_NM - 427.8)/0.35)**2) # Blue-purple
        spd += 0.3 * jnp.exp(-0.5*((WAVELENGTHS_NM - 670.5)/0.5)**2)  # Weak red
        
        # Add faint continuum from night sky
        spd += 0.07 * jnp.exp(-0.5*((WAVELENGTHS_NM - 555.0)/300.0)**2)
        
        return jnp.clip(spd * intensity_scale, 0.0, 3.0)

    elif profile_type == "quasar_spectrum":
        # Simulates high-redshift quasar spectrum with broad emission lines
        # and Lyman-alpha forest absorption
        z = value if value is not None else 2.5  # Redshift parameter
        spd = jnp.ones_like(WAVELENGTHS_NM) * 0.05  # Continuum
        
        # Broad emission lines (rest wavelengths)
        lines = [
            (121.6*(1+z), 3.0, 8.0),   # Lyman-alpha (redshifted)
            (154.9*(1+z), 2.5, 6.0),   # C IV
            (190.9*(1+z), 1.8, 5.0),   # C III]
            (279.8*(1+z), 2.2, 7.0)    # Mg II
        ]
        
        # Add emission lines
        for wl, amp, width in lines:
            spd += amp * jnp.exp(-0.5*((WAVELENGTHS_NM - wl)/width)**2)
        
        # Add Lyman-alpha forest absorption (random narrow dips)
        forest_density = 0.3
        for i in range(50):
            wl = 121.6*(1 + z - jax.random.uniform(jax.random.PRNGKey(i), minval=0.01, maxval=0.1))
            spd *= 1.0 - forest_density * jnp.exp(-0.5*((WAVELENGTHS_NM - wl)/0.3)**2)
        
        return jnp.maximum(spd, 0.0)

    elif profile_type == "metamaterial_resonance":
        # Artificial metamaterial with multiple sharp resonance peaks
        # from engineered nanostructures
        intensity_scale = value if value is not None else 1.2
        spd = jnp.zeros_like(WAVELENGTHS_NM)
        
        # Add multiple resonant modes
        resonances = [
            (380, 0.8, 1.5), (425, 1.2, 2.0), (480, 0.9, 1.8),
            (530, 1.5, 2.2), (610, 0.7, 3.0), (675, 1.1, 2.5),
            (720, 0.6, 4.0), (810, 0.9, 3.5)
        ]
        
        for wl, amp, width in resonances:
            spd += amp * jnp.exp(-0.5*((WAVELENGTHS_NM - wl)/width)**2)
        
        return jnp.clip(spd * intensity_scale, 0.0, 2.5)

    elif profile_type == "neutron_star_accretion":
        # Extreme environment spectrum from neutron star accretion disk
        # Features relativistic Doppler shifts and iron line emission
        intensity_scale = value if value is not None else 5.0
        spd = jnp.zeros_like(WAVELENGTHS_NM)
        
        # Base blackbody for hot accretion disk (1e6 K)
        temp = 1e6
        wavelengths_m = WAVELENGTHS_NM * 1e-9
        planck = 1e30 / (wavelengths_m**5 * (jnp.exp(0.0143877/(wavelengths_m*temp)) - 1))
        
        # Broad iron K-alpha line at 6.4 keV (Doppler shifted)
        iron_line_rest = 0.1936e3  # 6.4 keV in nm
        doppler_factor = jnp.array([0.8, 1.0, 1.2])  # Blue/rest/red shifted
        for df in doppler_factor:
            shifted_wl = iron_line_rest * df
            spd += 2.0 * jnp.exp(-0.5*((WAVELENGTHS_NM - shifted_wl)/(10*df))**2)
        
        return jnp.clip((planck + spd) * intensity_scale * 1e-12, 0.0, 10.0)

    elif profile_type == "bioluminescent_alien":
        # Hypothetical alien organism with tri-spectral bioluminescence
        intensity_scale = value if value is not None else 1.8
        spd = jnp.zeros_like(WAVELENGTHS_NM)
        
        # Primary communication wavelength (narrow)
        spd += 1.5 * jnp.exp(-0.5*((WAVELENGTHS_NM - 412.5)/1.2)**2)
        
        # Secondary photosynthetic peak (medium width)
        spd += 1.2 * jnp.exp(-0.5*((WAVELENGTHS_NM - 680.0)/15.0)**2)
        
        # Warning color emission (broad)
        spd += 0.9 * jnp.exp(-0.5*((WAVELENGTHS_NM - 550.0)/80.0)**2)
        
        # Add harmonic resonances
        for harmonic in range(2,5):
            spd += 0.3/harmonic * jnp.exp(-0.5*((WAVELENGTHS_NM - 412.5*harmonic)/1.2)**2)
        
        return jnp.clip(spd * intensity_scale, 0.0, 2.0)

    elif profile_type == "dark_matter_annihilation":
        # Theoretical spectrum from WIMP annihilation (χχ → γγ, γZ, etc.)
        # Sharp lines at predicted energy wavelengths
        intensity_scale = value if value is not None else 4.0
        spd = jnp.zeros_like(WAVELENGTHS_NM)
        
        # Predicted line features (converted from keV to nm)
        lines = [
            (0.0248, 3.0, 0.01),  # 511 keV (electron mass)
            (0.00248, 1.8, 0.005),  # 5 TeV WIMP annihilation
            (0.0124, 2.2, 0.008),  # 100 GeV SUSY neutralino
            (0.0496, 1.5, 0.015)   # 25 GeV dark photon
        ]
        
        # Convert from energy (keV) to wavelength (nm)
        for energy_keV, amp, width in lines:
            wavelength = 1.23984193 / energy_keV  # keV•nm ≈ 1.23984193
            spd += amp * jnp.exp(-0.5*((WAVELENGTHS_NM - wavelength)/width)**2)
        
        return jnp.clip(spd * intensity_scale, 0.0, 5.0)
    
    elif profile_type == "oil_slick":
        # Simulates thin-film interference (like oil on water) with oscillating reflectance.
        # 'value' scales the overall brightness/contrast of the effect.
        # 
        # Note: This appears pink/purple because:
        # 1. The phase values (400-550nm) emphasize blue/violet wavelengths
        # 2. The oscillation frequencies create constructive interference in the blue-violet range
        # 3. The cosine waves have peaks that align more in the shorter wavelengths
        #
        # To adjust color balance, modify phase values toward greens (550nm) and reds (650nm)
        
        base_reflectance = 0.05 # Base reflection (e.g., water surface)
        amplitude_scale = value if value is not None else 0.8
        
        # Adjusted components for more balanced color spectrum
        # Component 1: Slower oscillation (shifted toward green)
        freq1 = 0.025
        phase1 = 550.0  # Shifted from 400 to 550 (green region)
        amp1 = 0.4 * amplitude_scale
        
        # Component 2: Faster oscillation (shifted toward red)
        freq2 = 0.06
        phase2 = 650.0  # Shifted from 550 to 650 (red region)
        amp2 = 0.25 * amplitude_scale

        # Component 3: Medium oscillation (balanced middle)
        freq3 = 0.04
        phase3 = 500.0  # Shifted from 450 to 500 (cyan region)
        amp3 = 0.3 * amplitude_scale

        oscillation1 = amp1 * jnp.cos(freq1 * (WAVELENGTHS_NM - phase1))
        oscillation2 = amp2 * jnp.cos(freq2 * (WAVELENGTHS_NM - phase2))
        oscillation3 = amp3 * jnp.cos(freq3 * (WAVELENGTHS_NM - phase3))

        # Combine base reflectance with oscillations
        spd = base_reflectance + oscillation1 + oscillation2 + oscillation3
        
        # Clip the result to physically plausible range [0, 1]
        return jnp.clip(spd, 0.0, 1.0)
    
    elif profile_type == "uv_blacklight":
        # Simulates a UV-A blacklight source, peaking in the near-UV.
        # 'value' scales the overall intensity/power.
        # Note: Effectiveness depends on WAVELENGTHS_NM covering < 400nm.
        peak_wavelength = 385.0 # Peak emission wavelength (nm) - common for LED blacklights
        peak_width = 35.0       # Width of the emission peak (nm)
        intensity_scale = value if value is not None else 1.0 # Base intensity

        # Gaussian function for the emission peak
        spd = intensity_scale * jnp.exp(-0.5 * ((WAVELENGTHS_NM - peak_wavelength) / peak_width)**2)

        # Ensure non-negative power
        return jnp.maximum(spd, 0.0)

    elif profile_type == "ir_emitter":
        # Simulates a near-infrared (NIR) emitter, like an IR LED.
        # 'value' scales the overall intensity/power.
        # Note: Effectiveness depends on WAVELENGTHS_NM covering > 700nm.
        peak_wavelength = 850.0 # Peak emission wavelength (nm) - common for NIR LEDs
        peak_width = 30.0       # Width of the emission peak (nm)
        intensity_scale = value if value is not None else 1.0 # Base intensity

        # Gaussian function for the emission peak
        spd = intensity_scale * jnp.exp(-0.5 * ((WAVELENGTHS_NM - peak_wavelength) / peak_width)**2)

        # Ensure non-negative power
        return jnp.maximum(spd, 0.0)
    
    elif profile_type == "laser_red":
        # Simulates a red laser pointer with a very narrow emission spectrum.
        # 'value' scales the overall intensity/power.
        peak_wavelength = 650.0 # Peak emission wavelength (nm)
        peak_width = 1.0        # Very narrow width (nm) - approximating monochromatic
        intensity_scale = value if value is not None else 10.0 # Lasers are typically intense

        # Gaussian function for the emission peak
        spd = intensity_scale * jnp.exp(-0.5 * ((WAVELENGTHS_NM - peak_wavelength) / peak_width)**2)

        # Ensure non-negative power
        return jnp.maximum(spd, 0.0)
        
    elif profile_type == "laser_green":
        # Simulates a green laser pointer with a very narrow emission spectrum.
        # 'value' scales the overall intensity/power.
        peak_wavelength = 532.0 # Peak emission wavelength (nm) - common DPSS green laser
        peak_width = 1.0        # Very narrow width (nm) - approximating monochromatic
        intensity_scale = value if value is not None else 10.0 # Lasers are typically intense

        # Gaussian function for the emission peak
        spd = intensity_scale * jnp.exp(-0.5 * ((WAVELENGTHS_NM - peak_wavelength) / peak_width)**2)

        # Ensure non-negative power
        return jnp.maximum(spd, 0.0)
        
    elif profile_type == "laser_blue":
        # Simulates a blue laser pointer with a very narrow emission spectrum.
        # 'value' scales the overall intensity/power.
        peak_wavelength = 460.0 # Peak emission wavelength (nm) - common blue diode laser
        peak_width = 1.0        # Very narrow width (nm) - approximating monochromatic
        intensity_scale = value if value is not None else 10.0 # Lasers are typically intense

        # Gaussian function for the emission peak
        spd = intensity_scale * jnp.exp(-0.5 * ((WAVELENGTHS_NM - peak_wavelength) / peak_width)**2)

        # Ensure non-negative power
        return jnp.maximum(spd, 0.0)
        
    elif profile_type == "laser_violet":
        # Simulates a violet/purple laser pointer with a very narrow emission spectrum.
        # 'value' scales the overall intensity/power.
        peak_wavelength = 425.0 # Peak emission wavelength (nm) - common violet diode laser
        peak_width = 2.0        # Very narrow width (nm) - approximating monochromatic
        intensity_scale = value if value is not None else 10.0 # Lasers are typically intense

        # Gaussian function for the emission peak
        spd = intensity_scale * jnp.exp(-0.5 * ((WAVELENGTHS_NM - peak_wavelength) / peak_width)**2)

        # Ensure non-negative power
        return jnp.maximum(spd, 0.0)
        
    elif profile_type == "laser_yellow":
        # Simulates a yellow laser with a very narrow emission spectrum.
        # 'value' scales the overall intensity/power.
        peak_wavelength = 589.0 # Peak emission wavelength (nm) - sodium D line
        peak_width = 1.0        # Very narrow width (nm) - approximating monochromatic
        intensity_scale = value if value is not None else 10.0 # Lasers are typically intense

        # Gaussian function for the emission peak
        spd = intensity_scale * jnp.exp(-0.5 * ((WAVELENGTHS_NM - peak_wavelength) / peak_width)**2)

        # Ensure non-negative power
        return jnp.maximum(spd, 0.0)
        
    elif profile_type == "laser_cyan":
        # Simulates a cyan laser with a very narrow emission spectrum.
        # 'value' scales the overall intensity/power.
        peak_wavelength = 488.0 # Peak emission wavelength (nm) - argon-ion laser line
        peak_width = 1.0        # Very narrow width (nm) - approximating monochromatic
        intensity_scale = value if value is not None else 10.0 # Lasers are typically intense

        # Gaussian function for the emission peak
        spd = intensity_scale * jnp.exp(-0.5 * ((WAVELENGTHS_NM - peak_wavelength) / peak_width)**2)

        # Ensure non-negative power
        return jnp.maximum(spd, 0.0)

    elif profile_type == "led_white_cool":
        # Simulates a cool white LED source (blue LED + yellow/green phosphor).
        # 'value' scales the overall intensity/power.
        
        # Blue LED peak parameters
        blue_peak_wl = 455.0
        blue_peak_width = 15.0
        blue_peak_rel_intensity = 0.8 # Relative intensity of blue peak

        # Phosphor emission parameters (broad yellow/green)
        phosphor_peak_wl = 560.0
        phosphor_peak_width = 60.0
        phosphor_peak_rel_intensity = 1.0 # Relative intensity of phosphor emission

        intensity_scale = value if value is not None else 1.0 # Overall intensity scale

        # Calculate components
        blue_component = blue_peak_rel_intensity * jnp.exp(-0.5 * ((WAVELENGTHS_NM - blue_peak_wl) / blue_peak_width)**2)
        phosphor_component = phosphor_peak_rel_intensity * jnp.exp(-0.5 * ((WAVELENGTHS_NM - phosphor_peak_wl) / phosphor_peak_width)**2)

        # Combine and scale
        spd = intensity_scale * (blue_component + phosphor_component)

        # Ensure non-negative power
        return jnp.maximum(spd, 0.0)
    
    elif profile_type == "led_white_warm":
        # Simulates a warm white LED source (blue LED + yellow/orange phosphor).
        # 'value' scales the overall intensity/power.
        
        # Blue LED peak parameters
        blue_peak_wl = 455.0
        blue_peak_width = 15.0
        blue_peak_rel_intensity = 0.6  # Reduced blue component compared to cool white
        
        # Phosphor emission parameters (broader and more orange-shifted)
        phosphor_peak_wl = 590.0
        phosphor_peak_width = 70.0
        phosphor_peak_rel_intensity = 1.2  # Stronger phosphor emission for warmer appearance
        
        intensity_scale = value if value is not None else 1.0  # Overall intensity scale
        
        # Calculate components
        blue_component = blue_peak_rel_intensity * jnp.exp(-0.5 * ((WAVELENGTHS_NM - blue_peak_wl) / blue_peak_width)**2)
        phosphor_component = phosphor_peak_rel_intensity * jnp.exp(-0.5 * ((WAVELENGTHS_NM - phosphor_peak_wl) / phosphor_peak_width)**2)
        
        # Combine and scale
        spd = intensity_scale * (blue_component + phosphor_component)
        
        # Ensure non-negative power
        return jnp.maximum(spd, 0.0)
    
    elif profile_type == "fluorescent_cool":
        # Simulates a cool white fluorescent tube with characteristic mercury emission lines
        # 'value' scales the overall intensity/power.
        
        # Base phosphor emission (broad spectrum)
        phosphor_peak_wl = 550.0
        phosphor_peak_width = 80.0
        phosphor_intensity = 0.8
        
        # Mercury emission lines (sharp peaks)
        hg_lines = [
            (405.0, 0.15, 3.0),  # (wavelength, relative intensity, width)
            (436.0, 0.3, 3.0),
            (546.0, 0.4, 3.0),
            (578.0, 0.15, 3.0)
        ]
        
        intensity_scale = value if value is not None else 1.0
        
        # Calculate phosphor component (broad base)
        phosphor_component = phosphor_intensity * jnp.exp(-0.5 * ((WAVELENGTHS_NM - phosphor_peak_wl) / phosphor_peak_width)**2)
        
        # Add mercury emission lines
        mercury_component = jnp.zeros_like(WAVELENGTHS_NM)
        for wl, intensity, width in hg_lines:
            mercury_component += intensity * jnp.exp(-0.5 * ((WAVELENGTHS_NM - wl) / width)**2)
        
        # Combine and scale
        spd = intensity_scale * (phosphor_component + mercury_component)
        
        return jnp.maximum(spd, 0.0)
    
    elif profile_type == "fluorescent_warm":
        # Simulates a warm white fluorescent tube with characteristic mercury emission lines
        # 'value' scales the overall intensity/power.
        
        # Base phosphor emission (broad spectrum, shifted toward yellow/orange)
        phosphor_peak_wl = 580.0
        phosphor_peak_width = 90.0
        phosphor_intensity = 1.0
        
        # Mercury emission lines (sharp peaks)
        hg_lines = [
            (405.0, 0.1, 3.0),   # (wavelength, relative intensity, width)
            (436.0, 0.2, 3.0),
            (546.0, 0.3, 3.0),
            (578.0, 0.15, 3.0)
        ]
        
        intensity_scale = value if value is not None else 1.0
        
        # Calculate phosphor component (broad base)
        phosphor_component = phosphor_intensity * jnp.exp(-0.5 * ((WAVELENGTHS_NM - phosphor_peak_wl) / phosphor_peak_width)**2)
        
        # Add mercury emission lines
        mercury_component = jnp.zeros_like(WAVELENGTHS_NM)
        for wl, intensity, width in hg_lines:
            mercury_component += intensity * jnp.exp(-0.5 * ((WAVELENGTHS_NM - wl) / width)**2)
        
        # Combine and scale
        spd = intensity_scale * (phosphor_component + mercury_component)
        
        return jnp.maximum(spd, 0.0)
    
    elif profile_type == "daylight_noon":
        # Simulates noon daylight (direct sunlight) with a smooth spectrum
        # 'value' scales the overall intensity/power.
        
        # Parameters for a smooth curve approximating daylight at noon
        intensity_scale = value if value is not None else 1.0
        
        # Slight dip in the UV/blue region
        blue_dip = 0.05 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 420.0) / 30.0)**2)
        
        # Slight dip in the infrared region
        ir_dip = 0.1 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 720.0) / 40.0)**2)
        
        # Base curve (slightly higher in middle wavelengths)
        base = 0.9 + 0.1 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 550.0) / 120.0)**2)
        
        # Combine components
        spd = intensity_scale * (base - blue_dip - ir_dip)
        
        return jnp.maximum(spd, 0.0)
    
    elif profile_type == "daylight_sunset":
        # Simulates sunset daylight with enhanced red/orange components
        # 'value' scales the overall intensity/power.
        
        intensity_scale = value if value is not None else 1.0
        
        # Strong attenuation in blue/violet (Rayleigh scattering)
        blue_attenuation = 0.8 * (1.0 - jnp.exp(-0.5 * ((WAVELENGTHS_NM - 400.0) / 80.0)**2))
        
        # Enhanced red/orange components
        red_enhancement = 0.3 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 650.0) / 60.0)**2)
        
        # Base daylight curve
        base = 0.7 + 0.3 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 550.0) / 120.0)**2)
        
        # Combine components
        spd = intensity_scale * ((base - blue_attenuation) + red_enhancement)
        
        return jnp.maximum(spd, 0.0)
    
    elif profile_type == "neon_red":
        # Simulates a red neon light with characteristic emission lines
        # 'value' scales the overall intensity/power.
        
        intensity_scale = value if value is not None else 1.0
        
        # Neon emission lines (primary in red region)
        emission_lines = [
            (640.0, 1.0, 5.0),   # (wavelength, relative intensity, width)
            (633.0, 0.8, 4.0),
            (612.0, 0.3, 4.0),
            (585.0, 0.2, 4.0)
        ]
        
        # Calculate emission spectrum
        spd = jnp.zeros_like(WAVELENGTHS_NM)
        for wl, intensity, width in emission_lines:
            spd += intensity * jnp.exp(-0.5 * ((WAVELENGTHS_NM - wl) / width)**2)
        
        return intensity_scale * jnp.maximum(spd, 0.0)
    
    elif profile_type == "neon_blue":
        # Simulates a blue/argon neon light with characteristic emission lines
        # 'value' scales the overall intensity/power.
        
        intensity_scale = value if value is not None else 1.0
        
        # Argon/mercury emission lines (blue region)
        emission_lines = [
            (436.0, 0.7, 4.0),   # (wavelength, relative intensity, width)
            (450.0, 0.9, 5.0),
            (470.0, 1.0, 5.0),
            (490.0, 0.5, 4.0)
        ]
        
        # Calculate emission spectrum
        spd = jnp.zeros_like(WAVELENGTHS_NM)
        for wl, intensity, width in emission_lines:
            spd += intensity * jnp.exp(-0.5 * ((WAVELENGTHS_NM - wl) / width)**2)
        
        return intensity_scale * jnp.maximum(spd, 0.0)
    
    elif profile_type == "camera_flash":
        # Simulates a camera xenon flash tube with broad spectrum and xenon emission lines
        # 'value' scales the overall intensity/power.
        
        intensity_scale = value if value is not None else 10.0  # Flashes are very intense
        
        # Broad continuous spectrum (approximating blackbody)
        temp = 6000.0  # Approximate color temperature
        h = 6.626e-34  # Planck's constant
        c = 2.998e8    # Speed of light
        k = 1.381e-23  # Boltzmann constant
        
        # Convert wavelengths from nm to m
        wavelengths_m = WAVELENGTHS_NM * 1e-9
        
        # Simplified Planck's law (unnormalized)
        exponent = h * c / (wavelengths_m * k * temp)
        planck = 1.0 / (wavelengths_m**5 * (jnp.exp(exponent) - 1.0))
        
        # Normalize to 0-1 range
        planck_normalized = planck / jnp.max(planck)
        
        # Xenon emission lines
        xe_lines = [
            (450.0, 0.2, 3.0),   # (wavelength, relative intensity, width)
            (470.0, 0.15, 3.0),
            (490.0, 0.1, 3.0),
            (820.0, 0.3, 4.0),
            (880.0, 0.25, 4.0)
        ]
        
        # Add emission lines to continuous spectrum
        for wl, intensity, width in xe_lines:
            if wl <= WAVELENGTHS_NM[-1] and wl >= WAVELENGTHS_NM[0]:
                planck_normalized += intensity * jnp.exp(-0.5 * ((WAVELENGTHS_NM - wl) / width)**2)
        
        return intensity_scale * jnp.maximum(planck_normalized, 0.0)
    
    elif profile_type == "led_rgb":
        # Simulates an RGB LED with three distinct peaks
        # 'value' scales the overall intensity/power.
        
        intensity_scale = value if value is not None else 1.0
        
        # RGB peak parameters
        red_peak_wl = 630.0
        red_peak_width = 20.0
        red_intensity = 1.0
        
        green_peak_wl = 530.0
        green_peak_width = 20.0
        green_intensity = 0.8
        
        blue_peak_wl = 465.0
        blue_peak_width = 20.0
        blue_intensity = 0.9
        
        # Calculate RGB components
        red_component = red_intensity * jnp.exp(-0.5 * ((WAVELENGTHS_NM - red_peak_wl) / red_peak_width)**2)
        green_component = green_intensity * jnp.exp(-0.5 * ((WAVELENGTHS_NM - green_peak_wl) / green_peak_width)**2)
        blue_component = blue_intensity * jnp.exp(-0.5 * ((WAVELENGTHS_NM - blue_peak_wl) / blue_peak_width)**2)
        
        # Combine and scale
        spd = intensity_scale * (red_component + green_component + blue_component)
        
        return jnp.maximum(spd, 0.0)

    elif profile_type == "butterfly_wing_blue":
        # Simulates structural color (like Morpho butterfly) with a sharp reflectance peak.
        # 'value' scales the peak intensity.
        peak_wavelength = 470.0 # Center of the blue reflectance peak (nm)
        peak_width = 25.0       # Narrow width for saturated color (nm)
        peak_intensity = value if value is not None else 0.9 # Max reflectance at the peak
        base_reflectance = 0.01 # Very low reflectance outside the peak

        # Gaussian function for the sharp peak
        gaussian_peak = peak_intensity * jnp.exp(-0.5 * ((WAVELENGTHS_NM - peak_wavelength) / peak_width)**2)
        
        spd = base_reflectance + gaussian_peak
        
        # Clip to ensure values are within a reasonable range
        return jnp.clip(spd, 0.0, 1.0)

    elif profile_type == "stained_glass_red":
        # Simulates the transmission spectrum of red stained glass (acting as a filter).
        # High transmission in red, low elsewhere.
        # 'value' scales the maximum transmission level.
        
        cutoff_wavelength = 590.0 # Wavelength where transmission starts increasing (nm)
        steepness = 0.08          # Sharpness of the transmission cutoff
        min_transmission = 0.01   # Minimum transmission in absorbed regions
        max_transmission = value if value is not None else 0.85 # Max transmission in red region
        
        # Sigmoid function for the transmission curve
        scaled_wavelengths = (WAVELENGTHS_NM - cutoff_wavelength) * steepness
        sigmoid_vals = 1.0 / (1.0 + jnp.exp(-scaled_wavelengths))
        
        spd = min_transmission + (max_transmission - min_transmission) * sigmoid_vals
        
        # Clip the result
        return jnp.clip(spd, 0.0, 1.0)

    elif profile_type == "blood":
        # Simplified spectral reflectance model for blood, emphasizing hemoglobin absorption.
        # Based on general features from sources like https://omlc.org/spectra/hemoglobin/
        # Note: This is a qualitative approximation, not precise measured data.
        # 'value' acts as a scaling factor for the overall reflectance intensity.

        # --- Define Hemoglobin Absorption Features (using Gaussian dips) ---
        # These represent reductions from a base reflectance level.

        # Soret band (strong absorption around 415 nm)
        soret_peak = 415.0
        soret_width = 15.0  # Relatively narrow peak
        soret_rel_depth = 0.95 # Relative depth of absorption (0 to 1)

        # Q-bands (absorption in the green-yellow region, peaks differ for HbO2 vs Hb)
        # We approximate combined/average features here.
        q1_peak = 542.0
        q1_width = 12.0
        q1_rel_depth = 0.75

        q2_peak = 577.0
        q2_width = 12.0
        q2_rel_depth = 0.85

        # --- Define Base Reflectance ---
        # Blood reflectance is low in blue/green, rising significantly in the red region.
        # Use a sigmoid function to model this rise.
        base_center_wl = 595.0 # Wavelength around which reflectance starts rising sharply
        base_steepness = 0.03  # Controls how sharp the rise is
        min_reflectance = 0.01 # Minimum reflectance in the blue/violet
        max_reflectance = 0.50 # Maximum reflectance in the deep red (can be scaled by 'value')

        scaled_wavelengths = (WAVELENGTHS_NM - base_center_wl) * base_steepness
        sigmoid_vals = 1.0 / (1.0 + jnp.exp(-scaled_wavelengths))
        base_reflectance = min_reflectance + (max_reflectance - min_reflectance) * sigmoid_vals

        # --- Calculate Absorption Dips ---
        # Absorption depth is relative to the base reflectance at that wavelength.
        absorption_soret = (base_reflectance * soret_rel_depth *
                            jnp.exp(-0.5 * ((WAVELENGTHS_NM - soret_peak) / soret_width)**2))
        absorption_q1 = (base_reflectance * q1_rel_depth *
                         jnp.exp(-0.5 * ((WAVELENGTHS_NM - q1_peak) / q1_width)**2))
        absorption_q2 = (base_reflectance * q2_rel_depth *
                         jnp.exp(-0.5 * ((WAVELENGTHS_NM - q2_peak) / q2_width)**2))

        # --- Combine Base Reflectance and Absorption ---
        spd = base_reflectance - absorption_soret - absorption_q1 - absorption_q2

        # --- Apply Scaling Factor and Clip ---
        # 'value' scales the overall intensity. Default to 1.0 if not provided.
        scale_factor = value if value is not None else 1.0
        spd = spd * scale_factor

        # Clip to ensure physically plausible reflectance (non-negative)
        # Allow maximum to potentially exceed 1 slightly if scale_factor is large,
        # but keep minimum at 0. A hard clip at 1.0 might be safer depending on usage.
        spd = jnp.maximum(spd, 0.0)
        # Optional: Clip maximum for strict reflectance: spd = jnp.clip(spd, 0.0, 1.0)

        return spd

    elif profile_type == "absorption_bands":
        # Material with specific absorption bands
        base = 0.8 * jnp.ones_like(WAVELENGTHS_NM)
        absorption = 0.4 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 450.0) / 10.0)**2)
        absorption += 0.6 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 600.0) / 5.0)**2)
        spd = jnp.clip(base - absorption, 0.1, 1.0) * (value if value else 1.0)
        return spd
    else:
        raise ValueError(f"Unknown spectrum profile type: {profile_type}") 
