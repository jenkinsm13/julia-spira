# Library of spectral power distributions for materials and light sources

using LinearAlgebra
using StaticArrays

"""
    create_spectrum_profile(profile_type::String, value=nothing)

Creates a spectral power distribution based on the specified profile type and value.
Returns a Spectrum (Vector{Float32} of length N_WAVELENGTHS)
"""
function create_spectrum_profile(profile_type::String, value=nothing)
    # Initialize a flat spectrum
    spectrum = zeros(Float32, N_WAVELENGTHS)
    
    if profile_type == "flat"
        # Python: return jnp.full((N_WAVELENGTHS,), value if value is not None else 0.8)
        actual_value = value === nothing ? Float32(0.8) : Float32(value)
        fill!(spectrum, actual_value)
        
    elseif profile_type == "red"
        # Python:
        # center_wl = 630.0
        # steepness = 0.1
        # min_reflectance = 0.02
        # max_reflectance_py = value if value is not None else 0.9
        # scaled_wavelengths = (WAVELENGTHS_NM - center_wl) * steepness
        # sigmoid_vals = 1.0 / (1.0 + jnp.exp(-scaled_wavelengths))
        # spd = min_reflectance + (max_reflectance_py - min_reflectance) * sigmoid_vals
        
        center_wl = Float32(630.0)
        steepness = Float32(0.1)
        min_reflectance = Float32(0.02)
        max_reflectance_val = value === nothing ? Float32(0.9) : Float32(value)
        
        for i in 1:N_WAVELENGTHS
            scaled_wavelength = (WAVELENGTHS_NM[i] - center_wl) * steepness
            sigmoid_val = Float32(1.0) / (Float32(1.0) + exp(-scaled_wavelength))
            spectrum[i] = min_reflectance + (max_reflectance_val - min_reflectance) * sigmoid_val
        end
        
    elseif profile_type == "green"
        # Python:
        # center_wl = 540.0
        # sigma = 40.0
        # peak_value_py = value if value is not None else 0.8
        # trough_value = 0.05
        # spd = trough_value + (peak_value_py - trough_value) * jnp.exp(-0.5 * ((WAVELENGTHS_NM - center_wl) / sigma)**2)
        
        center_wl = Float32(540.0)
        sigma = Float32(40.0)
        peak_value_val = value === nothing ? Float32(0.8) : Float32(value)
        trough_value = Float32(0.05)
        
        for i in 1:N_WAVELENGTHS
            spectrum[i] = trough_value + (peak_value_val - trough_value) * exp(Float32(-0.5) * ((WAVELENGTHS_NM[i] - center_wl) / sigma)^2)
        end
        
    elseif profile_type == "blue"
        # Python:
        # center_wl = 490.0
        # steepness = 0.05
        # min_reflectance = 0.05
        # max_reflectance_py = value if value is not None else 0.8
        # scaled_wavelengths = (WAVELENGTHS_NM - center_wl) * steepness
        # sigmoid_vals = 1.0 / (1.0 + jnp.exp(-scaled_wavelengths))
        # spd = min_reflectance + (max_reflectance_py - min_reflectance) * (1.0 - sigmoid_vals)

        center_wl = Float32(490.0)
        steepness = Float32(0.05)
        min_reflectance = Float32(0.05)
        max_reflectance_val = value === nothing ? Float32(0.8) : Float32(value)
        
        for i in 1:N_WAVELENGTHS
            scaled_wavelength = (WAVELENGTHS_NM[i] - center_wl) * steepness
            sigmoid_val = Float32(1.0) / (Float32(1.0) + exp(-scaled_wavelength))
            spectrum[i] = min_reflectance + (max_reflectance_val - min_reflectance) * (Float32(1.0) - sigmoid_val)
        end
        
    elseif profile_type == "gold"
        # Python:
        # base_reflectance_py = value if value is not None else 0.92
        # wl_peak = 700.0
        # wl_width = 200.0
        # spd = base_reflectance_py * jnp.exp(-0.5 * ((WAVELENGTHS_NM - wl_peak) / wl_width)**2)
        # spd += 0.3 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 450.0) / 50.0)**2)
        # return jnp.clip(spd, 0.0, 1.0)
        
        base_reflectance_val = value === nothing ? Float32(0.92) : Float32(value)
        wl_peak = Float32(700.0)
        wl_width = Float32(200.0)
        
        for i in 1:N_WAVELENGTHS
            main_peak_val = base_reflectance_val * exp(Float32(-0.5) * ((WAVELENGTHS_NM[i] - wl_peak) / wl_width)^2)
            blue_absorption_val = Float32(0.3) * exp(Float32(-0.5) * ((WAVELENGTHS_NM[i] - Float32(450.0)) / Float32(50.0))^2)
            spectrum[i] = clamp(main_peak_val + blue_absorption_val, Float32(0.0), Float32(1.0))
        end
        
    elseif profile_type == "copper"
        # Python:
        # main_peak = 0.8 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 620.0) / 80.0)**2)
        # secondary_peak = 0.4 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 450.0) / 30.0)**2)
        # spd = (value if value is not None else 1.0) * (main_peak + secondary_peak)
        # return jnp.clip(spd, 0.0, 0.95)
        
        scale_val = value === nothing ? Float32(1.0) : Float32(value)
        for i in 1:N_WAVELENGTHS
            main_peak_val = Float32(0.8) * exp(Float32(-0.5) * ((WAVELENGTHS_NM[i] - Float32(620.0)) / Float32(80.0))^2)
            secondary_peak_val = Float32(0.4) * exp(Float32(-0.5) * ((WAVELENGTHS_NM[i] - Float32(450.0)) / Float32(30.0))^2)
            spectrum[i] = clamp(scale_val * (main_peak_val + secondary_peak_val), Float32(0.0), Float32(0.95))
        end
        
    elseif profile_type == "silver"
        # Python:
        # base = jnp.full_like(WAVELENGTHS_NM, 0.95 * (value if value else 1.0)) # Python 'if value else 1.0' means if value is not None and not 0
        # blue_enhance = 0.1 * jnp.exp(-0.5 * ((WAVELENGTHS_NM - 450.0) / 40.0)**2)
        # return jnp.clip(base + blue_enhance, 0.0, 1.0)

        effective_value_for_silver = Float32(1.0) # Default if value is None or 0.0
        if value !== nothing
            val_float = Float32(value)
            if val_float != Float32(0.0)
                effective_value_for_silver = val_float
            end
        end
        base_fill_val = Float32(0.95) * effective_value_for_silver
        
        for i in 1:N_WAVELENGTHS
            blue_enhance_val = Float32(0.1) * exp(Float32(-0.5) * ((WAVELENGTHS_NM[i] - Float32(450.0)) / Float32(40.0))^2)
            spectrum[i] = clamp(base_fill_val + blue_enhance_val, Float32(0.0), Float32(1.0))
        end
        
    elseif profile_type == "skin_medium"
        base_center_wl = Float32(550.0); base_steepness = Float32(0.015); min_r = Float32(0.15); max_r = Float32(0.65)
        soret_peak = Float32(415.0); soret_width = Float32(25.0); soret_depth = Float32(0.04)
        q1_peak = Float32(542.0); q1_width = Float32(18.0); q1_depth = Float32(0.02)
        q2_peak = Float32(577.0); q2_width = Float32(18.0); q2_depth = Float32(0.025)
        melanin_center_wl = Float32(450.0); melanin_steepness = Float32(0.02); melanin_max_absorption = Float32(0.45)
        scale_factor = value === nothing ? Float32(1.0) : Float32(value)

        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            scaled_wl_base = (wl - base_center_wl) * base_steepness
            base_spd_val = min_r + (max_r - min_r) * (Float32(1.0) / (Float32(1.0) + exp(-scaled_wl_base)))
            
            hb_absorption_val = (
                soret_depth * exp(Float32(-0.5) * ((wl - soret_peak) / soret_width)^2) +
                q1_depth * exp(Float32(-0.5) * ((wl - q1_peak) / q1_width)^2) +
                q2_depth * exp(Float32(-0.5) * ((wl - q2_peak) / q2_width)^2)
            )
            
            scaled_wl_melanin = (wl - melanin_center_wl) * melanin_steepness
            melanin_absorption_factor_val = Float32(1.0) - (Float32(1.0) / (Float32(1.0) + exp(-scaled_wl_melanin)))
            melanin_absorption_val = melanin_max_absorption * melanin_absorption_factor_val
            
            unscaled_spd = base_spd_val - hb_absorption_val - melanin_absorption_val
            spectrum[i] = clamp(unscaled_spd * scale_factor, Float32(0.0), Float32(0.95))
        end

    elseif profile_type == "skin_dark"
        base_center_wl = Float32(560.0); base_steepness = Float32(0.016); min_r = Float32(0.08); max_r = Float32(0.55)
        soret_peak = Float32(415.0); soret_width = Float32(25.0); soret_depth = Float32(0.04)
        q1_peak = Float32(542.0); q1_width = Float32(18.0); q1_depth = Float32(0.02)
        q2_peak = Float32(577.0); q2_width = Float32(18.0); q2_depth = Float32(0.025)
        melanin_center_wl = Float32(460.0); melanin_steepness = Float32(0.025); melanin_max_absorption = Float32(0.80)
        scale_factor = value === nothing ? Float32(1.0) : Float32(value)

        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            scaled_wl_base = (wl - base_center_wl) * base_steepness
            base_spd_val = min_r + (max_r - min_r) * (Float32(1.0) / (Float32(1.0) + exp(-scaled_wl_base)))
            
            hb_absorption_val = (
                soret_depth * exp(Float32(-0.5) * ((wl - soret_peak) / soret_width)^2) +
                q1_depth * exp(Float32(-0.5) * ((wl - q1_peak) / q1_width)^2) +
                q2_depth * exp(Float32(-0.5) * ((wl - q2_peak) / q2_width)^2)
            )
            
            scaled_wl_melanin = (wl - melanin_center_wl) * melanin_steepness
            melanin_absorption_factor_val = Float32(1.0) - (Float32(1.0) / (Float32(1.0) + exp(-scaled_wl_melanin)))
            melanin_absorption_val = melanin_max_absorption * melanin_absorption_factor_val
            
            unscaled_spd = base_spd_val - hb_absorption_val - melanin_absorption_val
            spectrum[i] = clamp(unscaled_spd * scale_factor, Float32(0.0), Float32(0.95))
        end

    elseif profile_type == "skin_fair"
        base_center_wl = Float32(550.0); base_steepness = Float32(0.018); min_r = Float32(0.35); max_r = Float32(0.75)
        soret_peak = Float32(415.0); soret_width = Float32(25.0); soret_depth = Float32(0.035)
        q1_peak = Float32(542.0); q1_width = Float32(18.0); q1_depth = Float32(0.02)
        q2_peak = Float32(577.0); q2_width = Float32(18.0); q2_depth = Float32(0.025)
        melanin_center_wl = Float32(480.0); melanin_steepness = Float32(0.020); melanin_max_absorption = Float32(0.20)
        scale_factor = value === nothing ? Float32(1.0) : Float32(value)

        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            scaled_wl_base = (wl - base_center_wl) * base_steepness
            base_spd_val = min_r + (max_r - min_r) * (Float32(1.0) / (Float32(1.0) + exp(-scaled_wl_base)))
            
            hb_absorption_val = (
                soret_depth * exp(Float32(-0.5) * ((wl - soret_peak) / soret_width)^2) +
                q1_depth * exp(Float32(-0.5) * ((wl - q1_peak) / q1_width)^2) +
                q2_depth * exp(Float32(-0.5) * ((wl - q2_peak) / q2_width)^2)
            )
            
            scaled_wl_melanin = (wl - melanin_center_wl) * melanin_steepness
            melanin_absorption_factor_val = Float32(1.0) - (Float32(1.0) / (Float32(1.0) + exp(-scaled_wl_melanin)))
            melanin_absorption_val = melanin_max_absorption * melanin_absorption_factor_val
            
            unscaled_spd = base_spd_val - hb_absorption_val - melanin_absorption_val
            spectrum[i] = clamp(unscaled_spd * scale_factor, Float32(0.0), Float32(1.0))
        end

    elseif profile_type == "vegetation"
        effective_scale_val = Float32(1.0)
        if value !== nothing
            val_float = Float32(value)
            if val_float != Float32(0.0)
                effective_scale_val = val_float
            end
        end

        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            green_peak_val = Float32(0.8) * exp(Float32(-0.5) * ((wl - Float32(550.0)) / Float32(40.0))^2)
            red_absorption_val = Float32(0.6) * exp(Float32(-0.5) * ((wl - Float32(680.0)) / Float32(10.0))^2)
            nir_reflectance_val = Float32(0.9) * (wl > Float32(700.0) ? Float32(1.0) : Float32(0.0))
            unscaled_spd = green_peak_val - red_absorption_val + nir_reflectance_val
            spectrum[i] = clamp(unscaled_spd * effective_scale_val, Float32(0.0), Float32(1.0))
        end
        
    elseif profile_type == "water"
        depth_factor_val = value === nothing ? Float32(5.0) : Float32(value)
        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            absorption_coeff_val = wl < Float32(500.0) ? Float32(0.2) : Float32(0.8)
            # Python does not clip, so Julia should not either by default. Max value is 0.98
            spectrum[i] = Float32(0.98) * exp(-absorption_coeff_val * depth_factor_val * wl / Float32(1000.0))
            spectrum[i] = max(spectrum[i], Float32(0.0)) # Ensure non-negative just in case
        end

    elseif profile_type == "fluorescent_pink"
        base_reflect_level = Float32(0.04)
        uv_abs_depth = Float32(1.5); uv_abs_center = Float32(385.0); uv_abs_width = Float32(15.0)
        blue_abs_depth = Float32(0.6); blue_abs_center = Float32(460.0); blue_abs_width = Float32(40.0)
        total_abs_potential = uv_abs_depth + blue_abs_depth

        emission_center_red = Float32(640.0); emission_width_red = Float32(30.0); emission_peak_red = Float32(1.0)
        emission_center_blue = Float32(445.0); emission_width_blue = Float32(25.0); emission_peak_blue = Float32(0.35)
        
        quantum_yield_factor = Float32(1.5)
        emission_scale = total_abs_potential * quantum_yield_factor

        scale_factor = value === nothing ? Float32(1.0) : Float32(value)
        
        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            
            _base_reflect_peak_red = Float32(0.30) * exp(Float32(-0.5) * ((wl - Float32(620.0)) / Float32(50.0))^2)
            _base_reflect_peak_blue = Float32(0.15) * exp(Float32(-0.5) * ((wl - Float32(440.0)) / Float32(35.0))^2)
            _base_reflection = clamp(base_reflect_level + _base_reflect_peak_red + _base_reflect_peak_blue, Float32(0.0), Float32(1.0))

            uv_absorption_frac = uv_abs_depth * exp(Float32(-0.5) * ((wl - uv_abs_center) / uv_abs_width)^2)
            blue_absorption_frac = blue_abs_depth * exp(Float32(-0.5) * ((wl - blue_abs_center) / blue_abs_width)^2)
            _absorption_spec = uv_absorption_frac + blue_absorption_frac

            emission_shape_red = emission_peak_red * exp(Float32(-0.5) * ((wl - emission_center_red) / emission_width_red)^2)
            emission_shape_blue = emission_peak_blue * exp(Float32(-0.5) * ((wl - emission_center_blue) / emission_width_blue)^2)
            _emission_shape = emission_shape_red + emission_shape_blue
            
            reflected_light = _base_reflection * (Float32(1.0) - _absorption_spec) 
            emitted_light = emission_scale * _emission_shape
            
            unscaled_spd = reflected_light + emitted_light
            spectrum[i] = clamp(unscaled_spd * scale_factor, Float32(0.0), Float32(40.0))
        end

    elseif profile_type == "fluorescent_green"
        base_reflect_level = Float32(0.05)
        uv_abs_center = Float32(385.0); uv_abs_width = Float32(25.0); uv_abs_depth = Float32(5.0)
        blue_abs_center = Float32(440.0); blue_abs_width = Float32(15.0); blue_abs_depth = Float32(1.5)
        total_abs_potential = uv_abs_depth + blue_abs_depth

        emission_center_green = Float32(532.0); emission_width_green = Float32(10.0); emission_peak_green = Float32(1.0)
        
        quantum_yield_factor = Float32(0.7)
        emission_scale = total_abs_potential * quantum_yield_factor

        scale_factor = value === nothing ? Float32(1.0) : Float32(value)

        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            
            _base_reflect_peak_green = Float32(0.03) * exp(Float32(-0.5) * ((wl - Float32(540.0)) / Float32(80.0))^2)
            _base_reflection = base_reflect_level + _base_reflect_peak_green

            uv_absorption_frac = uv_abs_depth * exp(Float32(-0.5) * ((wl - uv_abs_center) / uv_abs_width)^2)
            blue_absorption_frac = blue_abs_depth * exp(Float32(-0.5) * ((wl - blue_abs_center) / blue_abs_width)^2)
            _absorption_spec = uv_absorption_frac + blue_absorption_frac

            _emission_shape = emission_peak_green * exp(Float32(-0.5) * ((wl - emission_center_green) / emission_width_green)^2)

            reflected_light = _base_reflection * (Float32(1.0) - _absorption_spec)
            emitted_light = emission_scale * _emission_shape
            
            unscaled_spd = reflected_light + emitted_light
            spectrum[i] = clamp(unscaled_spd * scale_factor, Float32(0.0), Float32(60.0))
        end
        
    elseif profile_type == "fluorescent_yellow"
        base_reflect_level = Float32(0.04)
        uv_abs_center = Float32(385.0); uv_abs_width = Float32(55.0); uv_abs_depth = Float32(5.0)
        blue_abs_center = Float32(470.0); blue_abs_width = Float32(30.0); blue_abs_depth = Float32(0.7)
        total_abs_potential = uv_abs_depth + blue_abs_depth

        emission_center_yellow = Float32(590.0); emission_width_yellow = Float32(25.0); emission_peak_yellow = Float32(1.0)
        
        quantum_yield_factor = Float32(0.7)
        emission_scale = total_abs_potential * quantum_yield_factor

        scale_factor = value === nothing ? Float32(1.0) : Float32(value)
        
        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            
            _base_reflect_peak_yellow = Float32(0.08) * exp(Float32(-0.5) * ((wl - Float32(580.0)) / Float32(100.0))^2)
            _base_reflection = base_reflect_level + _base_reflect_peak_yellow
            
            uv_absorption_frac = uv_abs_depth * exp(Float32(-0.5) * ((wl - uv_abs_center) / uv_abs_width)^2)
            blue_absorption_frac = blue_abs_depth * exp(Float32(-0.5) * ((wl - blue_abs_center) / blue_abs_width)^2)
            _absorption_spec = uv_absorption_frac + blue_absorption_frac

            _emission_shape = emission_peak_yellow * exp(Float32(-0.5) * ((wl - emission_center_yellow) / emission_width_yellow)^2)
            
            reflected_light = _base_reflection * (Float32(1.0) - _absorption_spec)
            emitted_light = emission_scale * _emission_shape
            
            unscaled_spd = reflected_light + emitted_light
            spectrum[i] = clamp(unscaled_spd * scale_factor, Float32(0.0), Float32(40.0))
        end

    elseif profile_type == "fluorescent_orange"
        base_reflect_level = Float32(0.04)
        uv_abs_center = Float32(385.0); uv_abs_width = Float32(55.0); uv_abs_depth = Float32(15.0)
        blue_abs_center = Float32(460.0); blue_abs_width = Float32(40.0); blue_abs_depth = Float32(0.9)
        total_abs_potential = uv_abs_depth + blue_abs_depth

        emission_center_orange = Float32(660.0); emission_width_orange = Float32(35.0); emission_peak_orange = Float32(1.0)
        
        quantum_yield_factor = Float32(0.8)
        emission_scale = total_abs_potential * quantum_yield_factor
            
        scale_factor = value === nothing ? Float32(1.0) : Float32(value)

        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            
            _base_reflect_peak_orange = Float32(0.08) * exp(Float32(-0.5) * ((wl - Float32(600.0)) / Float32(120.0))^2)
            _base_reflection = base_reflect_level + _base_reflect_peak_orange

            uv_absorption_frac = uv_abs_depth * exp(Float32(-0.5) * ((wl - uv_abs_center) / uv_abs_width)^2)
            blue_absorption_frac = blue_abs_depth * exp(Float32(-0.5) * ((wl - blue_abs_center) / blue_abs_width)^2)
            _absorption_spec = uv_absorption_frac + blue_absorption_frac

            _emission_shape = emission_peak_orange * exp(Float32(-0.5) * ((wl - emission_center_orange) / emission_width_orange)^2)

            reflected_light = _base_reflection * (Float32(1.0) - _absorption_spec)
            emitted_light = emission_scale * _emission_shape
            
            unscaled_spd = reflected_light + emitted_light
            spectrum[i] = clamp(unscaled_spd * scale_factor, Float32(0.0), Float32(40.0))
        end
        
    elseif profile_type == "blackbody_emitter"
        default_temp = Float32(3500.0)
        min_visible_temp = Float32(1500.0)

        temp = value === nothing ? default_temp : Float32(value)
        temp = max(temp, min_visible_temp)

        C2 = Float32(1.4388e7) # Planck's second constant in nm*K
        scaling_factor = Float32(1.0e18)

        for i in 1:N_WAVELENGTHS
            wavelength_safe = WAVELENGTHS_NM[i] + Float32(1.0e-9)
            x = C2 / (wavelength_safe * temp)
            # jnp.expm1(x) is exp(x) - 1, for better precision with small x.
            denominator = (wavelength_safe^5 * expm1(x)) + Float32(1.0e-30) 
            spectrum[i] = max(scaling_factor / denominator, Float32(0.0))
        end

    elseif profile_type == "white_paint"
        scale_factor = value === nothing ? Float32(0.85) : Float32(value)
        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            base_val = Float32(0.90) * exp(Float32(-0.5) * ((wl - Float32(580.0)) / Float32(250.0))^2)
            blue_boost_val = Float32(0.05) * exp(Float32(-0.5) * ((wl - Float32(450.0)) / Float32(80.0))^2)
            unscaled_spd_val = base_val + blue_boost_val
            spectrum[i] = clamp(unscaled_spd_val * scale_factor, Float32(0.0), Float32(1.0))
        end
        
    elseif profile_type == "led_white_cool"
        blue_peak_wl = Float32(455.0)
        blue_fwhm = Float32(20.0) 
        phosphor_peak_wl = Float32(555.0) 
        phosphor_fwhm = Float32(90.0) 
        blue_to_phosphor_ratio = Float32(0.7)

        fwhm_to_sigma_factor = Float32(2.0 * sqrt(2.0 * log(2.0)))
        sigma_blue = blue_fwhm / fwhm_to_sigma_factor
        sigma_phosphor = phosphor_fwhm / fwhm_to_sigma_factor

        unnormalized_spd_vals = zeros(Float32, N_WAVELENGTHS)
        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            blue_component = exp(Float32(-0.5) * ((wl - blue_peak_wl) / sigma_blue)^2)
            phosphor_component = exp(Float32(-0.5) * ((wl - phosphor_peak_wl) / sigma_phosphor)^2)
            unnormalized_spd_vals[i] = blue_to_phosphor_ratio * blue_component + phosphor_component
        end

        max_val = maximum(unnormalized_spd_vals)
        normalized_spd_vals = unnormalized_spd_vals ./ max(max_val, Float32(1e-7))
        
        scale_factor = value === nothing ? Float32(15.0) : Float32(value)
        spectrum .= normalized_spd_vals .* scale_factor
        spectrum .= max.(spectrum, Float32(0.0))

    elseif profile_type == "led_white_warm"
        blue_peak_wl_ww = Float32(455.0); blue_peak_width_ww = Float32(15.0); blue_rel_intensity_ww = Float32(0.6)
        phosphor_peak_wl_ww = Float32(590.0); phosphor_peak_width_ww = Float32(70.0); phosphor_rel_intensity_ww = Float32(1.2)
        intensity_scale = value === nothing ? Float32(1.0) : Float32(value)

        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            blue_component = blue_rel_intensity_ww * exp(Float32(-0.5) * ((wl - blue_peak_wl_ww) / blue_peak_width_ww)^2)
            phosphor_component = phosphor_rel_intensity_ww * exp(Float32(-0.5) * ((wl - phosphor_peak_wl_ww) / phosphor_peak_width_ww)^2)
            spectrum[i] = max(intensity_scale * (blue_component + phosphor_component), Float32(0.0))
        end
        
    elseif profile_type == "fluorescent_cool"
        phosphor_peak_wl_fc = Float32(550.0); phosphor_peak_width_fc = Float32(80.0); phosphor_intensity_fc = Float32(0.8)
        hg_lines_data_fc = [(Float32(405.0), Float32(0.15), Float32(3.0)), (Float32(436.0), Float32(0.3), Float32(3.0)), 
                            (Float32(546.0), Float32(0.4), Float32(3.0)), (Float32(578.0), Float32(0.15), Float32(3.0))]
        intensity_scale = value === nothing ? Float32(1.0) : Float32(value)

        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            phosphor_component = phosphor_intensity_fc * exp(Float32(-0.5) * ((wl - phosphor_peak_wl_fc) / phosphor_peak_width_fc)^2)
            mercury_component = Float32(0.0)
            for (wl_hg, int_hg, wid_hg) in hg_lines_data_fc
                mercury_component += int_hg * exp(Float32(-0.5) * ((wl - wl_hg) / wid_hg)^2)
            end
            spectrum[i] = max(intensity_scale * (phosphor_component + mercury_component), Float32(0.0))
        end
        
    elseif profile_type == "fluorescent_warm"
        phosphor_peak_wl_fw = Float32(580.0); phosphor_peak_width_fw = Float32(90.0); phosphor_intensity_fw = Float32(1.0)
        hg_lines_data_fw = [(Float32(405.0), Float32(0.1), Float32(3.0)), (Float32(436.0), Float32(0.2), Float32(3.0)), 
                            (Float32(546.0), Float32(0.3), Float32(3.0)), (Float32(578.0), Float32(0.15), Float32(3.0))]
        intensity_scale = value === nothing ? Float32(1.0) : Float32(value)

        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            phosphor_component = phosphor_intensity_fw * exp(Float32(-0.5) * ((wl - phosphor_peak_wl_fw) / phosphor_peak_width_fw)^2)
            mercury_component = Float32(0.0)
            for (wl_hg, int_hg, wid_hg) in hg_lines_data_fw
                mercury_component += int_hg * exp(Float32(-0.5) * ((wl - wl_hg) / wid_hg)^2)
            end
            spectrum[i] = max(intensity_scale * (phosphor_component + mercury_component), Float32(0.0))
        end

    elseif profile_type == "daylight_noon"
        intensity_scale = value === nothing ? Float32(1.0) : Float32(value)
        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            blue_dip_val = Float32(0.05) * exp(Float32(-0.5) * ((wl - Float32(420.0)) / Float32(30.0))^2)
            ir_dip_val = Float32(0.1) * exp(Float32(-0.5) * ((wl - Float32(720.0)) / Float32(40.0))^2)
            base_val = Float32(0.9) + Float32(0.1) * exp(Float32(-0.5) * ((wl - Float32(550.0)) / Float32(120.0))^2)
            spectrum[i] = max(intensity_scale * (base_val - blue_dip_val - ir_dip_val), Float32(0.0))
        end
        
    elseif profile_type == "daylight_sunset"
        intensity_scale = value === nothing ? Float32(1.0) : Float32(value)
        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            blue_attenuation_val = Float32(0.8) * (Float32(1.0) - exp(Float32(-0.5) * ((wl - Float32(400.0)) / Float32(80.0))^2))
            red_enhancement_val = Float32(0.3) * exp(Float32(-0.5) * ((wl - Float32(650.0)) / Float32(60.0))^2)
            base_val = Float32(0.7) + Float32(0.3) * exp(Float32(-0.5) * ((wl - Float32(550.0)) / Float32(120.0))^2)
            spectrum[i] = max(intensity_scale * ((base_val - blue_attenuation_val) + red_enhancement_val), Float32(0.0))
        end
        
    elseif profile_type == "color_shift"
        effective_scale_val = Float32(1.2) 
        if value !== nothing
            val_float = Float32(value)
            if val_float != Float32(0.0)
                effective_scale_val = val_float
            end
        end
        
        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            green_absorption_val = Float32(0.7) * exp(Float32(-0.5) * ((wl - Float32(550.0)) / Float32(40.0))^2)
            stokes_red_peak_val = Float32(2.2) * exp(Float32(-0.5) * ((wl - Float32(640.0)) / Float32(10.0))^2)
            stokes_ir_peak_val = Float32(0.8) * exp(Float32(-0.5) * ((wl - Float32(720.0)) / Float32(50.0))^2)
            blue_reflection_val = Float32(1.1) * exp(Float32(-0.5) * ((wl - Float32(460.0)) / Float32(25.0))^2)
            
            unscaled_spd = (blue_reflection_val + stokes_red_peak_val + stokes_ir_peak_val - green_absorption_val)
            spectrum[i] = clamp(unscaled_spd * effective_scale_val, Float32(0.0), Float32(1.5))
        end

    elseif profile_type == "rainbow_dispersive"
        scale_val = value === nothing ? Float32(0.8) : Float32(value)
        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            red_region_val = exp(Float32(-0.5) * ((wl - Float32(650.0)) / Float32(40.0))^2) * Float32(1.0)
            green_region_val = exp(Float32(-0.5) * ((wl - Float32(550.0)) / Float32(30.0))^2) * Float32(0.9)
            blue_region_val = exp(Float32(-0.5) * ((wl - Float32(450.0)) / Float32(35.0))^2) * Float32(1.1)
            violet_region_val = exp(Float32(-0.5) * ((wl - Float32(400.0)) / Float32(25.0))^2) * Float32(0.8)
            yellow_region_val = exp(Float32(-0.5) * ((wl - Float32(580.0)) / Float32(20.0))^2) * Float32(0.85)
            rainbow_base_val = red_region_val + green_region_val + blue_region_val + violet_region_val + yellow_region_val
            
            theta_val = deg2rad(wl * Float32(0.2)) 
            interference_val = (sin(theta_val * Float32(2.0))^2 + cos(theta_val * Float32(3.5))^2) * Float32(0.3)
            
            unscaled_spd = rainbow_base_val + interference_val
            spectrum[i] = clamp(unscaled_spd * scale_val, Float32(0.0), Float32(1.5))
        end
        
    elseif profile_type == "aurora_borealis"
        intensity_scale = value === nothing ? Float32(2.5) : Float32(value)
        temp_spd_sum = zeros(Float32, N_WAVELENGTHS)
        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            spd_val = Float32(0.0)
            spd_val += Float32(1.2) * exp(Float32(-0.5)*((wl - Float32(557.7))/Float32(0.3))^2)
            spd_val += Float32(0.6) * exp(Float32(-0.5)*((wl - Float32(630.0))/Float32(0.4))^2)
            spd_val += Float32(0.4) * exp(Float32(-0.5)*((wl - Float32(636.4))/Float32(0.4))^2)
            spd_val += Float32(0.9) * exp(Float32(-0.5)*((wl - Float32(427.8))/Float32(0.35))^2)
            spd_val += Float32(0.3) * exp(Float32(-0.5)*((wl - Float32(670.5))/Float32(0.5))^2)
            spd_val += Float32(0.07) * exp(Float32(-0.5)*((wl - Float32(555.0))/Float32(300.0))^2)
            temp_spd_sum[i] = spd_val
        end
        spectrum .= clamp.(temp_spd_sum .* intensity_scale, Float32(0.0), Float32(3.0))

    elseif profile_type == "metamaterial_resonance"
        intensity_scale = value === nothing ? Float32(1.2) : Float32(value)
        temp_spd_sum_meta = zeros(Float32, N_WAVELENGTHS)
        resonances_data = [
            (Float32(380.0), Float32(0.8), Float32(1.5)), (Float32(425.0), Float32(1.2), Float32(2.0)), (Float32(480.0), Float32(0.9), Float32(1.8)),
            (Float32(530.0), Float32(1.5), Float32(2.2)), (Float32(610.0), Float32(0.7), Float32(3.0)), (Float32(675.0), Float32(1.1), Float32(2.5)),
            (Float32(720.0), Float32(0.6), Float32(4.0)), (Float32(810.0), Float32(0.9), Float32(3.5))
        ]
        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            spd_val = Float32(0.0)
            for (wl_r, amp, width_r) in resonances_data
                spd_val += amp * exp(Float32(-0.5)*((wl - wl_r)/width_r)^2)
            end
            temp_spd_sum_meta[i] = spd_val
        end
        spectrum .= clamp.(temp_spd_sum_meta .* intensity_scale, Float32(0.0), Float32(2.5))

    elseif profile_type == "quasar_spectrum"
        z = value === nothing ? Float32(2.5) : Float32(value)
        fill!(spectrum, Float32(0.05))
        
        py_lines_data = [
            (Float32(121.6)*(Float32(1.0)+z), Float32(3.0), Float32(8.0)), (Float32(154.9)*(Float32(1.0)+z), Float32(2.5), Float32(6.0)),
            (Float32(190.9)*(Float32(1.0)+z), Float32(1.8), Float32(5.0)), (Float32(279.8)*(Float32(1.0)+z), Float32(2.2), Float32(7.0))
        ]

        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            current_spd_val_at_wl = spectrum[i] 
            for (wl_l, amp, width_l) in py_lines_data
                current_spd_val_at_wl += amp * exp(Float32(-0.5)*((wl - wl_l)/width_l)^2)
            end
            spectrum[i] = current_spd_val_at_wl
        end
        
        @warn "Quasar spectrum: Lyman-alpha forest component using jax.random is not fully replicated due to PRNG differences. Results will vary from Python for this part of the quasar spectrum."
        spectrum .= max.(spectrum, Float32(0.0))

    elseif profile_type == "neutron_star_accretion"
        intensity_scale = value === nothing ? Float32(5.0) : Float32(value)
        temp_ns = Float32(1.0e6)
        hc_k_mK = Float32(0.0143877) 
        iron_line_rest_nm = Float32(0.1936e3)
        doppler_factors = [Float32(0.8), Float32(1.0), Float32(1.2)]

        planck_spectrum_component = zeros(Float32, N_WAVELENGTHS)
        for i in 1:N_WAVELENGTHS
            wl_nm = WAVELENGTHS_NM[i]
            wl_m = wl_nm * Float32(1.0e-9)
            exponent_val = hc_k_mK / (wl_m * temp_ns)
            planck_spectrum_component[i] = Float32(1.0e30) / (wl_m^5 * expm1(exponent_val) + Float32(1.0e-38)) 
        end

        iron_lines_component = zeros(Float32, N_WAVELENGTHS)
        for i in 1:N_WAVELENGTHS
            wl_nm = WAVELENGTHS_NM[i]
            current_iron_val_at_wl = Float32(0.0)
            for df_val in doppler_factors
                shifted_wl_nm = iron_line_rest_nm * df_val
                current_iron_val_at_wl += Float32(2.0) * exp(Float32(-0.5)*((wl_nm - shifted_wl_nm)/(Float32(10.0)*df_val))^2)
            end
            iron_lines_component[i] = current_iron_val_at_wl
        end
        
        for i in 1:N_WAVELENGTHS
            unclipped_val = (planck_spectrum_component[i] + iron_lines_component[i]) * intensity_scale * Float32(1.0e-12)
            spectrum[i] = clamp(unclipped_val, Float32(0.0), Float32(10.0))
        end

    elseif profile_type == "bioluminescent_alien"
        intensity_scale = value === nothing ? Float32(1.8) : Float32(value)
        temp_spd_sum = zeros(Float32, N_WAVELENGTHS)
        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            spd_val_at_wl = Float32(0.0)
            spd_val_at_wl += Float32(1.5) * exp(Float32(-0.5)*((wl - Float32(412.5))/Float32(1.2))^2)
            spd_val_at_wl += Float32(1.2) * exp(Float32(-0.5)*((wl - Float32(680.0))/Float32(15.0))^2)
            spd_val_at_wl += Float32(0.9) * exp(Float32(-0.5)*((wl - Float32(550.0))/Float32(80.0))^2)
            for harmonic_py_int in 2:4
                harmonic = Float32(harmonic_py_int)
                spd_val_at_wl += (Float32(0.3)/harmonic) * exp(Float32(-0.5)*((wl - Float32(412.5)*harmonic)/Float32(1.2))^2)
            end
            temp_spd_sum[i] = spd_val_at_wl
        end
        spectrum .= clamp.(temp_spd_sum .* intensity_scale, Float32(0.0), Float32(2.0))

    elseif profile_type == "dark_matter_annihilation"
        intensity_scale = value === nothing ? Float32(4.0) : Float32(value)
        temp_spd_sum_dma = zeros(Float32, N_WAVELENGTHS)
        lines_data = [
            (Float32(0.0248), Float32(3.0), Float32(0.01)), (Float32(0.00248), Float32(1.8), Float32(0.005)),
            (Float32(0.0124), Float32(2.2), Float32(0.008)), (Float32(0.0496), Float32(1.5), Float32(0.015))
        ]
        keV_nm_conversion_const = Float32(1.23984193)

        for i in 1:N_WAVELENGTHS
            wl_obs = WAVELENGTHS_NM[i]
            spd_val_at_wl = Float32(0.0)
            for (energy_keV, amp, width_dma) in lines_data
                wavelength_line_nm = keV_nm_conversion_const / energy_keV
                spd_val_at_wl += amp * exp(Float32(-0.5)*((wl_obs - wavelength_line_nm)/width_dma)^2)
            end
            temp_spd_sum_dma[i] = spd_val_at_wl
        end
        spectrum .= clamp.(temp_spd_sum_dma .* intensity_scale, Float32(0.0), Float32(5.0))

    elseif profile_type == "oil_slick"
        base_reflectance = Float32(0.05)
        amplitude_scale = value === nothing ? Float32(0.8) : Float32(value)
        
        freq1 = Float32(0.025); phase1 = Float32(550.0); amp1_calc = Float32(0.4) * amplitude_scale
        freq2 = Float32(0.06); phase2 = Float32(650.0); amp2_calc = Float32(0.25) * amplitude_scale
        freq3 = Float32(0.04); phase3 = Float32(500.0); amp3_calc = Float32(0.3) * amplitude_scale

        for i in 1:N_WAVELENGTHS
            wl = WAVELENGTHS_NM[i]
            oscillation1 = amp1_calc * cos(freq1 * (wl - phase1))
            oscillation2 = amp2_calc * cos(freq2 * (wl - phase2))
            oscillation3 = amp3_calc * cos(freq3 * (wl - phase3))
            spectrum[i] = clamp(base_reflectance + oscillation1 + oscillation2 + oscillation3, Float32(0.0), Float32(1.0))
        end

    elseif profile_type == "uv_blacklight"
        peak_wavelength = Float32(385.0)
        peak_width = Float32(35.0)
        intensity_scale = value === nothing ? Float32(1.0) : Float32(value)
        for i in 1:N_WAVELENGTHS
            spectrum[i] = max(intensity_scale * exp(Float32(-0.5) * ((WAVELENGTHS_NM[i] - peak_wavelength) / peak_width)^2), Float32(0.0))
        end

    elseif profile_type == "ir_emitter"
        peak_wavelength = Float32(850.0)
        peak_width = Float32(30.0)
        intensity_scale = value === nothing ? Float32(1.0) : Float32(value)
        for i in 1:N_WAVELENGTHS
            spectrum[i] = max(intensity_scale * exp(Float32(-0.5) * ((WAVELENGTHS_NM[i] - peak_wavelength) / peak_width)^2), Float32(0.0))
        end

    elseif profile_type == "laser_red"
        peak_wavelength = Float32(650.0)
        peak_width = Float32(1.0)
        intensity_scale = value === nothing ? Float32(10.0) : Float32(value)
        for i in 1:N_WAVELENGTHS
            spectrum[i] = max(intensity_scale * exp(Float32(-0.5) * ((WAVELENGTHS_NM[i] - peak_wavelength) / peak_width)^2), Float32(0.0))
        end
        
    elseif profile_type == "laser_green"
        peak_wavelength = Float32(532.0)
        peak_width = Float32(1.0)
        intensity_scale = value === nothing ? Float32(10.0) : Float32(value)
        for i in 1:N_WAVELENGTHS
            spectrum[i] = max(intensity_scale * exp(Float32(-0.5) * ((WAVELENGTHS_NM[i] - peak_wavelength) / peak_width)^2), Float32(0.0))
        end
        
    elseif profile_type == "laser_blue"
        peak_wavelength = Float32(460.0)
        peak_width = Float32(1.0)
        intensity_scale = value === nothing ? Float32(10.0) : Float32(value) 
        for i in 1:N_WAVELENGTHS
            # spectrum[i] = intensity_scale * exp(Float32(-0.5) * ((WAVELENGTHS_NM[i] - peak_wavelength) / peak_width)^2)
            # Ensuring max with 0.0 like other lasers and Python jnp.maximum(spd, 0.0)
            spectrum[i] = max(intensity_scale * exp(Float32(-0.5) * ((WAVELENGTHS_NM[i] - peak_wavelength) / peak_width)^2), Float32(0.0))
        end
        
    elseif profile_type == "laser_violet"
        peak_wavelength = Float32(425.0)
        peak_width = Float32(2.0)
        intensity_scale = value === nothing ? Float32(10.0) : Float32(value)
        for i in 1:N_WAVELENGTHS
            spectrum[i] = max(intensity_scale * exp(Float32(-0.5) * ((WAVELENGTHS_NM[i] - peak_wavelength) / peak_width)^2), Float32(0.0))
        end
        
    elseif profile_type == "laser_yellow"
        peak_wavelength = Float32(589.0)
        peak_width = Float32(1.0)
        intensity_scale = value === nothing ? Float32(10.0) : Float32(value)
        for i in 1:N_WAVELENGTHS
            spectrum[i] = max(intensity_scale * exp(Float32(-0.5) * ((WAVELENGTHS_NM[i] - peak_wavelength) / peak_width)^2), Float32(0.0))
        end
        
    elseif profile_type == "laser_cyan"
        peak_wavelength = Float32(488.0)
        peak_width = Float32(1.0)
        intensity_scale = value === nothing ? Float32(10.0) : Float32(value)
        for i in 1:N_WAVELENGTHS
            spectrum[i] = max(intensity_scale * exp(Float32(-0.5) * ((WAVELENGTHS_NM[i] - peak_wavelength) / peak_width)^2), Float32(0.0))
        end
        
    else
        default_fill_value = value === nothing ? Float32(0.5) : Float32(value)
        fill!(spectrum, default_fill_value) 
        @warn "Unknown spectrum profile type: $profile_type. Using flat spectrum with value $default_fill_value instead."
    end
    
    return Spectrum(spectrum) # Convert to SVector before returning
end