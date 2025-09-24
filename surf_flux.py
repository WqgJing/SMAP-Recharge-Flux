import numpy as np


def synth_surface_flux(
    total_days=30,
    dt_minutes=30,
    seed=42,
    et_base_mm_day=3.0,  # mean daily ET (mm/day)
    et_amp_mm_day=1.5,  # diurnal ET amplitude (mm/day peak-to-mean)
    et_noise_frac=0.10,  # ET Gaussian noise as fraction of (base + diurnal)
    storm_rate_per_day=0.4,  # Poisson rate of storm onsets (events/day)
    min_storm_hours=0.5,
    max_storm_hours=6.0,
    min_intensity_mm_hr=3.0,  # hyetograph peak intensity range (mm/hr)
    max_intensity_mm_hr=25.0,
    dry_start_hours=24,  # enforce no rain at beginning (q>0 initially)
    triangular_storm=True,
):
    """
    Returns:
        t (np.ndarray): seconds from 0 to total_days (inclusive)
        q (np.ndarray): surface flux in m/s ( + = upward ET, - = downward rain )
    Notes:
        - ET units are converted from mm/day to m/s.
        - Rainfall intensities are converted from mm/hr to m/s.
    """
    rng = np.random.default_rng(seed)

    # Time grid
    dt_days = dt_minutes / (24 * 60)
    t_days = np.arange(0.0, total_days + 1e-12, dt_days)
    t_sec = t_days * 86400.0

    # --- Evapotranspiration (upward, positive) ---
    # base ET (mm/day) with a diurnal (solar) cycle: peak around local noon each day
    # sin term over 24h; shift so minimum near dawn, maximum mid-day
    diurnal = np.sin(2 * np.pi * t_days)  # one cycle per day
    et_mm_day = et_base_mm_day + et_amp_mm_day * np.clip(
        diurnal, 0, None
    )  # no ET at night

    # add light random variability
    et_noise = et_noise_frac * et_mm_day * rng.normal(size=t_days.size)
    et_mm_day = np.clip(et_mm_day + et_noise, 0.0, None)

    # convert ET to m/s (upward = +)
    et_m_per_s = (et_mm_day / 1000.0) / 86400.0

    # --- Rainfall (downward, negative) via Poisson storms ---
    rain_m_per_s = np.zeros_like(t_days)

    # draw storm onsets as a Poisson process (thinning via rate * total_days)
    expected_events = storm_rate_per_day * total_days
    n_events = rng.poisson(expected_events)

    # candidate onset times (days), uniformly distributed
    onsets = rng.uniform(0, total_days, size=n_events)
    # enforce dry start window
    onsets = onsets[onsets * 24.0 >= dry_start_hours]
    onsets.sort()

    for onset_day in onsets:
        dur_hours = rng.uniform(min_storm_hours, max_storm_hours)
        dur_days = dur_hours / 24.0
        t0 = onset_day
        t1 = onset_day + dur_days

        # peak intensity (mm/hr)
        peak_mm_hr = rng.uniform(min_intensity_mm_hr, max_intensity_mm_hr)

        # build hyetograph over indices covering [t0, t1]
        mask = (t_days >= t0) & (t_days <= t1)
        if not np.any(mask):
            # ensure at least one step receives rain
            idx = np.searchsorted(t_days, t0)
            if 0 <= idx < t_days.size:
                mask = np.zeros_like(t_days, dtype=bool)
                mask[idx] = True

        if triangular_storm:
            # normalized 0→1→0 over storm duration
            tau = (t_days[mask] - t0) / max(1e-12, dur_days)
            shape = np.where(tau <= 0.5, 2 * tau, 2 * (1 - tau))
        else:
            # block storm
            shape = np.ones(np.count_nonzero(mask))

        # convert mm/hr to m/s and apply shape
        intens_m_s = (peak_mm_hr / 1000.0) / 3600.0
        rain_m_per_s[mask] -= intens_m_s * shape  # negative = downward

    # --- Combine ET and Rain ---
    q = et_m_per_s + rain_m_per_s

    # small additive sensor-like noise (e.g., 2% of typical ET magnitude)
    typical_et = (max(1e-12, et_base_mm_day) / 1000.0) / 86400.0
    q += 0.02 * typical_et * rng.normal(size=q.size)

    return t_sec, q


# ---------------- Example usage ----------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t, q = synth_surface_flux(
        total_days=15,
        dt_minutes=30,
        seed=1,
        et_base_mm_day=1,
        et_amp_mm_day=0.0,
        storm_rate_per_day=0.1,
        dry_start_hours=24,
    )

    print(
        f"Generated {q.size} points. q stats (m/s): min={q.min():.2e}, max={q.max():.2e}, mean={q.mean():.2e}"
    )
    # quick check: first value should be > 0 (drought start)
    print(f"q[0] = {q[0]:.2e} m/s")

    # plot (positive up, negative down)
    plt.figure(figsize=(10, 4))
    plt.plot(t / 86400.0, q, lw=1)
    plt.axhline(0, lw=0.8)
    plt.xlabel("Time (days)")
    plt.ylabel("Surface flux q (m/s)\n(+ up = ET, – down = rain)")
    plt.title("Synthetic Surface Flux for PINN Training")
    plt.tight_layout()
    plt.show()
