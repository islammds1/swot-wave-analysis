"""
ambiguity.py
============
180° directional ambiguity resolution for SWOT SSH × σ₀ cross-spectra,
and coordinate-frame conversion utilities.

Theory
------
The 2D power spectrum E(kx, ky) is symmetric about the origin, so a
spectral peak at (ka, kc) is indistinguishable from one at (−ka, −kc)
on amplitude alone.  The SSH–σ₀ cross-spectrum C = F_ssh · conj(F_σ₀)
carries two physically distinct phase signals that together resolve this
ambiguity for any propagation direction:

1. Tilt modulation (cross-track / range component)
   ------------------------------------------------
   The radar-facing slope of each wave crest modulates σ₀.  For waves
   with a cross-track wavenumber component kc, this produces a
   cross-spectrum with phase ≈ ±90°, so

       Im(C) ∝ kc · E(k)

   RIGHT swath (look ≈ +cross-track):  Im(C) > 0 → wave toward radar
                                        Im(C) < 0 → wave away  → flip

   This signal is strong when |kc| is large (near-cross-track swell)
   and vanishes for purely along-track propagation.

2. Velocity bunching (along-track component)
   ------------------------------------------
   KaRIN Doppler beam-forming displaces ocean features along the
   satellite track in proportion to their line-of-sight velocity.
   This imprints σ₀ oscillations that are OUT OF PHASE with SSH when
   the wave propagates in the satellite flight direction (phase ≈ 180°),
   and IN PHASE when the wave opposes the satellite (phase ≈ 0°):

       Re(C) > 0 → wave opposes satellite (ka < 0)
       Re(C) < 0 → wave same as satellite (ka > 0)

   Equivalently, the VB disambiguation signal is −sign(ka) · Re(C),
   which is positive when the candidate direction is confirmed.

   This signal is strong when |ka| is large (near-along-track swell)
   and vanishes for purely cross-track propagation.

Combined signal
---------------
The two signals are weighted by the propagation direction of the
candidate peak so that each contributes in proportion to its reliability:

    w_tilt = |kc_cand| / K_cand   (cross-track fraction)
    w_vb   = |ka_cand| / K_cand   (along-track fraction)

    D = w_tilt · (sign_tilt · Im_avg)
      + w_vb   · (−sign(ka_cand) · Re_avg)

    D ≥ 0  →  keep candidate (ka_cand, kc_cand)
    D < 0  →  flip to (−ka_cand, −kc_cand)

This formulation reduces to the pure tilt rule for cross-track waves
and to the pure VB rule (following Ardhuin et al. 2024) for along-track
waves, with a smooth directional blend in between.

References
----------
Ardhuin, F. et al. (2024). Phase-resolved swells across ocean basins in
SWOT altimetry data. Geophysical Research Letters, 51, e2024GL109658.
https://doi.org/10.1029/2024GL109658
"""

import numpy as np


# ---------------------------------------------------------------------------
# 180° ambiguity resolution
# ---------------------------------------------------------------------------

def resolve_180_ardhuin2024(
    ka_cand,
    kc_cand,
    C_patch,
    Ka,
    Kc,
    coh_patch,
    look_side="right",
    dk_rel=0.30,
    dtheta_win=30.0,
    coh_min=0.05,
):
    """
    Resolve the 180° directional ambiguity using both tilt modulation
    (Im(C)) and velocity bunching (Re(C)) signals from the SSH–σ₀
    cross-spectrum, following the physical framework of Ardhuin et al. (2024).

    The two signals are weighted by the candidate wave's propagation
    direction so that the method is reliable for both cross-track and
    along-track propagating waves:

        D = w_tilt · (sign_tilt · Im_avg)
          + w_vb   · (−sign(ka_cand) · Re_avg)

    where w_tilt = |kc_cand|/K and w_vb = |ka_cand|/K.

    Parameters
    ----------
    ka_cand, kc_cand : float
        Candidate peak wavenumber components [rad/m] from the ka ≥ 0
        half-plane.  The ambiguous alternative is (−ka_cand, −kc_cand).
    C_patch : ndarray, shape (Nwin, Nwin), complex
        Per-patch SSH–σ₀ cross-spectrum  F_ssh · conj(F_σ₀).
    Ka, Kc : ndarray, shape (Nwin, Nwin)
        Along-track and cross-track wavenumber grids [rad/m].
    coh_patch : ndarray, shape (Nwin, Nwin)
        Per-patch SSH–σ₀ coherence (0–1).
    look_side : {'right', 'left'}
        SWOT swath side.  Right swath → radar look ≈ +cross-track.
    dk_rel : float
        Fractional half-bandwidth around K_peak for the spectral
        averaging window.  Default 0.30 (±30 %).
    dtheta_win : float
        Angular half-window around the candidate direction [°].
        Default 30°.
    coh_min : float
        Minimum coherence threshold to include a pixel in the average.
        Default 0.05.

    Returns
    -------
    ka_true, kc_true : float
        Resolved wavenumber components [rad/m].
    D : float
        Combined disambiguation score.  D > 0 confirms the input
        candidate; D < 0 means the direction was flipped.
    im_avg : float
        Diagnostic: coherence-weighted mean Im(C) (tilt signal).
    re_avg : float
        Diagnostic: coherence-weighted mean Re(C) (VB signal).
    """
    K_cand = float(np.sqrt(ka_cand**2 + kc_cand**2))
    if K_cand == 0.0:
        return float(ka_cand), float(kc_cand), np.nan, np.nan, np.nan

    # ── Directional weights (unit-normalised) ─────────────────────────────────
    # w_tilt: cross-track fraction → tilt modulation reliability
    # w_vb  : along-track fraction → velocity bunching reliability
    w_tilt = float(abs(kc_cand) / K_cand)
    w_vb   = float(abs(ka_cand) / K_cand)

    # ── Spectral averaging window ─────────────────────────────────────────────
    K_mag      = np.sqrt(Ka**2 + Kc**2)
    theta_cand = np.degrees(np.arctan2(kc_cand, ka_cand))
    theta_grid = np.degrees(np.arctan2(Kc, Ka))
    dtheta     = ((theta_grid - theta_cand + 180.0) % 360.0) - 180.0

    mask = (
        np.isfinite(C_patch.real) &
        np.isfinite(C_patch.imag) &
        (np.abs(K_mag - K_cand) <= dk_rel * K_cand) &
        (np.abs(dtheta)          <= dtheta_win) &
        (coh_patch               >= coh_min)
    )

    if not np.any(mask):
        # Fallback: nearest grid point, no coherence gate
        idx = np.unravel_index(
            np.argmin((Ka - ka_cand)**2 + (Kc - kc_cand)**2), Ka.shape)
        im_avg = float(C_patch[idx].imag)
        re_avg = float(C_patch[idx].real)
    else:
        # Coherence-weighted means of Im(C) and Re(C)
        w     = coh_patch[mask]
        w_sum = float(w.sum())
        if w_sum > 0:
            im_avg = float(np.dot(C_patch.imag[mask], w) / w_sum)
            re_avg = float(np.dot(C_patch.real[mask], w) / w_sum)
        else:
            im_avg = float(np.mean(C_patch.imag[mask]))
            re_avg = float(np.mean(C_patch.real[mask]))

    # ── Tilt modulation signal ────────────────────────────────────────────────
    # Right swath: look ≈ +kc, so Im(C) > 0 confirms kc_cand > 0
    # Left  swath: look ≈ −kc, sign flips
    sign_tilt  = +1.0 if look_side == "right" else -1.0
    tilt_signal = sign_tilt * im_avg

    # ── Velocity bunching signal ──────────────────────────────────────────────
    # Waves opposing satellite (ka < 0): Re(C) > 0  → confirms candidate
    # Waves same as satellite  (ka > 0): Re(C) < 0  → also confirms candidate
    # So the confirmatory VB signal is −sign(ka_cand) · Re(C)
    # Guard against ka_cand == 0 (pure cross-track: w_vb = 0 anyway)
    sign_ka  = float(np.sign(ka_cand)) if ka_cand != 0.0 else 1.0
    vb_signal = -sign_ka * re_avg

    # ── Combined disambiguation score ─────────────────────────────────────────
    D = w_tilt * tilt_signal + w_vb * vb_signal

    if D >= 0.0:
        return float(ka_cand),  float(kc_cand),  D, im_avg, re_avg
    else:
        return float(-ka_cand), float(-kc_cand), D, im_avg, re_avg


# ---------------------------------------------------------------------------
# Coordinate-frame conversion
# ---------------------------------------------------------------------------

def sat_frame_to_geo_north(ka, kc, Vsat_E, Vsat_N):
    """
    Convert a resolved satellite-frame wave vector (ka, kc) to the
    meteorological "coming FROM" direction in degrees clockwise from
    True North.

    Satellite frame → geographic East/North
    ----------------------------------------
        k_E = ka · Vsat_E + kc · Vsat_N
        k_N = ka · Vsat_N − kc · Vsat_E

    where (Vsat_E, Vsat_N) is the unit along-track velocity vector.

    dir_to   = atan2(k_E, k_N)        [° CW from North, propagation TO]
    dir_from = (dir_to + 180) % 360   [meteorological convention]

    Parameters
    ----------
    ka, kc        : float   Along-track and cross-track wavenumber [rad/m]
    Vsat_E, Vsat_N: float   Unit along-track velocity components (East, North)

    Returns
    -------
    dir_from : float   Direction wave is coming FROM [° CW from True North]
    """
    k_E      = ka * Vsat_E + kc * Vsat_N
    k_N      = ka * Vsat_N - kc * Vsat_E
    dir_to   = float(np.degrees(np.arctan2(k_E, k_N)))
    dir_from = (dir_to + 180.0) % 360.0
    return dir_from


# ---------------------------------------------------------------------------
# Circular statistics for direction uncertainty
# ---------------------------------------------------------------------------

def circular_stats(angles_deg):
    """
    Compute the circular mean, circular standard deviation, and
    mean resultant length R for an array of directions.

    Parameters
    ----------
    angles_deg : array-like   Directions [°]  (any convention)

    Returns
    -------
    mean_deg : float   Circular mean [°]
    std_deg  : float   Circular std (Fisher 1993) [°]
    R        : float   Mean resultant length ∈ [0, 1]
                       1 = perfectly clustered, 0 = uniformly scattered
    n        : int     Number of finite samples used
    """
    a = np.asarray(angles_deg, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)

    if n == 0:
        return np.nan, np.nan, np.nan, 0

    rad      = np.radians(a)
    S        = np.mean(np.sin(rad))
    C        = np.mean(np.cos(rad))
    R        = float(np.sqrt(S**2 + C**2))
    mean_deg = float(np.degrees(np.arctan2(S, C)) % 360.0)
    std_deg  = float(np.degrees(np.sqrt(-2.0 * np.log(R)))) if R < 1.0 else 0.0

    return mean_deg, std_deg, R, n


def ci95_circular(std_deg, n):
    """
    95 % confidence interval half-width on the circular mean.

    ci95 = 1.96 · std / √n

    Parameters
    ----------
    std_deg : float   Circular standard deviation [°]
    n       : int     Sample size

    Returns
    -------
    ci95 : float   [°]
    """
    if n < 2 or not np.isfinite(std_deg):
        return np.nan
    return 1.96 * std_deg / np.sqrt(n)