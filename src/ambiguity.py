"""
ambiguity.py
============
180° directional ambiguity resolution for SWOT SSH × σ₀ cross-spectra,
and coordinate-frame conversion utilities.

Theory (Ardhuin et al. 2024)
-----------------------------
The imaginary part of the SSH × σ₀ cross-spectrum satisfies:

    Im[ C_SSH,σ₀(k) ]  ∝  k_range · E(k)

where k_range = projection of the wave vector **k** onto the radar
range direction (≈ cross-track for SWOT).

RIGHT swath  (look direction ≈ +cross-track):
    Im(C) > 0  →  positive range-direction group velocity
               →  wave propagates TOWARD the satellite  →  keep candidate
    Im(C) < 0  →  wave propagates AWAY from the satellite  →  flip 180°

No separate along-track / cross-track branch is needed; the sign rule
is universal because Im(C) encodes the full range component of **k**.

References
----------
Ardhuin, F. et al. (2024)  [full citation to be added]
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
    Resolve the 180° directional ambiguity following Ardhuin et al. (2024).

    Parameters
    ----------
    ka_cand, kc_cand : float
        Candidate peak wavenumber components [rad/m] from the ka ≥ 0
        half-plane.  These are ambiguous: the true direction is either
        (ka_cand, kc_cand) or (−ka_cand, −kc_cand).
    C_patch : ndarray, shape (Nwin, Nwin), complex
        Per-patch cross-spectrum  F_ssh · conj(F_σ₀).
    Ka, Kc : ndarray, shape (Nwin, Nwin)
        Along-track and cross-track wavenumber grids [rad/m].
    coh_patch : ndarray, shape (Nwin, Nwin)
        Per-patch coherence (0–1).
    look_side : {'right', 'left'}
        SWOT swath side.  Right swath → range ≈ +cross-track.
    dk_rel : float
        Fractional half-bandwidth around K_peak for the averaging window.
        Default 0.30 (±30 %).
    dtheta_win : float
        Angular half-window in the satellite frame [°].  Default 30°.
    coh_min : float
        Minimum coherence to include a pixel in the Im(C) average.
        Default 0.05.

    Returns
    -------
    ka_true, kc_true : float
        Resolved wavenumber components [rad/m].
    im_mean : float
        Diagnostic: coherence-weighted mean Im(C) used for the decision.
        Positive ↔ wave toward satellite (right swath).
    """
    K_cand = float(np.sqrt(ka_cand**2 + kc_cand**2))
    if K_cand == 0.0:
        return float(ka_cand), float(kc_cand), np.nan

    # Satellite-frame direction of candidate [°, CCW from along-track]
    theta_cand = np.degrees(np.arctan2(kc_cand, ka_cand))

    # Grid quantities
    K_mag      = np.sqrt(Ka**2 + Kc**2)
    theta_grid = np.degrees(np.arctan2(Kc, Ka))

    # Signed angular distance — wraps correctly
    dtheta = ((theta_grid - theta_cand + 180.0) % 360.0) - 180.0

    # ── Averaging window ──────────────────────────────────────────────────────
    mask = (
        np.isfinite(C_patch.imag) &
        (np.abs(K_mag - K_cand) <= dk_rel * K_cand) &
        (np.abs(dtheta)          <= dtheta_win) &
        (coh_patch               >= coh_min)
    )

    if not np.any(mask):
        # Fallback: single grid point closest to candidate, no coherence gate
        idx     = np.unravel_index(
            np.argmin((Ka - ka_cand)**2 + (Kc - kc_cand)**2), Ka.shape)
        im_mean = float(C_patch[idx].imag)
    else:
        # Coherence-weighted mean of Im(C)
        w     = coh_patch[mask]
        w_sum = float(w.sum())
        im_mean = (float(np.dot(C_patch.imag[mask], w) / w_sum)
                   if w_sum > 0 else float(np.mean(C_patch.imag[mask])))

    # ── Ambiguity decision ────────────────────────────────────────────────────
    # Right swath: Im > 0 → toward satellite → keep candidate
    # Left  swath: sign convention flips
    sign = +1 if look_side == "right" else -1

    if sign * im_mean >= 0.0:
        return float(ka_cand),  float(kc_cand),  im_mean   # confirmed
    else:
        return float(-ka_cand), float(-kc_cand), im_mean   # flipped 180°


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
    a  = np.asarray(angles_deg, dtype=float)
    a  = a[np.isfinite(a)]
    n  = len(a)

    if n == 0:
        return np.nan, np.nan, np.nan, 0

    rad   = np.radians(a)
    S     = np.mean(np.sin(rad))
    C     = np.mean(np.cos(rad))
    R     = float(np.sqrt(S**2 + C**2))

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
