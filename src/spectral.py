"""
spectral.py
===========
2D Welch cross-spectrum analysis for SWOT HR PIXC data.

Pipeline per Welch patch
------------------------
1.  Bin scatter points onto a regular satellite-frame grid.
2.  Detrend (remove planar mean) and apply a 2D Hann window.
3.  Compute 2D FFT → SSH spectrum E_p and cross-spectrum C_p.
4.  Find the SSH spectral peak in the swell band (ka ≥ 0 half-plane).
5.  Resolve the 180° directional ambiguity via Ardhuin et al. (2024).
6.  Convert to geographic direction and wave period.
7.  Aggregate per-patch estimates → mean ± std ± 95% CI.

All satellite-frame rotation is handled by injected callables
(utm_to_sat_fn, resolve_fn, geo_dir_fn) so this module has no
hard dependency on a specific geometry configuration.
"""

import numpy as np
from numpy.fft import fft2, fftfreq, fftshift
from scipy.signal.windows import hann


# ---------------------------------------------------------------------------
# Grid preparation
# ---------------------------------------------------------------------------

def detrend_plane(Z):
    """
    Remove a least-squares planar fit from a 2-D array.

    Parameters
    ----------
    Z : ndarray, shape (Ny, Nx)   Input field (may contain NaNs only
                                  if they have already been filled).

    Returns
    -------
    Z_dt : ndarray, shape (Ny, Nx)   Detrended field.
    """
    Ny, Nx = Z.shape
    yy, xx = np.mgrid[0:Ny, 0:Nx]
    A      = np.c_[xx.ravel(), yy.ravel(), np.ones(Nx * Ny)]
    beta, *_ = np.linalg.lstsq(A, Z.ravel(), rcond=None)
    return Z - (A @ beta).reshape(Ny, Nx)


def bin_to_sat_grid(x_pts, y_pts, ssh, sig0,
                    cx, cy, half_a, half_c, res,
                    utm_to_sat_fn):
    """
    Bin scattered PIXC points onto a regular satellite-frame grid.

    The box is truly rotated with the satellite track: selection and
    binning are done in the (along-track, cross-track) coordinate system.

    Parameters
    ----------
    x_pts, y_pts  : ndarray   UTM Easting / Northing of all PIXC points [m]
    ssh, sig0     : ndarray   SSH [m] and σ₀ [dB] at each point
    cx, cy        : float     Box centre UTM [m]
    half_a, half_c: float     Half-widths in satellite frame [m]
    res           : float     Grid resolution [m]
    utm_to_sat_fn : callable  utm_to_sat(dx, dy) → (u_along, u_cross)

    Returns
    -------
    ssh_grid  : ndarray (Ny, Nx) or None   Bin-averaged SSH [m]
    sig0_grid : ndarray (Ny, Nx) or None   Bin-averaged σ₀ [dB]
    a_centers : ndarray (Ny,)               Along-track bin centres [m]
    c_centers : ndarray (Nx,)               Cross-track bin centres [m]
    n_pts     : int                         Number of points inside box
    Returns (None, None, None, None, None) if fewer than 10 points.
    """
    u_a_all, u_c_all = utm_to_sat_fn(x_pts - cx, y_pts - cy)
    mask = (np.abs(u_a_all) <= half_a) & (np.abs(u_c_all) <= half_c)

    if mask.sum() < 10:
        return None, None, None, None, None

    u_a = u_a_all[mask];  u_c = u_c_all[mask]

    a_edges = np.arange(-half_a, half_a + res, res)
    c_edges = np.arange(-half_c, half_c + res, res)

    cnt,  _, _ = np.histogram2d(u_a, u_c, bins=[a_edges, c_edges])
    ssh_s, _, _ = np.histogram2d(u_a, u_c, bins=[a_edges, c_edges],
                                  weights=ssh[mask])
    sig_s, _, _ = np.histogram2d(u_a, u_c, bins=[a_edges, c_edges],
                                  weights=sig0[mask])

    v  = cnt > 0
    sg = np.full(cnt.shape, np.nan)
    gg = np.full(cnt.shape, np.nan)
    sg[v] = ssh_s[v] / cnt[v]
    gg[v] = sig_s[v] / cnt[v]

    a_centers = 0.5 * (a_edges[:-1] + a_edges[1:])
    c_centers = 0.5 * (c_edges[:-1] + c_edges[1:])

    return sg, gg, a_centers, c_centers, int(mask.sum())


# ---------------------------------------------------------------------------
# Spectral peak detection
# ---------------------------------------------------------------------------

def peak_in_swell_band(E, Ka, Kc, lam_min, lam_max):
    """
    Find the SSH spectral peak inside the swell band, restricted to
    the ka ≥ 0 half-plane to yield a single candidate before the
    180° ambiguity resolution step.

    Parameters
    ----------
    E            : ndarray (Nwin, Nwin)   SSH power spectrum
    Ka, Kc       : ndarray (Nwin, Nwin)   Wavenumber grids [rad/m]
    lam_min, lam_max : float              Swell band wavelength limits [m]

    Returns
    -------
    K_peak : float   Peak wavenumber magnitude [rad/m]  (nan if none found)
    ka_p   : float   Along-track component [rad/m]
    kc_p   : float   Cross-track component [rad/m]
    dir_p  : float   Ambiguous direction [° CCW from along-track]
    """
    K_mag = np.sqrt(Ka**2 + Kc**2)
    k_min = 2.0 * np.pi / lam_max
    k_max = 2.0 * np.pi / lam_min

    band  = (K_mag >= k_min) & (K_mag <= k_max) & (Ka >= 0)
    if not np.any(band):
        return np.nan, np.nan, np.nan, np.nan

    E_band = np.where(band, E, 0.0)
    idx    = np.unravel_index(np.argmax(E_band), E_band.shape)
    ka_p   = float(Ka[idx]);  kc_p = float(Kc[idx])
    K_p    = float(np.sqrt(ka_p**2 + kc_p**2))
    dir_p  = float(np.degrees(np.arctan2(kc_p, ka_p)))

    return K_p, ka_p, kc_p, dir_p


# ---------------------------------------------------------------------------
# Welch cross-spectrum
# ---------------------------------------------------------------------------

def welch_cross_spectrum(
    ssh_grid,
    sig0_grid,
    res,
    lwin,
    lam_min,
    lam_max,
    resolve_fn,
    geo_dir_fn,
    g=9.81,
    h=None,
    coh_min_patch=0.3,
):
    """
    Compute the Welch-averaged 2D cross-spectrum and per-patch wave
    parameter estimates with Ardhuin 2024 ambiguity resolution.

    Parameters
    ----------
    ssh_grid, sig0_grid : ndarray (Ny, Nx)
        Bin-averaged SSH [m] and σ₀ [dB] on the satellite-frame grid.
    res   : float   Grid resolution [m]
    lwin  : float   Welch window size [m]
    lam_min, lam_max : float   Swell band wavelength limits [m]
    resolve_fn : callable
        resolve_fn(ka_cand, kc_cand, C_patch, Ka, Kc, coh_patch)
        → (ka_true, kc_true, im_mean)
        See ambiguity.resolve_180_ardhuin2024.
    geo_dir_fn : callable
        geo_dir_fn(ka_true, kc_true) → dir_from_deg  [° CW from North]
        See ambiguity.sat_frame_to_geo_north (partially applied).
    g  : float   Gravitational acceleration [m/s²]
    h  : float or None
        Water depth [m] for dispersion calculation.
        If None, wave period is not computed (patch_T will be all NaN).
    coh_min_patch : float
        Minimum coherence at the spectral peak for a patch to contribute
        to the direction estimate.  Default 0.3.

    Returns
    -------
    dict with keys:
        E_ssh      : ndarray (Nwin, Nwin)   Welch-averaged SSH spectrum
        E_sig0     : ndarray (Nwin, Nwin)   Welch-averaged σ₀ spectrum
        C_cross    : ndarray complex (Nwin, Nwin)  Cross-spectrum
        coherence  : ndarray (Nwin, Nwin)   Coherence |C| / sqrt(E_ssh·E_sig0)
        phase_deg  : ndarray (Nwin, Nwin)   Cross-spectrum phase [°]
        E_var_map  : ndarray (Nwin, Nwin)   Std of E_ssh across patches
        k_along    : ndarray (Nwin,)         Along-track wavenumber axis [rad/m]
        k_cross    : ndarray (Nwin,)         Cross-track wavenumber axis [rad/m]
        Ka, Kc     : ndarray (Nwin, Nwin)   Wavenumber grids [rad/m]
        patch_K    : ndarray (n_pat,)        Peak |k| per patch [rad/m]
        patch_T    : ndarray (n_pat,)        Peak period per patch [s]
        patch_dir_sat : ndarray (n_pat,)     Resolved dir, satellite frame [°]
        patch_dir_geo : ndarray (n_pat,)     Resolved dir, geo from N [°]
        n_patches  : int
    Returns None if no valid patches are found.
    """
    ny, nx = ssh_grid.shape
    Nwin   = int(lwin // res)
    step   = Nwin - Nwin // 2   # 50 % overlap

    if Nwin > ny or Nwin > nx:
        raise ValueError(
            f"Welch window ({Nwin} px) exceeds grid size ({ny}×{nx}). "
            f"Reduce LWIN or increase the box size."
        )

    win2d = hann(Nwin)[:, None] * hann(Nwin)[None, :]
    U     = float(np.mean(win2d**2))
    norm  = (res**2 / Nwin**2) / U

    k_along = fftshift(2.0 * np.pi * fftfreq(Nwin, d=res))
    k_cross = fftshift(2.0 * np.pi * fftfreq(Nwin, d=res))
    Ka, Kc  = np.meshgrid(k_along, k_cross, indexing="ij")

    S_ssh = np.zeros((Nwin, Nwin))
    S_sig = np.zeros((Nwin, Nwin))
    S_crs = np.zeros((Nwin, Nwin), dtype=complex)

    patch_K        = []
    patch_T        = []
    patch_dir_sat  = []
    patch_dir_geo  = []
    patch_E        = []

    eps = 1e-12

    for i in range(0, ny - Nwin + 1, step):
        for j in range(0, nx - Nwin + 1, step):
            sp = ssh_grid [i:i+Nwin, j:j+Nwin].copy()
            si = sig0_grid[i:i+Nwin, j:j+Nwin].copy()

            vm = np.isfinite(sp) & np.isfinite(si)
            if vm.mean() < 0.8:
                continue

            # Fill NaNs with patch mean before windowing
            sp[~vm] = float(np.nanmean(sp[vm]))
            si[~vm] = float(np.nanmean(si[vm]))

            sp = detrend_plane(sp) * win2d
            si = detrend_plane(si) * win2d

            Fs = fftshift(fft2(sp))
            Fi = fftshift(fft2(si))

            E_p   = norm * np.abs(Fs)**2
            E_s_p = norm * np.abs(Fi)**2
            C_p   = norm * Fs * np.conj(Fi)

            # Per-patch coherence
            coh_p = np.clip(np.abs(C_p) / (np.sqrt(E_p * E_s_p) + eps), 0, 1)

            S_ssh += np.abs(Fs)**2
            S_sig += np.abs(Fi)**2
            S_crs += Fs * np.conj(Fi)

            # ── Spectral peak ─────────────────────────────────────────────────
            K_p, ka_p, kc_p, _ = peak_in_swell_band(
                E_p, Ka, Kc, lam_min, lam_max)

            if np.isnan(K_p):
                patch_K.append(np.nan);   patch_T.append(np.nan)
                patch_dir_sat.append(np.nan);  patch_dir_geo.append(np.nan)
                patch_E.append(E_p)
                continue

            # Coherence at peak
            idx_peak = np.unravel_index(
                np.argmin((Ka - ka_p)**2 + (Kc - kc_p)**2), Ka.shape)
            coh_peak = float(coh_p[idx_peak])

            # ── Ardhuin 2024 ambiguity resolution ─────────────────────────────
            ka_true, kc_true, _ = resolve_fn(
                ka_p, kc_p, C_p, Ka, Kc, coh_p)

            dir_sat = float(np.degrees(np.arctan2(kc_true, ka_true)))
            dir_geo = float(geo_dir_fn(ka_true, kc_true))

            # ── Wave period ───────────────────────────────────────────────────
            if h is not None and np.isfinite(h) and h > 0:
                omega_p = float(np.sqrt(g * K_p * np.tanh(K_p * h)))
                T_p     = 2.0 * np.pi / omega_p
            else:
                T_p = np.nan

            patch_K.append(K_p)
            patch_T.append(T_p)
            patch_E.append(E_p)

            if coh_peak >= coh_min_patch:
                patch_dir_sat.append(dir_sat)
                patch_dir_geo.append(dir_geo)
            else:
                patch_dir_sat.append(np.nan)
                patch_dir_geo.append(np.nan)

    n = len(patch_K)
    if n == 0:
        return None

    E_ssh   = norm * S_ssh / n
    E_sig0  = norm * S_sig / n
    C_cross = norm * S_crs / n

    coh_avg   = np.clip(np.abs(C_cross) / (np.sqrt(E_ssh * E_sig0) + eps), 0, 1)
    phase_deg = np.degrees(np.angle(C_cross))

    patch_E_stack = np.stack(patch_E, axis=0)
    E_var_map     = np.nanstd(patch_E_stack, axis=0, ddof=1)

    return {
        "E_ssh"        : E_ssh,
        "E_sig0"       : E_sig0,
        "C_cross"      : C_cross,
        "coherence"    : coh_avg,
        "phase_deg"    : phase_deg,
        "E_var_map"    : E_var_map,
        "k_along"      : k_along,
        "k_cross"      : k_cross,
        "Ka"           : Ka,
        "Kc"           : Kc,
        "patch_K"      : np.array(patch_K),
        "patch_T"      : np.array(patch_T),
        "patch_dir_sat": np.array(patch_dir_sat),
        "patch_dir_geo": np.array(patch_dir_geo),
        "n_patches"    : n,
    }
