"""
bathymetry.py
=============
Utilities for extracting mean water depth inside a track-aligned
SWOT analysis box from a CF-convention bathymetry NetCDF file.

The box is defined in UTM coordinates (centre + half-widths in the
satellite frame), so depth is averaged over the *rotated* footprint —
exactly the same region used for the PIXC spectral analysis.

Dependencies
------------
numpy, xarray, pyproj
"""

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mean_depth_for_box(
    cx, cy, half_a, half_c,
    elevation,
    utm_to_sat_fn,
    proj,
    lat_name="lat",
    lon_name="lon",
    lon_buffer_deg=0.02,
    lat_buffer_deg=0.02,
):
    """
    Return the mean water depth [m, positive downward] inside a
    track-aligned UTM box.

    Parameters
    ----------
    cx, cy          : float
        Box centre in UTM [m]  (Easting, Northing).
    half_a, half_c  : float
        Half-widths of the box in the satellite frame [m]
        (along-track and cross-track respectively).
    elevation       : xarray.DataArray
        2-D elevation array with lat/lon dimensions (negative = ocean).
    utm_to_sat_fn   : callable
        Function  utm_to_sat(dx, dy) → (u_along, u_cross)
        that rotates UTM offsets into the satellite frame.
        Typically the closure defined in run_analysis.py using Vsat_E/N.
    proj            : pyproj.Proj
        UTM projection object used to forward/inverse-project coordinates.
    lat_name        : str   Name of the latitude  dimension (default 'lat').
    lon_name        : str   Name of the longitude dimension (default 'lon').
    lon_buffer_deg  : float Extra lon margin around the bounding box [°].
    lat_buffer_deg  : float Extra lat margin around the bounding box [°].

    Returns
    -------
    h_mean : float
        Mean water depth inside the box [m, positive].
        Returns np.nan if no valid ocean pixels are found.

    Notes
    -----
    Algorithm
    ~~~~~~~~~
    1. Convert all 4 UTM corners of the rotated box to lat/lon.
    2. Subset the elevation DataArray to the bounding envelope
       (with a small buffer to avoid edge effects).
    3. Back-project every grid point to UTM and rotate into the
       satellite frame.
    4. Keep only pixels inside |u_a| ≤ half_a  AND  |u_c| ≤ half_c
       AND elevation ≤ 0 (ocean only).
    5. Return nanmean(-elevation) as depth.
    """
    # ── 1. UTM corners → lat/lon bounding envelope ───────────────────────────
    corners_utm = _box_corners_utm(cx, cy, half_a, half_c, utm_to_sat_fn)
    lons_c, lats_c = proj(corners_utm[:, 0], corners_utm[:, 1], inverse=True)

    lon_min = lons_c.min() - lon_buffer_deg
    lon_max = lons_c.max() + lon_buffer_deg
    lat_min = lats_c.min() - lat_buffer_deg
    lat_max = lats_c.max() + lat_buffer_deg

    # ── 2. Subset bathymetry ──────────────────────────────────────────────────
    lat_coord     = elevation[lat_name].values
    lat_increases = bool(np.diff(lat_coord).mean() > 0)
    lat_sl        = (slice(lat_min, lat_max) if lat_increases
                     else slice(lat_max, lat_min))

    sub = elevation.sel(**{lat_name: lat_sl,
                           lon_name: slice(lon_min, lon_max)})

    if sub.size == 0:
        return np.nan

    elev_vals = sub.values                       # shape (n_lat, n_lon)
    lats_1d   = sub[lat_name].values
    lons_1d   = sub[lon_name].values
    LON, LAT  = np.meshgrid(lons_1d, lats_1d)   # (n_lat, n_lon)

    # ── 3. Convert grid → UTM → satellite frame ───────────────────────────────
    X_grid, Y_grid = proj(LON.ravel(), LAT.ravel())
    u_a, u_c       = utm_to_sat_fn(X_grid - cx, Y_grid - cy)

    # ── 4. Mask: inside rotated box AND ocean pixel ───────────────────────────
    inside = (
        (np.abs(u_a) <= half_a) &
        (np.abs(u_c) <= half_c) &
        np.isfinite(elev_vals.ravel()) &
        (elev_vals.ravel() <= 0)
    )

    if not np.any(inside):
        return np.nan

    # ── 5. Mean depth ─────────────────────────────────────────────────────────
    return float(np.nanmean(-elev_vals.ravel()[inside]))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _box_corners_utm(cx, cy, half_a, half_c, utm_to_sat_fn):
    """
    Compute the 4 UTM corners of a track-aligned box.

    The box is defined in the satellite frame (along-track × cross-track);
    corners are rotated back to UTM using the inverse of utm_to_sat_fn.

    Parameters
    ----------
    cx, cy          : float   Box centre UTM [m]
    half_a, half_c  : float   Half-widths [m]
    utm_to_sat_fn   : callable  utm_to_sat(dx, dy) → (u_a, u_c)

    Returns
    -------
    corners : ndarray, shape (4, 2)   UTM (Easting, Northing) of each corner
    """
    # Satellite-frame corner offsets
    sa = np.array([-half_a,  half_a,  half_a, -half_a])
    sc = np.array([-half_c, -half_c,  half_c,  half_c])

    # Inverse rotation: sat-frame → UTM offset
    # utm_to_sat is an orthogonal rotation, so its inverse = transpose
    # if utm_to_sat(dx, dy) = (dx·Ve + dy·Vn,  dx·Vn - dy·Ve)
    # then sat_to_utm(ua, uc) = (ua·Ve + uc·Vn,  ua·Vn - uc·Ve)  [same form]
    # We derive Ve, Vn by evaluating the function on unit vectors:
    Ve, Vn = _extract_vsat(utm_to_sat_fn)
    dx = sa * Ve + sc * Vn
    dy = sa * Vn - sc * Ve

    return np.column_stack([cx + dx, cy + dy])


def _extract_vsat(utm_to_sat_fn):
    """
    Recover (Vsat_E, Vsat_N) from the utm_to_sat closure by probing it
    with the canonical basis vectors.
    """
    ua_e, _ = utm_to_sat_fn(np.array([1.0]), np.array([0.0]))
    uc_e, _ = utm_to_sat_fn(np.array([0.0]), np.array([1.0]))
    # utm_to_sat(1, 0) = (Ve, Vn)  for the along-track component
    return float(ua_e[0]), float(uc_e[0])
