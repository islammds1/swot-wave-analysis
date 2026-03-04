# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:58:29 2026

@author: islammds
"""
"""
swot_geometry.py
================
Reusable tool to extract SWOT satellite geometry from a L3 LR NetCDF file.

Usage
-----
    from swot_geometry import get_swot_geometry

    geo = get_swot_geometry("your_file.nc", swath="right")

    geo["trackangle_deg"]       # float
    geo["pass_direction"]       # 'ascending' or 'descending'
    geo["Vsat_EN"]              # np.array([East, North])
    geo["look_angle_deg"]       # 1-D array, one value per swath pixel
    geo["xtrk_km"]              # 1-D array, cross-track distances [km]
    geo["pixel_indices"]        # column indices in the full pixel array
"""

import numpy as np
import xarray as xr


def get_swot_geometry(filepath, swath="right", alt_km=890.0):
    """
    Compute SWOT satellite geometry from a L3 LR SSH NetCDF file.

    Parameters
    ----------
    filepath : str
        Path to the SWOT L3 LR .nc file.
    swath : str
        "right"  → positive cross_track_distance (starboard)
        "left"   → negative cross_track_distance (port)
        "both"   → returns signed look angle over all pixels
    alt_km : float
        Satellite altitude in km (default 890 km for SWOT).

    Returns
    -------
    dict with keys:
        trackangle_deg   : float   — satellite heading, degrees from East CCW
        pass_direction   : str     — 'ascending' or 'descending'
        Vsat_EN          : ndarray — [East, North] unit vector
        look_angle_deg   : ndarray — per-pixel look angle (degrees)
                           right swath: positive values
                           left  swath: positive values (magnitude only)
                           both        : signed (neg=left, pos=right, nan=gap)
        xtrk_km          : ndarray — cross-track distances for selected pixels [km]
        pixel_indices    : ndarray — column indices in the full pixel dimension
    """

    # ── Load ─────────────────────────────────────────────────────────────────
    ds   = xr.open_dataset(filepath)
    lat  = ds["latitude"].values               # (num_lines, num_pixels)
    lon  = ds["longitude"].values
    xtrk = ds["cross_track_distance"].values   # (num_pixels,)

    # ── Swath mask ────────────────────────────────────────────────────────────
    if swath == "right":
        mask = xtrk > 0
    elif swath == "left":
        mask = xtrk < 0
    elif swath == "both":
        mask = ~np.isnan(xtrk)
    else:
        raise ValueError("swath must be 'right', 'left', or 'both'")

    pixel_idx  = np.where(mask)[0]
    xtrk_sel   = xtrk[mask]

    if len(pixel_idx) == 0:
        raise ValueError(f"No pixels found for swath='{swath}'. "
                         f"xtrk range: {np.nanmin(xtrk):.2f} to {np.nanmax(xtrk):.2f}")

    # ── Track angle: use innermost pixel of selected swath ───────────────────
    indxc = pixel_idx[np.argmin(np.abs(xtrk_sel))]

    j1, j2 = 10, lat.shape[0] - 10
    dlat    = lat[j2, indxc] - lat[j1, indxc]
    dlon    = lon[j2, indxc] - lon[j1, indxc]
    midlat  = 0.5 * (lat[j1, indxc] + lat[j2, indxc])

    trackangle = -90.0 - np.degrees(np.arctan2(dlat, dlon * np.cos(np.radians(midlat))))
    trackangle = (trackangle + 180) % 360 - 180

    pass_direction = "ascending" if dlat > 0 else "descending"

    # ── Vsat unit vector ──────────────────────────────────────────────────────
    heading_from_north = 90.0 - trackangle
    Vsat = np.array([
        np.sin(np.radians(heading_from_north)),   # East
        np.cos(np.radians(heading_from_north)),   # North
    ])

    # ── Look angle ────────────────────────────────────────────────────────────
    if swath == "both":
        look_angle = np.full(xtrk.shape, np.nan)
        left_m  = xtrk < 0
        right_m = xtrk > 0
        look_angle[left_m]  = -np.degrees(np.arctan(np.abs(xtrk[left_m])  / alt_km))
        look_angle[right_m] =  np.degrees(np.arctan(np.abs(xtrk[right_m]) / alt_km))
        look_angle_out = look_angle   # full signed array
    else:
        look_angle_out = np.degrees(np.arctan(np.abs(xtrk_sel) / alt_km))

    return {
        "trackangle_deg" : float(trackangle),
        "pass_direction" : pass_direction,
        "Vsat_EN"        : Vsat,
        "look_angle_deg" : look_angle_out,
        "xtrk_km"        : xtrk_sel,
        "pixel_indices"  : pixel_idx,
    }


# ── Quick test when run directly ─────────────────────────────────────────────
if __name__ == "__main__":
    FILE = 'D:/PhD/Data/SWOT_L3_LR_SSH_Unsmoothed_481_016_20230405T094254_20230405T103359_v2.0.1.nc'

    geo = get_swot_geometry(FILE, swath="right")

    print(f"Track angle    : {geo['trackangle_deg']:.2f} deg")
    print(f"Pass direction : {geo['pass_direction']}")
    print(f"Vsat [E, N]    : {geo['Vsat_EN']}")
    print(f"Look angle     : {geo['look_angle_deg'].min():.2f} – {geo['look_angle_deg'].max():.2f} deg")
    print(f"xtrk range     : {geo['xtrk_km'].min():.2f} – {geo['xtrk_km'].max():.2f} km")
