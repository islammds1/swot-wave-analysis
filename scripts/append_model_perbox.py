"""
scripts/append_model_perbox.py
================================
Reads the SWOT results Excel, extracts NWShelf model VPED and VTPK
spatially averaged over each rotated box footprint, and saves a new
Excel file with two extra columns appended at the end.

All original columns are preserved exactly as-is.

New columns added
-----------------
  model_VTPK_s  : model spectral peak period  [s]  (mean over box)
  model_VPED_d  : model energy-weighted period [s]  (mean over box)

Usage
-----
  python scripts/append_model_perbox.py
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

# ============================================================
# 0) PATHS & SETTINGS
# ============================================================

EXCEL_FILE = r"D:/PhD/SWOT_L2_HR_PIXC/Pass016/swot_wave_results_20230405_cycle481_pass016.xlsx"
MODEL_FILE = r"D:/PhD/Data/nwshelf/model_mathis.nc"
MODEL_TIME = "2023-04-05 09:00:00"

# Model variable names — update if yours differ
VAR_VTPK = "VTPK"
VAR_VPED = "VPED"
LON_NAME = "longitude"
LAT_NAME = "latitude"

# Output file (saved alongside the input)
OUT_FILE = EXCEL_FILE.replace(".xlsx", "_+model.xlsx")

# ============================================================
# 1) LOAD SWOT EXCEL  (keep everything as-is)
# ============================================================

df = pd.read_excel(EXCEL_FILE)
print(f"Loaded: {os.path.basename(EXCEL_FILE)}  →  {len(df)} boxes")

# ============================================================
# 2) SATELLITE GEOMETRY FROM EXCEL
# ============================================================

track_deg = float(df["track_angle_deg"].iloc[0])
track_rad = np.radians(track_deg)
Vsat_E    = np.sin(track_rad)
Vsat_N    = np.cos(track_rad)


def utm_to_sat(dx, dy):
    """UTM offset → satellite frame (along-track, cross-track)."""
    return dx * Vsat_E + dy * Vsat_N,  dx * Vsat_N - dy * Vsat_E


# ============================================================
# 3) LOAD MODEL  →  select time  →  convert to UTM 30N
# ============================================================

print(f"Loading model …")
ds = xr.open_dataset(MODEL_FILE)

if "time" in ds.dims or "time" in ds.coords:
    ds_t = ds.sel(time=MODEL_TIME, method="nearest")
    print(f"  Time selected: {str(ds_t['time'].values)[:19]}")
else:
    ds_t = ds

df_model = ds_t[[VAR_VTPK, VAR_VPED]].to_dataframe().reset_index()
df_model = df_model.dropna(subset=[VAR_VTPK, VAR_VPED])
print(f"  Valid model grid points: {len(df_model):,}")

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)
df_model["X"], df_model["Y"] = transformer.transform(
    df_model[LON_NAME].values,
    df_model[LAT_NAME].values,
)

# ============================================================
# 4) EXTRACT MODEL VALUES PER BOX
# ============================================================

vtpk_vals = []
vped_vals = []

for _, row in df.iterrows():
    cx     = float(row["cx_utm_m"])
    cy     = float(row["cy_utm_m"])
    half_a = float(row["half_along_m"])
    half_c = float(row["half_cross_m"])
    lbl    = row["box_id"]

    u_a, u_c = utm_to_sat(
        df_model["X"].values - cx,
        df_model["Y"].values - cy,
    )
    inside = (np.abs(u_a) <= half_a) & (np.abs(u_c) <= half_c)

    if inside.sum() == 0:
        print(f"  {lbl}: ⚠ no model points inside box → NaN")
        vtpk_vals.append(np.nan)
        vped_vals.append(np.nan)
        continue

    vtpk = float(df_model.loc[inside, VAR_VTPK].max())
    vped = float(df_model.loc[inside, VAR_VPED].mean())

    print(f"  {lbl}: {inside.sum()} pts  |  "
          f"VTPK = {vtpk:.3f} s  |  VPED = {vped:.3f} degN")

    vtpk_vals.append(round(vtpk, 4))
    vped_vals.append(round(vped, 4))

# ============================================================
# 5) APPEND COLUMNS AND SAVE
# ============================================================

df["model_VTPK_s"] = vtpk_vals
df["model_VPED_degN"] = vped_vals

with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="SWOT_wave_results")
    ws = writer.sheets["SWOT_wave_results"]
    for col_cells in ws.columns:
        max_len = max(
            (len(str(c.value)) for c in col_cells if c.value is not None),
            default=10,
        )
        ws.column_dimensions[col_cells[0].column_letter].width = max_len + 3

print(f"\n✓ Saved: {OUT_FILE}")
print(f"  Columns: {list(df.columns)}")
print("\nFinal table:")
print(df[["box_id", "period_s", "model_VTPK_s", "model_VPED_degN"]].to_string(index=False))