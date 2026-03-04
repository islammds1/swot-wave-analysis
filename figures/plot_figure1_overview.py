"""
figures/plot_figure1_overview.py
=================================
Figure 1 — Study area overview map for the SWOT HR PIXC wave analysis.

Layout
------
  Row 0 (top)   : English Channel bathymetry + SWOT Cal/Val HR tile outlines
                  + satellite basemap (Esri World Imagery)
  Row 1 (bottom): Two PIXC sub-panels
                    Left  → Tile 067R  (SSH, buoy 6200059)
                    Right → Tile 065R  (SSH, buoys 6201005 & 6201007)
  Shared horizontal colorbar below both bottom panels.

Output
------
  Figure_1_overview.jpg  (600 dpi, saved to OUT_PATH)

Dependencies
------------
  xarray, geopandas, matplotlib, numpy, contextily,
  cartopy, mpl_toolkits, pandas, fiona
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import xarray as xr
import geopandas as gpd
import contextily as ctx
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable
import fiona

# ============================================================
# 0) PATHS  — edit these or set SWOT_DATA_ROOT env variable
# ============================================================

_ROOT      = os.environ.get("SWOT_DATA_ROOT", "D:/PhD")
_PIXC_ROOT = os.path.join(_ROOT, "SWOT_L2_HR_PIXC/Pass016")
_DATA_ROOT = os.path.join(_ROOT, "Data")

BATHY_FILE   = os.path.join(_DATA_ROOT, "Bathymetry/E4_2024.nc")
CALVAL_KML   = os.path.join(_ROOT, "Data/Passes and In situs/swot_calval_hr_Dec2022-v07_perPass.kml")

PIXC_067R    = os.path.join(_PIXC_ROOT,
    "SWOT_L2_HR_PIXC_481_016_067R_20230405T095353_20230405T095402_PGC0_01.nc")
PIXC_065R    = os.path.join(_PIXC_ROOT,
    "SWOT_L2_HR_PIXC_481_016_065R_20230405T095331_20230405T095342_PGC0_01.nc")

OUT_PATH     = os.path.join(
    _ROOT,
    "Manuscript_GRL_2025/Wave_spectrum_literature/Review/Figure_1_overview.jpg"
)

# ============================================================
# 1) GLOBAL FONT SETTINGS
# ============================================================

matplotlib.rcParams.update({
    "font.family"      : "Times New Roman",
    "font.size"        : 16,
    "font.weight"      : "bold",
    "axes.labelweight" : "bold",
    "axes.titleweight" : "bold",
    "legend.fontsize"  : 13,
    "xtick.labelsize"  : 14,
    "ytick.labelsize"  : 14,
})

# ============================================================
# 2) BUOY DATA
# ============================================================

buoy_data = {
    "Buoy_code": ["6201005", "6201006", "6201007", "6201008",
                  "6201009", "6200290", "6200059"],
    "Lon": [-2.75034, -2.5229962, -2.413325, -1.839827,
            -1.6154932, -1.719333, -1.620],
    "Lat": [50.693493, 50.602486, 50.62299, 50.711323,
            50.711823, 50.63333, 49.695],
}
gdf_buoys = gpd.GeoDataFrame(
    buoy_data,
    geometry=gpd.points_from_xy(buoy_data["Lon"], buoy_data["Lat"]),
    crs="EPSG:4326",
)

# Buoys shown in each bottom sub-panel
BUOYS_067R = {
    "6200059": {"lon": -1.620,    "lat": 49.695,    "color": "yellow"},
}
BUOYS_065R = {
    "6201005": {"lon": -2.75034,  "lat": 50.693493, "color": "red"},
    "6201007": {"lon": -2.413325, "lat": 50.62299,  "color": "red"},
}

# ============================================================
# 3) LOAD BATHYMETRY
# ============================================================

print("Loading bathymetry …")
ds_bathy  = xr.open_dataset(BATHY_FILE)
z         = ds_bathy["elevation"].values[::-1, :]   # flip latitude axis
lat_bathy = ds_bathy["lat"].values
lon_bathy = ds_bathy["lon"].values

# Downsample for faster rendering
FACTOR       = 2
z_down       = z[::FACTOR, ::FACTOR]
lon_down     = lon_bathy[::FACTOR]
lat_down     = lat_bathy[::-1][::FACTOR]
lon_grid, lat_grid = np.meshgrid(lon_down, lat_down)

# ============================================================
# 4) LOAD SWOT CAL/VAL HR TILE OUTLINES  (Pass 16)
# ============================================================

print("Loading Cal/Val HR tile outlines …")
calval_pass    = "16"
calval_layers  = fiona.listlayers(CALVAL_KML)
matching       = [
    lyr for lyr in calval_layers
    if lyr.startswith("Segment") and f"Pass {calval_pass}" in lyr
]

calval_hr_tiles = None
if matching:
    parts = []
    for lyr in matching:
        gdf_tmp = gpd.read_file(CALVAL_KML, driver="KML", layer=lyr)
        gdf_tmp["pass_id"] = calval_pass
        parts.append(gdf_tmp)
    calval_hr_tiles = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), crs="EPSG:4326"
    )
    print(f"  Loaded {len(calval_hr_tiles)} Cal/Val HR tiles for Pass {calval_pass}")
else:
    print("  No matching Cal/Val HR layers found.")

# ============================================================
# 5) LOAD PIXC DATA
# ============================================================

def load_pixc(path):
    """
    Load SWOT HR PIXC pixel cloud, returning only open-ocean pixels
    (ancillary_surface_classification_flag == 0).

    Returns
    -------
    lon, lat, height : ndarray  (1-D, masked to valid ocean pixels)
    """
    print(f"  Loading PIXC: {os.path.basename(path)} …")
    ds   = xr.open_mfdataset(path, group="pixel_cloud", engine="h5netcdf")
    lat  = ds["latitude"].values.flatten()
    lon  = ds["longitude"].values.flatten()
    h    = ds["height"].values.flatten()
    flag = ds["ancillary_surface_classification_flag"].values.flatten()
    mask = (flag == 0) & np.isfinite(lat) & np.isfinite(lon) & np.isfinite(h)
    print(f"    {mask.sum():,} valid pixels")
    return lon[mask], lat[mask], h[mask]


print("Loading PIXC tiles …")
lon_067, lat_067, h_067 = load_pixc(PIXC_067R)
lon_065, lat_065, h_065 = load_pixc(PIXC_065R)

# ============================================================
# 6) HELPER — plot buoys on a Cartopy axis
# ============================================================

def plot_buoys_cartopy(ax, buoys):
    for name, coords in buoys.items():
        ax.plot(
            coords["lon"], coords["lat"],
            marker="o", color=coords["color"],
            markersize=10,
            markeredgecolor="black", markeredgewidth=0.8,
            transform=ccrs.PlateCarree(), zorder=10,
        )

# ============================================================
# 7) BUILD FIGURE
# ============================================================

print("Building figure …")

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(
    2, 2, figure=fig,
    height_ratios=[1.4, 1.0],
    hspace=0.12,
    wspace=0.08,
)

# ── 7a  TOP PANEL — bathymetry overview (spans both columns) ─────────────────

ax_top = fig.add_subplot(gs[0, :])

p = ax_top.pcolormesh(
    lon_grid, lat_grid, -z_down,
    cmap="jet", shading="auto", vmin=0, vmax=150,
)

# Cal/Val HR tile outlines
if calval_hr_tiles is not None:
    calval_hr_tiles.plot(
        ax=ax_top,
        facecolor="none", edgecolor="white",
        linewidth=2, alpha=0.9, zorder=3,
        label="HR tiles (Cal/Val)",
    )

# Satellite basemap
ctx.add_basemap(
    ax_top, crs="EPSG:4326",
    source=ctx.providers.Esri.WorldImagery,
    attribution=False,
)

ax_top.set_xlim([-6, 2.5])
ax_top.set_ylim([48, 51.5])
ax_top.set_xlabel("Longitude")
ax_top.set_ylabel("Latitude")
ax_top.set_aspect("equal")

# Colorbar — attached to top panel
divider  = make_axes_locatable(ax_top)
cax_top  = divider.append_axes("right", size="2%", pad=0.08)
cbar_top = plt.colorbar(p, cax=cax_top)
cbar_top.set_label("Depth (m)", fontsize=16, fontweight="bold")

# Legend
legend_patch = mpatches.Patch(
    edgecolor="white", facecolor="none", linewidth=2,
    label="HR tiles (Cal/Val)",
)
ax_top.legend(
    handles=[legend_patch], loc="lower right",
    framealpha=0.6, facecolor="gray", edgecolor="white",
    labelcolor="white", fontsize=12,
)

# ── 7b  BOTTOM LEFT — Tile 067R ───────────────────────────────────────────────

ax_bl = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())

sc1 = ax_bl.scatter(
    lon_067, lat_067, c=h_067,
    cmap="ocean", s=0.5,
    vmin=46.5, vmax=49.5,
    transform=ccrs.PlateCarree(), zorder=5,
)
ax_bl.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=6)
ax_bl.add_feature(cfeature.LAND,      facecolor="lightgray", zorder=4)
ax_bl.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle=":", zorder=6)

gl1 = ax_bl.gridlines(
    draw_labels=True, linewidth=0.5,
    color="gray", alpha=0.7, linestyle="--",
)
gl1.top_labels   = False
gl1.right_labels = False

plot_buoys_cartopy(ax_bl, BUOYS_067R)
ax_bl.set_extent([-2.6, -1.55, 49.1, 49.8], crs=ccrs.PlateCarree())
ax_bl.set_title("Tile 067R", fontsize=16, fontweight="bold")

# ── 7c  BOTTOM RIGHT — Tile 065R ──────────────────────────────────────────────

ax_br = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())

sc2 = ax_br.scatter(
    lon_065, lat_065, c=h_065,
    cmap="ocean", s=0.5,
    vmin=46.5, vmax=49.5,
    transform=ccrs.PlateCarree(), zorder=5,
)
ax_br.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=6)
ax_br.add_feature(cfeature.LAND,      facecolor="lightgray", zorder=4)
ax_br.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle=":", zorder=6)

gl2 = ax_br.gridlines(
    draw_labels=True, linewidth=0.5,
    color="gray", alpha=0.7, linestyle="--",
)
gl2.top_labels   = False
gl2.right_labels = False

plot_buoys_cartopy(ax_br, BUOYS_065R)
ax_br.set_extent([-3.15, -2.0, 50.1, 50.9], crs=ccrs.PlateCarree())
ax_br.set_title("Tile 065R", fontsize=16, fontweight="bold")

# ── 7d  SHARED COLORBAR below both bottom panels ─────────────────────────────

# Finalize layout so axes positions are available
fig.canvas.draw()

pos_bl = ax_bl.get_position()
pos_br = ax_br.get_position()

cax_shared = fig.add_axes([
    pos_bl.x0,
    pos_bl.y0 - 0.045,
    pos_br.x1 - pos_bl.x0,
    0.018,
])
cbar_shared = fig.colorbar(sc1, cax=cax_shared, orientation="horizontal")
cbar_shared.set_label(
    "SSH (m)", fontsize=16, fontweight="bold",
    fontfamily="Times New Roman",
)

# ============================================================
# 8) SAVE & SHOW
# ============================================================

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=600, bbox_inches="tight")
print(f"\n✓ Saved: {OUT_PATH}")
plt.show()
