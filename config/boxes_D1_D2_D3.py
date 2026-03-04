"""
config/boxes_D1_D2_D3.py
=========================
Scene-specific configuration for the SWOT HR PIXC analysis.

Edit ONLY this file when changing:
  - Input data paths
  - Box definitions
  - Spectral / physical parameters

Everything else (physics, FFT, ambiguity resolution) lives in src/.
"""

import os

# ============================================================
# DATA PATHS
# Edit these to point to your local data copies.
# Tip: set the environment variable SWOT_DATA_ROOT to avoid
# hardcoding your personal drive letter.
# ============================================================

_ROOT      = os.environ.get("SWOT_DATA_ROOT", "D:/PhD")
_PIXC_ROOT = os.path.join(_ROOT, "SWOT_L2_HR_PIXC/Pass016")
_LR_ROOT   = os.path.join(_ROOT, "Data")
_BATHY_ROOT = os.path.join(_ROOT, "Data/Bathymetry")

PIXC_FILE  = os.path.join(
    _PIXC_ROOT,
    "SWOT_L2_HR_PIXC_481_016_067R_20230405T095353_20230405T095402_PGC0_01.nc"
)
PIXC_FILE2 = os.path.join(
    _PIXC_ROOT,
    "SWOT_L2_HR_PIXC_481_016_065R_20230405T095331_20230405T095342_PGC0_01.nc"
)
LR_FILE    = os.path.join(
    _LR_ROOT,
    "SWOT_L3_LR_SSH_Unsmoothed_481_016_20230405T094254_20230405T103359_v2.0.1.nc"
)
BATHY_FILE = os.path.join(_BATHY_ROOT, "E4_2024.nc")

GROUP = "pixel_cloud"   # HDF5 group inside the PIXC file

# ============================================================
# ANALYSIS BOXES
# Format: (centre_utm_x_m, centre_utm_y_m, half_along_m, half_cross_m)
# Coordinate system: UTM Zone 30N
# half_along / half_cross are in the satellite frame
# ============================================================

BOXES = [
    (562_500.0, 5_497_500.0, 2500.0, 2500.0),   # D1A  ← tile 067R
    (567_500.0, 5_497_500.0, 2500.0, 2500.0),   # D1B
    (572_500.0, 5_497_500.0, 2500.0, 2500.0),   # D1C
    (577_500.0, 5_497_500.0, 2500.0, 2500.0),   # D1D
    (562_500.0, 5_479_500.0, 7500.0, 7500.0),   # D2
    (557_500.0, 5_474_500.0, 2500.0, 2500.0),   # D2A
    (557_500.0, 5_484_500.0, 2500.0, 2500.0),   # D2B
    (547_500.0, 5_587_500.0, 2500.0, 2500.0),   # D3A  ← tile 065R
    (547_500.0, 5_592_500.0, 2500.0, 2500.0),   # D3B
    (547_500.0, 5_597_500.0, 2500.0, 2500.0),   # D3C
    (547_500.0, 5_602_500.0, 2500.0, 2500.0),   # D3D
]

BOX_LABELS = ["D1A", "D1B", "D1C", "D1D", "D2",
              "D2A", "D2B", "D3A", "D3B", "D3C", "D3D"]

# Maps each box index → which PIXC file to use (0 = PIXC_FILE, 1 = PIXC_FILE2)
BOX_TILE   = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

# ============================================================
# SPECTRAL PARAMETERS
# ============================================================

RES        = 30.0     # grid resolution [m]
LWIN       = 1500.0   # Welch window size [m]
LAMBDA_MIN = 50.0     # swell band lower wavelength limit [m]
LAMBDA_MAX = 500.0    # swell band upper wavelength limit [m]

# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

G = 9.81   # gravitational acceleration [m/s²]

# ============================================================
# SATELLITE / SWATH CONFIGURATION
# ============================================================

SWATH      = "right"   # "right" or "left"
LOOK_SIDE  = "right"   # same as SWATH for standard SWOT geometry

# ============================================================
# ARDHUIN 2024 AMBIGUITY RESOLUTION WINDOW
# ============================================================

ARD_DK_REL   = 0.30   # ±30 % of peak wavenumber
ARD_DTHETA   = 30.0   # ±30° angular half-window [degrees]
ARD_COH_MIN  = 0.05   # minimum coherence in the Im(C) window

# ============================================================
# QUALITY GATES
# ============================================================

COH_MIN_PATCH = 0.3   # minimum coherence at the spectral peak for
                      # a patch to contribute to direction statistics

# ============================================================
# OUTPUT
# ============================================================

OUT_DIR = os.path.dirname(PIXC_FILE)   # save results alongside data
