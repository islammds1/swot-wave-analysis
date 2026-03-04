# swot-wave-analysis

2D cross-spectrum wave analysis for **SWOT HR PIXC** (Level-2 High Rate
Pixel Cloud) data, with directional ambiguity resolution following
**Ardhuin et al. (2024)**.

---

## What it does

For each user-defined analysis box (track-aligned in UTM):

1. Bins scattered PIXC points (SSH, σ₀) onto a regular satellite-frame grid
2. Computes a **Welch 2D cross-spectrum** (SSH × σ₀) with 50 % patch overlap
3. Finds the spectral peak in a user-defined swell band
4. Resolves the **180° directional ambiguity** via the sign of Im(C_SSH,σ₀)  
   — physically: Im(C) ∝ k_range · E(k) (Ardhuin et al. 2024)
5. Extracts **mean water depth** from a bathymetry file over the exact
   rotated box footprint
6. Computes **peak wave period** via the full dispersion relation
   ω² = g·k·tanh(k·h) using the box-specific depth
7. Reports all results with **1-sigma and 95% CI uncertainties** derived
   from the spread across Welch patches
8. Saves results to **Excel + CSV**

---

## Repository structure

```
swot-wave-analysis/
├── src/
│   ├── swot_geometry.py   — satellite track angle, Vsat, look angle
│   ├── spectral.py        — Welch cross-spectrum, peak detection
│   ├── ambiguity.py       — Ardhuin 2024 resolution + geo conversion
│   ├── bathymetry.py      — depth extraction for rotated boxes
│   └── dispersion.py      — ω²=g·k·tanh(k·h), period, group speed
├── config/
│   └── boxes_D1_D2_D3.py  — file paths, box definitions, parameters
├── scripts/
│   └── run_analysis.py    — main orchestration script
├── tests/
│   └── test_dispersion.py — unit tests (pytest)
├── output/                — figures and results go here (git-ignored)
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/swot-wave-analysis.git
cd swot-wave-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure paths

Edit **`config/boxes_D1_D2_D3.py`** and set your data root:

```python
# Option A — environment variable (recommended)
export SWOT_DATA_ROOT=/path/to/your/data

# Option B — edit directly
_ROOT = "/path/to/your/data"
```

### 4. Run

```bash
python scripts/run_analysis.py
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Input data

| File | Description |
|------|-------------|
| `SWOT_L2_HR_PIXC_*.nc` | SWOT Level-2 HR Pixel Cloud (HDF5/NetCDF4) |
| `SWOT_L3_LR_SSH_Unsmoothed_*.nc` | SWOT Level-3 LR SSH (for satellite geometry) |
| `E4_2024.nc` | Bathymetry NetCDF with `elevation` variable on a `lat/lon` grid |

---

## Output files

| File | Description |
|------|-------------|
| `swot_hr_boxes_phase.png` | Diagnostic: E_ssh, Coherence, Phase per box |
| `swot_hr_pub_spectra_phase.jpg` | Publication figure (Times New Roman, 600 dpi) |
| `swot_wave_results_<date>_cycle<N>_pass<M>.xlsx` | Full results spreadsheet |
| `swot_wave_results_<date>_cycle<N>_pass<M>.csv` | Same as CSV |

### Results columns

| Column | Description |
|--------|-------------|
| `date` | Acquisition date from filename |
| `cycle` / `pass` | SWOT cycle and pass numbers |
| `box_id` | Box label (D1A, D1B, …) |
| `zone` | Coarse zone (D1, D2, D3) |
| `tile` | PIXC tile (067R / 065R) |
| `depth_m` | Mean water depth over box [m] |
| `wavelength_m` | Peak wavelength [m] |
| `wavelength_1sig_m` | 1σ std of wavelength [m] |
| `wavelength_95ci_m` | 95% CI half-width [m] |
| `period_s` | Peak period from dispersion [s] |
| `dispersion_regime` | deep / intermediate / shallow |
| `direction_degN` | Wave coming FROM direction [° CW from N] |
| `direction_1sig` | 1σ circular std [°] |
| `direction_95ci` | 95% CI half-width [°] |
| `dir_R` | Circular resultant length (0–1) |

---

## Method reference

> Ardhuin, F. et al. (2024). *Wave directional spectra from SWOT*.  
> *(Full citation to be added upon publication.)*

---

## Requirements

```
numpy >= 1.24
xarray >= 2023.1
pyproj >= 3.5
scipy >= 1.10
matplotlib >= 3.7
pandas >= 2.0
openpyxl >= 3.1
h5netcdf >= 1.1
```

---

## License

MIT © 2024 — see `LICENSE`.
