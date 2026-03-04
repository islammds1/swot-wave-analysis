"""
scripts/plot_model_vs_swot.py
==============================
Reads the merged Excel (_+model.xlsx) and produces:

  Panel (a) — Scatter: SWOT peak period    vs model VTPK  (1:1 line, bias/RMSE/r)
  Panel (b) — Scatter: SWOT direction      vs model VPED  (1:1 line, circular stats)
  Panel (c) — Bar:     SWOT direction per box ± 1σ with model VPED overlaid as diamonds

Note on VPED
------------
  VPED is the peak wave DIRECTION [°], not a period.
  SWOT direction_degN uses coming-FROM convention.
  If your model uses going-TO convention, set FLIP_VPED = True below.

Usage
-----
  python scripts/plot_model_vs_swot.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# 0) SETTINGS
# ============================================================

EXCEL_FILE = r"D:/PhD/SWOT_L2_HR_PIXC/Pass016/swot_wave_results_20230405_cycle481_pass016_+model.xlsx"
OUT_DIR    = os.path.dirname(EXCEL_FILE)

# Set True if model VPED is going-TO and SWOT direction is coming-FROM
FLIP_VPED  = False

matplotlib.rcParams.update({
    "font.family"      : "Times New Roman",
    "font.size"        : 16,
    "font.weight"      : "bold",
    "axes.labelweight" : "bold",
    "axes.titleweight" : "bold",
    "xtick.labelsize"  : 14,
    "ytick.labelsize"  : 14,
})

# ============================================================
# 1) LOAD DATA
# ============================================================

df = pd.read_excel(EXCEL_FILE)
df = df[df["box_id"] != "D2"].reset_index(drop=True)

if FLIP_VPED:
    df["model_VPED_degN"] = (df["model_VPED_degN"] + 180) % 360
    print("VPED flipped +180° (going-TO → coming-FROM)")

labels    = df["box_id"].values
n         = len(labels)
zone_color = {"D1": "#1f77b4", "D2": "#ff7f0e", "D3": "#2ca02c"}
pt_colors  = [zone_color.get(str(row["zone"]), "gray") for _, row in df.iterrows()]

# ============================================================
# 2) STATISTICS HELPERS
# ============================================================

def linear_stats(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan, 0
    d = y[mask] - x[mask]
    return float(np.mean(d)), float(np.sqrt(np.mean(d**2))), \
           float(np.corrcoef(x[mask], y[mask])[0, 1]), int(mask.sum())


def circular_diff(a, b):
    return ((a - b + 180) % 360) - 180


def circular_stats_dir(swot_deg, model_deg):
    mask  = np.isfinite(swot_deg) & np.isfinite(model_deg)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan, 0
    diffs = circular_diff(swot_deg[mask], model_deg[mask])
    bias  = float(np.mean(diffs))
    rmse  = float(np.sqrt(np.mean(diffs**2)))
    a = np.radians(swot_deg[mask]);  b = np.radians(model_deg[mask])
    num = np.sum(np.sin(a - np.mean(a)) * np.sin(b - np.mean(b)))
    den = np.sqrt(np.sum(np.sin(a - np.mean(a))**2) *
                  np.sum(np.sin(b - np.mean(b))**2))
    r = float(num / den) if den > 1e-12 else np.nan
    return bias, rmse, r, int(mask.sum())

# ============================================================
# 3) FIGURE  (1 row × 3 panels)
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.subplots_adjust(wspace=0.32)

# ── (a) Period scatter: SWOT T vs VTPK ───────────────────────────────────────
ax = axes[0]
sx = df["model_VTPK_s"].values.astype(float)
sy = df["period_s"].values.astype(float)

ax.scatter(sx, sy, c=pt_colors, s=130, zorder=5,
           edgecolors="k", linewidths=0.8)
for i, lbl in enumerate(labels):
    if np.isfinite(sx[i]) and np.isfinite(sy[i]):
        ax.annotate(lbl, (sx[i], sy[i]),
                    textcoords="offset points", xytext=(7, 4),
                    fontsize=12, fontweight="bold", color=pt_colors[i])

valid = np.isfinite(sx) & np.isfinite(sy)
lo = min(sx[valid].min(), sy[valid].min()) - 0.4
hi = max(sx[valid].max(), sy[valid].max()) + 0.4
ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="1:1", zorder=3)
ax.set_xlim(lo, hi);  ax.set_ylim(lo, hi);  ax.set_aspect("equal")

bias_t, rmse_t, r_t, N_t = linear_stats(sx, sy)
ax.text(0.04, 0.97,
        f"bias = {bias_t:+.2f} s\nRMSE = {rmse_t:.2f} s\nr = {r_t:.2f}  (N={N_t})",
        transform=ax.transAxes, va="top", ha="left", fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", pad=4))
ax.set_xlabel("Model VTPK [s]", fontsize=15)
ax.set_ylabel("SWOT $T_{peak}$ [s]", fontsize=15)
ax.set_title("(a)  Peak Period", fontsize=16, loc="left")
ax.legend(fontsize=13);  ax.grid(linestyle="--", alpha=0.35)

# ── (b) Direction scatter: SWOT dir vs VPED ──────────────────────────────────
ax = axes[1]
mx = df["model_VPED_degN"].values.astype(float)
my = df["direction_degN"].values.astype(float)

ax.scatter(mx, my, c=pt_colors, s=130, zorder=5,
           edgecolors="k", linewidths=0.8)
for i, lbl in enumerate(labels):
    if np.isfinite(mx[i]) and np.isfinite(my[i]):
        ax.annotate(lbl, (mx[i], my[i]),
                    textcoords="offset points", xytext=(7, 4),
                    fontsize=12, fontweight="bold", color=pt_colors[i])

valid_d = np.isfinite(mx) & np.isfinite(my)
lo_d = min(mx[valid_d].min(), my[valid_d].min()) - 5
hi_d = max(mx[valid_d].max(), my[valid_d].max()) + 5
ax.plot([lo_d, hi_d], [lo_d, hi_d], "k--", lw=1.5, label="1:1", zorder=3)
ax.set_xlim(lo_d, hi_d);  ax.set_ylim(lo_d, hi_d);  ax.set_aspect("equal")

bias_d, rmse_d, r_d, N_d = circular_stats_dir(my, mx)
ax.text(0.04, 0.97,
        f"bias = {bias_d:+.1f}°\nRMSE = {rmse_d:.1f}°\nr = {r_d:.2f}  (N={N_d})",
        transform=ax.transAxes, va="top", ha="left", fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", pad=4))
ax.set_xlabel("Model VPED [°N]", fontsize=15)
ax.set_ylabel("SWOT direction coming FROM [°N]", fontsize=15)
ax.set_title("(b)  Peak Direction", fontsize=16, loc="left")
ax.legend(fontsize=13);  ax.grid(linestyle="--", alpha=0.35)
ax.text(0.97, 0.03,
        "VPED = peak direction [°]" + ("  (+180° flip)" if FLIP_VPED else ""),
        transform=ax.transAxes, va="bottom", ha="right",
        fontsize=10, style="italic", color="gray")

# ── (c) Direction bar chart ± 1σ + model VPED diamonds ──────────────────────
ax = axes[2]
x_pos    = np.arange(n)
dirs     = df["direction_degN"].values.astype(float)
dir_std  = df["direction_1sig"].values.astype(float)
dir_ci95 = df["direction_95ci"].values.astype(float)
vped     = df["model_VPED_degN"].values.astype(float)

ax.bar(x_pos, dirs, color=pt_colors, edgecolor="k",
       linewidth=0.7, zorder=3, label="SWOT direction")
ax.errorbar(x_pos, dirs, yerr=dir_std,
            fmt="none", ecolor="k", elinewidth=2.0,
            capsize=6, capthick=1.8, zorder=5, label="±1σ")
ax.errorbar(x_pos, dirs, yerr=dir_ci95,
            fmt="none", ecolor="dimgray", elinewidth=1.0,
            capsize=4, linestyle="dotted", zorder=4, label="95% CI")
ax.scatter(x_pos, vped, marker="D", s=90, color="red",
           zorder=6, edgecolors="k", linewidths=0.7, label="Model VPED")

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=13)
ax.set_ylabel("Direction coming FROM [° CW from N]", fontsize=14)
ax.set_title("(c)  Direction per Box", fontsize=16, loc="left")
ax.set_ylim(0, 360);  ax.set_yticks(np.arange(0, 361, 45))
ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

zone_handles = [mpatches.Patch(facecolor=c, edgecolor="k", label=z)
                for z, c in zone_color.items()]
series_handles = [
    plt.Line2D([0], [0], color="k", lw=2, label="±1σ"),
    plt.Line2D([0], [0], color="dimgray", lw=1, linestyle="dotted", label="95% CI"),
    plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="red",
               markeredgecolor="k", markersize=9, label="Model VPED"),
]
ax.legend(handles=zone_handles + series_handles,
          fontsize=11, loc="upper right", ncol=2)

# ============================================================
# 4) SUPER TITLE & SAVE
# ============================================================

fig.suptitle(
    "SWOT HR PIXC  vs  NWShelf Model  |  2023-04-05  |  Pass 016 Cycle 481",
    fontsize=16, fontweight="bold", y=1.02,
)
out_jpg = os.path.join(OUT_DIR, "swot_vs_model_period_direction.jpg")
out_png = os.path.join(OUT_DIR, "swot_vs_model_period_direction.png")
plt.savefig(out_jpg, dpi=300, bbox_inches="tight")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.show()
print(f"✓ Saved: {out_jpg}")

# ============================================================
# 5) PRINT SUMMARY TABLE
# ============================================================

print(f"\n{'='*75}")
print(f"{'Box':<6} {'SWOT T':>8} {'VTPK':>7} {'ΔT':>6} "
      f"{'SWOT dir':>10} {'VPED':>7} {'Δdir':>7}")
print("-"*75)
for _, row in df.iterrows():
    dt   = row["period_s"] - row["model_VTPK_s"]
    ddir = circular_diff(row["direction_degN"], row["model_VPED_degN"])
    print(f"{row['box_id']:<6} "
          f"{row['period_s']:>8.2f} {row['model_VTPK_s']:>7.2f} {dt:>+6.2f} "
          f"{row['direction_degN']:>10.1f} {row['model_VPED_degN']:>7.1f} {ddir:>+7.1f}")

print(f"\nPeriod    — bias={bias_t:+.3f} s   RMSE={rmse_t:.3f} s   r={r_t:.3f}")
print(f"Direction — bias={bias_d:+.1f}°  RMSE={rmse_d:.1f}°  r={r_d:.3f}")
print("\nVPED = peak wave direction [°]  |  VTPK = peak wave period [s]")
if FLIP_VPED:
    print("VPED was flipped +180° (going-TO → coming-FROM)")
else:
    print("Set FLIP_VPED=True if model uses going-TO and SWOT uses coming-FROM")