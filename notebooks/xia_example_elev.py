#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:54:46 2026

@author: m484s199
"""

"""
Grounding point extraction from airborne IPR bed power + bed elevation
Following: Xia et al. (2025), IEEE TGRS 63, doi:10.1109/TGRS.2025.3620827
"Amery Ice Shelf Grounding Line Datapoints Automated Extraction
 From Airborne Ice-Penetrating Radar"

Both branches of Fig. 3 are implemented:
  AMPLITUDE BRANCH
    1. Butterworth low-pass  : order=5, Wn=0.15  (bed_power_dB)
    2. Normalize by max echo amplitude
    3. 1st derivative along-track → local maxima
    4. Threshold constraints T1-T4 on candidates
    5. Progressive threshold relaxation until candidates emerge

  ELEVATION BRANCH
    1. Interpolate hbed (surface_bottom wgs84) onto bed-power along-track grid
    2. Butterworth low-pass  : order=5, Wn=0.09  (hbed)
    3. Error function fit to smoothed elevation
    4. 3rd derivative zeros -> seaward basal topography break

  COMBINATION
    Grounding point = amplitude-gradient candidate seaward of, and
    closest to, the elevation-branch topo break.

Inputs
------
  bed_power_concat.csv   - slow_time, along_track (m), bed_power_dB
  surface_bottom.csv     - slow_time, along_track (m), wgs84 (m, bed elevation)

Outputs
-------
  bed_power_filtered.csv          - original columns + amp_filtered_dB,
                                    amp_gradient_dBm, hbed_filtered_m,
                                    hbed_gradient_mm
  grounding_point.csv             - single-row result table
  grounding_point_extraction.png
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, argrelextrema
from scipy.optimize import curve_fit
from scipy.special import erf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -----------------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------------
bp = pd.read_csv("../data/bed_power_concat.csv")
sb = pd.read_csv("../data/surface_bottom.csv")

x   = bp["along_track"].values      # m, bed-power grid (~679 m spacing)
amp = bp["bed_power_dB"].values      # dB, corrected bed echo amplitude

# Interpolate bed elevation (wgs84, ~14.7 m spacing) onto the coarser
# bed-power along-track grid. Points beyond SB coverage become NaN.
sb_clean = sb.dropna(subset=["wgs84"])
hbed = np.interp(
    x,
    sb_clean["along_track"].values,
    sb_clean["wgs84"].values,
    left=np.nan,
    right=np.nan,
)

# -----------------------------------------------------------------------------
# 2. Interpolate NaN gaps before filtering (NaNs are restored afterwards)
# -----------------------------------------------------------------------------
nan_mask_amp  = np.isnan(amp)
nan_mask_hbed = np.isnan(hbed)

amp_interp  = pd.Series(amp).interpolate(method="linear").ffill().bfill().values
hbed_interp = pd.Series(hbed).interpolate(method="linear").ffill().bfill().values

# -----------------------------------------------------------------------------
# 3. AMPLITUDE BRANCH -- Butterworth order=5, Wn=0.15
# -----------------------------------------------------------------------------
b_a, a_a = butter(5, 0.15, btype="low", analog=False)
amp_filt  = filtfilt(b_a, a_a, amp_interp)   # zero-phase (forward + backward)

amp_filt_out = amp_filt.copy()
amp_filt_out[nan_mask_amp] = np.nan

# Normalize: subtract global max so the brightest echo = 0 dB
amp_norm = amp_filt - np.max(amp_filt)

# 1st derivative of normalized filtered amplitude (dB/m)
dA = np.gradient(amp_norm, x)

# -----------------------------------------------------------------------------
# 4. ELEVATION BRANCH -- Butterworth order=5, Wn=0.09
# -----------------------------------------------------------------------------
b_e, a_e  = butter(5, 0.09, btype="low", analog=False)
hbed_filt = filtfilt(b_e, a_e, hbed_interp)

hbed_filt_out = hbed_filt.copy()
hbed_filt_out[nan_mask_hbed] = np.nan

# 1st derivative of smoothed bed elevation (m/m)
dhbed = np.gradient(hbed_filt, x)

# -----------------------------------------------------------------------------
# 5. Error function fit to bed elevation -> basal topography break
#
#    Model: hbed(x) = A * erf((x - x0) / w) + B
#
#    The 3rd derivative of erf has zeros at x0 +/- w/sqrt(2).
#    The seaward zero (x0 + w/sqrt(2)) is the abrupt basal slope-break
#    that marks the landward limit of the grounding zone.
# -----------------------------------------------------------------------------
def erf_model(t, A, x0, w, B):
    return A * erf((t - x0) / w) + B

# Centre the fitting window on the bed elevation minimum (deepest trough
# between grounded and floating regimes).
min_hbed_idx = int(np.argmin(hbed_filt))
gz_center_km = float(x[min_hbed_idx] / 1000)

half_win_elev = 80_000   # +/- 80 km
lo_e = int(np.searchsorted(x, x[min_hbed_idx] - half_win_elev))
hi_e = int(np.searchsorted(x, x[min_hbed_idx] + half_win_elev))
x_e_km   = x[lo_e:hi_e] / 1000
hbed_win = hbed_filt[lo_e:hi_e]

A0 = (hbed_win.max() - hbed_win.min()) / 2
popt_e, _ = curve_fit(
    erf_model, x_e_km, hbed_win,
    p0=[A0, gz_center_km, 15.0, hbed_win.mean()],
    maxfev=10_000,
)
A_e, x0_e, w_e, B_e = popt_e
hbed_erf = erf_model(x_e_km, *popt_e)

# Seaward and landward basal topography breaks (3rd-derivative zeros)
topo_break_km    = float(x0_e + w_e / np.sqrt(2))
landward_break_km = float(x0_e - w_e / np.sqrt(2))

print(f"Elevation erf fit   : center={x0_e:.2f} km, width={w_e:.2f} km")
print(f"Topo break (seaward): {topo_break_km:.2f} km")
print(f"Topo break (landward): {landward_break_km:.2f} km")

# -----------------------------------------------------------------------------
# 6. Amplitude gradient local maxima -> threshold filtering
#
#    Search for positive local maxima of dA within +/- 60 km of the
#    seaward topo break, then apply T1-T4 relative to window statistics.
#    Thresholds are relaxed progressively until candidates emerge.
# -----------------------------------------------------------------------------
half_win_amp = 60_000   # +/- 60 km around seaward topo break
lo_a = int(np.searchsorted(x, topo_break_km * 1000 - half_win_amp))
hi_a = int(np.searchsorted(x, topo_break_km * 1000 + half_win_amp))

x_amp  = x[lo_a:hi_a]
dA_amp = dA[lo_a:hi_a]
amp_s  = amp_filt[lo_a:hi_a]

pos_max = argrelextrema(dA_amp, np.greater, order=3)[0]

amp_win_max   = float(amp_s.max())
amp_win_range = float(amp_s.max() - amp_s.min())
dA_win_max    = float(np.abs(dA_amp).max())

candidates = []
for li in pos_max:
    gi  = lo_a + int(li)
    lo  = max(0, int(li) - 15)
    hi  = min(len(x_amp), int(li) + 15)

    # T1: amplitude at extremum >= 50% of window-max amplitude
    t1 = bool(amp_s[li] >= 0.50 * amp_win_max)
    # T2: amplitude range in local interval >= 45% of window amplitude range
    t2 = bool((amp_s[lo:hi].max() - amp_s[lo:hi].min()) >= 0.45 * amp_win_range)
    # T3: |gradient| at extremum >= 62% of window-max |gradient|
    t3 = bool(abs(dA_amp[li]) >= 0.62 * dA_win_max)
    # T4: gradient range in local interval >= 1/3 of window-max |gradient|
    t4 = bool((dA_amp[lo:hi].max() - dA_amp[lo:hi].min()) >= (1 / 3) * dA_win_max)

    dist = float(abs(x_amp[li] / 1000 - topo_break_km))
    candidates.append({
        "global_idx":        gi,
        "km":                round(float(x[gi] / 1000), 2),
        "amp_filt_dB":       round(float(amp_filt[gi]), 2),
        "gradient_dBm":      round(float(dA[gi]), 8),
        "hbed_filt_m":       round(float(hbed_filt[gi]), 2),
        "T1": t1, "T2": t2, "T3": t3, "T4": t4,
        "all_pass":          bool(t1 and t2 and t3 and t4),
        "dist_to_break_km":  round(dist, 2),
    })

# Progressive relaxation: T1+T2+T3+T4 -> T1+T3+T4 -> T1+T3 -> T3 only
gp = None
relaxation_levels = [
    (True,  True,  True,  True,  "T1+T2+T3+T4 (strict)"),
    (True,  False, True,  True,  "T1+T3+T4"),
    (True,  False, True,  False, "T1+T3"),
    (False, False, True,  False, "T3 only"),
]
for t1f, t2f, t3f, t4f, label in relaxation_levels:
    pool = [
        c for c in candidates
        if (not t1f or c["T1"]) and (not t2f or c["T2"])
        and (not t3f or c["T3"]) and (not t4f or c["T4"])
    ]
    if pool:
        # Prefer seaward candidates; within those, closest to topo break
        seaward = [c for c in pool if c["km"] >= topo_break_km - 15]
        gp = min(seaward if seaward else pool, key=lambda c: c["dist_to_break_km"])
        print(f"Threshold level used : {label}")
        break

if gp is None:
    gp = min(candidates, key=lambda c: c["dist_to_break_km"])
    print("Warning: no candidate passed T3 -- using closest to topo break.")

print(f"\n==> GROUNDING POINT")
print(f"    Along-track    : {gp['km']:.2f} km")
print(f"    Amp (filtered) : {gp['amp_filt_dB']:.2f} dB")
print(f"    Gradient       : {gp['gradient_dBm']*1e4:.3f} x10^-4 dB/m")
print(f"    hbed (filt)    : {gp['hbed_filt_m']:.1f} m")
print(f"    T1={gp['T1']} T2={gp['T2']} T3={gp['T3']} T4={gp['T4']}")
print(f"    Dist to topo break: {gp['dist_to_break_km']:.2f} km")

# -----------------------------------------------------------------------------
# 7. Save outputs
# -----------------------------------------------------------------------------
df_out = bp.copy()
df_out["amp_filtered_dB"]  = amp_filt_out
df_out["amp_gradient_dBm"] = dA
df_out["hbed_filtered_m"]  = hbed_filt_out
df_out["hbed_gradient_mm"] = dhbed
df_out.to_csv("bed_power_filtered.csv", index=False)

gp_out = pd.DataFrame([{
    "along_track_m":          gp["km"] * 1000,
    "along_track_km":         gp["km"],
    "amp_filtered_dB":        gp["amp_filt_dB"],
    "amp_gradient_dBm":       gp["gradient_dBm"],
    "hbed_filtered_m":        gp["hbed_filt_m"],
    "erf_center_km":          round(x0_e, 2),
    "topo_break_seaward_km":  round(topo_break_km, 2),
    "topo_break_landward_km": round(landward_break_km, 2),
    "dist_to_topo_break_km":  gp["dist_to_break_km"],
}])
# gp_out.to_csv("grounding_point.csv", index=False)
print("\nSaved: bed_power_filtered.csv, grounding_point.csv")

# -----------------------------------------------------------------------------
# 8. Plot -- three-panel figure (Fig. 7 style)
# -----------------------------------------------------------------------------
x_km = x / 1000

fig, axes = plt.subplots(3, 1, figsize=(14, 11))
fig.subplots_adjust(hspace=0.45)

C = {
    "raw":  "#b4b2a9",
    "filt": "#378add",
    "erf":  "#1d9e75",
    "topo": "#ef9f27",
    "gp":   "#e24b4a",
    "hbed": "#7f77dd",
}

# -- Panel A: full transect bed power -----------------------------------------
ax = axes[0]
ax.plot(x_km, amp, color=C["raw"], lw=0.7, alpha=0.7, label="Raw bed power")
ax.plot(x_km, amp_filt_out, color=C["filt"], lw=1.8,
        label="Butterworth filtered (order=5, $W_n$=0.15)")
ax.axvline(topo_break_km, color=C["topo"], lw=1.5, ls="--",
           label=f"Seaward topo break: {topo_break_km:.1f} km")
ax.axvline(gp["km"], color=C["gp"], lw=2,
           label=f"Grounding point: {gp['km']:.2f} km  ({gp['amp_filt_dB']:.1f} dB)")
ax.axvspan(x_amp[0] / 1000, x_amp[-1] / 1000, alpha=0.06, color=C["topo"])
ax.set_ylabel("Bed power (dB)")
ax.set_xlabel("Along-track (km)")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.2)
ax.set_title("(a) Amplitude -- Butterworth filtered (order=5, $W_n$=0.15)")

# -- Panel B: bed elevation with erf fit (grounding zone) ---------------------
ax = axes[1]
gz_lo = x_e_km[0] - 5
gz_hi = x_e_km[-1] + 5
mask_gz = (x_km >= gz_lo) & (x_km <= gz_hi)

ax.plot(x_km[mask_gz], hbed[mask_gz], color=C["raw"], lw=0.7, alpha=0.6,
        label="Raw hbed (interpolated from surface_bottom.csv)")
ax.plot(x_km[mask_gz], hbed_filt_out[mask_gz], color=C["hbed"], lw=1.8,
        label="Butterworth filtered hbed (order=5, $W_n$=0.09)")
ax.plot(x_e_km, hbed_erf, color=C["erf"], lw=1.5, ls="--",
        label=f"Error function fit (center={x0_e:.1f} km, w={w_e:.1f} km)")
ax.axvline(topo_break_km, color=C["topo"], lw=1.5, ls="--",
           label=f"Seaward topo break (x0+w/sqrt(2)): {topo_break_km:.1f} km")
ax.axvline(landward_break_km, color=C["topo"], lw=1, ls=":",
           label=f"Landward topo break (x0-w/sqrt(2)): {landward_break_km:.1f} km")
ax.axvline(gp["km"], color=C["gp"], lw=2, label=f"GP: {gp['km']:.2f} km")
ax.set_xlim(gz_lo, gz_hi)
ax.set_ylabel("Bed elevation (m, WGS84)")
ax.set_xlabel("Along-track (km)")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.2)
ax.set_title("(b) Bed elevation -- error function fit -> 3rd derivative zeros (topo break)")

# -- Panel C: amplitude gradient in grounding zone ----------------------------
ax = axes[2]
x_amp_km = x_amp / 1000

ax.plot(x_amp_km, dA_amp * 1e4, color=C["filt"], lw=1.5,
        label="dA/dx (amplitude gradient)")
ax.fill_between(x_amp_km, dA_amp * 1e4, 0,
                where=(dA_amp > 0), alpha=0.10, color=C["filt"])
ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
ax.axvline(topo_break_km, color=C["topo"], lw=1.5, ls="--",
           label=f"Seaward topo break: {topo_break_km:.1f} km")
ax.axvline(gp["km"], color=C["gp"], lw=2, label=f"GP: {gp['km']:.2f} km")

for c in candidates:
    passed = c["T3"]
    ax.scatter(c["km"], c["gradient_dBm"] * 1e4,
               s=70 if passed else 30,
               color=C["gp"] if passed else "#f09595",
               edgecolors="#a32d2d" if passed else C["gp"],
               linewidths=1.2, zorder=5)
    if passed:
        ax.annotate(f"GP\n{c['km']:.1f} km",
                    (c["km"], c["gradient_dBm"] * 1e4),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=9, color="#a32d2d")

ax.set_xlim(x_amp_km[0] - 2, x_amp_km[-1] + 2)
ax.set_ylabel("x10^-4 dB/m")
ax.set_xlabel("Along-track (km)")
p1 = mpatches.Patch(color=C["gp"],   label="Candidate passing T3 (selected GP)")
p2 = mpatches.Patch(color="#f09595", label="Candidate failing T3")
handles, llabels = ax.get_legend_handles_labels()
ax.legend(handles=handles + [p1, p2], fontsize=9)
ax.grid(True, alpha=0.2)
ax.set_title("(c) Amplitude gradient -- local maxima + threshold selection")

fig.suptitle("Grounding point extraction  |  Xia et al. (2025) two-branch method",
             fontsize=12, y=1.01)
plt.tight_layout()
# plt.savefig("grounding_point_extraction.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: grounding_point_extraction.png")