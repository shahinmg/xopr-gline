"""
Grounding point detection for tidewater glaciers (Greenland)
Adapted from: Xia et al. (2025), IEEE TGRS 63, doi:10.1109/TGRS.2025.3620827

WHY THE ICE-SHELF METHOD DOES NOT TRANSFER DIRECTLY
=====================================================
The Xia et al. (2025) method was designed for Antarctic ice shelves where:
  - The transect crosses a monotonic grounded -> floating transition
  - Bed elevation follows a sigmoid shape (suitable for erf fitting)
  - Bed power increases gradually as the ice-water interface replaces bedrock
  - The grounding zone is a broad, smooth feature (~km wide)

Tidewater glaciers in Greenland are fundamentally different:
  - The glacier is largely or entirely grounded to the terminus (no floating tongue)
  - The bed does NOT follow a monotonic sigmoid -- it is deeply overdeepened
    beneath thick inland ice and may rise near the terminus
  - Large NaN gaps are common (heavily crevassed zones, no coherent bed return)
  - Bed power transitions are ABRUPT (sharp step, not sigmoid) because the
    glacier meets ocean water directly at or very near the calving front
  - The amplitude contrast is large (~67 dB here: ~-120 dB grounded vs ~-42 dB
    wet/ocean-contacted bed), but concentrated over only a few km
  - The erf fit to elevation is inapplicable: no floating regime, non-monotonic
    bed topography, and the transect may cross the calving-front geometry twice

ADAPTED METHOD FOR TIDEWATER GLACIERS
=======================================
Step 1: Butterworth filter on amplitude (order=5, Wn=0.15) -- same as paper
Step 2: Characterise two regimes from the amplitude profile:
          - DARK  (grounded cold bed): determined from the clearly grounded
                  interior region (far from terminus, no ocean influence)
          - BRIGHT (wet bed / ocean contact): determined from clearly bright
                  terminus-adjacent region
          The midpoint of these two regime means is the transition threshold.
Step 3: Detect the grounding point as the LAST dark -> bright transition
        moving seaward (increasing along-track), i.e. the point where the
        amplitude first rises above the transition threshold and stays there.
        This is done by:
          a. Computing the amplitude gradient (dA/dx)
          b. Finding the largest positive gradient maximum in the transition zone
             (the steepest rise from dark to bright)
          c. Optionally validating against bed elevation: the grounding point
             should correspond to where the bed drops toward or below sea level
             on the seaward side (hbed approaching 0 or negative)
Step 4: Report the grounding point location, amplitude, and hbed at that point.

HELHEIM GEOMETRY (this transect)
==================================
  0   --  5 km : BRIGHT (-33 to -69 dB) -- terminus / calving front
  5   -- 32 km : NaN gap -- heavily crevassed zone, no coherent bed return
  32  --150 km : DARK (-90 to -142 dB)  -- grounded ice on bedrock (hbed > 300 m)
 150  --165 km : NaN gap -- second crevassed zone
 165  --176 km : DARK (-107 to -121 dB) -- grounded over deep subglacial trough
 176  --183 km : BRIGHT (-37 to -45 dB) -- calving front / terminus approach

The grounding point on the seaward end is at the dark->bright transition
around 175-176 km. The landward end (0-5 km) is also at the terminus but
the grounding point there is obscured by the large NaN gap.

Inputs
------
  Helheim_20080730_01_bed_power.csv  -- slow_time, along_track (m), bed_power_dB
  Helheim_20080730_01_bottom.csv     -- slow_time, ..., along_track (m), wgs84 (m)

Outputs
-------
  helheim_filtered.csv
  helheim_grounding_point.csv
  helheim_grounding_detection.png
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, argrelextrema
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# USER SETTINGS
# =============================================================================
AMP_BUTTER_ORDER = 5
AMP_BUTTER_WN    = 0.15

# Along-track ranges (m) defining the clearly GROUNDED interior
# (used to compute the dark-regime mean -- stay well away from NaN gaps
#  and bright terminus zones)
DARK_LO_M  =  60_000   # 60 km
DARK_HI_M  = 145_000   # 145 km

# Along-track range (m) defining the clearly BRIGHT terminus zone
# (used to compute the bright-regime mean)
BRIGHT_LO_M = 176_000  # 176 km
BRIGHT_HI_M = 183_100  # end of transect

# Search window for the grounding point (m)
# The transition is expected between the last NaN gap and the bright zone
SEARCH_LO_M = 150_000
SEARCH_HI_M = 183_100

# Minimum number of consecutive bright points to confirm the grounding point
# (avoids false positives from isolated bright spikes)
MIN_BRIGHT_CONSECUTIVE = 3

# Gradient order for argrelextrema (in samples)
GRAD_ORDER = 3
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------------
bp = pd.read_csv("../data/Helheim_20080730_01_bed_power.csv")
bt = pd.read_csv("../data/Helheim_20080730_01_bottom.csv")

x   = bp["along_track"].values
amp = bp["bed_power_dB"].values

bt_clean = bt.dropna(subset=["wgs84"])
hbed = np.interp(
    x,
    bt_clean["along_track"].values,
    bt_clean["wgs84"].values,
    left=np.nan, right=np.nan,
)

# -----------------------------------------------------------------------------
# 2. Butterworth filter on amplitude
# -----------------------------------------------------------------------------
nan_mask   = np.isnan(amp)
amp_interp = pd.Series(amp).interpolate(method="linear").ffill().bfill().values

b_a, a_a  = butter(AMP_BUTTER_ORDER, AMP_BUTTER_WN, btype="low", analog=False)
amp_filt  = filtfilt(b_a, a_a, amp_interp)

amp_filt_out = amp_filt.copy()
amp_filt_out[nan_mask] = np.nan

# Normalize: brightest = 0 dB (paper convention)
amp_norm = amp_filt - np.max(amp_filt)
dA       = np.gradient(amp_norm, x)   # 1st derivative (dB/m)

# -----------------------------------------------------------------------------
# 3. Characterise the two amplitude regimes
# -----------------------------------------------------------------------------
dark_mask   = (x >= DARK_LO_M)   & (x <= DARK_HI_M)   & ~nan_mask
bright_mask = (x >= BRIGHT_LO_M) & (x <= BRIGHT_HI_M) & ~nan_mask

dark_mean  = float(np.nanmean(amp[dark_mask]))
dark_std   = float(np.nanstd(amp[dark_mask]))
bright_mean = float(np.nanmean(amp[bright_mask]))
bright_std  = float(np.nanstd(amp[bright_mask]))

# Transition threshold: midpoint between the two regime means
transition_thresh = (dark_mean + bright_mean) / 2.0

print(f"Dark regime  (grounded interior, {DARK_LO_M/1000:.0f}-{DARK_HI_M/1000:.0f} km):")
print(f"  mean = {dark_mean:.1f} dB,  std = {dark_std:.1f} dB")
print(f"Bright regime (terminus,         {BRIGHT_LO_M/1000:.0f}-{BRIGHT_HI_M/1000:.0f} km):")
print(f"  mean = {bright_mean:.1f} dB,  std = {bright_std:.1f} dB")
print(f"Amplitude contrast: {bright_mean - dark_mean:.1f} dB")
print(f"Transition threshold: {transition_thresh:.1f} dB")

# -----------------------------------------------------------------------------
# 4. Detect the grounding point
#    Strategy: within the search window, find the largest positive gradient
#    maximum -- this marks the steepest dark->bright transition.
#    Then confirm by checking that amplitude stays above threshold for at
#    least MIN_BRIGHT_CONSECUTIVE subsequent valid (non-NaN) samples.
# -----------------------------------------------------------------------------
search_mask = (x >= SEARCH_LO_M) & (x <= SEARCH_HI_M)
x_s    = x[search_mask]
dA_s   = dA[search_mask]
amp_s  = amp_filt[search_mask]
amp_raw_s = amp[search_mask]
idx_s  = np.where(search_mask)[0]

# Positive local maxima of the gradient (amplitude rising)
pos_max = argrelextrema(dA_s, np.greater, order=GRAD_ORDER)[0]

# Threshold the gradient: must be positive and significant
dA_thresh = 0.10 * float(np.max(np.abs(dA_s)))  # at least 10% of peak gradient

candidates = []
for li in pos_max:
    gi = idx_s[li]
    if dA_s[li] < dA_thresh:
        continue
    # Count how many subsequent valid (non-NaN) samples stay above threshold
    subsequent_valid   = [amp[k] for k in range(gi, min(gi+30, len(x))) if not np.isnan(amp[k])]
    n_bright_after     = sum(1 for v in subsequent_valid[:MIN_BRIGHT_CONSECUTIVE+5]
                             if v > transition_thresh)
    dist_from_terminus = float(x[-1] - x[gi])
    candidates.append({
        "global_idx":        gi,
        "km":                round(float(x[gi] / 1000), 2),
        "amp_raw_dB":        round(float(amp[gi]), 2) if not np.isnan(amp[gi]) else None,
        "amp_filt_dB":       round(float(amp_filt[gi]), 2),
        "gradient_dBm":      round(float(dA[gi]), 8),
        "hbed_m":            round(float(hbed[gi]), 1) if not np.isnan(hbed[gi]) else None,
        "n_bright_after":    n_bright_after,
        "above_thresh":      bool(amp_filt[gi] > transition_thresh),
        "dist_from_terminus_km": round(dist_from_terminus / 1000, 2),
    })

# Select the grounding point: the candidate with the largest gradient that
# has at least MIN_BRIGHT_CONSECUTIVE bright samples after it
confirmed = [c for c in candidates if c["n_bright_after"] >= MIN_BRIGHT_CONSECUTIVE]
if confirmed:
    gp = max(confirmed, key=lambda c: c["gradient_dBm"])
else:
    # Fallback: largest gradient maximum in search window
    gp = max(candidates, key=lambda c: c["gradient_dBm"]) if candidates else None
    print("Warning: no confirmed grounding point -- using steepest gradient maximum.")

if gp is None:
    raise RuntimeError("No grounding point candidates found in search window.")

print(f"\n==> GROUNDING POINT")
print(f"    Along-track     : {gp['km']:.2f} km")
print(f"    Amp (raw)       : {gp['amp_raw_dB']} dB")
print(f"    Amp (filtered)  : {gp['amp_filt_dB']:.2f} dB")
print(f"    Gradient        : {gp['gradient_dBm']*1e4:.3f} x10^-4 dB/m")
print(f"    hbed            : {gp['hbed_m']} m (WGS84)")
print(f"    Bright points after: {gp['n_bright_after']}")
print(f"    Dist to terminus: {gp['dist_from_terminus_km']:.2f} km")

# -----------------------------------------------------------------------------
# 5. Save outputs
# -----------------------------------------------------------------------------
df_out = bp.copy()
df_out["amp_filtered_dB"]  = amp_filt_out
df_out["amp_gradient_dBm"] = dA
df_out["hbed_m"]           = hbed
df_out["regime"] = np.where(
    nan_mask, "nan",
    np.where(amp > transition_thresh, "bright", "dark")
)
df_out.to_csv("helheim_filtered.csv", index=False)

gp_out = pd.DataFrame([{
    "along_track_m":            gp["km"] * 1000,
    "along_track_km":           gp["km"],
    "amp_raw_dB":               gp["amp_raw_dB"],
    "amp_filtered_dB":          gp["amp_filt_dB"],
    "amp_gradient_dBm":         gp["gradient_dBm"],
    "hbed_m":                   gp["hbed_m"],
    "dark_regime_mean_dB":      round(dark_mean, 2),
    "bright_regime_mean_dB":    round(bright_mean, 2),
    "transition_threshold_dB":  round(transition_thresh, 2),
    "amplitude_contrast_dB":    round(bright_mean - dark_mean, 2),
    "n_bright_confirmed":       gp["n_bright_after"],
    "dist_to_terminus_km":      gp["dist_from_terminus_km"],
}])
gp_out.to_csv("helheim_grounding_point.csv", index=False)
print("\nSaved: helheim_filtered.csv, helheim_grounding_point.csv")

# -----------------------------------------------------------------------------
# 6. Plot -- three-panel figure
# -----------------------------------------------------------------------------
x_km = x / 1000

fig, axes = plt.subplots(3, 1, figsize=(14, 11))
fig.subplots_adjust(hspace=0.45)

C = {"raw": "#b4b2a9", "filt": "#378add", "gp": "#e24b4a",
     "thresh": "#ef9f27", "hbed": "#7f77dd", "dark": "#5f5e5a", "bright": "#1d9e75"}

# Panel A: full amplitude transect
ax = axes[0]
ax.scatter(x_km, amp, color=C["raw"], s=6, alpha=0.6, zorder=2, label="Raw bed power")
ax.plot(x_km, amp_filt_out, color=C["filt"], lw=1.8, zorder=3,
        label=f"Butterworth filtered (order={AMP_BUTTER_ORDER}, $W_n$={AMP_BUTTER_WN})")
ax.axhline(transition_thresh, color=C["thresh"], lw=1.5, ls="--",
           label=f"Transition threshold: {transition_thresh:.1f} dB")
ax.axhline(dark_mean,   color=C["dark"],   lw=1, ls=":",
           label=f"Dark mean: {dark_mean:.1f} dB")
ax.axhline(bright_mean, color=C["bright"], lw=1, ls=":",
           label=f"Bright mean: {bright_mean:.1f} dB")
ax.axvline(gp["km"], color=C["gp"], lw=2,
           label=f"Grounding point: {gp['km']:.2f} km  ({gp['amp_filt_dB']:.1f} dB)")
# Shade NaN regions
nan_regions = []
in_nan = False
for i, v in enumerate(nan_mask):
    if v and not in_nan:
        nan_start = x_km[i]; in_nan = True
    elif not v and in_nan:
        nan_regions.append((nan_start, x_km[i-1])); in_nan = False
if in_nan: nan_regions.append((nan_start, x_km[-1]))
for lo, hi in nan_regions:
    ax.axvspan(lo, hi, alpha=0.12, color="gray", zorder=1)
ax.set_ylabel("Bed power (dB)")
ax.set_xlabel("Along-track (km)")
ax.legend(fontsize=8, loc="lower left", ncol=2)
ax.grid(True, alpha=0.2)
ax.set_title(f"(a) Bed power -- Butterworth filtered (order={AMP_BUTTER_ORDER}, $W_n$={AMP_BUTTER_WN})")

# Panel B: bed elevation
ax = axes[1]
ax.plot(x_km, hbed, color=C["hbed"], lw=1.5, alpha=0.8, label="Bed elevation (WGS84)")
ax.axhline(0, color="steelblue", lw=1, ls="--", alpha=0.6, label="Sea level (0 m)")
ax.axvline(gp["km"], color=C["gp"], lw=2, label=f"GP: {gp['km']:.2f} km")
for lo, hi in nan_regions:
    ax.axvspan(lo, hi, alpha=0.12, color="gray", zorder=1)
ax.fill_between(x_km, hbed, 0, where=(hbed < 0), alpha=0.15, color="steelblue",
                label="Below sea level")
ax.set_ylabel("Bed elevation (m, WGS84)")
ax.set_xlabel("Along-track (km)")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.2)
ax.set_title("(b) Bed elevation -- no erf fit (non-monotonic tidewater glacier geometry)")

# Panel C: gradient in search window
ax = axes[2]
x_s_km = x_s / 1000
ax.plot(x_s_km, dA_s * 1e4, color=C["filt"], lw=1.5, label="dA/dx (amplitude gradient)")
ax.fill_between(x_s_km, dA_s * 1e4, 0, where=(dA_s > 0), alpha=0.12, color=C["filt"])
ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
ax.axhline(dA_thresh * 1e4, color=C["thresh"], lw=1, ls="--",
           label=f"Gradient threshold: {dA_thresh*1e4:.1f} x10$^{{-4}}$")
ax.axvline(gp["km"], color=C["gp"], lw=2, label=f"GP: {gp['km']:.2f} km")

for c in candidates:
    confirmed_c = c["n_bright_after"] >= MIN_BRIGHT_CONSECUTIVE
    ax.scatter(c["km"], c["gradient_dBm"] * 1e4,
               s=80 if confirmed_c else 30,
               color=C["gp"] if confirmed_c else "#f09595",
               edgecolors="#a32d2d" if confirmed_c else C["gp"],
               linewidths=1.2, zorder=5)
    if confirmed_c and c["gradient_dBm"] == gp["gradient_dBm"]:
        ax.annotate(f"GP\n{c['km']:.1f} km",
                    (c["km"], c["gradient_dBm"] * 1e4),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=9, color="#a32d2d")

ax.set_xlim(SEARCH_LO_M / 1000 - 2, SEARCH_HI_M / 1000 + 1)
ax.set_ylabel("x10$^{-4}$ dB/m")
ax.set_xlabel("Along-track (km)")
p1 = mpatches.Patch(color=C["gp"],   label="Confirmed GP candidate")
p2 = mpatches.Patch(color="#f09595", label="Unconfirmed candidate")
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles + [p1, p2], fontsize=9)
ax.grid(True, alpha=0.2)
ax.set_title("(c) Amplitude gradient -- search window for grounding point")

fig.suptitle(
    "Helheim Glacier grounding point detection  |  Tidewater glacier method\n"
    "Amplitude regime contrast + steepest dark->bright transition",
    fontsize=11, y=1.01,
)
plt.tight_layout()
# plt.savefig("helheim_grounding_detection.png", dpi=150, bbox_inches="tight")
# plt.show()
# print("Saved: helheim_grounding_detection.png")