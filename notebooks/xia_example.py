"""
Grounding point extraction from airborne IPR bed power
Following: Xia et al. (2025), IEEE TGRS 63, doi:10.1109/TGRS.2025.3620827
"Amery Ice Shelf Grounding Line Datapoints Automated Extraction
 From Airborne Ice-Penetrating Radar"

Paper workflow (Fig. 3 & Section III-B):
  1. Butterworth low-pass filter on amplitude  : order=5, Wn=0.15
  2. Butterworth low-pass filter on bed elev.  : order=5, Wn=0.09  (requires hbed)
  3. Error function fit to smoothed elevation  → 3rd derivative zeros → topo break
  4. 1st derivative of smoothed amplitude      → local maxima
  5. Threshold constraints (T1–T4) on the maxima
  6. Grounding point = seaward local maximum closest to the topo break

NOTE: This CSV contains bed_power_dB only (no bed elevation).
      Steps 1, 4–6 are fully implemented.
      Step 2–3 (elevation branch) is scaffolded — supply hbed to activate it.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, argrelextrema
from scipy.optimize import curve_fit
from scipy.special import erf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("bed_power_concat.csv")
x   = df["along_track"].values      # along-track distance (m)
amp = df["bed_power_dB"].values      # corrected bed echo amplitude (dB)

# Optional: supply bed elevation array to enable the elevation branch
# hbed = df["bed_elevation_m"].values   # uncomment if available

# ── 2. Interpolate NaNs (data gaps) ──────────────────────────────────────────
amp_interp = pd.Series(amp).interpolate(method="linear").ffill().bfill().values
nan_mask   = np.isnan(amp)

# ── 3. Butterworth low-pass filter — AMPLITUDE: order=5, Wn=0.15 ─────────────
b_a, a_a     = butter(5, 0.15, btype="low", analog=False)
amp_filt     = filtfilt(b_a, a_a, amp_interp)   # zero-phase
amp_filt_out = amp_filt.copy()
amp_filt_out[nan_mask] = np.nan

# ── 4. [ELEVATION BRANCH — activate when hbed is available] ──────────────────
# b_e, a_e  = butter(5, 0.09, btype="low", analog=False)
# hbed_filt = filtfilt(b_e, a_e, hbed_interp)
#
# def erf_model(t, A, x0, w, B):
#     return A * erf((t - x0) / w) + B
#
# popt, _ = curve_fit(erf_model, x / 1000, hbed_filt, p0=[...])
# A, x0_km, w_km, B = popt
# # 3rd derivative of erf: zeros at x0 ± w/sqrt(2)
# # The seaward inflection (abrupt slope break) is at:
# topo_break_km = x0_km + w_km / np.sqrt(2)

# ── 5. Find grounding zone center from amplitude gradient ─────────────────────
# (Substitute for elevation branch when hbed unavailable)
amp_norm   = amp_filt - np.max(amp_filt)   # normalize: brightest = 0 dB
dA         = np.gradient(amp_norm, x)      # 1st derivative (dB/m)

# Locate zone of maximum amplitude rate-of-change via rolling mean of gradient
rolling_dA    = pd.Series(dA).rolling(20, center=True, min_periods=5).mean().values
gz_center_idx = int(np.argmax(rolling_dA))
gz_center_km  = float(x[gz_center_idx] / 1000)

# Define search window (±60 km around gradient peak)
half_win_m = 60_000
lo_idx = int(np.searchsorted(x, x[gz_center_idx] - half_win_m))
hi_idx = int(np.searchsorted(x, x[gz_center_idx] + half_win_m))
x_win     = x[lo_idx:hi_idx]
amp_win   = amp_filt[lo_idx:hi_idx]
dA_win    = dA[lo_idx:hi_idx]
x_win_km  = x_win / 1000

# Error function fit to amplitude (proxy for elevation break when hbed absent)
def erf_model(t, A, x0, w, B):
    return A * erf((t - x0) / w) + B

A0 = (amp_win.max() - amp_win.min()) / 2
p0 = [A0, gz_center_km, 8.0, amp_win.mean()]
popt, _ = curve_fit(erf_model, x_win_km, amp_win, p0=p0, maxfev=5000)
amp_erf     = erf_model(x_win_km, *popt)
A_fit, x0_fit, w_fit, B_fit = popt
# Seaward inflection of the erf (analogous to topo break from elevation)
topo_break_km = float(x0_fit + w_fit / np.sqrt(2))

print(f"Grounding zone center (amplitude gradient peak) : {gz_center_km:.2f} km")
print(f"Error function inflection (proxy topo break)    : {topo_break_km:.2f} km")

# ── 6. Local maxima of amplitude gradient → apply thresholds ─────────────────
pos_max_local = argrelextrema(dA_win, np.greater, order=3)[0]

amp_win_max   = float(amp_win.max())
amp_win_range = float(amp_win.max() - amp_win.min())
dA_win_max    = float(np.abs(dA_win).max())

candidates = []
for li in pos_max_local:
    gi  = lo_idx + int(li)
    lo  = max(0, int(li) - 15)
    hi  = min(len(x_win), int(li) + 15)

    # Paper thresholds (Section III-B), applied within the local search window:
    # T1: amplitude at extremum >= 50% of window-maximum amplitude
    t1 = bool(amp_win[li] >= 0.50 * amp_win_max)
    # T2: amplitude range in interval >= 45% of window amplitude range
    t2 = bool((amp_win[lo:hi].max() - amp_win[lo:hi].min()) >= 0.45 * amp_win_range)
    # T3: |gradient| at extremum >= 62% of window-maximum |gradient|
    t3 = bool(abs(dA_win[li]) >= 0.62 * dA_win_max)
    # T4: gradient range in interval >= 1/3 of window-maximum |gradient|
    t4 = bool((dA_win[lo:hi].max() - dA_win[lo:hi].min()) >= (1 / 3) * dA_win_max)

    dist_km = float(abs(x_win[li] / 1000 - topo_break_km))
    candidates.append({
        "global_idx": gi,
        "km": round(float(x[gi] / 1000), 2),
        "amp_filt_dB": round(float(amp_filt[gi]), 2),
        "gradient_dBm": round(float(dA[gi]), 8),
        "T1": t1, "T2": t2, "T3": t3, "T4": t4,
        "all_pass": bool(t1 and t2 and t3 and t4),
        "dist_to_break_km": round(dist_km, 2),
    })

# Progressive threshold relaxation (paper Section III-B):
# Start strict (T1–T4), then lower progressively until candidates emerge.
gp = None
for t1f, t2f, t3f, t4f in [
    (True,  True,  True,  True),   # all four
    (True,  False, True,  True),   # relax T2
    (True,  False, True,  False),  # relax T2, T4
    (False, False, True,  False),  # T3 only (steepest gradient)
]:
    pool = [c for c in candidates
            if (not t1f or c["T1"]) and (not t2f or c["T2"])
            and (not t3f or c["T3"]) and (not t4f or c["T4"])]
    if pool:
        # Among qualifying candidates, select the seaward one closest to topo break
        seaward = [c for c in pool if c["km"] >= topo_break_km - 15]
        gp = min(seaward if seaward else pool, key=lambda c: c["dist_to_break_km"])
        break

if gp is None:
    gp = min(candidates, key=lambda c: c["dist_to_break_km"])
    print("Warning: no candidate passed any threshold — using closest to topo break.")

print(f"\n==> GROUNDING POINT")
print(f"    Position      : {gp['km']:.2f} km along-track")
print(f"    Filtered amp  : {gp['amp_filt_dB']:.2f} dB")
print(f"    Gradient      : {gp['gradient_dBm']*1e4:.3f} ×10⁻⁴ dB/m")
print(f"    T1={gp['T1']} T2={gp['T2']} T3={gp['T3']} T4={gp['T4']}")
print(f"    Dist to break : {gp['dist_to_break_km']:.2f} km")

# ── 7. Save results ───────────────────────────────────────────────────────────
df_out = df.copy()
df_out["amp_filtered_dB"]   = amp_filt_out
df_out["amp_gradient_dBm"]  = dA
# df_out.to_csv("bed_power_filtered.csv", index=False)

gp_df = pd.DataFrame([{
    "along_track_m":     gp["km"] * 1000,
    "along_track_km":    gp["km"],
    "amp_filtered_dB":   gp["amp_filt_dB"],
    "gradient_dBm":      gp["gradient_dBm"],
    "erf_center_km":     round(x0_fit, 2),
    "topo_break_km":     round(topo_break_km, 2),
}])
# gp_df.to_csv("grounding_point.csv", index=False)
print("\nSaved: bed_power_filtered.csv, grounding_point.csv")

# ── 8. Plot (Fig. 7 style from paper) ────────────────────────────────────────
x_km = x / 1000
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False,
                                gridspec_kw={"height_ratios": [3, 2.5]})
fig.subplots_adjust(hspace=0.35)

# ── Panel 1: full transect amplitude ─────────────────────────────────────────
ax1.plot(x_km, amp, color="#b4b2a9", lw=0.7, alpha=0.7, label="Raw bed power")
ax1.plot(x_km, amp_filt_out, color="#378add", lw=1.8, label="Butterworth filtered (order=5, Wn=0.15)")
ax1.plot(x_win_km, amp_erf, color="#1d9e75", lw=1.5, ls="--", label="Error function fit")
ax1.axvline(topo_break_km, color="#ef9f27", lw=1.5, ls="--",
            label=f"Topo break (erf 3rd deriv): {topo_break_km:.1f} km")
ax1.axvline(gp["km"], color="#e24b4a", lw=2,
            label=f"Grounding point: {gp['km']:.2f} km  ({gp['amp_filt_dB']:.1f} dB)")
ax1.axvspan(x_win[0] / 1000, x_win[-1] / 1000, alpha=0.06, color="#ef9f27")
ax1.set_ylabel("Bed power (dB)")
ax1.legend(fontsize=9, loc="lower right")
ax1.grid(True, alpha=0.2)
ax1.set_title("Grounding point extraction — Xia et al. (2025) method\n"
              "Butterworth (order=5, Wn=0.15) + error function + gradient thresholds")
ax1.set_xlabel("Along-track (km)")

# ── Panel 2: zoomed gradient in grounding zone ────────────────────────────────
ax2.plot(x_win_km, dA_win * 1e4, color="#1d9e75", lw=1.5,
         label="dA/dx (amplitude gradient)")
ax2.fill_between(x_win_km, dA_win * 1e4, 0,
                 where=(dA_win > 0), alpha=0.08, color="#1d9e75")
ax2.axhline(0, color="gray", lw=0.8, alpha=0.5)

# Plot all candidates
for c in candidates:
    km = c["km"]
    dA_val = c["gradient_dBm"] * 1e4
    col = "#e24b4a" if c["T3"] else "rgba(226,75,74,0.35)"
    sz  = 60 if c["T3"] else 25
    ax2.scatter(km, dA_val, s=sz, color="#e24b4a" if c["T3"] else "#f09595",
                zorder=5, edgecolors="#a32d2d" if c["T3"] else "#e24b4a", linewidths=1)
    if c["T3"]:
        ax2.annotate(f"GP\n{km:.1f} km", (km, dA_val),
                     textcoords="offset points", xytext=(6, 4),
                     fontsize=9, color="#a32d2d")

ax2.axvline(topo_break_km, color="#ef9f27", lw=1.5, ls="--")
ax2.axvline(gp["km"], color="#e24b4a", lw=2)
ax2.set_xlabel("Along-track (km)")
ax2.set_ylabel("×10⁻⁴ dB/m")
ax2.set_title("Amplitude gradient — grounding zone detail")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)

# Legend for scatter
gp_patch = mpatches.Patch(color="#e24b4a", label="Candidate (T3 pass)")
other_patch = mpatches.Patch(color="#f09595", label="Candidate (T3 fail)")
ax2.legend(handles=[gp_patch, other_patch], fontsize=9, loc="upper left")

plt.tight_layout()
# plt.savefig("grounding_point_extraction.png", dpi=150)
plt.show()
print("Saved: grounding_point_extraction.png")