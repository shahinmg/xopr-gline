"""
Three-panel grounding point profile figure
==========================================
Produces a publication-style figure showing:
  Panel A  Ice surface + bed elevation, ice column fill,
           bed coloured by hydrostatic flotation state,
           Butterworth-smoothed bed, erf fit, grounding point
  Panel B  Ice thickness along-track
  Panel C  Bed power (raw + Butterworth filtered), grounding point

Inputs (CSV)
------------
  bed_power   : slow_time, along_track (m), bed_power_dB
  bed_elev    : slow_time, along_track (m), wgs84 (m)  -- bed layer picks
  ice_surface : slow_time, along_track (m), wgs84 (m)  -- surface picks

Grounding-point detection
-------------------------
  Elevation branch : Butterworth(5, 0.09) on hbed -> erf fit ->
                     seaward 3rd-derivative zero (topo break)
  Amplitude branch : Butterworth(5, 0.15) -> normalize -> gradient ->
                     local maxima -> T1-T4 thresholds (Xia et al. 2025)
  GP               = amplitude candidate seaward of and closest to topo break

Usage
-----
  python plot_grounding_profile.py \\
      --power   bed_power_concat.csv \\
      --bed     surface_bottom.csv \\
      --surface ice_surface.csv \\
      [--label  "Glacier Name"] \\
      [--out    output_figure.png]

Reference
---------
  Xia et al. (2025), IEEE TGRS 63, doi:10.1109/TGRS.2025.3620827
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.signal import butter, filtfilt, argrelextrema
from scipy.optimize import curve_fit
from scipy.special import erf
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# SETTINGS — edit these to tune the detection
# =============================================================================
AMP_ORDER   = 5;  AMP_WN   = 0.15   # Butterworth: amplitude branch
ELEV_ORDER  = 5;  ELEV_WN  = 0.09   # Butterworth: elevation branch
ERF_WIN_KM  = 80                     # half-width of erf fitting window (km)
AMP_SEARCH_HALFWIN_KM = 60           # search window around topo break (km)
T1 = 0.50; T2 = 0.45                 # amplitude threshold fractions
T3 = 0.62; T4 = 1 / 3               # gradient threshold fractions
TIDAL_UNC_KM = 0.5                   # tidal uncertainty to shade (km)

RHO_ICE = 917; RHO_SW = 1028        # densities for flotation check

# Flotation thresholds (metres above predicted freeboard)
GROUNDED_THRESH  =  30   # delta > 30  m  -> grounded (brown)
FLOATING_THRESH  =   0   # delta < 0   m  -> floating (blue)
# Between 0 and 30 m                      -> near flotation (orange)

# Colours
C = dict(
    bg       = "#f6f4f1",
    surf     = "#4a6fa5",
    bed_raw  = "#c4b49a",
    bed_sm   = "#7a5230",
    bed_erf  = "#1d9e75",
    ice      = "#ccdff5",
    ocean    = "#3a7abf",
    grd_fill = "#a68a5b",
    grounded = "#8b6f47",
    near_flt = "#f0a500",
    floating = "#3a7abf",
    gp       = "#d94f3d",
    topo     = "#ef9f27",
    amp_raw  = "#b4b2a9",
    amp_filt = "#378add",
    thick    = "#7b5ea7",
    text     = "#1a1a1a",
)

# =============================================================================
# STEP 1 — Load and align data
# =============================================================================
def load_data(power_path, bed_path, surface_path):
    bp  = pd.read_csv(power_path)
    sb  = pd.read_csv(bed_path)
    sf  = pd.read_csv(surface_path)

    sb_c = sb.dropna(subset=["wgs84"])
    sf_c = sf.dropna(subset=["wgs84"]) if sf["wgs84"].isna().any() else sf

    x_bp  = bp["along_track"].values / 1000   # km
    amp   = bp["bed_power_dB"].values

    x_bed = sb_c["along_track"].values / 1000
    h_bed = sb_c["wgs84"].values

    x_sf  = sf_c["along_track"].values / 1000
    h_sf  = sf_c["wgs84"].values

    # Interpolate surface onto bed grid for thickness + flotation check
    h_sf_on_bed = np.interp(x_bed, x_sf, h_sf)
    H = h_sf_on_bed - h_bed

    # Flotation state
    freeboard_pred = H * (1 - RHO_ICE / RHO_SW)
    delta_free     = h_sf_on_bed - freeboard_pred

    return dict(
        x_bp=x_bp, amp=amp,
        x_bed=x_bed, h_bed=h_bed,
        x_sf=x_sf, h_sf=h_sf,
        h_sf_on_bed=h_sf_on_bed,
        H=H, delta_free=delta_free,
    )

# =============================================================================
# STEP 2 — Filter amplitude and bed elevation
# =============================================================================
def filter_data(d):
    x_bp  = d["x_bp"];  amp   = d["amp"]
    x_bed = d["x_bed"]; h_bed = d["h_bed"]
    x_sf  = d["x_sf"];  h_sf  = d["h_sf"]

    nan_mask = np.isnan(amp)

    # Amplitude: interpolate NaNs, Butterworth, restore NaNs
    amp_i    = pd.Series(amp).interpolate(method="linear").ffill().bfill().values
    b_a, a_a = butter(AMP_ORDER, AMP_WN, btype="low", analog=False)
    amp_f    = filtfilt(b_a, a_a, amp_i)
    amp_f_out = amp_f.copy(); amp_f_out[nan_mask] = np.nan

    # Bed elevation: interpolate onto amp grid, Butterworth
    hbed_bp = np.interp(x_bp, x_bed, h_bed, left=np.nan, right=np.nan)
    hbed_i  = pd.Series(hbed_bp).interpolate(method="linear").ffill().bfill().values
    b_e, a_e = butter(ELEV_ORDER, ELEV_WN, btype="low", analog=False)
    hbed_bw  = filtfilt(b_e, a_e, hbed_i)

    # Gradient of normalised filtered amplitude
    dA = np.gradient(amp_f - np.max(amp_f), x_bp)

    d.update(dict(
        nan_mask=nan_mask,
        amp_f=amp_f, amp_f_out=amp_f_out,
        hbed_bw=hbed_bw, dA=dA,
    ))
    return d

# =============================================================================
# STEP 3 — Elevation branch: erf fit -> topo break
# =============================================================================
def elevation_branch(d):
    x_bp = d["x_bp"]; hbed_bw = d["hbed_bw"]

    def erf_model(t, A, x0, w, B):
        return A * erf((t - x0) / w) + B

    mi   = int(np.argmin(hbed_bw))
    gz   = float(x_bp[mi])
    lo_e = int(np.searchsorted(x_bp, gz - ERF_WIN_KM))
    hi_e = int(np.searchsorted(x_bp, gz + ERF_WIN_KM))
    x_e  = x_bp[lo_e:hi_e]
    h_w  = hbed_bw[lo_e:hi_e]
    A0   = (h_w.max() - h_w.min()) / 2

    popt, _ = curve_fit(erf_model, x_e, h_w,
                        p0=[A0, gz, 15.0, h_w.mean()], maxfev=10_000)
    A_e, x0_e, w_e, B_e = popt
    hbed_erf = erf_model(x_e, *popt)
    topo_km  = float(x0_e + w_e / np.sqrt(2))
    land_km  = float(x0_e - w_e / np.sqrt(2))

    d.update(dict(
        x_e=x_e, hbed_erf=hbed_erf, popt=popt,
        topo_km=topo_km, land_km=land_km,
    ))
    return d

# =============================================================================
# STEP 4 — Amplitude branch: local maxima + T1-T4 thresholds
# =============================================================================
def amplitude_branch(d):
    x_bp     = d["x_bp"]
    amp_f    = d["amp_f"]
    nan_mask = d["nan_mask"]
    dA       = d["dA"]
    topo_km  = d["topo_km"]

    lo_a = int(np.searchsorted(x_bp, topo_km - AMP_SEARCH_HALFWIN_KM))
    hi_a = int(np.searchsorted(x_bp, topo_km + AMP_SEARCH_HALFWIN_KM))

    x_a   = x_bp[lo_a:hi_a]
    dA_a  = dA[lo_a:hi_a]
    amp_a = amp_f[lo_a:hi_a]

    pos_max  = argrelextrema(dA_a, np.greater, order=3)[0]
    aw_max   = float(amp_a.max())
    aw_range = float(amp_a.max() - amp_a.min())
    dw_max   = float(np.abs(dA_a).max())

    candidates = []
    for li in pos_max:
        gi  = lo_a + int(li)
        lo  = max(0, int(li) - 15)
        hi  = min(len(x_a), int(li) + 15)
        t1b = bool(amp_a[li] >= T1 * aw_max)
        t2b = bool((amp_a[lo:hi].max() - amp_a[lo:hi].min()) >= T2 * aw_range)
        t3b = bool(abs(dA_a[li]) >= T3 * dw_max)
        t4b = bool((dA_a[lo:hi].max() - dA_a[lo:hi].min()) >= T4 * dw_max)
        dist = float(abs(x_a[li] - topo_km))
        candidates.append(dict(
            gi=gi, km=round(float(x_bp[gi]), 2),
            amp=round(float(amp_f[gi]), 2),
            dA_val=round(float(dA[gi]), 8),
            t1=t1b, t2=t2b, t3=t3b, t4=t4b,
            dist=round(dist, 2),
        ))

    # Progressive threshold relaxation
    gp = None
    for t1f, t2f, t3f, t4f in [
        (1, 1, 1, 1), (1, 0, 1, 1), (1, 0, 1, 0), (0, 0, 1, 0)
    ]:
        pool = [c for c in candidates
                if (not t1f or c["t1"]) and (not t2f or c["t2"])
                and (not t3f or c["t3"]) and (not t4f or c["t4"])]
        if pool:
            sw = [c for c in pool if c["km"] >= topo_km - 15]
            gp = min(sw if sw else pool, key=lambda c: c["dist"])
            break
    if gp is None and candidates:
        gp = min(candidates, key=lambda c: c["dist"])

    d["gp"] = gp
    return d

# =============================================================================
# STEP 5 — NaN block finder (for shading)
# =============================================================================
def find_nan_blocks(x, nan_mask):
    diff  = np.diff(nan_mask.astype(int))
    starts = list(np.where(diff == 1)[0] + 1)
    ends   = list(np.where(diff == -1)[0] + 1)
    if nan_mask[0]:  starts = [0] + starts
    if nan_mask[-1]: ends   = ends + [len(x)]
    return [(x[s], x[min(e, len(x)-1)]) for s, e in zip(starts, ends)]

# =============================================================================
# STEP 6 — Plot
# =============================================================================
def plot(d, label="", out_path="grounding_profile.png"):
    x_bp  = d["x_bp"];  amp   = d["amp"]
    x_bed = d["x_bed"]; h_bed = d["h_bed"]
    x_sf  = d["x_sf"];  h_sf  = d["h_sf"]
    h_sf_on_bed = d["h_sf_on_bed"]
    H           = d["H"]
    delta_free  = d["delta_free"]
    amp_f_out   = d["amp_f_out"]
    hbed_bw     = d["hbed_bw"]
    dA          = d["dA"]
    nan_mask    = d["nan_mask"]
    x_e         = d["x_e"]
    hbed_erf    = d["hbed_erf"]
    popt        = d["popt"]
    topo_km     = d["topo_km"]
    land_km     = d["land_km"]
    gp          = d["gp"]

    GP_KM   = gp["km"]
    GP_BED  = float(np.interp(GP_KM, x_bp, hbed_bw))
    GP_SURF = float(np.interp(GP_KM, x_sf, h_sf))
    GP_H    = float(np.interp(GP_KM, x_bed, H))
    GP_AMP  = float(np.interp(GP_KM, x_bp, d["amp_f"]))

    nan_blocks = find_nan_blocks(x_bp, nan_mask)

    # ── Helpers ──────────────────────────────────────────────────────────
    def shade_nans(ax):
        for lo, hi in nan_blocks:
            if hi - lo > 1:
                ax.axvspan(lo, hi, alpha=0.09, color="gray", zorder=0)

    def draw_vlines(ax):
        ax.axvspan(GP_KM - TIDAL_UNC_KM, GP_KM + TIDAL_UNC_KM,
                   color=C["gp"], alpha=0.12, zorder=2)
        ax.axvline(GP_KM,   color=C["gp"],  lw=1.7, ls="--", alpha=0.75, zorder=3)
        ax.axvline(topo_km, color=C["topo"], lw=1.4, ls="--", alpha=0.80, zorder=3)
        ax.axvline(land_km, color=C["topo"], lw=1.0, ls=":",  alpha=0.65, zorder=3)

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12), facecolor=C["bg"])
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3.2, 1.4, 1.4], hspace=0.07,
                            left=0.07, right=0.97, top=0.94, bottom=0.07)
    ax_prof  = fig.add_subplot(gs[0])
    ax_thick = fig.add_subplot(gs[1], sharex=ax_prof)
    ax_amp   = fig.add_subplot(gs[2], sharex=ax_prof)

    for ax in [ax_prof, ax_thick, ax_amp]:
        ax.set_facecolor(C["bg"])
        for sp in ax.spines.values():
            sp.set_color("#cccccc")

    # ── Panel A: surface + bed profile ────────────────────────────────────
    ax = ax_prof
    shade_nans(ax)
    draw_vlines(ax)

    # Ocean cavity fill
    ax.fill_between(x_bed, h_bed.min() - 50, np.minimum(h_bed, 0),
                    color=C["ocean"], alpha=0.20, zorder=0,
                    label="Below sea level / ocean cavity")

    # Bedrock fill
    ax.fill_between(x_bed, h_bed.min() - 50, h_bed,
                    color=C["grd_fill"], alpha=0.45, zorder=1)

    # Ice column fill (surface to bed on the dense surface grid)
    h_bed_on_sf = np.interp(x_sf, x_bed, h_bed, left=np.nan, right=np.nan)
    ax.fill_between(x_sf, h_bed_on_sf, h_sf,
                    where=~np.isnan(h_bed_on_sf),
                    color=C["ice"], alpha=0.80, zorder=2, label="Ice column")

    # Bed coloured by flotation state
    for mask, col, lbl in [
        (delta_free >  GROUNDED_THRESH,                          C["grounded"], "Grounded"),
        ((delta_free >= FLOATING_THRESH) & (delta_free <= GROUNDED_THRESH),
                                                                  C["near_flt"], "Near flotation (Δ < 30 m)"),
        (delta_free <  FLOATING_THRESH,                          C["floating"], "At / below flotation"),
    ]:
        ax.scatter(x_bed[mask], h_bed[mask], s=3, color=col,
                   alpha=0.7, zorder=4, label=lbl)

    # Smoothed bed
    ax.plot(x_bp, hbed_bw, color=C["bed_sm"], lw=1.8, zorder=5,
            label=f"Bed smoothed (Butterworth order={ELEV_ORDER}, $W_n$={ELEV_WN})")

    # Erf fit
    ax.plot(x_e, hbed_erf, color=C["bed_erf"], lw=1.8, ls="--", zorder=6,
            label=f"Erf fit (c={popt[1]:.1f} km, w={popt[2]:.1f} km)")

    # Ice surface
    ax.plot(x_sf, h_sf, color=C["surf"], lw=1.8, zorder=5,
            label="Ice surface")

    # Sea level
    ax.axhline(0, color=C["ocean"], lw=1.0, ls=":", alpha=0.9, zorder=4)
    ax.text(4, 8, "Sea level", fontsize=7.5, color=C["ocean"], alpha=0.9)

    # Grounding point markers (surface dot + bed triangle)
    ax.scatter([GP_KM], [GP_SURF], s=180, color=C["gp"], zorder=10,
               edgecolors="white", linewidths=1.8,
               label=f"Grounding point: {GP_KM:.2f} km (±{TIDAL_UNC_KM} km)")
    ax.scatter([GP_KM], [GP_BED],  s=140, color=C["gp"], zorder=10,
               edgecolors="white", linewidths=1.5, marker="v")

    # Annotation
    ax.annotate(
        f"Grounding point\n{GP_KM:.2f} km\n"
        f"Surf: {GP_SURF:.0f} m\nBed: {GP_BED:.0f} m\nH: {GP_H:.0f} m",
        xy=(GP_KM, GP_SURF),
        xytext=(GP_KM - max(40, x_bp[-1] * 0.17), GP_SURF - h_sf.max() * 0.16),
        fontsize=9, color=C["gp"], fontweight="semibold",
        arrowprops=dict(arrowstyle="->", color=C["gp"], lw=1.3),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=C["gp"], alpha=0.93, lw=1.2),
        zorder=11,
    )

    # Topo break annotation
    ax.annotate(
        f"Topo break\n{topo_km:.1f} km",
        xy=(topo_km, GP_BED - 50),
        xytext=(topo_km + x_bp[-1] * 0.025, GP_BED - 200),
        fontsize=8.5, color=C["topo"],
        arrowprops=dict(arrowstyle="->", color=C["topo"], lw=1.1),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=C["topo"], alpha=0.85, lw=1.0),
    )

    ax.set_ylabel("Elevation (m, WGS84)", fontsize=11, color=C["text"])
    ax.set_ylim(h_bed.min() - 80, h_sf.max() * 1.05)
    ax.set_xlim(-3, x_bp[-1] + 5)
    ax.tick_params(labelbottom=False, colors=C["text"], labelsize=9)
    ax.legend(fontsize=8, loc="upper right", ncol=2,
              framealpha=0.92, edgecolor="#cccccc", facecolor="white")
    ax.grid(True, alpha=0.15, lw=0.7)
    ax.set_title(
        f"{label}  |  Surface + bed profile with grounding point\n"
        f"Bed coloured by hydrostatic flotation state  ·  "
        f"Xia et al. (2025) erf topo break + amplitude gradient",
        fontsize=11, color=C["text"], pad=9, fontweight="semibold",
    )

    # ── Panel B: ice thickness ────────────────────────────────────────────
    ax = ax_thick
    shade_nans(ax)
    draw_vlines(ax)

    ax.fill_between(x_bed, 0, H, color=C["thick"], alpha=0.30, zorder=1)
    ax.plot(x_bed, H, color=C["thick"], lw=1.6, zorder=2,
            label="Ice thickness")

    ax.scatter([GP_KM], [GP_H], s=140, color=C["gp"], zorder=5,
               edgecolors="white", linewidths=1.6)
    ax.text(GP_KM - 2, GP_H + H.max() * 0.02, f"{GP_H:.0f} m",
            fontsize=8.5, color=C["gp"], ha="right", fontweight="medium")

    ax.set_ylabel("Ice thickness (m)", fontsize=10, color=C["text"])
    ax.set_ylim(0, H.max() * 1.12)
    ax.tick_params(labelbottom=False, colors=C["text"], labelsize=9)
    ax.legend(fontsize=8.5, loc="upper right",
              framealpha=0.92, edgecolor="#cccccc", facecolor="white")
    ax.grid(True, alpha=0.15, lw=0.7)

    # ── Panel C: bed power ────────────────────────────────────────────────
    ax = ax_amp
    shade_nans(ax)
    draw_vlines(ax)

    ax.scatter(x_bp, amp, s=4, color=C["amp_raw"], alpha=0.5, zorder=2)
    ax.plot(x_bp, amp_f_out, color=C["amp_filt"], lw=1.8, zorder=3,
            label=f"Butterworth (order={AMP_ORDER}, $W_n$={AMP_WN})")

    # Dark / bright regime reference lines
    dark_mean   = float(np.nanmean(d["amp_f"][(x_bp < x_bp[-1] * 0.83) & ~nan_mask]))
    bright_mean = float(np.nanmean(d["amp_f"][(x_bp > x_bp[-1] * 0.97) & ~nan_mask]))
    ax.axhline(dark_mean,   color="#666",       lw=0.9, ls=":", alpha=0.75)
    ax.axhline(bright_mean, color=C["amp_filt"], lw=0.9, ls=":", alpha=0.75)
    ax.text(x_bp[-1] * 0.02, dark_mean   + 1.0, f"Dark (grounded): {dark_mean:.0f} dB",
            fontsize=7.5, color="#555")
    ax.text(x_bp[-1] * 0.02, bright_mean + 1.0, f"Bright (floating): {bright_mean:.0f} dB",
            fontsize=7.5, color=C["amp_filt"])

    ax.scatter([GP_KM], [GP_AMP], s=140, color=C["gp"], zorder=5,
               edgecolors="white", linewidths=1.6,
               label=f"GP: {GP_KM:.2f} km  ({GP_AMP:.0f} dB)")

    leg_extra = [
        Line2D([0],[0], color=C["gp"],  lw=1.7, ls="--",
               label=f"GP: {GP_KM:.2f} km (±{TIDAL_UNC_KM} km)"),
        Line2D([0],[0], color=C["topo"], lw=1.4, ls="--",
               label=f"Topo break: {topo_km:.1f} km"),
        Line2D([0],[0], color=C["topo"], lw=1.0, ls=":",
               label=f"Landward break: {land_km:.1f} km"),
    ]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + leg_extra, fontsize=8, loc="lower right",
              framealpha=0.92, edgecolor="#cccccc", facecolor="white")

    ax.set_ylabel("Bed power (dB)", fontsize=10, color=C["text"])
    ax.set_xlabel("Along-track distance (km)", fontsize=11, color=C["text"])
    ax.set_ylim(np.nanmin(amp) - 5, np.nanmax(amp) + 5)
    ax.tick_params(colors=C["text"], labelsize=9)
    ax.grid(True, alpha=0.15, lw=0.7)

    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=C["bg"])
    print(f"Saved: {out_path}")
    plt.show()

    # ── Print detection summary ───────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Grounding point  : {GP_KM:.2f} km")
    print(f"  Topo break       : {topo_km:.2f} km (seaward)")
    print(f"  Landward break   : {land_km:.2f} km")
    print(f"  Amp at GP        : {GP_AMP:.1f} dB (filtered)")
    print(f"  Bed at GP        : {GP_BED:.0f} m WGS84 (smoothed)")
    print(f"  Surface at GP    : {GP_SURF:.0f} m WGS84")
    print(f"  Ice thickness    : {GP_H:.0f} m")
    print(f"  Tidal uncertainty: ± {TIDAL_UNC_KM} km")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Three-panel grounding point profile figure")
    parser.add_argument("--power",   required=True, help="Bed power CSV")
    parser.add_argument("--bed",     required=True, help="Bed elevation CSV (wgs84)")
    parser.add_argument("--surface", required=True, help="Ice surface CSV (wgs84)")
    parser.add_argument("--label",   default="Glacier",
                        help="Glacier / transect label for title")
    parser.add_argument("--out",     default="grounding_profile.png",
                        help="Output figure path")
    args = parser.parse_args()

    data = load_data(args.power, args.bed, args.surface)
    data = filter_data(data)
    data = elevation_branch(data)
    data = amplitude_branch(data)
    plot(data, label=args.label, out_path=args.out)
