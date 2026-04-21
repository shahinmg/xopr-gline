"""
Grounding Point Detection via Offline Bayesian Changepoint Detection
=====================================================================
Runs the Fearnhead (2006) offline BOCPD with both normalised and raw
(un-normalised) features in a single pass

Usage
-----
  # Petermann (floating→grounded, InSAR GZ known)
  python bocpd_grounding_point_v2.py \\
      --power   petermann_bed_power.csv \\
      --bed     petermann_bottom.csv \\
      --surface petermann_surface.csv \\
      --gz_lo 95.38 --gz_hi 97.89 \\
      --search_lo 75 \\
      --label "Petermann 2010-04-20"

References
----------
  Fearnhead (2006). Statistics and Computing 16(2), 203-213.
  Adams & MacKay (2007). arXiv:0710.3742.
  Xuan & Murphy (2007). ICML, 1055-1062.
  Ciracì et al. (2023). PNAS 120(20). doi:10.1073/pnas.2220924120
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from functools import partial
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore")

try:
    import bayesian_changepoint_detection.offline_changepoint_detection as offcd
except ImportError:
    raise ImportError(
        "Install first:  pip install bayesian-changepoint-detection decorator"
    )

# =============================================================================
# SETTINGS
# =============================================================================
RHO_ICE = 917.0
RHO_SW  = 1028.0

AMP_ORDER, AMP_WN   = 5, 0.15
ELEV_ORDER, ELEV_WN = 5, 0.09
SLOPE_WN            = 0.02

# Floating reference: x < FLOAT_FRAC × gz_lo is used for normalisation
# (only applies to floating→grounded transects; overridden by --ref_lo/ref_hi)
FLOAT_FRAC = 0.73

GRADIENT_MAX_ABS     = 5.0    # dB/km — clip gradient artifacts
GRADIENT_ARTIFACT_LO = 97.5   # km — NaN cluster region (Petermann-specific)
GRADIENT_ARTIFACT_HI = 103.0  # km

TRUNCATE    = -50.0   # log-prob truncation for Fearnhead algorithm
SMOOTH_SIZE = 3       # posterior smoothing window

# Flag threshold: if |MAP_norm - MAP_raw| > this, mark as discordant
DISCORD_THRESHOLD_KM = 3.0

# Colours
C = dict(
    bg      = "#f6f4f1",
    surf    = "#4a6fa5",
    bed_s   = "#7a5230",
    ice     = "#ccdff5",
    ocean   = "#3a7abf",
    grd     = "#a68a5b",
    gz      = "#7b2d8b",
    grad    = "#d94f3d",
    norm    = "#1a8f5a",   # normalised run → green
    raw     = "#d4700a",   # raw run        → orange
    gauss   = "#e87722",
    ifm     = "#9b59b6",
    fcov    = "#16a085",
    amp_f   = "#378add",
    amp_r   = "#b4b2a9",
    dfree   = "#1a6bbf",
    text    = "#1a1a1a",
)


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(power_path, bed_path, surface_path):
    bp = pd.read_csv(power_path)
    bt = pd.read_csv(bed_path).dropna(subset=["wgs84"])
    sf = pd.read_csv(surface_path).dropna(subset=["wgs84"])
    return dict(
        x_bp  = bp["along_track"].values / 1000,
        amp   = bp["bed_power_dB"].values,
        x_bed = bt["along_track"].values / 1000,
        h_bed = bt["wgs84"].values,
        x_sf  = sf["along_track"].values / 1000,
        h_sf  = sf["wgs84"].values,
    )


# =============================================================================
# FEATURE COMPUTATION
# =============================================================================
def compute_features(d):
    """Compute all four features on the bed-power grid. No normalisation here."""
    x_bp, amp     = d["x_bp"], d["amp"]
    x_bed, h_bed  = d["x_bed"], d["h_bed"]
    x_sf,  h_sf   = d["x_sf"],  d["h_sf"]

    nan_mask = np.isnan(amp)

    # Filtered amplitude
    amp_i    = pd.Series(amp).interpolate().ffill().bfill().values
    b, a     = butter(AMP_ORDER, AMP_WN, btype="low", analog=False)
    amp_f    = filtfilt(b, a, amp_i)
    amp_f_out = amp_f.copy(); amp_f_out[nan_mask] = np.nan

    # Amplitude gradient with artifact masking
    dA = np.gradient(amp_f, x_bp)
    dA[(x_bp > GRADIENT_ARTIFACT_LO) & (x_bp < GRADIENT_ARTIFACT_HI)] = np.nan
    dA[np.abs(dA) > GRADIENT_MAX_ABS] = np.nan
    dA = pd.Series(dA).interpolate(limit=5).ffill().bfill().values

    # Flotation residual Δfree
    h_sf_on_bed   = np.interp(x_bed, x_sf, h_sf)
    H_bed         = h_sf_on_bed - h_bed
    b2, a2        = butter(ELEV_ORDER, SLOPE_WN, btype="low", analog=False)
    h_surf_sm     = filtfilt(b2, a2, h_sf_on_bed)
    H_sm          = h_surf_sm - filtfilt(b2, a2, h_bed)
    delta_free_sm = h_surf_sm - H_sm * (1 - RHO_ICE / RHO_SW)
    delta_free_bp = np.interp(x_bp, x_bed, delta_free_sm)

    # Surface slope
    h_sf_bp_sm = filtfilt(b2, a2, np.interp(x_bp, x_sf, h_sf))
    d_surf_bp  = np.gradient(h_sf_bp_sm, x_bp)

    # Smoothed bed for plotting
    hbed_bp  = np.interp(x_bp, x_bed, h_bed, left=np.nan, right=np.nan)
    hbed_i   = pd.Series(hbed_bp).interpolate().ffill().bfill().values
    be, ae   = butter(AMP_ORDER, ELEV_WN, btype="low", analog=False)
    hbed_bw  = filtfilt(be, ae, hbed_i)
    hbed_bw_out = hbed_bw.copy(); hbed_bw_out[np.isnan(hbed_bp)] = np.nan

    d.update(dict(
        nan_mask        = nan_mask,
        amp_f           = amp_f,
        amp_f_out       = amp_f_out,
        dA              = dA,
        delta_free_bp   = delta_free_bp,
        delta_free_bed  = np.interp(x_bed, x_bp, delta_free_bp),
        d_surf_bp       = d_surf_bp,
        H_bp            = np.interp(x_bp, x_bed, H_bed),
        h_sf_bp         = np.interp(x_bp, x_sf, h_sf),
        hbed_bw         = hbed_bw,
        hbed_bw_out     = hbed_bw_out,
        H_bed           = H_bed,
    ))
    return d


# =============================================================================
# CORE BOCPD
# =============================================================================
def run_fearnhead(obs_1d, signal_4d, prior, label=""):
    """
    Run Gaussian + IFM + FullCov models, return per-model and combined posteriors.

    obs_1d   : 1D array (first feature, used for the Gaussian model)
    signal_4d: (n, 4) array of all features (used for IFM and FullCov)
    prior    : prior function (const_prior or geometric_prior)

    Returns dict with cp_gauss, cp_ifm, cp_fcov, cp_comb (all length n-1)
    and MAP statistics.
    """
    _, _, Pcp_g = offcd.offline_changepoint_detection(
        obs_1d, prior, offcd.gaussian_obs_log_likelihood, truncate=TRUNCATE)
    cp_g = np.exp(Pcp_g).sum(0); cp_g /= cp_g.sum()

    _, _, Pcp_i = offcd.offline_changepoint_detection(
        signal_4d, prior, offcd.ifm_obs_log_likelihood, truncate=TRUNCATE)
    cp_i = np.exp(Pcp_i).sum(0); cp_i /= cp_i.sum()

    _, _, Pcp_f = offcd.offline_changepoint_detection(
        signal_4d, prior, offcd.fullcov_obs_log_likelihood, truncate=TRUNCATE)
    cp_f = np.exp(Pcp_f).sum(0); cp_f /= cp_f.sum()

    # Equal-weight geometric mean combination
    log_c = (np.log(cp_g + 1e-15) +
             np.log(cp_i + 1e-15) +
             np.log(cp_f + 1e-15)) / 3.0
    log_c -= np.logaddexp.reduce(log_c)
    cp_c  = uniform_filter1d(np.exp(log_c), size=SMOOTH_SIZE)
    cp_c /= cp_c.sum()

    return dict(cp_gauss=cp_g, cp_ifm=cp_i, cp_fcov=cp_f, cp_comb=cp_c)


def compute_ci(cp_comb, x_cp):
    """MAP and credible intervals from a normalised posterior."""
    n1  = len(x_cp)
    cdf = np.cumsum(cp_comb)
    MAP  = float(x_cp[np.argmax(cp_comb)])
    lo68 = float(x_cp[np.searchsorted(cdf, 0.160)])
    hi68 = float(x_cp[min(np.searchsorted(cdf, 0.840), n1 - 1)])
    lo95 = float(x_cp[np.searchsorted(cdf, 0.025)])
    hi95 = float(x_cp[min(np.searchsorted(cdf, 0.975), n1 - 1)])
    return dict(MAP=MAP, lo68=lo68, hi68=hi68, lo95=lo95, hi95=hi95)


def run_both(d, gz_lo_km, search_lo_km, search_hi_km,
             ref_lo_km=None, ref_hi_km=None):
    """
    Run BOCPD with both normalised and raw features.

    ref_lo_km / ref_hi_km: x range of the stable reference section used for
      normalisation. Default: x < FLOAT_FRAC × gz_lo (floating section).
      For PINNED_GZ transects, pass the grounded dark block bounds instead.
    """
    x_bp = d["x_bp"]

    # Reference section for normalisation
    if ref_lo_km is not None and ref_hi_km is not None:
        ref_mask = (x_bp >= ref_lo_km) & (x_bp <= ref_hi_km) & ~d["nan_mask"]
    else:
        ref_mask = x_bp < FLOAT_FRAC * gz_lo_km

    def norm(f):
        return (f - np.mean(f[ref_mask])) / (np.std(f[ref_mask]) + 1e-9)

    # Raw and normalised feature arrays (full transect length)
    feats_raw  = dict(
        amp_f   = d["amp_f"],
        dA      = d["dA"],
        dfree   = d["delta_free_bp"],
        dsurf   = d["d_surf_bp"],
    )
    feats_norm = {k: norm(v) for k, v in feats_raw.items()}

    # Search window
    win   = (x_bp >= search_lo_km) & (x_bp <= search_hi_km)
    x_win = x_bp[win]
    n     = int(win.sum())
    x_cp  = x_win[:-1]   # posteriors align to x[:-1] (n-1 length)
    prior = partial(offcd.const_prior, l=n + 1)

    results = {}
    for run_label, feats in [("normalised", feats_norm), ("raw", feats_raw)]:
        print(f"  Running {run_label}...")
        signal = np.column_stack([
            feats["amp_f"][win],
            feats["dA"][win],
            feats["dfree"][win],
            feats["dsurf"][win],
        ])
        posteriors = run_fearnhead(feats["dfree"][win], signal, prior,
                                   label=run_label)
        ci = compute_ci(posteriors["cp_comb"], x_cp)
        results[run_label] = dict(
            feats      = feats,
            posteriors = posteriors,
            ci         = ci,
            x_cp       = x_cp,
            x_win      = x_win,
        )

    return results


# =============================================================================
# GRADIENT METHOD 
# =============================================================================
def gradient_gp(d, search_lo_km, search_hi_km):
    """Steepest amplitude gradient in the search window."""
    x_bp  = d["x_bp"]
    dA    = d["dA"].copy()
    dA[np.abs(dA) > GRADIENT_MAX_ABS] = 0.0
    mask  = (x_bp >= search_lo_km) & (x_bp <= search_hi_km)
    # Positive gradient = grounded→floating; negative = floating→grounded
    # Use absolute max to work for both transect directions
    idx   = np.argmax(np.abs(dA[mask]))
    return float(x_bp[mask][idx])


# =============================================================================
# NaN blocks helper
# =============================================================================
def get_nan_blocks(x_bp, nan_mask, min_width=0.3):
    diff_nm = np.diff(nan_mask.astype(int))
    starts  = list(np.where(diff_nm == 1)[0] + 1)
    ends    = list(np.where(diff_nm == -1)[0] + 1)
    if nan_mask[0]:  starts = [0] + starts
    if nan_mask[-1]: ends   = ends + [len(x_bp)]
    return [(x_bp[s], x_bp[min(e, len(x_bp)-1)])
            for s, e in zip(starts, ends)
            if x_bp[min(e, len(x_bp)-1)] - x_bp[s] > min_width]


# =============================================================================
# PLOTTING
# =============================================================================
def _shade_nans(ax, nan_blocks):
    for lo, hi in nan_blocks:
        ax.axvspan(lo, hi, alpha=0.09, color="gray", zorder=0)


def _draw_ref_lines(ax, gz_lo, gz_hi, gp_grad,
                    MAP_norm, ci_norm, MAP_raw, ci_raw,
                    search_lo, search_hi):
    ax.axvspan(search_lo, search_hi, color="#e8f4e8", alpha=0.30, zorder=0)
    if gz_lo and gz_hi:
        ax.axvspan(gz_lo, gz_hi,  color=C["gz"],  alpha=0.20, zorder=2)
        ax.axvline(gz_lo, color=C["gz"],  lw=1.4, ls="--", alpha=0.88, zorder=3)
        ax.axvline(gz_hi, color=C["gz"],  lw=1.4, ls="--", alpha=0.88, zorder=3)
    if gp_grad:
        ax.axvline(gp_grad, color=C["grad"], lw=1.6, ls="--", alpha=0.70, zorder=3)
    # CI bands then MAP lines
    ax.axvspan(ci_norm["lo95"], ci_norm["hi95"], color=C["norm"], alpha=0.07, zorder=1)
    ax.axvspan(ci_norm["lo68"], ci_norm["hi68"], color=C["norm"], alpha=0.15, zorder=1)
    ax.axvspan(ci_raw["lo95"],  ci_raw["hi95"],  color=C["raw"],  alpha=0.07, zorder=1)
    ax.axvspan(ci_raw["lo68"],  ci_raw["hi68"],  color=C["raw"],  alpha=0.12, zorder=1)
    ax.axvline(MAP_norm, color=C["norm"], lw=2.0, ls="-",  alpha=0.88, zorder=4)
    ax.axvline(MAP_raw,  color=C["raw"],  lw=2.0, ls="--", alpha=0.88, zorder=4)


def plot_combined(d, results, label, gz_lo, gz_hi,
                  gp_grad, search_lo, search_hi, out_path):
    """
    4-panel comparison figure showing both normalised and raw runs together.
    """
    x_bp          = d["x_bp"]
    x_bed, h_bed  = d["x_bed"], d["h_bed"]
    x_sf,  h_sf   = d["x_sf"],  d["h_sf"]
    nan_mask       = d["nan_mask"]
    amp            = d["amp"]
    amp_f_out      = d["amp_f_out"]
    delta_free_bed = d["delta_free_bed"]
    hbed_bw_out    = d["hbed_bw_out"]
    H_bp           = d["H_bp"]

    r_n  = results["normalised"]
    r_r  = results["raw"]
    ci_n = r_n["ci"]; MAP_n = ci_n["MAP"]
    ci_r = r_r["ci"]; MAP_r = ci_r["MAP"]
    x_cp = r_n["x_cp"]

    nan_blocks = get_nan_blocks(x_bp, nan_mask)

    def shade(ax): _shade_nans(ax, nan_blocks)
    def ref(ax):   _draw_ref_lines(ax, gz_lo, gz_hi, gp_grad,
                                   MAP_n, ci_n, MAP_r, ci_r,
                                   search_lo, search_hi)

    fig = plt.figure(figsize=(16, 18), facecolor=C["bg"])
    gs  = gridspec.GridSpec(4, 1, height_ratios=[2.6, 1.4, 1.8, 1.3],
                             hspace=0.07, left=0.07, right=0.97,
                             top=0.95, bottom=0.06)
    axs = [fig.add_subplot(gs[i]) for i in range(4)]
    for i in range(1, 4):
        axs[i].sharex(axs[0])
    for ax in axs:
        ax.set_facecolor(C["bg"])
        for sp in ax.spines.values(): sp.set_color("#cccccc")

    # ── Panel A: elevation profile ─────────────────────────────────────────
    ax = axs[0]; shade(ax); ref(ax)
    ax.fill_between(x_bed, h_bed.min()-80, np.minimum(h_bed, 0),
                    color=C["ocean"], alpha=0.18, zorder=0)
    ax.fill_between(x_bed, h_bed.min()-80, h_bed, color=C["grd"], alpha=0.40, zorder=1)
    hbc = np.interp(x_sf, x_bed, h_bed, left=np.nan, right=np.nan)
    ax.fill_between(x_sf, hbc, h_sf, where=~np.isnan(hbc),
                    color=C["ice"], alpha=0.80, zorder=2)
    for mask, col, lbl in [
        (delta_free_bed >  30,                           C["grd"],    "Grounded"),
        ((delta_free_bed >= 0) & (delta_free_bed <= 30), "#f0a500",   "Near flotation"),
        (delta_free_bed <   0,                           C["ocean"],  "Floating"),
    ]:
        ax.scatter(x_bed[mask], h_bed[mask], s=2.5, color=col,
                   alpha=0.65, zorder=4, label=lbl)
    ax.plot(x_bp, hbed_bw_out, color=C["bed_s"], lw=1.8, zorder=5, label="Bed (smoothed)")
    ax.plot(x_sf, h_sf,        color=C["surf"],  lw=1.8, zorder=5, label="Ice surface")
    ax.axhline(0, color=C["ocean"], lw=0.9, ls=":", alpha=0.8)
    ax.text(x_bp[1], 15, "Sea level", fontsize=7, color=C["ocean"])

    for MAP, col, mk, run_lbl in [
        (MAP_n, C["norm"], "o", f"Normalised MAP: {MAP_n:.1f} km"),
        (MAP_r, C["raw"],  "s", f"Raw MAP: {MAP_r:.1f} km"),
    ]:
        gp_s = float(np.interp(MAP, x_sf, h_sf))
        gp_b = float(np.interp(MAP, x_bp, d["hbed_bw"]))
        ax.scatter([MAP], [gp_s], s=200, color=col, zorder=11,
                   edgecolors="white", lw=1.8, marker=mk, label=run_lbl)
        ax.scatter([MAP], [gp_b], s=150, color=col, zorder=11,
                   edgecolors="white", lw=1.5, marker="v")

    for MAP, col, xoff, yoff in [
        (MAP_n, C["norm"], -30,  260),
        (MAP_r, C["raw"],  +12,  260),
    ]:
        gp_s = float(np.interp(MAP, x_sf, h_sf))
        ci   = ci_n if col == C["norm"] else ci_r
        ax.annotate(f'{"Norm" if col == C["norm"] else "Raw"}\n{MAP:.1f} km',
                    xy=(MAP, gp_s), xytext=(MAP + xoff, gp_s + yoff),
                    fontsize=8, color=col, fontweight="semibold",
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.1),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=col, alpha=0.93, lw=1.2), zorder=12)

    if gp_grad:
        gpgs = float(np.interp(gp_grad, x_sf, h_sf))
        ax.scatter([gp_grad], [gpgs], s=130, color=C["grad"], zorder=10,
                   edgecolors="white", lw=1.6, marker="D",
                   label=f"Gradient GP: {gp_grad:.1f} km")

    if gz_lo and gz_hi:
        gz_s = float(np.interp((gz_lo + gz_hi) / 2, x_sf, h_sf))
        ax.annotate("InSAR GZ\n(reference)", xy=((gz_lo + gz_hi) / 2, gz_s),
                    xytext=((gz_lo + gz_hi) / 2 + 20, gz_s + 240),
                    fontsize=8, color=C["gz"], fontweight="semibold",
                    arrowprops=dict(arrowstyle="->", color=C["gz"], lw=1.1),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=C["gz"], alpha=0.9, lw=1.0), zorder=11)

    disc = abs(MAP_n - MAP_r)
    disc_txt = (f"⚠ Discordant  |Δ| = {disc:.1f} km"
                if disc > DISCORD_THRESHOLD_KM
                else f"Agreement  |Δ| = {disc:.1f} km")
    ax.text(0.98, 0.05, disc_txt, transform=ax.transAxes, ha="right",
            fontsize=9, color=C["raw"] if disc > DISCORD_THRESHOLD_KM else C["norm"],
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=C["raw"] if disc > DISCORD_THRESHOLD_KM else C["norm"],
                      alpha=0.88, lw=1.0))

    ax.set_ylabel("Elevation (m WGS84)", fontsize=10, color=C["text"])
    ax.set_ylim(h_bed.min() - 100, h_sf.max() * 1.07)
    ax.set_xlim(x_bp[0] - 1, x_bp[-1] + 2)
    ax.tick_params(labelbottom=False, colors=C["text"], labelsize=8.5)
    ax.legend(fontsize=7.5, loc="upper left", ncol=3,
              framealpha=0.9, edgecolor="#ccc", facecolor="white")
    ax.grid(True, alpha=0.13, lw=0.6)
    ax.set_title(
        f"{label}  ·  Offline Bayesian Changepoint Detection\n"
        "hildensia/bayesian_changepoint_detection (Fearnhead 2006)  ·  "
        "4 features  ·  Equal weights  ·  Green = normalised  ·  Orange = raw",
        fontsize=10.5, color=C["text"], pad=8, fontweight="semibold")

    # ── Panel B: feature profiles ──────────────────────────────────────────
    ax = axs[1]; shade(ax); ref(ax)
    ax.plot(x_bp, d["amp_f"],        color="#c45ab3", lw=1.4, label="Amplitude level (dB)", zorder=3)
    ax.plot(x_bp, d["dA"],           color=C["gauss"],lw=1.4, label="dA/dx (dB/km)", zorder=3)
    ax2 = ax.twinx(); ax2.set_facecolor(C["bg"])
    ax2.plot(x_bp, d["delta_free_bp"], color=C["dfree"], lw=1.4, alpha=0.85,
             label="Δfree (m)", zorder=3)
    ax2.plot(x_bp, d["d_surf_bp"],     color="#888",      lw=1.2, alpha=0.65,
             label="Surface slope (m/km)", zorder=2)
    ax2.set_ylabel("Δfree (m) / slope (m/km)", fontsize=8.5, color=C["dfree"])
    ax2.tick_params(axis="y", colors=C["dfree"], labelsize=8)
    ax.set_ylabel("Amplitude (dB)", fontsize=8.5, color="#c45ab3")
    ax.tick_params(axis="x", labelbottom=False); ax.tick_params(axis="y", labelsize=8.5)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7.5, loc="upper left", ncol=4,
              framealpha=0.9, edgecolor="#ccc", facecolor="white")
    ax.grid(True, alpha=0.13, lw=0.6)

    # ── Panel C: posteriors comparison ─────────────────────────────────────
    ax = axs[2]; shade(ax); ref(ax)
    ax.axvspan(search_lo, search_hi if search_hi else gz_lo,
               color="#e8f4e8", alpha=0.40, zorder=0)
    cp_n = r_n["posteriors"]["cp_comb"]
    cp_r = r_r["posteriors"]["cp_comb"]
    sc   = max(cp_n.max(), cp_r.max(),
               r_n["posteriors"]["cp_gauss"].max(),
               r_r["posteriors"]["cp_gauss"].max())

    # Individual model posteriors (light, dashed)
    for run, col_run, ls in [("normalised", C["norm"], "-"),
                              ("raw",        C["raw"],  "--")]:
        ps = results[run]["posteriors"]
        for cp, col_m, lw in [(ps["cp_gauss"], C["gauss"], 1.3),
                               (ps["cp_ifm"],   C["ifm"],   1.3),
                               (ps["cp_fcov"],  C["fcov"],  1.1)]:
            ax.plot(x_cp, cp / sc, color=col_m, lw=lw, ls=ls, alpha=0.50, zorder=3)

    # Combined posteriors (heavy)
    ax.fill_between(x_cp, 0, cp_n / sc, color=C["norm"], alpha=0.18, zorder=1)
    ax.fill_between(x_cp, 0, cp_r / sc, color=C["raw"],  alpha=0.12, zorder=1)
    ax.plot(x_cp, cp_n / sc, color=C["norm"], lw=2.8, zorder=6,
            label=f"Normalised  MAP={MAP_n:.1f}km  68%CI=[{ci_n['lo68']:.0f}–{ci_n['hi68']:.0f}]")
    ax.plot(x_cp, cp_r / sc, color=C["raw"],  lw=2.8, ls="--", zorder=6,
            label=f"Raw         MAP={MAP_r:.1f}km  68%CI=[{ci_r['lo68']:.0f}–{ci_r['hi68']:.0f}]")

    ax.axvline(MAP_n, color=C["norm"], lw=2.0, zorder=7)
    ax.axvline(MAP_r, color=C["raw"],  lw=2.0, ls="--", zorder=7)

    # CI brackets
    for ci, col, ypos in [(ci_n, C["norm"], 0.09), (ci_r, C["raw"], 0.02)]:
        ax.annotate("", xy=(ci["lo68"], ypos), xytext=(ci["hi68"], ypos),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.4))
        ax.text((ci["lo68"] + ci["hi68"]) / 2, ypos + 0.04,
                f"68% [{ci['lo68']:.0f}–{ci['hi68']:.0f}]",
                ha="center", fontsize=6.5, color=col)

    # Legend for model line styles
    from matplotlib.lines import Line2D
    legend_extra = [
        Line2D([0], [0], color="#888", lw=1.3, ls="-",
               label="Solid = normalised  (Gaussian / IFM / FullCov)"),
        Line2D([0], [0], color="#888", lw=1.3, ls="--",
               label="Dashed = raw"),
    ]
    handles, labs = ax.get_legend_handles_labels()
    ax.legend(handles + legend_extra, labs + [h.get_label() for h in legend_extra],
              fontsize=7.5, loc="upper left", ncol=2,
              framealpha=0.9, edgecolor="#ccc", facecolor="white")
    ax.set_ylabel("P(GP=x)\n(normalised)", fontsize=9.5, color=C["text"])
    ax.tick_params(labelbottom=False, colors=C["text"], labelsize=8.5)
    ax.grid(True, alpha=0.13, lw=0.6); ax.set_ylim(bottom=0)

    # ── Panel D: amplitude ─────────────────────────────────────────────────
    ax = axs[3]; shade(ax); ref(ax)
    ax.scatter(x_bp, amp, s=5, color=C["amp_r"], alpha=0.55, zorder=2)
    ax.plot(x_bp, amp_f_out, color=C["amp_f"], lw=1.8, zorder=3, label="Filtered amplitude")
    for MAP, col, mk, lbl in [
        (MAP_n, C["norm"], "o", f"Normalised MAP: {MAP_n:.1f} km"),
        (MAP_r, C["raw"],  "s", f"Raw MAP: {MAP_r:.1f} km"),
    ]:
        ax.scatter([MAP], [float(np.interp(MAP, x_bp, d["amp_f"]))],
                   s=160, color=col, zorder=7, edgecolors="white",
                   lw=1.5, marker=mk, label=lbl)
    if gp_grad:
        ax.scatter([gp_grad], [float(np.interp(gp_grad, x_bp, d["amp_f"]))],
                   s=120, color=C["grad"], marker="D", zorder=7,
                   edgecolors="white", lw=1.3,
                   label=f"Gradient GP: {gp_grad:.1f} km")
    if gz_lo and gz_hi:
        ax.scatter([gz_lo, gz_hi],
                   [float(np.interp(gz_lo, x_bp, d["amp_f"])),
                    float(np.interp(gz_hi, x_bp, d["amp_f"]))],
                   s=80, color=C["gz"], marker="D", zorder=7,
                   edgecolors="white", lw=1.2, label="InSAR GZ bounds")

    bm = float(np.nanmean(d["amp_f"][x_bp < x_bp[int(len(x_bp)*0.3)]]))
    dm = float(np.nanmean(d["amp_f"][x_bp > x_bp[int(len(x_bp)*0.7)]]))
    ax.axhline(bm, color=C["amp_f"], lw=0.8, ls=":", alpha=0.6)
    ax.axhline(dm, color="#666",     lw=0.8, ls=":", alpha=0.6)
    ax.text(x_bp[2], bm + 0.4, f"Bright: {bm:.0f} dB", fontsize=7, color=C["amp_f"])
    ax.text(x_bp[2], dm + 0.4, f"Dark: {dm:.0f} dB",   fontsize=7, color="#666")

    ax.set_ylabel("Bed power (dB)", fontsize=10, color=C["text"])
    ax.set_xlabel("Along-track distance (km)", fontsize=10, color=C["text"])
    ax.set_ylim(np.nanmin(amp) - 3, np.nanmax(amp) + 3)
    ax.tick_params(colors=C["text"], labelsize=8.5)
    ax.legend(fontsize=7.5, loc="lower left", ncol=3,
              framealpha=0.9, edgecolor="#ccc", facecolor="white")
    ax.grid(True, alpha=0.13, lw=0.6)

    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=C["bg"])
    print(f"  Saved: {out_path}")


# =============================================================================
# SUMMARY TABLE
# =============================================================================
def print_summary(label, results, gp_grad, gz_lo, gz_hi):
    r_n = results["normalised"]; r_r = results["raw"]
    ci_n = r_n["ci"]; ci_r = r_r["ci"]
    x_cp = r_n["x_cp"]
    disc  = abs(ci_n["MAP"] - ci_r["MAP"])
    flag  = "⚠ DISCORDANT" if disc > DISCORD_THRESHOLD_KM else "✓ Agreement"

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  {'':25s}  {'Normalised':>12}  {'Raw':>12}")
    print(f"  {'-'*52}")
    for run_lbl, ci, ps in [
        ("normalised", ci_n, r_n["posteriors"]),
        ("raw",        ci_r, r_r["posteriors"]),
    ]:
        pass  # printed in table below

    rows = [
        ("MAP (km)",        f"{ci_n['MAP']:.2f}",  f"{ci_r['MAP']:.2f}"),
        ("68% CI lo (km)",  f"{ci_n['lo68']:.1f}", f"{ci_r['lo68']:.1f}"),
        ("68% CI hi (km)",  f"{ci_n['hi68']:.1f}", f"{ci_r['hi68']:.1f}"),
        ("95% CI lo (km)",  f"{ci_n['lo95']:.1f}", f"{ci_r['lo95']:.1f}"),
        ("95% CI hi (km)",  f"{ci_n['hi95']:.1f}", f"{ci_r['hi95']:.1f}"),
        ("68% CI width",    f"{ci_n['hi68']-ci_n['lo68']:.1f} km",
                            f"{ci_r['hi68']-ci_r['lo68']:.1f} km"),
        ("Gaussian MAP",
            f"{x_cp[np.argmax(r_n['posteriors']['cp_gauss'])]:.2f}",
            f"{x_cp[np.argmax(r_r['posteriors']['cp_gauss'])]:.2f}"),
        ("IFM MAP",
            f"{x_cp[np.argmax(r_n['posteriors']['cp_ifm'])]:.2f}",
            f"{x_cp[np.argmax(r_r['posteriors']['cp_ifm'])]:.2f}"),
        ("FullCov MAP",
            f"{x_cp[np.argmax(r_n['posteriors']['cp_fcov'])]:.2f}",
            f"{x_cp[np.argmax(r_r['posteriors']['cp_fcov'])]:.2f}"),
    ]
    for name, val_n, val_r in rows:
        print(f"  {name:25s}  {val_n:>12}  {val_r:>12}")

    print(f"  {'-'*52}")
    if gp_grad:
        print(f"  {'Gradient GP (km)':25s}  {gp_grad:>12.2f}")
    if gz_lo:
        print(f"  {'InSAR GZ seaward (km)':25s}  {gz_lo:>12.2f}")
        print(f"  {'Offset norm from GZ':25s}  {gz_lo-ci_n['MAP']:>+11.1f} km")
        print(f"  {'Offset raw from GZ':25s}  {gz_lo-ci_r['MAP']:>+11.1f} km")
    print(f"  {'|MAP_norm − MAP_raw|':25s}  {disc:>12.2f} km  {flag}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BOCPD grounding point: normalised vs raw, 4 features")
    parser.add_argument("--power",      required=True)
    parser.add_argument("--bed",        required=True)
    parser.add_argument("--surface",    required=True)
    parser.add_argument("--gz_lo",      type=float, default=None,
                        help="InSAR GZ seaward edge (km) — search window upper bound")
    parser.add_argument("--gz_hi",      type=float, default=None,
                        help="InSAR GZ landward edge (km) — display only")
    parser.add_argument("--search_lo",  type=float, default=75.0,
                        help="GP search window lower bound (km)")
    parser.add_argument("--search_hi",  type=float, default=None,
                        help="GP search window upper bound (km); defaults to gz_lo")
    parser.add_argument("--ref_lo",     type=float, default=None,
                        help="Normalisation reference section lower bound (km)")
    parser.add_argument("--ref_hi",     type=float, default=None,
                        help="Normalisation reference section upper bound (km)")
    parser.add_argument("--label",      default="Glacier")
    parser.add_argument("--out",        default="bocpd_result.png")
    args = parser.parse_args()

    gz_lo      = args.gz_lo
    gz_hi      = args.gz_hi
    search_hi  = args.search_hi if args.search_hi else gz_lo
    if search_hi is None:
        raise ValueError("Provide either --gz_lo or --search_hi to bound the search window.")

    print(f"\n{args.label}")
    print(f"  Search window: {args.search_lo:.1f} – {search_hi:.1f} km")
    if gz_lo:
        print(f"  InSAR GZ:      {gz_lo:.2f} – {gz_hi:.2f} km")

    d = load_data(args.power, args.bed, args.surface)
    d = compute_features(d)

    results = run_both(d, gz_lo_km=gz_lo if gz_lo else search_hi,
                       search_lo_km=args.search_lo,
                       search_hi_km=search_hi,
                       ref_lo_km=args.ref_lo,
                       ref_hi_km=args.ref_hi)

    gp_grad = gradient_gp(d, args.search_lo, search_hi)
    print(f"  Gradient GP:   {gp_grad:.2f} km")

    plot_combined(d, results,
                  label=args.label,
                  gz_lo=gz_lo, gz_hi=gz_hi,
                  gp_grad=gp_grad,
                  search_lo=args.search_lo,
                  search_hi=search_hi,
                  out_path=args.out)

    print_summary(args.label, results, gp_grad, gz_lo, gz_hi)
