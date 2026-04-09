"""
Grounding Point Detection via Offline Bayesian Changepoint Detection
=====================================================================
Uses the hildensia/bayesian_changepoint_detection package:
    pip install bayesian-changepoint-detection decorator

Implements Fearnhead (2006) offline BOCPD with three likelihood models:
  - 1D Gaussian         : amplitude level alone
  - IFM (independent)   : 4 normalised features, independent channels
  - FullCov             : 4 normalised features, full covariance model

The package implements Fearnhead's (2006) exact offline algorithm:

    Q(t)   = log P(x_{t:n} | last CP at or before t)
    P(t,s) = log P(x_{t:s} | one segment)   [observation likelihood]
    Pcp    = (n-1 × n-1) matrix where Pcp[j, t] =
             log P(data has j+1 changepoints, last one at index t)

Marginalising Pcp over j gives the posterior P(CP at index t | data).

IMPORTANT: Pcp has shape (n-1, n-1) not (n, n), so cp probabilities
align to x[:-1], i.e., the CP at index t is the transition BETWEEN
samples t-1 and t.

Features (all normalised to zero mean / unit std in the floating section)
---------
  f_amp   : Butterworth-filtered amplitude level  (primary)
  f_dA    : amplitude gradient dA/dx              (NaN-masked around artifact clusters)
  f_dfree : flotation residual  Δfree = h_surf − H·(1 − ρ_ice/ρ_sw)
  f_dsurf : ice surface slope   d(h_surf)/dx

Physics constraints
-------------------
  - The grounding point must be DOWNFLOW (seaward, lower x) of the InSAR GZ.
  - Search window is restricted to [search_lo_km, gz_lo_km].
  - For Petermann (x=0 = calving front, x increases inland) the GP is
    where amplitude transitions from bright (floating) to dark (grounded).

Installation
------------
  pip install bayesian-changepoint-detection decorator

Usage
-----
  python bocpd_hildensia_grounding_point.py \\
      --power   petermann_bed_power.csv \\
      --bed     petermann_bottom.csv \\
      --surface petermann_surface.csv \\
      --gz_lo   95.38 \\
      --gz_hi   97.89 \\
      --search_lo 75 \\
      --label   "Petermann Glacier" \\
      --out     result.png

References
----------
  Fearnhead (2006). "Exact and Efficient Bayesian Inference for Multiple
    Changepoint Problems." Statistics and Computing 16(2), 203-213.
  Adams & MacKay (2007). "Bayesian Online Changepoint Detection."
    arXiv:0710.3742.
  Ciracì et al. (2023). PNAS 120(20). doi:10.1073/pnas.2220924120
    (InSAR grounding zone reference for Petermann)
"""

import argparse
import warnings
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
        "Install the package first:\n"
        "  pip install bayesian-changepoint-detection decorator"
    )

# =============================================================================
# SETTINGS
# =============================================================================
RHO_ICE = 917.0
RHO_SW  = 1028.0

AMP_ORDER, AMP_WN   = 5, 0.15   # amplitude Butterworth filter
ELEV_ORDER, ELEV_WN = 5, 0.09   # bed elevation filter
SLOPE_WN            = 0.02      # surface/bed slope smoothing

# Search window: fraction of gz_lo to use as floating reference
FLOAT_FRAC      = 0.73    # x < FLOAT_FRAC * gz_lo → floating reference
DEFAULT_SEARCH_LO = 75.0  # km — default lower bound of GP search window

# Gradient artifact masking
GRADIENT_MAX_ABS     = 5.0   # dB/km
GRADIENT_ARTIFACT_LO = 97.5  # km  — NaN cluster start
GRADIENT_ARTIFACT_HI = 103.0 # km  — NaN cluster end

# BOCPD truncation threshold (log-prob below this is ignored for speed)
TRUNCATE = -50.0

# Combination weights (higher → more influence)
W_GAUSS = 2.5
W_IFM   = 3.0
W_FCOV  = 1.5

# Posterior smoothing
SMOOTH_SIZE = 3


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(power_path, bed_path, surface_path):
    bp  = pd.read_csv(power_path)
    bt  = pd.read_csv(bed_path).dropna(subset=["wgs84"])
    sf  = pd.read_csv(surface_path).dropna(subset=["wgs84"])
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
    """
    Compute four normalised features on the bed-power grid.

    f_amp   : filtered amplitude level
    f_dA    : amplitude gradient (NaN artifact region masked)
    f_dfree : flotation residual Δfree
    f_dsurf : ice surface slope

    Each is normalised to zero mean, unit std in the floating section.
    All arrays have the same length as x_bp.
    """
    x_bp, amp = d["x_bp"], d["amp"]
    x_bed, h_bed = d["x_bed"], d["h_bed"]
    x_sf, h_sf = d["x_sf"], d["h_sf"]

    nan_mask = np.isnan(amp)

    # Filtered amplitude
    amp_i    = pd.Series(amp).interpolate().ffill().bfill().values
    b, a     = butter(AMP_ORDER, AMP_WN, btype="low", analog=False)
    amp_f    = filtfilt(b, a, amp_i)
    amp_f_out = amp_f.copy(); amp_f_out[nan_mask] = np.nan

    # Amplitude gradient with NaN artifact masking
    dA = np.gradient(amp_f, x_bp)
    dA[(x_bp > GRADIENT_ARTIFACT_LO) & (x_bp < GRADIENT_ARTIFACT_HI)] = np.nan
    dA[np.abs(dA) > GRADIENT_MAX_ABS] = np.nan
    dA = pd.Series(dA).interpolate(limit=5).ffill().bfill().values

    # Flotation residual Δfree = h_surf − H·(1 − ρ_ice/ρ_sw)
    # > 0  → grounded,  ≈ 0  → floating,  < 0  → below flotation
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
        nan_mask       = nan_mask,
        amp_f          = amp_f,
        amp_f_out      = amp_f_out,
        dA             = dA,
        delta_free_bp  = delta_free_bp,
        delta_free_bed = np.interp(x_bed, x_bp, delta_free_bp),
        d_surf_bp      = d_surf_bp,
        H_bp           = np.interp(x_bp, x_bed, H_bed),
        h_sf_bp        = np.interp(x_bp, x_sf, h_sf),
        hbed_bw        = hbed_bw,
        hbed_bw_out    = hbed_bw_out,
        H_bed          = H_bed,
    ))
    return d


def normalise_features(d, gz_lo_km):
    """Normalise each feature to zero mean, unit std in the floating section."""
    x_bp       = d["x_bp"]
    float_mask = x_bp < FLOAT_FRAC * gz_lo_km

    def norm(f):
        return (f - np.mean(f[float_mask])) / (np.std(f[float_mask]) + 1e-9)

    d["f_amp"]   = norm(d["amp_f"])
    d["f_dA"]    = norm(d["dA"])
    d["f_dfree"] = norm(d["delta_free_bp"])
    d["f_dsurf"] = norm(d["d_surf_bp"])
    return d


# =============================================================================
# BOCPD via hildensia/bayesian_changepoint_detection
# =============================================================================
def run_bocpd(d, gz_lo_km, search_lo_km=DEFAULT_SEARCH_LO):
    """
    Run Fearnhead (2006) offline BOCPD with three likelihood models.

    The package's offline_changepoint_detection(data, prior, likelihood)
    returns (Q, P, Pcp) where:
        Pcp[j, t] = log P(data has j+1 CPs, last one at index t)
    Pcp has shape (n-1, n-1), so:
        cp_prob = exp(Pcp).sum(axis=0)  has length n-1
        aligns to x[:-1] (CP between samples t-1 and t)

    Three likelihood models are run:
        1D Gaussian  : offcd.gaussian_obs_log_likelihood  — amplitude only
        IFM          : offcd.ifm_obs_log_likelihood       — 4 features, indep.
        FullCov      : offcd.fullcov_obs_log_likelihood   — 4 features, full cov

    Their log-posteriors are combined with weights W_GAUSS, W_IFM, W_FCOV.
    """
    x_bp = d["x_bp"]
    win  = (x_bp >= search_lo_km) & (x_bp < gz_lo_km)
    x_win = x_bp[win]
    n     = int(win.sum())

    signal = np.column_stack([
        d["f_amp"][win],
        d["f_dA"][win],
        d["f_dfree"][win],
        d["f_dsurf"][win],
    ])

    # Uniform prior: const_prior(r, l) = 1/l  — equal probability per position
    prior = partial(offcd.const_prior, l=n + 1)

    print("Running 1D Gaussian model...")
    _, _, Pcp1 = offcd.offline_changepoint_detection(
        d["f_amp"][win], prior, offcd.gaussian_obs_log_likelihood,
        truncate=TRUNCATE)
    cp_gauss = np.exp(Pcp1).sum(0); cp_gauss /= cp_gauss.sum()

    print("Running IFM (4-feature independent) model...")
    _, _, Pcp3 = offcd.offline_changepoint_detection(
        signal, prior, offcd.ifm_obs_log_likelihood, truncate=TRUNCATE)
    cp_ifm = np.exp(Pcp3).sum(0); cp_ifm /= cp_ifm.sum()

    print("Running FullCov (4-feature covariance) model...")
    _, _, Pcp2 = offcd.offline_changepoint_detection(
        signal, prior, offcd.fullcov_obs_log_likelihood, truncate=TRUNCATE)
    cp_fcov = np.exp(Pcp2).sum(0); cp_fcov /= cp_fcov.sum()

    # Pcp has shape (n-1, n-1) → align to x_win[:-1]
    x_cp = x_win[:-1]

    # Combine in log-space with reliability weights
    log_comb = (W_GAUSS * np.log(cp_gauss + 1e-15)
                + W_IFM  * np.log(cp_ifm   + 1e-15)
                + W_FCOV * np.log(cp_fcov  + 1e-15)) / (W_GAUSS + W_IFM + W_FCOV)
    log_comb -= np.logaddexp.reduce(log_comb)
    cp_comb  = np.exp(log_comb)
    cp_comb  = uniform_filter1d(cp_comb, size=SMOOTH_SIZE)
    cp_comb /= cp_comb.sum()

    # MAP and credible intervals
    n1  = len(x_cp)
    cdf = np.cumsum(cp_comb)
    GP_km  = float(x_cp[np.argmax(cp_comb)])
    lo68   = float(x_cp[np.searchsorted(cdf, 0.160)])
    hi68   = float(x_cp[min(np.searchsorted(cdf, 0.840), n1 - 1)])
    lo95   = float(x_cp[np.searchsorted(cdf, 0.025)])
    hi95   = float(x_cp[min(np.searchsorted(cdf, 0.975), n1 - 1)])

    return dict(
        x_cp     = x_cp,
        x_win    = x_win,
        cp_gauss = cp_gauss,
        cp_ifm   = cp_ifm,
        cp_fcov  = cp_fcov,
        cp_comb  = cp_comb,
        gp_km    = GP_km,
        lo68=lo68, hi68=hi68,
        lo95=lo95, hi95=hi95,
        search_lo_km = search_lo_km,
        gz_lo_km     = gz_lo_km,
    )


# =============================================================================
# REFERENCE: amplitude gradient method
# =============================================================================
def gradient_gp(d, gz_lo_km):
    """Steepest negative amplitude gradient downflow of GZ (for comparison)."""
    x_bp  = d["x_bp"]
    dA    = d["dA"].copy()
    dA[np.abs(dA) > GRADIENT_MAX_ABS] = 0.0
    down  = x_bp < gz_lo_km
    return float(x_bp[down][np.argmin(dA[down])])


# =============================================================================
# PLOTTING
# =============================================================================
def plot(d, result, label="", gz_lo_km=None, gz_hi_km=None,
         gradient_gp_km=None, out_path="bocpd_result.png"):

    x_bp  = d["x_bp"];  x_bed = d["x_bed"];  x_sf = d["x_sf"]
    h_bed = d["h_bed"]; h_sf  = d["h_sf"];   amp  = d["amp"]
    nan_mask        = d["nan_mask"]
    amp_f_out       = d["amp_f_out"]
    delta_free_bed  = d["delta_free_bed"]
    hbed_bw_out     = d["hbed_bw_out"]
    H_bp            = d["H_bp"]

    gp    = result["gp_km"]
    x_cp  = result["x_cp"]
    lo68, hi68 = result["lo68"], result["hi68"]
    lo95, hi95 = result["lo95"], result["hi95"]
    cp_gauss = result["cp_gauss"]
    cp_ifm   = result["cp_ifm"]
    cp_fcov  = result["cp_fcov"]
    cp_comb  = result["cp_comb"]
    search_lo = result["search_lo_km"]

    BG      = "#f6f4f1"
    C_SURF  = "#4a6fa5"; C_BED_S = "#7a5230"; C_ICE   = "#ccdff5"
    C_OCEAN = "#3a7abf"; C_GRD   = "#a68a5b"; C_GZ    = "#7b2d8b"
    C_GRAD  = "#d94f3d"; C_BOCPD = "#1a8f5a"
    C_GAUSS = "#e87722"; C_IFM   = "#9b59b6"; C_FCOV  = "#16a085"
    C_AMP_F = "#378add"; C_AMP_R = "#b4b2a9"; TEXT    = "#1a1a1a"

    # NaN blocks
    diff_nm = np.diff(nan_mask.astype(int))
    starts  = list(np.where(diff_nm == 1)[0] + 1)
    ends    = list(np.where(diff_nm == -1)[0] + 1)
    if nan_mask[0]:  starts = [0] + starts
    if nan_mask[-1]: ends   = ends + [len(x_bp)]
    nan_blocks = [(x_bp[s], x_bp[min(e, len(x_bp)-1)])
                  for s, e in zip(starts, ends)
                  if x_bp[min(e, len(x_bp)-1)] - x_bp[s] > 0.3]

    def shade_nans(ax):
        for lo, hi in nan_blocks:
            ax.axvspan(lo, hi, alpha=0.09, color="gray", zorder=0)

    def draw_ref(ax):
        if gz_lo_km and gz_hi_km:
            ax.axvspan(gz_lo_km, gz_hi_km, color=C_GZ, alpha=0.20, zorder=2)
            ax.axvline(gz_lo_km, color=C_GZ, lw=1.4, ls="--", alpha=0.88, zorder=3)
            ax.axvline(gz_hi_km, color=C_GZ, lw=1.4, ls="--", alpha=0.88, zorder=3)
        if gradient_gp_km:
            ax.axvline(gradient_gp_km, color=C_GRAD, lw=1.6, ls="--", alpha=0.70, zorder=3)
        ax.axvspan(lo95, hi95, color=C_BOCPD, alpha=0.09, zorder=1)
        ax.axvspan(lo68, hi68, color=C_BOCPD, alpha=0.17, zorder=1)
        ax.axvline(gp,   color=C_BOCPD, lw=2.2, ls="-",  alpha=0.88, zorder=4)

    fig = plt.figure(figsize=(16, 16), facecolor=BG)
    gs  = gridspec.GridSpec(4, 1, height_ratios=[2.8, 1.3, 1.7, 1.3],
                             hspace=0.07, left=0.07, right=0.97,
                             top=0.95, bottom=0.06)
    ax_prof = fig.add_subplot(gs[0])
    ax_feat = fig.add_subplot(gs[1], sharex=ax_prof)
    ax_post = fig.add_subplot(gs[2], sharex=ax_prof)
    ax_amp  = fig.add_subplot(gs[3], sharex=ax_prof)
    for ax in [ax_prof, ax_feat, ax_post, ax_amp]:
        ax.set_facecolor(BG)
        for sp in ax.spines.values(): sp.set_color("#cccccc")

    # ── Panel A: elevation profile ─────────────────────────────────────────
    ax = ax_prof; shade_nans(ax); draw_ref(ax)
    ax.fill_between(x_bed, h_bed.min()-80, np.minimum(h_bed, 0),
                    color=C_OCEAN, alpha=0.18, zorder=0)
    ax.fill_between(x_bed, h_bed.min()-80, h_bed, color=C_GRD, alpha=0.40, zorder=1)
    hbc = np.interp(x_sf, x_bed, h_bed, left=np.nan, right=np.nan)
    ax.fill_between(x_sf, hbc, h_sf, where=~np.isnan(hbc),
                    color=C_ICE, alpha=0.80, zorder=2)
    for mask, col, lbl in [
        (delta_free_bed >  30,                          C_GRD,    "Grounded"),
        ((delta_free_bed >= 0) & (delta_free_bed <= 30),"#f0a500","Near flotation"),
        (delta_free_bed <   0,                          C_OCEAN,  "Floating"),
    ]:
        ax.scatter(x_bed[mask], h_bed[mask], s=2.5, color=col,
                   alpha=0.65, zorder=4, label=lbl)
    ax.plot(x_bp, hbed_bw_out, color=C_BED_S, lw=1.8, zorder=5, label="Bed (smoothed)")
    ax.plot(x_sf, h_sf, color=C_SURF, lw=1.8, zorder=5, label="Ice surface")
    ax.axhline(0, color=C_OCEAN, lw=0.9, ls=":", alpha=0.8)
    ax.text(1.5, 15, "Sea level", fontsize=7, color=C_OCEAN)

    gp_s = float(np.interp(gp, x_sf, h_sf))
    gp_b = float(np.interp(gp, x_bp, hbed_bw_out))
    gp_H = float(np.interp(gp, x_bp, H_bp))
    ax.scatter([gp], [gp_s], s=220, color=C_BOCPD, zorder=11,
               edgecolors="white", linewidths=1.8, label=f"BOCPD GP: {gp:.1f} km")
    ax.scatter([gp], [gp_b], s=160, color=C_BOCPD, zorder=11,
               edgecolors="white", linewidths=1.5, marker="v")
    ax.annotate(
        f"BOCPD MAP GP\n{gp:.1f} km\nH = {gp_H:.0f} m\n68% CI [{lo68:.0f}–{hi68:.0f}]",
        xy=(gp, gp_s), xytext=(gp - 36, gp_s + 260),
        fontsize=8.5, color=C_BOCPD, fontweight="semibold",
        arrowprops=dict(arrowstyle="->", color=C_BOCPD, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_BOCPD,
                  alpha=0.93, lw=1.2), zorder=12)

    if gradient_gp_km:
        gpgs = float(np.interp(gradient_gp_km, x_sf, h_sf))
        ax.scatter([gradient_gp_km], [gpgs], s=150, color=C_GRAD, zorder=10,
                   edgecolors="white", lw=1.6, marker="D",
                   label=f"Gradient GP: {gradient_gp_km:.1f} km")
        ax.annotate(f"Gradient GP\n{gradient_gp_km:.1f} km",
                    xy=(gradient_gp_km, gpgs),
                    xytext=(gradient_gp_km - 14, gpgs - 330),
                    fontsize=8, color=C_GRAD,
                    arrowprops=dict(arrowstyle="->", color=C_GRAD, lw=1.1),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=C_GRAD, alpha=0.9, lw=1.0), zorder=11)

    if gz_lo_km and gz_hi_km:
        gz_s = float(np.interp((gz_lo_km + gz_hi_km) / 2, x_sf, h_sf))
        ax.annotate("InSAR GZ\n(reference)",
                    xy=((gz_lo_km + gz_hi_km) / 2, gz_s),
                    xytext=((gz_lo_km + gz_hi_km) / 2 + 20, gz_s + 260),
                    fontsize=8, color=C_GZ, fontweight="semibold",
                    arrowprops=dict(arrowstyle="->", color=C_GZ, lw=1.1),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=C_GZ, alpha=0.9, lw=1.0), zorder=11)

    ax.text(38, 300, "Floating tongue", ha="center",
            fontsize=8.5, color=C_OCEAN, style="italic")
    ax.text(148, 850, "Grounded\nice sheet", ha="center",
            fontsize=8.5, color="#5a4020", style="italic")
    ax.annotate("Ice flow →", xy=(0.13, 0.06), xycoords="axes fraction",
                fontsize=8.5, color="#555", style="italic",
                xytext=(0.02, 0.06),
                arrowprops=dict(arrowstyle="<-", color="#555", lw=1.1))

    ax.set_ylabel("Elevation (m WGS84)", fontsize=10, color=TEXT)
    ax.set_ylim(h_bed.min() - 100, h_sf.max() * 1.07)
    ax.set_xlim(-2, x_bp[-1] + 3)
    ax.tick_params(labelbottom=False, colors=TEXT, labelsize=8.5)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2,
              framealpha=0.9, edgecolor="#ccc", facecolor="white")
    ax.grid(True, alpha=0.13, lw=0.6)
    ax.set_title(
        f"{label}  ·  Offline Bayesian Changepoint Detection\n"
        "hildensia/bayesian_changepoint_detection  "
        "(Fearnhead 2006 / Adams & MacKay 2007)\n"
        "Green: combined MAP ± CI  ·  Red dashed: amplitude gradient  ·  "
        "Purple: InSAR GZ (reference)",
        fontsize=10.5, color=TEXT, pad=8, fontweight="semibold")

    # ── Panel B: normalised features ───────────────────────────────────────
    ax = ax_feat; shade_nans(ax); draw_ref(ax)
    ax.plot(x_bp, d["f_amp"],   color="#c45ab3", lw=1.5, label="Amplitude level", zorder=3)
    ax.plot(x_bp, d["f_dA"],    color=C_GAUSS,   lw=1.5, label="Amplitude gradient (dA/dx)", zorder=3)
    ax.plot(x_bp, d["f_dfree"], color="#1a6bbf",  lw=1.5, label="Flotation residual (Δfree)", zorder=3)
    ax.plot(x_bp, d["f_dsurf"], color="#888",     lw=1.3, label="Surface slope", alpha=0.7, zorder=3)
    ax.axvspan(search_lo, gz_lo_km, color="#e8f4e8", alpha=0.45, zorder=0,
               label="BOCPD search window")
    ax.axhline(0, color="#999", lw=0.8, ls=":")
    ax.set_ylabel("Normalised\nfeature (σ)", fontsize=9.5, color=TEXT)
    ax.tick_params(labelbottom=False, colors=TEXT, labelsize=8.5)
    ax.legend(fontsize=7.5, loc="upper left", ncol=5,
              framealpha=0.9, edgecolor="#ccc", facecolor="white")
    ax.grid(True, alpha=0.13, lw=0.6); ax.set_ylim(-9, 6)

    # ── Panel C: posteriors ────────────────────────────────────────────────
    ax = ax_post; shade_nans(ax); draw_ref(ax)
    ax.axvspan(search_lo, gz_lo_km, color="#e8f4e8", alpha=0.40, zorder=0)
    sc = max(cp_comb.max(), cp_gauss.max(), cp_ifm.max(), cp_fcov.max())

    ax.fill_between(x_cp, 0, cp_comb / sc, color=C_BOCPD, alpha=0.22, zorder=1)
    ax.plot(x_cp, cp_comb / sc,  color=C_BOCPD, lw=2.8, zorder=6,
            label=f"Combined  MAP={gp:.1f}km")
    ax.plot(x_cp, cp_gauss / sc, color=C_GAUSS, lw=1.7, ls="--", alpha=0.85, zorder=4,
            label=f"1D Gaussian  MAP={x_cp[np.argmax(cp_gauss)]:.1f}km")
    ax.plot(x_cp, cp_ifm / sc,   color=C_IFM,   lw=1.7, ls="--", alpha=0.85, zorder=4,
            label=f"IFM (4 feat)  MAP={x_cp[np.argmax(cp_ifm)]:.1f}km")
    ax.plot(x_cp, cp_fcov / sc,  color=C_FCOV,  lw=1.5, ls=":",  alpha=0.80, zorder=3,
            label=f"FullCov (4 feat)  MAP={x_cp[np.argmax(cp_fcov)]:.1f}km")
    ax.axvline(gp, color=C_BOCPD, lw=2.2, zorder=7)
    ax.text(gp + 0.3, 0.62, f"MAP\n{gp:.1f}km",
            fontsize=8, color=C_BOCPD, fontweight="semibold")
    ax.annotate("", xy=(lo68, 0.08), xytext=(hi68, 0.08),
                arrowprops=dict(arrowstyle="<->", color=C_BOCPD, lw=1.6))
    ax.text((lo68 + hi68) / 2, 0.13, f"68% CI  [{lo68:.0f}–{hi68:.0f} km]",
            ha="center", fontsize=7.5, color=C_BOCPD)
    ax.set_ylabel("P(GP=x)\n(normalised,\nsearch window)", fontsize=9.5, color=TEXT)
    ax.tick_params(labelbottom=False, colors=TEXT, labelsize=8.5)
    ax.legend(fontsize=8, loc="upper left", ncol=2,
              framealpha=0.9, edgecolor="#ccc", facecolor="white")
    ax.grid(True, alpha=0.13, lw=0.6); ax.set_ylim(bottom=0)

    # ── Panel D: amplitude ─────────────────────────────────────────────────
    ax = ax_amp; shade_nans(ax); draw_ref(ax)
    ax.scatter(x_bp, amp, s=5, color=C_AMP_R, alpha=0.55, zorder=2)
    ax.plot(x_bp, amp_f_out, color=C_AMP_F, lw=1.8, zorder=3, label="Filtered amplitude")
    ax.scatter([gp], [float(np.interp(gp, x_bp, d["amp_f"]))],
               s=180, color=C_BOCPD, zorder=7, edgecolors="white", lw=1.6,
               label=f"BOCPD GP: {gp:.1f} km")
    if gradient_gp_km:
        ax.scatter([gradient_gp_km],
                   [float(np.interp(gradient_gp_km, x_bp, d["amp_f"]))],
                   s=130, color=C_GRAD, marker="D", zorder=7,
                   edgecolors="white", lw=1.4, label=f"Gradient GP: {gradient_gp_km:.1f} km")
    if gz_lo_km and gz_hi_km:
        ax.scatter([gz_lo_km, gz_hi_km],
                   [float(np.interp(gz_lo_km, x_bp, d["amp_f"])),
                    float(np.interp(gz_hi_km, x_bp, d["amp_f"]))],
                   s=90, color=C_GZ, marker="D", zorder=7,
                   edgecolors="white", lw=1.2, label="InSAR GZ bounds")
    bm = float(np.nanmean(d["amp_f"][x_bp < 50]))
    dm = float(np.nanmean(d["amp_f"][x_bp > 130]))
    ax.axhline(bm, color=C_AMP_F, lw=0.8, ls=":", alpha=0.6)
    ax.axhline(dm, color="#666",  lw=0.8, ls=":", alpha=0.6)
    ax.text(2, bm + 0.4, f"Floating: {bm:.0f} dB", fontsize=7, color=C_AMP_F)
    ax.text(2, dm + 0.4, f"Grounded: {dm:.0f} dB", fontsize=7, color="#666")
    ax.set_ylabel("Bed power (dB)", fontsize=10, color=TEXT)
    ax.set_xlabel("Along-track distance (km)", fontsize=10, color=TEXT)
    ax.set_ylim(np.nanmin(amp) - 3, np.nanmax(amp) + 3)
    ax.tick_params(colors=TEXT, labelsize=8.5)
    ax.legend(fontsize=7.5, loc="lower left", ncol=2,
              framealpha=0.9, edgecolor="#ccc", facecolor="white")
    ax.grid(True, alpha=0.13, lw=0.6)

    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out_path}")

    # Print summary
    x_cp = result["x_cp"]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  BOCPD MAP GP       : {gp:.2f} km")
    print(f"  68% CI             : [{lo68:.1f}, {hi68:.1f}] km")
    print(f"  95% CI             : [{lo95:.1f}, {hi95:.1f}] km")
    print(f"  1D Gaussian MAP    : {x_cp[np.argmax(cp_gauss)]:.2f} km  (P={cp_gauss.max():.4f})")
    print(f"  IFM MAP            : {x_cp[np.argmax(cp_ifm)]:.2f} km  (P={cp_ifm.max():.4f})")
    print(f"  FullCov MAP        : {x_cp[np.argmax(cp_fcov)]:.2f} km  (P={cp_fcov.max():.4f})")
    if gradient_gp_km:
        print(f"  Gradient method GP : {gradient_gp_km:.2f} km")
    if gz_lo_km:
        print(f"  InSAR GZ           : {gz_lo_km:.2f} – {gz_hi_km:.2f} km")
        print(f"  GP offset from GZ  : {gz_lo_km - gp:.1f} km downflow")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline BOCPD grounding point detection from IPR data")
    parser.add_argument("--power",     required=True)
    parser.add_argument("--bed",       required=True)
    parser.add_argument("--surface",   required=True)
    parser.add_argument("--gz_lo",     type=float, default=None,
                        help="InSAR GZ seaward edge (km) — sets search window upper bound")
    parser.add_argument("--gz_hi",     type=float, default=None,
                        help="InSAR GZ landward edge (km) — display only")
    parser.add_argument("--search_lo", type=float, default=DEFAULT_SEARCH_LO,
                        help=f"GP search window lower bound (km, default={DEFAULT_SEARCH_LO})")
    parser.add_argument("--label",     default="Glacier")
    parser.add_argument("--out",       default="bocpd_result.png")
    args = parser.parse_args()

    gz_lo = args.gz_lo if args.gz_lo else 999.0

    d = load_data(args.power, args.bed, args.surface)
    d = compute_features(d)
    d = normalise_features(d, gz_lo)

    result       = run_bocpd(d, gz_lo_km=gz_lo, search_lo_km=args.search_lo)
    grad_gp_km   = gradient_gp(d, gz_lo)

    plot(d, result,
         label=args.label,
         gz_lo_km=args.gz_lo, gz_hi_km=args.gz_hi,
         gradient_gp_km=grad_gp_km,
         out_path=args.out)
