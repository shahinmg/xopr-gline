"""
Bayesian Change-Point Detection of Glacier Grounding Points from IPR Data
=========================================================================
Implements the closed-form Bayesian 2-segment split-point posterior for
detecting grounding points in airborne ice-penetrating radar (IPR) data.

Why NOT classical online BOCPD (Adams & MacKay 2007)
-----------------------------------------------------
The classic online algorithm is designed for high-frequency streaming data
where signal is stationary within each segment. IPR transects violate both
assumptions:
  - Sparse sampling (~260 points over 179 km)
  - Within-segment variance is comparable to the between-segment shift
    (SNR < 3 for the amplitude gradient feature)
  - We have all data at once, so online detection is unnecessary

The Bayesian 2-segment split-point posterior is the correct offline
analogue. For each candidate split index t*, it computes:

    P(t* | data) ∝ P(data[0:t*] | θ₁) × P(data[t*:] | θ₂)

where the segment parameters θ₁, θ₂ (mean and variance) are analytically
marginalised out using the Normal-Inverse-Gamma (NIG) conjugate model.
This gives a proper posterior distribution over all possible GP locations,
rather than a point estimate.

Features used
-------------
  f_amp   : filtered amplitude level (primary — most robust to NaN artifacts)
  f_dA    : amplitude gradient dA/dx (direct GP signal but NaN-sensitive)
  f_dfree : flotation residual Δfree = h_surf − H·(1 − ρ_ice/ρ_sw)
  f_dsurf : ice surface slope d(h_surf)/dx

Each feature is normalised to zero mean, unit std in the floating section
(x < 70 km for Petermann). The NIG prior is set to match the floating
reference (mu0=0, tight variance), so deviations from the floating baseline
drive the posterior toward a change point.

Posteriors are combined in log-space with reliability weights:
    log P_combined ∝ 3·log P_amp + 2.5·log P_dA + 1.5·log P_dfree + 1·log P_dsurf

Search window
-------------
Physics constrains the GP to be downflow (seaward) of any InSAR-mapped
grounding zone. Restricting the search window to [75 km, GZ_seaward_edge]
prevents the posterior from being contaminated by features elsewhere on
the transect (e.g. NaN artifacts at 98–101 km in this dataset).

Inputs (CSV, CReSIS format)
---------------------------
  bed_power   : slow_time, along_track (m), bed_power_dB
  bed_elev    : slow_time, along_track (m), wgs84 (m)  — bed layer picks
  ice_surface : slow_time, along_track (m), wgs84 (m)  — surface picks

Usage
-----
  python bocpd_grounding_point.py \\
      --power   petermann_20100420_03_bed_power.csv \\
      --bed     petermann_20100420_03_bottom.csv \\
      --surface petermann_20100420_03_surface.csv \\
      [--gz_lo  95.38]   # InSAR GZ seaward edge (km) — optional prior
      [--gz_hi  97.89]   # InSAR GZ landward edge (km) — optional prior
      [--search_lo 75]   # GP search window start (km)
      [--label  "Petermann Glacier"]
      [--out    bocpd_result.png]

Reference
---------
  Adams & MacKay (2007), "Bayesian Online Changepoint Detection",
  arXiv:0710.3742 — the original BOCPD paper (online variant)

  This implementation uses the offline 2-segment version with the
  Normal-Inverse-Gamma conjugate, which is more appropriate for
  sparse, batch IPR data.

  InSAR GZ reference for Petermann:
  Ciracì et al. (2023), PNAS 120(20), doi:10.1073/pnas.2220924120
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt
from scipy.special import gammaln
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# SETTINGS
# =============================================================================
RHO_ICE = 917.0    # kg/m³
RHO_SW  = 1028.0   # kg/m³

# Butterworth filter settings
AMP_ORDER, AMP_WN   = 5, 0.15   # amplitude low-pass
ELEV_ORDER, ELEV_WN = 5, 0.09   # bed elevation low-pass
SLOPE_WN            = 0.02      # surface/bed slope smoothing

# NIG prior hyperparameters
# mu0=0   : floating section is reference (normalised to zero)
# kappa0  : prior "sample size" — higher = tighter prior on mean
# alpha0  : prior degrees of freedom / 2
# beta0   : prior scale — set to match expected within-segment variance
NIG_MU0    = 0.0
NIG_KAPPA0 = 2.0
NIG_ALPHA0 = 3.0
NIG_BETA0  = 3.0
MIN_SEG    = 5     # minimum samples per segment

# Feature combination weights (higher = more influence on posterior)
W_AMP   = 3.0    # amplitude level  (most robust primary feature)
W_DA    = 2.5    # amplitude gradient (strong but NaN-sensitive)
W_DFREE = 1.5    # flotation residual
W_DSURF = 1.0    # surface slope

# GP search window (km)  — physics constrains GP to be downflow of GZ
DEFAULT_SEARCH_LO_KM = 75.0

# Floating section threshold for normalisation
# Points with x < FLOAT_FRAC × gz_lo are assumed to be firmly floating
FLOAT_FRAC = 0.73

# NaN artifact masking: gradient values in regions with dense NaN clusters
# can produce large spurious spikes — mask these before computing features
GRADIENT_MAX_ABS = 5.0   # dB/km — larger values are treated as artifacts


# =============================================================================
# STEP 1 — Load data
# =============================================================================
def load_data(power_path, bed_path, surface_path):
    bp   = pd.read_csv(power_path)
    bt   = pd.read_csv(bed_path).dropna(subset=["wgs84"])
    sf   = pd.read_csv(surface_path).dropna(subset=["wgs84"])

    x_bp  = bp["along_track"].values / 1000   # km
    amp   = bp["bed_power_dB"].values

    x_bed = bt["along_track"].values / 1000
    h_bed = bt["wgs84"].values

    x_sf  = sf["along_track"].values / 1000
    h_sf  = sf["wgs84"].values

    return dict(x_bp=x_bp, amp=amp, x_bed=x_bed, h_bed=h_bed, x_sf=x_sf, h_sf=h_sf)


# =============================================================================
# STEP 2 — Compute features
# =============================================================================
def compute_features(d, nan_artifact_mask=None):
    """
    Compute four physically motivated features on the bed-power grid:
      f_amp   : normalised filtered amplitude level
      f_dA    : normalised amplitude gradient (dA/dx), NaN artifacts masked
      f_dfree : normalised flotation residual Δfree
      f_dsurf : normalised surface slope
    All normalised to zero mean, unit std in the floating section.
    """
    x_bp  = d["x_bp"];  amp   = d["amp"]
    x_bed = d["x_bed"]; h_bed = d["h_bed"]
    x_sf  = d["x_sf"];  h_sf  = d["h_sf"]

    nan_mask = np.isnan(amp)

    # ── Filtered amplitude ────────────────────────────────────────────────
    amp_i    = pd.Series(amp).interpolate(method="linear").ffill().bfill().values
    b_a, a_a = butter(AMP_ORDER, AMP_WN, btype="low", analog=False)
    amp_f    = filtfilt(b_a, a_a, amp_i)
    amp_f_out = amp_f.copy(); amp_f_out[nan_mask] = np.nan

    # ── Amplitude gradient ────────────────────────────────────────────────
    dA = np.gradient(amp_f, x_bp)
    if nan_artifact_mask is not None:
        dA[nan_artifact_mask] = np.nan
    dA[np.abs(dA) > GRADIENT_MAX_ABS] = np.nan
    dA = pd.Series(dA).interpolate(limit=5).ffill().bfill().values

    # ── Δfree (flotation residual) ────────────────────────────────────────
    # Δfree > 0  → grounded (surface higher than predicted freeboard)
    # Δfree ≈ 0  → floating (in hydrostatic equilibrium)
    # Δfree < 0  → below flotation (full hydrostatic)
    h_sf_on_bed   = np.interp(x_bed, x_sf, h_sf)
    H_bed         = h_sf_on_bed - h_bed
    b2, a2        = butter(ELEV_ORDER, SLOPE_WN, btype="low", analog=False)
    h_bed_sm      = filtfilt(b2, a2, h_bed)
    h_surf_sm     = filtfilt(b2, a2, h_sf_on_bed)
    H_sm          = h_surf_sm - h_bed_sm
    delta_free_sm = h_surf_sm - H_sm * (1.0 - RHO_ICE / RHO_SW)
    delta_free_bp = np.interp(x_bp, x_bed, delta_free_sm)

    # ── Surface slope ─────────────────────────────────────────────────────
    # Near-zero on floating ice (surface decoupled from bed)
    # Large on grounded ice (surface follows ice dynamics)
    h_sf_bp    = np.interp(x_bp, x_sf, h_sf)
    h_sf_bp_sm = filtfilt(b2, a2, h_sf_bp)
    d_surf_bp  = np.gradient(h_sf_bp_sm, x_bp)

    d.update(dict(
        nan_mask    = nan_mask,
        amp_f       = amp_f,
        amp_f_out   = amp_f_out,
        dA          = dA,
        delta_free_bp = delta_free_bp,
        d_surf_bp   = d_surf_bp,
        H_bp        = np.interp(x_bp, x_bed, H_bed),
        h_sf_bp     = h_sf_bp,
        h_bed_sm_bp = np.interp(x_bp, x_bed, h_bed_sm),
        delta_free_bed = np.interp(x_bed, x_bp, delta_free_bp),
        H_bed       = H_bed,
    ))
    return d


# =============================================================================
# STEP 3 — Core Bayesian change-point detection
# =============================================================================
def log_nig_evidence(data, mu0=NIG_MU0, kappa0=NIG_KAPPA0,
                     alpha0=NIG_ALPHA0, beta0=NIG_BETA0):
    """
    Closed-form log marginal likelihood P(data | NIG prior).

    Analytically marginalises over unknown Gaussian mean μ and variance σ²
    using the Normal-Inverse-Gamma conjugate model:

        μ, σ² ~ NIG(mu0, kappa0, alpha0, beta0)
        x_i   ~ Normal(μ, σ²)

    After observing n data points with sample mean x̄ and SS = Σ(xᵢ-x̄)²:

        kappa_n = kappa0 + n
        alpha_n = alpha0 + n/2
        beta_n  = beta0 + SS/2 + kappa0·n·(x̄ - mu0)² / (2·kappa_n)

    The log marginal likelihood is:

        log P(data) = log Γ(alpha_n) - log Γ(alpha0)
                    + alpha0·log(beta0) - alpha_n·log(beta_n)
                    + 0.5·log(kappa0/kappa_n)
                    - (n/2)·log(π)

    This is the Student-t predictive distribution integrated over all
    possible segment means and variances.
    """
    n = len(data)
    if n == 0:
        return 0.0
    xbar    = np.mean(data)
    kappa_n = kappa0 + n
    alpha_n = alpha0 + n / 2.0
    ss      = np.sum((data - xbar) ** 2)
    beta_n  = (beta0
               + 0.5 * ss
               + (kappa0 * n * (xbar - mu0) ** 2) / (2.0 * kappa_n))
    return (gammaln(alpha_n) - gammaln(alpha0)
            + alpha0 * np.log(beta0) - alpha_n * np.log(beta_n)
            + 0.5 * np.log(kappa0 / kappa_n)
            - (n / 2.0) * np.log(np.pi))


def split_point_posterior(obs, mu0=NIG_MU0, kappa0=NIG_KAPPA0,
                           alpha0=NIG_ALPHA0, beta0=NIG_BETA0,
                           min_seg=MIN_SEG,
                           search_lo_idx=None, search_hi_idx=None):
    """
    Bayesian 2-segment split-point posterior.

    For each candidate split index t* in [search_lo_idx, search_hi_idx]:

        log P(t* | obs) ∝ log P(obs[:t*] | NIG) + log P(obs[t*:] | NIG)

    The posterior is normalised over all valid split points.

    Parameters
    ----------
    obs            : 1D array of normalised feature values
    mu0            : NIG prior mean (0 = floating section is reference)
    kappa0, alpha0, beta0 : NIG hyperparameters (see log_nig_evidence)
    min_seg        : minimum samples required in each segment
    search_lo_idx  : restrict search to indices >= this value
    search_hi_idx  : restrict search to indices < this value

    Returns
    -------
    post : normalised posterior probability array (same length as obs)
           post[t] = P(change point just before obs[t])
    """
    T  = len(obs)
    lo = search_lo_idx if search_lo_idx is not None else min_seg
    hi = search_hi_idx if search_hi_idx is not None else T - min_seg

    log_post = np.full(T, -np.inf)
    for t in range(lo, hi):
        log_post[t] = (log_nig_evidence(obs[:t],  mu0, kappa0, alpha0, beta0)
                       + log_nig_evidence(obs[t:], mu0, kappa0, alpha0, beta0))

    # Normalise over valid split points
    finite = np.isfinite(log_post)
    if finite.sum() > 0:
        log_post[finite] -= np.logaddexp.reduce(log_post[finite])

    return np.exp(log_post)


def run_bocpd(d, gz_lo_km, gz_hi_km=None, search_lo_km=DEFAULT_SEARCH_LO_KM,
              smooth_size=3):
    """
    Run the Bayesian split-point change-point detection on four features.

    Steps:
    1. Identify floating section (x < FLOAT_FRAC × gz_lo_km) for normalisation
    2. Normalise each feature: subtract floating mean, divide by floating std
    3. Compute split-point posterior for each feature within search window
    4. Combine posteriors in log-space with reliability weights
    5. Return MAP estimate and credible intervals

    Parameters
    ----------
    d           : data dict from compute_features
    gz_lo_km    : seaward edge of InSAR grounding zone (km)
                  GP search window upper bound
    gz_hi_km    : landward edge of InSAR GZ (km, for display only)
    search_lo_km: lower bound of GP search window (km)
    smooth_size : rolling average window for posterior smoothing

    Returns
    -------
    result dict with posteriors, MAP, and credible intervals
    """
    x_bp       = d["x_bp"]
    float_mask = x_bp < FLOAT_FRAC * gz_lo_km   # firmly floating section

    def normalise(f):
        mu  = np.mean(f[float_mask])
        sig = np.std(f[float_mask]) + 1e-9
        return (f - mu) / sig

    f_amp   = normalise(d["amp_f"])
    f_dA    = normalise(d["dA"])
    f_dfree = normalise(d["delta_free_bp"])
    f_dsurf = normalise(d["d_surf_bp"])

    # Search window indices
    lo_idx = int(np.searchsorted(x_bp, search_lo_km))
    hi_idx = int(np.searchsorted(x_bp, gz_lo_km))

    prior = dict(mu0=NIG_MU0, kappa0=NIG_KAPPA0,
                 alpha0=NIG_ALPHA0, beta0=NIG_BETA0)
    kw    = dict(**prior, min_seg=MIN_SEG,
                 search_lo_idx=lo_idx, search_hi_idx=hi_idx)

    p_amp   = uniform_filter1d(split_point_posterior(f_amp,   **kw), size=smooth_size)
    p_dA    = uniform_filter1d(split_point_posterior(f_dA,    **kw), size=smooth_size)
    p_dfree = uniform_filter1d(split_point_posterior(f_dfree, **kw), size=smooth_size)
    p_dsurf = uniform_filter1d(split_point_posterior(f_dsurf, **kw), size=smooth_size)

    # Combine in log-space with reliability weights
    log_comb = (W_AMP   * np.log(p_amp   + 1e-15)
                + W_DA  * np.log(p_dA    + 1e-15)
                + W_DFREE * np.log(p_dfree + 1e-15)
                + W_DSURF * np.log(p_dsurf + 1e-15)) / (W_AMP + W_DA + W_DFREE + W_DSURF)
    log_comb -= np.logaddexp.reduce(log_comb)
    p_comb   = np.exp(log_comb)

    # MAP and credible intervals (within search window only)
    win       = (x_bp >= search_lo_km) & (x_bp < gz_lo_km)
    x_win     = x_bp[win]
    p_win     = p_comb[win] / p_comb[win].sum()
    cdf       = np.cumsum(p_win)
    gp_km     = float(x_win[np.argmax(p_win)])
    gp_prob   = float(p_win.max())
    lo68      = float(x_win[np.searchsorted(cdf, 0.160)])
    hi68      = float(x_win[np.searchsorted(cdf, 0.840)])
    lo95      = float(x_win[np.searchsorted(cdf, 0.025)])
    hi95      = float(x_win[np.searchsorted(cdf, 0.975)])

    return dict(
        f_amp=f_amp, f_dA=f_dA, f_dfree=f_dfree, f_dsurf=f_dsurf,
        p_amp=p_amp, p_dA=p_dA, p_dfree=p_dfree, p_dsurf=p_dsurf,
        p_comb=p_comb,
        gp_km=gp_km, gp_prob=gp_prob,
        lo68=lo68, hi68=hi68, lo95=lo95, hi95=hi95,
        search_lo_km=search_lo_km, gz_lo_km=gz_lo_km,
        float_mask=float_mask,
    )


# =============================================================================
# STEP 4 — Amplitude gradient GP (reference method)
# =============================================================================
def gradient_gp(d, gz_lo_km):
    """
    Reference method: steepest negative amplitude gradient downflow of GZ.
    For floating→grounded transects (x=0 bright, x=max dark).
    """
    x_bp  = d["x_bp"]
    amp_f = d["amp_f"]
    dA    = np.gradient(amp_f, x_bp)
    dA_clean = dA.copy()
    dA_clean[np.abs(dA_clean) > GRADIENT_MAX_ABS] = 0.0
    down = x_bp < gz_lo_km
    idx  = np.argmin(dA_clean[down])
    return float(x_bp[down][idx])


# =============================================================================
# STEP 5 — Plot
# =============================================================================
def plot(d, result, label="", gz_lo_km=None, gz_hi_km=None,
         gradient_gp_km=None, out_path="bocpd_result.png"):

    x_bp      = d["x_bp"]
    x_bed     = d["x_bed"]
    h_bed     = d["h_bed"]
    x_sf      = d["x_sf"]
    h_sf      = d["h_sf"]
    nan_mask  = d["nan_mask"]
    amp       = d["amp"]
    amp_f_out = d["amp_f_out"]
    H_bed     = d["H_bed"]
    delta_free_bed = d["delta_free_bed"]

    gp        = result["gp_km"]
    lo68, hi68 = result["lo68"], result["hi68"]
    lo95, hi95 = result["lo95"], result["hi95"]
    p_comb    = result["p_comb"]
    search_lo = result["search_lo_km"]

    BG      = "#f6f4f1"
    C_SURF  = "#4a6fa5"
    C_BED_S = "#7a5230"
    C_ICE   = "#ccdff5"
    C_OCEAN = "#3a7abf"
    C_GRD   = "#a68a5b"
    C_GZ    = "#7b2d8b"
    C_GRAD  = "#d94f3d"
    C_BOCPD = "#1a8f5a"
    C_AMP_F = "#378add"
    C_AMP_R = "#b4b2a9"
    TEXT    = "#1a1a1a"

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
            ax.axvline(gradient_gp_km, color=C_GRAD, lw=1.6, ls="--", alpha=0.75, zorder=3)
        ax.axvspan(lo95, hi95, color=C_BOCPD, alpha=0.09, zorder=1)
        ax.axvspan(lo68, hi68, color=C_BOCPD, alpha=0.16, zorder=1)
        ax.axvline(gp,   color=C_BOCPD, lw=2.2, ls="-",  alpha=0.88, zorder=4)

    # Figure
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
    ax = ax_prof
    shade_nans(ax); draw_ref(ax)
    ax.fill_between(x_bed, h_bed.min()-80, np.minimum(h_bed, 0),
                    color=C_OCEAN, alpha=0.18, zorder=0)
    ax.fill_between(x_bed, h_bed.min()-80, h_bed, color=C_GRD, alpha=0.40, zorder=1)
    h_bed_col = np.interp(x_sf, x_bed, h_bed, left=np.nan, right=np.nan)
    ax.fill_between(x_sf, h_bed_col, h_sf, where=~np.isnan(h_bed_col),
                    color=C_ICE, alpha=0.80, zorder=2)
    for mask, col, lbl in [
        (delta_free_bed >  30,                          C_GRD,    "Grounded"),
        ((delta_free_bed >= 0) & (delta_free_bed <= 30),"#f0a500","Near flotation"),
        (delta_free_bed <   0,                          C_OCEAN,  "Floating"),
    ]:
        ax.scatter(x_bed[mask], h_bed[mask], s=2.5, color=col,
                   alpha=0.65, zorder=4, label=lbl)
    ax.plot(x_bp, d["h_bed_sm_bp"], color=C_BED_S, lw=1.8, zorder=5,
            label="Bed (smoothed)")
    ax.plot(x_sf, h_sf, color=C_SURF, lw=1.8, zorder=5, label="Ice surface")
    ax.axhline(0, color=C_OCEAN, lw=0.9, ls=":", alpha=0.8)
    ax.text(1.5, 15, "Sea level", fontsize=7, color=C_OCEAN)

    gp_surf  = float(np.interp(gp, x_sf, h_sf))
    gp_bed_v = float(np.interp(gp, x_bp, d["h_bed_sm_bp"]))
    gp_H     = float(np.interp(gp, x_bp, d["H_bp"]))
    ax.scatter([gp], [gp_surf], s=220, color=C_BOCPD, zorder=11,
               edgecolors="white", linewidths=1.8,
               label=f"BOCPD GP: {gp:.1f} km")
    ax.scatter([gp], [gp_bed_v], s=160, color=C_BOCPD, zorder=11,
               edgecolors="white", linewidths=1.5, marker="v")
    ax.annotate(
        f"BOCPD MAP GP\n{gp:.1f} km\nH = {gp_H:.0f} m\n68% CI [{lo68:.0f}–{hi68:.0f}]",
        xy=(gp, gp_surf), xytext=(gp - 34, gp_surf + 280),
        fontsize=8.5, color=C_BOCPD, fontweight="semibold",
        arrowprops=dict(arrowstyle="->", color=C_BOCPD, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_BOCPD,
                  alpha=0.93, lw=1.2), zorder=12)

    if gradient_gp_km:
        gp_grad_surf = float(np.interp(gradient_gp_km, x_sf, h_sf))
        ax.scatter([gradient_gp_km], [gp_grad_surf], s=150, color=C_GRAD,
                   zorder=10, edgecolors="white", lw=1.6, marker="D",
                   label=f"Gradient GP: {gradient_gp_km:.1f} km")
        ax.annotate(f"Gradient GP\n{gradient_gp_km:.1f} km",
                    xy=(gradient_gp_km, gp_grad_surf),
                    xytext=(gradient_gp_km - 14, gp_grad_surf - 320),
                    fontsize=8, color=C_GRAD,
                    arrowprops=dict(arrowstyle="->", color=C_GRAD, lw=1.1),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=C_GRAD, alpha=0.9, lw=1.0), zorder=11)

    if gz_lo_km and gz_hi_km:
        gz_mid  = (gz_lo_km + gz_hi_km) / 2
        gz_surf = float(np.interp(gz_mid, x_sf, h_sf))
        ax.annotate("InSAR GZ\n(reference)",
                    xy=(gz_mid, gz_surf), xytext=(gz_mid + 20, gz_surf + 260),
                    fontsize=8, color=C_GZ, fontweight="semibold",
                    arrowprops=dict(arrowstyle="->", color=C_GZ, lw=1.1),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=C_GZ, alpha=0.9, lw=1.0), zorder=11)

    ax.set_ylabel("Elevation (m WGS84)", fontsize=10, color=TEXT)
    ax.set_ylim(h_bed.min() - 100, h_sf.max() * 1.07)
    ax.set_xlim(-2, x_bp[-1] + 3)
    ax.tick_params(labelbottom=False, colors=TEXT, labelsize=8.5)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2,
              framealpha=0.9, edgecolor="#ccc", facecolor="white")
    ax.grid(True, alpha=0.13, lw=0.6)
    ax.set_title(
        f"{label}  |  Bayesian 2-segment split-point change-point detection\n"
        f"Green: BOCPD MAP ± CI  ·  Red dashed: amplitude gradient  ·  "
        f"Purple: InSAR GZ (reference)",
        fontsize=10.5, color=TEXT, pad=8, fontweight="semibold")

    # ── Panel B: normalised features ───────────────────────────────────────
    ax = ax_feat; shade_nans(ax); draw_ref(ax)
    ax.plot(x_bp, result["f_amp"],   color="#c45ab3", lw=1.5,
            label="Amplitude level", zorder=3)
    ax.plot(x_bp, result["f_dA"],    color="#e87722", lw=1.5,
            label="Amplitude gradient (dA/dx)", zorder=3)
    ax.plot(x_bp, result["f_dfree"], color="#1a6bbf", lw=1.5,
            label="Flotation residual (Δfree)", zorder=3)
    ax.plot(x_bp, result["f_dsurf"], color="#888",    lw=1.3,
            label="Surface slope", alpha=0.7, zorder=3)
    ax.axhline(0, color="#999", lw=0.8, ls=":")
    ax.set_ylabel("Normalised\nfeature (σ)", fontsize=9.5, color=TEXT)
    ax.tick_params(labelbottom=False, colors=TEXT, labelsize=8.5)
    ax.legend(fontsize=7.5, loc="upper left", ncol=4,
              framealpha=0.9, edgecolor="#ccc", facecolor="white")
    ax.grid(True, alpha=0.13, lw=0.6); ax.set_ylim(-9, 6)

    # ── Panel C: posteriors ────────────────────────────────────────────────
    ax = ax_post; shade_nans(ax); draw_ref(ax)
    win = (x_bp >= search_lo) & (x_bp < gz_lo_km)
    sc  = max(p_comb[win].max(),
              result["p_amp"][win].max(),
              result["p_dA"][win].max(),
              result["p_dfree"][win].max()) or 1.0
    scale = 1.0 / sc
    ax.fill_between(x_bp[win], 0, p_comb[win]*scale,
                    color=C_BOCPD, alpha=0.25, zorder=1)
    ax.plot(x_bp[win], p_comb[win]*scale,        color=C_BOCPD, lw=2.5,
            label=f"Combined P(GP=x)  MAP={gp:.1f}km", zorder=5)
    ax.plot(x_bp[win], result["p_amp"][win]*scale,   color="#c45ab3",
            lw=1.3, ls="--", alpha=0.8, label="Amplitude level", zorder=4)
    ax.plot(x_bp[win], result["p_dA"][win]*scale,    color="#e87722",
            lw=1.3, ls="--", alpha=0.8, label="Amplitude gradient", zorder=4)
    ax.plot(x_bp[win], result["p_dfree"][win]*scale, color="#1a6bbf",
            lw=1.3, ls="--", alpha=0.8, label="Flotation residual", zorder=4)
    ax.plot(x_bp[win], result["p_dsurf"][win]*scale, color="#888",
            lw=1.1, ls="--", alpha=0.6, label="Surface slope", zorder=3)
    ax.axvline(gp, color=C_BOCPD, lw=2.2, zorder=6)
    ax.text(gp + 0.3, 0.55, f"MAP\n{gp:.1f}km",
            fontsize=8, color=C_BOCPD, fontweight="semibold")
    ax.set_ylabel("P(GP = x)\n(search window)", fontsize=9.5, color=TEXT)
    ax.tick_params(labelbottom=False, colors=TEXT, labelsize=8.5)
    ax.legend(fontsize=7.5, loc="upper left", ncol=3,
              framealpha=0.9, edgecolor="#ccc", facecolor="white")
    ax.grid(True, alpha=0.13, lw=0.6); ax.set_ylim(bottom=0)

    # ── Panel D: amplitude ─────────────────────────────────────────────────
    ax = ax_amp; shade_nans(ax); draw_ref(ax)
    ax.scatter(x_bp, amp, s=5, color=C_AMP_R, alpha=0.55, zorder=2)
    ax.plot(x_bp, amp_f_out, color=C_AMP_F, lw=1.8, zorder=3,
            label="Filtered amplitude")
    ax.scatter([gp], [float(np.interp(gp, x_bp, d["amp_f"]))],
               s=180, color=C_BOCPD, zorder=7, edgecolors="white", lw=1.6,
               label=f"BOCPD GP: {gp:.1f} km")
    if gradient_gp_km:
        ax.scatter([gradient_gp_km],
                   [float(np.interp(gradient_gp_km, x_bp, d["amp_f"]))],
                   s=130, color=C_GRAD, marker="D", zorder=7,
                   edgecolors="white", lw=1.4,
                   label=f"Gradient GP: {gradient_gp_km:.1f} km")
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
    plt.show()

    # Print summary
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  BOCPD MAP GP   : {gp:.2f} km")
    print(f"  68% CI         : [{lo68:.1f}, {hi68:.1f}] km")
    print(f"  95% CI         : [{lo95:.1f}, {hi95:.1f}] km")
    if gradient_gp_km:
        print(f"  Gradient GP    : {gradient_gp_km:.2f} km")
    if gz_lo_km:
        print(f"  InSAR GZ       : {gz_lo_km:.2f} – {gz_hi_km:.2f} km")
        print(f"  Offset (MAP)   : {gz_lo_km - gp:.1f} km downflow of GZ seaward edge")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bayesian change-point detection of grounding points from IPR data")
    parser.add_argument("--power",     required=True, help="Bed power CSV")
    parser.add_argument("--bed",       required=True, help="Bed elevation CSV")
    parser.add_argument("--surface",   required=True, help="Ice surface CSV")
    parser.add_argument("--gz_lo",     type=float, default=None,
                        help="InSAR GZ seaward edge (km) — sets search window upper bound")
    parser.add_argument("--gz_hi",     type=float, default=None,
                        help="InSAR GZ landward edge (km) — display only")
    parser.add_argument("--search_lo", type=float, default=DEFAULT_SEARCH_LO_KM,
                        help=f"GP search window start (km, default={DEFAULT_SEARCH_LO_KM})")
    parser.add_argument("--label",     default="Glacier",
                        help="Label for figure title")
    parser.add_argument("--out",       default="bocpd_result.png",
                        help="Output figure path")
    args = parser.parse_args()

    # If no InSAR GZ provided, use full transect as search window
    gz_lo = args.gz_lo if args.gz_lo else 999.0

    # Load and process
    d = load_data(args.power, args.bed, args.surface)
    d["amp_f"] = d["amp"]  # placeholder before compute_features
    d = compute_features(d)

    # Run detection
    result = run_bocpd(d, gz_lo_km=gz_lo, gz_hi_km=args.gz_hi,
                       search_lo_km=args.search_lo)
    grad_gp = gradient_gp(d, gz_lo)

    # Plot
    plot(d, result,
         label=args.label,
         gz_lo_km=args.gz_lo, gz_hi_km=args.gz_hi,
         gradient_gp_km=grad_gp,
         out_path=args.out)
