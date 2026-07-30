"""
Figures for grounding point detection.

Takes a profile and the results detected on it; knows nothing about how they
were produced.
"""

from typing import Optional, Sequence

import numpy as np

from . import features as _features
from .profile import GlacierProfile
from .result import DetectionResult

# Okabe-Ito, colourblind-safe.
C = {
    "onset": "#009E73",
    "surf": "#0072B2",
    "bed": "#8a6d3b",
    "amp": "#0072B2",
    "resid": "#009E73",
    "map": "#D55E00",
    "gz": "#CC79A7",
    "window": "#009E73",
    "ink": "#1a1a1a",
    "muted": "#6b6b6b",
}
MODEL_C = {"gauss": "#E69F00", "ifm": "#56B4E9", "fullcov": "#009E73"}


def plot_detection(profile: GlacierProfile,
                   results: Sequence[DetectionResult],
                   out_path: str,
                   gz: Optional[tuple] = None,
                   threshold_m: float = 30.0,
                   pad_km: float = 25.0,
                   title: Optional[str] = None):
    """
    Four panels: geometry, the flotation residual that set the window, bed
    power, and the posterior.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_name = {r.detector: r for r in results}
    primary = by_name.get("bocpd", results[0])
    onset = by_name.get("onset")
    lo, hi = primary.search_window
    x0, x1 = lo - pad_km, hi + pad_km

    fig, axs = plt.subplots(4, 1, figsize=(13, 14), sharex=True,
                            gridspec_kw=dict(height_ratios=[2.1, 1.3, 1.3, 1.7],
                                             hspace=0.09))
    for ax in axs:
        ax.grid(True, alpha=0.15, lw=0.6)
        ax.tick_params(colors=C["ink"], labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
        ax.axvspan(lo, hi, color=C["window"], alpha=0.09, zorder=0)
        if gz:
            ax.axvspan(gz[0], gz[1], color=C["gz"], alpha=0.22, zorder=1)

    vis = (profile.x >= x0) & (profile.x <= x1)

    # -- geometry ---------------------------------------------------------
    ax = axs[0]
    ax.plot(profile.x, profile.h_surf, color=C["surf"], lw=2.0, label="Ice surface")
    ax.plot(profile.x, profile.h_bed, color=C["bed"], lw=2.0, label="Bed")
    ax.axhline(0, color=C["muted"], lw=0.9, ls=":")
    ax.axvline(primary.map_km, color=C["map"], lw=1.8, ls="--", alpha=0.9)
    if onset is not None:
        ax.axvline(onset.map_km, color=C["onset"], lw=2.2, alpha=0.95)
        ax.scatter([onset.map_km],
                   [float(np.interp(onset.map_km, profile.x, profile.h_surf))],
                   s=180, color=C["onset"], edgecolors="white", lw=1.8, zorder=7,
                   label=f"grounding point (onset) {onset.map_km:.2f} km")
    ax.scatter([primary.map_km],
               [float(np.interp(primary.map_km, profile.x, profile.h_surf))],
               s=120, color=C["map"], marker="s", edgecolors="white", lw=1.6,
               zorder=6, label=f"changepoint {primary.map_km:.2f} km")
    _tight_ylim(ax, profile.h_bed[vis], profile.h_surf[vis])
    ax.set_ylabel("Elevation (m WGS84)", fontsize=10, color=C["ink"])
    ax.legend(fontsize=8.5, loc="upper left", ncol=3, framealpha=0.9,
              edgecolor="#ccc", facecolor="white")
    ax.set_title(title or f"{profile.source}", fontsize=12, color=C["ink"],
                 pad=10, fontweight="semibold")

    # -- flotation residual -----------------------------------------------
    ax = axs[1]
    residual = _features.FlotationResidual().compute(profile)
    ax.plot(profile.x, residual, color=C["resid"], lw=1.8,
            label="Flotation residual (smoothed)")
    ax.axhline(threshold_m, color=C["ink"], lw=1.1, ls="--",
               label=f"grounded threshold {threshold_m:g} m")
    ax.axhline(0, color=C["muted"], lw=0.9, ls=":")
    ax.set_ylabel("Residual (m)", fontsize=10, color=C["ink"])
    lo_v, hi_v = np.nanmin(residual[vis]), np.nanmax(residual[vis])
    ax.set_ylim(min(lo_v, -10), max(hi_v, threshold_m * 2.2))
    ax.legend(fontsize=8.5, loc="upper left", ncol=2, framealpha=0.9,
              edgecolor="#ccc", facecolor="white")

    # -- bed power --------------------------------------------------------
    ax = axs[2]
    ax.plot(profile.x, profile.amp, color=C["muted"], lw=0.9, alpha=0.6,
            label="Bed power")
    ax.plot(profile.x, _features.AmplitudeLevel().compute(profile),
            color=C["amp"], lw=1.9, label="Filtered")
    for gap_lo, gap_hi in profile.nan_blocks():
        ax.axvspan(gap_lo, gap_hi, color="gray", alpha=0.16, zorder=0)
    if onset is not None:
        ax.axhline(onset.extra["baseline_dB"], color=C["onset"], lw=1.1, ls=":",
                   label=f"floating baseline {onset.extra['baseline_dB']:.1f} dB")
        ax.axhline(onset.extra["threshold_dB"], color=C["onset"], lw=1.1,
                   ls="--", label="onset threshold")
        ax.axvline(onset.map_km, color=C["onset"], lw=2.2,
                   label=f"onset {onset.map_km:.2f} km")
    finite = profile.amp[vis][np.isfinite(profile.amp[vis])]
    if finite.size:
        ax.set_ylim(finite.min() - 3, finite.max() + 3)
    if onset is not None:
        # After set_ylim, so the annotation lands inside the panel.
        y0, y1 = ax.get_ylim()
        y_arrow = y0 + 0.10 * (y1 - y0)
        ax.annotate("", xy=(onset.map_km, y_arrow),
                    xytext=(primary.map_km, y_arrow),
                    arrowprops=dict(arrowstyle="<->", color=C["ink"], lw=1.3))
        ax.text(0.5 * (onset.map_km + primary.map_km), y_arrow + 0.03 * (y1 - y0),
                f"transition {primary.map_km - onset.map_km:.1f} km",
                ha="center", fontsize=8.5, color=C["ink"])
    ax.set_ylabel("Bed power (dB)", fontsize=10, color=C["ink"])
    ax.legend(fontsize=8.5, loc="lower left", ncol=2, framealpha=0.9,
              edgecolor="#ccc", facecolor="white")

    # -- posterior --------------------------------------------------------
    ax = axs[3]
    for name, post in primary.extra.get("model_posteriors", {}).items():
        ax.plot(primary.x_km, post / post.max(), color=MODEL_C.get(name, "#999"),
                lw=1.4, ls=":", alpha=0.85,
                label=f"{name}  {primary.extra['model_maps'][name]:.2f} km")
    if primary.posterior is not None:
        p = primary.posterior / primary.posterior.max()
        ax.fill_between(primary.x_km, 0, p, color=C["map"], alpha=0.18)
        ax.plot(primary.x_km, p, color=C["map"], lw=2.4,
                label=f"combined  {primary.map_km:.2f} km")
        for mass, alpha in ((0.95, 0.10), (0.68, 0.20)):
            ci_lo, ci_hi = primary.credible_interval(mass)
            ax.axvspan(ci_lo, ci_hi, color=C["map"], alpha=alpha, zorder=0)
    for r in results[1:]:
        ax.axvline(r.map_km, color=C["ink"], lw=1.4, ls="--", alpha=0.7,
                   label=f"{r.detector}  {r.map_km:.2f} km")
    ax.set_ylim(0, 1.15)
    ax.set_xlim(x0, x1)
    ax.set_ylabel("P(GP = x), scaled", fontsize=10, color=C["ink"])
    ax.set_xlabel("Along-track distance (km)", fontsize=10, color=C["ink"])
    ax.legend(fontsize=8.5, loc="upper left", ncol=3, framealpha=0.9,
              edgecolor="#ccc", facecolor="white")

    if gz:
        axs[0].text(np.mean(gz), axs[0].get_ylim()[1], " InSAR GZ", ha="center",
                    va="top", fontsize=9, color=C["gz"], fontweight="semibold")

    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _tight_ylim(ax, lower, upper, pad_frac=0.12):
    lo = float(np.nanmin(lower))
    hi = float(np.nanmax(upper))
    pad = pad_frac * (hi - lo)
    ax.set_ylim(lo - pad, hi + 2.5 * pad)
