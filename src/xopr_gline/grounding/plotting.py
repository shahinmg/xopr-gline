"""
Figures for grounding point detection.

Takes a profile and the results detected on it; knows nothing about how they
were produced. Styled after polartoolkit's profile cross sections: filled
ice/water/earth layers with black interfaces, GMT frames, legends outside the
axes, and A/B labels on the section ends.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import features as _features
from .geoid import sample_bedmachine
from .profile import GlacierProfile
from .result import DetectionResult

# Cross-section fills, from polartoolkit.profiles.default_layers.
LAYER_C = {"ice": "lightskyblue", "water": "darkblue", "earth": "#c8a165"}

# Matches GlacierProfile.ice_mask, which the fills cannot use once the picks
# have been bridged across their gaps.
_MIN_THICKNESS_M = 25.0

# What the column holds at a sample. _NODATA and _NOBASE are their own states
# on purpose: a stretch with no pick is unknown, not ocean.
_NODATA, _GROUNDED, _FLOATING, _WATER, _NOBASE = 0, 1, 2, 3, 4

# Longest bed-pick gap worth interpolating across. 20060608_01 has 8 km of
# shelf with no bottom pick, and bridging it drew the ice base 80 m above sea
# level with ocean underneath. A gap this wide is not interpolation.
_MAX_BRIDGE_KM = 2.5

# Okabe-Ito, colourblind-safe.
C = {
    "onset": "#009E73",
    "surf": "#0072B2",
    "bed": "#8a6d3b",
    "amp": "#0072B2",
    "resid": "#009E73",
    "map": "#D55E00",
    "gz": "#CC79A7",
    "terminus": "#E69F00",
    "window": "#009E73",
    "ink": "#1a1a1a",
    "muted": "#6b6b6b",
}
MODEL_C = {"gauss": "#E69F00", "ifm": "#56B4E9", "fullcov": "#009E73"}

# Layer picks over the greyscale radargram, where the section's browns and
# blues disappear into the echo.
RADAR_C = {"surf": "#56B4E9", "bed": "#F0E442"}

# GMT defaults: Helvetica, thin black frame, outward ticks, 300 dpi.
GMT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "black",
    "axes.labelsize": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "savefig.dpi": 300,
}


def plot_detection(profile: GlacierProfile,
                   results: Sequence[DetectionResult],
                   out_path: str,
                   gz: Optional[tuple] = None,
                   threshold_m: float = 30.0,
                   pad_km: float = 25.0,
                   title: Optional[str] = None,
                   fill_layers: bool = True,
                   project_bathymetry: bool = True,
                   bathymetry=None,
                   radargram=None,
                   gz_label: Optional[str] = None,
                   features: Optional[Sequence] = None):
    """
    Geometry, the flotation residual that set the window, bed power, and the
    posterior, under an optional radargram.

    bathymetry draws the seabed under floating ice, where radar stops at the ice
    bottom: a BedMachine path to sample 'bed' along the line, or an array on the
    profile's grid. Without it the seabed is drawn flat, and either way the
    radar picks are untouched.

    radargram is a Radargram on the profile's along-track axis, drawn as a top
    panel with the layer picks and the detected changepoints over the echo.

    gz shades a reference grounding zone: one (lo_km, hi_km) band or several.
    gz_label names it in the legend, e.g. the survey epoch '2011-2015'.

    features is the channel set the detector was given, so the bed-power panel
    draws what it actually read. Without it the panel falls back to the shipped
    cutoff, which misrepresents any run that was tuned away from it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    with plt.rc_context(GMT_RC):
        by_name = {r.detector: r for r in results}
        primary = by_name.get("bocpd", results[0])
        onset = by_name.get("onset")
        lo, hi = primary.search_window
        x0, x1 = _data_extent(profile, lo, hi, pad_km)
        bands = _gz_bands(gz)
        # Orientation from the searched stretch, not the whole segment: a
        # segment that turns fits a sign pointing the wrong way, which flips
        # the front, the grounded side and the held bottom all at once.
        sign = profile.landward_sign(window=(lo, hi))

        ratios = {"radar": 1.9, "geom": 2.1, "resid": 1.3, "amp": 1.3,
                  "post": 1.7}
        panels = (["radar"] if radargram is not None else []) + [
            "geom", "resid", "amp", "post"]
        fig, axs = plt.subplots(
            len(panels), 1, figsize=(13, sum(ratios[p] for p in panels) * 2.19),
            sharex=True,
            gridspec_kw=dict(height_ratios=[ratios[p] for p in panels],
                             hspace=0.09))
        panel = dict(zip(panels, axs))
        for ax in axs:
            _gmt_frame(ax)
            # Not over the radargram: a translucent wash reads as a colour cast
            # on the greyscale echo, and the panels below already show the band.
            if ax is panel.get("radar"):
                continue
            ax.axvspan(lo, hi, color=C["window"], alpha=0.09, zorder=4)
            for band in bands:
                ax.axvspan(band[0], band[1], color=C["gz"], alpha=0.22, zorder=4)

        # Calving front from the data, where thickness collapses. Absent on a
        # transect that never reaches open water, e.g. Petermann 7:11. The
        # other crossings are bergs in the fjord: drawn, because they explain
        # the bed power, but faint, because the grounding point is measured
        # against the front alone.
        # From the searched stretch's landward edge, matching _qc_fields:
        # not the profile's end, which is the far side on a segment that
        # turns, and not the pick, which can sit just seaward of the front.
        front = profile.calving_front_km(lo if sign < 0 else hi, sign=sign)
        termini = [t for t in profile.terminus_crossings_km() if x0 <= t <= x1]
        bergs = [t for t in termini if front is None or t != front]
        for ax in axs:
            for berg in bergs:
                ax.axvline(berg, color=C["terminus"], lw=1.0, ls=":",
                           alpha=0.45, zorder=5)
            if front is not None and x0 <= front <= x1:
                ax.axvline(front, color=C["terminus"], lw=1.6, ls="-.",
                           alpha=0.9, zorder=5)

        vis = (profile.x >= x0) & (profile.x <= x1)
        sea_level = _sea_level(profile)

        # -- radargram ------------------------------------------------------
        if radargram is not None:
            _draw_radargram(panel["radar"], radargram, profile, vis, primary,
                            onset)

        # -- geometry -----------------------------------------------------
        ax = panel["geom"]
        # The seabed sets the floor along with the picks: BedMachine's trough
        # runs 200 m below Helheim's deepest bed pick and would be clipped off.
        bathy = _resolve_bathymetry(profile, bathymetry)
        _tight_ylim(ax, _depth_bound(profile, bathy, vis), profile.h_surf[vis])
        floor = ax.get_ylim()[0]
        # The changepoint is the grounding point, so it is what splits grounded
        # from floating in the section. The onset is where the bed echo starts
        # to change, i.e. the seaward edge of the transition, not the pick.
        sec = _build_section(profile, primary.map_km, sea_level, bathy, floor,
                             project_bathymetry, sign=sign)
        ice = sec.ice
        handles = []
        if fill_layers:
            _fill_section(ax, profile.x, sec, sea_level, floor)
            handles += [Patch(fc=LAYER_C["ice"], ec="black", lw=0.8, label="Ice"),
                        Patch(fc=LAYER_C["water"], ec="black", lw=0.8,
                              label="Water"),
                        Patch(fc=LAYER_C["earth"], ec="black", lw=0.8,
                              label="Bed")]
            surf_c = bed_c = "black"
        else:
            surf_c, bed_c = C["surf"], C["bed"]
            handles += [Line2D([], [], color=surf_c, lw=1.6, label="Ice surface"),
                        Line2D([], [], color=bed_c, lw=1.6, label="Bed")]
        # Only where there is glacier ice. Past the calving front the layer
        # picks collapse onto the sea surface, and drawing them there would
        # trace the water rather than the ice this figure is about.
        ax.plot(profile.x, np.where(~sec.surf_bridged, sec.surf, np.nan),
                color=surf_c, lw=1.2, zorder=5)
        ax.plot(profile.x, np.where(~sec.base_bridged, sec.base, np.nan),
                color=bed_c, lw=1.2, zorder=5)
        # Bridged stretches, drawn one sample wide either side so they join the
        # measured pick rather than floating free of it.
        for values, bridged, color in ((sec.surf, sec.surf_bridged, surf_c),
                                       (sec.base, sec.base_bridged, bed_c)):
            if bridged.any():
                ax.plot(profile.x, np.where(_dilate(bridged) & ice, values,
                                            np.nan),
                        color=color, lw=1.2, ls=(0, (4, 2)), zorder=5)
        if sec.base_bridged.any() or sec.surf_bridged.any():
            handles.append(Line2D([], [], color=bed_c, lw=1.2, ls=(0, (4, 2)),
                                  label="Interpolated across gap"))
        if sec.projected.any():
            source = "BedMachine" if bathymetry is not None else "projected"
            handles.append(Line2D([], [], color="black", lw=1.2, ls="--",
                                  label=f"Seabed ({source})"))
        if (sec.state == _NODATA).any() or sec.unresolved.any():
            handles.append(Patch(fc="none", ec=C["muted"], lw=0.6, hatch="///",
                                 label="no bottom pick"))
        ax.axhline(sea_level, color="black", lw=0.7, ls="--", alpha=0.8,
                   zorder=5)
        if onset is not None:
            ax.axvline(onset.map_km, color=C["onset"], lw=1.6, ls=":",
                       alpha=0.9, zorder=6)
            ax.scatter([onset.map_km], [_surf_at(profile, sec, onset.map_km)],
                       s=110, color=C["onset"], marker="v", edgecolors="white",
                       lw=1.4, zorder=7)
            handles.append(Line2D([], [], color=C["onset"], marker="v", lw=1.6,
                                  ls=":", mec="white", mew=1.2, ms=8,
                                  label=f"bed-echo onset {onset.map_km:.2f} km"))
        ax.axvline(primary.map_km, color=C["map"], lw=2.2, alpha=0.95, zorder=6)
        ax.scatter([primary.map_km], [_surf_at(profile, sec, primary.map_km)],
                   s=180, color=C["map"], edgecolors="white", lw=1.8, zorder=7)
        handles.append(Line2D([], [], color=C["map"], marker="o", lw=2.2,
                              mec="white", mew=1.4, ms=10,
                              label=f"grounding point (changepoint) "
                                    f"{primary.map_km:.2f} km"))
        if front is not None and x0 <= front <= x1:
            handles.append(Line2D([], [], color=C["terminus"], lw=1.6, ls="-.",
                                  label=f"calving front {front:.2f} km"))
        if bergs:
            handles.append(Line2D([], [], color=C["terminus"], lw=1.0, ls=":",
                                  alpha=0.45,
                                  label=f"berg / other ice ({len(bergs)})"))
        if bands:
            handles.append(Patch(fc=C["gz"], ec="none", alpha=0.45,
                                 label=f"InSAR GZ {gz_label}" if gz_label
                                 else "InSAR GZ"))
        handles.append(Line2D([], [], color="black", lw=0.7, ls="--",
                              label="sea level"))
        ax.set_ylabel("Elevation (m WGS84)")
        _gmt_legend(ax, handles=handles)
        # Title and end labels on the topmost panel, whichever that is.
        axs[0].set_title(title or f"{profile.source}", fontsize=12,
                         color="black", pad=16, fontweight="semibold")
        _start_end_labels(axs[0])

        # -- flotation residual -------------------------------------------
        ax = panel["resid"]
        # The same quantity flotation_window used, not the 71.6 km
        # FlotationResidual feature: that cutoff exceeds a short leg's whole
        # length, so it collapses to the mean and draws a flat line that hides
        # the crossing it is meant to explain.
        residual = profile.smoothed_residual()
        ax.plot(profile.x, residual, color=C["resid"], lw=1.8,
                label="Flotation residual (5 km smoothed)")
        ax.axhline(threshold_m, color="black", lw=1.0, ls="--",
                   label=f"grounded threshold {threshold_m:g} m")
        ax.axhline(0, color=C["muted"], lw=0.7, ls=":")
        ax.set_ylabel("Residual (m)")
        lo_v, hi_v = np.nanmin(residual[vis]), np.nanmax(residual[vis])
        ax.set_ylim(min(lo_v, -10), max(hi_v, threshold_m * 2.2))
        _gmt_legend(ax)

        # -- bed power ------------------------------------------------------
        ax = panel["amp"]
        ax.plot(profile.x, profile.amp, color=C["muted"], lw=0.9, alpha=0.6,
                label="Bed power")
        amp_feature = _amp_feature(features)
        ax.plot(profile.x, amp_feature.compute(profile),
                color=C["amp"], lw=1.9, label=_amp_label(amp_feature))
        for gap_lo, gap_hi in profile.nan_blocks():
            ax.axvspan(gap_lo, gap_hi, color="gray", alpha=0.16, zorder=0)
        if onset is not None:
            ax.axhline(onset.extra["baseline_dB"], color=C["onset"], lw=1.1,
                       ls=":",
                       label=f"floating baseline {onset.extra['baseline_dB']:.1f} dB")
            ax.axhline(onset.extra["threshold_dB"], color=C["onset"], lw=1.1,
                       ls="--", label="onset threshold")
            ax.axvline(onset.map_km, color=C["onset"], lw=1.6, ls=":",
                       label=f"bed-echo onset {onset.map_km:.2f} km")
        ax.axvline(primary.map_km, color=C["map"], lw=2.2, alpha=0.95,
                   label=f"grounding point {primary.map_km:.2f} km")
        finite = profile.amp[vis][np.isfinite(profile.amp[vis])]
        if finite.size:
            ax.set_ylim(finite.min() - 3, finite.max() + 3)
        if onset is not None:
            # After set_ylim, so the annotation lands inside the panel.
            y0, y1 = ax.get_ylim()
            y_arrow = y0 + 0.10 * (y1 - y0)
            ax.annotate("", xy=(onset.map_km, y_arrow),
                        xytext=(primary.map_km, y_arrow),
                        arrowprops=dict(arrowstyle="<->", color="black", lw=1.3))
            ax.text(0.5 * (onset.map_km + primary.map_km),
                    y_arrow + 0.03 * (y1 - y0),
                    f"transition {primary.map_km - onset.map_km:.1f} km",
                    ha="center", fontsize=8.5, color="black")
        ax.set_ylabel("Bed power (dB)")
        _gmt_legend(ax)

        # -- posterior ------------------------------------------------------
        ax = panel["post"]
        for name, post in primary.extra.get("model_posteriors", {}).items():
            ax.plot(primary.x_km, post / post.max(),
                    color=MODEL_C.get(name, "#999"), lw=1.4, ls=":", alpha=0.85,
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
            ax.axvline(r.map_km, color="black", lw=1.4, ls="--", alpha=0.7,
                       label=f"{r.detector}  {r.map_km:.2f} km")
        ax.set_ylim(0, 1.15)
        ax.set_xlim(x0, x1)
        ax.set_ylabel("P(GP = x), scaled")
        ax.set_xlabel("Along-track distance (km)")
        _gmt_legend(ax)

        if bands:
            # Labelled once, on the crossing the pick is measured against.
            geom = panel["geom"]
            near = min(bands, key=lambda b: abs(np.mean(b) - primary.map_km))
            geom.text(np.mean(near), geom.get_ylim()[1], " InSAR GZ",
                      ha="center", va="top", fontsize=9, color=C["gz"],
                      fontweight="semibold", zorder=8)

        fig.savefig(out_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return out_path


def _gz_bands(gz) -> list:
    """
    Reference zones as a list of (lo_km, hi_km).

    Accepts a single band, as the CLI's --gz_lo/--gz_hi give, or several, as a
    track crossing the same zone twice does.
    """
    if gz is None or len(gz) == 0:
        return []
    if np.isscalar(gz[0]):
        return [(float(gz[0]), float(gz[1]))]
    return [(float(lo), float(hi)) for lo, hi in gz]


def _draw_radargram(ax, rad, profile, vis, primary, onset):
    """
    Echo image with the layer picks and the detected changepoints over it.

    The picks are the raw ones the detectors ran on, not the gap-bridged
    section: the point of the panel is to show where the picker put them
    against the return it put them on.
    """
    from matplotlib.lines import Line2D

    vmin, vmax = rad.clim()
    ax.imshow(rad.power, extent=rad.extent, origin="lower", aspect="auto",
              cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest",
              zorder=0)
    # The grid would read as layering in the echo.
    ax.grid(False)

    handles = []
    for values, colour, label in (
        (profile.h_surf, RADAR_C["surf"], "surface pick"),
        (profile.h_bed, RADAR_C["bed"], "bed pick"),
    ):
        ax.plot(profile.x, values, color=colour, lw=1.1, ls=":", zorder=6)
        handles.append(Line2D([], [], color=colour, lw=1.4, ls=":", label=label))

    if onset is not None:
        ax.axvline(onset.map_km, color=C["onset"], lw=1.6, ls=":", alpha=0.9,
                   zorder=6)
        handles.append(Line2D([], [], color=C["onset"], lw=1.6, ls=":",
                              label=f"bed-echo onset {onset.map_km:.2f} km"))
    ax.axvline(primary.map_km, color=C["map"], lw=2.2, alpha=0.95, zorder=6)
    handles.append(Line2D([], [], color=C["map"], lw=2.2,
                          label=f"grounding point (changepoint) "
                                f"{primary.map_km:.2f} km"))

    # Clipped to the loaded grid: padding out past it would open a white band
    # above the echo rather than showing more of it.
    picks = np.concatenate([profile.h_bed[vis], profile.h_surf[vis]])
    picks = picks[np.isfinite(picks)]
    if picks.size:
        _tight_ylim(ax, picks, picks)
    lo, hi = ax.get_ylim()
    ax.set_ylim(max(lo, float(rad.z[0])) if picks.size else float(rad.z[0]),
                min(hi, float(rad.z[-1])) if picks.size else float(rad.z[-1]))
    ax.set_ylabel("Elevation (m WGS84)")
    _gmt_legend(ax, handles=handles)


def _gmt_frame(ax):
    """Black frame ticked on all four sides, faint grid."""
    from matplotlib.ticker import AutoMinorLocator

    ax.grid(True, color="gray", alpha=0.25, lw=0.4, ls=":")
    ax.set_axisbelow(False)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="major", length=5, width=0.8)
    ax.tick_params(which="minor", length=2.5, width=0.6)
    ax.tick_params(top=True, right=True, labeltop=False, labelright=False)


def _gmt_legend(ax, handles=None):
    """Legend outside the bottom right corner, unboxed, as GMT JBR+jBL."""
    kw = dict(loc="lower left", bbox_to_anchor=(1.012, 0.0), frameon=False,
              borderaxespad=0, handlelength=1.8, labelspacing=0.35)
    if handles is None:
        ax.legend(**kw)
    else:
        ax.legend(handles=handles, **kw)


def _amp_feature(features):
    """The bed-power channel the detector was given, or the shipped default."""
    for f in features or ():
        if getattr(f, "name", None) == "amp":
            return f
    return _features.AmplitudeLevel()


def _amp_label(feature) -> str:
    """Name the bed-power curve by what was actually done to it."""
    cutoff = getattr(getattr(feature, "filt", None), "wavelength_km", None)
    despike = getattr(feature, "despike_km", 0.0)
    parts = []
    if despike:
        parts.append(f"despiked {despike:g} km")
    parts.append(f"{cutoff:g} km low-pass" if cutoff else "no low-pass")
    return ", ".join(parts).capitalize()


def _start_end_labels(ax, start="A", end="B"):
    """Boxed A/B labels on the section ends."""
    box = dict(facecolor="white", edgecolor="black", linewidth=0.8,
               boxstyle="square,pad=0.25")
    for x, text, ha in ((0.0, start, "right"), (1.0, end, "left")):
        ax.text(x, 1.005, text, transform=ax.transAxes, ha=ha, va="bottom",
                fontsize=13, fontweight="bold", color="black", bbox=box,
                clip_on=False, zorder=9)


def _sea_level(profile) -> float:
    """Sea surface in the profile's elevation datum, i.e. the geoid."""
    g = np.broadcast_to(np.asarray(profile.geoid_separation_m, dtype=float),
                        profile.x.shape)
    return float(np.nanmean(g))


@dataclass
class _Section:
    """
    One classification of the column at every sample, with ordered interfaces.

    Every fill reads from this, so the three polygons tile the column exactly
    once. The old code derived a mask per fill from three different versions of
    the picks, and where they disagreed the section grew holes and painted
    ocean inside the glacier.
    """

    surf: np.ndarray        # ice top, NaN off the ice
    base: np.ndarray        # ice bottom, NaN off the ice
    seabed: np.ndarray      # top of solid ground, NaN where unknown
    state: np.ndarray       # _NODATA / _GROUNDED / _FLOATING / _WATER / _NOBASE
    surf_bridged: np.ndarray
    base_bridged: np.ndarray
    projected: np.ndarray   # seabed inferred rather than picked

    @property
    def ice(self):
        return (self.state == _GROUNDED) | (self.state == _FLOATING)

    @property
    def submerged(self):
        """Where the column holds water, i.e. where the water fill may go."""
        return (self.state == _FLOATING) | (self.state == _WATER)

    @property
    def unresolved(self):
        """Surface known, bottom not. Neither ice nor water can be drawn."""
        return self.state == _NOBASE


def _build_section(profile, gp_km, sea_level, bathymetry, floor,
                   project_bathymetry=True, keel_clearance_m=15.0, sign=None):
    """
    Interfaces and a per-sample state for the geometry panel.

    Radar gives the ice bottom, not the seabed, so what lies below the pick is
    set by the detected grounding point: ocean seaward of it, bed landward. The
    seabed is then clamped under the ice above it, which is what stops a
    disagreeing BedMachine bed drawing a line through the middle of the ice.

    `sign` is the local orientation. Taken over a whole segment it can point
    the wrong way, which flips every side-of-the-pick judgement below.
    """
    if sign is None:
        sign = profile.landward_sign()
    surf, surf_bridged = _interp_gaps(profile.x, profile.h_surf, _MAX_BRIDGE_KM)
    bed, base_bridged = _interp_gaps(profile.x, profile.h_bed, _MAX_BRIDGE_KM)
    grounded = ((profile.x >= gp_km) if sign > 0
                else (profile.x <= gp_km))

    # The ice runs to the terminus, but its bottom pick collapses before then,
    # so the bottom is carried out at the depth it last held.
    collapsed = _collapsed_bottom(profile, surf, bed, grounded)
    if collapsed.any():
        bed, held = _hold_front_bottom(profile, bed, collapsed, sea_level,
                                       sign=sign)
        base_bridged = (base_bridged | held) & np.isfinite(bed)

    ice = _ice_mask(surf, bed)
    ahead = _ahead_of_terminus(profile, ice, gp_km, sign=sign)
    ice = ice & ~ahead

    # A stretch with no pick is unknown, not open water. Keeping that its own
    # state is what stops a bed-pick gap being filled in as a fjord, and a
    # surface with no bottom under it being filled in as ice.
    state = np.full(profile.n, _NODATA, dtype=np.int8)
    state[np.isfinite(surf) & ~ahead] = _NOBASE
    state[ahead & np.isfinite(profile.h_surf)] = _WATER
    state[ice & grounded] = _GROUNDED
    state[ice & ~grounded] = _FLOATING

    submerged = (state == _FLOATING) | (state == _WATER)
    seabed = np.where(state == _GROUNDED, bed, np.nan)
    under = _seabed_under_water(profile, bed, gp_km, grounded, ice, floor,
                                bathymetry, project_bathymetry)
    projected = np.zeros(profile.n, dtype=bool)
    if under is not None:
        # Also under the unresolved columns, so the fjord floor stays continuous
        # across a stretch whose ice bottom is missing.
        below = submerged | (state == _NOBASE)
        projected = below & np.isfinite(under)
        # Ordered, always: the seabed cannot rise through the ice above it or
        # break the sea surface.
        ceiling = np.where(ice, bed - keel_clearance_m, sea_level)
        seabed = np.where(projected, np.clip(under, floor, ceiling), seabed)

    # The surface is kept over the unresolved columns too: it is measured there,
    # and it is the top of the band that marks what the bottom pick did not say.
    known_top = ice | (state == _NOBASE)
    return _Section(
        surf=np.where(known_top, surf, np.nan),
        base=np.where(ice, bed, np.nan),
        seabed=seabed,
        state=state,
        surf_bridged=surf_bridged & known_top,
        base_bridged=base_bridged & ice,
        projected=projected,
    )


def _fill_section(ax, x, sec, sea_level, floor):
    """Ice, water and earth polygons in polartoolkit's layer colours."""
    drawn = sec.state != _NODATA
    ax.fill_between(x, floor, sec.seabed, where=np.isfinite(sec.seabed),
                    color=LAYER_C["earth"], lw=0, zorder=2)
    # Seabed to sea level, flat on top: the ice is drawn over it and takes back
    # the part of the column it occupies, which keeps the sea surface level
    # right up against the calving face instead of dipping to the ice bottom.
    # Grown one sample into whatever is drawn beside it so the two polygons
    # share an edge, but never into a gap, which is what used to leak ocean
    # into the middle of the glacier.
    ax.fill_between(x, np.where(np.isfinite(sec.seabed), sec.seabed, floor),
                    sea_level, where=_grow(sec.submerged, drawn),
                    color=LAYER_C["water"], lw=0, zorder=1)
    ax.fill_between(x, sec.base, sec.surf, where=sec.ice,
                    color=LAYER_C["ice"], lw=0, zorder=3)
    # Ice whose bottom was interpolated rather than picked.
    inferred = _grow(sec.base_bridged, sec.ice)
    if inferred.any():
        ax.fill_between(x, sec.base, sec.surf, where=inferred, facecolor="none",
                        edgecolor="white", hatch="///", lw=0, zorder=3.5)
    # Surface measured, bottom not: the column below it is left unresolved
    # rather than guessed at as ice or as ocean.
    if sec.unresolved.any():
        ax.fill_between(x, np.where(np.isfinite(sec.seabed), sec.seabed, floor),
                        sec.surf, where=_grow(sec.unresolved, drawn),
                        facecolor="white", edgecolor=C["muted"], hatch="///",
                        lw=0, alpha=0.9, zorder=3.2)
    if sec.projected.any():
        ax.plot(x, np.where(_grow(sec.projected, np.isfinite(sec.seabed)),
                            sec.seabed, np.nan),
                color="black", lw=1.2, ls="--", zorder=5)
    for lo, hi in _runs_km(x, sec.state == _NODATA):
        ax.axvspan(lo, hi, facecolor="none", edgecolor=C["muted"], lw=0.6,
                   hatch="///", alpha=0.55, zorder=2.6)


def _resolve_bathymetry(profile, spec):
    """
    Seabed elevations on the profile's grid, in its ellipsoidal datum.

    A path is sampled from BedMachine, whose bed sits on the geoid, so the
    geoid separation is added to bring it onto the ellipsoid with the picks.
    """
    if spec is None:
        return None
    if isinstance(spec, (str, Path)):
        if profile.lat is None or profile.lon is None:
            raise ValueError("profile has no coordinates to sample BedMachine "
                             "along; pass an array of elevations instead")
        bed = sample_bedmachine(profile.lat, profile.lon, "bed", path=spec)
        return bed + profile.geoid_separation_m
    values = np.asarray(spec, dtype=float)
    if values.shape != profile.x.shape:
        raise ValueError(f"bathymetry has shape {values.shape}, expected "
                         f"{profile.x.shape}")
    return values


def _seabed_under_water(profile, bed, gp_km, grounded, ice, floor, bathymetry,
                        project_bathymetry):
    """Seabed elevations to use wherever the radar stopped at the ice bottom."""
    if bathymetry is not None:
        return np.asarray(bathymetry, dtype=float)
    if not project_bathymetry:
        return None
    level = _project_seabed(profile, bed, gp_km, grounded, ice, floor)
    return None if level is None else np.full(profile.n, level)


def _project_seabed(profile, bed, gp_km, grounded, ice, floor, window_km=2.0,
                    clearance_m=20.0):
    """
    Flat seabed under the floating ice, at its depth at the grounding point.

    Radar stops at the ice bottom once the ice is afloat, so the seabed there is
    a projection, not a measurement. It is drawn level rather than sloped: any
    slope would come from extrapolating a few km of grounded bed tens of km out,
    which is not evidence of the fjord's shape.

    Where the shelf keel hangs deeper than the bed at the grounding point, the
    level drops below the deepest keel instead of following it, so the line
    stays flat and never draws the seabed through floating ice. The keel alone
    sets it when the grounding point falls in a bed-pick gap, as on Helheim.
    """
    depths = []
    near = grounded & np.isfinite(bed) & (
        np.abs(profile.x - gp_km) <= window_km)
    if near.any():
        depths.append(float(np.median(bed[near])))
    keels = bed[~grounded & ice]
    if keels.size:
        depths.append(float(keels.min()) - clearance_m)
    if not depths:
        return None
    return max(min(depths), floor)


def _depth_bound(profile, bathymetry, vis):
    """
    How deep the panel has to reach, from the picks and the seabed together.

    BedMachine's deepest sample is taken at its 1st percentile rather than its
    minimum: one spurious trough would otherwise set the scale and flatten the
    ice into a strip at the top of the panel.
    """
    bed = profile.h_bed[vis]
    if bathymetry is None:
        return bed
    sea = bathymetry[vis]
    sea = sea[np.isfinite(sea)]
    if not sea.size:
        return bed
    return np.append(bed, np.percentile(sea, 1.0))


def _grow(mask, into):
    """
    Mask extended one sample into `into`, so adjacent fills share an edge.

    Layer boundaries fall between samples, so two fills that stop on their own
    last sample leave a white wedge at every contact. Growing only into what is
    actually drawn beside them keeps the wedge closed without spilling a fill
    into a stretch that has no data.
    """
    out = mask.copy()
    out[:-1] |= mask[1:] & into[:-1]
    out[1:] |= mask[:-1] & into[1:]
    return out


def _runs_km(x, mask, min_width_km=0.0):
    """Contiguous True runs as (lo_km, hi_km) pairs."""
    spans = []
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    for start, stop in zip(edges[::2], edges[1::2]):
        lo, hi = float(x[start]), float(x[stop - 1])
        if hi - lo >= min_width_km:
            spans.append((lo, hi))
    return spans


def _data_extent(profile, lo, hi, pad_km):
    """
    x limits: the valid data, from its upflow end to its downflow end.

    Anything past the first and last sample carrying a pick is blank paper, and
    Helheim's cropped leg would otherwise sit in 40 km of it. pad_km still caps
    how far either side of the search window the view opens out, so a long
    transect stays zoomed on the transition instead of showing the whole flight.
    """
    valid = (np.isfinite(profile.h_surf) | np.isfinite(profile.h_bed)
             | np.isfinite(profile.amp))
    if not valid.any():
        return lo - pad_km, hi + pad_km
    x_valid = profile.x[valid]
    x0 = max(float(x_valid[0]), lo - pad_km)
    x1 = min(float(x_valid[-1]), hi + pad_km)
    return (x0, x1) if x1 > x0 else (lo - pad_km, hi + pad_km)


def _interp_gaps(x, y, max_gap_km=None):
    """
    Layer bridged linearly across interior gaps, plus a mask of what was made up.

    Only gaps between picks are filled, and only ones narrower than max_gap_km.
    Beyond the first and last pick there is nothing to interpolate between, and
    a straight run across tens of km is invention rather than interpolation.
    """
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return y, np.zeros(len(y), dtype=bool)
    filled = np.interp(x, x[ok], y[ok])
    inside = (x >= x[ok][0]) & (x <= x[ok][-1])
    bridged = inside & ~ok
    if max_gap_km is not None:
        for lo, hi in _runs_km(x, bridged):
            # One sample either side, the picks the bridge would span.
            if (hi - lo) + 2 * float(np.median(np.diff(x))) > max_gap_km:
                too_wide = (x >= lo) & (x <= hi)
                filled[too_wide] = np.nan
                bridged[too_wide] = False
    return np.where(inside, filled, np.nan), bridged


def _ice_mask(surf, bed):
    """
    True where the bridged picks show real ice.

    GlacierProfile.ice_mask works off the raw picks, which the fills and the
    layer lines no longer use once the gaps are bridged.
    """
    thickness = surf - bed
    return np.isfinite(thickness) & (thickness >= _MIN_THICKNESS_M)


def _collapsed_bottom(profile, surf, bed, grounded, tolerance=0.5):
    """
    Floating-side bottom picks too shallow to be an ice base.

    Nearing the calving front the bottom pick climbs to meet the surface, and it
    is well on its way up before it breaks the sea surface: Helheim's last two
    picks read -60 m under 85 m of freeboard, which would be a keel a tenth of
    the depth flotation demands. A floating column's draft follows from its
    freeboard, so anything shallower than tolerance of that is the picker losing
    the bottom rather than ice thinning.
    """
    freeboard = surf - profile.geoid_separation_m
    draft = freeboard * (1.0 - 1.0 / profile.constants.flotation_factor)
    return (~grounded & np.isfinite(bed) & (freeboard > 0)
            & (bed - profile.geoid_separation_m > tolerance * draft))


def _hold_front_bottom(profile, bed, collapsed, sea_level, min_draft_m=20.0,
                       sign=None):
    """
    Ice bottom carried out to the terminus at the depth it last held.

    The pick does not fail all at once: it shoals for a few samples before it
    breaks the surface, so the last pick standing is already too shallow.
    Helheim's is -469 m where the ice either side is -571 m, and holding it
    would draw 100 m of ice as ocean. This walks back to where the pick stopped
    shoaling and holds that depth out to the front.

    An anchor that is itself above the waterline is not held: on 20060608_01 the
    pick sits on the surface across the whole shelf, and holding it drew 8 km of
    ice with its base 80 m above sea level and ocean underneath. That bottom is
    left unknown instead.

    Returns the bottom and a mask of the samples it made up.
    """
    if sign is None:
        sign = profile.landward_sign()
    seaward = slice(None) if sign < 0 else slice(None, None, -1)
    b = bed[seaward].copy()
    anchor = int(np.argmax(collapsed[seaward]))
    while anchor > 0 and np.isfinite(b[anchor - 1]) and b[anchor - 1] < b[anchor]:
        anchor -= 1

    held = np.zeros(len(b), dtype=bool)
    if np.isfinite(b[anchor]) and b[anchor] < sea_level - min_draft_m:
        held[anchor + 1:] = True
        b[anchor + 1:] = b[anchor]
    else:
        b[anchor:] = np.nan
    return b[seaward], held[seaward]


def _ahead_of_terminus(profile, ice, gp_km, sign=None):
    """
    Samples in front of the glacier, seaward of its calving face.

    The face is the terminus the profile finds seaward of the grounding point.
    It cannot be read off the ice mask alone: in front of the front the picks
    wander around the sea surface and throw up metres of "ice" over open water,
    which would put the calving face out among the bergs. Falls back to the
    seaward end of the ice the grounding point sits in, for a transect that
    never reaches open water.
    """
    if sign is None:
        sign = profile.landward_sign()
    front = profile.calving_front_km(gp_km, sign=sign)
    if front is None:
        if not ice.any():
            return np.zeros(len(profile.x), dtype=bool)
        edges = np.flatnonzero(np.diff(np.r_[False, ice, False].astype(np.int8)))
        gp_i = int(np.argmin(np.abs(profile.x - gp_km)))
        start, stop = min(zip(edges[::2], edges[1::2]),
                          key=lambda run: 0 if run[0] <= gp_i < run[1]
                          else min(abs(run[0] - gp_i), abs(run[1] - 1 - gp_i)))
        front = profile.x[start] if sign > 0 else profile.x[stop - 1]
    return (profile.x < front) if sign > 0 else (profile.x > front)


def _dilate(mask):
    """Mask grown by one sample either side."""
    out = mask.copy()
    out[:-1] |= mask[1:]
    out[1:] |= mask[:-1]
    return out


def _surf_at(profile, sec, x_km):
    """
    Ice surface at a position, for placing a marker on it.

    Off the interpolated surface it falls back to the raw pick, so a marker
    landing in a gap still sits on the section rather than vanishing.
    """
    for values in (sec.surf, profile.h_surf):
        ok = np.isfinite(values)
        if ok.any():
            return float(np.interp(x_km, profile.x[ok], values[ok]))
    return np.nan


def _tight_ylim(ax, lower, upper, pad_frac=0.12):
    lo = float(np.nanmin(lower))
    hi = float(np.nanmax(upper))
    pad = pad_frac * (hi - lo)
    ax.set_ylim(lo - pad, hi + 2.5 * pad)
