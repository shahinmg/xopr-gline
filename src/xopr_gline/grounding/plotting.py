"""
Figures for grounding point detection.

Takes a profile and the results detected on it; knows nothing about how they
were produced. Styled after polartoolkit's profile cross sections: filled
ice/water/earth layers with black interfaces, GMT frames, legends outside the
axes, and A/B labels on the section ends.
"""

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
                   bathymetry=None):
    """
    Four panels: geometry, the flotation residual that set the window, bed
    power, and the posterior.

    bathymetry gives the seabed under the floating ice, where radar stops at the
    ice bottom: a path to BedMachine to sample 'bed' along the flight line, or
    an array of elevations on the profile's grid. Without it the seabed is drawn
    flat at its depth at the grounding point. Either way the radar picks are
    untouched; this only sets the projected bed line.
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

        fig, axs = plt.subplots(4, 1, figsize=(13, 14), sharex=True,
                                gridspec_kw=dict(
                                    height_ratios=[2.1, 1.3, 1.3, 1.7],
                                    hspace=0.09))
        for ax in axs:
            _gmt_frame(ax)
            ax.axvspan(lo, hi, color=C["window"], alpha=0.09, zorder=4)
            if gz:
                ax.axvspan(gz[0], gz[1], color=C["gz"], alpha=0.22, zorder=4)

        # Calving front from the data, where thickness collapses. Absent on a
        # transect that never reaches open water, e.g. Petermann 7:11.
        termini = [t for t in profile.terminus_crossings_km() if x0 <= t <= x1]
        for ax in axs:
            for terminus in termini:
                ax.axvline(terminus, color=C["terminus"], lw=1.6, ls="-.",
                           alpha=0.9, zorder=5)

        vis = (profile.x >= x0) & (profile.x <= x1)
        sea_level = _sea_level(profile)

        # -- geometry -----------------------------------------------------
        ax = axs[0]
        # The seabed sets the floor along with the picks: BedMachine's trough
        # runs 200 m below Helheim's deepest bed pick and would be clipped off.
        bathy = _resolve_bathymetry(profile, bathymetry)
        floor = (profile.h_bed[vis] if bathy is None
                 else np.concatenate([profile.h_bed[vis], bathy[vis]]))
        _tight_ylim(ax, floor, profile.h_surf[vis])
        # Gaps in the layer picks are bridged so the section reads continuously,
        # and the bridges are dashed: interpolated, not measured.
        surf, surf_gap = _interp_gaps(profile.x, profile.h_surf)
        bed, bed_gap = _interp_gaps(profile.x, profile.h_bed)
        gap = surf_gap | bed_gap
        gp = onset.map_km if onset is not None else primary.map_km
        grounded = ((profile.x >= gp) if profile.landward_sign() > 0
                    else (profile.x <= gp))
        # The ice runs to the terminus, but its bottom pick collapses before
        # then, so the bottom is carried out at the depth it last held.
        collapsed = _collapsed_bottom(profile, surf, bed, grounded)
        patched = np.zeros(profile.n, dtype=bool)
        if collapsed.any():
            bed, patched = _hold_front_bottom(profile, bed, collapsed)
            gap = gap | patched
        ahead = _ahead_of_terminus(profile, _ice_mask(surf, bed), gp)
        ice = _ice_mask(surf, bed) & ~ahead
        handles = []
        if fill_layers:
            # The ice stops at the calving face, so the fill follows the raw
            # pick up it. The held bottom stays the seabed's business.
            projected = _fill_section(ax, profile, surf, bed,
                                      np.where(patched, profile.h_bed, bed),
                                      ice, ahead, grounded, sea_level, gp,
                                      project_bathymetry, bathy)
            handles += [Patch(fc=LAYER_C["ice"], ec="black", lw=0.8, label="Ice"),
                        Patch(fc=LAYER_C["water"], ec="black", lw=0.8,
                              label="Water"),
                        Patch(fc=LAYER_C["earth"], ec="black", lw=0.8,
                              label="Bed")]
            if projected is not None:
                source = "BedMachine" if bathymetry is not None else "projected"
                handles.append(Line2D([], [], color="black", lw=1.2, ls="--",
                                      label=f"Bed ({source})"))
            surf_c = bed_c = "black"
        else:
            surf_c, bed_c = C["surf"], C["bed"]
            handles += [Line2D([], [], color=surf_c, lw=1.6, label="Ice surface"),
                        Line2D([], [], color=bed_c, lw=1.6, label="Bed")]
        # Only where there is glacier ice. Past the calving front the layer
        # picks collapse onto the sea surface, and drawing them there would
        # trace the water rather than the ice this figure is about.
        ax.plot(profile.x, np.where(ice, profile.h_surf, np.nan), color=surf_c,
                lw=1.2, zorder=5)
        ax.plot(profile.x, np.where(ice & ~patched, profile.h_bed, np.nan),
                color=bed_c, lw=1.2, zorder=5)
        if (gap & ice).any():
            ax.plot(profile.x, np.where(ice, _bridge(surf, surf_gap), np.nan),
                    color=surf_c, lw=1.2, ls=(0, (4, 2)), zorder=5)
            ax.plot(profile.x, np.where(ice, _bridge(bed, bed_gap), np.nan),
                    color=bed_c, lw=1.2, ls=(0, (4, 2)), zorder=5)
            # The calving face: the bottom pick climbing to meet the surface. It
            # is where the picks stop describing an ice bottom, so it is drawn
            # like the rest of what was not measured.
            ax.plot(profile.x,
                    np.where(_dilate(patched) & ice, profile.h_bed, np.nan),
                    color=bed_c, lw=1.2, ls=(0, (4, 2)), zorder=5)
            handles.append(Line2D([], [], color=surf_c, lw=1.2, ls=(0, (4, 2)),
                                  label="Interpolated across gap"))
        ax.axhline(sea_level, color="black", lw=0.7, ls="--", alpha=0.8,
                   zorder=5)
        ax.axvline(primary.map_km, color=C["map"], lw=1.8, ls="--", alpha=0.9,
                   zorder=6)
        if onset is not None:
            ax.axvline(onset.map_km, color=C["onset"], lw=2.2, alpha=0.95,
                       zorder=6)
            ax.scatter([onset.map_km],
                       [float(np.interp(onset.map_km, profile.x, profile.h_surf))],
                       s=180, color=C["onset"], edgecolors="white", lw=1.8,
                       zorder=7)
            handles.append(Line2D([], [], color=C["onset"], marker="o", lw=2.2,
                                  mec="white", mew=1.4, ms=10,
                                  label=f"grounding point (onset) "
                                        f"{onset.map_km:.2f} km"))
        ax.scatter([primary.map_km],
                   [float(np.interp(primary.map_km, profile.x, profile.h_surf))],
                   s=120, color=C["map"], marker="s", edgecolors="white", lw=1.6,
                   zorder=7)
        handles.append(Line2D([], [], color=C["map"], marker="s", lw=1.8,
                              ls="--", mec="white", mew=1.2, ms=8,
                              label=f"changepoint {primary.map_km:.2f} km"))
        if termini:
            handles.append(Line2D([], [], color=C["terminus"], lw=1.6, ls="-.",
                                  label=f"terminus {termini[0]:.2f} km"))
        handles.append(Line2D([], [], color="black", lw=0.7, ls="--",
                              label="sea level"))
        ax.set_ylabel("Elevation (m WGS84)")
        _gmt_legend(ax, handles=handles)
        ax.set_title(title or f"{profile.source}", fontsize=12, color="black",
                     pad=16, fontweight="semibold")
        _start_end_labels(ax)

        # -- flotation residual -------------------------------------------
        ax = axs[1]
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
        ax = axs[2]
        ax.plot(profile.x, profile.amp, color=C["muted"], lw=0.9, alpha=0.6,
                label="Bed power")
        ax.plot(profile.x, _features.AmplitudeLevel().compute(profile),
                color=C["amp"], lw=1.9, label="Filtered")
        for gap_lo, gap_hi in profile.nan_blocks():
            ax.axvspan(gap_lo, gap_hi, color="gray", alpha=0.16, zorder=0)
        if onset is not None:
            ax.axhline(onset.extra["baseline_dB"], color=C["onset"], lw=1.1,
                       ls=":",
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
                        arrowprops=dict(arrowstyle="<->", color="black", lw=1.3))
            ax.text(0.5 * (onset.map_km + primary.map_km),
                    y_arrow + 0.03 * (y1 - y0),
                    f"transition {primary.map_km - onset.map_km:.1f} km",
                    ha="center", fontsize=8.5, color="black")
        ax.set_ylabel("Bed power (dB)")
        _gmt_legend(ax)

        # -- posterior ------------------------------------------------------
        ax = axs[3]
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

        if gz:
            axs[0].text(np.mean(gz), axs[0].get_ylim()[1], " InSAR GZ",
                        ha="center", va="top", fontsize=9, color=C["gz"],
                        fontweight="semibold", zorder=8)

        fig.savefig(out_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return out_path


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


def _fill_section(ax, profile, surf, bed, ice_bottom, ice, ahead, grounded,
                  sea_level, gp_km, project_bathymetry=True, bathymetry=None):
    """
    Ice, water and earth polygons in polartoolkit's layer colours.

    Radar gives the ice bottom, not the seabed, so what lies below the pick is
    coloured by the detected grounding point: ocean seaward of it, bed landward.

    Gaps in the picks are filled from the bridged surface and bed, so a short
    dropout does not slice the section into blocks. A dropout is not water,
    though: only what lies in front of the calving face is filled as ocean, so a
    stretch with no pick stays blank rather than becoming a fjord.
    """
    y0 = ax.get_ylim()[0]
    seabed = np.where(grounded, bed, np.nan)
    projected = None
    if bathymetry is not None:
        projected = np.where(grounded | ~np.isfinite(bathymetry), np.nan,
                             np.maximum(bathymetry, y0))
    elif project_bathymetry:
        projected = _project_seabed(profile, bed, gp_km, grounded, ice, y0)
    if projected is not None:
        seabed = np.where(np.isfinite(projected), projected,
                          np.where(grounded, bed, np.nan))

    ax.fill_between(profile.x, y0, seabed, where=np.isfinite(seabed),
                    color=LAYER_C["earth"], lw=0, zorder=2)
    # Seabed to sea level, flat on top wherever there is water at all: the ice
    # is drawn over it and takes back the part of the column it occupies, which
    # keeps the sea surface level right up against the calving face instead of
    # dipping to meet the ice bottom. One sample wider than the water itself,
    # because the seabed steps down from the measured bed to the projected floor
    # between two samples and that step is otherwise left as a white wedge.
    ax.fill_between(profile.x, np.where(np.isfinite(seabed), seabed, y0),
                    sea_level,
                    where=_dilate((~grounded & ice) | ahead),
                    color=LAYER_C["water"], lw=0, zorder=1)
    ax.fill_between(profile.x, ice_bottom, surf, where=ice,
                    color=LAYER_C["ice"], lw=0, zorder=3)
    if projected is not None:
        ax.plot(np.where(grounded, np.nan, profile.x), projected, color="black",
                lw=1.2, ls="--", zorder=5)
    return projected


def _resolve_bathymetry(profile, spec):
    """
    Seabed elevations on the profile's grid, in its ellipsoidal datum.

    A path is read as BedMachine and sampled along the flight line. Its bed sits
    on the geoid, so the profile's geoid separation is added to bring it onto
    the ellipsoid with the radar picks, which are left alone.
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
    return np.where(grounded, np.nan, max(min(depths), floor))


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


def _interp_gaps(x, y):
    """
    Layer bridged linearly across interior gaps, plus a mask of what was made up.

    Only gaps between picks are filled. Beyond the first and last pick there is
    nothing to interpolate between, and a straight run off the end of the data
    would be invention rather than interpolation.
    """
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return y, np.zeros(len(y), dtype=bool)
    filled = np.interp(x, x[ok], y[ok])
    inside = (x >= x[ok][0]) & (x <= x[ok][-1])
    return np.where(inside, filled, np.nan), inside & ~ok


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


def _hold_front_bottom(profile, bed, collapsed):
    """
    Ice bottom carried out to the terminus at the depth it last held.

    The pick does not fail all at once: it shoals for a few samples before it
    breaks the surface, so the last pick standing is already too shallow.
    Helheim's is -469 m where the ice either side is -571 m, and holding it
    would draw 100 m of ice as ocean. This walks back to where the pick stopped
    shoaling and holds that depth out to the front.

    Returns the bottom and a mask of the samples it made up.
    """
    seaward = slice(None) if profile.landward_sign() < 0 else slice(None, None, -1)
    b = bed[seaward].copy()
    anchor = int(np.argmax(collapsed[seaward]))
    while anchor > 0 and np.isfinite(b[anchor - 1]) and b[anchor - 1] < b[anchor]:
        anchor -= 1
    if not np.isfinite(b[anchor]):
        return bed, np.zeros(len(bed), dtype=bool)

    patched = np.zeros(len(b), dtype=bool)
    patched[anchor + 1:] = True
    b[anchor + 1:] = b[anchor]
    return b[seaward], patched[seaward]


def _ahead_of_terminus(profile, ice, gp_km):
    """
    Samples in front of the glacier, seaward of its calving face.

    The face is the terminus the profile finds seaward of the grounding point.
    It cannot be read off the ice mask alone: in front of the front the picks
    wander around the sea surface and throw up metres of "ice" over open water,
    which would put the calving face out among the bergs. Falls back to the
    seaward end of the ice the grounding point sits in, for a transect that
    never reaches open water.
    """
    sign = profile.landward_sign()
    seaward = [t for t in profile.terminus_crossings_km()
               if (t > gp_km if sign < 0 else t < gp_km)]
    if seaward:
        front = min(seaward) if sign < 0 else max(seaward)
    elif ice.any():
        edges = np.flatnonzero(np.diff(np.r_[False, ice, False].astype(np.int8)))
        gp_i = int(np.argmin(np.abs(profile.x - gp_km)))
        start, stop = min(zip(edges[::2], edges[1::2]),
                          key=lambda run: 0 if run[0] <= gp_i < run[1]
                          else min(abs(run[0] - gp_i), abs(run[1] - 1 - gp_i)))
        front = profile.x[start] if sign > 0 else profile.x[stop - 1]
    else:
        return np.zeros(len(profile.x), dtype=bool)
    return (profile.x < front) if sign > 0 else (profile.x > front)


def _dilate(mask):
    """Mask grown by one sample either side."""
    out = mask.copy()
    out[:-1] |= mask[1:]
    out[1:] |= mask[:-1]
    return out


def _bridge(y, gap):
    """The bridged samples only, joined to the measured pick either side."""
    return np.where(_dilate(gap), y, np.nan)


def _tight_ylim(ax, lower, upper, pad_frac=0.12):
    lo = float(np.nanmin(lower))
    hi = float(np.nanmax(upper))
    pad = pad_frac * (hi - lo)
    ax.set_ylim(lo - pad, hi + 2.5 * pad)
