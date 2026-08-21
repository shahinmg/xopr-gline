"""
A glacier centreline profile on a uniform along-track grid.

Detection methods should take a GlacierProfile rather than raw xOPR datasets or
CSVs, so the same code runs on any segment. Bed power and layer elevations
arrive from xOPR on different slow_time grids; the constructors join them and
resample onto one uniform grid so downstream code never interpolates again.
"""

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter1d

import xopr
import xopr.opr_access
import xopr.radar_util

from ..constants import DEFAULT_CONSTANTS, PhysicalConstants
from ..xopr_utils import surface_bed_reflection_power
from .geoid import resolve_geoid


@dataclass(frozen=True)
class FrameSpan:
    """One frame's identity and the along-track km it covers on this profile."""

    granule: Optional[str] = None
    frame: Optional[int] = None
    lo_km: float = np.nan
    hi_km: float = np.nan
    start_time: Optional[pd.Timestamp] = None
    end_time: Optional[pd.Timestamp] = None

    def contains(self, x_km: float) -> bool:
        return bool(self.lo_km <= x_km <= self.hi_km)

    def __str__(self) -> str:
        return f"{self.granule} [{self.lo_km:.2f}-{self.hi_km:.2f} km]"


@dataclass(frozen=True)
class ProfileSource:
    """
    Where a profile came from, so a result can be traced back.

    `frames` carries each frame's along-track extent on the same axis as
    GlacierProfile.x, which is what turns a detected km into a named frame.
    """

    collection: Optional[str] = None
    segment: Optional[str] = None
    frames: tuple = ()
    xopr_version: str = field(default_factory=lambda: xopr.__version__)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def date(self) -> Optional[pd.Timestamp]:
        """Acquisition start of the earliest frame, or None if untimed."""
        times = [f.start_time for f in self.frames if f.start_time is not None]
        return min(times) if times else None

    def frame_at_km(self, x_km: float) -> Optional[FrameSpan]:
        """The frame covering this along-track position, or None if outside."""
        for span in self.frames:
            if span.contains(x_km):
                return span
        return None

    def frames_between(self, lo_km: float, hi_km: float) -> tuple:
        """Frames overlapping [lo_km, hi_km], i.e. the frames a detector saw."""
        return tuple(s for s in self.frames
                     if s.hi_km >= lo_km and s.lo_km <= hi_km)

    def __str__(self) -> str:
        return f"{self.collection}/{self.segment} ({self.n_frames} frames)"


@dataclass
class GlacierProfile:
    """
    Along-track profile with bed power and surface/bed elevations on one grid.

    Parameters
    ----------
    x: np.ndarray
        Along-track distance in km, uniformly spaced and increasing.

    amp: np.ndarray
        Bed echo power in dB. NaN where no bed pick exists.

    h_surf, h_bed: np.ndarray
        Ice surface and bed elevation in m WGS84. NaN across stretches with no
        layer pick, rather than a straight line drawn over the gap.

    constants: PhysicalConstants
        Densities used by the flotation properties.

    geoid_separation_m: float or np.ndarray
        Geoid height above the WGS84 ellipsoid, scalar or per-sample. xOPR
        returns ellipsoidal elevations but flotation is referenced to sea level,
        so this is subtracted before any flotation calculation. It varies along
        track (13-21 m across Petermann), so prefer sampling it from BedMachine.
        Leaving it at 0 biases the flotation residual by this amount.

    lat, lon: np.ndarray or None
        Point coordinates, carried so the geoid can be sampled and results
        georeferenced.

    source: ProfileSource or None
        Provenance, set by from_xopr.
    """

    x: np.ndarray
    amp: np.ndarray
    h_surf: np.ndarray
    h_bed: np.ndarray
    constants: PhysicalConstants = DEFAULT_CONSTANTS
    geoid_separation_m: Union[float, np.ndarray] = 0.0
    lat: Optional[np.ndarray] = None
    lon: Optional[np.ndarray] = None
    source: Optional[ProfileSource] = None

    def __post_init__(self):
        n = len(self.x)
        for name in ("amp", "h_surf", "h_bed"):
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"{name} has length {len(getattr(self, name))}, expected {n}"
                )
        if n < 2:
            raise ValueError("profile needs at least 2 samples")
        if not np.all(np.diff(self.x) > 0):
            raise ValueError("x must be strictly increasing")
        for name in ("lat", "lon"):
            v = getattr(self, name)
            if v is not None and len(v) != n:
                raise ValueError(f"{name} has length {len(v)}, expected {n}")
        g = self.geoid_separation_m
        if np.ndim(g) and len(g) != n:
            raise ValueError(
                f"geoid_separation_m has length {len(g)}, expected {n}"
            )

    # -- geometry ---------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.x)

    @property
    def dx(self) -> float:
        """Sample spacing in km."""
        return float(np.median(np.diff(self.x)))

    @property
    def extent(self) -> tuple:
        return float(self.x[0]), float(self.x[-1])

    @property
    def nan_mask(self) -> np.ndarray:
        """True where the bed power has no pick."""
        return np.isnan(self.amp)

    def position_at(self, x_km: float) -> tuple:
        """
        (lat, lon) at an along-track position, NaN if the profile has no track.

        Interpolated on the profile grid, so a detected km becomes a point that
        can go straight into a geodataframe.
        """
        if self.lat is None or self.lon is None:
            return (np.nan, np.nan)
        return (float(np.interp(x_km, self.x, self.lat)),
                float(np.interp(x_km, self.x, self.lon)))

    def cutoff_to_wn(self, wavelength_km: float) -> float:
        """
        Convert a physical cutoff wavelength to a normalised Butterworth Wn.

        Filter cutoffs must be specified in km, not in Wn, or the same setting
        filters a different physical scale on every flight.
        """
        wn = 2.0 * self.dx / wavelength_km
        if not 0.0 < wn < 1.0:
            raise ValueError(
                f"wavelength {wavelength_km} km gives Wn={wn:.3f}; must be in "
                f"(0, 1). Sample spacing is {self.dx:.4f} km, so the cutoff "
                f"must exceed {2 * self.dx:.4f} km."
            )
        return wn

    # -- flotation --------------------------------------------------------
    @property
    def h_surf_msl(self) -> np.ndarray:
        """Surface elevation above sea level (geoid), in m."""
        return self.h_surf - self.geoid_separation_m

    @property
    def h_bed_msl(self) -> np.ndarray:
        """Bed elevation above sea level (geoid), in m."""
        return self.h_bed - self.geoid_separation_m

    @property
    def thickness(self) -> np.ndarray:
        """Ice thickness in m. Unaffected by the geoid offset."""
        return self.h_surf - self.h_bed

    @property
    def flotation_residual(self) -> np.ndarray:
        """
        h_surf_msl - H * (1 - rho_ice/rho_sw), in m.

        Positive where the surface sits above hydrostatic flotation (grounded),
        negative where it sits below (floating). Referenced to sea level, so an
        uncorrected geoid separation offsets this directly.
        """
        return self.h_surf_msl - self.thickness * self.constants.flotation_factor

    @property
    def height_above_buoyancy(self) -> np.ndarray:
        """
        H - (rho_sw/rho_ice) * water_depth, in m.

        The same criterion as flotation_residual in a different scaling; both
        cross zero at flotation. Matches empirical.hab.
        """
        water_depth = -self.h_bed_msl
        return self.thickness - self.constants.density_ratio * water_depth

    # -- subsetting -------------------------------------------------------
    def window(self, lo_km: float, hi_km: float) -> "GlacierProfile":
        """Return the profile restricted to [lo_km, hi_km]."""
        sel = (self.x >= lo_km) & (self.x <= hi_km)
        if sel.sum() < 2:
            raise ValueError(f"window [{lo_km}, {hi_km}] km selects < 2 samples")
        return GlacierProfile(
            x=self.x[sel],
            amp=self.amp[sel],
            h_surf=self.h_surf[sel],
            h_bed=self.h_bed[sel],
            constants=self.constants,
            geoid_separation_m=(self.geoid_separation_m[sel]
                                if np.ndim(self.geoid_separation_m)
                                else self.geoid_separation_m),
            lat=None if self.lat is None else self.lat[sel],
            lon=None if self.lon is None else self.lon[sel],
            source=self.source,
        )

    def smoothed_residual(self, smooth_km: float = 5.0) -> np.ndarray:
        """Flotation residual with short-wavelength noise removed."""
        filled = pd.Series(self.flotation_residual).interpolate(
            limit_direction="both"
        ).values
        return uniform_filter1d(filled, size=max(3, int(round(smooth_km / self.dx))))

    def landward_sign(self, smooth_km: float = 5.0) -> int:
        """
        +1 if x increases landward, -1 if it increases seaward.

        Grounded ice sits further above flotation than floating ice, so the
        sign of the residual's trend along x gives the orientation. Petermann
        frames 7:11 run landward (+1); Helheim frames 14:16 run seaward (-1).
        """
        residual = self.smoothed_residual(smooth_km)
        ok = np.isfinite(residual)
        if ok.sum() < 2:
            raise ValueError("flotation residual is undefined on this profile")
        slope = np.polyfit(self.x[ok], residual[ok], 1)[0]
        return 1 if slope >= 0 else -1

    def floating_mask(self, threshold_m: float = 10.0,
                      smooth_km: float = 5.0) -> np.ndarray:
        """
        True where the ice is close enough to hydrostatic balance to be treated
        as afloat. Used to pick a reference stretch of shelf per glacier instead
        of hardcoding along-track bounds.
        """
        return self.smoothed_residual(smooth_km) < threshold_m

    def flotation_crossing_km(self, threshold_m: float = 30.0,
                              smooth_km: float = 5.0) -> float:
        """
        Where the smoothed flotation residual first rises past threshold_m
        going landward, in km.

        This is the geometry's own estimate of where grounding begins, derived
        from elevation alone and independent of anything a detector does with
        the bed echo. Used both to place the search window and to sanity-check
        a detector's answer against it.
        """
        sign = self.landward_sign(smooth_km)
        grounded = self.smoothed_residual(smooth_km) > threshold_m
        if sign < 0:
            # Scan landward, which is decreasing x on a seaward-running profile.
            grounded = grounded[::-1]
        if not grounded.any():
            raise ValueError(
                f"residual never exceeds {threshold_m} m; the transect may be "
                f"entirely afloat. Pass an explicit search window."
            )
        if grounded[0]:
            raise ValueError(
                "profile starts already grounded, so the seaward transition is "
                "not in range. Pass an explicit search window."
            )

        idx = int(np.argmax(grounded))
        if sign < 0:
            idx = len(grounded) - 1 - idx
        return float(self.x[idx])

    def flotation_window(self, margin_km: float = 12.0,
                         threshold_m: float = 30.0,
                         smooth_km: float = 5.0,
                         min_thickness_m: float = 25.0) -> tuple:
        """
        Search window derived from where the ice becomes grounded.

        Replaces bounding the search by an InSAR grounding zone, which requires
        already knowing the answer.

        Floating ice sits near zero residual, not below it, so a sign change is
        noise rather than signal. This uses the first sustained rise above
        threshold_m on the smoothed residual, matching the >30 m "grounded"
        classification the CSV-era script used for plotting.

        The seaward edge is cut at the terminus. Past the front the residual
        drops back under the threshold, so an unclamped window would hand the
        detectors open water to find a changepoint in.
        """
        sign = self.landward_sign(smooth_km)
        x_cross = self.flotation_crossing_km(threshold_m, smooth_km)
        lo = max(float(self.x[0]), x_cross - margin_km)
        hi = min(float(self.x[-1]), x_cross + margin_km)

        # The seaward edge snaps to the terminus rather than being capped by it:
        # the grounding point must lie between the crossing and the front. Only
        # termini within reach count, so a long shelf cannot stretch the window.
        reach = 2.0 * margin_km
        for terminus in self.terminus_crossings_km(min_thickness_m):
            if abs(terminus - x_cross) > reach:
                continue
            if sign > 0 and terminus <= x_cross:
                lo = terminus
            elif sign < 0 and terminus >= x_cross:
                hi = terminus

        if not hi > lo:
            raise ValueError(
                f"window collapses at the terminus: the flotation crossing at "
                f"{x_cross:.1f} km sits at the calving front, leaving no "
                f"grounded ice to search. Pass an explicit search window."
            )
        return lo, hi

    # -- terminus ---------------------------------------------------------
    @property
    def degenerate_pick(self) -> np.ndarray:
        """
        True where the bottom pick sits exactly on the surface pick.

        That is the picker finding no bottom return and defaulting to the
        surface, not zero-thickness ice. Open water also reads near zero but
        gives scattered non-zero thicknesses and a much brighter bed echo.
        """
        return self.h_surf == self.h_bed

    def ice_mask(self, min_thickness_m: float = 25.0) -> np.ndarray:
        """
        True where there is real ice, i.e. a measured thickness above the floor.

        Past the calving front the surface and bottom picks collapse onto the
        same reflector, so thickness falls to a few metres or goes negative.
        Those samples are open water, not thin ice, and must be kept out of any
        flotation reasoning: with H ~ 0 the flotation residual reduces to the
        water surface height, which reads as "near flotation" for free.
        """
        return np.isfinite(self.thickness) & (self.thickness >= min_thickness_m)

    def open_water(self, min_thickness_m: float = 25.0,
                   min_run_km: float = 1.0) -> np.ndarray:
        """
        True where the profile is over water rather than ice.

        Sub-threshold thickness is water unless the picker merely defaulted the
        bottom onto the surface. That defaulting does count as water once it
        persists past min_run_km, which no picker glitch does: Petermann opens
        with 19.5 km of it at sea level with no bed return at all.
        """
        thin = np.isfinite(self.thickness) & ~self.ice_mask(min_thickness_m)
        degenerate = self.degenerate_pick
        sustained = np.zeros(self.n, dtype=bool)
        for start, stop in _mask_runs(degenerate):
            if self.x[stop] - self.x[start] >= min_run_km:
                sustained[start:stop + 1] = True
        return thin & (~degenerate | sustained)

    def terminus_crossings_km(self, min_thickness_m: float = 25.0,
                              min_open_km: float = 1.0) -> list:
        """
        Along-track distances where ice meets open water.

        A crossing needs measured water, not missing data, on one side: NaN
        thickness is unknown rather than open, so a bed-pick gap is not a
        terminus. Returns the ice-side x of each boundary.
        """
        ice = self.ice_mask(min_thickness_m)
        water = self.open_water(min_thickness_m, min_open_km)

        crossings = []
        for start, stop in _mask_runs(water):
            if self.x[stop] - self.x[start] < min_open_km:
                continue
            if start > 0 and ice[start - 1]:
                crossings.append(float(self.x[start - 1]))
            if stop < self.n - 1 and ice[stop + 1]:
                crossings.append(float(self.x[stop + 1]))
        return sorted(crossings)

    def nan_blocks(self, min_width_km: float = 0.3) -> list:
        """Contiguous runs of missing bed power wider than min_width_km."""
        mask = self.nan_mask
        edges = np.diff(mask.astype(int))
        starts = list(np.where(edges == 1)[0] + 1)
        ends = list(np.where(edges == -1)[0] + 1)
        if mask[0]:
            starts = [0] + starts
        if mask[-1]:
            ends = ends + [len(mask) - 1]
        blocks = []
        for s, e in zip(starts, ends):
            e = min(e, len(self.x) - 1)
            if self.x[e] - self.x[s] > min_width_km:
                blocks.append((float(self.x[s]), float(self.x[e])))
        return blocks

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_arrays(cls, x_km, amp, h_surf, h_bed, constants=DEFAULT_CONSTANTS,
                    geoid_separation_m=0.0, source=None) -> "GlacierProfile":
        """Build directly from arrays that already share one grid."""
        return cls(
            x=np.asarray(x_km, dtype=float),
            amp=np.asarray(amp, dtype=float),
            h_surf=np.asarray(h_surf, dtype=float),
            h_bed=np.asarray(h_bed, dtype=float),
            constants=constants,
            geoid_separation_m=geoid_separation_m,
            source=source,
        )

    @classmethod
    def from_xopr(cls, collection: str, segment: str,
                  opr: Optional[xopr.opr_access.OPRConnection] = None,
                  frame_slice: Optional[slice] = None,
                  frames: Optional[Sequence[int]] = None,
                  dx_km: Optional[float] = None,
                  resample_interval: str = "2s",
                  geoid: Union[float, str, None] = None,
                  constants: PhysicalConstants = DEFAULT_CONSTANTS,
                  ) -> "GlacierProfile":
        """
        Build a profile for one segment straight from xOPR.

        Parameters
        ----------
        collection, segment: str
            e.g. "2010_Greenland_DC8" and "20100420_03".

        opr: OPRConnection or None
            Reused if given, otherwise a default connection is created.

        frame_slice: slice or None
            Positional subset of the segment's frames, after sorting by frame
            number. None uses the whole segment.

        frames: sequence of int or None
            Frame numbers to keep, e.g. [4, 5]. Takes precedence over
            frame_slice and is the safe choice when the numbers are known.

        dx_km: float or None
            Grid spacing. None uses the median bed-power sample spacing, which
            is the coarser of the two inputs.

        resample_interval: str
            Along-track averaging applied to the radar frames before picking
            peak power. Sets dx: '2s' gives ~287 m, '5s' gives ~717 m.
        """
        opr = opr or xopr.opr_access.OPRConnection()
        stac_items = opr.query_frames(
            collections=[collection], segment_paths=[segment]
        )
        stac_items = _sort_by_frame(stac_items)
        if frames is not None:
            wanted = set(int(f) for f in frames)
            stac_items = stac_items.iloc[
                [i for i, p in enumerate(stac_items["properties"])
                 if _frame_number(p) in wanted]
            ]
        elif frame_slice is not None:
            stac_items = stac_items.iloc[frame_slice]
        if len(stac_items) == 0:
            raise ValueError(f"no frames found for {collection}/{segment}")

        x_elev, h_surf, h_bed, x_track, lat_t, lon_t, spans = _load_elevations(
            opr, stac_items
        )
        x_amp, amp = _load_bed_power(opr, stac_items, resample_interval)

        if dx_km is None:
            dx_km = float(np.median(np.diff(x_amp)))

        lo = max(x_elev[0], x_amp[0])
        hi = min(x_elev[-1], x_amp[-1])
        if not hi > lo:
            raise ValueError("bed power and elevation cover disjoint extents")
        x = np.arange(lo, hi + dx_km, dx_km)
        x = x[x <= hi]

        # From the coordinate track, not the pick-filtered grid, so the path
        # follows the aircraft through stretches with no layer pick.
        lat = np.interp(x, x_track, lat_t)
        lon = np.interp(x, x_track, lon_t)

        gap_tol_km = 2.0 * dx_km
        return cls(
            x=x,
            amp=_regrid_preserving_gaps(x_amp, amp, x, gap_tol_km=gap_tol_km),
            h_surf=_regrid_preserving_gaps(x_elev, h_surf, x, gap_tol_km),
            h_bed=_regrid_preserving_gaps(x_elev, h_bed, x, gap_tol_km),
            constants=constants,
            geoid_separation_m=resolve_geoid(lat, lon, geoid),
            lat=lat,
            lon=lon,
            source=ProfileSource(
                collection=collection, segment=segment, frames=spans
            ),
        )


# -- helpers --------------------------------------------------------------
def _frame_number(properties) -> Optional[int]:
    """Frame number from a STAC item's properties, or None if absent."""
    return properties.get("opr:frame") if isinstance(properties, dict) else None


def _sort_by_frame(stac_items):
    """
    Put catalog items in frame order.

    The STAC query makes no ordering promise, and frame_slice is positional, so
    an unsorted return would silently select the wrong frames.
    """
    nums = [_frame_number(p) for p in stac_items["properties"]]
    if any(n is None for n in nums):
        return stac_items
    return stac_items.iloc[np.argsort(nums, kind="stable")]


def _mask_runs(mask: np.ndarray) -> list:
    """Contiguous True runs as (start, stop) inclusive index pairs."""
    if not mask.any():
        return []
    edges = np.diff(mask.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    stops = list(np.where(edges == -1)[0])
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        stops = stops + [len(mask) - 1]
    return list(zip(starts, stops))


def _load_elevations(opr, stac_items):
    """
    Surface and bed WGS84 elevation against along-track distance in km.

    Returns the elevations on their own valid-pick grid plus a separate
    coordinate track. The two are filtered differently on purpose: a stretch
    with no layer pick still has a real flight path, so dropping those samples
    from lat/lon would let the interpolation cut the corner.

    The surface is kept wherever it is picked, even with no bottom under it.
    Requiring both put a hole in the surface at every bed-pick gap, and the
    surface is picked on every sample: 20080730_01 has 17550 of them and 3974
    with no bottom.
    """
    frames = opr.load_frames(stac_items)
    flight_line = xopr.merge_frames(frames)
    # Overlapping frames repeat a slow_time, which every later lookup refuses
    # to align to, so this happens once and before along_track is measured.
    flight_line = flight_line.drop_duplicates("slow_time")
    flight_line = xopr.radar_util.add_along_track(flight_line)
    spans = _frame_spans(frames, flight_line)

    layers = opr.get_layers(flight_line)
    needed = ("standard:surface", "standard:bottom")
    if layers is None or any(name not in layers for name in needed):
        raise ValueError("segment has no surface/bottom layer picks")

    surface = layers["standard:surface"]
    converted = {
        name: xopr.layer_twtt_to_range(
            layer, surface, vertical_coordinate="wgs84"
        )
        for name, layer in layers.items()
    }

    # Layers carry slow_time only, so borrow along_track from the flight line.
    target = converted["standard:surface"].slow_time
    along_track = flight_line["along_track"].reindex(slow_time=target,
                                                     method="nearest")
    lat = flight_line["Latitude"].reindex(slow_time=target, method="nearest")
    lon = flight_line["Longitude"].reindex(slow_time=target, method="nearest")

    # The bottom is picked on a subset of the surface's slow_time, so it is
    # aligned onto that grid exactly: an unpicked sample stays NaN rather than
    # borrowing its neighbour's depth.
    x_km = along_track.values / 1000.0
    h_surf = converted["standard:surface"]["wgs84"].values
    h_bed = converted["standard:bottom"]["wgs84"].reindex(
        slow_time=target).values
    lat_v, lon_v = lat.values, lon.values

    order = np.argsort(x_km)
    x_km, h_surf, h_bed = x_km[order], h_surf[order], h_bed[order]
    lat_v, lon_v = lat_v[order], lon_v[order]

    unique = np.concatenate([[True], np.diff(x_km) > 0])
    track = np.isfinite(x_km) & np.isfinite(lat_v) & np.isfinite(lon_v) & unique
    # h_bed keeps its NaNs; _regrid_preserving_gaps drops them on its own, so
    # the bed comes out of the regrid unchanged and only the surface gains.
    keep = np.isfinite(x_km) & np.isfinite(h_surf) & unique
    return (x_km[keep], h_surf[keep], h_bed[keep],
            x_km[track], lat_v[track], lon_v[track], spans)


def _frame_spans(frames, flight_line) -> tuple:
    """
    Locate each loaded frame on the merged along-track axis, in km.

    Done before the pick filtering so every frame is represented, including
    ones with no layer picks at all.
    """
    along_track = flight_line["along_track"]
    spans = []
    for f in frames:
        t = np.asarray(f["slow_time"].values)
        if t.size == 0:
            continue
        edges = along_track.sel(slow_time=[t.min(), t.max()],
                                method="nearest").values / 1000.0
        timed = np.issubdtype(t.dtype, np.datetime64)
        spans.append(FrameSpan(
            granule=f.attrs.get("granule"),
            frame=f.attrs.get("frame"),
            lo_km=float(np.min(edges)),
            hi_km=float(np.max(edges)),
            start_time=pd.Timestamp(t.min()) if timed else None,
            end_time=pd.Timestamp(t.max()) if timed else None,
        ))
    return _close_seams(sorted(spans, key=lambda s: s.lo_km))


def _close_seams(spans, max_gap_km: float = 1.0) -> tuple:
    """
    Extend each frame up to the next so lookups never fall between frames.

    Consecutive frames leave a one-sample seam. Gaps wider than max_gap_km are
    a genuinely missing frame and are left alone.
    """
    closed = []
    for span, nxt in zip(spans, list(spans[1:]) + [None]):
        if nxt is not None and 0 < nxt.lo_km - span.hi_km <= max_gap_km:
            span = replace(span, hi_km=nxt.lo_km)
        closed.append(span)
    return tuple(closed)


def _load_bed_power(opr, stac_items, resample_interval="2s"):
    """
    Bed echo power in dB against along-track distance in km.

    A frame whose repick finds no samples in the layer margin is dropped rather
    than taking the whole segment with it; the gap it leaves survives the
    regrid as NaN.
    """
    frames, failed = [], []
    for i in range(len(stac_items)):
        try:
            reflectivity = surface_bed_reflection_power(
                stac_items.iloc[i], opr=opr, resample_interval=resample_interval
            )
        except Exception as exc:
            failed.append(f"{type(exc).__name__}: {exc}")
            continue
        if reflectivity is not None:
            frames.append(reflectivity)
    if not frames:
        detail = f"; {failed[0]}" if failed else ""
        raise ValueError(f"no frames yielded bed power{detail}")

    # Not merge_frames: surface_bed_reflection_power drops the granule
    # attribute that merge_frames parses.
    merged = xr.concat(frames, dim="slow_time").sortby("slow_time")
    merged = merged.drop_duplicates("slow_time")
    merged = xopr.radar_util.add_along_track(merged)

    x_km = merged["along_track"].values / 1000.0
    amp = merged["bed_power_dB"].values

    order = np.argsort(x_km)
    x_km, amp = x_km[order], amp[order]
    keep = np.isfinite(x_km) & np.concatenate([[True], np.diff(x_km) > 0])
    return x_km[keep], amp[keep]


def _regrid_preserving_gaps(x_src, values, x_dst, gap_tol_km):
    """
    Interpolate onto x_dst but keep genuine data gaps as NaN.

    Plain interpolation would bridge missing bed picks with invented values,
    which later hides exactly the regions a detector must know about.
    """
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return np.full_like(x_dst, np.nan, dtype=float)

    out = np.interp(x_dst, x_src[valid], values[valid])

    # Blank anything further than gap_tol_km from a real sample.
    nearest = np.searchsorted(x_src[valid], x_dst).clip(1, valid.sum() - 1)
    left = x_src[valid][nearest - 1]
    right = x_src[valid][nearest]
    distance = np.minimum(np.abs(x_dst - left), np.abs(right - x_dst))
    out[distance > gap_tol_km] = np.nan
    return out
