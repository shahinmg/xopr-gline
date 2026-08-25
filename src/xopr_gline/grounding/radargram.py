"""
The echo image behind a profile, for figures.

Detection reduces each frame to layer picks and bed power; this loads the
radargram itself so a pick can be judged against what the radar saw. It shares
GlacierProfile's along-track axis, so the two plot on one set of x limits.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

import xopr
import xopr.opr_access
import xopr.radar_util

from .profile import (GlacierProfile, ProfileSource, _frame_spans,
                      select_frames)

# interpolate_to_vertical_grid pads each trace with these above and below its
# own samples. They are fill, not measurements, and would otherwise paint a
# uniform slab across the sky and below the bed.
_PAD_VALUES = (-1.0, -2.0)


@dataclass
class Radargram:
    """
    Echo power on a uniform along-track / elevation grid.

    x is in km on GlacierProfile's axis and z is m WGS84 increasing upward, so
    the layer picks can be drawn straight over `power` without any conversion.
    """

    x: np.ndarray                       # along-track, km
    z: np.ndarray                       # elevation, m WGS84, increasing
    power: np.ndarray                   # dB, shape (len(z), len(x)), NaN in gaps
    source: Optional[ProfileSource] = None

    def __post_init__(self):
        if self.power.shape != (len(self.z), len(self.x)):
            raise ValueError(f"power has shape {self.power.shape}, expected "
                             f"{(len(self.z), len(self.x))}")

    @property
    def extent(self) -> tuple:
        """(left, right, bottom, top) for imshow with origin='lower'."""
        dx = 0.5 * float(self.x[1] - self.x[0])
        dz = 0.5 * float(self.z[1] - self.z[0])
        return (float(self.x[0]) - dx, float(self.x[-1]) + dx,
                float(self.z[0]) - dz, float(self.z[-1]) + dz)

    def clim(self, lo_pct: float = 5.0, hi_pct: float = 99.5) -> tuple:
        """
        Percentile colour limits, so a few bright returns cannot flatten the
        image. Returns (None, None) if the radargram is empty.
        """
        finite = self.power[np.isfinite(self.power)]
        if finite.size == 0:
            return (None, None)
        lo, hi = np.percentile(finite, [lo_pct, hi_pct])
        return (float(lo), float(hi))

    @classmethod
    def from_profile(cls, profile: GlacierProfile,
                     opr: Optional[xopr.opr_access.OPRConnection] = None,
                     pad_m: float = 350.0, **kwargs) -> "Radargram":
        """
        Load the radargram for the frames a profile was built from.

        The vertical grid is cropped to the picks plus pad_m, which keeps the
        panel on the ice instead of on a km of empty air above it. `profile`
        may be a windowed subset: the frames and the axis come from its source,
        so it still lines up.
        """
        source = profile.source
        if source is None or not source.collection or not source.segment:
            raise ValueError("profile carries no provenance to load frames from;"
                             " use Radargram.from_xopr")
        frames = [s.frame for s in source.frames if s.frame is not None]
        kwargs.setdefault("vert_range", _pick_range(profile, pad_m))
        return cls.from_xopr(source.collection, source.segment, opr=opr,
                             frames=frames or None, anchor=source, **kwargs)

    @classmethod
    def from_xopr(cls, collection: str, segment: str,
                  opr: Optional[xopr.opr_access.OPRConnection] = None,
                  frames: Optional[Sequence[int]] = None,
                  frame_slice: Optional[slice] = None,
                  resample_interval: str = "2s",
                  vert_spacing: float = 10.0,
                  vert_range: Optional[tuple] = None,
                  dx_km: Optional[float] = None,
                  anchor: Optional[ProfileSource] = None) -> "Radargram":
        """
        Build a radargram for one segment straight from xOPR.

        Parameters mirror GlacierProfile.from_xopr; `resample_interval` sets the
        along-track stacking and so the horizontal resolution, `vert_spacing`
        the vertical, and `vert_range` crops the elevation grid.

        `anchor` re-registers x onto an existing profile's along-track origin,
        which otherwise starts at whichever frame was loaded first.
        """
        opr = opr or xopr.opr_access.OPRConnection()
        stac_items = select_frames(opr, collection, segment, frames, frame_slice)

        raw = opr.load_frames(stac_items)
        flight_line = xopr.merge_frames(raw)
        flight_line = flight_line.drop_duplicates("slow_time")
        flight_line = xopr.radar_util.add_along_track(flight_line)
        spans = _frame_spans(raw, flight_line)

        # along_track is a coordinate, which the resample drops. Carried as a
        # variable it is averaged in the same bins as the echo, so a column's
        # position is the mean of the traces stacked into it.
        flight_line = flight_line.assign(
            along_track_m=flight_line["along_track"])
        stacked = flight_line.resample(slow_time=resample_interval).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            stacked["Data"] = 10 * np.log10(np.abs(stacked["Data"]))

        vert_min, vert_max = vert_range or (None, None)
        grid = xopr.radar_util.interpolate_to_vertical_grid(
            stacked, vertical_coordinate="wgs84", vert_min=vert_min,
            vert_max=vert_max, vert_spacing=vert_spacing,
        )

        power = np.asarray(grid["Data"].values, dtype=float)
        power[np.isin(power, _PAD_VALUES)] = np.nan
        x_src = np.asarray(grid["along_track_m"].values, dtype=float) / 1000.0
        x, power = _regrid_columns(x_src, power, dx_km)

        shift = _anchor_shift(spans, anchor)
        return cls(x=x + shift, z=np.asarray(grid["wgs84"].values, dtype=float),
                   power=power.T,
                   source=anchor or ProfileSource(collection=collection,
                                                  segment=segment,
                                                  frames=spans))


# -- helpers --------------------------------------------------------------
def _pick_range(profile: GlacierProfile, pad_m: float) -> Optional[tuple]:
    """Elevation range covering the layer picks, padded, or None if unpicked."""
    finite = np.concatenate([profile.h_bed[np.isfinite(profile.h_bed)],
                             profile.h_surf[np.isfinite(profile.h_surf)]])
    if finite.size == 0:
        return None
    return (float(finite.min()) - pad_m, float(finite.max()) + pad_m)


def _regrid_columns(x_src, power, dx_km=None, gap_factor=2.0):
    """
    Traces put on a uniform along-track grid by nearest neighbour.

    imshow needs even spacing, and the aircraft does not fly at a constant
    speed. Columns further than a couple of samples from any trace are left NaN
    so a frame gap stays a gap rather than being smeared across.
    """
    ok = np.isfinite(x_src)
    if ok.sum() < 2:
        raise ValueError("radargram has no positioned traces")
    order = np.argsort(x_src[ok])
    x_src, power = x_src[ok][order], power[ok][order]
    keep = np.concatenate([[True], np.diff(x_src) > 0])
    x_src, power = x_src[keep], power[keep]

    spacing = float(np.median(np.diff(x_src)))
    dx = float(dx_km or spacing)
    x = np.arange(x_src[0], x_src[-1] + dx, dx)
    x = x[x <= x_src[-1]]
    if len(x) < 2:
        raise ValueError("radargram covers less than two grid steps")

    idx = np.searchsorted(x_src, x).clip(0, len(x_src) - 1)
    left = (idx > 0) & (np.abs(x - x_src[idx - 1]) < np.abs(x - x_src[idx]))
    idx[left] -= 1
    out = power[idx]
    out[np.abs(x - x_src[idx]) > gap_factor * max(dx, spacing)] = np.nan
    return x, out


def _anchor_shift(spans, anchor: Optional[ProfileSource]) -> float:
    """
    Offset putting these frames on an existing profile's along-track axis.

    Both measure from the start of whatever was loaded, so a different frame
    subset shifts the origin. Matching on frame number and taking the median
    offset survives one frame being missing from either side.
    """
    if anchor is None:
        return 0.0
    have = {s.frame: s.lo_km for s in spans if s.frame is not None}
    offsets = [want.lo_km - have[want.frame] for want in anchor.frames
               if want.frame in have and np.isfinite(want.lo_km)]
    if not offsets:
        raise ValueError("radargram frames do not overlap the profile's; the "
                         "along-track axes cannot be registered")
    return float(np.median(offsets))
