"""
External reference zones located on a profile's along-track axis.

An InSAR grounding zone is a different measurement of a different thing, made in
a different year, so it is drawn and reported beside a detection but never used
to find one. See xopr_gline.validate for judging a whole sweep against one.
"""

import re
from pathlib import Path
from typing import Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

from .profile import GlacierProfile, _mask_runs

# Epoch leading a file name, e.g. '2011_2015_gzone+iceshelf.shp'.
_EPOCH = re.compile(r"^(\d{4})(?:_(\d{4}))?")


def load_zones(path: Union[str, Path], pattern: str = "*.shp") -> gpd.GeoDataFrame:
    """
    Reference zones from one vector file, or every file matching `pattern` in a
    directory.

    Each row keeps the file it came from and the epoch parsed from its name, so
    `zone_for_date` can pick the survey nearest a flight. A name leading with
    YYYY_YYYY or a single YYYY carries an epoch; anything else gets NaT and is
    only ever selected when nothing else is available.
    """
    path = Path(path)
    files = sorted(p for p in path.glob(pattern)
                   if not p.name.startswith("._")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"no files matching {pattern!r} in {path}")

    frames = []
    for f in files:
        g = gpd.read_file(f)
        start, end = _epoch(f.stem)
        frames.append(g.assign(zone_name=f.stem, epoch_start=start,
                               epoch_end=end))
    zones = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(zones, geometry="geometry", crs=frames[0].crs)


def zone_for_date(zones: gpd.GeoDataFrame,
                  date) -> Optional[gpd.GeoDataFrame]:
    """
    The zone whose epoch is nearest `date`, as a one-or-more row frame.

    Zero distance when the date falls inside an epoch, otherwise the gap to the
    nearer end. Returns every row sharing the winning name, since one survey can
    be several polygons.
    """
    if len(zones) == 0:
        return None
    date = pd.Timestamp(date)
    if pd.isna(date):
        return zones
    gap = zones.apply(lambda r: _epoch_gap(r["epoch_start"], r["epoch_end"],
                                           date), axis=1)
    if gap.isna().all():
        return zones
    return zones[zones["zone_name"] == zones.loc[gap.idxmin(), "zone_name"]]


def epoch_label(zones) -> Optional[str]:
    """
    A zone's epoch as 'YYYY-YYYY', for a figure legend.

    Falls back to the file name for a layer whose name carried no epoch.
    """
    if zones is None or len(zones) == 0:
        return None
    row = zones.iloc[0]
    start, end = row.get("epoch_start"), row.get("epoch_end")
    if pd.notna(start) and pd.notna(end):
        return f"{start.year}-{end.year}"
    name = row.get("zone_name")
    return None if name is None else str(name).replace("_", "-")


def zone_crossings_km(profile: GlacierProfile,
                      zones: Union[gpd.GeoDataFrame, gpd.GeoSeries],
                      min_run_km: float = 0.0) -> list:
    """
    Along-track stretches where the flight track lies inside a zone, in km.

    Returned as (lo_km, hi_km) pairs, one per crossing: an out-and-back line
    enters the same zone twice and both are real. Empty if the track misses it.
    """
    if profile.lat is None or profile.lon is None:
        raise ValueError("profile carries no coordinates to locate a zone on")
    if len(zones) == 0:
        return []

    track = gpd.GeoSeries(gpd.points_from_xy(profile.lon, profile.lat),
                          crs="EPSG:4326")
    # Into the zone's own projection rather than the other way about: these are
    # polar polygons, and their straight edges are straight in 3413.
    inside = track.to_crs(zones.crs).within(unary_union(zones.geometry.values))

    runs = []
    for start, stop in _mask_runs(np.asarray(inside)):
        lo, hi = float(profile.x[start]), float(profile.x[stop])
        if hi - lo >= min_run_km:
            runs.append((lo, hi))
    return runs


def offset_from_zone_km(profile: GlacierProfile, x_km: float,
                        bands: Sequence[tuple]) -> float:
    """
    Signed along-track distance from a pick to the nearest zone, in km.

    0 inside the zone, negative where the pick sits downflow of it and positive
    upflow, matching the convention run_grounding.py prints. NaN with no zone.
    """
    if not bands:
        return np.nan
    lo, hi = min(bands, key=lambda b: min(abs(x_km - b[0]), abs(x_km - b[1])))
    if lo <= x_km <= hi:
        return 0.0
    signed = (x_km - hi) if x_km > hi else (x_km - lo)
    return float(profile.landward_sign() * signed)


# -- helpers --------------------------------------------------------------
def _epoch(stem: str) -> tuple:
    """
    (start, end) timestamps from a leading YYYY_YYYY or YYYY, else (NaT, NaT).

    A single year is a one-year epoch, so a survey named for the year it was
    flown matches dates the same way a multi-year one does.
    """
    m = _EPOCH.match(stem)
    if not m:
        return (pd.NaT, pd.NaT)
    return (pd.Timestamp(year=int(m.group(1)), month=1, day=1),
            pd.Timestamp(year=int(m.group(2) or m.group(1)), month=12, day=31))


def _epoch_gap(start, end, date) -> float:
    """Days from `date` to an epoch, 0 inside it and NaN for an unparsed one."""
    if pd.isna(start) or pd.isna(end):
        return np.nan
    if start <= date <= end:
        return 0.0
    return float(min(abs((date - start).days), abs((date - end).days)))
