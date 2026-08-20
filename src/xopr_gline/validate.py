"""
Judge a sweep against something other than eyeballing the map.

Two references are available. The cheap one needs no external data at all: the
same glacier is usually flown in several years, and repeat grounding points
should cluster within a few km unless the glacier really moved, so the scatter
within a cluster measures the method. The other compares against a supplied
reference such as an InSAR grounding zone, which is a different measurement of
a different thing rather than truth: expect the radar grounding point slightly
downflow of the InSAR limit of tidal flexure.
"""

from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd

METRIC_CRS = "EPSG:3413"
DEFAULT_MAPPING = Path("data/glacier_to_radarframe_ids.geojson")


def attach_glacier_ids(points: gpd.GeoDataFrame,
                       mapping: Union[str, Path, gpd.GeoDataFrame, None] = None
                       ) -> gpd.GeoDataFrame:
    """
    Add the GlacierID each grounding point's frame belongs to.

    Joined on the frame granule rather than by position, so a point that landed
    in the wrong place is still attributed to the glacier it was looking for.
    """
    if mapping is None:
        mapping = DEFAULT_MAPPING
    if not isinstance(mapping, gpd.GeoDataFrame):
        mapping = gpd.read_file(mapping)

    lookup = (mapping[["stac_item_id", "GlacierID"]]
              .assign(frame=mapping["stac_item_id"].str.replace(
                  r"^Data_", "", regex=True))
              .drop_duplicates("frame")
              .set_index("frame")["GlacierID"])
    out = points.copy()
    out["GlacierID"] = out["frame"].map(lookup)
    return out


def repeat_consistency(points: gpd.GeoDataFrame,
                       group: str = "GlacierID",
                       tolerance_km: float = 3.0) -> gpd.GeoDataFrame:
    """
    Scatter of repeat grounding points on the same glacier, in km.

    Adds the distance from each point to its group's medoid and flags the ones
    beyond tolerance_km. Groups of one are kept with NaN, since a single visit
    is neither consistent nor inconsistent.
    """
    if group not in points.columns:
        raise ValueError(f"no {group!r} column; run attach_glacier_ids first")

    out = points.copy()
    metric = out.to_crs(METRIC_CRS)
    out["repeat_offset_km"] = np.nan
    out["repeat_n"] = np.nan

    for _, block in metric.groupby(group):
        located = block[block.geometry.notna()]
        if len(located) == 0:
            continue
        # Distances are to the medoid, not the centroid: one bad point drags a
        # centroid far enough to put every visit over the tolerance, which
        # reports four outliers where there is one.
        pairwise = np.array([located.geometry.distance(g).values
                             for g in located.geometry])
        centre = located.geometry.iloc[int(np.argmin(pairwise.sum(axis=1)))]
        out.loc[located.index, "repeat_offset_km"] = (
            located.geometry.distance(centre).values / 1000.0)
        out.loc[located.index, "repeat_n"] = len(located)

    out["repeat_outlier"] = ((out["repeat_n"] > 1)
                             & (out["repeat_offset_km"] > tolerance_km))
    return out


def repeat_summary(points: gpd.GeoDataFrame,
                   group: str = "GlacierID") -> pd.DataFrame:
    """One row per glacier: how many visits, over what years, and how tight."""
    if "repeat_offset_km" not in points.columns:
        points = repeat_consistency(points, group=group)
    years = (pd.to_datetime(points["date"], errors="coerce").dt.year
             if "date" in points else pd.Series(index=points.index, dtype=float))
    return (points.assign(year=years)
            .groupby(group)
            .agg(n=("repeat_offset_km", "size"),
                 years=("year", lambda s: sorted(set(s.dropna().astype(int)))),
                 median_offset_km=("repeat_offset_km", "median"),
                 max_offset_km=("repeat_offset_km", "max"),
                 outliers=("repeat_outlier", "sum"))
            .sort_values("max_offset_km", ascending=False))


def compare_to_reference(points: gpd.GeoDataFrame,
                         reference: Union[str, Path, gpd.GeoDataFrame],
                         name: str = "reference") -> gpd.GeoDataFrame:
    """
    Distance from each grounding point to the nearest reference feature, in km.

    `reference` is any line or point layer, e.g. a MEaSUREs InSAR grounding
    zone. Unsigned: it says how far off the answer is, not which side, since
    the expected offset is downflow and that needs a flow direction.
    """
    if not isinstance(reference, gpd.GeoDataFrame):
        reference = gpd.read_file(reference)

    metric = points.to_crs(METRIC_CRS)
    lines = reference.to_crs(METRIC_CRS).geometry.union_all()
    out = points.copy()
    out[f"{name}_distance_km"] = np.where(
        metric.geometry.notna(),
        metric.geometry.distance(lines) / 1000.0,
        np.nan,
    )
    return out


def report(points: gpd.GeoDataFrame,
           reference: Optional[Union[str, Path, gpd.GeoDataFrame]] = None,
           mapping: Union[str, Path, gpd.GeoDataFrame, None] = None
           ) -> gpd.GeoDataFrame:
    """Attach every available check to a sweep's points, in one call."""
    out = attach_glacier_ids(points, mapping=mapping)
    out = repeat_consistency(out)
    if reference is not None:
        out = compare_to_reference(out, reference)
    return out


def scoreboard(points: gpd.GeoDataFrame) -> pd.Series:
    """
    The handful of numbers to watch when tuning.

    Compare two sweeps on this rather than on a map: it is what tells a change
    that moved the answers from a change that improved them.
    """
    found = points[points["status"] == "ok"] if "status" in points else points
    n = len(found)

    def fraction(column, value=True):
        if column not in found or not n:
            return np.nan
        return float((found[column] == value).sum()) / n

    numbers = {
        "n": n,
        "edge_pinned": fraction("edge_pinned"),
        "at_window_edge": fraction("at_window_edge"),
        "at_terminus": fraction("at_terminus"),
        "snapped_to_terminus": fraction("snapped_to_terminus"),
        "confidence_high": fraction("confidence", "high"),
        "confidence_low": fraction("confidence", "low"),
    }
    for column in ("detector_spread_km", "bocpd_model_spread_km",
                   "reference_distance_km"):
        if column in found:
            numbers[f"median_{column}"] = float(found[column].median())

    # Only glaciers visited more than once say anything about consistency, and
    # they are the minority: including the singletons at their own zero offset
    # drags the median to nothing.
    if "repeat_offset_km" in found:
        repeats = found[found["repeat_n"] > 1]
        numbers["n_repeat"] = len(repeats)
        numbers["median_repeat_offset_km"] = float(
            repeats["repeat_offset_km"].median()) if len(repeats) else np.nan
        numbers["repeat_outliers"] = (float(repeats["repeat_outlier"].mean())
                                      if len(repeats) else np.nan)
    return pd.Series(numbers)
