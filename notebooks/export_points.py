"""
Write one geopackage per detector from a finished sweep, for QGIS.

Each layer holds that detector's own picks and nothing else, so loading three
files gives three comparable point sets rather than one column of mixed
methods. Rows where a detector did not answer are dropped rather than written
as null geometry, which QGIS shows as an empty row.

Adds `crossing_gap_km`, the distance from the pick to the flotation crossing.
That is the elevation's own estimate of where grounding begins, computed
without the bed echo, so it is the most useful thing to style or filter on:
small means two independent measurements agree.

Examples
--------
  python notebooks/export_points.py data/sweeps/greenland_v3.jsonl
  python notebooks/export_points.py data/sweeps/greenland_v3.jsonl --detectors bocpd
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xopr_gline.sweep import to_geodataframe

DETECTORS = ("bocpd", "onset", "gradient")


def load(path: Path) -> pd.DataFrame:
    if path.suffix == ".gpkg":
        import geopandas as gpd
        return gpd.read_file(path)
    return pd.DataFrame([json.loads(line) for line in path.open()
                         if line.strip()])


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep", help="the sweep's .jsonl (or .gpkg)")
    p.add_argument("--detectors", nargs="*", default=list(DETECTORS))
    p.add_argument("--out_dir", default=None,
                   help="where to write. Default: beside the sweep")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    sweep = Path(args.sweep)
    out_dir = Path(args.out_dir) if args.out_dir else sweep.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load(sweep)
    found = rows[rows["status"] == "ok"] if "status" in rows else rows
    print(f"{len(found)} rows that found something in {sweep.name}")

    for detector in args.detectors:
        try:
            points = to_geodataframe(found, detector=detector)
        except ValueError as exc:
            print(f"  {detector:9s} skipped: {exc}")
            continue

        points = points[points.geometry.notna()].copy()
        if not len(points):
            print(f"  {detector:9s} no picks")
            continue

        if "flotation_crossing_km" in points:
            points["crossing_gap_km"] = (
                points["map_km"] - points["flotation_crossing_km"]).abs()

        path = out_dir / f"{sweep.stem}_{detector}.gpkg"
        if path.exists() and not args.overwrite:
            print(f"  {detector:9s} {path.name} exists; pass --overwrite")
            continue
        # A geopackage cannot hold a python list, and nothing here needs one.
        for column in points.columns:
            if points[column].map(lambda v: isinstance(v, (list, dict))).any():
                points[column] = points[column].astype(str)
        points.to_file(path, driver="GPKG")

        gap = points.get("crossing_gap_km")
        detail = ("" if gap is None
                  else f", median {gap.median():.2f} km from the crossing")
        print(f"  {detector:9s} {len(points):3d} points -> {path.name}{detail}")


if __name__ == "__main__":
    main()
