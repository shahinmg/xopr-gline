"""
Score one sweep, or set two side by side.

Tuning without this is guesswork: a change that moves every answer looks
identical to a change that improves them. Compares on the quality flags, the
scatter of repeat visits to the same glacier, and optionally an external
reference layer such as an InSAR grounding zone.

Examples
--------
  python notebooks/compare_sweeps.py data/sweeps/greenland_v2.jsonl
  python notebooks/compare_sweeps.py data/sweeps/greenland.jsonl \
      data/sweeps/greenland_v2.jsonl
  python notebooks/compare_sweeps.py a.jsonl b.jsonl --reference gz_lines.gpkg
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from xopr_gline.sweep import sweep_log, to_geodataframe
from xopr_gline.validate import report, repeat_summary, scoreboard


def load(path: Path):
    """A sweep's points, from its JSONL or its geopackage."""
    path = Path(path)
    if path.suffix == ".gpkg":
        import geopandas as gpd
        return gpd.read_file(path)
    rows = [json.loads(line) for line in path.open() if line.strip()]
    return to_geodataframe(pd.DataFrame(rows))


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweeps", nargs="+", help="one or more .jsonl or .gpkg")
    p.add_argument("--reference", default=None,
                   help="external layer to measure against, e.g. an InSAR "
                        "grounding zone. Unsigned distance in km")
    p.add_argument("--mapping", default=None,
                   help="frame to GlacierID lookup. Default: "
                        "data/glacier_to_radarframe_ids.geojson")
    p.add_argument("--worst", type=int, default=10,
                   help="how many of the least consistent glaciers to list")
    args = p.parse_args()

    boards, scored = {}, {}
    for path in args.sweeps:
        points = load(path)
        found = points[points["status"] == "ok"] if "status" in points else points
        checked = report(found, reference=args.reference, mapping=args.mapping)
        scored[path] = checked
        boards[Path(path).stem] = scoreboard(checked)

    print(pd.DataFrame(boards).to_string(float_format=lambda v: f"{v:.3f}"))

    for path, checked in scored.items():
        print(f"\n=== {path}: least consistent glaciers")
        summary = repeat_summary(checked)
        print(summary[summary["n"] > 1].head(args.worst).to_string())

        history = sweep_log(Path(path).with_suffix(".jsonl"))
        if len(history):
            last = history.iloc[-1]
            print(f"\nlast run {last.get('sweep_id')} at commit "
                  f"{last.get('commit')}"
                  + (" (dirty)" if last.get("dirty") else ""))


if __name__ == "__main__":
    main()
