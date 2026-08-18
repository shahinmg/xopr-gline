"""
Run grounding point detection over every vetted flight line.

Writes one row per run to JSONL as it goes, so an interrupted sweep resumes
where it stopped, and a geopackage of grounding points at the end.

Examples
--------
  python notebooks/run_sweep.py --out data/sweeps/greenland.jsonl
  python notebooks/run_sweep.py --margin_km 5 --workers 12 --limit 20
"""

import argparse
from pathlib import Path

import pandas as pd

from xopr_gline.sweep import build_tasks, forget, run_sweep


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/sweeps/greenland.jsonl",
                   help="JSONL of rows, appended to and resumed from")
    p.add_argument("--gpkg", default=None,
                   help="write the grounding points here. Default: --out with "
                        "a .gpkg suffix")
    p.add_argument("--lines", default=None, help="vetted flight lines gpkg")
    p.add_argument("--margin_km", type=float, default=12.0,
                   help="half-width of the search window around flotation. "
                        "The answer is sensitive to it: Helheim moves 9 km "
                        "between 5 and 12")
    p.add_argument("--pad_frames", type=int, default=1,
                   help="frames to load either side of each vetted cluster")
    p.add_argument("--geoid", default="data/bedmachine/BedMachineGreenland-v5.nc")
    p.add_argument("--fig_dir", default=None,
                   help="write a four-panel figure per grounding point here, "
                        "named by run_id. Drawing is serialised, so expect it "
                        "to add a second or two per run")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="run only the first N tasks, for a smoke test")
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--no_screen", action="store_true",
                   help="detect on every run, screened or not")
    p.add_argument("--retry", nargs="*", default=None,
                   metavar="STATUS",
                   help="forget rows with these statuses and run them again, "
                        "for after a bug fix. Default: the failure statuses")
    p.add_argument("--cache_dir", default=None)
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.retry is not None:
        statuses = args.retry or ["no_profile", "not_in_catalog",
                                  "no_detection", "error"]
        print(f"forgot {forget(out, statuses)} rows: {', '.join(statuses)}")

    frames = None
    if args.lines:
        from xopr_gline.sweep import useable_frames
        frames = useable_frames(args.lines)
    tasks = build_tasks(frames=frames, pad_frames=args.pad_frames)
    if args.limit:
        tasks = tasks.head(args.limit)
    print(f"{len(tasks)} runs over {tasks['segment'].nunique()} segments, "
          f"{tasks['frames'].map(lambda f: len(f) if f else 0).sum()} frames")

    points = run_sweep(tasks, out=out, workers=args.workers,
                       resume=not args.no_resume, cache_dir=args.cache_dir,
                       geoid=args.geoid, margin_km=args.margin_km,
                       screen=not args.no_screen, fig_dir=args.fig_dir)

    print("\n" + points["status"].value_counts().to_string())
    gpkg = Path(args.gpkg) if args.gpkg else out.with_suffix(".gpkg")
    found = points[points["status"] == "ok"]
    if len(found):
        found.to_file(gpkg, driver="GPKG")
        print(f"\n{len(found)} grounding points -> {gpkg}")
    else:
        print("\nno grounding points found")
    if args.fig_dir and "figure" in points:
        drawn = points["figure"].notna().sum()
        print(f"{drawn} figures -> {args.fig_dir}"
              + (f"  ({len(found) - drawn} failed to draw)"
                 if drawn < len(found) else ""))


if __name__ == "__main__":
    main()
