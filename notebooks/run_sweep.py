"""
Run grounding point detection over every vetted flight line.

Writes one row per run to JSONL as it goes, so an interrupted sweep resumes
where it stopped, and a geopackage of grounding points at the end.

Every invocation is recorded in a sibling .runs.jsonl with its settings and
commit, and every row carries the sweep_id that produced it.

Examples
--------
  python notebooks/run_sweep.py --out data/sweeps/greenland_v2.jsonl
  python notebooks/run_sweep.py --margin_km 5 --workers 12 --limit 20
  python notebooks/run_sweep.py --elev_cutoff_km 20 --out data/sweeps/cut20.jsonl
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from xopr_gline.grounding.features import (AMP_CUTOFF_KM, ELEV_CUTOFF_KM,
                                           SATURATE_M)
from xopr_gline.sweep import (DEFAULT_DX_KM, build_tasks, forget, run_sweep,
                              to_geodataframe)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/sweeps/greenland_v2.jsonl",
                   help="JSONL of rows, appended to and resumed from. Pick a "
                        "new name per experiment; a sweep resumes into an "
                        "existing file rather than replacing it")
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
    p.add_argument("--dx_km", type=float, default=DEFAULT_DX_KM,
                   help="along-track grid spacing. Pinned by default: left to "
                        "itself it follows the frames loaded and no two runs "
                        "are comparable")
    p.add_argument("--pad_km", type=float, default=4.0,
                   help="context beyond the search window given to BOCPD but "
                        "not reported on, so edge transients cannot win. 0 "
                        "restores the old trim-only behaviour")
    p.add_argument("--headline", default="bocpd",
                   choices=["bocpd", "onset", "gradient"],
                   help="which detector's pick becomes the geometry")
    p.add_argument("--also_gpkg", nargs="*", default=["bocpd"],
                   metavar="DETECTOR",
                   help="write extra geopackages keyed on these detectors' own "
                        "picks, e.g. bocpd for the raw changepoint. Pass with "
                        "no names to write only the headline one")
    p.add_argument("--snap", action="store_true",
                   help="move a pick pinned against the calving front onto the "
                        "front. Off by default: the geometry should be what a "
                        "detector found, and at_terminus / terminus_gap_km / "
                        "edge_pinned carry that reading without overwriting it")
    p.add_argument("--amp_cutoff_km", type=float, default=AMP_CUTOFF_KM,
                   help="low-pass wavelength for the bed power channels")
    p.add_argument("--elev_cutoff_km", type=float, default=ELEV_CUTOFF_KM,
                   help="low-pass wavelength for the elevation channels. The "
                        "default is inherited from the Petermann CSV era and "
                        "is worth sweeping")
    p.add_argument("--saturate_m", type=float, default=SATURATE_M,
                   help="tanh scale for the flotation residual channel, so the "
                        "inland ramp cannot dominate. 0 disables")
    p.add_argument("--geoid", default="data/bedmachine/BedMachineGreenland-v5.nc")
    p.add_argument("--fig_dir", default=None,
                   help="write a four-panel figure per grounding point here, "
                        "named by run_id. Drawing is serialised, so expect it "
                        "to add a second or two per run")
    p.add_argument("--workers", type=int, default=4,
                   help="threads. Memory, not CPU, is the limit: each holds a "
                        "run's full radar arrays, and 8 with --fig_dir was "
                        "OOM-killed at 23 GB. Raise it only without figures")
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
                       invocation={"argv": sys.argv, "args": vars(args)},
                       geoid=args.geoid, margin_km=args.margin_km,
                       dx_km=args.dx_km, pad_km=args.pad_km,
                       headline=args.headline,
                       snap_to_terminus=args.snap,
                       amp_cutoff_km=args.amp_cutoff_km,
                       elev_cutoff_km=args.elev_cutoff_km,
                       saturate_m=args.saturate_m,
                       screen=not args.no_screen, fig_dir=args.fig_dir)

    print("\n" + points["status"].value_counts().to_string())
    gpkg = Path(args.gpkg) if args.gpkg else out.with_suffix(".gpkg")
    found = points[points["status"] == "ok"]
    if len(found):
        # Never in place: an existing geopackage is a previous experiment.
        if gpkg.exists():
            raise SystemExit(f"{gpkg} exists; pass --gpkg to write elsewhere "
                             f"so the earlier sweep is not overwritten")
        found.to_file(gpkg, driver="GPKG")
        print(f"\n{len(found)} grounding points -> {gpkg}")

        # A companion keyed on the raw changepoint, so the detector's own
        # answer can be mapped without the terminus snapping in the way.
        for detector in args.also_gpkg:
            try:
                raw = to_geodataframe(found, detector=detector)
            except ValueError as exc:
                print(f"\nno {detector} geopackage: {exc}")
                continue
            path = gpkg.with_name(f"{gpkg.stem}_{detector}{gpkg.suffix}")
            if path.exists():
                raise SystemExit(f"{path} exists; move it aside first")
            raw.to_file(path, driver="GPKG")
            print(f"{raw.geometry.notna().sum()} {detector} points -> {path}")

        from xopr_gline.validate import scoreboard
        print("\n" + scoreboard(found).to_string())
    else:
        print("\nno grounding points found")
    if args.fig_dir and "figure" in points:
        drawn = points["figure"].notna().sum()
        print(f"{drawn} figures -> {args.fig_dir}"
              + (f"  ({len(found) - drawn} failed to draw)"
                 if drawn < len(found) else ""))


if __name__ == "__main__":
    main()
