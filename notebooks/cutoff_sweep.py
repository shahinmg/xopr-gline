"""
Sweep one filter cutoff over a set of lines, holding everything else fixed.

One sweep per --cutoffs value, each with its own jsonl, gpkg and figure dir, so
picks can be compared at settings that differ in nothing else. The radar
downloads are shared through one cache, so only the first cutoff pays for them.

  python notebooks/cutoff_sweep.py --vary elev \
      --tasks_from /home/laserglaciers/igs_2026/greenland_v6_bocpd_cleaned_terminus_gap_helheim.gpkg \
      --baseline /home/laserglaciers/igs_2026/greenland_v6_bocpd_cleaned_terminus_gap_helheim.gpkg \
      --prefix data/sweeps/helheim_elev --fig_root data/figs/helheim_elev_cutoff
  python notebooks/cutoff_sweep.py --vary amp --near data/petermann_grounding_zones/petermann_grounding_zones
"""

import argparse
import glob
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from xopr_gline.sweep import (DEFAULT_DX_KM, build_tasks, run_sweep,
                              tasks_from_points, to_geodataframe,
                              useable_frames)

ZONE_PATTERN = "[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9].shp"
DEFAULT_CUTOFFS = {"amp": [9.55, 4.0, 2.5, 1.0], "elev": [71.6, 40.0, 20.0, 10.0]}


def near_frames(lines: str, zone_dir: str, buffer_m: float):
    """Vetted frames whose flight line passes within buffer_m of a zone."""
    zones = pd.concat([gpd.read_file(p) for p in
                       sorted(glob.glob(f"{zone_dir}/{ZONE_PATTERN}"))])
    area = gpd.GeoSeries([zones.to_crs(3413).geometry.union_all()],
                         crs=3413).buffer(buffer_m).iloc[0]
    vetted = gpd.read_file(lines).to_crs(3413)
    near = set(vetted[vetted.intersects(area)]["Name"].astype(str))
    frames = useable_frames(lines)
    return frames[frames["granule"].isin(near)].reset_index(drop=True)


def tag(cutoff: float) -> str:
    """File-safe label for a cutoff, e.g. 9.55 -> 9p55."""
    return f"{cutoff:g}".replace(".", "p")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vary", choices=["amp", "elev"], default="amp",
                   help="which low-pass the loop varies; the other is held at "
                        "its default and recorded on every row")
    p.add_argument("--cutoffs", type=float, nargs="+", default=None,
                   help="wavelengths in km. Defaults per --vary; put the "
                        "current setting first so it is the baseline")
    p.add_argument("--tasks_from", default=None,
                   help="rerun the frames of a previous sweep's output, gpkg "
                        "or jsonl. Takes precedence over --near")
    p.add_argument("--all_lines", action="store_true",
                   help="every vetted flight line, i.e. an ice-sheet-wide run")
    p.add_argument("--near", default=None,
                   help="select vetted lines passing near these zone "
                        "shapefiles instead")
    p.add_argument("--buffer_km", type=float, default=40.0)
    p.add_argument("--lines", default="data/flight_lines/cresis_useable_flight_lines.gpkg")
    p.add_argument("--reference", default=None,
                   help="grounding zone layers to report against. Omit where "
                        "none exist and the gz columns are simply absent")
    p.add_argument("--amp_cutoff_km", type=float, default=None,
                   help="held value for the bed-power low-pass while --vary is "
                        "elev. 0 disables it. Default: the shipped 9.55")
    p.add_argument("--elev_cutoff_km", type=float, default=None,
                   help="held value for the elevation low-pass while --vary is "
                        "amp. 0 disables it. Default: the shipped 71.6")
    p.add_argument("--da_cutoff_km", type=float, default=None,
                   help="low-pass for the dA gradient channel alone. Set this "
                        "when running --amp_cutoff_km 0, or dA becomes the "
                        "gradient of raw dB")
    p.add_argument("--despike_km", type=float, default=0.0,
                   help="Hampel window for the bed-power channel, km. Removes "
                        "berg spikes without smearing the step; pair with "
                        "--amp_cutoff_km 0 to keep the raw shape")
    p.add_argument("--models", nargs="+", default=None,
                   metavar="MODEL",
                   help="BOCPD likelihoods to combine, e.g. ifm. Default is "
                        "the ifm+fullcov ensemble; a geometric mean follows "
                        "whichever is sharper, so a weak model can override a "
                        "good one")
    p.add_argument("--channels", nargs="+", default=None,
                   help="channel names to feed the detector, e.g. amp dfree "
                        "dsurf to drop dA entirely")
    p.add_argument("--reference_pattern", default=ZONE_PATTERN,
                   help="glob for the zone files inside --reference when it is "
                        "a directory")
    p.add_argument("--baseline", default=None,
                   help="previous output to measure each pick's shift from, "
                        "joined on segment and frames_requested")
    p.add_argument("--prefix", default="data/sweeps/cutoff",
                   help="jsonl and gpkg are named <prefix>_<cutoff>.*")
    p.add_argument("--fig_root", default=None,
                   help="figures go in <fig_root>/<cutoff>/. Default: derived "
                        "from --prefix")
    p.add_argument("--cache_dir", default=None,
                   help="shared xOPR download cache. Strongly recommended: "
                        "without it every cutoff refetches the same frames")
    p.add_argument("--margin_km", type=float, default=12.0)
    p.add_argument("--dx_km", type=float, default=DEFAULT_DX_KM)
    p.add_argument("--pad_km", type=float, default=4.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no_figs", action="store_true")
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    cutoffs = args.cutoffs or DEFAULT_CUTOFFS[args.vary]
    fig_root = args.fig_root or f"data/figs/{Path(args.prefix).name}_cutoff"

    if args.tasks_from:
        tasks = tasks_from_points(args.tasks_from)
    elif args.all_lines:
        tasks = build_tasks(frames=useable_frames(args.lines), pad_frames=1)
    elif args.near:
        tasks = build_tasks(frames=near_frames(args.lines, args.near,
                                               args.buffer_km * 1000.0),
                            pad_frames=1)
    else:
        raise SystemExit("pass --tasks_from, --all_lines or --near")
    if args.limit:
        tasks = tasks.head(args.limit)
    print(f"{len(tasks)} runs over {tasks['segment'].nunique()} segments, "
          f"{tasks['frames'].map(lambda f: len(f) if f else 0).sum()} frames")
    print(f"varying {args.vary}_cutoff_km: {cutoffs}\n")

    baseline = load_baseline(args.baseline)
    summaries = []
    for cutoff in cutoffs:
        out = Path(f"{args.prefix}_{tag(cutoff)}.jsonl")
        gpkg = out.with_suffix(".gpkg")
        fig_dir = None if args.no_figs else f"{fig_root}/{tag(cutoff)}"
        print(f"\n=== {args.vary}_cutoff_km = {cutoff} -> {out} ===", flush=True)

        cuts = {f"{args.vary}_cutoff_km": cutoff}
        # The channel not being swept is pinned here rather than left on its
        # shipped default, so both can be set from physical scales.
        for name, held in (("amp", args.amp_cutoff_km),
                           ("elev", args.elev_cutoff_km)):
            if name != args.vary and held is not None:
                cuts[f"{name}_cutoff_km"] = held
        points = run_sweep(
            tasks, out=out, workers=args.workers,
            resume=not args.no_resume, cache_dir=args.cache_dir,
            invocation={"argv": sys.argv, "args": vars(args),
                        "note": f"{args.vary} cutoff sweep"},
            geoid="data/bedmachine/BedMachineGreenland-v5.nc",
            margin_km=args.margin_km, dx_km=args.dx_km, pad_km=args.pad_km,
            headline="bocpd", snap_to_terminus=False,
            screen=True, fig_dir=fig_dir,
            datum_correct=True, radargram=not args.no_figs,
            crop_features=False,
            reference=args.reference, reference_pattern=args.reference_pattern,
            da_cutoff_km=args.da_cutoff_km, channels=args.channels,
            despike_km=args.despike_km, models=args.models,
            **cuts,
        )

        print("\n" + points["status"].value_counts().to_string())
        found = points[points["status"] == "ok"]
        if not len(found):
            print("no grounding points")
            continue
        if gpkg.exists():
            print(f"{gpkg} exists, leaving it alone")
        else:
            found.to_file(gpkg, driver="GPKG")
            print(f"{len(found)} grounding points -> {gpkg}")
            to_geodataframe(found, detector="bocpd").to_file(
                gpkg.with_name(f"{gpkg.stem}_bocpd{gpkg.suffix}"), driver="GPKG")
        summaries.append(summarise(found, cutoff, baseline))

    if summaries:
        table = pd.DataFrame(summaries).set_index("cutoff_km")
        print(f"\n\n=== {args.vary} cutoff comparison ===")
        print(table.to_string(float_format=lambda v: f"{v:.2f}"))
        table.to_csv(f"{args.prefix}_comparison.csv")
        print(f"\n-> {args.prefix}_comparison.csv")


def load_baseline(path):
    """Previous picks keyed on segment and frame span, or None."""
    if not path:
        return None
    base = gpd.read_file(path) if str(path).endswith(".gpkg") else \
        pd.DataFrame([__import__("json").loads(l) for l in open(path)])
    if "bocpd_km" not in base:
        raise SystemExit(f"{path} has no bocpd_km to compare against")
    return base.set_index(
        base["segment"].astype(str) + "/" + base["frames_requested"].astype(str)
    )["bocpd_km"]


def summarise(found: gpd.GeoDataFrame, cutoff: float, baseline=None) -> dict:
    """One row per cutoff: how the picks sit against the zone, the baseline
    and each other."""
    from xopr_gline.validate import attach_glacier_ids, repeat_consistency

    row = {"n": len(found)}

    off = pd.to_numeric(found.get("gz_offset_km"), errors="coerce") \
        if "gz_offset_km" in found else pd.Series(dtype=float)
    if off.notna().any():
        row.update(n_with_zone=int(off.notna().sum()),
                   in_zone=int((off == 0).sum()),
                   median_gz_offset_km=float(off.abs().median()),
                   p90_gz_offset_km=float(off.abs().quantile(0.9)))

    if baseline is not None:
        key = (found["segment"].astype(str) + "/"
               + found["frames_requested"].astype(str))
        shift = (pd.to_numeric(found["bocpd_km"], errors="coerce").values
                 - key.map(baseline).values.astype(float))
        row.update(n_vs_baseline=int(np.isfinite(shift).sum()),
                   mean_abs_shift_km=float(np.nanmean(np.abs(shift))),
                   max_abs_shift_km=float(np.nanmax(np.abs(shift))))

    try:
        spread = repeat_consistency(attach_glacier_ids(found))
        row.update(median_repeat_km=float(spread["repeat_offset_km"].median()),
                   max_repeat_km=float(spread["repeat_offset_km"].max()))
    except Exception as exc:                      # no mapping, or no frame column
        row["repeat_reason"] = str(exc)[:60]

    for col in ("bocpd_model_spread_km", "ci95_width_km"):
        values = pd.to_numeric(found.get(col), errors="coerce")
        if values is not None and values.notna().any():
            row[f"median_{col}"] = float(values.median())
    row.update(
        edge_pinned=int(found.get("edge_pinned", pd.Series(dtype=bool)).sum()),
        at_terminus=int(found.get("at_terminus", pd.Series(dtype=bool)).sum()),
    )
    return {"cutoff_km": cutoff, **row}


if __name__ == "__main__":
    main()
