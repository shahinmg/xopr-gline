"""
Find the grounding point for one xOPR segment.

No CSVs and no hardcoded along-track constants: point it at a collection and
segment. The search window defaults to the flotation crossing, so an InSAR
grounding zone is validation, not input.

Examples
--------
  python notebooks/run_grounding.py \\
      --collection 2010_Greenland_DC8 --segment 20100420_03 \\
      --frames 7:11 --gz_lo 95.38 --gz_hi 97.89

  python notebooks/run_grounding.py \\
      --collection 2010_Greenland_DC8 --segment 20100420_03 \\
      --search_lo 75 --search_hi 96
"""

import argparse
import warnings

import numpy as np

import xopr.opr_access

from xopr_gline.grounding import (BOCPDDetector, GlacierProfile,
                                  GradientDetector, OnsetDetector,
                                  select_flotation_leg, transition_width_km)


def parse_slice(text):
    if text is None:
        return None
    lo, _, hi = text.partition(":")
    return slice(int(lo) if lo else None, int(hi) if hi else None)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True)
    p.add_argument("--segment", required=True)
    p.add_argument("--frames", default=None,
                   help="frame slice, e.g. '7:11'. Default: whole segment")
    p.add_argument("--resample", default="2s",
                   help="along-track averaging; '2s' ~287 m, '5s' ~717 m")
    p.add_argument("--geoid", default=None,
                   help="geoid separation: a number in m, or a path to "
                        "BedMachine to sample it along the flight line")
    p.add_argument("--bathymetry", default=None,
                   help="BedMachine path to draw the seabed under the floating "
                        "ice from. Figure only; the radar picks are untouched")
    p.add_argument("--dx_km", type=float, default=None,
                   help="grid spacing; default is the bed-power spacing")
    p.add_argument("--crop_lo", type=float, default=None,
                   help="restrict the profile to valid data before detecting")
    p.add_argument("--crop_hi", type=float, default=None)
    p.add_argument("--auto_leg", action="store_true",
                   help="crop to the along-flow leg that crosses flotation, "
                        "using ITS_LIVE velocities. Splits out-and-back lines")
    p.add_argument("--baseline_km", type=float, default=20.0,
                   help="length of the onset reference stretch")
    p.add_argument("--min_baseline_km", type=float, default=8.0)
    p.add_argument("--no_onset", action="store_true",
                   help="skip the onset detector. Use where no floating ice "
                        "sits downflow of the window to take a baseline from, "
                        "e.g. a glacier calving at its grounding zone")
    p.add_argument("--search_lo", type=float, default=None)
    p.add_argument("--search_hi", type=float, default=None)
    p.add_argument("--margin_km", type=float, default=12.0,
                   help="half-width of the auto window around flotation")
    p.add_argument("--gz_lo", type=float, default=None,
                   help="InSAR GZ seaward edge, km. Reported, never used to search")
    p.add_argument("--gz_hi", type=float, default=None)
    p.add_argument("--normalised", action="store_true",
                   help="also run with standardised features")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--fig", default=None, help="write a figure to this path")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.quiet:
        warnings.filterwarnings("ignore")

    geoid_spec = args.geoid
    if geoid_spec is not None:
        try:
            geoid_spec = float(geoid_spec)
        except ValueError:
            pass          # treat as a BedMachine path

    opr = xopr.opr_access.OPRConnection(cache_dir=args.cache_dir)
    profile = GlacierProfile.from_xopr(
        args.collection, args.segment, opr=opr,
        frame_slice=parse_slice(args.frames),
        dx_km=args.dx_km, resample_interval=args.resample,
        geoid=geoid_spec,
    )

    if args.crop_lo is not None or args.crop_hi is not None:
        profile = profile.window(
            args.crop_lo if args.crop_lo is not None else profile.extent[0],
            args.crop_hi if args.crop_hi is not None else profile.extent[1],
        )

    leg = None
    if args.auto_leg:
        leg = select_flotation_leg(profile, margin_km=args.margin_km)
        if leg is None:
            raise SystemExit(
                "no along-flow leg crosses flotation with bed power to work "
                "from; crop by hand or drop --auto_leg"
            )
        profile = profile.window(*leg)

    if args.search_lo is not None and args.search_hi is not None:
        window = (args.search_lo, args.search_hi)
        how = "explicit"
    else:
        window = profile.flotation_window(args.margin_km)
        how = f"auto (flotation crossing +/- {args.margin_km:g} km)"

    print(f"\n{profile.source}")
    print(f"  samples       : {profile.n}  dx={profile.dx*1000:.0f} m")
    print(f"  extent        : {profile.extent[0]:.2f} - {profile.extent[1]:.2f} km")
    if leg is not None:
        print(f"  along-flow leg: {leg[0]:.2f} - {leg[1]:.2f} km  [auto]")
    g = profile.geoid_separation_m
    g_txt = (f"{float(g):.1f} m" if not np.ndim(g)
             else f"{np.nanmin(g):.1f} - {np.nanmax(g):.1f} m (sampled)")
    print(f"  geoid corr    : {g_txt}")
    print(f"  orientation   : x increases "
          f"{'landward' if profile.landward_sign() > 0 else 'seaward'}")
    print(f"  search window : {window[0]:.2f} - {window[1]:.2f} km  [{how}]")
    gaps = profile.nan_blocks()
    print(f"  bed-power gaps: {len(gaps)}"
          + (f"  {[(round(a,1), round(b,1)) for a, b in gaps[:4]]}" if gaps else ""))

    onset = None
    if not args.no_onset:
        onset = OnsetDetector(baseline_km=args.baseline_km,
                              min_baseline_km=args.min_baseline_km
                              ).detect(profile, window)
    changepoint = BOCPDDetector().detect(profile, window)
    results = ([onset] if onset is not None else []) + [
        changepoint, GradientDetector().detect(profile, window)]
    if args.normalised:
        results.append(BOCPDDetector(normalise=True).detect(profile, window))

    print()
    for r in results:
        print(f"  {r.summary()}")
        if "model_maps" in r.extra:
            per_model = "  ".join(f"{k}={v:.2f}"
                                  for k, v in r.extra["model_maps"].items())
            print(f"    per-model MAP: {per_model}")
        if r.detector == "onset":
            lo_b, hi_b = r.extra["baseline_extent_km"]
            print(f"    baseline {r.extra['baseline_dB']:.1f} dB "
                  f"+/- {r.extra['sigma_dB']:.2f} from {lo_b:.1f}-{hi_b:.1f} km")

    if onset is not None:
        width = transition_width_km(onset, changepoint,
                                    profile.landward_sign())
        print(f"\n  grounding point (onset)   : {onset.map_km:.2f} km")
        print(f"  transition width          : {width:.2f} km  "
              f"(onset -> steepest; wide means a diffuse zone)")
    else:
        print(f"\n  grounding point (changepoint): {changepoint.map_km:.2f} km"
              f"  [onset skipped]")

    if args.gz_lo is not None:
        hi = args.gz_hi if args.gz_hi is not None else args.gz_lo
        # The InSAR GZ is older than the radar and is a zone, not a point. The
        # grounding point is expected slightly downflow of its seaward edge, so
        # a small negative offset is the good case -- not landing inside.
        print(f"\n  InSAR GZ (reference only): {args.gz_lo:.2f} - {hi:.2f} km")
        print(f"  {'':24s}offset from seaward edge (negative = downflow)")
        for r in results:
            offset = r.map_km - args.gz_lo
            note = "downflow" if offset < 0 else "upflow of GZ seaward edge"
            print(f"    {r.detector:22s} MAP={r.map_km:7.2f} km  "
                  f"{offset:+6.2f} km  {note}")

    if args.fig:
        from xopr_gline.grounding.plotting import plot_detection
        gz = None
        if args.gz_lo is not None:
            gz = (args.gz_lo, args.gz_hi if args.gz_hi is not None else args.gz_lo)
        # Title is the provenance alone; how the window was set is printed
        # above and shown by the shaded band.
        plot_detection(profile, results, args.fig, gz=gz,
                       bathymetry=args.bathymetry)
        print(f"\n  figure: {args.fig}")


if __name__ == "__main__":
    main()
