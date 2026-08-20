# notebooks

Driver scripts and exploratory notebooks for grounding point detection.

`run_grounding.py` is the entry point: give it a collection and segment and it
builds a `GlacierProfile` from xOPR, runs the onset / BOCPD / gradient
detectors, and prints the grounding point. `--help` lists every knob.

Run from the repo root — `--geoid` and `--fig` paths below are relative.

## Helheim, 20080730_01

```bash
uv run python notebooks/run_grounding.py \
    --collection 2008_Greenland_TO --segment 20080730_01 \
    --frames 3:5 --crop_lo 62 --margin_km 3.5 --no_onset \
    --geoid data/bedmachine/BedMachineGreenland-v5.nc \
    --bathymetry data/bedmachine/BedMachineGreenland-v5.nc \
    --fig data/figs/helheim_geoid.png
```

This reproduces `data/figs/helheim_geoid_original_20250728.png`, the
pre-refactor figure: shelf, calving front, and the changepoint 1.3 km inboard
of it.

Frames 004-005 are the two that hold the grounding zone. `--frames :5` is the
subset used in `helheim_example_20080730_01_005.ipynb` and gives the same
picture with the origin 101 km further back. `--frames=-2:` is *not* the
grounding zone: the last two frames of this segment are 015-016, out over the
fjord, where the median thickness is 14 m.

`--crop_lo 62` drops the inland leg due to large amount of missing data.

`--no_onset` is required. Helheim calves at its grounding zone, so no floating
ice sits downflow of the window for a bed-power baseline. 

## Petermann, 20100420_03

```bash
uv run python notebooks/run_grounding.py \
    --collection 2010_Greenland_DC8 --segment 20100420_03 \
    --frames 7:11 \
    --geoid data/bedmachine/BedMachineGreenland-v5.nc \
    --bathymetry data/bedmachine/BedMachineGreenland-v5.nc \
    --gz_lo 95.38 --gz_hi 97.89 \
    --fig data/figs/petermann_geoid.png
```

`--gz_lo/--gz_hi` are the InSAR grounding zone.

## Greenland sweep

```bash
uv run python notebooks/run_sweep.py --workers 10 --out data/sweeps/greenland_v2.jsonl
```

Runs every vetted flight line in
`data/flight_lines/cresis_useable_flight_lines.gpkg` and writes a geopackage of
grounding points beside the JSONL. Pick a new `--out` per experiment: a sweep
resumes into an existing JSONL and refuses to overwrite an existing geopackage.

### Reading the output

Every row carries the settings that produced it and a set of quality flags, so
the geopackage is meant to be filtered rather than taken whole:

| column | meaning |
|---|---|
| `confidence` | `high` / `medium` / `low`, from the flags below |
| `edge_pinned` | the search boundary decided the answer, not the data. Never trust these |
| `at_window_edge` | the change is at or beyond a search limit |
| `at_terminus` | the grounding point *is* the calving front |
| `snapped_to_terminus` | the position was moved onto the front; see below |
| `raw_map_km` | the detector's own position, kept whenever a row was snapped |
| `detector_spread_km` | how far apart onset, BOCPD and gradient landed |
| `bocpd_model_spread_km` | how far apart the IFM and FullCov likelihoods landed |
| `posterior_entropy` | 0 is a spike, 1 is flat. A spike on a boundary is a warning, not a result |
| `sweep_id` | which invocation produced the row; look it up in `*.runs.jsonl` |

### The terminus, the window, and the pick

Three separate things, easily conflated:

1. **The search window** is clamped at the terminus by `flotation_window`,
   because the grounding point cannot lie seaward of the front. This only
   bounds where a detector may look; it moves nothing.
2. **A pick pinned against that clamp** means the posterior wanted to sit at or
   beyond the seaward limit. On its own that is a statement about the window,
   not about the ice: a changepoint model piles mass at a truncation whether or
   not a real change lives there. It is reported as `edge_pinned` with
   `terminus_gap_km`.
3. **Whether the ice is actually afloat at the front** is answered by
   `flotation_crossing_km`, the elevation's own estimate of where grounding
   begins, which knows nothing about the bed echo. Over the first sweep it sat
   a median 1.5 km from the front, but up to 12.5 km inland on a handful of
   runs.

The geometry is always what a detector found. `--snap` opts in to moving a
pinned pick onto the front (recording `snapped_to_terminus` and keeping
`raw_map_km`), and even then only where `flotation_crossing_km` agrees within
5 km; otherwise it declines and records `snap_declined_km`. Off by default:
compare `map_km` against `flotation_crossing_km` and decide for yourself.

### Geopackages per detector

Every row carries `{detector}_lat` / `{detector}_lon` for onset, bocpd and
gradient, so any detector's answer can be mapped, not just the headline. The
sweep writes `<out>_bocpd.gpkg` alongside the main one; `--also_gpkg` chooses
which. In code, `to_geodataframe(rows, detector="bocpd")` does the same and
moves `map_km` with the geometry so the two cannot disagree.

`xopr_gline.validate` scores a sweep without eyeballing a map:
`repeat_consistency` measures how far apart repeat visits to one glacier land,
`compare_to_reference` takes any external layer such as an InSAR grounding
zone, and `scoreboard` reduces a sweep to the handful of numbers to compare two
runs on.

### Cost

Memory is the limit, not CPU: every worker holds a run's full radar arrays, and
`--workers 8 --fig_dir ...` was OOM-killed at 23 GB on a 23 GB machine. Draw
figures in a second pass rather than alongside detection — the plot lock
serialises drawing anyway, so threads waiting for it sit on their data doing
nothing. Four workers without figures is the safe default.

### Run log

Each invocation appends to `<out>.runs.jsonl`: the settings, the commit, the
host, the task counts and the status counts, with a start and a finish record
per sweep. `xopr_gline.sweep.sweep_log(out)` reads it back. Rows carry the
matching `sweep_id`, so a point on a map can be traced to the code and settings
behind it.

The unit of work is a *run*, not a segment: a contiguous cluster of vetted
frames padded by `--pad_frames`. One segment can cross several glaciers
(`19970528_01` has frames 6-7, 12-14 and 20), so clusters more than a frame
apart become separate runs. That also cuts the download from 3693 frames to
724.

Every run produces a row carrying a `status`, so failures are a column to
filter on rather than a crash: `ok`, `screened_out`, `no_profile`, `no_leg`,
`no_window`, `no_detection`, `not_in_catalog`. Rows are appended as they finish
and reruns skip what is already there, so an interrupted sweep resumes.

`--margin_km` is the one setting worth thinking about. The window is the
flotation crossing +/- that much, and the answer moves with it: Helheim gives
94.90 km at 5 km and 85.94 km at 12, Petermann 98.24 km at 5 and 95.67 km at 12
(the InSAR GZ is 95.38-97.89). Every row records the window it used.

`--dx_km` is pinned at 0.25 km. Left to itself the grid follows the bed-power
sampling of whichever frames a run loaded, which ranged 121-277 m over the
first sweep, so every km-denominated setting downstream meant a different
number of samples on every run. Pin it before comparing anything.

`--pad_km` (4 km) is context the BOCPD models see but do not report on, so the
edge transient every likelihood produces lands in the padding instead of
becoming the answer. It stops at the profile ends and at the calving front,
since past the front the ice/water contrast is a far stronger changepoint than
the grounding transition. Where the pad is blocked the old trim still applies
and the row is flagged `edge_pinned`. `--pad_km 0` restores the old behaviour.

`--elev_cutoff_km` (71.6) is inherited from the Petermann CSV era and is worth
sweeping: it is a low-pass wavelength half the length of a typical profile,
applied to the channel BOCPD keys on. Run it at a few values into separate
`--out` files and compare with `validate.scoreboard`.

## Notes

- `--geoid` takes a number in m or a path to BedMachine, which is sampled
  along the flight line. Omit it (or pass `0`) for the uncorrected run.
- `--bathymetry` draws the seabed under the floating ice from BedMachine's bed,
  sampled along the same line. It is a figure setting only: no detector sees it
  and the radar picks are untouched. Omit it and the seabed is drawn flat at its
  depth at the grounding point, which on Helheim misses a fjord that runs from
  -600 m at the grounding zone to -824 m two km out. Radar and BedMachine
  disagree by a median 52 m over Helheim's grounded leg but only 15 m near the
  grounding point, so the handover there is barely visible; it is drawn raw
  rather than tied to the radar bed.
- The search window defaults to the flotation crossing +/- `--margin_km`
  (12 km). `--search_lo/--search_hi` override it.
- `--normalised` adds a standardised-feature BOCPD run alongside the raw one.
  The raw run is the headline result.
- `--no_onset` drops the onset detector, for transects with no shelf downflow
  of the window. Without it those runs stop with "floating reference is 0.0 km".
  That message is now correct rather than incidental: the baseline is taken
  from floating *ice* only. Past the calving front thickness collapses to ~0,
  so the flotation residual reduces to the water surface and the bright sea
  return used to set the baseline tens of dB too high, which put the onset
  exactly on the terminus in 28 of the first sweep's 44 onset rows.
- The flotation residual channel is referenced to sea level and its far field
  is compressed with `--saturate_m` (100 m). Unsaturated, the residual climbs
  past 1000 m inland while the transition lives within tens of metres of zero,
  and that ramp sets the covariance and pulls the changepoint onto bed
  topography.
- `frames`, `position` and `frames searched` are the provenance a geodataframe
  row needs: the granules loaded, the grounding point's lat/lon, and which
  frames the search window touched.
- Along-track km are relative to the start of the frame slice, so they only
  mean anything alongside the `--frames` used.
- `data/**` is gitignored; figures written there stay local.

## Other files

| file | what |
|---|---|
| `01_search_flight_lines.py` | STAC query for OPR lines crossing Greenland termini |
| `bocpd_petermann.py` | original Petermann-only offline BOCPD script, pre-refactor |
| `helheim_example_*.ipynb` | pre-refactor Helheim walkthroughs; no geoid correction |
| `petermann_grounding_zone_comp*.ipynb` | grounding zone comparison scratch work |
| `bulk_processing_gline.ipynb` | batch runs over many segments |
| `emperical_grounding_point.ipynb` | erf topography model fitting |
