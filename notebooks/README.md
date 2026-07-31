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

`--gz_lo/--gz_hi` are the InSAR grounding zone. They are printed for
comparison only and never steer the search — expect the grounding point
slightly downflow of the seaward edge (a small negative offset).

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
- Every run prints a `terminus check`, saying whether the grounding point is
  the calving front. Helheim comes back `grounded_terminus` — it really does
  ground 1.3 km inboard of its face. `terminus_artifact` means the ice behind
  the front is afloat, so the detector found the ice/water contrast rather than
  a grounding transition, and the result should not be used.
- The search window defaults to the flotation crossing +/- `--margin_km`
  (12 km). `--search_lo/--search_hi` override it.
- `--normalised` adds a standardised-feature BOCPD run alongside the raw one.
  The raw run is the headline result.
- `--no_onset` drops the onset detector, for transects with no shelf downflow
  of the window. Without it those runs stop with "floating reference is 0.0 km".
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
