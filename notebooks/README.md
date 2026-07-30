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
    --frames :5 \
    --geoid data/bedmachine/BedMachineGreenland-v5.nc \
    --fig data/figs/helheim_geoid.png
```

`--frames :5` is the subset used in `helheim_example_20080730_01_005.ipynb`.
Drop it for the whole segment, or `--frames -2:` for the last two frames.

## Petermann, 20100420_03

```bash
uv run python notebooks/run_grounding.py \
    --collection 2010_Greenland_DC8 --segment 20100420_03 \
    --frames 7:11 \
    --geoid data/bedmachine/BedMachineGreenland-v5.nc \
    --gz_lo 95.38 --gz_hi 97.89 \
    --fig data/figs/petermann_geoid.png
```

`--gz_lo/--gz_hi` are the InSAR grounding zone. They are printed for
comparison only and never steer the search — expect the grounding point
slightly downflow of the seaward edge (a small negative offset).

## Notes

- `--geoid` takes a number in m or a path to BedMachine, which is sampled
  along the flight line. Omit it (or pass `0`) for the uncorrected run.
- The search window defaults to the flotation crossing +/- `--margin_km`
  (12 km). `--search_lo/--search_hi` override it.
- `--normalised` adds a standardised-feature BOCPD run alongside the raw one.
  The raw run is the headline result.
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
