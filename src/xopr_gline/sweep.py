"""
Run grounding point detection over many segments and collect the results.

Every segment produces exactly one row carrying a status, so a failure is a
value to filter on rather than an exception that kills the sweep. Rows go to
JSONL as they finish, which makes a run resumable and lets tuning replay
without redownloading.

A status of "ok" only means a detector answered. Each row also carries quality
flags and the settings behind it, and each invocation is recorded in a sibling
.runs.jsonl, so a point on a map can be traced to the code that produced it.
Filter on `confidence` before using the geometry; see xopr_gline.validate.
"""

import json
import platform
import re
import subprocess
import threading
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

import xopr
import xopr.opr_access

from .grounding import (BOCPDDetector, GlacierProfile, GradientDetector,
                        OnsetDetector, screen_profile, select_flotation_leg,
                        transition_width_km)
from .grounding.features import (AMP_CUTOFF_KM, ELEV_CUTOFF_KM, SATURATE_M,
                                 default_features)
from .grounding.geoid import DEFAULT_BEDMACHINE

GRANULE = re.compile(r"^(\d{8})_(\d{2})_(\d{3})$")
USEABLE_LINES = Path("data/flight_lines/cresis_useable_flight_lines.gpkg")

# Pinned rather than derived. Left to itself the grid follows the bed-power
# sampling of whichever frames a run happens to load, which ran 121-277 m over
# the first Greenland sweep; every km-denominated setting downstream then means
# a different number of samples on every run and no two runs are comparable.
DEFAULT_DX_KM = 0.25

# plot_detection draws through pyplot, which keeps global state.
_PLOT_LOCK = threading.Lock()


# -- candidate segments ----------------------------------------------------
def useable_frames(path=USEABLE_LINES) -> pd.DataFrame:
    """
    The manually vetted flight lines as one row per granule.

    The gpkg splits some granules across several line parts; those collapse to
    a single frame here.
    """
    lines = gpd.read_file(path)
    parts = lines["Name"].astype(str).str.extract(GRANULE)
    if parts[0].isna().any():
        bad = lines["Name"][parts[0].isna()].unique()[:5]
        raise ValueError(f"unparseable granule names: {list(bad)}")
    frames = pd.DataFrame({
        "granule": lines["Name"].astype(str),
        "segment": parts[0] + "_" + parts[1],
        "frame": parts[2].astype(int),
        "date": pd.to_datetime(parts[0], format="%Y%m%d"),
    })
    return frames.drop_duplicates("granule").sort_values(
        ["segment", "frame"]).reset_index(drop=True)


def resolve_collections(segments: Sequence[str],
                        opr: Optional[xopr.opr_access.OPRConnection] = None,
                        batch: int = 25) -> pd.DataFrame:
    """
    Look each segment up in the STAC catalog to find its collection.

    The vetted flight lines name segments but not seasons. Segments the catalog
    does not hold come back with a null collection rather than being dropped.
    """
    opr = opr or xopr.opr_access.OPRConnection()
    segments = sorted(set(segments))
    found = []
    for i in range(0, len(segments), batch):
        items = opr.query_frames(segment_paths=segments[i:i + batch],
                                 exclude_geometry=True)
        if items is None or len(items) == 0:
            continue
        props = pd.json_normalize(items["properties"])
        found.append(pd.DataFrame({
            "segment": (props["opr:date"].astype(str) + "_"
                        + props["opr:segment"].map("{:02d}".format)).values,
            "collection": items["collection"].values,
            "frame": props["opr:frame"].values,
        }))

    if not found:
        return pd.DataFrame({"segment": segments, "collection": None,
                             "catalog_frames": 0})
    catalog = pd.concat(found, ignore_index=True)
    per_segment = catalog.groupby("segment").agg(
        collection=("collection", "first"),
        catalog_frames=("frame", "nunique"),
    ).reset_index()
    return pd.DataFrame({"segment": segments}).merge(per_segment, how="left",
                                                     on="segment")


def build_tasks(frames: Optional[pd.DataFrame] = None,
                opr: Optional[xopr.opr_access.OPRConnection] = None,
                pad_frames: Optional[int] = 1,
                max_gap: int = 1) -> pd.DataFrame:
    """
    One row per run: a contiguous cluster of vetted frames and its collection.

    A segment can cross several glaciers, so clusters more than max_gap frames
    apart become separate runs rather than one over-long profile. pad_frames
    widens each cluster so the flotation window has room; None loads the whole
    segment as a single run.
    """
    frames = useable_frames() if frames is None else frames
    catalog = resolve_collections(frames["segment"].unique(), opr=opr)
    available = dict(zip(catalog["segment"], catalog["catalog_frames"]))

    rows = []
    for segment, group in frames.groupby("segment"):
        numbers = sorted(group["frame"])
        last = available.get(segment)
        last = int(last) if last is not None and np.isfinite(last) else None
        for i, cluster in enumerate(_clusters(numbers, max_gap), 1):
            rows.append({
                "run_id": f"{segment}#{i}",
                "segment": segment,
                "vetted_frames": cluster,
                "frames": _pad(cluster, pad_frames, last),
                "date": group["date"].min(),
            })

    tasks = pd.DataFrame(rows).merge(catalog, on="segment", how="left")
    return tasks.sort_values("run_id").reset_index(drop=True)


def _clusters(numbers: Sequence[int], max_gap: int = 1) -> list:
    """Split frame numbers wherever they jump by more than max_gap."""
    groups = [[numbers[0]]]
    for n in numbers[1:]:
        if n - groups[-1][-1] > max_gap:
            groups.append([])
        groups[-1].append(n)
    return groups


def _pad(cluster: Sequence[int], pad_frames: Optional[int],
         last: Optional[int]) -> Optional[list]:
    """Widen a cluster by pad_frames on each side, clipped to the segment."""
    if pad_frames is None or last is None:
        return None
    lo = max(1, min(cluster) - pad_frames)
    hi = min(last, max(cluster) + pad_frames)
    return list(range(lo, hi + 1))


# -- one segment -----------------------------------------------------------
def run_segment(collection: Optional[str], segment: str,
                frames: Optional[Sequence[int]] = None,
                *,
                opr: Optional[xopr.opr_access.OPRConnection] = None,
                geoid=DEFAULT_BEDMACHINE,
                resample_interval: str = "2s",
                dx_km: Optional[float] = DEFAULT_DX_KM,
                margin_km: float = 12.0,
                auto_leg: bool = True,
                screen: bool = True,
                headline: str = "bocpd",
                pad_km: float = 4.0,
                snap_to_terminus: bool = False,
                amp_cutoff_km: float = AMP_CUTOFF_KM,
                elev_cutoff_km: float = ELEV_CUTOFF_KM,
                saturate_m: float = SATURATE_M,
                fig_dir=None,
                bathymetry=None,
                **extra) -> dict:
    """
    Detect a grounding point on one segment, returning a row whatever happens.

    Never raises: every failure lands in `status` and `reason` so a sweep can
    carry on and the outcome stays visible in the output table.

    Every tuning knob is written back onto the row, so a result can be
    reproduced from the output alone without knowing what the sweep was
    invoked with.
    """
    row = {"collection": collection, "segment": segment,
           "frames_requested": _frames_text(frames),
           "margin_km": margin_km, "dx_km": dx_km, "pad_km": pad_km,
           "headline": headline, "snap_to_terminus": snap_to_terminus,
           "amp_cutoff_km": amp_cutoff_km, "elev_cutoff_km": elev_cutoff_km,
           "saturate_m": saturate_m, "resample_interval": resample_interval,
           "status": "error", "reason": "", **extra}
    if not collection or (isinstance(collection, float) and np.isnan(collection)):
        row["status"] = "not_in_catalog"
        row["reason"] = "segment not found in the STAC catalog"
        return row

    opr = opr or xopr.opr_access.OPRConnection()
    try:
        profile = GlacierProfile.from_xopr(
            collection, segment, opr=opr, frames=frames, dx_km=dx_km,
            resample_interval=resample_interval, geoid=geoid,
        )
    except Exception as exc:                      # any xOPR or layer failure
        row["status"] = "no_profile"
        row["reason"] = _reason(exc)
        return row

    row.update(_profile_fields(profile))

    # screen_profile picks the along-flow leg itself and judges on it, so the
    # profile must NOT be cropped first or the leg gets selected twice.
    leg = None
    if screen:
        try:
            verdict = screen_profile(profile, margin_km=margin_km)
        except Exception as exc:
            row["status"] = "screen_failed"
            row["reason"] = _reason(exc)
            return row
        row.update(_screen_fields(verdict))
        leg = verdict.leg_km
        if not verdict.usable:
            row["status"] = "screened_out"
            row["reason"] = verdict.reason
            return row
    elif auto_leg:
        try:
            leg = select_flotation_leg(profile, margin_km=margin_km)
        except Exception as exc:
            row["reason"] = _reason(exc)

    if auto_leg:
        if leg is None:
            row["status"] = "no_leg"
            row["reason"] = row["reason"] or "no along-flow leg crosses flotation"
            return row
        profile = profile.window(*leg)
        row["leg_lo_km"], row["leg_hi_km"] = leg

    try:
        window = profile.flotation_window(margin_km)
    except Exception as exc:
        row["status"] = "no_window"
        row["reason"] = _reason(exc)
        return row
    row["window_lo_km"], row["window_hi_km"] = window
    # The elevation's own estimate of where grounding begins, independent of
    # anything the bed-echo detectors do. Recorded so a pick sitting on the
    # seaward clamp can be judged against the geometry after the fact.
    try:
        row["flotation_crossing_km"] = profile.flotation_crossing_km()
    except ValueError:
        row["flotation_crossing_km"] = np.nan

    features = default_features(amp_cutoff_km, elev_cutoff_km, saturate_m)
    by_name = {f.name: f for f in features}
    results = {}
    for name, detector in (
        ("onset", OnsetDetector(feature=by_name["amp"])),
        ("bocpd", BOCPDDetector(features=features, pad_km=pad_km)),
        ("gradient", GradientDetector(feature=by_name["dA"])),
    ):
        try:
            results[name] = detector.detect(profile, window)
        except Exception as exc:
            row[f"{name}_reason"] = _reason(exc)
    row.update({f"{k}_km": v.map_km for k, v in results.items()})

    if not results:
        row["status"] = "no_detection"
        row["reason"] = row.get("bocpd_reason", "every detector failed")
        return row

    # The headline must be one method across the whole output or the geometry
    # column mixes detectors and the table is not one dataset. A fallback is
    # still better than no row, but it is recorded rather than silent.
    pick = results.get(headline)
    row["headline_fallback"] = pick is None
    if pick is None:
        pick = results.get("bocpd") or next(iter(results.values()))
    row.update(_result_fields(pick, results, profile))
    # Snap first so the confidence knows about it, apply after so it overrides
    # the position the detector reported.
    snapped = _snap_to_terminus(pick, profile, snap=snap_to_terminus)
    row.update(_qc_fields(pick, results, profile,
                          snapped=snapped["snapped_to_terminus"]))
    row.update(snapped)
    row["status"] = "ok"

    if fig_dir is not None:
        row.update(_write_figure(profile, results, fig_dir,
                                 row.get("run_id") or segment,
                                 bathymetry if bathymetry is not None else geoid))
    return row


def _write_figure(profile, results, fig_dir, name, bathymetry) -> dict:
    """
    Write the four-panel detection figure for one run.

    plot_detection drives pyplot, whose state is global, so figures are drawn
    one at a time however many threads the sweep runs. Drawing is seconds
    against a download of tens, so the lock costs little.
    """
    from .grounding.plotting import plot_detection

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    # Same order as run_grounding.py: onset first, then the changepoint it is
    # measured against, then the rest.
    ordered = [results[k] for k in ("onset", "bocpd", "gradient") if k in results]
    try:
        with _PLOT_LOCK:
            plot_detection(profile, ordered, str(path), bathymetry=bathymetry)
    except Exception as exc:
        return {"figure_reason": _reason(exc)}
    return {"figure": str(path)}


# -- the sweep -------------------------------------------------------------
def run_sweep(tasks: Optional[pd.DataFrame] = None,
              out: Optional[Path] = None,
              workers: int = 8,
              resume: bool = True,
              cache_dir: Optional[str] = None,
              progress: bool = True,
              invocation: Optional[dict] = None,
              **kwargs) -> gpd.GeoDataFrame:
    """
    Run every task in a thread pool and return the rows as a geodataframe.

    Work is I/O bound on the xOPR downloads, so threads are enough. Rows are
    appended to `out` as JSONL, and a rerun with resume=True skips segments
    already recorded there.

    Each invocation is recorded in a sibling `.runs.jsonl` and every row is
    stamped with its sweep_id, so a point in the output can be traced back to
    the settings and the commit that produced it. See sweep_log().
    """
    tasks = build_tasks() if tasks is None else tasks
    done = _completed(out) if resume else set()
    pending = tasks[~tasks["run_id"].isin(done)]

    sweep_id = uuid.uuid4().hex[:8]
    started = pd.Timestamp.utcnow()
    log_run(out, sweep_id=sweep_id, event="start", started=str(started),
            n_tasks=int(len(pending)), n_skipped=int(len(tasks) - len(pending)),
            workers=workers, resume=resume, cache_dir=cache_dir,
            params=dict(kwargs), invocation=invocation)

    opr = xopr.opr_access.OPRConnection(cache_dir=cache_dir)
    lock = threading.Lock()
    rows = []

    def work(task):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return run_segment(task.get("collection"), task["segment"],
                               task.get("frames"), opr=opr,
                               run_id=task["run_id"], sweep_id=sweep_id,
                               date=_iso(task.get("date")), **kwargs)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(work, t): t["run_id"]
                       for t in pending.to_dict("records")}
            for i, future in enumerate(as_completed(futures), 1):
                row = future.result()
                rows.append(row)
                if out is not None:
                    with lock:
                        with open(out, "a") as fh:
                            fh.write(json.dumps(row, default=str) + "\n")
                if progress:
                    print(f"[{i}/{len(futures)}] {row['run_id']:18s} "
                          f"{row['status']:14s} {row.get('reason', '')[:60]}",
                          flush=True)
    finally:
        # Logged even on an interrupt, so a partial sweep still leaves a record
        # of what it was doing.
        counts = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
        log_run(out, sweep_id=sweep_id, event="finish",
                finished=str(pd.Timestamp.utcnow()),
                duration_s=round((pd.Timestamp.utcnow() - started)
                                 .total_seconds(), 1),
                n_completed=len(rows), statuses=counts)

    if out is not None and resume:
        rows = _read_jsonl(out)
    return to_geodataframe(pd.DataFrame(rows))


# -- run log ---------------------------------------------------------------
def runs_path(out) -> Optional[Path]:
    """Where the invocation log for an output file lives."""
    return None if out is None else Path(out).with_suffix(".runs.jsonl")


def log_run(out, **fields) -> Optional[Path]:
    """
    Append one record about a sweep invocation.

    Append-only and one line per event, so an interrupted sweep still leaves a
    start record and a resume adds to the history rather than replacing it.
    """
    path = runs_path(out)
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"logged": str(pd.Timestamp.utcnow()), **_environment(), **fields}
    with open(path, "a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    return path


def sweep_log(out) -> pd.DataFrame:
    """The invocation history for an output file, newest last."""
    path = runs_path(out)
    return pd.DataFrame(_read_jsonl(path) if path else [])


def _environment() -> dict:
    """Code version and host, so a run can be reproduced or explained."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True,
                                    timeout=5).stdout.strip())
    except Exception:
        sha, dirty = None, None
    return {"commit": sha or None, "dirty": dirty,
            "xopr_version": getattr(xopr, "__version__", None),
            "host": platform.node()}


def forget(out, statuses: Sequence[str] = ("no_profile", "not_in_catalog",
                                           "no_detection", "error"),
           reason_contains: Optional[str] = None) -> int:
    """
    Drop rows with these statuses from the JSONL so a resume retries them.

    A sweep already running holds the code it started with, so fixing a bug
    means forgetting the runs that failed for the old reason. Returns how many
    rows were dropped.
    """
    out = Path(out)
    rows = _read_jsonl(out)
    keep = [r for r in rows
            if r.get("status") not in statuses
            or (reason_contains is not None
                and reason_contains not in r.get("reason", ""))]
    with open(out, "w") as fh:
        for row in keep:
            fh.write(json.dumps(row, default=str) + "\n")
    return len(rows) - len(keep)


def forget_runs(out, run_ids: Sequence[str]) -> int:
    """
    Drop these specific runs from the JSONL so a resume redoes them.

    `forget` works by status, which is no help after a fix that changes correct
    rows rather than failing ones. Returns how many rows were dropped.
    """
    out = Path(out)
    wanted = set(run_ids)
    rows = _read_jsonl(out)
    keep = [r for r in rows if r.get("run_id") not in wanted]
    with open(out, "w") as fh:
        for row in keep:
            fh.write(json.dumps(row, default=str) + "\n")
    return len(rows) - len(keep)


def to_geodataframe(rows: pd.DataFrame,
                    detector: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Rows as points on the grounding point, in EPSG:4326.

    `detector` puts a named detector's own pick in the geometry instead of the
    reported one, e.g. "bocpd" for the raw changepoint on rows whose reported
    position was moved onto the terminus. The km column moves with it, so the
    geometry and `map_km` never disagree.
    """
    if not len(rows):
        return gpd.GeoDataFrame(rows, geometry=[], crs="EPSG:4326")

    rows = rows.copy()
    lat_col, lon_col = "lat", "lon"
    if detector is not None:
        lat_col, lon_col = f"{detector}_lat", f"{detector}_lon"
        missing = [c for c in (lat_col, lon_col, f"{detector}_km")
                   if c not in rows]
        if missing:
            raise ValueError(
                f"no {detector!r} coordinates in these rows (missing "
                f"{missing}). Sweeps written before per-detector coordinates "
                f"were recorded only carry the reported position."
            )
        rows["map_km"] = rows[f"{detector}_km"]
        rows["geometry_source"] = detector

    blank = pd.Series(np.nan, index=rows.index)
    lon = pd.to_numeric(rows[lon_col] if lon_col in rows else blank,
                        errors="coerce")
    lat = pd.to_numeric(rows[lat_col] if lat_col in rows else blank,
                        errors="coerce")
    # Rows that found nothing get a null geometry, not POINT (NaN NaN).
    points = gpd.points_from_xy(lon, lat)
    geometry = gpd.GeoSeries(points, index=rows.index).where(
        lon.notna() & lat.notna())
    return gpd.GeoDataFrame(rows, geometry=geometry, crs="EPSG:4326")


# -- helpers ---------------------------------------------------------------
def _frames_text(frames) -> Optional[str]:
    if frames is None:
        return None
    frames = list(frames)
    return f"{min(frames):03d}-{max(frames):03d}" if frames else None


def _reason(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


def _iso(value):
    return None if value is None or pd.isna(value) else str(pd.Timestamp(value).date())


def _profile_fields(profile: GlacierProfile) -> dict:
    source = profile.source
    return {
        "n_samples": profile.n,
        "dx_m": profile.dx * 1000.0,
        "extent_lo_km": profile.extent[0],
        "extent_hi_km": profile.extent[1],
        "n_frames": None if source is None else source.n_frames,
        "acquired": None if source is None or source.date is None
                    else str(source.date),
    }


def _screen_fields(verdict) -> dict:
    return {
        "terminus_km": verdict.terminus_km,
        "min_hab_m": verdict.min_hab_m,
        "median_angle_deg": verdict.median_angle_deg,
        "monotonic_fraction": verdict.monotonic_fraction,
        "ice_fraction": verdict.ice_fraction,
        "distance_to_terminus_km": verdict.distance_to_terminus_km,
    }


def _result_fields(pick, results, profile) -> dict:
    ci68 = pick.credible_interval(0.68)
    ci95 = pick.credible_interval(0.95)
    frame = pick.frame
    width = (_transition(results, profile)
             if "onset" in results and "bocpd" in results else np.nan)
    # Per-model MAPs, because the ensemble can outvote a model that was right:
    # on Helheim ifm lands within 1 km of the manual answer and fullcov 9 km
    # away, and the geometric mean follows fullcov. Their spread flags it.
    models = results["bocpd"].extra.get("model_maps", {}) if "bocpd" in results else {}
    return {
        **{f"bocpd_{name}_km": value for name, value in models.items()},
        "bocpd_model_spread_km": (max(models.values()) - min(models.values())
                                  if len(models) > 1 else np.nan),
        # Each detector's own coordinates, not just the headline's. Without
        # these the only geometry in the table is whichever pick was promoted,
        # and any other detector's answer cannot be mapped after the fact.
        **{f"{name}_{axis}": getattr(result, axis)
           for name, result in results.items()
           for axis in ("lat", "lon")},
        "headline_detector": pick.detector,
        "map_km": pick.map_km,
        "lat": pick.lat,
        "lon": pick.lon,
        "frame": None if frame is None else frame.granule,
        "frames_used": ",".join(str(f.frame) for f in pick.frames_used),
        "ci68_lo_km": ci68[0], "ci68_hi_km": ci68[1],
        "ci95_lo_km": ci95[0], "ci95_hi_km": ci95[1],
        "transition_width_km": width,
    }


def _qc_fields(pick, results, profile, snapped: bool = False) -> dict:
    """
    Flags that say how much to trust the row, so the output can be filtered.

    The first Greenland sweep had no way to separate a real detection from a
    boundary artifact, which is what made a table of 99 "ok" rows unusable.
    """
    picks = {k: v.map_km for k, v in results.items()}
    models = (results["bocpd"].extra.get("model_maps", {})
              if "bocpd" in results else {})
    ci95 = pick.credible_interval(0.95)

    # Measured against the calving front, not the nearest ice/water crossing:
    # bergs in the fjord each make a crossing, and a pick sitting 2 km inland of
    # the front read as 0.2 km from a berg drifting past it. What is wanted is
    # the changepoint of upflow ice relative to the terminus.
    front = profile.calving_front_km()
    gap = abs(pick.map_km - front) if front is not None else np.nan
    # Signed, so a pick beyond the front is distinguishable from one inland of
    # it. Positive is inland, which is the only side a grounding point can be.
    offset = ((pick.map_km - front) * profile.landward_sign()
              if front is not None else np.nan)
    # A pick displaced off the terminus by the trim is still a pick at the
    # terminus, so the tolerance grows to whatever the trim could have moved it.
    reach = max(2.0 * profile.dx, float(pick.extra.get("edge_reach_km", 0.0)))
    fields = {
        "edge_pinned": bool(pick.extra.get("edge_pinned", False)),
        "at_window_edge": bool(pick.extra.get("at_window_edge", False)),
        "edge_distance_km": pick.extra.get("edge_distance_km", np.nan),
        # CLAUDE.md step 3: record when the grounding point is the terminus
        # rather than sitting somewhere up a floating tongue.
        "at_terminus": bool(np.isfinite(gap) and gap <= reach),
        "terminus_gap_km": gap,
        "calving_front_km": front if front is not None else np.nan,
        "terminus_offset_km": offset,
        "detector_spread_km": (max(picks.values()) - min(picks.values())
                               if len(picks) > 1 else np.nan),
        "bocpd_model_spread_km": (max(models.values()) - min(models.values())
                                  if len(models) > 1 else np.nan),
        "posterior_entropy": pick.posterior_entropy,
        "ci95_width_km": ci95[1] - ci95[0],
        "onset_upflow": bool(("onset" in results and "bocpd" in results)
                             and _transition(results, profile) < 0),
        "snapped_to_terminus": bool(snapped),
    }
    fields["confidence"] = _confidence(fields)
    return fields


def _snap_to_terminus(pick, profile, snap: bool = True,
                      tolerance_km: float = 0.75,
                      max_crossing_km: float = 5.0) -> dict:
    """
    Report the calving front when that is what the pick is really saying.

    flotation_window clamps the search at the terminus, because the grounding
    point cannot lie seaward of the front, and the models cannot put their
    maximum on that clamp: the trim reserves the first few samples. So a
    tidewater glacier grounded at its front comes back a fixed ~1.75 km inland,
    on every glacier, which is a bias rather than a measurement.

    A pileup against that clamp is on its own only evidence that the posterior
    wanted to be at or beyond the seaward limit, not that the ice is afloat
    there: a changepoint model piles mass at a truncation whether or not a real
    change lives on it. So the geometry has to agree independently, through the
    flotation crossing, before the point is moved. Where it does not, the pick
    is left alone and stays edge_pinned and low confidence, which is the honest
    reading: the window truncated a transition that lies further inland.

    The detector's own value is kept in raw_map_km, so nothing is lost.
    """
    edge = pick.extra.get("pinned_edge")
    blank = {"snapped_to_terminus": False}
    if not snap or edge is None:
        return blank

    edge_km = pick.extra.get("pinned_edge_km", np.nan)
    crossings = profile.terminus_crossings_km()
    at_edge = [t for t in crossings if abs(t - edge_km) <= tolerance_km]
    if not at_edge:
        return blank

    # The elevation says where grounding begins; only snap where it puts that
    # near the front too. Measured over the first Greenland sweep, 48 of 54
    # snapped rows had it within 5 km, and the other 6 up to 12.5 km inland.
    try:
        crossing = profile.flotation_crossing_km()
    except ValueError:
        return blank
    if abs(crossing - min(at_edge, key=lambda t: abs(t - edge_km))) > max_crossing_km:
        return {"snapped_to_terminus": False,
                "snap_declined_km": float(abs(crossing - edge_km))}

    terminus = min(at_edge, key=lambda t: abs(t - edge_km))
    lat, lon = profile.position_at(terminus)
    moved = {
        "snapped_to_terminus": True,
        "raw_map_km": pick.map_km,
        "map_km": float(terminus),
        "lat": lat,
        "lon": lon,
        "terminus_gap_km": 0.0,
        "at_terminus": True,
    }

    # The frame is named from the position, so moving the position can move it
    # across a frame seam. Only matters where the window spans two frames, but
    # a row naming a frame that does not contain its own point is wrong.
    source = profile.source
    if source is not None:
        span = source.frame_at_km(terminus)
        if span is not None:
            moved["frame"] = span.granule
    return moved


def _confidence(qc: dict) -> str:
    """
    Three-way verdict from the QC flags, for filtering the output.

    low: an arbitrary boundary or a detector disagreement decided the answer.
    medium: the detectors agree only loosely, the pick sits at a search limit,
    or it is the calving front. high: everything agrees within a couple of km
    away from the edges.

    A snapped row is not "low": the position is the calving front, which is
    measured, and what is unresolved is only whether a distinct grounding point
    sits inland of it. That is medium, and it stays low if the detectors also
    disagree.
    """
    spread = np.nanmax([qc["detector_spread_km"],
                        qc["bocpd_model_spread_km"], 0.0])
    if spread > 5.0 or qc["onset_upflow"]:
        return "low"
    if qc.get("snapped_to_terminus"):
        return "medium"
    if qc["edge_pinned"]:
        return "low"
    if qc["at_window_edge"] or spread > 2.0:
        return "medium"
    return "high"


def _transition(results, profile) -> float:
    try:
        return transition_width_km(results["onset"], results["bocpd"],
                                   profile.landward_sign())
    except Exception:
        return np.nan


def _read_jsonl(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _completed(out) -> set:
    """Runs already in the output, so a rerun picks up where it stopped."""
    return {row["run_id"] for row in _read_jsonl(out)} if out else set()
