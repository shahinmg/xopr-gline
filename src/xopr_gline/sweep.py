"""
Run grounding point detection over many segments and collect the results.

Every segment produces exactly one row carrying a status, so a failure is a
value to filter on rather than an exception that kills the sweep. Rows go to
JSONL as they finish, which makes a run resumable and lets tuning replay
without redownloading.
"""

import json
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

import xopr.opr_access

from .grounding import (BOCPDDetector, GlacierProfile, GradientDetector,
                        OnsetDetector, screen_profile, select_flotation_leg,
                        transition_width_km)
from .grounding.geoid import DEFAULT_BEDMACHINE

GRANULE = re.compile(r"^(\d{8})_(\d{2})_(\d{3})$")
USEABLE_LINES = Path("data/flight_lines/cresis_useable_flight_lines.gpkg")

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
                margin_km: float = 12.0,
                auto_leg: bool = True,
                screen: bool = True,
                headline: str = "onset",
                fig_dir=None,
                bathymetry=None,
                **extra) -> dict:
    """
    Detect a grounding point on one segment, returning a row whatever happens.

    Never raises: every failure lands in `status` and `reason` so a sweep can
    carry on and the outcome stays visible in the output table.
    """
    row = {"collection": collection, "segment": segment,
           "frames_requested": _frames_text(frames),
           "margin_km": margin_km,
           "status": "error", "reason": "", **extra}
    if not collection or (isinstance(collection, float) and np.isnan(collection)):
        row["status"] = "not_in_catalog"
        row["reason"] = "segment not found in the STAC catalog"
        return row

    opr = opr or xopr.opr_access.OPRConnection()
    try:
        profile = GlacierProfile.from_xopr(
            collection, segment, opr=opr, frames=frames,
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

    results = {}
    for name, detector in (("onset", OnsetDetector()),
                           ("bocpd", BOCPDDetector()),
                           ("gradient", GradientDetector())):
        try:
            results[name] = detector.detect(profile, window)
        except Exception as exc:
            row[f"{name}_reason"] = _reason(exc)
    row.update({f"{k}_km": v.map_km for k, v in results.items()})

    if not results:
        row["status"] = "no_detection"
        row["reason"] = row.get("bocpd_reason", "every detector failed")
        return row

    pick = results.get(headline) or results.get("bocpd") or next(iter(results.values()))
    row.update(_result_fields(pick, results, profile))
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
              **kwargs) -> gpd.GeoDataFrame:
    """
    Run every task in a thread pool and return the rows as a geodataframe.

    Work is I/O bound on the xOPR downloads, so threads are enough. Rows are
    appended to `out` as JSONL, and a rerun with resume=True skips segments
    already recorded there.
    """
    tasks = build_tasks() if tasks is None else tasks
    done = _completed(out) if resume else set()
    pending = tasks[~tasks["run_id"].isin(done)]

    opr = xopr.opr_access.OPRConnection(cache_dir=cache_dir)
    lock = threading.Lock()
    rows = []

    def work(task):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return run_segment(task.get("collection"), task["segment"],
                               task.get("frames"), opr=opr,
                               run_id=task["run_id"],
                               date=_iso(task.get("date")), **kwargs)

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

    if out is not None and resume:
        rows = _read_jsonl(out)
    return to_geodataframe(pd.DataFrame(rows))


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


def to_geodataframe(rows: pd.DataFrame) -> gpd.GeoDataFrame:
    """Rows as points on the grounding point, in EPSG:4326."""
    if not len(rows):
        return gpd.GeoDataFrame(rows, geometry=[], crs="EPSG:4326")
    blank = pd.Series(np.nan, index=rows.index)
    lon = pd.to_numeric(rows["lon"] if "lon" in rows else blank, errors="coerce")
    lat = pd.to_numeric(rows["lat"] if "lat" in rows else blank, errors="coerce")
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
    width = np.nan
    if "onset" in results and "bocpd" in results:
        try:
            width = transition_width_km(results["onset"], results["bocpd"],
                                        profile.landward_sign())
        except Exception:
            width = np.nan
    # Per-model MAPs, because the ensemble can outvote a model that was right:
    # on Helheim ifm lands within 1 km of the manual answer and fullcov 9 km
    # away, and the geometric mean follows fullcov. Their spread flags it.
    models = results["bocpd"].extra.get("model_maps", {}) if "bocpd" in results else {}
    return {
        **{f"bocpd_{name}_km": value for name, value in models.items()},
        "bocpd_model_spread_km": (max(models.values()) - min(models.values())
                                  if len(models) > 1 else np.nan),
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


def _read_jsonl(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _completed(out) -> set:
    """Runs already in the output, so a rerun picks up where it stopped."""
    return {row["run_id"] for row in _read_jsonl(out)} if out else set()
