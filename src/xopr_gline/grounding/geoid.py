"""
Geoid and bed sampling from BedMachine.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pyproj
import xarray as xr

DEFAULT_BEDMACHINE = Path("data/bedmachine/BedMachineGreenland-v5.nc")
BEDMACHINE_CRS = "EPSG:3413"


def sample_bedmachine(lat, lon,
                      variable: str = "geoid",
                      path: Union[str, Path, None] = None,
                      crs: str = BEDMACHINE_CRS) -> np.ndarray:
    """
    A BedMachine variable at each lat/lon, in m.

    Nearest-neighbour on BedMachine's 150 m grid. Note the datums differ:
    'geoid' is height above the WGS84 ellipsoid, while 'bed' and 'surface' are
    heights above that geoid, so adding the geoid puts them on the ellipsoid
    with xOPR's elevations.

    Parameters
    ----------
    lat, lon: array-like
        Point coordinates in degrees.

    variable: str
        Variable to sample, e.g. 'geoid' or 'bed'.

    path: str or Path or None
        BedMachine NetCDF. Defaults to data/bedmachine/BedMachineGreenland-v5.nc.
    """
    return _sample(lat, lon, variable, path, crs)


def sample_geoid(lat, lon,
                 path: Union[str, Path, None] = None,
                 crs: str = BEDMACHINE_CRS,
                 variable: str = "geoid") -> np.ndarray:
    """
    Geoid height above the WGS84 ellipsoid at each lat/lon, in m.

    Nearest-neighbour on BedMachine's 150 m grid.

    Parameters
    ----------
    lat, lon: array-like
        Point coordinates in degrees.

    path: str or Path or None
        BedMachine NetCDF. Defaults to data/bedmachine/BedMachineGreenland-v5.nc.

    variable: str
        Name of the geoid variable in the file.
    """
    return _sample(lat, lon, variable, path, crs)


def _sample(lat, lon, variable, path, crs) -> np.ndarray:
    path = Path(path) if path is not None else DEFAULT_BEDMACHINE
    if not path.exists():
        raise FileNotFoundError(
            f"BedMachine not found at {path}. Pass an explicit path or a scalar "
            f"geoid separation instead."
        )

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise KeyError(f"{path.name} has no {variable!r}; found "
                           f"{sorted(ds.data_vars)}")
        transformer = pyproj.Transformer.from_crs("EPSG:4326", crs,
                                                  always_xy=True)
        px, py = transformer.transform(lon, lat)
        sampled = ds[variable].sel(
            x=xr.DataArray(px, dims="point"),
            y=xr.DataArray(py, dims="point"),
            method="nearest",
        )
        values = sampled.values.astype(float)

    outside = (px < float(ds.x.min())) | (px > float(ds.x.max())) | \
              (py < float(ds.y.min())) | (py > float(ds.y.max()))
    if np.any(outside):
        values = values.copy()
        values[outside] = np.nan

    return values


def resolve_geoid(lat, lon, spec: Optional[Union[float, str, Path]]) -> object:
    """
    Turn a geoid argument into either a scalar or a per-sample array.

    spec may be None (0.0), a number (used as-is), or a path to BedMachine.
    """
    if spec is None:
        return 0.0
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return float(spec)
    return sample_geoid(lat, lon, path=spec)
