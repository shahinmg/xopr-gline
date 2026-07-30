"""
Tests for geospatial functions.
"""

import geopandas as gpd
import numpy as np
import pyproj
import pytest
import xarray as xr

from xopr_gline.geospatial import get_greenland_termini, open_itslive_velocity


def test_get_greenland_termini():
    """
    Ensure get_greenland_termini function returns a geopandas.GeoDataFrame with
    columns including the glacier placenames and geometries of type LineString.
    """
    gdf = get_greenland_termini(end_year=2021)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 239
    assert set(gdf.columns) == {
        "Image_ID",
        "Sensor",
        "Quality_Fl",
        "SourceDate",
        "geometry",
        "POINT_X",
        "POINT_Y",
        "GrnlndcNam",
        "Official_n",
        "AltName",
    }
    assert gdf.geometry.geom_type.unique().tolist() == ["LineString"]
    assert gdf.geometry.crs == "OGC:CRS84"
    assert gdf.index.is_monotonic_increasing


def test_open_itslive_velocity():
    """
    Windowed read of a 40 km box over Petermann returns a georeferenced
    DataArray, whether it comes from a cached download or straight off S3.
    """
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:3413", always_xy=True
    )
    x, y = transformer.transform(-60.5, 80.6)
    box = (x - 20_000, y - 20_000, x + 20_000, y + 20_000)

    vx = open_itslive_velocity(component="vx")
    assert isinstance(vx, xr.DataArray)
    assert vx.name == "vx"
    assert vx.dims == ("y", "x")
    assert vx.rio.crs == "EPSG:3413"

    subset = vx.rio.clip_box(*box).compute()
    assert subset.shape == (334, 335)
    # Petermann flows fast enough that the box cannot be all fill.
    assert np.isfinite(subset).any()
    assert np.nanmin(subset) < -100  # m/yr, northwest-flowing


def test_open_itslive_velocity_bad_component():
    """Unknown components name the valid ones rather than 404ing later."""
    with pytest.raises(KeyError, match="vx"):
        open_itslive_velocity(component="vz")
