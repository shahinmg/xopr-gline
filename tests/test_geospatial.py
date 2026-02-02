"""
Tests for geospatial functions.
"""

import geopandas as gpd

from xopr_gline.geospatial import get_greenland_termini


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
