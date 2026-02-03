"""
Functions for retrieving geospatial datasets and making spatiotemporal queries.
"""

import datetime
import os
import tempfile

import earthaccess
import geopandas as gpd


def get_greenland_termini(end_year: int = 2021) -> gpd.GeoDataFrame:
    """
    Load Greenland outlet glacier termini positions.

    Citation:
    - Joughin, I., Moon, T., Joughin, J. & Black, T. (2021). MEaSUREs Annual Greenland
      Outlet Glacier Terminus Positions from SAR Mosaics. (NSIDC-0642, Version 2).
      [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center
      Distributed Active Archive Center. https://doi.org/10.5067/ESFWE11AVFKW.

    Parameters
    ----------
    end_year : int
        Year to select for the glacier termini position. Function will pull in data
        from end_year - 1 to end_year. Default is 2021, i.e. 2020-2021.

    Returns
    -------
    gpd.GeoDataFrame
        A geopandas.GeoDataFrame of the glacier termini positions for one year.
        Linestring coordinates are in OGC:CRS84, i.e. longitude/latitude.

        | GlacierID | ... |    geometry     | ... | Official_n | ... |
        |-----------|-----|-----------------|-----|------------|-----|
        |     1     |     | LINESTRING(...) |     | ? Gletsjer |     |

    """
    # Authenticate to Earthdata login
    earthaccess.login()

    # Search for granules in https://nsidc.org/data/nsidc-0642/versions/2
    end_date = datetime.datetime(year=end_year, month=12, day=31)
    start_date = datetime.datetime(year=end_year, month=1, day=1)
    granules = earthaccess.search_data(
        collection_concept_id="C3292900075-NSIDC_CPRD",
        temporal=(start_date, end_date),
    )
    _files = earthaccess.download(granules=granules, local_path=tempfile.tempdir)

    # Join glacier placenames to their termini geometry
    df_glacierid = gpd.read_file(
        filename=os.path.join(tempfile.tempdir, "GlacierIDs_v02.0.shp"),
        read_geometry=False,
    ).set_index(keys="GlacierID")
    gdf_termini_ = gpd.read_file(
        filename=os.path.join(tempfile.tempdir, "termini_2020_2021_v02.0.shp")
    ).set_index(keys="Glacier_ID")
    gdf_termini = gdf_termini_.merge(
        right=df_glacierid,
        left_index=True,  # left_on="Glacier_ID"
        right_index=True,  # right_on="GlacierID"
    )

    return gdf_termini.to_crs(crs="OGC:CRS84")
