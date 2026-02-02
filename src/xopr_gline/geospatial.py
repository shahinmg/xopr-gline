"""
Functions for retrieving geospatial datasets and making spatiotemporal queries.
"""

import earthaccess
import geopandas as gpd


def get_greenland_termini() -> gpd.GeoDataFrame:
    """
    Load Greenland outlet glacier termini positions.

    Citation:
    - Joughin, I., Moon, T., Joughin, J. & Black, T. (2021). MEaSUREs Annual Greenland
      Outlet Glacier Terminus Positions from SAR Mosaics. (NSIDC-0642, Version 2).
      [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center
      Distributed Active Archive Center. https://doi.org/10.5067/ESFWE11AVFKW.

    Returns
    -------
    gpd.GeoDataFrame
        A geopandas.GeoDataFrame of the glacier termini positions for one year.

        | GlacierID | ... |    geometry     | ... | Official_n | ... |
        |-----------|-----|-----------------|-----|------------|-----|
        |     1     |     | LINESTRING(...) |     | ? Gletsjer |     |

    """
    # Authenticate to Earthdata login
    earthaccess.login()

    # Search for granules in https://nsidc.org/data/nsidc-0642/versions/2
    granules = earthaccess.search_data(
        collection_concept_id="C3292900075-NSIDC_CPRD",
        temporal=("2021-01-01", "2021-12-31"),  # TODO make this a parameter?
    )
    _files = earthaccess.download(granules=granules, local_path="data/")

    # Join glacier placenames to their termini geometry
    df_glacierid = gpd.read_file(
        filename="data/GlacierIDs_v02.0.shp", read_geometry=False
    ).set_index(keys="GlacierID")
    gdf_termini_ = gpd.read_file(filename="data/termini_2020_2021_v02.0.shp").set_index(
        keys="Glacier_ID"
    )
    gdf_termini = gdf_termini_.merge(
        right=df_glacierid,
        left_index=True,  # left_on="Glacier_ID"
        right_index=True,  # right_on="GlacierID"
    )

    return gdf_termini
