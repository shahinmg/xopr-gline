"""
Functions for retrieving geospatial datasets and making spatiotemporal queries.
"""

import datetime
import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Union

import earthaccess
import geopandas as gpd
import requests
import rioxarray  # noqa: F401  registers the .rio accessor
import xarray as xr
from tqdm.auto import tqdm

ITSLIVE_VELOCITY_URLS = {
    "vx": "https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2.1/static/cog/"
          "ITS_LIVE_velocity_120m_RGI05A_0000_V02.1_vx.tif",
    "vy": "https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2.1/static/cog/"
          "ITS_LIVE_velocity_120m_RGI05A_0000_V02.1_vy.tif",
}
DEFAULT_ITSLIVE_DIR = Path("data/its_live")


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
    with tempfile.TemporaryDirectory() as tmpdirname:
        _files = earthaccess.download(granules=granules, local_path=tmpdirname)

        # Join glacier placenames to their termini geometry
        df_glacierid = gpd.read_file(
            filename=os.path.join(tmpdirname, "GlacierIDs_v02.0.shp"),
            read_geometry=False,
        ).set_index(keys="GlacierID")
        gdf_termini_ = gpd.read_file(
            filename=os.path.join(tmpdirname, "termini_2020_2021_v02.0.shp")
        ).set_index(keys="Glacier_ID")
        gdf_termini = gdf_termini_.merge(
            right=df_glacierid,
            left_index=True,  # left_on="Glacier_ID"
            right_index=True,  # right_on="GlacierID"
        ).sort_index(axis="index")

        return gdf_termini.to_crs(crs="OGC:CRS84")


def get_itslive_velocity(
    components: Iterable[str] = ("vx", "vy"),
    directory: Union[str, Path, None] = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """
    Download the ITS_LIVE 120 m Greenland (RGI05A) velocity mosaic components.

    Citation:
    - Gardner, A. S., Fahnestock, M. A. & Scambos, T. A. (2025). MEaSUREs
      ITS_LIVE Regional Glacier and Ice Sheet Surface Velocities, Version 2.
      [Data Set]. NASA National Snow and Ice Data Center Distributed Active
      Archive Center. https://doi.org/10.5067/9SM8CTFHF3AZ.

    Parameters
    ----------
    components : iterable of str
        Which components to fetch. Keys of ITSLIVE_VELOCITY_URLS.

    directory : str or Path or None
        Where to save the COGs. Defaults to data/its_live.

    overwrite : bool
        Re-download even if the file exists.

    Returns
    -------
    dict[str, Path]
        Component name to downloaded GeoTIFF.
    """
    directory = Path(directory) if directory is not None else DEFAULT_ITSLIVE_DIR
    directory.mkdir(parents=True, exist_ok=True)

    paths = {}
    for component in components:
        if component not in ITSLIVE_VELOCITY_URLS:
            raise KeyError(f"unknown component {component!r}; expected one of "
                           f"{sorted(ITSLIVE_VELOCITY_URLS)}")
        url = ITSLIVE_VELOCITY_URLS[component]
        destination = directory / url.rsplit("/", 1)[-1]
        paths[component] = destination

        if destination.exists() and not overwrite:
            continue

        # Stream to a .part file so an interrupted download is not mistaken for
        # a complete one on the next call.
        partial = destination.with_suffix(destination.suffix + ".part")
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0)) or None
            with open(partial, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=destination.name
            ) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    progress.update(len(chunk))

        partial.replace(destination)

    return paths


def open_itslive_velocity(
    component: str = "vx",
    path: Union[str, Path, None] = None,
    directory: Union[str, Path, None] = None,
    chunks: Optional[dict] = None,
) -> xr.DataArray:
    """
    Open an ITS_LIVE velocity component, lazily, in m/yr on EPSG:3413.

    Parameters
    ----------
    component : str
        Key of ITSLIVE_VELOCITY_URLS, i.e. 'vx' or 'vy'.

    path : str or Path or None
        Explicit file to open, skipping the cache lookup.

    directory : str or Path or None
        Where to look for a cached download. Defaults to data/its_live.

    chunks : dict or None
        Dask chunks. Defaults to the COG's 512 px tiling.

    Returns
    -------
    xr.DataArray
        The component, band dimension squeezed out.
    """
    if component not in ITSLIVE_VELOCITY_URLS:
        raise KeyError(f"unknown component {component!r}; expected one of "
                       f"{sorted(ITSLIVE_VELOCITY_URLS)}")

    url = ITSLIVE_VELOCITY_URLS[component]
    if path is not None:
        source = str(path)
    else:
        directory = Path(directory) if directory is not None else DEFAULT_ITSLIVE_DIR
        cached = directory / url.rsplit("/", 1)[-1]
        source = str(cached) if cached.exists() else url

    if source.startswith("http"):
        # Stop GDAL listing the bucket prefix on every open, which costs a
        # round trip per read and finds nothing useful here.
        os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

    da = rioxarray.open_rasterio(
        source, masked=True, chunks=chunks or {"x": 512, "y": 512}
    )
    if "band" in da.dims and da.sizes["band"] == 1:
        da = da.squeeze("band", drop=True)
    return da.rename(component)
