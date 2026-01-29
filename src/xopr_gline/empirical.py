"""
An empirical approach to automatically identify grounding points from xOPR accessed radar data.
This is an xOPR approach to the methods described in Xia et al., 2025 https://ieeexplore.ieee.org/document/11202578/

"""
import numpy as np
import xarray as xr


def hab(ds: xr.Dataset) -> xr.Dataset:
    """
    Add height above buoyancy and thickness.

    Parameters
    ----------
    ds : xr.Dataset
        xOPR Dataset with surface and bottom variables.

    Returns
    -------
    xr.Dataset
        Dataset with height above buoyancy and thickness added in meters.

    """
    
    return ds