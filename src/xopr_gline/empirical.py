"""
An empirical approach to automatically identify grounding points from xOPR accessed radar data.
This is an xOPR approach to the methods described in Xia et al., 2025 https://ieeexplore.ieee.org/document/11202578/

"""
import numpy as np
import xarray as xr
from scipy.special import erf
from scipy.optimize import curve_fit

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

def erf_topography_model(elevation, amp, b, x0, v_off):
    """
    S-shaped model for topographic transitions.
    Described in Xia et al., 2025 https://ieeexplore.ieee.org/document/11202578/

    Parameters:
    - elevation: elevation profile from xOPR
    - amp: amplitdue of elevation profile
    - b: slope
    - x0: center of the transition
    - v_off: vertical offset
    """
    
    return amp * erf(b * (elevation - x0)) + v_off

def get_derivatives(elevation, amp, b, x0):
    """
    Calculates the 1st, 2nd, and 3rd analytical derivatives of the fit.
    
    Parameters:
    - elevation: elevation profile from xOPR
    - amp: amplitdue of elevation profile
    - b: slope
    - x0: center of the transition
    
    Returns:
    - a tuple of the first, second, and third analytical derivatives of the modeled elevation fit

    example
    
    _, _, z3 = get_derivatives(bed_elevation['along_track'], a_fit, b_fit, x0_fit)
    """
    
    u = b * (elevation - x0)
    # 1st derivative: Gaussian
    z_prime = (2 * amp * b / np.sqrt(np.pi)) * np.exp(-u**2)
    # 2nd derivative
    z_double_prime = -(4 * amp * b**2 / np.sqrt(np.pi)) * u * np.exp(-u**2)
    # 3rd derivative
    z_triple_prime = (4 * amp * b**3 / np.sqrt(np.pi)) * (2 * u**2 - 1) * np.exp(-u**2)
    
    return z_prime, z_double_prime, z_triple_prime