"""
Tests for emperical functions.
"""

from xopr_gline.constants import DEFAULT_CONSTANTS
from xopr_gline.empirical import hab
import xopr.opr_access
import pytest
import xarray as xr
import numpy as np

selected_collection = "2010_Greenland_DC8"
selected_segment = "20100420_03"


@pytest.fixture(scope="module")
def layers():
    """Load a short segment and convert its layers to WGS84 elevations."""
    opr = xopr.opr_access.OPRConnection(cache_dir="/tmp")

    # Query frames
    stac_items = opr.query_frames(
        collections=[selected_collection], segment_paths=[selected_segment]
    )

    stac_items = stac_items.iloc[7:11]
    frames = opr.load_frames(stac_items)
    flight_line = xopr.merge_frames(frames)
    layers = opr.get_layers(flight_line)

    # get_layers returns picks as two-way travel time, but hab() works in
    # WGS84 elevation, so convert each layer relative to the surface.
    surface = layers["standard:surface"]
    return {
        name: xopr.layer_twtt_to_range(
            layer, surface, vertical_coordinate="wgs84"
        )
        for name, layer in layers.items()
    }


@pytest.fixture(scope="module")
def result(layers):
    """Run hab once and reuse across all tests."""
    return hab(layers)


# --- Tests ---

class TestHab:
    def test_returns_dataarray(self, result):
        assert isinstance(result, xr.DataArray)

    def test_output_shape_matches_input(self, result, layers):
        """Output should have the same spatial shape as the input layers."""
        expected_shape = layers["standard:surface"]["wgs84"].shape
        assert result.shape == expected_shape

    def test_no_all_nan(self, result):
        """Result should not be entirely NaN."""
        assert not np.all(np.isnan(result.values))

    def test_hab_formula_matches_manual_calculation(self, result, layers):
        """Cross-check output against manually applying the formula."""
        rho_sw = DEFAULT_CONSTANTS.rho_sw
        rho_ice = DEFAULT_CONSTANTS.rho_ice
        surface = layers["standard:surface"]["wgs84"]
        bottom = layers["standard:bottom"]["wgs84"]
        H = surface - bottom
        expected = H - (rho_sw / rho_ice) * (bottom * -1)
        np.testing.assert_allclose(result.values, expected.values)

    def test_custom_densities_differ_from_defaults(self, result, layers):
        """Changing densities should produce a different result."""
        result_custom = hab(layers, rho_sw=1025, rho_ice=900)
        assert not np.allclose(result.values, result_custom.values, equal_nan=True)

    def test_dimensions_preserved(self, result, layers):
        """Output dimensions should match input surface dimensions."""
        expected_dims = layers["standard:surface"]["wgs84"].dims
        assert result.dims == expected_dims
