"""
Locating an external grounding zone on a profile's along-track axis.

The zone is drawn and reported beside a detection, never used to make one, so
what matters here is that it lands in the right place and that the sign of the
offset says which side of it the pick fell on.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from xopr_gline.constants import DEFAULT_CONSTANTS
from xopr_gline.grounding.plotting import _gz_bands
from xopr_gline.grounding.profile import GlacierProfile
from xopr_gline.grounding.reference import (_epoch, epoch_label,
                                            offset_from_zone_km,
                                            zone_crossings_km, zone_for_date)

GEOID = 30.0
F = DEFAULT_CONSTANTS.flotation_factor


def _profile(n=200, dx=0.25):
    """A glacier running seaward along a line of longitude, grounded inland."""
    x = np.arange(n) * dx
    h = np.linspace(800.0, 300.0, n)
    bed_msl = np.linspace(-300.0, -900.0, n)
    surf_msl = np.maximum(bed_msl + h, h * F)
    lat = np.linspace(80.0, 80.45, n)          # x increases northward
    return GlacierProfile(x=x, amp=np.full(n, -110.0),
                          h_surf=surf_msl + GEOID, h_bed=surf_msl - h + GEOID,
                          geoid_separation_m=GEOID,
                          lat=lat, lon=np.full(n, -60.0))


def _zone(lat_lo, lat_hi, name="2011_2015"):
    poly = Polygon([(-61, lat_lo), (-59, lat_lo), (-59, lat_hi), (-61, lat_hi)])
    return gpd.GeoDataFrame({"zone_name": [name],
                             "epoch_start": [pd.Timestamp("2011-01-01")],
                             "epoch_end": [pd.Timestamp("2015-12-31")]},
                            geometry=[poly], crs="EPSG:4326")


def test_the_crossing_lands_where_the_track_enters_the_zone():
    p = _profile()
    lat_lo, lat_hi = 80.2, 80.25
    bands = zone_crossings_km(p, _zone(lat_lo, lat_hi))
    assert len(bands) == 1
    lo, hi = bands[0]
    expected_lo = float(np.interp(lat_lo, p.lat, p.x))
    expected_hi = float(np.interp(lat_hi, p.lat, p.x))
    assert lo == pytest.approx(expected_lo, abs=p.dx)
    assert hi == pytest.approx(expected_hi, abs=p.dx)


def test_a_zone_the_track_misses_gives_no_crossing():
    assert zone_crossings_km(_profile(), _zone(70.0, 71.0)) == []


def test_a_track_crossing_the_same_zone_twice_reports_both():
    """An out-and-back line enters once on each leg; both are real."""
    p = _profile()
    out_and_back = np.r_[p.lat, p.lat[::-1]]
    p = GlacierProfile(x=np.arange(len(out_and_back)) * p.dx,
                       amp=np.tile(p.amp, 2), h_surf=np.tile(p.h_surf, 2),
                       h_bed=np.tile(p.h_bed, 2), geoid_separation_m=GEOID,
                       lat=out_and_back, lon=np.full(len(out_and_back), -60.0))
    assert len(zone_crossings_km(p, _zone(80.2, 80.25))) == 2


def test_the_offset_is_negative_downflow_of_the_zone():
    """Matches the convention run_grounding.py prints: negative = downflow."""
    p = _profile()
    bands = zone_crossings_km(p, _zone(80.2, 80.25))
    lo, hi = bands[0]
    seaward = p.landward_sign()
    downflow = lo - 3.0 if seaward > 0 else hi + 3.0
    upflow = hi + 3.0 if seaward > 0 else lo - 3.0

    assert offset_from_zone_km(p, downflow, bands) == pytest.approx(-3.0, abs=0.01)
    assert offset_from_zone_km(p, upflow, bands) == pytest.approx(3.0, abs=0.01)
    assert offset_from_zone_km(p, 0.5 * (lo + hi), bands) == 0.0
    assert np.isnan(offset_from_zone_km(p, lo, []))


def test_the_epoch_nearest_the_flight_is_chosen():
    zones = pd.concat([_zone(80.2, 80.25, "2011_2015"),
                       _zone(80.2, 80.25, "2016_2017")], ignore_index=True)
    zones.loc[1, "epoch_start"] = pd.Timestamp("2016-01-01")
    zones.loc[1, "epoch_end"] = pd.Timestamp("2017-12-31")
    zones = gpd.GeoDataFrame(zones, geometry="geometry", crs="EPSG:4326")

    inside = zone_for_date(zones, pd.Timestamp("2013-04-20"))
    assert inside["zone_name"].tolist() == ["2011_2015"]
    # 2019 is two years past 2016_2017 and four past 2011_2015.
    after = zone_for_date(zones, pd.Timestamp("2019-04-17"))
    assert after["zone_name"].tolist() == ["2016_2017"]


def test_an_unparsed_file_name_gets_no_epoch():
    assert _epoch("gzone_final")[0] is pd.NaT
    assert _epoch("2011_2015_gzone+iceshelf")[0] == pd.Timestamp("2011-01-01")


def test_the_figure_takes_one_band_or_several():
    """--gz_lo/--gz_hi give a bare pair; a layer gives a list of them."""
    assert _gz_bands((95.4, 97.9)) == [(95.4, 97.9)]
    assert _gz_bands([(10.0, 12.0), (30.0, 31.0)]) == [(10.0, 12.0), (30.0, 31.0)]
    assert _gz_bands(None) == []
    assert _gz_bands([]) == []


def test_the_legend_names_the_epoch_not_the_kilometres():
    assert epoch_label(_zone(80.2, 80.25)) == "2011-2015"


def test_a_layer_with_no_epoch_falls_back_to_its_name():
    zones = _zone(80.2, 80.25, "gzone_final")
    zones.loc[0, ["epoch_start", "epoch_end"]] = pd.NaT
    assert epoch_label(zones) == "gzone-final"
    assert epoch_label(None) is None


# -- elevation datum diagnostic -------------------------------------------
def _with_open_water(offset_m, n=200, dx=0.25, water_km=15.0):
    """A shelf running out into open water whose surface sits at offset_m."""
    x = np.arange(n) * dx
    h = np.full(n, 400.0)
    surf_msl = h * F
    bed_msl = surf_msl - h
    water = x < water_km                       # seaward end is open water
    surf_msl[water] = 0.0
    bed_msl[water] = 0.0
    return GlacierProfile(x=x, amp=np.full(n, -110.0),
                          h_surf=surf_msl + GEOID + offset_m,
                          h_bed=bed_msl + GEOID + offset_m,
                          geoid_separation_m=GEOID)


def test_open_water_reports_the_flights_datum_offset():
    """Open water is at sea level, so anything else is the datum error."""
    for offset in (0.0, -29.3, 6.1):
        p = _with_open_water(offset)
        assert p.sea_level_offset_m() == pytest.approx(offset, abs=0.01)
        assert p.sea_level_samples() > 10


def test_a_line_with_no_open_water_reports_no_offset():
    p = _with_open_water(0.0, water_km=0.0)
    assert np.isnan(p.sea_level_offset_m())
    assert p.sea_level_samples() == 0


def test_the_datum_offset_matches_the_shelf_residual():
    """The offset shifts every flotation quantity by the same amount."""
    clean, biased = _with_open_water(0.0), _with_open_water(-29.3)
    ice = clean.ice_mask()
    shift = (np.median(biased.flotation_residual[ice])
             - np.median(clean.flotation_residual[ice]))
    assert shift == pytest.approx(-29.3, abs=0.01)
    assert biased.sea_level_offset_m() == pytest.approx(shift, abs=0.01)


# -- single-year surveys ---------------------------------------------------
def test_a_single_year_file_name_gets_a_one_year_epoch():
    """A TLS survey named for the year it was flown, e.g.
    2018_grounding_zone_convex_hull.gpkg."""
    start, end = _epoch("2018_grounding_zone_convex_hull")
    assert (start.year, end.year) == (2018, 2018)


def test_two_one_year_surveys_stay_separate():
    """Unparsed they would share a NaT epoch and merge into one reference."""
    zones = pd.concat([_zone(80.2, 80.25, "2018_gz"),
                       _zone(80.2, 80.25, "2019_gz")], ignore_index=True)
    for i, year in enumerate((2018, 2019)):
        zones.loc[i, "epoch_start"] = pd.Timestamp(f"{year}-01-01")
        zones.loc[i, "epoch_end"] = pd.Timestamp(f"{year}-12-31")
    zones = gpd.GeoDataFrame(zones, geometry="geometry", crs="EPSG:4326")

    assert zone_for_date(zones, pd.Timestamp("2017-04-22"))[
        "zone_name"].tolist() == ["2018_gz"]
    assert zone_for_date(zones, pd.Timestamp("2019-08-01"))[
        "zone_name"].tolist() == ["2019_gz"]
