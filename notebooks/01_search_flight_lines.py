"""
Script to loop over 200+ Greenland glaciers termini, and use the line geometries to find
intersecting OPR flight lines via a STAC query.

Results are saved to a GeoJSON file with the following table structure:

|     stac_item_id     | GlacierID |     collection    |    geometry     |
|----------------------|-----------|-------------------|-----------------|
| Data_20020531_06_001 |     1     | 2002_Greenland_P3 | LINESTRING(...) |
"""

import time

import geopandas as gpd
import tqdm
import xopr

import xopr_gline

# %%
# Set up connection
opr = xopr.OPRConnection(
    collection_url="https://data.cresis.ku.edu/data/",
    stac_parquet_href="gs://opr_stac/catalog/**/*.parquet",
)

# Get termini positions
gdf_termini: gpd.GeoDataFrame = xopr_gline.geospatial.get_greenland_termini(
    end_year=2021
)
assert len(gdf_termini) == 239

# %%
# Loop through glacier termini, save out stac_item_id
for row in tqdm.tqdm(iterable=gdf_termini.itertuples(), total=len(gdf_termini)):
    print("\n")
    print(f"Looking at GlacierID {row.Index}:", row.Official_n)
    # Search for radar frames intersecting termini linestring geometry
    tic = time.time()
    stac_items: gpd.GeoDataFrame = opr.query_frames(geometry=row.geometry)
    toc = time.time()
    print(f"Took {toc - tic} seconds")
    if stac_items is not None:
        print(f"Found {len(stac_items)} rows")

        # Save stac_item_id per GlacierID to GeoJSON file
        stac_items["GlacierID"] = row.Index  # add GlacierID to every row
        stac_items[["GlacierID", "collection", "geometry"]].to_file(
            filename := "data/glacier_to_radarframe_ids.geojson",
            RFC7946=True,  # https://gdal.org/en/stable/drivers/vector/geojson.html#rfc-7946-write-support
            mode="w" if row.Index == 1 else "a",
        )
    else:
        print("Found 0 rows")

print(f"Indexes saved to {filename}")
