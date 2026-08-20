"""
Script to apply grounding line detection algorithm across Greenland glaciers.
"""

import geopandas as gpd
import tqdm
import xarray as xr
import xopr

# %%
# Set up OPR connection and load flight line indexes from GeoJSON
opr = xopr.OPRConnection(
    collection_url="https://data.cresis.ku.edu/data/",
    cache_dir="data/radar_cache",
    stac_parquet_href="gs://opr_stac/catalog/**/*.parquet",
)
gdf: gpd.GeoDataFrame = gpd.read_file(filename="data/glacier_to_radarframe_ids.geojson")


# %%
# Main loop
unique_flightlines = gdf.stac_item_id.unique()
for idx, flightline_id in tqdm.tqdm(
    iterable=enumerate(unique_flightlines), total=len(unique_flightlines)
):
    # Check which glacier termini this flight line crosses
    glacier_ids: list[str] = gdf.query(
        expr=f"stac_item_id == '{flightline_id}'"
    ).GlacierID.to_list()
    print("\n")
    print(
        f"Index: {idx}, applying gline algorithm\n"
        f"on flight_line: {flightline_id}\n"
        f"which crosses GlacierIDs: {glacier_ids}"
    )

    # Perform exact STAC search on ID, return one xarray.Dataset frame
    stac_items: gpd.GeoDataFrame = opr.query_frames(
        search_kwargs={"ids": [flightline_id]}
    )
    assert len(stac_items) == 1  # should only return one flight line

    frames: list[xr.Dataset] = opr.load_frames(stac_items=stac_items)
    assert len(frames) == 1, len(frames)  # only one radar frame per flight line segment
    flight_line: xr.Dataset = xopr.merge_frames(frames=frames)

    # Apply grounding line algorithm here
    # TODO

    # Save grounding line prediction output
    # TODO

    if idx > 5:
        break

print("Done!")
