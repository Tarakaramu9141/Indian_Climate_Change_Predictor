
import geopandas as gpd
gdf = gpd.read_file("data/india_states.geojson")
print("Columns:", gdf.columns.tolist())
