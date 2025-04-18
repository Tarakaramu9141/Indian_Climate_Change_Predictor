import geopandas as gpd

# Load GeoJSON
geojson_path = "data/india_states.geojson"
gdf = gpd.read_file(geojson_path)

# Print column names and NAME_1 values
print("Columns in GeoJSON:", gdf.columns.tolist())
print("\nState names in NAME_1 column:")
print(gdf['NAME_1'].tolist())