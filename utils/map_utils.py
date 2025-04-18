import folium
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Specify state name column
STATE_NAME_COLUMN = "NAME_1"  # Identified from GeoJSON inspection

def create_map(weather_data, feature):
    """Create a Folium map with weather data overlaid on Indian states."""
    # Load GeoJSON
    geojson_path = os.path.join("data", "india_states.geojson")
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"GeoJSON file not found at {geojson_path}")
    gdf = gpd.read_file(geojson_path)
    
    # Log all columns and sample data
    logger.info(f"GeoJSON columns: {list(gdf.columns)}")
    logger.info(f"Sample GeoJSON data (first row):")
    for col in gdf.columns:
        logger.info(f"  {col}: {gdf[col].iloc[0]}")
    
    # Use manual state name column if specified
    if STATE_NAME_COLUMN is not None and STATE_NAME_COLUMN in gdf.columns:
        state_col = STATE_NAME_COLUMN
    else:
        # Auto-detect state name column
        possible_cols = ["ST_NM", "NAME", "state", "State", "state_name", "name", "NAME_1"]
        state_col = None
        for col in possible_cols:
            if col in gdf.columns:
                state_col = col
                break
    
    if state_col is None:
        logger.error(f"No state name column found in GeoJSON. Tried: {possible_cols}")
        raise KeyError("No state name column found in GeoJSON")
    
    logger.info(f"Using state name column: {state_col}")
    
    # Standardize state names
    gdf[state_col] = gdf[state_col].replace({
        "Jammu & Kashmir": "Jammu and Kashmir",
        "Andaman & Nicobar": "Andaman and Nicobar Islands",
        "Telengana": "Telangana"  # Common GeoJSON misspelling
    })
    
    # Log sample state names after standardization
    logger.info(f"Sample state names after standardization: {gdf[state_col].head().tolist()}")
    
    # Merge weather data with GeoJSON
    merged = gdf.merge(weather_data, left_on=state_col, right_on="state", how="left")
    
    # Initialize map centered on India
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="cartodbpositron")
    
    # Define color scale
    if feature == "temp":
        colorscale = folium.LinearColormap(
            colors=["blue", "green", "yellow", "red"],
            vmin=-10,
            vmax=40,
            caption="Temperature (°C)"
        )
    elif feature == "precip":
        colorscale = folium.LinearColormap(
            colors=["white", "lightblue", "blue", "darkblue"],
            vmin=0,
            vmax=100,
            caption="Precipitation (mm)"
        )
    elif feature == "wind_speed":
        colorscale = folium.LinearColormap(
            colors=["white", "lightgreen", "green", "darkgreen"],
            vmin=0,
            vmax=50,
            caption="Wind Speed (km/h)"
        )
    else:  # wind_dir
        colorscale = folium.LinearColormap(
            colors=["white", "purple", "darkpurple"],
            vmin=0,
            vmax=360,
            caption="Wind Direction (°)"
        )
    
    # Add choropleth layer
    folium.Choropleth(
        geo_data=merged,
        name="choropleth",
        data=merged,
        columns=[state_col, feature],
        key_on=f"feature.properties.{state_col}",
        fill_color="YlOrRd" if feature == "temp" else "Blues" if feature == "precip" else "Greens",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"{feature.capitalize()}",
        nan_fill_color="gray"
    ).add_to(m)
    
    # Add tooltips
    folium.GeoJson(
        merged,
        style_function=lambda x: {"fillColor": "transparent", "color": "black", "weight": 1},
        tooltip=folium.GeoJsonTooltip(
            fields=[state_col, "temp", "precip", "wind_speed", "wind_dir"],
            aliases=["State", "Temperature (°C)", "Precipitation (mm)", "Wind Speed (km/h)", "Wind Direction (°)"],
            localize=True
        )
    ).add_to(m)
    
    # Add color scale
    colorscale.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m