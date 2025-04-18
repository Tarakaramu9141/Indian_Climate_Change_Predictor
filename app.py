import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from utils.weather import get_current_weather, get_forecast_weather
from utils.map_utils import create_map

# Streamlit page configuration
st.set_page_config(page_title="Indian Climate Change Predictor", layout="wide")

# Title
st.title("Indian Climate Change Predictor")

# Fetch current weather
current_weather = get_current_weather()
st.write("Current Weather:")
st.write(current_weather)

# Create and display current weather map
if not current_weather.empty:
    current_map = create_map(current_weather, "temp")
    st.subheader("Current Weather Map (Temperature)")
    st.write("No Geodata.json for Orissa and uttarnchal")
    folium_static(current_map, width=1200, height=600)
else:
    st.error("No current weather data available.")

# Forecast day slider
forecast_day = st.slider("Select Forecast Day", min_value=1, max_value=10, value=1)

# Fetch forecast weather
forecast_weather = get_forecast_weather(forecast_day)
st.write(f"Forecast Weather for Day {forecast_day}:")
st.write("No Geodata.json for Orissa and uttarnchal")
st.write(forecast_weather)

# Create and display forecast weather map
if not forecast_weather.empty:
    forecast_map = create_map(forecast_weather, "temp")
    st.subheader(f"Forecast Weather Map for Day {forecast_day} (Temperature)")
    folium_static(forecast_map, width=1200, height=600)
else:
    st.error(f"No forecast data available for day {forecast_day}.")