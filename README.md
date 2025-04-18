# Indian Climate Change Predictor

A Streamlit-based web application to visualize current and forecasted weather across Indian states using Open-Meteo API and a custom-trained neural network model.
Features.

- Current Weather: Displays current temperature, precipitation, wind speed, and direction for 29 Indian states and 5 Union territories of India using Open-Meteo API.
- Forecast Weather: Provides forecasts for up to 10 days:
- Days 1–7: Uses Open-Meteo forecast API.
- Days 8–10: Uses a trained dense neural network model.
- Interactive Maps: Visualizes weather data on an interactive map using Folium and GeoJSON.

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt
- GeoJSON file: data/india_states.geojson (not included; source from a reliable provider like data.gov.in)

## Setup

- Clone the Repository:
git clone <repository-url>
cd Indian-Climate_Change_Predictor

- Create Virtual Environment:
python -m venv .venv
.venv\Scripts\activate  # Windows

- Install Dependencies:
pip install -r requirements.txt

- Prepare GeoJSON:

- Place india_states.geojson in the data/ directory.

- Ensure it contains Indian states with a state name column, typically NAME_1.

- State names should match STATE_NAME_MAPPING in utils/weather.py (e.g., "Jammu and Kashmir", "Andaman and Nicobar Islands").

- Inspect GeoJSON columns and data:
import geopandas as gpd
gdf = gpd.read_file("data/india_states.geojson")
print("Columns:", gdf.columns.tolist())
print("Sample data (first row):")
for col in gdf.columns:
    print(f"  {col}: {gdf[col].iloc[0]}")
print("All state names:", gdf["NAME_1"].tolist())

- If NAME_1 is present, utils/map_utils.py is configured to use it (STATE_NAME_COLUMN = "NAME_1"). If another column is used, update STATE_NAME_COLUMN in utils/map_utils.py.

- Train the Model:
python utils/model.py

- Fetches historical weather data from Open-Meteo.
Trains a dense neural network and saves it as models/lstm_weather_model.h5.
Saves normalization parameters to data/normalization_params.csv.

- Run the Application:
streamlit run app.py

Open http://localhost:8501 in your browser.
View current weather and forecast maps using the day slider.



## Project Structure

app.py: Streamlit app for the web interface.
utils/model.py: Fetches historical data, trains, and saves the model.
utils/weather.py: Fetches current and forecast weather data.
utils/map_utils.py: Generates Folium maps.
requirements.txt: Python dependencies.
data/: Directory for historical_weather.csv, normalization_params.csv, and india_states.geojson.
models/: Directory for lstm_weather_model.h5.

##  Troubleshooting

- No Forecast Data:

Ensure models/lstm_weather_model.h5 exists for days 8+:
dir models

- Debug day 1 forecast:
import logging
logging.basicConfig(level=logging.DEBUG)
from utils.weather import get_forecast_weather
df = get_forecast_weather(1)
print(df)

- Test Open-Meteo forecast API:
import requests
url = "https://api.open-meteo.com/v1/forecast?latitude=15.9129&longitude=79.7400&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_max,wind_direction_10m_dominant&forecast_days=7"
response = requests.get(url)
print(response.status_code, response.json())

Map Issues (KeyError: 'No state name column found in GeoJSON'):

Verify NAME_1 in india_states.geojson:
import geopandas as gpd
gdf = gpd.read_file("data/india_states.geojson")
print("Columns:", gdf.columns.tolist())
print("Sample data (first row):")
for col in gdf.columns:
    print(f"  {col}: {gdf[col].iloc[0]}")
print("All state names:", gdf["NAME_1"].tolist())


- If NAME_1 is missing, set STATE_NAME_COLUMN in utils/map_utils.py to the correct column.

Ensure state names match STATE_NAME_MAPPING. Update the replace dictionary in map_utils.py for mismatches (e.g., "Telengana": "Telangana").

- Training Errors:

Run python utils/model.py and check for NaN loss, No gradients, or InvalidArgumentError.

- Test weighted_mse:
import tensorflow as tf
import numpy as np
def weighted_mse(y_true, y_pred):
    weights = tf.constant([0.4, 0.1, 0.3, 0.2], dtype=tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    squared_diff = tf.square(y_true - y_pred)
    weighted_diff = squared_diff * tf.expand_dims(weights, 0)
    return tf.reduce_mean(weighted_diff, axis=-1)
y_true = tf.constant(np.random.randn(10, 4), dtype=tf.float32)
y_pred = tf.constant(np.random.randn(10, 4), dtype=tf.float32)
print(tf.reduce_mean(weighted_mse(y_true, y_pred)).numpy())


- Test model training:
import pandas as pd
from utils.model import prepare_data
df = pd.read_csv("data/historical_weather.csv")
X, y, _, _ = prepare_data(df)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
model = Sequential([Dense(4, activation="tanh", input_shape=(7*4,))])
optimizer = SGD(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: tf.reduce_mean(tf.square(y_true - y_pred)))
model.fit(X, y, epochs=1, batch_size=32, verbose=1)

- API Errors:

Open-Meteo requires no key. Check internet or API status.

## Notes

The model uses a single dense layer with SGD optimizer to predict temperature, precipitation, wind speed, and direction.
Historical data is fetched for 365 days from Open-Meteo’s archive API.
Forecasts for days 8+ rely on the trained model and may be less accurate due to iterative predictions.

## License
MIT License

