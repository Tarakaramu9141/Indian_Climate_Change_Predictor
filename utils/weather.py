import requests
import pandas as pd
import numpy as np
import logging
from tensorflow.keras.models import load_model
import time
import urllib.parse
import tensorflow as tf
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping to standardize GeoJSON state names
STATE_NAME_MAPPING = {
    "Jammu & Kashmir": "Jammu and Kashmir",
    "Andaman and Nicobar": "Andaman and Nicobar Islands",
}

# State coordinates (approximate centroids)
STATE_COORDS = {
    "Andhra Pradesh": {"lat": 15.9129, "lon": 79.7400},
    "Arunachal Pradesh": {"lat": 28.2180, "lon": 94.7278},
    "Assam": {"lat": 26.2006, "lon": 92.9376},
    "Bihar": {"lat": 25.0961, "lon": 85.3131},
    "Chhattisgarh": {"lat": 21.2787, "lon": 81.8661},
    "Goa": {"lat": 15.2993, "lon": 74.1240},
    "Gujarat": {"lat": 22.2587, "lon": 71.1924},
    "Haryana": {"lat": 29.0588, "lon": 76.0856},
    "Himachal Pradesh": {"lat": 31.1048, "lon": 77.1734},
    "Jharkhand": {"lat": 23.6102, "lon": 85.2799},
    "Karnataka": {"lat": 15.3173, "lon": 75.7139},
    "Kerala": {"lat": 10.8505, "lon": 76.2711},
    "Madhya Pradesh": {"lat": 22.9734, "lon": 78.6569},
    "Maharashtra": {"lat": 19.7515, "lon": 75.7139},
    "Manipur": {"lat": 24.6637, "lon": 93.9063},
    "Meghalaya": {"lat": 25.4670, "lon": 91.3662},
    "Mizoram": {"lat": 23.1645, "lon": 92.9376},
    "Nagaland": {"lat": 26.1584, "lon": 94.5624},
    "Odisha": {"lat": 20.9517, "lon": 85.0985},
    "Punjab": {"lat": 31.1471, "lon": 75.3412},
    "Rajasthan": {"lat": 27.0238, "lon": 74.2179},
    "Sikkim": {"lat": 27.5330, "lon": 88.5122},
    "Tamil Nadu": {"lat": 11.1271, "lon": 78.6569},
    "Telangana": {"lat": 18.1124, "lon": 79.0193},
    "Tripura": {"lat": 23.9408, "lon": 91.9882},
    "Uttar Pradesh": {"lat": 27.5706, "lon": 80.0982},
    "Uttarakhand": {"lat": 30.0668, "lon": 79.0193},
    "West Bengal": {"lat": 22.9868, "lon": 87.8550},
    "Jammu and Kashmir": {"lat": 33.7782, "lon": 76.5762},
    "Ladakh": {"lat": 34.1526, "lon": 77.5771},
    "Delhi": {"lat": 28.7041, "lon": 77.1025},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794},
    "Puducherry": {"lat": 11.9416, "lon": 79.8083},
    "Andaman and Nicobar Islands": {"lat": 11.7401, "lon": 92.6586},
    "Lakshadweep": {"lat": 10.5667, "lon": 72.6417},
}

def get_current_weather():
    """Fetch current weather data for all Indian states."""
    weather_data = []
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "current": "temperature_2m,precipitation,wind_speed_10m,wind_direction_10m"
    }
    
    for state, coords in STATE_COORDS.items():
        params.update({"latitude": coords["lat"], "longitude": coords["lon"]})
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        for attempt in range(5):
            try:
                response = requests.get(url, timeout=15)
                logger.info(f"Current weather response for {state}: {response.status_code}")
                response.raise_for_status()
                data = response.json()
                logger.debug(f"Raw response for {state}: {data}")
                current = data.get("current", {})
                if not current:
                    logger.warning(f"No current weather data for {state}")
                    break
                weather_data.append({
                    "state": STATE_NAME_MAPPING.get(state, state),
                    "temp": current.get("temperature_2m", np.nan),
                    "precip": current.get("precipitation", 0),
                    "wind_speed": current.get("wind_speed_10m", np.nan),
                    "wind_dir": current.get("wind_direction_10m", np.nan)
                })
                logger.info(f"Current weather fetched for {state}: {weather_data[-1]}")
                break
            except requests.RequestException as e:
                logger.error(f"Attempt {attempt + 1} failed for {state}: {e}")
                time.sleep(2)
                if attempt == 4:
                    logger.error(f"Failed to fetch current weather for {state} after 5 attempts")
    df = pd.DataFrame(weather_data)
    if df.empty:
        logger.error("No current weather data retrieved.")
    else:
        logger.info(f"Current weather data shape: {df.shape}")
    return df

def get_historical_data(state, lat, lon, start_date, end_date):
    """Fetch historical weather data for a state."""
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max,wind_direction_10m_dominant"
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=15)
            logger.info(f"Historical data response for {state}: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Raw historical response for {state}: {data}")
            daily = data.get("daily", {})
            if not daily or not daily.get("time"):
                logger.warning(f"No historical data for {state}")
                return None
            df = pd.DataFrame({
                "date": daily["time"],
                "temp": daily["temperature_2m_mean"],
                "precip": daily["precipitation_sum"],
                "wind_speed": daily["wind_speed_10m_max"],
                "wind_dir": daily["wind_direction_10m_dominant"]
            })
            df["state"] = STATE_NAME_MAPPING.get(state, state)
            if df.empty:
                logger.warning(f"Empty historical data for {state}")
                return None
            # Clip outliers
            df["temp"] = df["temp"].clip(lower=-10, upper=45)
            df["precip"] = df["precip"].clip(lower=0, upper=500)
            df["wind_speed"] = df["wind_speed"].clip(lower=0, upper=100)
            df["wind_dir"] = df["wind_dir"].clip(lower=0, upper=360)
            # Log data quality before imputation
            logger.info(f"Pre-imputation data quality for {state}: {df[['temp', 'precip', 'wind_speed', 'wind_dir']].describe()}")
            # Impute missing values
            df["temp"] = df["temp"].fillna(df["temp"].mean() if not df["temp"].isna().all() else 25.0)
            df["precip"] = df["precip"].fillna(0)
            df["wind_speed"] = df["wind_speed"].fillna(df["wind_speed"].mean() if not df["wind_speed"].isna().all() else 5.0)
            df["wind_dir"] = df["wind_dir"].fillna(df["wind_dir"].mean() if not df["wind_dir"].isna().all() else 180.0)
            # Log data quality after imputation
            logger.info(f"Post-imputation data quality for {state}: {df[['temp', 'precip', 'wind_speed', 'wind_dir']].describe()}")
            # Ensure no NaN or inf
            if df[['temp', 'precip', 'wind_speed', 'wind_dir']].isna().any().any() or np.isinf(df[['temp', 'precip', 'wind_speed', 'wind_dir']]).any().any():
                logger.warning(f"Invalid values after imputation for {state}")
                return None
            logger.info(f"Historical data for {state}: {df.shape[0]} days")
            return df
        except requests.RequestException as e:
            logger.error(f"Attempt {attempt + 1} failed for {state}: {e}")
            time.sleep(2)
            if attempt == 4:
                logger.error(f"Failed to fetch historical data for {state} after 5 attempts")
                return None
    return None

def get_forecast_weather(day):
    """Predict weather for all Indian states for the specified day."""
    forecast_data = []
    
    # Use Open-Meteo API for days 1-7
    if day <= 7:
        base_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": None,
            "longitude": None,
            "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max,wind_direction_10m_dominant",
            "forecast_days": 7
        }
        
        for state, coords in STATE_COORDS.items():
            params.update({"latitude": coords["lat"], "longitude": coords["lon"]})
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            logger.debug(f"Forecast URL for {state}: {url}")
            for attempt in range(5):
                try:
                    response = requests.get(url, timeout=15)
                    logger.info(f"Forecast response for {state} (day {day}): status={response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    logger.debug(f"Raw forecast response for {state}: {data}")
                    daily = data.get("daily", {})
                    if not daily or not daily.get("time"):
                        logger.warning(f"No forecast data in response for {state} on day {day}")
                        break
                    index = day - 1  # day 1 corresponds to index 0
                    if index >= len(daily["time"]):
                        logger.warning(f"Day {day} out of range for {state}: {len(daily['time'])} days available")
                        break
                    forecast_data.append({
                        "state": STATE_NAME_MAPPING.get(state, state),
                        "temp": daily["temperature_2m_mean"][index],
                        "precip": daily["precipitation_sum"][index],
                        "wind_speed": daily["wind_speed_10m_max"][index],
                        "wind_dir": daily["wind_direction_10m_dominant"][index]
                    })
                    logger.info(f"Forecast fetched for {state} on day {day}: {forecast_data[-1]}")
                    break
                except requests.RequestException as e:
                    logger.error(f"Attempt {attempt + 1} failed for {state}: {e}")
                    time.sleep(2)
                    if attempt == 4:
                        logger.error(f"Failed to fetch forecast for {state} after 5 attempts")
    # Use model for days 8+
    else:
        try:
            model = load_model("models/lstm_weather_model.h5", custom_objects={"weighted_mse": lambda y_true, y_pred: tf.reduce_mean(tf.square(y_true - y_pred))})
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return pd.DataFrame()
        
        try:
            historical_df = pd.read_csv("data/historical_weather.csv")
            norm_params = pd.read_csv("data/normalization_params.csv")
            mean_vals = norm_params["mean"]
            std_vals = norm_params["std"]
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return pd.DataFrame()
        
        for state, coords in STATE_COORDS.items():
            state_data = historical_df[historical_df["state"] == STATE_NAME_MAPPING.get(state, state)]
            if state_data.empty:
                logger.warning(f"No historical data for {state}")
                continue
            
            features = ["temp", "precip", "wind_speed", "wind_dir"]
            recent_data = state_data[features].tail(7).values.astype(np.float32)
            if len(recent_data) < 7:
                logger.warning(f"Insufficient data for {state}: {len(recent_data)} days")
                continue
            
            # Standardize data
            normalized_data = np.zeros_like(recent_data, dtype=np.float32)
            for i, feature in enumerate(features):
                if feature in ["precip", "wind_speed"]:
                    recent_data[:, i] = np.log1p(recent_data[:, i] + 0.1)
                normalized_data[:, i] = (recent_data[:, i] - mean_vals.iloc[i]) / std_vals.iloc[i]
            normalized_data = np.nan_to_num(normalized_data, nan=0.0)
            flattened_data = normalized_data.flatten()
            
            # Iteratively predict up to the specified day
            current_sequence = flattened_data.copy()
            for _ in range(day):
                sequence_input = current_sequence.reshape(1, -1)
                prediction = model.predict(sequence_input, verbose=0)[0]
                current_sequence = np.concatenate([current_sequence[len(features):], prediction])
            
            # Denormalize the final prediction
            prediction = np.zeros_like(current_sequence[-len(features):])
            for i, feature in enumerate(features):
                prediction[i] = current_sequence[-len(features) + i] * std_vals.iloc[i] + mean_vals.iloc[i]
                if feature in ["precip", "wind_speed"]:
                    prediction[i] = np.expm1(prediction[i] - 0.1)
            prediction[1] = max(prediction[1], 0)
            prediction[2] = max(prediction[2], 0)
            prediction[3] = prediction[3] % 360
            
            forecast_data.append({
                "state": STATE_NAME_MAPPING.get(state, state),
                "temp": prediction[0],
                "precip": prediction[1],
                "wind_speed": prediction[2],
                "wind_dir": prediction[3]
            })
    
    df = pd.DataFrame(forecast_data)
    if df.empty:
        logger.error(f"No forecast weather data retrieved for day {day}. Number of states processed: {len(forecast_data)}")
    else:
        logger.info(f"Forecast data shape for day {day}: {df.shape}")
    return df