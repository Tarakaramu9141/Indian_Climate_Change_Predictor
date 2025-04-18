import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import GlorotNormal
from datetime import datetime, timedelta
from utils.weather import get_historical_data, STATE_COORDS, STATE_NAME_MAPPING
import logging
import tensorflow as tf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_historical_data():
    """Fetch historical data for all states."""
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    all_data = []
    
    for state, coords in STATE_COORDS.items():
        logger.info(f"Fetching historical data for {state}")
        df = get_historical_data(state, coords["lat"], coords["lon"], start_date, end_date)
        if df is not None and not df.empty:
            all_data.append(df)
        else:
            logger.warning(f"No valid historical data for {state}")
    
    if not all_data:
        logger.error("No historical data fetched for any state.")
        return None
    
    historical_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Historical data shape: {historical_df.shape}")
    logger.info(f"Data quality: {historical_df[['temp', 'precip', 'wind_speed', 'wind_dir']].describe()}")
    historical_df.to_csv("data/historical_weather.csv", index=False)
    return historical_df

def prepare_data(df, sequence_length=7):
    """Prepare data for training with standardization."""
    if df is None or df.empty:
        logger.error("No data provided for training.")
        return [], [], None, None
    
    features = ["temp", "precip", "wind_speed", "wind_dir"]
    X, y = [], []
    
    # Clip outliers
    df["temp"] = df["temp"].clip(lower=-10, upper=45)
    df["precip"] = df["precip"].clip(lower=0, upper=500)
    df["wind_speed"] = df["wind_speed"].clip(lower=0, upper=100)
    df["wind_dir"] = df["wind_dir"].clip(lower=0, upper=360)
    
    for state in df["state"].unique():
        state_data = df[df["state"] == state][features].dropna()
        if len(state_data) < sequence_length + 1:
            logger.warning(f"Insufficient data for {state}: {len(state_data)} days")
            continue
        data_array = state_data.values.astype(np.float32)
        for i in range(len(data_array) - sequence_length):
            X.append(data_array[i:i + sequence_length].flatten())
            y.append(data_array[i + sequence_length])
    
    if not X:
        logger.error("No valid training sequences generated.")
        return [], [], None, None
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    logger.info(f"Raw training data shape: X={X.shape}, y={y.shape}")
    
    # Check for NaN or inf
    if np.isnan(X).any() or np.isnan(y).any() or np.isinf(X).any() or np.isinf(y).any():
        logger.error("NaN or inf values found in raw training data.")
        return [], [], None, None
    
    # Standardize features (mean=0, std=1)
    mean_vals = df[features].mean()
    std_vals = df[features].std()
    std_vals = std_vals.where(std_vals != 0, 1.0)
    logger.info(f"Mean values: {mean_vals}")
    logger.info(f"Std values: {std_vals}")
    X_normalized = np.zeros_like(X, dtype=np.float32)
    y_normalized = np.zeros_like(y, dtype=np.float32)
    
    for i in range(len(features)):
        if i == 1 or i == 2:  # precip, wind_speed
            X[:, i::len(features)] = np.log1p(X[:, i::len(features)] + 0.1)
            y[:, i] = np.log1p(y[:, i] + 0.1)
            mean_vals.iloc[i] = np.log1p(mean_vals.iloc[i] + 0.1)
            std_vals.iloc[i] = np.std(np.log1p(df[features[i]] + 0.1))
        X_normalized[:, i::len(features)] = (X[:, i::len(features)] - mean_vals.iloc[i]) / std_vals.iloc[i]
        y_normalized[:, i] = (y[:, i] - mean_vals.iloc[i]) / std_vals.iloc[i]
    
    X_normalized = np.nan_to_num(X_normalized, nan=0.0)
    y_normalized = np.nan_to_num(y_normalized, nan=0.0)
    
    # Add small noise
    X_normalized += np.random.normal(0, 1e-5, X_normalized.shape).astype(np.float32)
    y_normalized += np.random.normal(0, 1e-5, y_normalized.shape).astype(np.float32)
    
    # Final check for valid data
    if np.isnan(X_normalized).any() or np.isnan(y_normalized).any() or np.isinf(X_normalized).any() or np.isinf(y_normalized).any():
        logger.error("NaN or inf values after normalization.")
        return [], [], None, None
    
    # Log sample data
    logger.info(f"Sample X[0]: {X_normalized[0]}")
    logger.info(f"Sample y[0]: {y_normalized[0]}")
    
    logger.info(f"Normalized training data shape: X={X_normalized.shape}, y={y_normalized.shape}")
    return X_normalized, y_normalized, mean_vals, std_vals

def weighted_mse(y_true, y_pred):
    """Custom MSE loss with feature weights."""
    weights = tf.constant([0.4, 0.1, 0.3, 0.2], dtype=tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    squared_diff = tf.square(y_true - y_pred)
    weighted_diff = squared_diff * tf.expand_dims(weights, 0)
    return tf.reduce_mean(weighted_diff, axis=-1)

def train_model():
    """Train and save the model."""
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists("data/historical_weather.csv"):
        df = fetch_historical_data()
        if df is None:
            logger.error("Cannot train model without historical data.")
            return
    else:
        df = pd.read_csv("data/historical_weather.csv")
    
    X, y, mean_vals, std_vals = prepare_data(df)
    if len(X) == 0:
        logger.error("No training data available.")
        return
    
    model = Sequential([
        Dense(4, activation="tanh", input_shape=(7*4,), kernel_initializer=GlorotNormal())
    ])
    
    optimizer = SGD(learning_rate=0.00001, clipnorm=0.5)
    model.compile(optimizer=optimizer, loss=weighted_mse, metrics=[MeanSquaredError()])
    
    # Test predictions before training
    y_pred = model.predict(X[:10], verbose=0)
    logger.info(f"Sample predictions before training: {y_pred[:2]}")
    try:
        loss = tf.reduce_mean(weighted_mse(y[:10], y_pred)).numpy()
        logger.info(f"Initial loss: {loss}")
    except Exception as e:
        logger.error(f"Failed to compute initial loss: {e}")
        return
    
    # Debug first batch
    logger.info("Debugging first batch...")
    with tf.GradientTape() as tape:
        pred = model(X[:32], training=True)
        loss = tf.reduce_mean(weighted_mse(y[:32], pred))
    grads = tape.gradient(loss, model.trainable_variables)
    for var, grad in zip(model.trainable_variables, grads):
        if grad is None:
            logger.error(f"No gradient for {var.name}")
        elif tf.reduce_any(tf.math.is_nan(grad)):
            logger.error(f"NaN gradient for {var.name}")
        else:
            logger.info(f"First batch gradient norm for {var.name}: {tf.norm(grad).numpy()}")
    logger.info(f"First batch loss: {loss.numpy()}")
    
    # Monitor loss and gradients
    class MonitorCallback(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if np.isnan(logs['loss']):
                logger.error(f"NaN loss detected at batch {batch}")
                self.model.stop_training = True
            # Log gradients
            with tf.GradientTape() as tape:
                predictions = self.model(X[:1], training=True)
                loss = tf.reduce_mean(weighted_mse(y[:1], predictions))
            grads = tape.gradient(loss, self.model.trainable_variables)
            for var, grad in zip(self.model.trainable_variables, grads):
                if grad is None or tf.reduce_any(tf.math.is_nan(grad)):
                    logger.warning(f"Invalid gradient for {var.name}")
                else:
                    logger.info(f"Gradient norm for {var.name}: {tf.norm(grad).numpy()}")
    
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1, callbacks=[MonitorCallback()])
    
    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_weather_model.h5")
    logger.info("Model saved to models/lstm_weather_model.h5")
    
    # Save normalization parameters
    pd.DataFrame({"mean": mean_vals, "std": std_vals}).to_csv("data/normalization_params.csv", index=False)
    logger.info("Normalization parameters saved to data/normalization_params.csv")

if __name__ == "__main__":
    train_model()