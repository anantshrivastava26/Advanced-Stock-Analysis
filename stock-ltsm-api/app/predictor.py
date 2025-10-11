# app/predictor.py
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from .config import MODEL_PATH, SCALER_PRICE_PATH, SCALER_SENT_PATH, WINDOW_SIZE
from .utils import create_sequences
import os

_model = None
_price_scaler = None
_sent_scaler = None

def ensure_loaded():
    global _model, _price_scaler, _sent_scaler
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model not found. Train it first via /train.")
        _model = load_model(str(MODEL_PATH))
    if _price_scaler is None:
        _price_scaler = joblib.load(SCALER_PRICE_PATH)
    if _sent_scaler is None:
        _sent_scaler = joblib.load(SCALER_SENT_PATH)
    return _model, _price_scaler, _sent_scaler

def predict_from_sequences(price_series, sent_series):
    """
    price_series, sent_series: raw (unscaled) 1D arrays aligned day-by-day
    We use the last WINDOW_SIZE days to predict next day.
    """
    model, price_scaler, sent_scaler = ensure_loaded()
    prices = np.array(price_series).reshape(-1,1)
    sents = np.array(sent_series).reshape(-1,1)
    scaled_prices = price_scaler.transform(prices)
    scaled_sents = sent_scaler.transform(sents)
    if len(scaled_prices) < WINDOW_SIZE:
        raise ValueError("Need at least WINDOW_SIZE days of data to predict.")
    # take last WINDOW_SIZE
    X = np.column_stack((scaled_prices[-WINDOW_SIZE:,0], scaled_sents[-WINDOW_SIZE:,0])).reshape(1, WINDOW_SIZE, 2)
    pred_scaled = model.predict(X)
    pred = price_scaler.inverse_transform(pred_scaled.reshape(-1,1))[0,0]
    return float(pred)
