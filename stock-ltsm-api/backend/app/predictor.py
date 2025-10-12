# backend/app/predictor.py
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from .config import MODEL_DIR, SCALER_PRICE_PATH, SCALER_SENT_PATH, WINDOW_SIZE
import glob
import os

_model = None
_price_scaler = None
_sent_scaler = None
_loaded_model_path = None  # keep track of loaded file, so we can reload if changed

def _find_latest_model_files():
    """
    Find the latest model and matching scalers in MODEL_DIR by timestamp in filename.
    Returns (model_path, price_scaler_path, sent_scaler_path) or (None,None,None)
    """
    models = sorted(glob.glob(os.path.join(MODEL_DIR, "lstm_*.h5")))
    if not models:
        return None, None, None
    latest_model = models[-1]
    # derive timestamp from filename: lstm_<ts>.h5
    base = os.path.basename(latest_model)
    if base.startswith("lstm_") and base.endswith(".h5"):
        ts = base[len("lstm_"):-len(".h5")]
        price_path = os.path.join(MODEL_DIR, f"scaler_price_{ts}.save")
        sent_path = os.path.join(MODEL_DIR, f"scaler_sent_{ts}.save")
        # if scaler files missing, fallback to any scaler files (best effort)
        if not os.path.exists(price_path):
            price_candidates = sorted(glob.glob(os.path.join(MODEL_DIR, "scaler_price_*.save")))
            price_path = price_candidates[-1] if price_candidates else None
        if not os.path.exists(sent_path):
            sent_candidates = sorted(glob.glob(os.path.join(MODEL_DIR, "scaler_sent_*.save")))
            sent_path = sent_candidates[-1] if sent_candidates else None
        return latest_model, price_path, sent_path
    return None, None, None

def ensure_loaded():
    global _model, _price_scaler, _sent_scaler, _loaded_model_path
    model_path, price_path, sent_path = _find_latest_model_files()
    if model_path is None:
        raise FileNotFoundError("Model not found. Train it first via /train.")
    # if different from currently loaded, reload
    if _loaded_model_path != model_path:
        _model = load_model(model_path)
        _price_scaler = joblib.load(price_path) if price_path and os.path.exists(price_path) else None
        _sent_scaler = joblib.load(sent_path) if sent_path and os.path.exists(sent_path) else None
        _loaded_model_path = model_path
    return _model, _price_scaler, _sent_scaler

def predict_from_sequences(price_series, sent_series):
    model, price_scaler, sent_scaler = ensure_loaded()
    if price_scaler is None or sent_scaler is None:
        raise FileNotFoundError("Scalers not found. Ensure model/scalers were saved.")
    prices = np.array(price_series).reshape(-1,1)
    sents = np.array(sent_series).reshape(-1,1)
    scaled_prices = price_scaler.transform(prices)
    scaled_sents = sent_scaler.transform(sents)
    if len(scaled_prices) < WINDOW_SIZE:
        raise ValueError('Need at least WINDOW_SIZE days of data to predict.')
    X = np.column_stack((scaled_prices[-WINDOW_SIZE:,0], scaled_sents[-WINDOW_SIZE:,0])).reshape(1, WINDOW_SIZE, 2)
    pred_scaled = model.predict(X)
    pred = price_scaler.inverse_transform(pred_scaled.reshape(-1,1))[0,0]
    return float(pred)
