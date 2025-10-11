# app/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lstm_stock_model.h5"
SCALER_PRICE_PATH = MODEL_DIR / "scaler_price.save"
SCALER_SENT_PATH = MODEL_DIR / "scaler_sent.save"

WINDOW_SIZE = 60
EPOCHS = 100
BATCH_SIZE = 32
