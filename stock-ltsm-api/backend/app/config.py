# backend/app/config.py (ensure these are present)
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# If you still want a default static path constant (deprecated when using timestamped names),
# keep these for compatibility but predictor now uses MODEL_DIR globbing.
MODEL_PATH = MODEL_DIR / "lstm_stock_model.h5"
SCALER_PRICE_PATH = MODEL_DIR / "scaler_price.save"
SCALER_SENT_PATH = MODEL_DIR / "scaler_sent.save"

WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', 60))
EPOCHS = int(os.getenv('EPOCHS', 50))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
