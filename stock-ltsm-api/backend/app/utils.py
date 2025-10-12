import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from .config import SCALER_PRICE_PATH, SCALER_SENT_PATH, WINDOW_SIZE

def create_sequences(prices: np.ndarray, sents: np.ndarray, window: int = WINDOW_SIZE):
    X, y = [], []
    for i in range(window, len(prices)):
        price_seq = prices[i-window:i, 0]
        sent_seq = sents[i-window:i, 0]
        X.append(np.column_stack((price_seq, sent_seq)))
        y.append(prices[i, 0])
    return np.array(X), np.array(y)

def save_scaler(scaler, path):
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)
