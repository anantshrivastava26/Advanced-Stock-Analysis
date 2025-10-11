# app/utils.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import joblib
from .config import SCALER_PRICE_PATH, SCALER_SENT_PATH, WINDOW_SIZE

def create_sequences(prices: np.ndarray, sents: np.ndarray, window: int = WINDOW_SIZE):
    """
    Input:
      prices: (n,1) array of scaled prices
      sents: (n,1) array of scaled sentiments aligned day-by-day
    Return:
      X shape (num_samples, window, 2), y shape (num_samples,)
    """
    X, y = [], []
    for i in range(window, len(prices)):
        price_seq = prices[i-window:i, 0]
        sent_seq = sents[i-window:i, 0]
        stacked = np.column_stack((price_seq, sent_seq))
        X.append(stacked)
        y.append(prices[i, 0])
    return np.array(X), np.array(y)

def save_scaler(scaler, path):
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)

def fit_or_load_scalers(price_array, sent_array):
    # price_array, sent_array are 2D arrays (n,1)
    try:
        price_scaler = load_scaler(SCALER_PRICE_PATH)
        sent_scaler = load_scaler(SCALER_SENT_PATH)
    except Exception:
        price_scaler = MinMaxScaler()
        sent_scaler = MinMaxScaler()
        price_scaler.fit(price_array)
        sent_scaler.fit(sent_array)
        save_scaler(price_scaler, SCALER_PRICE_PATH)
        save_scaler(sent_scaler, SCALER_SENT_PATH)
    return price_scaler, sent_scaler
