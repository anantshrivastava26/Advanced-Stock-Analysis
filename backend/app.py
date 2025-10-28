# backend/app.py
import os
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import joblib
import requests
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# optional imports
try:
    from xgboost import XGBRegressor
    xgb_available = True
except Exception:
    xgb_available = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    tf_available = True
except Exception:
    tf_available = False

load_dotenv()
ALPHA_KEY = os.getenv("API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # optional, required for news fetch

if not ALPHA_KEY:
    raise RuntimeError("Missing ALPHA VANTAGE API_KEY in backend/.env")

app = FastAPI(title="StockSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    ticker: str
    model: str  # e.g. "linear", "random_forest", "gb", "xgboost", "lstm"

def fetch_daily_alpha(ticker: str) -> pd.DataFrame:
    """Fetch daily time series (Alpha Vantage)."""
    base = "https://www.alphavantage.co/query"
    params = {"function": "TIME_SERIES_DAILY", "symbol": ticker, "outputsize": "full", "apikey": ALPHA_KEY}
    r = requests.get(base, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    if "Time Series (Daily)" not in j:
        raise ValueError("Alpha Vantage returned unexpected response: " + str(j))
    df = pd.DataFrame(j["Time Series (Daily)"]).T
    # ensure numeric
    df = df.rename(columns=lambda s: s.strip())
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def fetch_headlines(ticker: str, limit: int = 10):
    """Fetch headlines from NewsAPI (if key present)."""
    if not NEWS_API_KEY:
        return [], []
    url = "https://newsapi.org/v2/everything"
    params = {"q": ticker, "language": "en", "pageSize": limit, "sortBy": "publishedAt", "apiKey": NEWS_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        j = r.json()
        articles = j.get("articles", [])[:limit]
        titles = [a.get("title", "") for a in articles]
        published = [a.get("publishedAt", None) for a in articles]
        return titles, published
    except Exception:
        return [], []

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sent_analyzer = SentimentIntensityAnalyzer()

@app.post("/api/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    ticker = req.ticker.strip().upper()
    model_choice = req.model.strip().lower()

    # fetch price data
    try:
        df = fetch_daily_alpha(ticker)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Alpha Vantage error: {e}")

    # Compute moving average (20) shifted by 1 to avoid leakage
    if "4. close" not in df.columns:
        raise HTTPException(status_code=500, detail="Alpha Vantage output missing close column")
    df["moving_avg"] = df["4. close"].rolling(window=20).mean().shift(1)
    df.dropna(inplace=True)

    # Prepare X/y
    X = df[["moving_avg"]].copy()
    y = df["4. close"].copy()

    # chronological split
    split_idx = int(0.8 * len(X))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # train selected model
    model_name = model_choice
    next_day_pred = None
    history = None

    if model_choice == "linear":
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        # next day prediction: use last available moving_avg (most recent row)
        last_ma = X["moving_avg"].iloc[-1]
        next_day_pred = float(model.predict([[last_ma]])[0])

    elif model_choice == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        last_ma = X["moving_avg"].iloc[-1]
        next_day_pred = float(model.predict([[last_ma]])[0])

    elif model_choice in ("gb", "gradient_boosting"):
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        last_ma = X["moving_avg"].iloc[-1]
        next_day_pred = float(model.predict([[last_ma]])[0])

    elif model_choice == "xgboost":
        if not xgb_available:
            raise HTTPException(status_code=400, detail="XGBoost not available on server")
        model = XGBRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        last_ma = X["moving_avg"].iloc[-1]
        next_day_pred = float(model.predict([[last_ma]])[0])

    elif model_choice == "lstm":
        if not tf_available:
            raise HTTPException(status_code=400, detail="TensorFlow/Keras not available on server")
        # Build univariate sequences from moving_avg
        WINDOW = 20
        feat = X["moving_avg"].values.reshape(-1, 1)
        targ = y.values.reshape(-1, 1)
        # Fit scalers on train only
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        scaler_X.fit(feat[:split_idx])
        scaler_y.fit(targ[:split_idx])
        feat_s = scaler_X.transform(feat)
        targ_s = scaler_y.transform(targ)
        X_seqs = []
        y_seqs = []
        for i in range(WINDOW, len(feat_s)):
            X_seqs.append(feat_s[i-WINDOW:i, 0])
            y_seqs.append(targ_s[i, 0])
        X_seqs = np.array(X_seqs).reshape(-1, WINDOW, 1)
        y_seqs = np.array(y_seqs)
        seq_split = int(0.8 * len(X_seqs))
        X_train_seq, X_test_seq = X_seqs[:seq_split], X_seqs[seq_split:]
        y_train_seq, y_test_seq = y_seqs[:seq_split], y_seqs[seq_split:]

        model = Sequential([
            LSTM(64, input_shape=(WINDOW, 1), return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

        history = model.fit(X_train_seq, y_train_seq, validation_data=(X_test_seq, y_test_seq),
                            epochs=100, batch_size=32, callbacks=[es], verbose=0)

        preds_s = model.predict(X_test_seq)
        preds = scaler_y.inverse_transform(preds_s.reshape(-1, 1)).flatten()
        y_test_vals = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

        # next day: build last WINDOW sequence from most recent moving_avg values
        last_window = feat_s[-WINDOW:].reshape(1, WINDOW, 1)
        next_s = model.predict(last_window)
        next_day_pred = float(scaler_y.inverse_transform(next_s.reshape(-1, 1))[0][0])

        # override y_test variable used below
        y_test = pd.Series(y_test_vals, index=np.arange(len(y_test_vals)))
        # also set preds as numpy array
        preds = np.array(preds)

    else:
        raise HTTPException(status_code=400, detail="Unknown model choice")

    # If not LSTM, get y_test values for metrics (they're pd.Series originally)
    if model_choice != "lstm":
        preds = np.array(preds)
        y_test_vals = y_test.values

    mae = float(mean_absolute_error(y_test_vals, preds))

    # Headlines + sentiment
    headlines, published = fetch_headlines(ticker)
    sentiments = [float(sent_analyzer.polarity_scores(t)["compound"]) for t in headlines] if headlines else []
    avg_sent = float(np.mean(sentiments)) if sentiments else 0.0

    # Prepare timeline arrays (string dates) for charting -- use the test index dates
    test_dates = list(map(lambda d: str(d.date()), X_test.index))
    actual_vals = list(y_test_vals.tolist())
    predicted_vals = list(preds.tolist())

    return {
        "ticker": ticker,
        "model_name": model_choice,
        "dates": test_dates,
        "actual": actual_vals,
        "predicted": predicted_vals,
        "headlines": headlines,
        "sentiment": sentiments,
        "metrics": {"mae": mae, "avg_sentiment": avg_sent},
        "next_day_prediction": next_day_pred
    }


@app.post("/api/save-model")
def save_model(payload: Dict[str, Any]):
    """
    Optional: save the last trained model to disk. For simplicity this endpoint expects:
    { "model_name": "rf_model.joblib", "model_obj": not provided }
    In practice you'd implement model persistence in the training flow. This route is a placeholder.
    """
    return {"ok": True, "msg": "Implement model saving during training (endpoint placeholder)"}
