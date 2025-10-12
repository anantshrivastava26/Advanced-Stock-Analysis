# backend/app/trainer.py
import numpy as np
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib
from .utils import create_sequences
from .config import MODEL_PATH, SCALER_PRICE_PATH, SCALER_SENT_PATH, WINDOW_SIZE, EPOCHS, BATCH_SIZE, NEWSAPI_KEY, MODEL_DIR
from .news_client import headlines_by_date
from datetime import datetime
import os
import glob
import shutil

analyzer = SentimentIntensityAnalyzer()

def fetch_stock_data(ticker: str, start: str, end: str):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def compute_sentiments_for_df(df, ticker, api_key=None):
    start = df['Date'].dt.strftime('%Y-%m-%d').iloc[0]
    end = df['Date'].dt.strftime('%Y-%m-%d').iloc[-1]
    news_dict = headlines_by_date(ticker, start, end, api_key=api_key)
    sents = []
    for d in df['Date'].dt.strftime('%Y-%m-%d'):
        texts = news_dict.get(d, [])
        if texts:
            compounds = [analyzer.polarity_scores(t).get('compound', 0.0) for t in texts]
            sents.append(np.mean(compounds))
        else:
            sents.append(0.0)
    return np.array(sents).reshape(-1,1)

def prepare_data(df, ticker, use_news=True, api_key=None):
    df = df[['Date','Close']].dropna().reset_index(drop=True)
    prices = df['Close'].values.reshape(-1,1).astype(float)
    if use_news:
        sents = compute_sentiments_for_df(df, ticker, api_key=api_key)
    else:
        sents = np.zeros_like(prices)
    price_scaler = MinMaxScaler()
    sent_scaler = MinMaxScaler()
    scaled_prices = price_scaler.fit_transform(prices)
    scaled_sents = sent_scaler.fit_transform(sents)
    X, y = create_sequences(scaled_prices, scaled_sents, WINDOW_SIZE)
    return X, y, df, price_scaler, sent_scaler

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def _cleanup_old_files(exclude_prefix):
    """
    Remove older model and scaler files in MODEL_DIR except those starting with exclude_prefix.
    Expected naming:
      lstm_<timestamp>.h5
      scaler_price_<timestamp>.save
      scaler_sent_<timestamp>.save
    """
    patterns = [
        os.path.join(MODEL_DIR, "lstm_*.h5"),
        os.path.join(MODEL_DIR, "scaler_price_*.save"),
        os.path.join(MODEL_DIR, "scaler_sent_*.save"),
    ]
    for pat in patterns:
        for fp in glob.glob(pat):
            fname = os.path.basename(fp)
            if not fname.startswith(exclude_prefix):
                try:
                    os.remove(fp)
                except Exception:
                    try:
                        if os.path.isdir(fp):
                            shutil.rmtree(fp)
                    except Exception:
                        pass

def train(ticker='AAPL', start='2024-04-01', end=None, use_news=True, api_key=None, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Train a new model, save model and scalers with timestamped names, and remove older models/scalers.
    Returns the saved model filename (basename).
    """
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')
    df = fetch_stock_data(ticker, start, end)
    if len(df) < WINDOW_SIZE + 1:
        raise ValueError('Not enough rows for window_size')
    X, y, df_clean, price_scaler, sent_scaler = prepare_data(df, ticker, use_news, api_key)
    model = build_model((X.shape[1], X.shape[2]))
    es = EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[es])

    # timestamp and filenames
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_fname = f"lstm_{ts}.h5"
    price_scaler_fname = f"scaler_price_{ts}.save"
    sent_scaler_fname = f"scaler_sent_{ts}.save"

    model_path = os.path.join(MODEL_DIR, model_fname)
    price_scaler_path = os.path.join(MODEL_DIR, price_scaler_fname)
    sent_scaler_path = os.path.join(MODEL_DIR, sent_scaler_fname)

    # save
    model.save(model_path)
    joblib.dump(price_scaler, price_scaler_path)
    joblib.dump(sent_scaler, sent_scaler_path)

    # cleanup older models/scalers (keep only the new prefix)
    exclude_prefix = f"lstm_{ts}"
    _cleanup_old_files(exclude_prefix)

    return {
        "model_filename": model_fname,
        "scalers": [price_scaler_fname, sent_scaler_fname],
        "rows": len(df_clean),
        "trained_on": f"{start} to {end}",
    }
