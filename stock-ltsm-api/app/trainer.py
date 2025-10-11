# app/trainer.py
import numpy as np
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import tensorflow as tf
from .utils import create_sequences, save_scaler, load_scaler
from .config import MODEL_PATH, SCALER_PRICE_PATH, SCALER_SENT_PATH, WINDOW_SIZE, EPOCHS, BATCH_SIZE
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from datetime import datetime, timedelta

analyzer = SentimentIntensityAnalyzer()

def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    df = df.reset_index()
    # ensure date column
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def fetch_sentiment_for_dates(dates, news_texts=None):
    """
    If you have real news data per date, pass news_texts as dict {date: [texts]}.
    Otherwise we create neutral zeros (placeholder).
    """
    sent_scores = []
    for d in dates:
        texts = []
        if news_texts and d in news_texts:
            texts = news_texts[d]
        if texts:
            # combine and compute compound
            compounds = [analyzer.polarity_scores(t)['compound'] for t in texts]
            sent_scores.append(np.mean(compounds))
        else:
            sent_scores.append(0.0)
    return np.array(sent_scores).reshape(-1, 1)

def prepare_data(df: pd.DataFrame, news_dict=None):
    df = df.copy()
    df = df[['Date', 'Close']]
    df = df.dropna().reset_index(drop=True)
    dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
    # Sentiment aligned per trading day (placeholder)
    sents = fetch_sentiment_for_dates(dates, news_dict)
    prices = df['Close'].values.reshape(-1,1).astype(float)
    price_scaler = MinMaxScaler()
    sent_scaler = MinMaxScaler()
    scaled_prices = price_scaler.fit_transform(prices)
    scaled_sents = sent_scaler.fit_transform(sents)
    # save scalers
    joblib.dump(price_scaler, SCALER_PRICE_PATH)
    joblib.dump(sent_scaler, SCALER_SENT_PATH)
    X, y = create_sequences(scaled_prices, scaled_sents, WINDOW_SIZE)
    return X, y, price_scaler, sent_scaler, df

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train(ticker='AAPL', start='2024-04-01', end=None, news_dict=None, epochs=EPOCHS, batch_size=BATCH_SIZE):
    if end is None:
        end = (pd.Timestamp.today() - pd.Timedelta(days=0)).strftime('%Y-%m-%d')
    df = fetch_stock_data(ticker, start, end)
    X, y, price_scaler, sent_scaler, df_clean = prepare_data(df, news_dict)
    if len(X) == 0:
        raise ValueError("Not enough data to create sequences. Need more rows than WINDOW_SIZE.")
    model = build_model((X.shape[1], X.shape[2]))
    es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[es])
    # save model
    model.save(MODEL_PATH)
    return {
        "model_path": str(MODEL_PATH),
        "rows": len(df_clean),
        "trained_on": f"{start} to {end}"
    }

if __name__ == "__main__":
    print("Training locally (this is typically invoked from the API).")
    r = train()
    print(r)
