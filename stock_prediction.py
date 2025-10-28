import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from alpha_vantage.timeseries import TimeSeries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import os
from dotenv import load_dotenv

# Optional: XGBoost support (install if not already)
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False
    print("⚠️ XGBoost not installed. Run: pip install xgboost")

# Optional: TensorFlow/Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    tf_available = True
except Exception:
    tf_available = False
    print("⚠️ TensorFlow/Keras not installed. Install with: pip install tensorflow")

# --- Load environment variables ---
load_dotenv()

# --- Ask user for stock ticker ---
ticker = input("Enter the stock ticker symbol (e.g., AAPL, TSLA, MSFT): ").upper().strip()

# --- Alpha Vantage API ---
api_key = os.getenv("API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
if not api_key:
    raise ValueError("❌ Missing API_KEY in .env file")

ts = TimeSeries(key=api_key, output_format="pandas")

print(f"\nFetching stock data for {ticker}...")
try:
    data, meta_data = ts.get_daily(symbol=ticker, outputsize="full")
    print(f"✅ Data for {ticker} fetched successfully!")
except Exception as e:
    print(f"❌ Error fetching data for {ticker}: {e}")
    exit()

# --- News API (using NEWS_API_KEY from .env) ---
news_url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}"

print("\nFetching latest news articles for ...")
print(f"📰 {ticker}")
try:
    response = requests.get(news_url)
    response.raise_for_status()
    news_data = response.json()
    if "articles" in news_data:
        articles = [article["title"] for article in news_data["articles"][:10]]
        print("\n📰 Sample News Headlines:")
        for headline in articles:
            print("•", headline)
    else:
        articles = ["No news data available."]
except Exception as e:
    print(f"❌ Error fetching news: {e}")
    articles = ["No news data available."]

# --- Sentiment Analysis ---
analyzer = SentimentIntensityAnalyzer()
def get_sentiment_score(text):
    return analyzer.polarity_scores(text)["compound"]

sentiment_scores = [get_sentiment_score(article) for article in articles]
print("\n🧠 Sentiment Scores:", sentiment_scores)

# --- Feature Engineering ---
data["moving_avg"] = data["4. close"].rolling(window=20).mean().shift(1)
data.dropna(inplace=True)

X = data[["moving_avg"]]
y = data["4. close"]

# Chronological split (no data leakage)
split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# --- Model Selection Menu ---
print("\n📊 Select a model for prediction:")
print("1️⃣  Linear Regression")
print("2️⃣  Random Forest Regressor")
print("3️⃣  Gradient Boosting Regressor")
if xgb_available:
    print("4️⃣  XGBoost Regressor")
if tf_available:
    print("5️⃣  LSTM (Keras)")

choice = input("\nEnter your choice (1/2/3/4/5): ").strip()

# Default variables
model = None
model_name = "Model"
predictions = None
y_true_for_plot = None

# --- Handle traditional models ---
if choice == "1":
    model = LinearRegression()
    model_name = "Linear Regression"
    print(f"\n🔍 Training {model_name} on {ticker} data...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    y_true_for_plot = y_test.values

elif choice == "2":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model_name = "Random Forest Regressor"
    print(f"\n🔍 Training {model_name} on {ticker} data...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    y_true_for_plot = y_test.values

elif choice == "3":
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_name = "Gradient Boosting Regressor"
    print(f"\n🔍 Training {model_name} on {ticker} data...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    y_true_for_plot = y_test.values

elif choice == "4" and xgb_available:
    model = XGBRegressor(n_estimators=200, random_state=42)
    model_name = "XGBoost Regressor"
    print(f"\n🔍 Training {model_name} on {ticker} data...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    y_true_for_plot = y_test.values

elif choice == "5" and tf_available:
    # --- LSTM pipeline (univariate sequences using moving_avg) ---
    print("\n🔁 User selected LSTM — building LSTM pipeline using the same moving_avg feature...")

    WINDOW = 20  # number of timesteps to look back
    BATCH_SIZE = 32
    EPOCHS = 100
    PATIENCE = 10

    # Prepare values
    feature_vals = X["moving_avg"].values.reshape(-1, 1)  # shape (n,1)
    target_vals = y.values.reshape(-1, 1)

    # Fit scalers on training portion only to avoid leakage
    train_cutoff = split_index
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(feature_vals[:train_cutoff])
    scaler_y.fit(target_vals[:train_cutoff])

    feature_scaled = scaler_X.transform(feature_vals)
    target_scaled = scaler_y.transform(target_vals)

    # build sequences
    X_seqs = []
    y_seqs = []
    for i in range(WINDOW, len(feature_scaled)):
        X_seqs.append(feature_scaled[i-WINDOW:i, 0])  # sequence of length WINDOW
        y_seqs.append(target_scaled[i, 0])
    X_seqs = np.array(X_seqs)  # shape (n_samples, WINDOW)
    y_seqs = np.array(y_seqs)

    # reshape for LSTM: (samples, timesteps, features=1)
    X_seqs = X_seqs.reshape((X_seqs.shape[0], X_seqs.shape[1], 1))

    # Determine new split index for sequences (80% train)
    seq_split_idx = int(0.8 * len(X_seqs))
    X_train_lstm, X_test_lstm = X_seqs[:seq_split_idx], X_seqs[seq_split_idx:]
    y_train_lstm, y_test_lstm = y_seqs[:seq_split_idx], y_seqs[seq_split_idx:]

    # build model
    model = Sequential([
        LSTM(64, input_shape=(WINDOW, 1), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

    # train
    print(f"Training LSTM (epochs={EPOCHS}, batch_size={BATCH_SIZE})...")
    history = model.fit(
        X_train_lstm, y_train_lstm,
        validation_data=(X_test_lstm, y_test_lstm),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=2
    )

    # predict (scaled), then inverse transform
    preds_scaled = model.predict(X_test_lstm)
    predictions = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    y_true_for_plot = scaler_y.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_true_for_plot, predictions)
    print(f"\nLSTM Test MAE: {mae:.4f}")

else:
    # invalid choice or selected option not available
    print("⚠️ Invalid choice or option not available. Defaulting to Random Forest Regressor.")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model_name = "Random Forest Regressor"
    print(f"\n🔍 Training {model_name} on {ticker} data...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    y_true_for_plot = y_test.values

# If a non-LSTM model was used, compute MAE and show plot (LSTM already computed MAE above)
if not (choice == "5" and tf_available):
    mae = mean_absolute_error(y_true_for_plot, predictions)
    print(f"\nTest MAE ({model_name}): {mae:.4f}")

    # --- Visualization for traditional models ---
    plt.figure(figsize=(12, 4))
    plt.plot(y_true_for_plot, label="Actual Price", color="blue")
    plt.plot(predictions, label="Predicted Price", color="orange")
    plt.title(f"{ticker} Stock Price Prediction using {model_name}")
    plt.xlabel("Samples")
    plt.ylabel("Stock Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.show()

print(f"\n✅ Prediction complete using {model_name if model is not None else 'selected model'}!")
