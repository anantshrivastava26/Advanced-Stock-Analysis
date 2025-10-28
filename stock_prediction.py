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
import os
from dotenv import load_dotenv

# Optional: XGBoost support (install if not already)
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False
    print("⚠️ XGBoost not installed. Run: pip install xgboost")

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

# --- News API (using your provided URL) ---
news_url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}"

print("\nFetching latest news articles for Apple...")
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
choice = input("\nEnter your choice (1/2/3/4): ").strip()

if choice == "1":
    model = LinearRegression()
    model_name = "Linear Regression"
elif choice == "2":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model_name = "Random Forest Regressor"
elif choice == "3":
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_name = "Gradient Boosting Regressor"
elif choice == "4" and xgb_available:
    model = XGBRegressor(n_estimators=200, random_state=42)
    model_name = "XGBoost Regressor"
else:
    print("⚠️ Invalid choice, defaulting to Random Forest.")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model_name = "Random Forest Regressor"

print(f"\n🔍 Training {model_name} on {ticker} data...")

# --- Train and Predict ---
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# --- Visualization ---
plt.figure(figsize=(12, 4))
plt.plot(y_test.values, label="Actual Price", color="blue")
plt.plot(predictions, label="Predicted Price", color="orange")
plt.title(f"{ticker} Stock Price Prediction using {model_name}")
plt.xlabel("Samples")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.tight_layout()
plt.show()

print(f"\n✅ Prediction complete using {model_name}!")
