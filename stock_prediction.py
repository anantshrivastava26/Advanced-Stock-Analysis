import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from alpha_vantage.timeseries import TimeSeries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Ask user for stock ticker ---
ticker = input("Enter the stock ticker symbol (e.g., AAPL, TSLA, MSFT): ").upper().strip()

# --- Alpha Vantage API ---
api_key = os.getenv("API_KEY")
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

# --- News API ---
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
if NEWS_API_KEY:
    news_url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}"
    response = requests.get(news_url)
    news_data = response.json()
    
    if "articles" in news_data:
        articles = [article["title"] for article in news_data["articles"][:10]]
        print(f"\n📰 Sample News Headlines for {ticker}:")
        for headline in articles:
            print("•", headline)
    else:
        print("⚠️ No news articles found.")
else:
    print("⚠️ NEWS_API_KEY not found in .env, skipping news fetch.")
    articles = ["No news data available."]

# --- Sentiment Analysis ---
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    return analyzer.polarity_scores(text)["compound"]

sentiment_scores = [get_sentiment_score(article) for article in articles]
print("\n🧠 Sentiment Scores:", sentiment_scores)

# --- Stock Price Prediction ---
data["moving_avg"] = data["4. close"].rolling(window=20).mean().shift(1)  # shifted to avoid leakage
data.dropna(inplace=True)

X = data[["moving_avg"]]
y = data["4. close"]

# Chronological split instead of random shuffle
split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# --- Visualization ---
plt.figure(figsize=(12, 4))
plt.plot(y_test.values, label="Actual Price", color="blue")
plt.plot(predictions, label="Predicted Price", color="orange")
plt.title(f"{ticker} Stock Price Prediction (Random Forest)")
plt.xlabel("Samples")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.tight_layout()
plt.show()
