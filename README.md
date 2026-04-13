# Advanced Stock Analysis

A full-stack stock analysis and prediction project with:

- FastAPI backend for data fetching, model training, forecasting, sentiment scoring, and API endpoints.
- React frontend for interactive model selection, charting, metrics, and news sentiment display.

The app fetches stock data from Alpha Vantage, computes a moving-average based feature, trains a selected model, and returns:

- Actual vs predicted values for the test window
- Next-day prediction
- MAE metric
- News headlines with sentiment scores

## Project Structure

```text
Advanced-Stock-Analysis/
  backend/
    app.py
    stock_prediction.py
    requirements.txt
    stock-market-analysis-prediction-using-lstm.ipynb
  frontend/
    package.json
    public/index.html
    src/App.jsx
    src/index.js
    src/styles.css
```

## Tech Stack

### Backend

- FastAPI
- Uvicorn
- pandas, numpy, scikit-learn
- vaderSentiment
- requests, python-dotenv
- Optional: xgboost
- Optional: tensorflow/keras (for LSTM)

### Frontend

- React 18
- Recharts
- react-scripts

## Prerequisites

Install these before running:

- Python 3.10+ recommended
- Node.js 18+ and npm
- Alpha Vantage API key (required)
- News API key (optional, but recommended for headlines)

## Environment Variables

Create a file at `backend/.env`:

```env
API_KEY=your_alpha_vantage_api_key
NEWS_API_KEY=your_newsapi_key
```

Notes:

- `API_KEY` is mandatory. Backend startup will fail if it is missing.
- `NEWS_API_KEY` is optional. If absent, prediction still works but news/sentiment may be empty.

## Install Dependencies

Open terminal at project root.

### 1. Backend dependencies

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 2. Frontend dependencies

In a new terminal:

```powershell
cd frontend
npm install
```

## Run Backend and Frontend

Run both services in separate terminals.

### Terminal A: Start backend

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Backend URL:

- http://localhost:8000
- Swagger docs: http://localhost:8000/docs

### Terminal B: Start frontend

```powershell
cd frontend
npm start
```

Frontend URL:

- http://localhost:3000

The frontend is configured to call the backend at `http://localhost:8000/api/predict`.

## How to Use

1. Open the frontend in browser: `http://localhost:3000`
2. Enter a stock ticker (for example: AAPL, TSLA, MSFT)
3. Select model:
   - linear
   - random_forest
   - gb
   - xgboost
   - lstm
4. Click Run
5. View:
   - Prediction chart (actual vs predicted)
   - Next-day prediction
   - MAE and average sentiment
   - Latest headlines and per-headline sentiment

## Backend API

### POST `/api/predict`

Request body:

```json
{
  "ticker": "AAPL",
  "model": "random_forest"
}
```

Valid `model` values:

- `linear`
- `random_forest`
- `gb` (or `gradient_boosting`)
- `xgboost` (requires xgboost installed)
- `lstm` (requires tensorflow installed)

Response includes:

- `ticker`, `model_name`
- `dates`, `actual`, `predicted`
- `next_day_prediction`
- `headlines`, `sentiment`
- `metrics` (`mae`, `avg_sentiment`)

### POST `/api/save-model`

Placeholder endpoint currently returning a success message. Persistent model saving is not implemented in the training flow yet.

## Optional Dependencies

`requirements.txt` includes optional heavy packages (`xgboost`, `tensorflow`) for additional model choices. If those packages are unavailable:

- App still runs for other models
- API returns a clear error when unsupported model is requested

## Troubleshooting

### 1. Backend fails with missing API key

Symptom:

- Runtime error about missing Alpha Vantage `API_KEY`

Fix:

- Ensure `backend/.env` exists and contains valid `API_KEY`

### 2. CORS or connection error from frontend

Symptom:

- Browser error when calling backend

Fix:

- Confirm backend is running on port 8000
- Confirm frontend is running on port 3000
- Verify URL in `frontend/src/App.jsx` is `http://localhost:8000/api/predict`

### 3. `xgboost` model fails

Symptom:

- API error: XGBoost not available

Fix:

```powershell
pip install xgboost
```

### 4. `lstm` model fails

Symptom:

- API error: TensorFlow/Keras not available

Fix:

```powershell
pip install tensorflow==2.16.1
```

### 5. PowerShell activation policy error

Fix:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate venv again.

## Development Notes

- Backend split is chronological (80/20) to reduce data leakage in time-series prediction.
- Moving average feature is shifted by 1 day to avoid same-day target leakage.
- News sentiment uses VADER compound scores.

## Future Improvements

- Persist trained models and add model registry
- Add ticker validation and caching
- Add more time-series features and technical indicators
- Add tests for API and frontend workflows
- Add Docker setup for one-command local run
