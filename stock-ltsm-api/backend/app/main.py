from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from .models import TrainRequest, PredictRequest, PredictResponse
from .trainer import train, fetch_stock_data, compute_sentiments_for_df
from .predictor import predict_from_sequences
from .config import MODEL_PATH, NEWSAPI_KEY, WINDOW_SIZE
import pandas as pd

app = FastAPI(title='Stock LSTM API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/health')
async def health():
    return {'status': 'ok', 'model_exists': MODEL_PATH.exists()}

@app.post('/train')
async def train_endpoint(req: TrainRequest, background_tasks: BackgroundTasks):
    def _run():
        try:
            train(
                ticker=req.ticker,
                start=req.start,
                end=req.end,
                use_news=req.use_news,
                api_key=NEWSAPI_KEY,
                epochs=req.epochs,
                batch_size=req.batch_size
            )
            print('Training completed.')
        except Exception as e:
            print('Training error:', e)
    background_tasks.add_task(_run)
    return {'status': 'training_started', 'ticker': req.ticker}

@app.get('/historical')
async def historical(ticker: str = 'AAPL', start: str = '2024-04-01', end: str = None):
    """Return OHLC data for a given ticker and date range"""
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')
    try:
        df = fetch_stock_data(ticker, start, end)

        # Drop multi-level columns if any (in case yfinance returns tuples)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        # Ensure columns exist
        cols = [c for c in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
        out = df[cols].copy()

        # Cast datetime to string
        if 'Date' in out.columns:
            out['Date'] = pd.to_datetime(out['Date']).dt.strftime('%Y-%m-%d')

        # Convert all NumPy types to Python native
        data = [
            {col: (val.item() if hasattr(val, 'item') else val) for col, val in row.items()}
            for row in out.to_dict(orient='records')
        ]

        return data
    except Exception as e:
        import traceback
        print("Historical endpoint error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/sentiment')
async def sentiment(ticker: str = 'AAPL', start: str = '2024-04-01', end: str = None):
    """Return daily sentiment data for a ticker"""
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')
    try:
        df = fetch_stock_data(ticker, start, end)
        sents = compute_sentiments_for_df(df, ticker, api_key=NEWSAPI_KEY)
        dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()

        # Build serializable list
        data = [
            {"date": d, "sentiment": float(s)}
            for d, s in zip(dates, sents.reshape(-1))
        ]
        return data
    except Exception as e:
        import traceback
        print("Sentiment endpoint error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/predict', response_model=PredictResponse)
async def predict_endpoint(req: PredictRequest):
    try:
        if len(req.prices) != len(req.sents):
            raise HTTPException(status_code=400, detail='prices and sents must be same length')

        if len(req.prices) < WINDOW_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least {WINDOW_SIZE} days of data to predict. Received {len(req.prices)}."
            )

        pred = predict_from_sequences(req.prices, req.sents)
        return PredictResponse(predicted_price=pred, input_length=len(req.prices))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# in backend/app/main.py (train endpoint)
@app.post('/train')
async def train_endpoint(req: TrainRequest, background_tasks: BackgroundTasks):
    # Kick off training as a background task to avoid blocking the request
    def _run():
        try:
            res = train(
                ticker=req.ticker,
                start=req.start,
                end=req.end,
                use_news=req.use_news,
                api_key=NEWSAPI_KEY,
                epochs=req.epochs,
                batch_size=req.batch_size
            )
            print('Training completed. Saved model:', res.get("model_filename"))
        except Exception as e:
            print('Training error:', e)
    background_tasks.add_task(_run)
    return {'status': 'training_started', 'ticker': req.ticker}
