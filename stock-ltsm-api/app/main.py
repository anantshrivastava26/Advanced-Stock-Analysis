# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import TrainRequest, PredictRequest, PredictResponse
from .trainer import train
from .predictor import predict_from_sequences
from .config import MODEL_PATH
import uvicorn
import os

app = FastAPI(title="Stock LSTM API", version="1.0")

# Allow local dev frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status":"ok", "model_exists": MODEL_PATH.exists()}

@app.post("/train")
async def train_endpoint(req: TrainRequest):
    try:
        res = train(ticker=req.ticker, start=req.start, end=req.end, epochs=req.epochs, batch_size=req.batch_size)
        return {"status":"trained", **res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(req: PredictRequest):
    try:
        if len(req.prices) != len(req.sents):
            raise HTTPException(status_code=400, detail="prices and sents must be same length")
        pred = predict_from_sequences(req.prices, req.sents)
        return PredictResponse(predicted_price=pred, input_length=len(req.prices))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
