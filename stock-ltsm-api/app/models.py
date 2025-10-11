# app/models.py
from pydantic import BaseModel, Field
from typing import List, Optional

class TrainRequest(BaseModel):
    ticker: str = Field(default="AAPL")
    start: str = Field(default="2024-04-01")
    end: Optional[str] = None
    epochs: Optional[int] = 100
    batch_size: Optional[int] = 32

class PredictRequest(BaseModel):
    prices: List[float]      # aligned list of historic close prices (last N days, N >= WINDOW_SIZE)
    sents: List[float]       # aligned list of sentiment scores per day (-1..1), same length as prices

class PredictResponse(BaseModel):
    predicted_price: float
    input_length: int
