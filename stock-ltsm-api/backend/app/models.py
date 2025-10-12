from pydantic import BaseModel
from typing import List, Optional

class TrainRequest(BaseModel):
    ticker: str = 'AAPL'
    start: str = '2024-04-01'
    end: Optional[str] = None
    use_news: Optional[bool] = True
    epochs: Optional[int] = 50
    batch_size: Optional[int] = 32

class PredictRequest(BaseModel):
    prices: List[float]
    sents: List[float]

class PredictResponse(BaseModel):
    predicted_price: float
    input_length: int
