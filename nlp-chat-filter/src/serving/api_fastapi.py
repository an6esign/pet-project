from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from ..models.infer import load_router

infer = load_router
app = FastAPI(title="Ticket Router (TF-IDF)")

class PredictRequest(BaseModel):
    text: str

class PredictBatchRequest(BaseModel):
    texts: List[str]

class PredictResponse(BaseModel):
    label_id: int
    label_name: str
    proba: Optional[List[float]] = None

class PredictBatchResponse(BaseModel):
    results: List[PredictResponse]
    
@app.get('/health')
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    pred = infer(req.text)[0]
    return pred.__dict__

@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest):
    preds = infer(req.texts)
    return {"results": [p.__dict__ for p in preds]}