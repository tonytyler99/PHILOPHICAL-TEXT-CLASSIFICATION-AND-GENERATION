"""PHIL-TEXT: FastAPI Model Serving"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI(title="PHIL-TEXT API", version="0.1.0")
classifier = None
id2label = None

class ClassifyRequest(BaseModel):
    text: str
    top_k: int = 3

class ClassifyResponse(BaseModel):
    predicted_philosopher: str
    confidence: float
    top_predictions: list

@app.on_event("startup")
def load_models():
    global classifier, id2label
    model_path = Path("models/saved/classifier.pkl")
    if model_path.exists():
        classifier = joblib.load(model_path)
    labels_path = Path("models/saved/id2label.json")
    if labels_path.exists():
        import json
        with open(labels_path) as f:
            id2label = {int(k): v for k, v in json.load(f).items()}

@app.get("/")
def root():
    return {"service": "PHIL-TEXT API", "status": "active"}

@app.get("/health")
def health():
    return {"status": "healthy", "classifier_loaded": classifier is not None}

@app.post("/classify", response_model=ClassifyResponse)
def classify_text(request: ClassifyRequest):
    if classifier is None:
        raise HTTPException(503, "Model yuklenmedi")
    try:
        probs = classifier.predict_proba([request.text])[0]
        pred_id = int(probs.argmax())
        top_k = probs.argsort()[-request.top_k:][::-1]
        return ClassifyResponse(
            predicted_philosopher=id2label.get(pred_id, str(pred_id)),
            confidence=round(float(probs[pred_id]), 4),
            top_predictions=[{"philosopher": id2label.get(int(i), str(int(i))),
                              "probability": round(float(probs[i]), 4)} for i in top_k])
    except Exception as e:
        raise HTTPException(400, str(e))
