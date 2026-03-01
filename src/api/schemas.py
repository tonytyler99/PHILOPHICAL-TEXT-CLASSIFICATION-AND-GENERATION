"""PHIL-TEXT API: Pydantic istek/yanıt şemaları"""
from typing import Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        max_length=10_000,
        description="Sınıflandırılacak felsefe metni (min 10, maks 10.000 karakter)",
        examples=["The soul is immortal and undergoes many incarnations."],
    )


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="Metin listesi (min 1, maks 32 öğe)",
    )


class Top3Item(BaseModel):
    philosopher: str
    probability: float


class Prediction(BaseModel):
    philosopher: str
    confidence: float
    top_3: list[Top3Item]
    text_preview: str


class PredictResponse(BaseModel):
    prediction: Prediction
    model: str
    processing_time_ms: float


class BatchPredictResponse(BaseModel):
    predictions: list[Prediction]
    count: int
    model: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_name: str
    test_accuracy: float
    test_f1_weighted: float
    num_classes: int
    label_mapping: dict[str, str]
    train_samples: int
    test_samples: int
