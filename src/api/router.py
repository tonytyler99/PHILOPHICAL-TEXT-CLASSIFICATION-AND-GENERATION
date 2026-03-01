"""PHIL-TEXT API: Endpoint router'ı"""
import time

from fastapi import APIRouter, HTTPException

from .model_manager import model_manager
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ModelInfoResponse,
    Prediction,
    PredictRequest,
    PredictResponse,
    Top3Item,
)

router = APIRouter()


def _to_prediction(raw: dict) -> Prediction:
    return Prediction(
        philosopher  = raw["philosopher"],
        confidence   = raw["confidence"],
        top_3        = [Top3Item(**t) for t in raw["top_3"]],
        text_preview = raw["text_preview"],
    )


# ── Tahmin ────────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Tek metin sınıflandırma",
    description="Verilen felsefe metnini 10 filozofdan birine sınıflandırır.",
)
def predict(req: PredictRequest):
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")
    t0 = time.perf_counter()
    try:
        raw = model_manager.predict([req.text])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {exc}") from exc
    elapsed = (time.perf_counter() - t0) * 1000
    return PredictResponse(
        prediction         = _to_prediction(raw[0]),
        model              = model_manager.meta.get("model_name", "unknown"),
        processing_time_ms = round(elapsed, 2),
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Toplu metin sınıflandırma",
    description="En fazla 32 metni tek istekle sınıflandırır.",
)
def predict_batch(req: BatchPredictRequest):
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")
    if len(req.texts) > 32:
        raise HTTPException(status_code=422, detail="Maks. 32 metin gönderilebilir")
    t0 = time.perf_counter()
    try:
        raws = model_manager.predict(req.texts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {exc}") from exc
    elapsed = (time.perf_counter() - t0) * 1000
    return BatchPredictResponse(
        predictions        = [_to_prediction(r) for r in raws],
        count              = len(raws),
        model              = model_manager.meta.get("model_name", "unknown"),
        processing_time_ms = round(elapsed, 2),
    )


# ── Model Bilgisi ─────────────────────────────────────────────────────────────

@router.get(
    "/models/info",
    response_model=ModelInfoResponse,
    summary="Model meta bilgileri",
    description="Yüklü modelin doğruluk metrikleri ve etiket eşlemesini döndürür.",
)
def model_info():
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")
    m = model_manager.meta
    return ModelInfoResponse(
        model_name       = m["model_name"],
        test_accuracy    = m["test_accuracy"],
        test_f1_weighted = m["test_f1_weighted"],
        num_classes      = m["num_classes"],
        label_mapping    = m["label_mapping"],
        train_samples    = m["train_samples"],
        test_samples     = m["test_samples"],
    )


# ── Filozoflar listesi ────────────────────────────────────────────────────────

@router.get(
    "/philosophers",
    summary="Desteklenen filozoflar",
    description="Modelin tahmin edebileceği 10 filozofun listesini döndürür.",
)
def philosophers():
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")
    mapping = model_manager.meta.get("label_mapping", {})
    return {
        "count": len(mapping),
        "philosophers": sorted(mapping.values()),
    }
