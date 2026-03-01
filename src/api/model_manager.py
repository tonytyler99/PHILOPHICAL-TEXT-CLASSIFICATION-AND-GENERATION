"""PHIL-TEXT API: Model yükleme ve tahmin yöneticisi (singleton)"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
from loguru import logger

MODEL_PATH = Path("models/best_model_linearsvc.joblib")
META_PATH  = Path("models/best_model_meta.json")


class ModelManager:
    def __init__(self):
        self._pipeline  = None
        self._meta      = None
        self._id2label  = None
        self._loaded_at = None

    # ── Yükleme ──────────────────────────────────────────────────────────────

    def load(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_PATH}")
        if not META_PATH.exists():
            raise FileNotFoundError(f"Meta dosyası bulunamadı: {META_PATH}")

        logger.info(f"Model yükleniyor: {MODEL_PATH}")
        self._pipeline  = joblib.load(MODEL_PATH)
        self._meta      = json.loads(META_PATH.read_text(encoding="utf-8"))
        self._id2label  = {int(k): v for k, v in self._meta["label_mapping"].items()}
        self._loaded_at = time.time()
        logger.info(f"Model hazır: {self._meta['model_name']} — "
                    f"test_acc={self._meta['test_accuracy']}")

    # ── Özellikler ───────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def meta(self) -> dict:
        return self._meta or {}

    # ── Tahmin ───────────────────────────────────────────────────────────────

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def predict(self, texts: list[str]) -> list[dict]:
        if not self.is_loaded:
            raise RuntimeError("Model henüz yüklenmedi")

        labels = self._pipeline.predict(texts)          # (n,) int
        scores = self._pipeline.decision_function(texts)  # (n, 10) raw scores

        results = []
        for i, text in enumerate(texts):
            probs   = self._softmax(scores[i])
            pred_id = int(labels[i])
            top3    = probs.argsort()[-3:][::-1]

            results.append({
                "philosopher":  self._id2label[pred_id],
                "confidence":   round(float(probs[pred_id]), 4),
                "top_3": [
                    {
                        "philosopher": self._id2label[int(j)],
                        "probability": round(float(probs[j]), 4),
                    }
                    for j in top3
                ],
                "text_preview": text[:120] + "..." if len(text) > 120 else text,
            })
        return results


# Uygulama genelinde tek örnek
model_manager = ModelManager()
