"""
PHIL-TEXT API — Giriş noktası

Çalıştırma:
    python app.py
    uvicorn app:app --reload --port 8000

Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
Health:      http://localhost:8000/health
"""
import uvicorn
from src.api.main import app  # noqa: F401 — uvicorn için gerekli

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
