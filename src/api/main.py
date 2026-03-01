"""PHIL-TEXT API: FastAPI uygulama fabrikası"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from loguru import logger

from .model_manager import model_manager
from .router import router
from .schemas import HealthResponse

_START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Başlatma: model yükle. Kapatma: kaynakları serbest bırak."""
    logger.info("PHIL-TEXT API başlatılıyor...")
    model_manager.load()
    logger.info("API hazır — http://localhost:8000/docs")
    yield
    logger.info("PHIL-TEXT API kapatılıyor.")


app = FastAPI(
    title="PHIL-TEXT API",
    description=(
        "Felsefe metni sınıflandırma REST API'si.\n\n"
        "10 büyük filozofu TF-IDF + LinearSVC modeli ile sınıflandırır.\n\n"
        "**Desteklenen filozoflar:** Aristoteles, Descartes, Hume, Kant, Locke, "
        "Marcus Aurelius, Nietzsche, Platon, Schopenhauer, Spinoza"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["Siniflandirma"])


@app.get("/", include_in_schema=False)
def root():
    """Swagger UI'ya yönlendir."""
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse, tags=["Sistem"],
         summary="Servis sağlık kontrolü")
def health():
    return HealthResponse(
        status         = "ok",
        model_loaded   = model_manager.is_loaded,
        model_name     = model_manager.meta.get("model_name"),
        uptime_seconds = round(time.time() - _START_TIME, 1),
    )
