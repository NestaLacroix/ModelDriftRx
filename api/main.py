"""
DriftRx - FastAPI application entry point.

Start the server with:
    uvicorn api.main:app --reload

Or via the Makefile:
    make serve
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.drift import router as drift_router
from api.routers.health import router as health_router
from api.routers.history import router as history_router
from api.routers.predict import router as predict_router
from src.utils.config import CONFIG


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Create required directories on startup; nothing special on shutdown."""
    CONFIG.ensure_directories()
    yield


app = FastAPI(
    title="DriftRx API",
    description=(
        "Self-healing ML model monitoring - detects data drift, diagnoses root "
        "causes, auto-retrains models, and serves full incident reports."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Allow any origin so the Streamlit dashboard and CLI tools can reach the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, tags=["monitoring"])
app.include_router(predict_router, tags=["inference"])
app.include_router(drift_router, tags=["drift"])
app.include_router(history_router, tags=["incidents"])

