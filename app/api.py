"""FastAPI app: upload an image, get a generated caption back.

Run:
    uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

Then open http://localhost:8000/docs for the interactive Swagger UI.
"""
from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from src.inference.predict import Predictor

#can be also CHECKPOINT_PATH = Path("models/base_resnet_lstm/best_model.pt")
CHECKPOINT_PATH = Path("models/resnet_lstm_regularized/best_model.pt") #change this to which ever arch we choose to use
VOCAB_PATH = Path("data/processed/vocab.json")

app = FastAPI(title="Image Caption Generator", version="1.0")

_predictor: Predictor | None = None


class CaptionResponse(BaseModel):
    caption: str


@app.on_event("startup")
def load_model() -> None:
    global _predictor
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Checkpoint not found at {CHECKPOINT_PATH}. Train a model first.")
    _predictor = Predictor(
        checkpoint_path=CHECKPOINT_PATH,
        vocab_path=VOCAB_PATH,
        device="cpu",  # keep API deployment CPU-safe by default; override if GPU available
    )
    print("Model loaded.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.post("/predict", response_model=CaptionResponse)
async def predict(file: UploadFile = File(...)) -> CaptionResponse:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    caption = _predictor.predict(image)
    return CaptionResponse(caption=caption)