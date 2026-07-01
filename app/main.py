"""
FastAPI app serving the cat/dog classifier.

Run with:
    uvicorn app.main:app --reload --port 8000

Then:
    curl -X POST -F "file=@image.jpg" http://localhost:8000/predict
"""

import base64
import io
import logging
import sys
import os

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from config import BEST_MODEL_PATH, CLASS_NAMES  # noqa: E402
from dataset import eval_transforms  # noqa: E402
from model import load_model_for_inference  # noqa: E402
from grad_cam import generate_gradcam_overlay  # noqa: E402
from prediction_logger import log_prediction, compute_stats  # noqa: E402

UNCERTAIN_THRESHOLD = 0.75  # below this confidence, flag as uncertain
ENTROPY_THRESHOLD = 0.55    # normalized entropy above this = model is genuinely confused

import math

def is_uncertain(probs_tensor) -> tuple[bool, str]:
    """
    Two-stage uncertainty check:
    1. Confidence below threshold (model weakly prefers one class)
    2. High entropy (probability mass nearly equally split between classes)
    Returns (uncertain: bool, reason: str)
    """
    p = [probs_tensor[i].item() for i in range(len(probs_tensor))]
    confidence = max(p)

    # Normalized Shannon entropy (0 = certain, 1 = maximum confusion)
    entropy = -sum(pi * math.log(pi + 1e-9) for pi in p)
    max_entropy = math.log(len(p))
    norm_entropy = entropy / max_entropy

    if norm_entropy > ENTROPY_THRESHOLD:
        return True, "high_entropy"
    if confidence < UNCERTAIN_THRESHOLD:
        return True, "low_confidence"
    return False, "confident"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cat vs Dog Classifier", version="1.0")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = None  # loaded on startup


@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(BEST_MODEL_PATH):
        logger.warning(f"No checkpoint found at {BEST_MODEL_PATH}. /predict will fail until trained.")
        return
    model = load_model_for_inference(BEST_MODEL_PATH, device=DEVICE)
    logger.info(f"Model loaded on {DEVICE}")


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_homepage():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "device": DEVICE}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    tensor = eval_transforms(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()

    pred_class = CLASS_NAMES[pred_idx]
    confidence = round(probs[pred_idx].item(), 4)
    uncertain, uncertainty_reason = is_uncertain(probs)

    log_prediction(pred_class, confidence, uncertain, source="predict")

    return {
        "class": pred_class,
        "confidence": confidence,
        "probabilities": {CLASS_NAMES[i]: round(p.item(), 4) for i, p in enumerate(probs)},
        "uncertain": uncertain,
        "uncertainty_reason": uncertainty_reason,
    }


@app.post("/predict-gradcam")
async def predict_gradcam(file: UploadFile = File(...)):
    """
    Same as /predict, but also returns a Grad-CAM heatmap overlay
    (base64-encoded PNG) showing which image regions drove the decision.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    overlay, pred_class, confidence = generate_gradcam_overlay(model, image, device=DEVICE)

    with torch.no_grad():
        tensor2 = eval_transforms(image).unsqueeze(0).to(DEVICE)
        outputs2 = model(tensor2)
        probs2 = torch.softmax(outputs2, dim=1)[0]
    uncertain, uncertainty_reason = is_uncertain(probs2)

    log_prediction(pred_class, confidence, uncertain, source="predict-gradcam")

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "class": pred_class,
        "confidence": confidence,
        "uncertain": uncertain,
        "uncertainty_reason": uncertainty_reason,
        "heatmap_base64": overlay_b64,
    }


@app.get("/stats")
def stats():
    """Aggregated prediction statistics for the dashboard."""
    return compute_stats()


@app.get("/dashboard")
def serve_dashboard():
    return FileResponse(os.path.join(STATIC_DIR, "dashboard.html"))
