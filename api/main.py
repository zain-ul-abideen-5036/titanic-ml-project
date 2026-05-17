"""
api/main.py  —  FastAPI application with beautiful HTML UI + JSON API.

Endpoints:
  GET  /           → Beautiful HTML dashboard
  GET  /health     → JSON health check
  GET  /metrics    → JSON prediction statistics
  POST /predict    → JSON prediction (called by the UI via JavaScript)

Run locally:
    uvicorn api.main:app --reload
    Then open:  http://localhost:8000
"""
import os, sys, logging
from datetime import datetime
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import predict as ml_predict

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="predictions.log", level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Titanic Survival Predictor",
    description="End-to-end ML portfolio project: RandomForest model served via FastAPI.",
    version="2.0.0",
)

# ── In-memory metrics ─────────────────────────────────────────────────────────
_metrics = {
    "total_predictions": 0,
    "survived_count": 0,
    "did_not_survive_count": 0,
    "started_at": datetime.utcnow().isoformat(),
}

# ── Pydantic models ───────────────────────────────────────────────────────────
class PassengerInput(BaseModel):
    pclass  : int   = Field(..., ge=1, le=3,   example=1)
    sex     : str   = Field(...,               example="female")
    age     : float = Field(..., ge=0, le=120, example=29.0)
    sibsp   : int   = Field(..., ge=0,         example=0)
    parch   : int   = Field(..., ge=0,         example=0)
    fare    : float = Field(..., ge=0,         example=100.0)
    embarked: Optional[str] = Field("S",       example="S")

    @field_validator("sex")
    @classmethod
    def val_sex(cls, v):
        if v.lower() not in ("male","female"):
            raise ValueError("sex must be 'male' or 'female'")
        return v.lower()

    @field_validator("embarked")
    @classmethod
    def val_embarked(cls, v):
        if v and v.upper() not in ("C","Q","S"):
            raise ValueError("embarked must be C, Q, or S")
        return v.upper() if v else "S"

class PredictionResponse(BaseModel):
    survived   : bool
    probability: float
    confidence : str
    message    : str

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    """Serve the beautiful HTML UI."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>UI not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.get("/health", tags=["Status"])
def health_check():
    model_ok = os.path.exists("artifacts/model.pkl")
    if not model_ok:
        raise HTTPException(status_code=503,
            detail="Model not found. Run 'python src/train.py' first.")
    return {"status": "healthy", "model": "loaded"}

@app.get("/metrics", tags=["Status"])
def get_metrics():
    total = _metrics["total_predictions"]
    return {
        **_metrics,
        "survival_rate": round(_metrics["survived_count"] / total, 4) if total > 0 else None,
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def make_prediction(passenger: PassengerInput):
    try:
        result = ml_predict(passenger.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    _metrics["total_predictions"] += 1
    if result["survived"]:
        _metrics["survived_count"] += 1
    else:
        _metrics["did_not_survive_count"] += 1

    logging.info(
        f"pclass={passenger.pclass} sex={passenger.sex} age={passenger.age} "
        f"fare={passenger.fare} → survived={result['survived']} prob={result['probability']}"
    )

    pct = f"{result['probability']:.0%}"
    msg = (f"This passenger likely survived (probability: {pct})."
           if result["survived"]
           else f"This passenger likely did not survive (survival probability: {pct}).")

    return PredictionResponse(
        survived=result["survived"],
        probability=result["probability"],
        confidence=result["confidence"],
        message=msg,
    )
