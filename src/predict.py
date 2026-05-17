"""
src/predict.py  —  Load model and predict for a single passenger.
"""
import os, sys
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess_for_inference

MODEL_PATH = "artifacts/model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run 'python src/train.py' first.")
    return joblib.load(MODEL_PATH)

def predict(passenger_data: dict) -> dict:
    model = load_model()
    X = preprocess_for_inference(passenger_data)
    prediction  = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    distance    = abs(probability - 0.5)
    confidence  = "high" if distance >= 0.35 else ("medium" if distance >= 0.15 else "low")
    return {
        "survived"   : bool(prediction),
        "probability": round(float(probability), 4),
        "confidence" : confidence,
    }
