import os

import joblib
import pandas as pd

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(features: dict) -> dict:
    model = get_model()
    df = pd.DataFrame([features])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    return {
        "prediction": prediction,
        "label": "malignant" if prediction == 0 else "benign",
        "probability": probability,
    }
