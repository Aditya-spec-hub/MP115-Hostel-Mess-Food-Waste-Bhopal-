"""FastAPI service for MP115 (hostel mess food waste)."""
import json
import logging
import os
import sys
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
FEATURES = ["students_registered", "attendance_pct", "menu_rating",
            "temp_c", "cooked_kg", "plate_size_cm", "meal", "hostel"]


class JsonFormatter(logging.Formatter):
    """One JSON object per log line."""

    def format(self, record):
        entry = {"level": record.levelname, "message": record.getMessage()}
        entry.update(getattr(record, "extra_fields", {}))
        return json.dumps(entry)


logger = logging.getLogger("mp115")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)  # screen only, never a file
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class PredictRequest(BaseModel):
    students_registered: int = Field(..., ge=0)
    attendance_pct: float = Field(..., ge=0, le=100)
    menu_rating: float = Field(..., ge=0, le=5)
    temp_c: float
    cooked_kg: float = Field(..., ge=0, le=1000)   # client rule 1: above 1000 impossible
    plate_size_cm: float = Field(..., ge=10)       # client rule 2: below 10 is a typo
    meal: str
    hostel: str


model = None


@asynccontextmanager
async def lifespan(app):
    """Load the model once when the service starts."""
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info("model loaded", extra={"extra_fields": {"path": MODEL_PATH}})
    else:
        logger.warning("model file not found", extra={"extra_fields": {"path": MODEL_PATH}})
    yield


app = FastAPI(title="MP115 Food Waste API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    row = pd.DataFrame([req.model_dump()])[FEATURES]
    prediction = float(model.predict(row)[0])
    logger.info("prediction", extra={"extra_fields": {
        "hostel": req.hostel, "meal": req.meal, "prediction": round(prediction, 3)}})
    return {"prediction": prediction}
