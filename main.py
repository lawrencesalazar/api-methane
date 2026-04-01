# main.py
import os
import json
import logging
from typing import List
import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, db

# ==============================
# LOGGING
# ==============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# ==============================
# FastAPI INIT
# ==============================
app = FastAPI(title="Methane Gas Monitoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local dev / dashboard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# FIREBASE INIT
# ==============================
firebase_db = None

def init_firebase():
    global firebase_db
    try:
        if firebase_admin._apps:
            firebase_db = db.reference()
            return True

        cred_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
        cred = credentials.Certificate(json.loads(cred_json))

        firebase_admin.initialize_app(cred, {
            "databaseURL": os.environ.get("FIREBASE_DB_URL")
        })

        firebase_db = db.reference()
        logger.info("🔥 Firebase Connected")
        return True

    except Exception as e:
        logger.error(f"Firebase error: {e}")
        return False

if not init_firebase():
    logger.error("Firebase connection failed. Exiting...")
    raise SystemExit(1)

# ==============================
# LOAD TRAINED MODEL & SCALER
# ==============================
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
METRICS_PATH = "metrics.pkl"

model = None
scaler = None
model_metrics = None

def load_model():
    global model, scaler, model_metrics
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(METRICS_PATH, "rb") as f:
            model_metrics = pickle.load(f)
        logger.info("✅ Model, scaler, and metrics loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ML artifacts: {e}")
        raise SystemExit(1)

load_model()

# ==============================
# HELPERS
# ==============================
def list_sensors() -> List[str]:
    data = firebase_db.child("/sensorReadings/latest").get() or {}
    return list(data.keys())

def get_summary(sensor_id: str):
    data = firebase_db.child(f"/sensorReadings/latest/{sensor_id}").get() or {}
    return data

def get_risk(sensor_id: str):
    """
    Get fuzzy risk using ML model prediction
    """
    data = get_summary(sensor_id)
    try:
        # Prepare features: order must match training
        features = np.array([
            float(data.get("methane", 0)),
            float(data.get("co2", 0)),
            float(data.get("temperature", 0)),
            float(data.get("humidity", 0))
        ]).reshape(1, -1)

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict numeric risk score
        predicted_risk = float(model.predict(features_scaled)[0])

        # Apply fuzzy logic thresholds
        if predicted_risk < 0.4:
            level, score, explosion_risk = "LOW", int(predicted_risk*25), 5
        elif predicted_risk < 0.7:
            level, score, explosion_risk = "MEDIUM", int(predicted_risk*50), 25
        else:
            level, score, explosion_risk = "HIGH", int(predicted_risk*100), 60

    except Exception as e:
        logger.error(f"Fuzzy risk calculation error for {sensor_id}: {e}")
        # Fallback to threshold logic
        methane = float(data.get("methane", 0))
        if methane < 3:
            level, score, explosion_risk = "LOW", 10, 5
        elif methane < 7:
            level, score, explosion_risk = "MEDIUM", 50, 25
        else:
            level, score, explosion_risk = "HIGH", 90, 60

    return {"level": level, "score": score, "explosion_risk": explosion_risk}

def get_forecast(sensor_id: str):
    history = firebase_db.child(f"/sensorReadings/history/{sensor_id}").order_by_key().limit_to_last(5).get()
    if history:
        return [float(v["methane"]) for v in history.values()]
    return []

def get_metrics(sensor_id: str):
    return model_metrics or {"RMSE": 0.5, "MAE": 0.3}

def get_alerts(sensor_id: str):
    risk = get_risk(sensor_id)
    return [{"type": "METHANE", "level": risk["level"], "score": risk["score"]}]

def get_chart(sensor_id: str):
    history = firebase_db.child(f"/sensorReadings/history/{sensor_id}").order_by_key().limit_to_last(20).get()
    if not history:
        return {"timestamps": [], "methane": [], "co2": []}
    timestamps = [v["timestamp"] for v in history.values()]
    methane = [float(v["methane"]) for v in history.values()]
    co2 = [float(v["co2"]) for v in history.values()]
    return {"timestamps": timestamps, "methane": methane, "co2": co2}

# ==============================
# API ENDPOINTS
# ==============================
@app.get("/api/sensors", response_model=List[str])
def api_sensors():
    return list_sensors()

@app.get("/api/sensor/summary/{sensor_id}")
def api_summary(sensor_id: str):
    return get_summary(sensor_id)

@app.get("/api/fuzzy/{sensor_id}")
def api_fuzzy(sensor_id: str):
    """
    Returns fuzzy logic risk based on ML model prediction
    """
    return {"risk": get_risk(sensor_id)}

@app.get("/api/forecast/{sensor_id}")
def api_forecast(sensor_id: str):
    return {"forecast": get_forecast(sensor_id)}

@app.get("/api/model/metrics/{sensor_id}")
def api_metrics(sensor_id: str):
    """
    Return saved model performance metrics
    """
    return get_metrics(sensor_id)

@app.get("/api/sensor/alerts/{sensor_id}")
def api_alerts(sensor_id: str):
    return {"alerts": get_alerts(sensor_id)}

@app.get("/api/visualization/chart/{sensor_id}")
def api_chart(sensor_id: str):
    return get_chart(sensor_id)