# main.py
import os
import json
import logging
from typing import List
import numpy as np
import joblib   # ✅ switched from pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, db
import sys

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
    allow_origins=["*"],
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
# LOAD MODEL USING JOBLIB
# ==============================
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
METRICS_PATH = "metrics.pkl"

model = None
scaler = None
model_metrics = None

def load_model_safe():
    global model, scaler, model_metrics
    try:
        # Check files exist
        for path in [MODEL_PATH, SCALER_PATH, METRICS_PATH]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")

        # ✅ Load using joblib
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        model_metrics = joblib.load(METRICS_PATH)

        logger.info("✅ Joblib model, scaler, and metrics loaded successfully")

    except FileNotFoundError as e:
        logger.error(f"❌ File error: {e}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Failed to load ML artifacts (Joblib): {e}")
        sys.exit(1)

load_model_safe()

# ==============================
# HELPERS
# ==============================
def list_sensors() -> List[str]:
    data = firebase_db.child("/sensorReadings/latest").get() or {}
    return list(data.keys())

def get_summary(sensor_id: str):
    return firebase_db.child(f"/sensorReadings/latest/{sensor_id}").get() or {}

def get_risk(sensor_id: str):
    """
    ML + Fuzzy Logic Risk
    """
    data = get_summary(sensor_id)

    try:
        # Feature vector (match training order!)
        features = np.array([
            float(data.get("methane", 0)),
            float(data.get("co2", 0)),
            float(data.get("temperature", 0)),
            float(data.get("humidity", 0))
        ]).reshape(1, -1)

        # Scale
        features_scaled = scaler.transform(features)

        # Predict
        predicted_risk = float(model.predict(features_scaled)[0])

        # Fuzzy logic mapping
        if predicted_risk < 0.4:
            level = "LOW"
            score = int(predicted_risk * 25)
            explosion_risk = 5

        elif predicted_risk < 0.7:
            level = "MEDIUM"
            score = int(predicted_risk * 50)
            explosion_risk = 25

        else:
            level = "HIGH"
            score = int(predicted_risk * 100)
            explosion_risk = 60

    except Exception as e:
        logger.warning(f"⚠️ ML failed for {sensor_id}, fallback used: {e}")

        methane = float(data.get("methane", 0))
        if methane < 3:
            level, score, explosion_risk = "LOW", 10, 5
        elif methane < 7:
            level, score, explosion_risk = "MEDIUM", 50, 25
        else:
            level, score, explosion_risk = "HIGH", 90, 60

    return {
        "level": level,
        "score": score,
        "explosion_risk": explosion_risk
    }

def get_forecast(sensor_id: str):
    history = firebase_db.child(f"/sensorReadings/history/{sensor_id}") \
        .order_by_key().limit_to_last(5).get()

    if history:
        return [float(v["methane"]) for v in history.values()]
    return []

def get_metrics(sensor_id: str):
    return model_metrics or {"RMSE": 0.5, "MAE": 0.3}

def get_alerts(sensor_id: str):
    risk = get_risk(sensor_id)
    return [{"type": "METHANE", "level": risk["level"], "score": risk["score"]}]

def get_chart(sensor_id: str):
    history = firebase_db.child(f"/sensorReadings/history/{sensor_id}") \
        .order_by_key().limit_to_last(20).get()

    if not history:
        return {"timestamps": [], "methane": [], "co2": []}

    timestamps = [v["timestamp"] for v in history.values()]
    methane = [float(v["methane"]) for v in history.values()]
    co2 = [float(v["co2"]) for v in history.values()]

    return {
        "timestamps": timestamps,
        "methane": methane,
        "co2": co2
    }

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
    return {"risk": get_risk(sensor_id)}

@app.get("/api/forecast/{sensor_id}")
def api_forecast(sensor_id: str):
    return {"forecast": get_forecast(sensor_id)}

@app.get("/api/model/metrics/{sensor_id}")
def api_metrics(sensor_id: str):
    return get_metrics(sensor_id)

@app.get("/api/sensor/alerts/{sensor_id}")
def api_alerts(sensor_id: str):
    return {"alerts": get_alerts(sensor_id)}

@app.get("/api/visualization/chart/{sensor_id}")
def api_chart(sensor_id: str):
    return get_chart(sensor_id)