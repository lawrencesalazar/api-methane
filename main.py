# main.py
import os
import json
import logging
from typing import List
import numpy as np
import joblib
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
# FASTAPI INIT
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
# ENVIRONMENT VARIABLES
# ==============================
FIREBASE_SERVICE_ACCOUNT = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
FIREBASE_DB_URL = os.environ.get("FIREBASE_DB_URL")
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
SCALER_PATH = os.environ.get("SCALER_PATH", "scaler.pkl")
METRICS_PATH = os.environ.get("METRICS_PATH", "metrics.pkl")

# ==============================
# GLOBALS
# ==============================
firebase_db = None
model = None
scaler = None
model_metrics = None

# ==============================
# STARTUP EVENT
# ==============================
@app.on_event("startup")
def startup_event():
    global firebase_db, model, scaler, model_metrics
    logger.info("🚀 Starting API...")

    # ----- FIREBASE INIT -----
    try:
        if not firebase_admin._apps:
            if not FIREBASE_SERVICE_ACCOUNT or not FIREBASE_DB_URL:
                logger.warning("⚠️ Firebase environment variables not set")
            else:
                cred = credentials.Certificate(json.loads(FIREBASE_SERVICE_ACCOUNT))
                firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
                firebase_db = db.reference()
                logger.info("🔥 Firebase connected")
        else:
            firebase_db = db.reference()
    except Exception as e:
        logger.error(f"❌ Firebase initialization failed: {e}")
        firebase_db = None

    # ----- LOAD ML MODEL -----
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            model_metrics = joblib.load(METRICS_PATH)
            logger.info("✅ Model, scaler, and metrics loaded")
        else:
            logger.warning("⚠️ Model files not found, fuzzy fallback only")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        model = None
        scaler = None
        model_metrics = None

# ==============================
# ROOT & HEALTH
# ==============================
@app.get("/")
def root():
    return {"status": "API running"}

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "firebase_connected": firebase_db is not None,
        "model_loaded": model is not None
    }

@app.get("/api/health/firebase")
def firebase_health():
    if firebase_db is None:
        return {"status": "error", "message": "Firebase not initialized"}
    try:
        data = firebase_db.get()
        return {"status": "connected", "has_data": data is not None}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==============================
# HELPERS
# ==============================
def list_sensors() -> List[str]:
    try:
        if firebase_db is None:
            return []
        data = firebase_db.child("sensorReadings/latest").get() or {}
        return list(data.keys())
    except Exception as e:
        logger.error(f"❌ list_sensors: {e}")
        return []

def get_summary(sensor_id: str):
    try:
        if firebase_db is None:
            return {}
        return firebase_db.child(f"sensorReadings/latest/{sensor_id}").get() or {}
    except Exception as e:
        logger.error(f"❌ get_summary: {e}")
        return {}

def get_risk(sensor_id: str):
    data = get_summary(sensor_id)
    try:
        if model is None or scaler is None:
            raise Exception("Model not loaded")

        features = np.array([
            float(data.get("methane", 0)),
            float(data.get("co2", 0)),
            float(data.get("temperature", 0)),
            float(data.get("humidity", 0))
        ]).reshape(1, -1)

        features_scaled = scaler.transform(features)
        predicted = float(model.predict(features_scaled)[0])

        if predicted < 0.4:
            level, score, explosion = "LOW", int(predicted*25), 5
        elif predicted < 0.7:
            level, score, explosion = "MEDIUM", int(predicted*50), 25
        else:
            level, score, explosion = "HIGH", int(predicted*100), 60

    except Exception as e:
        logger.warning(f"⚠️ ML failed, fallback risk used: {e}")
        methane = float(data.get("methane", 0))
        if methane < 3:
            level, score, explosion = "LOW", 10, 5
        elif methane < 7:
            level, score, explosion = "MEDIUM", 50, 25
        else:
            level, score, explosion = "HIGH", 90, 60

    return {"level": level, "score": score, "explosion_risk": explosion}

def get_forecast(sensor_id: str):
    try:
        if firebase_db is None:
            return []
        history = firebase_db.child(f"sensorReadings/history/{sensor_id}") \
            .order_by_key().limit_to_last(5).get()
        return [float(v["methane"]) for v in history.values()] if history else []
    except Exception as e:
        logger.error(f"❌ get_forecast: {e}")
        return []

def get_metrics(sensor_id: str):
    return model_metrics or {"RMSE": 0.5, "MAE": 0.3}

def get_alerts(sensor_id: str):
    risk = get_risk(sensor_id)
    return [{"type": "METHANE", "level": risk["level"], "score": risk["score"]}]

def get_chart(sensor_id: str):
    try:
        if firebase_db is None:
            return {"timestamps": [], "methane": [], "co2": []}
        history = firebase_db.child(f"sensorReadings/history/{sensor_id}") \
            .order_by_key().limit_to_last(20).get()
        if not history:
            return {"timestamps": [], "methane": [], "co2": []}

        return {
            "timestamps": [v["timestamp"] for v in history.values()],
            "methane": [float(v["methane"]) for v in history.values()],
            "co2": [float(v["co2"]) for v in history.values()]
        }
    except Exception as e:
        logger.error(f"❌ get_chart: {e}")
        return {"timestamps": [], "methane": [], "co2": []}

# ==============================
# API ENDPOINTS
# ==============================
@app.get("/api/sensors")
def api_sensors():
    return list_sensors()

@app.get("/api/sensor/summary/{sensor_id}")
def api_sensor_summary(sensor_id: str):
    return get_summary(sensor_id)

@app.get("/api/fuzzy/{sensor_id}")
def api_sensor_fuzzy(sensor_id: str):
    return {"risk": get_risk(sensor_id)}

@app.get("/api/forecast/{sensor_id}")
def api_sensor_forecast(sensor_id: str):
    return {"forecast": get_forecast(sensor_id)}

@app.get("/api/model/metrics/{sensor_id}")
def api_sensor_metrics(sensor_id: str):
    return get_metrics(sensor_id)

@app.get("/api/sensor/alerts/{sensor_id}")
def api_sensor_alerts(sensor_id: str):
    return {"alerts": get_alerts(sensor_id)}

@app.get("/api/visualization/chart/{sensor_id}")
def api_sensor_chart(sensor_id: str):
    return get_chart(sensor_id)

# ==============================
# RENDER ENTRYPOINT
# ==============================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)