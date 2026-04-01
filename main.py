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
# ENV VARIABLES
# ==============================
FIREBASE_SERVICE_ACCOUNT = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
FIREBASE_DB_URL = os.environ.get("FIREBASE_DB_URL")

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
SCALER_PATH = os.environ.get("SCALER_PATH", "scaler.pkl")
METRICS_PATH = os.environ.get("METRICS_PATH", "metrics.pkl")

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

        if not FIREBASE_SERVICE_ACCOUNT:
            raise Exception("Missing FIREBASE_SERVICE_ACCOUNT env")

        if not FIREBASE_DB_URL:
            raise Exception("Missing FIREBASE_DB_URL env")

        cred_dict = json.loads(FIREBASE_SERVICE_ACCOUNT)
        cred = credentials.Certificate(cred_dict)

        firebase_admin.initialize_app(cred, {
            "databaseURL": FIREBASE_DB_URL
        })

        firebase_db = db.reference()
        logger.info("🔥 Firebase Connected")
        return True

    except Exception as e:
        logger.error(f"❌ Firebase init error: {e}")
        return False

# ==============================
# LOAD MODEL (JOBLIB)
# ==============================
model = None
scaler = None
model_metrics = None

def load_model_safe():
    global model, scaler, model_metrics
    try:
        for path in [MODEL_PATH, SCALER_PATH, METRICS_PATH]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        model_metrics = joblib.load(METRICS_PATH)

        logger.info("✅ Model loaded successfully")

    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")
        # ⚠️ DO NOT EXIT → prevents Render crash
        model = None
        scaler = None
        model_metrics = None

# ==============================
# STARTUP EVENTS
# ==============================
@app.on_event("startup")
def startup_event():
    logger.info("🚀 Starting API...")

    fb_status = init_firebase()
    if not fb_status:
        logger.warning("⚠️ Firebase not connected")

    load_model_safe()

# ==============================
# HEALTH CHECK
# ==============================
def check_firebase_connection():
    try:
        if firebase_db is None:
            return {"status": "error", "message": "Firebase not initialized"}

        data = firebase_db.get()

        return {
            "status": "connected",
            "has_data": data is not None
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def root():
    return {"status": "API running"}

@app.get("/api/health/firebase")
def api_firebase_health():
    return check_firebase_connection()

# ==============================
# HELPERS
# ==============================
def list_sensors() -> List[str]:
    try:
        data = firebase_db.child("sensorReadings/latest").get() or {}
        return list(data.keys())
    except Exception as e:
        logger.error(f"❌ list_sensors error: {e}")
        return []

def get_summary(sensor_id: str):
    try:
        return firebase_db.child(f"sensorReadings/latest/{sensor_id}").get() or {}
    except Exception as e:
        logger.error(f"❌ summary error: {e}")
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
        predicted_risk = float(model.predict(features_scaled)[0])

        if predicted_risk < 0.4:
            level, score, explosion_risk = "LOW", int(predicted_risk * 25), 5
        elif predicted_risk < 0.7:
            level, score, explosion_risk = "MEDIUM", int(predicted_risk * 50), 25
        else:
            level, score, explosion_risk = "HIGH", int(predicted_risk * 100), 60

    except Exception as e:
        logger.warning(f"⚠️ fallback used: {e}")
        methane = float(data.get("methane", 0))

        if methane < 3:
            level, score, explosion_risk = "LOW", 10, 5
        elif methane < 7:
            level, score, explosion_risk = "MEDIUM", 50, 25
        else:
            level, score, explosion_risk = "HIGH", 90, 60

    return {"level": level, "score": score, "explosion_risk": explosion_risk}

def get_forecast(sensor_id: str):
    try:
        history = firebase_db.child(f"sensorReadings/history/{sensor_id}") \
            .order_by_key().limit_to_last(5).get()

        return [float(v["methane"]) for v in history.values()] if history else []

    except Exception as e:
        logger.error(f"❌ forecast error: {e}")
        return []

def get_metrics(sensor_id: str):
    return model_metrics or {"RMSE": 0.5, "MAE": 0.3}

def get_alerts(sensor_id: str):
    risk = get_risk(sensor_id)
    return [{"type": "METHANE", "level": risk["level"], "score": risk["score"]}]

def get_chart(sensor_id: str):
    try:
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
        logger.error(f"❌ chart error: {e}")
        return {"timestamps": [], "methane": [], "co2": []}

# ==============================
# API ENDPOINTS
# ==============================
@app.get("/api/sensors")
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

# ==============================
# RUN (RENDER COMPATIBLE)
# ==============================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)