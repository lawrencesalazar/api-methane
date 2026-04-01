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
# APP INIT
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
# GLOBALS
# ==============================
firebase_db = None
model = None
scaler = None
model_metrics = None

# ==============================
# STARTUP (IMPORTANT FOR RENDER)
# ==============================
@app.on_event("startup")
def startup():
    global firebase_db, model, scaler, model_metrics

    logger.info("🚀 Starting API...")

    # --------------------------
    # FIREBASE INIT
    # --------------------------
    try:
        if not firebase_admin._apps:
            cred_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
            db_url = os.environ.get("FIREBASE_DB_URL")

            if cred_json and db_url:
                cred = credentials.Certificate(json.loads(cred_json))
                firebase_admin.initialize_app(cred, {
                    "databaseURL": db_url
                })
                firebase_db = db.reference()
                logger.info("🔥 Firebase connected")
            else:
                logger.warning("⚠️ Firebase ENV not set")

    except Exception as e:
        logger.error(f"❌ Firebase init failed: {e}")

    # --------------------------
    # LOAD MODEL (JOBLIB)
    # --------------------------
    try:
        if os.path.exists("model.pkl"):
            model = joblib.load("model.pkl")
            scaler = joblib.load("scaler.pkl")
            model_metrics = joblib.load("metrics.pkl")
            logger.info("✅ Model loaded")
        else:
            logger.warning("⚠️ Model files not found")

    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")

# ==============================
# ROOT (REQUIRED FOR RENDER TEST)
# ==============================
@app.get("/")
def root():
    return {"status": "API is running"}

# ==============================
# HEALTH CHECK
# ==============================
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "firebase": firebase_db is not None,
        "model_loaded": model is not None
    }

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
        logger.error(f"❌ summary: {e}")
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
        pred = float(model.predict(features_scaled)[0])

        if pred < 0.4:
            return {"level": "LOW", "score": int(pred*25), "explosion_risk": 5}
        elif pred < 0.7:
            return {"level": "MEDIUM", "score": int(pred*50), "explosion_risk": 25}
        else:
            return {"level": "HIGH", "score": int(pred*100), "explosion_risk": 60}

    except:
        methane = float(data.get("methane", 0))
        if methane < 3:
            return {"level": "LOW", "score": 10, "explosion_risk": 5}
        elif methane < 7:
            return {"level": "MEDIUM", "score": 50, "explosion_risk": 25}
        else:
            return {"level": "HIGH", "score": 90, "explosion_risk": 60}

# ==============================
# API ENDPOINTS
# ==============================
@app.get("/api/sensors")
def sensors():
    return list_sensors()

@app.get("/api/sensor/{sensor_id}")
def summary(sensor_id: str):
    return get_summary(sensor_id)

@app.get("/api/fuzzy/{sensor_id}")
def fuzzy(sensor_id: str):
    return {"risk": get_risk(sensor_id)}

# ==============================
# RUN FOR RENDER
# ==============================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)