# main.py
import os
import json
import logging
from typing import List
from datetime import datetime
import pytz
import numpy as np
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    allow_origins=["*"],  # Allow dashboard
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
        if not cred_json:
            raise Exception("FIREBASE_SERVICE_ACCOUNT env not set")
        cred = credentials.Certificate(json.loads(cred_json))

        firebase_admin.initialize_app(cred, {
            "databaseURL": os.environ.get("FIREBASE_DB_URL")
        })

        firebase_db = db.reference()
        logger.info("🔥 Firebase Connected")
        return True

    except Exception as e:
        logger.error(f"Firebase init error: {e}")
        return False

if not init_firebase():
    logger.error("Firebase connection failed. Exiting...")
    sys.exit(1)

# ==============================
# SAFE FIREBASE READ
# ==============================
def safe_get(ref, default=None):
    try:
        return ref.get()
    except Exception as e:
        logger.error(f"Firebase read failed: {e}")
        return default

# ==============================
# CHECK FIREBASE CONNECTION
# ==============================
def check_firebase_connection():
    try:
        if firebase_db is None:
            return {"status": "error", "message": "Firebase not initialized"}

        # Test simple read
        data = safe_get(firebase_db.child("sensorReadings/latest"), {})
        return {
            "status": "connected",
            "message": "Firebase reachable",
            "has_data": bool(data)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/health/firebase")
def api_firebase_health():
    return check_firebase_connection()

# ==============================
# LOAD MODEL + SCALER + METRICS
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
        for path in [MODEL_PATH, SCALER_PATH, METRICS_PATH]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing ML artifact: {path}")

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        model_metrics = joblib.load(METRICS_PATH)

        logger.info("✅ Joblib ML artifacts loaded")

    except Exception as e:
        logger.error(f"Failed to load ML artifacts: {e}")
        sys.exit(1)

load_model_safe()

# ==============================
# TIMEZONE
# ==============================
PH_TZ = pytz.timezone("Asia/Manila")

def current_ph_time():
    return datetime.now(PH_TZ).strftime("%Y-%m-%d %H:%M:%S")

# ==============================
# HELPERS
# ==============================
import pandas as pd

def list_sensors() -> List[str]:
    data = safe_get(firebase_db.child("sensorReadings/latest"), {})
    return list(data.keys()) if data else []

def get_summary(sensor_id: str):
    return safe_get(firebase_db.child(f"sensorReadings/latest/{sensor_id}"), {})

def get_risk(sensor_id: str):
    data = get_summary(sensor_id)
    try:
        # Feature names must match training
        features = pd.DataFrame([{
            "methane": float(data.get("methane", 0)),
            "co2": float(data.get("co2", 0)),
            "temperature": float(data.get("temperature", 0)),
            "humidity": float(data.get("humidity", 0))
        }])

        features_scaled = scaler.transform(features)
        predicted_risk = float(model.predict(features_scaled)[0])

        # Fuzzy mapping
        if predicted_risk < 0.4:
            level, score, explosion_risk = "LOW", int(predicted_risk*25), 5
        elif predicted_risk < 0.7:
            level, score, explosion_risk = "MEDIUM", int(predicted_risk*50), 25
        else:
            level, score, explosion_risk = "HIGH", int(predicted_risk*100), 60

    except Exception as e:
        logger.warning(f"ML fallback for {sensor_id}: {e}")
        methane = float(data.get("methane", 0))
        if methane < 3:
            level, score, explosion_risk = "LOW", 10, 5
        elif methane < 7:
            level, score, explosion_risk = "MEDIUM", 50, 25
        else:
            level, score, explosion_risk = "HIGH", 90, 60

    return {"level": level, "score": score, "explosion_risk": explosion_risk}

def get_forecast(sensor_id: str):
    history = safe_get(firebase_db.child(f"sensorReadings/history/{sensor_id}")
                       .order_by_key().limit_to_last(5), {})
    return [float(v.get("methane", 0)) for v in history.values()] if history else []

def get_metrics(sensor_id: str):
    return model_metrics or {"RMSE": 0.5, "MAE": 0.3}

def get_alerts(sensor_id: str):
    risk = get_risk(sensor_id)
    return [{"type": "METHANE", "level": risk["level"], "score": risk["score"]}]

def get_chart(sensor_id: str):
    history = safe_get(firebase_db.child(f"sensorReadings/history/{sensor_id}")
                       .order_by_key().limit_to_last(20), {})

    if not history:
        return {"timestamps": [], "methane": [], "co2": []}

    return {
        "timestamps": [v.get("timestamp") for v in history.values()],
        "methane": [float(v.get("methane", 0)) for v in history.values()],
        "co2": [float(v.get("co2", 0)) for v in history.values()]
    }

# ==============================
# INSERT SENSOR DATA
# ==============================
@app.post("/api/sensor/insert")
async def api_sensor_insert(request: Request):
    try:
        payload = await request.json()
        sensor_id = payload.get("sensor_id")
        if not sensor_id:
            return JSONResponse({"status": "error", "message": "sensor_id required"}, status_code=400)

        # Add timestamp
        payload["timestamp"] = current_ph_time()

        # Save latest
        firebase_db.child(f"sensorReadings/latest/{sensor_id}").set(payload)

        # Save history
        firebase_db.child(f"sensorReadings/history/{sensor_id}/{payload['timestamp']}").set(payload)

        return {"status": "success", "message": "Sensor data inserted", "timestamp": payload["timestamp"]}

    except Exception as e:
        logger.error(f"Insert failed: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

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

# ==============================
# WEBSOCKET FOR REAL-TIME
# ==============================
connected_clients: List[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    try:
        while True:
            data = await ws.receive_text()
            # Optional: Echo or send live updates
            await ws.send_text(f"Server received: {data}")
    except WebSocketDisconnect:
        connected_clients.remove(ws)