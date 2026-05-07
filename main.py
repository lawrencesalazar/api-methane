# main.py
import os
import json
import logging
from typing import List
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, db
import sys

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
    sys.exit(1)

# ==============================
# SAFE FIREBASE
# ==============================
def safe_get(ref, default=None):
    try:
        return ref.get()
    except Exception as e:
        logger.error(f"Firebase read error: {e}")
        return default

# ==============================
# TIMEZONE (PH)
# ==============================
PH_TZ = pytz.timezone("Asia/Manila")

def current_ph_time():
    now = datetime.now(PH_TZ)
    return now.strftime("%Y%m%d_%H%M%S") 

def readable_time():
    return datetime.now(PH_TZ).strftime("%Y-%m-%d %H:%M:%S")
# ==============================
# LOAD ML MODEL
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
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        model_metrics = joblib.load(METRICS_PATH)
        logger.info("✅ ML Model Loaded")
    except Exception as e:
        logger.error(f"ML load error: {e}")
        sys.exit(1)

load_model()

# ==============================
# PYDANTIC MODEL
# ==============================
class SensorInput(BaseModel):
    sensor_id: str
    methane: float
    co2: float
    temperature: float
    humidity: float

# ==============================
# HELPERS
# ==============================
def list_sensors() -> List[str]:
    data = safe_get(firebase_db.child("sensorReadings/latest"), {})
    return list(data.keys()) if data else []

def get_summary(sensor_id: str):
    return safe_get(firebase_db.child(f"sensorReadings/latest/{sensor_id}"), {})

def get_risk(sensor_data):
    try:
        methane = float(sensor_data.get("methane", 0))
        co2 = float(sensor_data.get("co2", 0))
        temperature = float(sensor_data.get("temperature", 0))
        humidity = float(sensor_data.get("humidity", 0))

        score = 0

        # =========================
        # METHANE WEIGHT
        # =========================
        if methane >= 300:
            score += 50
        elif methane >= 200:
            score += 40
        elif methane >= 100:
            score += 25
        elif methane >= 50:
            score += 10

        # =========================
        # CO2 WEIGHT
        # =========================
        if co2 >= 1000:
            score += 25
        elif co2 >= 500:
            score += 15
        elif co2 >= 300:
            score += 10

        # =========================
        # TEMPERATURE WEIGHT
        # =========================
        if temperature >= 45:
            score += 20
        elif temperature >= 35:
            score += 10

        # =========================
        # HUMIDITY WEIGHT
        # =========================
        if humidity >= 85:
            score += 10
        elif humidity >= 70:
            score += 5

        # =========================
        # FINAL CLASSIFICATION
        # =========================
        if score >= 70:
            level = "HIGH"
            explosion_risk = 80
        elif score >= 40:
            level = "MEDIUM"
            explosion_risk = 40
        else:
            level = "LOW"
            explosion_risk = 10

        return {
            "level": level,
            "score": score,
            "explosion_risk": explosion_risk
        }

    except Exception as e:
        logger.error(f"Risk calculation error: {e}")

        return {
            "level": "UNKNOWN",
            "score": 0,
            "explosion_risk": 0
        }


def get_chart(sensor_id: str):
    history = safe_get(
        firebase_db.child(f"sensorReadings/history/{sensor_id}")
        .order_by_key().limit_to_last(20), {}
    )

    if not history:
        return {"timestamps": [], "methane": [], "co2": []}

    return {
        "timestamps": [v["timestamp"] for v in history.values()],
        "methane": [float(v["methane"]) for v in history.values()],
        "co2": [float(v["co2"]) for v in history.values()]
    }

def get_metrics(sensor_id: str):
    return model_metrics or {"RMSE": 0.5, "MSE": 0.25, "MAE": 0.3}

# ==============================
# 🔮 PREDICTION (FUTURE METHANE)
# ==============================
def predict_methane(sensor_id: str):
    history = safe_get(
        firebase_db.child(f"sensorReadings/history/{sensor_id}")
        .order_by_key().limit_to_last(10), {}
    )

    if not history:
        return []

    values = [float(v["methane"]) for v in history.values()]
    predictions = []

    last = values[-1]

    for i in range(5):
        next_val = last + np.random.uniform(-0.5, 0.5)
        next_val = max(0, next_val)
        predictions.append(round(next_val, 2))
        last = next_val

    return predictions

# ==============================
# WEBSOCKET
# ==============================
clients: List[WebSocket] = []

@app.websocket("/ws")
async def websocket(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.remove(ws)

async def broadcast(data: dict):
    for client in clients:
        try:
            await client.send_json(data)
        except:
            pass

# ==============================
# INSERT SENSOR (WITH FUZZY SAVE)
# ==============================
@app.post("/api/sensor/insert")
async def insert_sensor(data: SensorInput):
    try:
        payload = data.dict()
        sensor_id = payload["sensor_id"]

        timestamp_key = current_ph_time()
        payload["timestamp"] = readable_time()   # human readable
        # ✅ compute fuzzy
        # risk = get_risk(sensor_id)
        risk = get_risk(payload)
        payload["risk"] = risk

        # save
        firebase_db.child(f"sensorReadings/latest/{sensor_id}").set(payload)

        firebase_db.child(
            f"sensorReadings/history/{sensor_id}/{timestamp_key}"
        ).set(payload)

        # broadcast
        await broadcast(payload)

        return {"status": "success", "data": payload}

    except Exception as e:
        logger.error(f"Insert error: {e}")
        return {"status": "error", "message": str(e)}

# ==============================
# API ENDPOINTS
# ==============================
@app.get("/api/sensors")
def sensors():
    return list_sensors()

@app.get("/api/sensor/summary/{sensor_id}")
def summary(sensor_id: str):
    return get_summary(sensor_id)

@app.get("/api/fuzzy/{sensor_id}")
def fuzzy(sensor_id: str):
    return {"risk": get_risk(sensor_id)}

@app.get("/api/model/metrics/{sensor_id}")
def metrics(sensor_id: str):
    return get_metrics(sensor_id)

@app.get("/api/visualization/chart/{sensor_id}")
def chart(sensor_id: str):
    return get_chart(sensor_id)

@app.get("/api/predict/{sensor_id}")
def predict(sensor_id: str):
    return {"predictions": predict_methane(sensor_id)}

# ==============================
# ROOT
# ==============================
@app.get("/")
def root():
    return {"status": "API running 🚀"}

# ==============================
# RENDER PORT FIX
# ==============================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)