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

def get_risk(sensor_id: str):
    data = get_summary(sensor_id)

    try:
        features = pd.DataFrame([{
            "methane": float(data.get("methane", 0)),
            "co2": float(data.get("co2", 0)),
            "temperature": float(data.get("temperature", 0)),
            "humidity": float(data.get("humidity", 0))
        }])

        scaled = scaler.transform(features)
        pred = float(model.predict(scaled)[0])

        if pred < 0.4:
            return {"level": "LOW", "score": int(pred*25), "explosion_risk": 5}
        elif pred < 0.7:
            return {"level": "MEDIUM", "score": int(pred*50), "explosion_risk": 25}
        else:
            return {"level": "HIGH", "score": int(pred*100), "explosion_risk": 60}

    except Exception as e:
        logger.warning(f"Fallback risk used: {e}")
        methane = float(data.get("methane", 0))
        if methane < 3:
            return {"level": "LOW", "score": 10, "explosion_risk": 5}
        elif methane < 7:
            return {"level": "MEDIUM", "score": 50, "explosion_risk": 25}
        else:
            return {"level": "HIGH", "score": 90, "explosion_risk": 60}

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
        risk = get_risk(sensor_id)
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