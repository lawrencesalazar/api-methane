# ==============================
# 🔥 GAS MONITORING API (CLEAN)
# ==============================

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

import firebase_admin
from firebase_admin import credentials, db

# ==============================
# CONFIG
# ==============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gas-monitor")
PH_TZ = ZoneInfo("Asia/Manila")
firebase_db = None

# ==============================
# FIREBASE INIT
# ==============================
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

# ==============================
# LIFESPAN
# ==============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_firebase()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# WEBSOCKET
# ==============================
class Manager:
    def __init__(self):
        self.clients = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.append(ws)

    def disconnect(self, ws):
        if ws in self.clients:
            self.clients.remove(ws)

    async def broadcast(self, data):
        for ws in self.clients:
            try:
                await ws.send_json(data)
            except:
                pass

manager = Manager()

# ==============================
# THRESHOLDS
# ==============================
DEFAULT = {
    "methane_low": 5,
    "methane_high": 10,
    "co2_high": 1200
}

# ==============================
# FUZZY LOGIC
# ==============================
def fuzz(v, low, high):
    if v <= low: return 0
    if v >= high: return 1
    return (v - low) / (high - low)

def evaluate_risk(data):
    methane = float(data.get("methane", 0))
    co2 = float(data.get("co2", 0))
    temp = float(data.get("temperature", 0))
    hum = float(data.get("humidity", 0))

    m = fuzz(methane, DEFAULT["methane_low"], DEFAULT["methane_high"])
    c = fuzz(co2, 800, DEFAULT["co2_high"])

    score = (m * 0.6 + c * 0.4) * 100

    if score >= 75:
        level = "HIGH"
    elif score >= 40:
        level = "MEDIUM"
    else:
        level = "LOW"

    # 🔥 Explosion risk (methane critical)
    explosion = 0
    if 5 <= methane <= 15:
        explosion = ((methane - 5) / 10) * 100

    return {
        "level": level,
        "score": round(score, 2),
        "explosion_risk": round(explosion, 2)
    }

# ==============================
# INSERT SENSOR DATA
# ==============================
async def insert_data(data):

    now = datetime.now(PH_TZ)
    key = now.strftime("%Y%m%d_%H%M%S")

    data["timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")
    sid = data["sensor_id"]

    root = db.reference()

    # Save latest
    root.child(f"sensorReadings/latest/{sid}").set(data)

    # Save history
    root.child(f"sensorReadings/history/{sid}/{key}").set(data)

    # Compute risk
    risk = evaluate_risk(data)

    root.child(f"sensorReadings/risk/{sid}/{key}").set(risk)

    # Broadcast
    await manager.broadcast({
        "sensor": sid,
        "data": data,
        "risk": risk
    })

# ==============================
# API: INSERT
# ==============================
@app.post("/api/sensor/insert")
async def insert_api(data: Dict[str, Any], bg: BackgroundTasks):

    # 🔥 Validate (matches ESP32)
    required = ["sensor_id", "methane", "co2", "temperature", "humidity"]

    for r in required:
        if r not in data:
            raise HTTPException(400, f"Missing {r}")

    bg.add_task(insert_data, data)

    return {
        "status": "ok",
        "message": "Data received"
    }

# ==============================
# API: SUMMARY
# ==============================
@app.get("/api/sensor/summary/{sensor_id}")
def summary(sensor_id: str):

    data = db.reference(f"sensorReadings/latest/{sensor_id}").get()

    if not data:
        return {"error": "No data"}

    risk = evaluate_risk(data)

    return {
        "sensor_id": sensor_id,
        "timestamp": data.get("timestamp"),
        "methane": data.get("methane"),
        "co2": data.get("co2"),
        "temperature": data.get("temperature"),
        "humidity": data.get("humidity"),
        "risk": risk
    }

# ==============================
# API: FUZZY
# ==============================
@app.get("/api/fuzzy/{sensor_id}")
def fuzzy(sensor_id: str):

    data = db.reference(f"sensorReadings/latest/{sensor_id}").get()

    if not data:
        return {"error": "No data"}

    risk = evaluate_risk(data)

    return {
        "sensor_id": sensor_id,
        "risk": risk,
        "description": f"Risk is {risk['level']} (Score: {risk['score']})"
    }

# ==============================
# API: SENSORS
# ==============================
@app.get("/api/sensors")
def sensors():
    data = db.reference("sensorReadings/latest").get()
    return list(data.keys()) if data else []

# ==============================
# WEBSOCKET
# ==============================
@app.websocket("/ws")
async def websocket(ws: WebSocket):

    await manager.connect(ws)

    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)