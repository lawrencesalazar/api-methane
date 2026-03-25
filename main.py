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

# =========================================================
# CONFIG
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gas-monitoring")

PH_TZ = ZoneInfo("Asia/Manila")
firebase_db = None

# =========================================================
# FIREBASE INIT
# =========================================================
def initialize_firebase():
    global firebase_db
    try:
        if firebase_db:
            return True

        service_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
        cred = credentials.Certificate(json.loads(service_json))

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                "databaseURL": os.environ.get("FIREBASE_DB_URL")
            })

        firebase_db = db.reference()
        logger.info("🔥 Firebase Connected")
        return True
    except Exception as e:
        logger.error(f"Firebase error: {e}")
        return False

def is_firebase_connected():
    try:
        if not firebase_admin._apps:
            return False
        db.reference("health_check").get()
        return True
    except:
        return False

# =========================================================
# LIFESPAN
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_firebase()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# WEBSOCKET MANAGER
# =========================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)

    async def broadcast(self, data):
        for ws in self.active_connections:
            try:
                await ws.send_json(data)
            except:
                pass

manager = ConnectionManager()

# =========================================================
# DEFAULT THRESHOLDS
# =========================================================
DEFAULT_THRESHOLDS = {
    "methane_low": 5,
    "methane_high": 10,
    "co2_high": 1200,
    "ammonia_high": 25
}

# =========================================================
# ADAPTIVE THRESHOLDS
# =========================================================
def get_adaptive_thresholds(sensor_id):
    try:
        history = db.reference(f"sensorReadings/history/{sensor_id}").get()
        if not history:
            return DEFAULT_THRESHOLDS

        methane_vals = [
            float(history[k]["methane"])
            for k in history if "methane" in history[k]
        ]

        if len(methane_vals) < 5:
            return DEFAULT_THRESHOLDS

        avg = sum(methane_vals)/len(methane_vals)
        mx = max(methane_vals)

        return {
            **DEFAULT_THRESHOLDS,
            "methane_low": avg * 0.7,
            "methane_high": mx * 0.9
        }

    except:
        return DEFAULT_THRESHOLDS

# =========================================================
# FUZZY LOGIC
# =========================================================
def fuzzify(v, low, high):
    if v <= low: return 0
    if v >= high: return 1
    return (v-low)/(high-low)

def evaluate_risk(data):
    t = get_adaptive_thresholds(data["sensor_id"])
    m = fuzzify(data["methane"], t["methane_low"], t["methane_high"])
    c = fuzzify(data["co2"], 800, t["co2_high"])
    a = fuzzify(data["ammonia"], 10, t["ammonia_high"])

    score = (m*0.4 + c*0.3 + a*0.3)*100

    if score >= 75:
        return {"level":"HIGH","score":round(score,2)}
    elif score >= 40:
        return {"level":"MEDIUM","score":round(score,2)}
    return None

# =========================================================
# PREDICTIVE ALERT
# =========================================================
def predict_trend(sensor_id):
    try:
        history = db.reference(f"sensorReadings/history/{sensor_id}").get()
        if not history:
            return None

        values = [float(history[k]["methane"]) for k in sorted(history.keys())[-10:]]

        if len(values) < 5:
            return None

        slope = values[-1] - values[0]

        if slope > 2:
            return {"type":"PREDICTIVE","message":"Methane rising fast","trend":round(slope,2)}

    except:
        return None

# =========================================================
# INSERT SENSOR
# =========================================================
async def insert_sensor(data):
    now = datetime.now(PH_TZ)
    key = now.strftime('%Y%m%d_%H%M%S')
    data["timestamp"] = now.strftime('%Y-%m-%d %H:%M:%S')
    sid = data["sensor_id"]

    root = db.reference()
    root.child(f"sensorReadings/latest/{sid}").set(data)
    root.child(f"sensorReadings/history/{sid}/{key}").set(data)

    alert = evaluate_risk(data)
    if alert:
        root.child(f"sensorReadings/alerts/{sid}/{key}").set(alert)

    predict = predict_trend(sid)
    if predict:
        root.child(f"sensorReadings/alerts/{sid}/{key}").set(predict)

    await manager.broadcast(data)

# =========================================================
# APIs
# =========================================================
@app.post("/api/sensor/insert")
async def api_insert(data: Dict[str, Any], bg: BackgroundTasks):
    bg.add_task(insert_sensor, data)
    return {"status":"ok"}

@app.get("/api/sensors")
def sensors():
    data = db.reference("sensorReadings/latest").get()
    return list(data.keys()) if data else []

@app.get("/api/visualization/chart/{sensor_id}")
def chart(sensor_id):
    data = db.reference(f"sensorReadings/history/{sensor_id}").get() or {}
    labels, m, c, a = [], [], [], []

    for k in sorted(data.keys()):
        r = data[k]
        labels.append(r["timestamp"])
        m.append(r["methane"])
        c.append(r["co2"])
        a.append(r["ammonia"])

    return {
        "labels": labels,
        "datasets": [
            {"label":"Methane","data":m},
            {"label":"CO2","data":c},
            {"label":"Ammonia","data":a},
        ]
    }

@app.get("/api/forecast/{sensor_id}")
def forecast(sensor_id):
    data = db.reference(f"sensorReadings/history/{sensor_id}").get()
    values = [float(data[k]["methane"]) for k in sorted(data.keys())[-10:]]

    slope = (values[-1]-values[0])/len(values)
    future = []
    last = values[-1]

    for i in range(5):
        last += slope
        future.append(round(last,2))

    return {"forecast":future}

# =========================================================
# WEBSOCKET
# =========================================================
@app.websocket("/ws")
async def ws(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)