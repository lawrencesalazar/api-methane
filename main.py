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

PH_TZ = ZoneInfo("Asia/Manila")  # 🇵🇭 timezone
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
        if not service_json:
            raise Exception("Missing FIREBASE_SERVICE_ACCOUNT")

        cred = credentials.Certificate(json.loads(service_json))

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                "databaseURL": os.environ.get("FIREBASE_DB_URL")
            })

        firebase_db = db.reference()
        logger.info("🔥 Firebase Connected")
        return True

    except Exception as e:
        logger.error(f"❌ Firebase init error: {e}")
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

# =========================================================
# APP
# =========================================================
app = FastAPI(title="Gas Monitoring API", lifespan=lifespan)

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

    async def broadcast(self, data: dict):
        for ws in self.active_connections:
            try:
                await ws.send_json(data)
            except:
                pass

manager = ConnectionManager()

# =========================================================
# THRESHOLDS
# =========================================================
DEFAULT_THRESHOLDS = {
    "methane_low": 5,
    "methane_high": 10,
    "co2_low": 800,
    "co2_high": 1200,
    "ammonia_low": 10,
    "ammonia_high": 25,
}

def get_thresholds(sensor_id: str):
    try:
        data = db.reference(f"sensorReadings/thresholds/{sensor_id}").get()
        return {**DEFAULT_THRESHOLDS, **(data or {})}
    except:
        return DEFAULT_THRESHOLDS

# =========================================================
# FUZZY LOGIC
# =========================================================
def fuzzify(value, low, high):
    if value <= low: return 0
    if value >= high: return 1
    return (value - low) / (high - low)

def evaluate_risk(data):
    t = get_thresholds(data["sensor_id"])

    m = fuzzify(data["methane"], t["methane_low"], t["methane_high"])
    c = fuzzify(data["co2"], t["co2_low"], t["co2_high"])
    a = fuzzify(data["ammonia"], t["ammonia_low"], t["ammonia_high"])

    score = (m*0.3 + c*0.3 + a*0.4)*100

    if score >= 75:
        return {"level":"HIGH","score":round(score,2)}
    elif score >= 40:
        return {"level":"MEDIUM","score":round(score,2)}
    return None

# =========================================================
# INSERT SENSOR
# =========================================================
async def insert_sensor(data: Dict[str, Any]):
    if not is_firebase_connected():
        raise Exception("Firebase not connected")

    now = datetime.now(PH_TZ)
    key = now.strftime('%Y%m%d_%H%M%S')

    data["timestamp"] = now.strftime('%Y-%m-%d %H:%M:%S')
    sid = data["sensor_id"]

    root = db.reference()

    # latest
    root.child(f"sensorReadings/latest/{sid}").set(data)

    # history
    root.child(f"sensorReadings/history/{sid}/{key}").set(data)

    # alerts
    alert = evaluate_risk(data)
    if alert:
        root.child(f"sensorReadings/alerts/{sid}/{key}").set(alert)

    # 🔥 broadcast
    if manager.active_connections:
        await manager.broadcast(data)

# =========================================================
# API ROUTES
# =========================================================
@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/api/sensor/insert")
async def api_insert(data: Dict[str, Any], background_tasks: BackgroundTasks):
    background_tasks.add_task(insert_sensor, data)
    return {"status": "ok"}

@app.get("/api/sensor/latest/{sensor_id}")
def latest(sensor_id: str):
    return db.reference(f"sensorReadings/latest/{sensor_id}").get()

@app.get("/api/sensor/history/{sensor_id}")
def history(sensor_id: str):
    return db.reference(f"sensorReadings/history/{sensor_id}").get()

@app.get("/api/sensor/alerts/{sensor_id}")
def alerts(sensor_id: str):
    return db.reference(f"sensorReadings/alerts/{sensor_id}").get()

# =========================================================
# CHART API
# =========================================================
@app.get("/api/visualization/chart/{sensor_id}")
def chart(sensor_id: str):
    data = db.reference(f"sensorReadings/history/{sensor_id}").get() or {}

    labels, methane, co2, ammonia = [], [], [], []

    for k in sorted(data.keys()):
        r = data[k]
        labels.append(r.get("timestamp"))
        methane.append(float(r.get("methane",0)))
        co2.append(float(r.get("co2",0)))
        ammonia.append(float(r.get("ammonia",0)))

    return {
        "labels": labels,
        "datasets": [
            {"label":"Methane","data":methane},
            {"label":"CO2","data":co2},
            {"label":"Ammonia","data":ammonia},
        ]
    }

# =========================================================
# HEATMAP API
# =========================================================
@app.get("/api/fuzzy/heatmap/{sensor_id}")
def heatmap(sensor_id: str):
    data = db.reference(f"sensorReadings/latest/{sensor_id}").get()
    if not data:
        raise HTTPException(404,"No data")

    t = get_thresholds(sensor_id)

    return {
        "heatmap":[
            {"label":"Methane","value":min(100,int(data["methane"]/t["methane_high"]*100))},
            {"label":"CO2","value":min(100,int(data["co2"]/t["co2_high"]*100))},
            {"label":"Ammonia","value":min(100,int(data["ammonia"]/t["ammonia_high"]*100))}
        ]
    }

# =========================================================
# HEALTH CHECK
# =========================================================
@app.get("/api/health/firebase")
def health():
    if is_firebase_connected():
        return {"status":"connected"}
    raise HTTPException(500,"Firebase not connected")

# =========================================================
# WEBSOCKET
# =========================================================
@app.websocket("/ws")
async def websocket(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)