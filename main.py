import os
import json
import logging
import asyncio
import base64
from datetime import datetime
from typing import Dict, Any
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import firebase_admin
from firebase_admin import credentials, db

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 🔥 REQUIRED for Render / headless server
# =========================================================
# CONFIG
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gas-monitoring")

firebase_db = None

# =========================================================
# FIREBASE INIT
# =========================================================
def initialize_firebase():
    global firebase_db
    try:
        if firebase_db is not None:
            return True

        firebase_config_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")

        if firebase_config_json:
            service_account_info = json.loads(firebase_config_json)
            cred = credentials.Certificate(service_account_info)
        else:
            private_key = os.getenv("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n')

            service_account_info = {
                "type": "service_account",
                "project_id": os.getenv("FIREBASE_PROJECT_ID"),
                "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
                "private_key": private_key,
                "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
                "client_id": os.getenv("FIREBASE_CLIENT_ID"),
                "token_uri": "https://oauth2.googleapis.com/token",
            }

            cred = credentials.Certificate(service_account_info)

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                "databaseURL": os.getenv("FIREBASE_DB_URL")
            })

        firebase_db = db.reference()
        logger.info("🔥 Firebase Connected")
        return True

    except Exception as e:
        logger.error(f"Firebase init error: {e}")
        return False

# =========================================================
# LIFESPAN
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    if initialize_firebase():
        logger.info("🔥 Firebase initialized")
    else:
        logger.error("❌ Firebase init failed")

    # Check connection immediately
    if is_firebase_connected():
        logger.info("✅ Firebase connection verified")
    else:
        logger.error("❌ Firebase connection test failed")

    yield
# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(
    title="Gas Monitoring API",
    version="4.0.0",
    lifespan=lifespan
)

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

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, data: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
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
    "temp_high": 35,
    "humidity_high": 85
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

    score = (m*0.3 + c*0.3 + a*0.4) * 100

    if score >= 75:
        return {"level": "HIGH", "score": round(score,2)}
    elif score >= 40:
        return {"level": "MEDIUM", "score": round(score,2)}
    return None

def is_firebase_connected() -> bool:
    try:
        if not firebase_admin._apps:
            logger.error("❌ Firebase app not initialized")
            return False

        # Try a simple read operation
        test_ref = db.reference("health_check")
        test_ref.get()

        return True

    except Exception as e:
        logger.error(f"❌ Firebase connection failed: {e}")
        return False
@app.get("/api/health/firebase")
def firebase_health():
    if is_firebase_connected():
        return {"status": "connected"}
    else:
        raise HTTPException(500, "Firebase not connected")
# =========================================================
# INSERT SENSOR DATA
# =========================================================
def insert_sensor(data: Dict[str, Any]):
    if not is_firebase_connected():
        raise Exception("Firebase is not connected")

    now = datetime.now()
    key = now.strftime('%Y%m%d_%H%M%S')
    data["timestamp"] = now.strftime('%Y-%m-%d %H:%M:%S')

    root = db.reference()
    sid = data["sensor_id"]

    root.child(f"sensorReadings/latest/{sid}").set(data)
    root.child(f"sensorReadings/history/{sid}/{key}").set(data)

    alert = evaluate_risk(data)
    if alert:
        root.child(f"sensorReadings/alerts/{sid}/{key}").set(alert)
    if not firebase_admin._apps:
        raise Exception("Firebase not initialized")

    now = datetime.now()
    key = now.strftime('%Y%m%d_%H%M%S')
    data["timestamp"] = now.strftime('%Y-%m-%d %H:%M:%S')

    root = db.reference()
    sid = data["sensor_id"]

    root.child(f"sensorReadings/latest/{sid}").set(data)
    root.child(f"sensorReadings/history/{sid}/{key}").set(data)

    alert = evaluate_risk(data)
    if alert:
        root.child(f"sensorReadings/alerts/{sid}/{key}").set(alert)
# =========================================================
# ROUTES
# =========================================================
@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/api/sensor/insert")
def api_insert(data: Dict[str, Any]):
    insert_sensor(data)
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

@app.get("/api/thresholds/{sensor_id}")
def thresholds(sensor_id: str):
    return get_thresholds(sensor_id)

@app.post("/api/thresholds/update")
def update_thresholds(data: Dict[str, Any]):
    sid = data["sensor_id"]
    db.reference(f"sensorReadings/thresholds/{sid}").set(data)
    return {"status": "updated"}

# =========================================================
# WEBSOCKET ENDPOINT
# =========================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# =========================================================
# METHANE VISUALIZATION
# =========================================================
 @app.get("/api/visualization/methane/{sensor_id}")
def methane(sensor_id: str):
    try:
        data = db.reference(f"sensorReadings/history/{sensor_id}").get()

        if not data:
            raise HTTPException(404, "No data found")

        x = []
        y = []

        for k in sorted(data.keys()):
            record = data[k]

            # ✅ Safe extraction
            timestamp = record.get("timestamp")
            methane = record.get("methane")

            if timestamp is None or methane is None:
                continue

            try:
                methane = float(methane)
            except:
                continue

            x.append(timestamp)
            y.append(methane)

        if len(x) == 0:
            raise HTTPException(400, "No valid data to plot")

        # ✅ Plot
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, marker='o')

        plt.xticks(rotation=45)
        plt.title(f"Methane Levels - {sensor_id}")
        plt.xlabel("Time")
        plt.ylabel("Methane (ppm)")
        plt.tight_layout()

        # ✅ Convert to image
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        image_base64 = base64.b64encode(buf.read()).decode()

        plt.close()  # 🔥 IMPORTANT (prevents memory leak)

        return {
            "sensor_id": sensor_id,
            "points": len(x),
            "image": image_base64
        }

    except Exception as e:
        logger.error(f"Visualization error: {e}")
        raise HTTPException(500, str(e))
    
# =========================================================
# RUN
# =========================================================
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)