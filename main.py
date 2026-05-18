# ============================================================
# METHANE AI MONITORING SYSTEM v2.1 (FULL ENTERPRISE STABLE)
# ============================================================

import os
import json
import base64
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import pytz

from datetime import datetime
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel

import firebase_admin
from firebase_admin import credentials, db

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import skfuzzy as fuzz
from skfuzzy import control as ctrl

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("methane-ai")

# ============================================================
# APP INIT
# ============================================================

app = FastAPI(title="Methane AI API v2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

PH_TZ = pytz.timezone("Asia/Manila")

def now():
    return datetime.now(PH_TZ)

def ts():
    return now().strftime("%Y%m%d_%H%M%S")

def readable():
    return now().strftime("%Y-%m-%d %H:%M:%S")

# ============================================================
# FIREBASE
# ============================================================

firebase_db = None

def init_firebase():
    global firebase_db
    try:
        if firebase_admin._apps:
            firebase_db = db.reference()
            return True

        cred_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
        if not cred_json:
            return False

        cred = credentials.Certificate(json.loads(cred_json))
        firebase_admin.initialize_app(cred, {
            "databaseURL": os.environ.get("FIREBASE_DB_URL")
        })

        firebase_db = db.reference()
        return True

    except Exception as e:
        logger.error(e)
        return False

init_firebase()

# ============================================================
# MODELS
# ============================================================

class SensorInput(BaseModel):
    sensor_id: str
    methane: float
    co2: float
    temperature: float
    humidity: float

# ============================================================
# FUZZY SYSTEM (STABLE)
# ============================================================

class FuzzySystem:
    def __init__(self):
        m = ctrl.Antecedent(np.arange(0, 1001, 1), "methane")
        r = ctrl.Consequent(np.arange(0, 101, 1), "risk")

        m["low"] = fuzz.trimf(m.universe, [0, 0, 300])
        m["mid"] = fuzz.trimf(m.universe, [200, 500, 700])
        m["high"] = fuzz.trimf(m.universe, [600, 900, 1000])

        r["safe"] = fuzz.trimf(r.universe, [0, 10, 25])
        r["low"] = fuzz.trimf(r.universe, [20, 35, 50])
        r["med"] = fuzz.trimf(r.universe, [45, 65, 80])
        r["high"] = fuzz.trimf(r.universe, [75, 90, 100])

        rules = [
            ctrl.Rule(m["high"], r["high"]),
            ctrl.Rule(m["mid"], r["med"]),
            ctrl.Rule(m["low"], r["safe"])
        ]

        self.sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

    def run(self, methane):
        self.sim.input["methane"] = methane
        self.sim.compute()

        score = float(self.sim.output["risk"])

        if score > 80:
            level = "CRITICAL"
        elif score > 60:
            level = "HIGH"
        elif score > 40:
            level = "MEDIUM"
        elif score > 20:
            level = "LOW"
        else:
            level = "SAFE"

        return {"score": round(score,2), "level": level}

fuzzy = FuzzySystem()

# ============================================================
# AI ENGINE (FORECAST + ML)
# ============================================================

class AIEngine:
    def __init__(self):
        self.rf = RandomForestRegressor(n_estimators=120)
        self.ridge = Ridge()
        self.scaler = StandardScaler()
        self.trained = False
        self.r2 = 0

    def prepare(self, h):
        rows = []
        for i in range(3, len(h)):
            rows.append({
                "m1": h[i-1]["methane"],
                "m2": h[i-2]["methane"],
                "m3": h[i-3]["methane"],
                "co2": h[i]["co2"],
                "temp": h[i]["temperature"],
                "hum": h[i]["humidity"],
                "y": h[i]["methane"]
            })

        df = pd.DataFrame(rows)
        return df.drop("y", axis=1), df["y"]

    def train(self, h):
        if len(h) < 20:
            return {"success": False, "message": "Need 20+ records"}

        X, y = self.prepare(h)
        Xtr, Xte, ytr, yte = train_test_split(X, y)

        Xtr = self.scaler.fit_transform(Xtr)
        Xte = self.scaler.transform(Xte)

        self.rf.fit(Xtr, ytr)
        self.ridge.fit(Xtr, ytr)

        pred = (self.rf.predict(Xte)*0.7 + self.ridge.predict(Xte)*0.3)

        self.r2 = r2_score(yte, pred)
        self.trained = True

        return {
            "success": True,
            "accuracy": round(self.r2*100,2),
            "rmse": round(np.sqrt(mean_squared_error(yte,pred)),2),
            "mae": round(mean_absolute_error(yte,pred),2),
            "samples": len(h)
        }

    def predict(self, h):
        if not self.trained or len(h) < 5:
            return None

        last = h[-1]

        x = [[
            h[-1]["methane"],
            h[-2]["methane"],
            h[-3]["methane"],
            last["co2"],
            last["temperature"],
            last["humidity"]
        ]]

        x = self.scaler.transform(x)

        rf = self.rf.predict(x)[0]
        rd = self.ridge.predict(x)[0]

        pred = (rf*0.7 + rd*0.3)

        trend = (h[-1]["methane"] - h[-5]["methane"]) / 5
        pred = pred + trend * 0.5

        return max(0, float(pred))

ai = AIEngine()

# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def recommend(risk, methane):
    if risk["level"] == "CRITICAL":
        return "EVACUATE IMMEDIATELY"
    if risk["level"] == "HIGH":
        return "Activate ventilation system"
    if risk["level"] == "MEDIUM":
        return "Increase monitoring"
    return "Safe operation"

# ============================================================
# HISTORY FUNCTION
# ============================================================

def get_history(sensor_id, limit=100):
    data = firebase_db.child(f"sensorReadings/history/{sensor_id}") \
        .order_by_key().limit_to_last(limit).get() or {}

    return [{
        "methane": float(v.get("methane",0)),
        "co2": float(v.get("co2",0)),
        "temperature": float(v.get("temperature",0)),
        "humidity": float(v.get("humidity",0)),
        "timestamp": v.get("timestamp")
    } for v in data.values()]

# ============================================================
# FULL ENDPOINT LIST (REQUIRED)
# ============================================================

@app.get("/")
def root():
    return {
        "status": "Methane AI API Running",
        "version": "2.0",
        "features": [
            "Fuzzy Logic",
            "Machine Learning",
            "Random Forest",
            "Forecasting",
            "Realtime WebSocket",
            "Firebase",
            "ReactJS Ready",
            "Manual Training",
            "Base64 Model Storage"
        ],
        "endpoints": [
            "POST /api/sensor/insert",
            "POST /api/sensor/insert-gsm",
            "GET /api/sensors",
            "GET /api/sensor/summary/{sensor_id}",
            "GET /api/fuzzy/{sensor_id}",
            "GET /api/fuzzy/config",
            "POST /api/ml/train/{sensor_id}",
            "GET /api/ml/predict/{sensor_id}",
            "GET /api/ml/status/{sensor_id}",
            "GET /api/ml/retrain-check/{sensor_id}",
            "GET /api/model/metrics/{sensor_id}",
            "GET /api/visualization/chart/{sensor_id}",
            "GET /api/history/{sensor_id}",
            "GET /api/dashboard/{sensor_id}",
            "GET /api/health",
            "WS /ws"
        ]
    }

# ============================================================
# SENSOR INSERT
# ============================================================

@app.post("/api/sensor/insert")
async def insert(data: SensorInput):
    payload = data.dict()
    payload["timestamp"] = readable()

    risk = fuzzy.run(payload["methane"])
    payload["risk"] = risk
    payload["recommendation"] = recommend(risk, payload["methane"])

    firebase_db.child(f"sensorReadings/latest/{payload['sensor_id']}").set(payload)
    firebase_db.child(f"sensorReadings/history/{payload['sensor_id']}/{ts()}").set(payload)

    return {"success": True, "data": payload}

# ============================================================
# ALL REQUIRED ENDPOINTS (RESTORED)
# ============================================================

@app.get("/api/sensors")
def sensors():
    data = firebase_db.child("sensorReadings/latest").get() or {}
    return list(data.keys())

@app.get("/api/sensor/summary/{sensor_id}")
def summary(sensor_id: str):
    return firebase_db.child(f"sensorReadings/latest/{sensor_id}").get() or {}

@app.get("/api/fuzzy/{sensor_id}")
def fuzzy_api(sensor_id: str):
    latest = firebase_db.child(f"sensorReadings/latest/{sensor_id}").get()
    return {"risk": fuzzy.run(float(latest["methane"])), "sensor": latest}

@app.get("/api/fuzzy/config")
def fuzzy_config():
    return {"system": "Fuzzy Methane Risk v2"}

@app.post("/api/ml/train/{sensor_id}")
def train(sensor_id: str):
    h = get_history(sensor_id, 300)
    return {"training": ai.train(h)}

@app.get("/api/ml/predict/{sensor_id}")
def predict(sensor_id: str):
    h = get_history(sensor_id, 100)
    pred = ai.predict(h)

    latest = h[-1]
    risk = fuzzy.run(pred)

    return {
        "current": latest["methane"],
        "forecast": pred,
        "accuracy": round(ai.r2*100,2),
        "risk": risk,
        "recommendation": recommend(risk, pred),
        "generated_at": readable()
    }

@app.get("/api/ml/status/{sensor_id}")
def status(sensor_id: str):
    return {"trained": ai.trained, "accuracy": round(ai.r2*100,2)}

@app.get("/api/ml/retrain-check/{sensor_id}")
def retrain(sensor_id: str):
    h = get_history(sensor_id, 300)
    return {"should_retrain": len(h) > 50, "data_points": len(h)}

@app.get("/api/model/metrics/{sensor_id}")
def metrics(sensor_id: str):
    return {"accuracy": round(ai.r2*100,2)}

@app.get("/api/visualization/chart/{sensor_id}")
def chart(sensor_id: str, limit: int = 50):
    h = get_history(sensor_id, limit)
    return {
        "timestamps": [x["timestamp"] for x in h],
        "methane": [x["methane"] for x in h],
        "co2": [x["co2"] for x in h]
    }

@app.get("/api/history/{sensor_id}")
def hist(sensor_id: str):
    return {"records": get_history(sensor_id, 200)}

@app.get("/api/dashboard/{sensor_id}")
def dashboard(sensor_id: str):
    latest = firebase_db.child(f"sensorReadings/latest/{sensor_id}").get()
    risk = fuzzy.run(float(latest["methane"]))

    return {
        "latest": latest,
        "risk": risk,
        "forecast": ai.predict(get_history(sensor_id, 10))
    }

@app.get("/api/health")
def health():
    return {
        "status": "ONLINE",
        "server_time": readable(),
        "firebase_connected": firebase_db is not None
    }

# ============================================================
# WEBSOCKET
# ============================================================

clients: List[WebSocket] = []

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except:
        clients.remove(ws)