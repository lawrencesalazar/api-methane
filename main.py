# ============================================================
# IMPORTS
# ============================================================
import os
import json
import base64
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import firebase_admin
from firebase_admin import credentials, db

# optional fuzzy
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = FastAPI(title="Methane AI API", version="2.0")

# ============================================================
# FIREBASE INIT (REAL DATA)
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
            print("Firebase missing credentials")
            return False

        cred = credentials.Certificate(json.loads(cred_json))
        firebase_admin.initialize_app(cred, {
            "databaseURL": os.environ.get("FIREBASE_DB_URL")
        })

        firebase_db = db.reference()
        print("Firebase Connected")
        return True

    except Exception as e:
        print("Firebase error:", e)
        return False

init_firebase()

# ============================================================
# UTIL
# ============================================================
def readable_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_get(ref):
    try:
        return ref.get()
    except:
        return None

# ============================================================
# PYDANTIC MODEL
# ============================================================
class SensorInput(BaseModel):
    sensor_id: str
    methane: float
    co2: float
    temperature: float
    humidity: float

# ============================================================
# FIREBASE HISTORY PARSER (REAL)
# ============================================================
def get_history(sensor_id):
    data = safe_get(firebase_db.child("sensorReadings/history").child(sensor_id))
    if not data:
        return []

    flat = []

    for group in data.values():
        if isinstance(group, dict):
            for k, v in group.items():
                if isinstance(v, dict) and "methane" in v:
                    flat.append(v)

    flat.sort(key=lambda x: x.get("timestamp", ""))

    return flat

# ============================================================
# FUZZY LOGIC SYSTEM
# ============================================================
class FuzzyLogicSystem:
    def __init__(self):

        self.methane = ctrl.Antecedent(np.arange(0, 1001, 1), "methane")
        self.co2 = ctrl.Antecedent(np.arange(0, 5001, 1), "co2")
        self.temperature = ctrl.Antecedent(np.arange(0, 61, 1), "temperature")
        self.humidity = ctrl.Antecedent(np.arange(0, 101, 1), "humidity")

        self.risk = ctrl.Consequent(np.arange(0, 101, 1), "risk")

        self.methane["low"] = fuzz.trimf(self.methane.universe, [0, 0, 300])
        self.methane["medium"] = fuzz.trimf(self.methane.universe, [200, 450, 700])
        self.methane["high"] = fuzz.trimf(self.methane.universe, [600, 850, 1000])

        self.co2["normal"] = fuzz.trimf(self.co2.universe, [0, 400, 800])
        self.co2["danger"] = fuzz.trimf(self.co2.universe, [2000, 3500, 5000])

        self.temperature["normal"] = fuzz.trimf(self.temperature.universe, [15, 25, 35])
        self.temperature["hot"] = fuzz.trimf(self.temperature.universe, [30, 45, 60])

        self.humidity["normal"] = fuzz.trimf(self.humidity.universe, [40, 60, 80])
        self.humidity["wet"] = fuzz.trimf(self.humidity.universe, [70, 90, 100])

        self.risk["safe"] = fuzz.trimf(self.risk.universe, [0, 10, 25])
        self.risk["low"] = fuzz.trimf(self.risk.universe, [20, 35, 50])
        self.risk["medium"] = fuzz.trimf(self.risk.universe, [45, 60, 75])
        self.risk["high"] = fuzz.trimf(self.risk.universe, [70, 85, 100])

        rules = [
            ctrl.Rule(self.methane["high"], self.risk["high"]),
            ctrl.Rule(self.methane["medium"] & self.co2["danger"], self.risk["medium"]),
            ctrl.Rule(self.methane["low"] & self.co2["normal"], self.risk["safe"]),
        ]

        self.ctrl = ctrl.ControlSystem(rules)

    def calculate(self, m, c, t, h):
        sim = ctrl.ControlSystemSimulation(self.ctrl)
        sim.input["methane"] = m
        sim.input["co2"] = c
        sim.input["temperature"] = t
        sim.input["humidity"] = h
        sim.compute()

        score = float(sim.output["risk"])

        level = "SAFE"
        if score > 80:
            level = "CRITICAL"
        elif score > 60:
            level = "HIGH"
        elif score > 40:
            level = "MEDIUM"
        elif score > 20:
            level = "LOW"

        return {"score": score, "level": level}

fuzzy = FuzzyLogicSystem()

# ============================================================
# ML ENGINE
# ============================================================
class AIEngine:
    def __init__(self):
        self.rf = RandomForestRegressor(n_estimators=120)
        self.ridge = Ridge()
        self.scaler = StandardScaler()
        self.trained = False

        self.rmse = 0
        self.mae = 0
        self.r2 = 0

    def train(self, history):
        if len(history) < 20:
            return {"success": False, "message": "Need 20 records"}

        rows = []
        for i in range(3, len(history)):
            rows.append({
                "m1": history[i-1]["methane"],
                "m2": history[i-2]["methane"],
                "m3": history[i-3]["methane"],
                "c": history[i]["co2"],
                "t": history[i]["temperature"],
                "h": history[i]["humidity"],
                "y": history[i]["methane"]
            })

        df = pd.DataFrame(rows)
        X = df.drop("y", axis=1)
        y = df["y"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.rf.fit(X_train, y_train)
        self.ridge.fit(X_train, y_train)

        pred = (self.rf.predict(X_test)*0.7 + self.ridge.predict(X_test)*0.3)

        self.rmse = np.sqrt(mean_squared_error(y_test, pred))
        self.mae = mean_absolute_error(y_test, pred)
        self.r2 = r2_score(y_test, pred)

        self.trained = True

        return {
            "success": True,
            "training": {
                "accuracy": round(self.r2 * 100, 2),
                "rmse": round(self.rmse, 2),
                "mae": round(self.mae, 2),
                "samples": len(history)
            }
        }

    def predict(self, recent):
        if not self.trained:
            return None

        x = [[
            recent[-1]["methane"],
            recent[-2]["methane"],
            recent[-3]["methane"],
            recent[-1]["co2"],
            recent[-1]["temperature"],
            recent[-1]["humidity"]
        ]]

        x = self.scaler.transform(x)

        return float(
            self.rf.predict(x)[0]*0.7 +
            self.ridge.predict(x)[0]*0.3
        )

ai = AIEngine()

# ============================================================
# ROUTES (ALL FIXED)
# ============================================================

@app.post("/api/sensor/insert")
def insert(data: SensorInput):
    firebase_db.child("sensorReadings/history").child(data.sensor_id).push({
        "methane": data.methane,
        "co2": data.co2,
        "temperature": data.temperature,
        "humidity": data.humidity,
        "sensor_id": data.sensor_id,
        "timestamp": readable_time()
    })
    return {"success": True}

@app.get("/api/sensors")
def sensors():
    data = safe_get(firebase_db.child("sensorReadings/history"))
    return list(data.keys()) if data else []

@app.get("/api/sensor/summary/{sensor_id}")
def summary(sensor_id: str):
    h = get_history(sensor_id)
    if not h:
        return {}

    last = h[-1]
    return last

@app.get("/api/fuzzy/{sensor_id}")
def fuzzy_api(sensor_id: str):
    h = get_history(sensor_id)
    if not h:
        return {}

    last = h[-1]
    return {
        "risk": fuzzy.calculate(
            last["methane"],
            last["co2"],
            last["temperature"],
            last["humidity"]
        )
    }

@app.post("/api/ml/train/{sensor_id}")
def train(sensor_id: str):
    return ai.train(get_history(sensor_id))

@app.get("/api/ml/predict/{sensor_id}")
def predict(sensor_id: str):
    h = get_history(sensor_id)
    return {"prediction": ai.predict(h[-5:])}

@app.get("/api/model/metrics/{sensor_id}")
def metrics(sensor_id: str):
    return {
        "RMSE": ai.rmse,
        "MAE": ai.mae,
        "R2": ai.r2,
        "model_trained": ai.trained
    }

@app.get("/api/visualization/chart/{sensor_id}")
def chart(sensor_id: str, limit: int = 20, offset: int = 0):
    h = get_history(sensor_id)
    sliced = h[::-1][offset:offset+limit]

    return {
        "timestamps": [x["timestamp"] for x in sliced],
        "methane": [x["methane"] for x in sliced],
        "co2": [x["co2"] for x in sliced],
        "total": len(h)
    }

@app.get("/api/history/{sensor_id}")
def history(sensor_id: str):
    return get_history(sensor_id)

@app.get("/api/dashboard/{sensor_id}")
def dashboard(sensor_id: str):
    return {
        "summary": summary(sensor_id),
        "risk": fuzzy_api(sensor_id),
        "ml": metrics(sensor_id)
    }

@app.get("/api/health")
def health():
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
        ]
    }

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    while True:
        await websocket.send_json({
            "status": "LIVE",
            "time": readable_time()
        })

# ==============================
# RENDER PORT FIX
# ==============================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)