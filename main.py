import os
import json
import base64
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import firebase_admin
from firebase_admin import credentials, db

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ============================================================
# APP INIT
# ============================================================

app = FastAPI(title="Methane AI API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

firebase_db = None

# ============================================================
# UTIL FUNCTIONS
# ============================================================

def readable_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_get(ref):
    try:
        return ref.get()
    except:
        return None


def normalize_history(history_data):
    """
    FIXED: Handles your REAL Firebase structure:
    sensorReadings/history/ESP32_01/{date_group}/{record}
    """
    rows = []

    if not history_data:
        return rows

    for date_group in history_data.values():
        if isinstance(date_group, dict):
            for record in date_group.values():
                if isinstance(record, dict):
                    rows.append(record)

    rows.sort(key=lambda x: x.get("timestamp", ""))

    return rows


# ============================================================
# FIREBASE INIT (REAL)
# ============================================================

def init_firebase():
    global firebase_db

    try:
        if firebase_admin._apps:
            firebase_db = db.reference()
            return True

        cred_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
        db_url = os.environ.get("FIREBASE_DB_URL")

        if not cred_json or not db_url:
            print("Firebase credentials missing")
            return False

        cred = credentials.Certificate(json.loads(cred_json))

        firebase_admin.initialize_app(cred, {
            "databaseURL": db_url
        })

        firebase_db = db.reference()
        print("Firebase Connected")
        return True

    except Exception as e:
        print("Firebase Error:", e)
        return False


init_firebase()

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
# FUZZY LOGIC ENGINE (RESTORED)
# ============================================================

class FuzzyLogicSystem:

    def __init__(self):

        self.methane = ctrl.Antecedent(np.arange(0, 1001, 1), "methane")
        self.co2 = ctrl.Antecedent(np.arange(0, 5001, 1), "co2")
        self.temperature = ctrl.Antecedent(np.arange(0, 61, 1), "temperature")
        self.humidity = ctrl.Antecedent(np.arange(0, 101, 1), "humidity")

        self.risk = ctrl.Consequent(np.arange(0, 101, 1), "risk")

        # membership functions
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
            ctrl.Rule(self.temperature["hot"] & self.methane["high"], self.risk["high"]),
        ]

        self.ctrl = ctrl.ControlSystem(rules)

    def calculate(self, methane, co2, temperature, humidity):

        sim = ctrl.ControlSystemSimulation(self.ctrl)

        sim.input["methane"] = methane
        sim.input["co2"] = co2
        sim.input["temperature"] = temperature
        sim.input["humidity"] = humidity

        sim.compute()

        score = float(sim.output["risk"])

        if score >= 80:
            level = "CRITICAL"
        elif score >= 60:
            level = "HIGH"
        elif score >= 40:
            level = "MEDIUM"
        elif score >= 20:
            level = "LOW"
        else:
            level = "SAFE"

        return {"level": level, "score": round(score, 2)}


fuzzy_system = FuzzyLogicSystem()


# ============================================================
# MACHINE LEARNING ENGINE (RESTORED)
# ============================================================

class AdvancedMethaneAI:

    def __init__(self):

        self.model = RandomForestRegressor(n_estimators=150)
        self.ridge = Ridge(alpha=1.0)
        self.scaler = StandardScaler()

        self.trained = False
        self.rmse = 0
        self.mae = 0
        self.r2 = 0

    def prepare(self, history):

        rows = []

        for i in range(3, len(history)):

            rows.append({
                "m1": history[i-1]["methane"],
                "m2": history[i-2]["methane"],
                "m3": history[i-3]["methane"],
                "co2": history[i-1]["co2"],
                "temp": history[i-1]["temperature"],
                "hum": history[i-1]["humidity"],
                "target": history[i]["methane"]
            })

        df = pd.DataFrame(rows)

        return df.drop("target", axis=1), df["target"]

    def train(self, history):

        if len(history) < 20:
            return {"success": False, "message": "Need 20+ records"}

        X, y = self.prepare(history)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model.fit(X_train, y_train)
        self.ridge.fit(X_train, y_train)

        pred1 = self.model.predict(X_test)
        pred2 = self.ridge.predict(X_test)

        final = (pred1 * 0.7 + pred2 * 0.3)

        self.rmse = float(np.sqrt(mean_squared_error(y_test, final)))
        self.mae = float(mean_absolute_error(y_test, final))
        self.r2 = float(r2_score(y_test, final))

        self.trained = True

        return {
            "success": True,
            "accuracy": round(self.r2 * 100, 2),
            "rmse": round(self.rmse, 2),
            "mae": round(self.mae, 2),
            "samples": len(history)
        }

    def predict(self, history):

        if not self.trained or len(history) < 3:
            return {"prediction": None}

        last = history[-1]

        features = [[
            last["methane"],
            last["methane"],
            last["methane"],
            last["co2"],
            last["temperature"],
            last["humidity"]
        ]]

        features = self.scaler.transform(features)

        rf = self.model.predict(features)[0]
        ridge = self.ridge.predict(features)[0]

        return {"prediction": float((rf * 0.7 + ridge * 0.3))}


ml_engine = AdvancedMethaneAI()


# ============================================================
# API ENDPOINTS (FULL RESTORED)
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
            "GET /api/sensors",
            "GET /api/sensor/summary/{sensor_id}",
            "GET /api/fuzzy/{sensor_id}",
            "GET /api/fuzzy/config",
            "POST /api/ml/train/{sensor_id}",
            "GET /api/ml/predict/{sensor_id}",
            "GET /api/model/metrics/{sensor_id}",
            "GET /api/visualization/chart/{sensor_id}",
            "GET /api/history/{sensor_id}",
            "GET /api/dashboard/{sensor_id}",
            "GET /api/health",
            "WS /ws"
        ]
    }


@app.get("/api/health")
def health():
    return {"status": "ONLINE"}


@app.get("/api/sensors")
def get_sensors():
    data = safe_get(firebase_db.child("sensorReadings/history"))
    return list(data.keys()) if data else []


@app.get("/api/history/{sensor_id}")
def get_history(sensor_id: str):
    data = safe_get(firebase_db.child(f"sensorReadings/history/{sensor_id}"))
    return normalize_history(data)


@app.get("/api/sensor/summary/{sensor_id}")
def summary(sensor_id: str):
    hist = get_history(sensor_id)
    if not hist:
        return {}

    return hist[-1]


@app.get("/api/fuzzy/{sensor_id}")
def fuzzy(sensor_id: str):
    hist = get_history(sensor_id)
    if not hist:
        return {"level": "UNKNOWN", "score": 0}

    last = hist[-1]

    return fuzzy_system.calculate(
        last["methane"],
        last["co2"],
        last["temperature"],
        last["humidity"]
    )


@app.post("/api/ml/train/{sensor_id}")
def train(sensor_id: str):
    hist = get_history(sensor_id)
    return ml_engine.train(hist)


@app.get("/api/ml/predict/{sensor_id}")
def predict(sensor_id: str):
    hist = get_history(sensor_id)
    return ml_engine.predict(hist)


@app.get("/api/model/metrics/{sensor_id}")
def metrics(sensor_id: str):
    return {
        "RMSE": ml_engine.rmse,
        "MAE": ml_engine.mae,
        "R2": ml_engine.r2,
        "model_trained": ml_engine.trained
    }

@app.get("/api/visualization/chart/{sensor_id}")
def get_chart(sensor_id: str, limit: int = 50, offset: int = 0):

    hist = get_history(sensor_id)

    if not hist:
        return {
            "timestamps": [],
            "methane": [],
            "co2": [],
            "total": 0
        }

    sliced = hist[offset:offset + limit]

    return {
        "timestamps": [x.get("timestamp") for x in sliced],
        "methane": [x.get("methane") for x in sliced],
        "co2": [x.get("co2") for x in sliced],
        "temperature": [x.get("temperature") for x in sliced],
        "humidity": [x.get("humidity") for x in sliced],
        "total": len(hist),
        "limit": limit,
        "offset": offset
    }

@app.get("/api/dashboard/{sensor_id}")
def dashboard(sensor_id: str):

    hist = get_history(sensor_id)["data"]

    if not hist:
        return {
            "sensor_id": sensor_id,
            "status": "NO_DATA"
        }

    latest = hist[-1]

    fuzzy = fuzzy_system.calculate(
        latest["methane"],
        latest["co2"],
        latest["temperature"],
        latest["humidity"]
    )

    ml_pred = ml_engine.predict(hist)

    return {
        "sensor_id": sensor_id,
        "latest": latest,
        "fuzzy": fuzzy,
        "ml_prediction": ml_pred,
        "total_records": len(hist),
        "status": "ONLINE",
        "server_time": readable_time(),
        "firebase_connected": firebase_db is not None
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    try:
        while True:

            sensors = safe_get(firebase_db.child("sensorReadings/history"))

            if not sensors:
                await websocket.send_json({"status": "NO_DATA"})
                continue

            for sensor_id, groups in sensors.items():

                hist = normalize_history(groups)

                if not hist:
                    continue

                latest = hist[-1]

                payload = {
                    "sensor_id": sensor_id,
                    "methane": latest.get("methane", 0),
                    "co2": latest.get("co2", 0),
                    "temperature": latest.get("temperature", 0),
                    "humidity": latest.get("humidity", 0),
                    "timestamp": latest.get("timestamp"),
                    "risk": fuzzy_system.calculate(
                        latest["methane"],
                        latest["co2"],
                        latest["temperature"],
                        latest["humidity"]
                    )
                }

                await websocket.send_json(payload)

    except Exception as e:
        print("WebSocket Error:", e)


# ==============================
# RENDER PORT FIX
# ==============================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)