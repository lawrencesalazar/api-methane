# ============================================================
# METHANE GAS MONITORING SYSTEM
# ============================================================
# FEATURES
# ------------------------------------------------------------
# FASTAPI API SERVER
# FIREBASE REALTIME DATABASE
# FUZZY LOGIC RISK ANALYSIS
# MACHINE LEARNING FORECASTING
# RANDOM FOREST + RIDGE HYBRID AI
# MANUAL TRAINING
# BASE64 MODEL STORAGE
# FORECAST HISTORY
# REALTIME WEBSOCKET
# THRESHOLD MANAGEMENT
# ALERT LOGGING
# GSM/ESP32 READY
# ============================================================

# ============================================================
# IMPORTS
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
from typing import Optional

# ============================================================
# FASTAPI
# ============================================================

from fastapi import FastAPI
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi import HTTPException

from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from pydantic import BaseModel
from pydantic import Field

# ============================================================
# FIREBASE
# ============================================================

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# ============================================================
# MACHINE LEARNING
# ============================================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.model_selection import train_test_split

# ============================================================
# FUZZY LOGIC
# ============================================================

import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ============================================================
# WARNING SETTINGS
# ============================================================

warnings.filterwarnings("ignore")

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("methane-ai-api")

# ============================================================
# FASTAPI INIT
# ============================================================

app = FastAPI(
    title="Methane AI Monitoring API",
    version="2.0",
    description="Methane Gas Monitoring API with Fuzzy Logic and AI Forecasting",
    redirect_slashes=False
)

# ============================================================
# MIDDLEWARE
# ============================================================

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

# ============================================================
# TIMEZONE
# ============================================================

PH_TZ = pytz.timezone("Asia/Manila")

# ============================================================
# TIME HELPERS
# ============================================================

def current_ph_timestamp():
    return datetime.now(PH_TZ).strftime("%Y%m%d_%H%M%S")

def readable_time():
    return datetime.now(PH_TZ).strftime("%Y-%m-%d %H:%M:%S")

# ============================================================
# FIREBASE INIT
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
            logger.warning("Firebase credentials missing")
            return False

        cred = credentials.Certificate(json.loads(cred_json))

        firebase_admin.initialize_app(
            cred,
            {"databaseURL": os.environ.get("FIREBASE_DB_URL")}
        )

        firebase_db = db.reference()
        logger.info("Firebase Connected")
        return True

    except Exception as e:
        logger.error(f"Firebase Init Error: {e}")
        return False

init_firebase()

# ============================================================
# SAFE FIREBASE READ
# ============================================================

def safe_get(ref, default=None):
    try:
        return ref.get()
    except Exception as e:
        logger.error(f"Firebase Read Error: {e}")
        return default

# ============================================================
# PYDANTIC MODELS
# ============================================================

class SensorInput(BaseModel):
    sensor_id: str
    methane: float
    co2: float
    temperature: float
    humidity: float

class ThresholdConfig(BaseModel):
    sensor_id: str
    warning_level: float = Field(default=500.0, ge=0, le=100000)
    danger_level: float = Field(default=5000.0, ge=0, le=100000)
    explosive_level: float = Field(default=50000.0, ge=0, le=200000)
    co2_warning: float = Field(default=1000.0, ge=0, le=10000)
    co2_danger: float = Field(default=5000.0, ge=0, le=20000)
    temp_warning: float = Field(default=35.0, ge=0, le=60)
    temp_danger: float = Field(default=45.0, ge=0, le=80)
    humidity_warning: float = Field(default=80.0, ge=0, le=100)
    humidity_danger: float = Field(default=90.0, ge=0, le=100)
    check_interval: int = Field(default=60, ge=10, le=3600)
    alert_enabled: bool = True
    sms_alert: bool = True
    email_alert: bool = False
    auto_shutdown: bool = False
    last_modified: str = ""

class AlertLog(BaseModel):
    sensor_id: str
    alert_type: str
    methane_level: float
    threshold_value: float
    message: str
    timestamp: str

class BulkThresholdUpdate(BaseModel):
    thresholds: List[ThresholdConfig]

# ============================================================
# FUZZY LOGIC SYSTEM
# ============================================================

class FuzzyLogicSystem:
    def __init__(self):
        # Input universes
        self.methane = ctrl.Antecedent(np.arange(0, 1001, 1), "methane")
        self.co2 = ctrl.Antecedent(np.arange(0, 5001, 1), "co2")
        self.temperature = ctrl.Antecedent(np.arange(0, 61, 1), "temperature")
        self.humidity = ctrl.Antecedent(np.arange(0, 101, 1), "humidity")
        self.risk = ctrl.Consequent(np.arange(0, 101, 1), "risk")

        # Methane membership functions
        self.methane["low"] = fuzz.trimf(self.methane.universe, [0, 0, 300])
        self.methane["medium"] = fuzz.trimf(self.methane.universe, [200, 450, 700])
        self.methane["high"] = fuzz.trimf(self.methane.universe, [600, 850, 1000])

        # CO2 membership functions
        self.co2["normal"] = fuzz.trimf(self.co2.universe, [0, 400, 800])
        self.co2["elevated"] = fuzz.trimf(self.co2.universe, [600, 1500, 2500])
        self.co2["danger"] = fuzz.trimf(self.co2.universe, [2000, 3500, 5000])

        # Temperature membership functions
        self.temperature["normal"] = fuzz.trimf(self.temperature.universe, [15, 25, 35])
        self.temperature["hot"] = fuzz.trimf(self.temperature.universe, [30, 45, 60])

        # Humidity membership functions
        self.humidity["normal"] = fuzz.trimf(self.humidity.universe, [40, 60, 80])
        self.humidity["wet"] = fuzz.trimf(self.humidity.universe, [70, 90, 100])

        # Risk membership functions
        self.risk["safe"] = fuzz.trimf(self.risk.universe, [0, 10, 25])
        self.risk["low"] = fuzz.trimf(self.risk.universe, [20, 35, 50])
        self.risk["medium"] = fuzz.trimf(self.risk.universe, [45, 60, 75])
        self.risk["high"] = fuzz.trimf(self.risk.universe, [70, 85, 100])

        # Fuzzy rules
        rules = [
            ctrl.Rule(self.methane["high"], self.risk["high"]),
            ctrl.Rule(self.methane["medium"] & self.co2["elevated"], self.risk["medium"]),
            ctrl.Rule(self.methane["high"] & self.temperature["hot"], self.risk["high"]),
            ctrl.Rule(self.humidity["wet"] & self.methane["medium"], self.risk["medium"]),
            ctrl.Rule(self.methane["low"] & self.co2["normal"], self.risk["safe"])
        ]

        self.risk_ctrl = ctrl.ControlSystem(rules)

    def calculate_risk(self, methane, co2, temperature, humidity):
        try:
            simulator = ctrl.ControlSystemSimulation(self.risk_ctrl)
            simulator.input["methane"] = methane
            simulator.input["co2"] = co2
            simulator.input["temperature"] = temperature
            simulator.input["humidity"] = humidity
            simulator.compute()

            score = float(simulator.output["risk"])

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

        except Exception as e:
            logger.error(f"Fuzzy Error: {e}")
            return {"level": "UNKNOWN", "score": 0}

# ============================================================
# FUZZY INIT
# ============================================================

fuzzy_system = FuzzyLogicSystem()

# ============================================================
# MACHINE LEARNING ENGINE
# ============================================================

class AdvancedMethaneAI:
    def __init__(self):
        self.rf_model = RandomForestRegressor(n_estimators=150, random_state=42)
        self.ridge_model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_accuracy = 0
        self.rmse = 0
        self.mae = 0
        self.r2 = 0
        self.last_training = None

    def prepare_dataset(self, history):
        rows = []
        for i in range(3, len(history)):
            prev1 = history[i - 1]
            prev2 = history[i - 2]
            prev3 = history[i - 3]
            current = history[i]

            rows.append({
                "methane_prev1": prev1["methane"],
                "methane_prev2": prev2["methane"],
                "methane_prev3": prev3["methane"],
                "co2_prev1": prev1["co2"],
                "temperature_prev1": prev1["temperature"],
                "humidity_prev1": prev1["humidity"],
                "target": current["methane"]
            })

        df = pd.DataFrame(rows)
        X = df.drop(columns=["target"])
        y = df["target"]
        return X, y

    def train(self, history):
        if len(history) < 20:
            return {"success": False, "message": "Need at least 20 records"}

        X, y = self.prepare_dataset(history)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.rf_model.fit(X_train_scaled, y_train)
        self.ridge_model.fit(X_train_scaled, y_train)

        rf_pred = self.rf_model.predict(X_test_scaled)
        ridge_pred = self.ridge_model.predict(X_test_scaled)
        final_pred = (rf_pred * 0.7) + (ridge_pred * 0.3)

        self.rmse = np.sqrt(mean_squared_error(y_test, final_pred))
        self.mae = mean_absolute_error(y_test, final_pred)
        self.r2 = r2_score(y_test, final_pred)
        self.training_accuracy = self.r2
        self.is_trained = True
        self.last_training = readable_time()

        return {
            "success": True,
            "accuracy": round(self.r2 * 100, 2),
            "rmse": round(self.rmse, 2),
            "mae": round(self.mae, 2),
            "samples": len(history)
        }

    def predict(self, recent):
        if not self.is_trained:
            return None

        features = [[
            recent[-1]["methane"],
            recent[-2]["methane"],
            recent[-3]["methane"],
            recent[-1]["co2"],
            recent[-1]["temperature"],
            recent[-1]["humidity"]
        ]]

        features_scaled = self.scaler.transform(features)
        rf = self.rf_model.predict(features_scaled)[0]
        ridge = self.ridge_model.predict(features_scaled)[0]
        prediction = (rf * 0.7) + (ridge * 0.3)
        prediction = max(0, prediction)
        return round(float(prediction), 2)
    
    def predict_sequence(self, recent, steps=10):
        if not self.is_trained:
            return None

        history = recent.copy()
        forecast = []

        for _ in range(steps):
            features = [[
                history[-1]["methane"],
                history[-2]["methane"],
                history[-3]["methane"],
                history[-1]["co2"],
                history[-1]["temperature"],
                history[-1]["humidity"]
            ]]

            features_scaled = self.scaler.transform(features)

            rf = self.rf_model.predict(features_scaled)[0]
            ridge = self.ridge_model.predict(features_scaled)[0]

            prediction = (rf * 0.7) + (ridge * 0.3)
            prediction = max(0, float(prediction))

            forecast.append(round(prediction, 2))

            # simulate next step (IMPORTANT for recursion)
            history.append({
                "methane": prediction,
                "co2": history[-1]["co2"],
                "temperature": history[-1]["temperature"],
                "humidity": history[-1]["humidity"]
            })

        return forecast
# ============================================================
# MODEL STORAGE FUNCTIONS
# ============================================================

def save_model_to_firebase(sensor_id, model):
    try:
        serialized = pickle.dumps(model)
        encoded = base64.b64encode(serialized).decode("utf-8")

        if firebase_db:
            firebase_db.child(f"mlModels/{sensor_id}").set({
                "model": encoded,
                "version": "2.0",
                "algorithm": "RandomForest + Ridge Hybrid",
                "accuracy": model.training_accuracy,
                "updated_at": readable_time()
            })
        return True
    except Exception as e:
        logger.error(f"Save Model Error: {e}")
        return False

def load_model_from_firebase(sensor_id):
    try:
        data = safe_get(firebase_db.child(f"mlModels/{sensor_id}"))
        if not data:
            return None

        encoded = data["model"]
        decoded = base64.b64decode(encoded)
        model = pickle.loads(decoded)
        return model
    except Exception as e:
        logger.error(f"Load Model Error: {e}")
        return None

# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def generate_recommendation(risk, methane):
    level = risk["level"]

    if level == "CRITICAL":
        return "Critical methane concentration detected. Immediate evacuation and emergency inspection required."
    elif level == "HIGH":
        return "High methane level detected. Increase ventilation immediately."
    elif level == "MEDIUM":
        return "Methane level increasing. Continuous monitoring recommended."
    elif level == "LOW":
        return "Methane level manageable but continue monitoring."

    return "Environment stable and safe."

# ============================================================
# HISTORY HELPER
# ============================================================

def get_history(sensor_id, limit=200):
    data = safe_get(
        firebase_db.child(f"sensorReadings/history/{sensor_id}").order_by_key().limit_to_last(limit),
        {}
    )

    history = []
    for _, value in data.items():
        history.append({
            "methane": float(value.get("methane", 0)),
            "co2": float(value.get("co2", 0)),
            "temperature": float(value.get("temperature", 25)),
            "humidity": float(value.get("humidity", 50)),
            "timestamp": value.get("timestamp")
        })
    return history

# ============================================================
# THRESHOLD MANAGEMENT FUNCTIONS
# ============================================================

def get_default_threshold(sensor_id):
    return {
        "sensor_id": sensor_id,
        "warning_level": 500.0,
        "danger_level": 5000.0,
        "explosive_level": 50000.0,
        "co2_warning": 1000.0,
        "co2_danger": 5000.0,
        "temp_warning": 35.0,
        "temp_danger": 45.0,
        "humidity_warning": 80.0,
        "humidity_danger": 90.0,
        "check_interval": 60,
        "alert_enabled": True,
        "sms_alert": True,
        "email_alert": False,
        "auto_shutdown": False,
        "last_modified": readable_time()
    }

def get_threshold_from_firebase(sensor_id):
    try:
        threshold = safe_get(firebase_db.child(f"thresholds/{sensor_id}"), None)
        if not threshold:
            threshold = get_default_threshold(sensor_id)
            # Save default to Firebase
            firebase_db.child(f"thresholds/{sensor_id}").set(threshold)
        return threshold
    except Exception as e:
        logger.error(f"Get threshold error: {e}")
        return get_default_threshold(sensor_id)

def check_threshold_alert(sensor_id, methane_level, co2_level, temp_level, humidity_level):
    try:
        threshold = get_threshold_from_firebase(sensor_id)
        alerts = []

        # Check methane levels
        if methane_level >= threshold["explosive_level"]:
            alerts.append({
                "type": "EXPLOSIVE",
                "level": methane_level,
                "threshold": threshold["explosive_level"],
                "message": f"EXPLOSIVE methane level: {methane_level} ppm (threshold: {threshold['explosive_level']} ppm)"
            })
        elif methane_level >= threshold["danger_level"]:
            alerts.append({
                "type": "DANGER",
                "level": methane_level,
                "threshold": threshold["danger_level"],
                "message": f"DANGER: High methane level: {methane_level} ppm (threshold: {threshold['danger_level']} ppm)"
            })
        elif methane_level >= threshold["warning_level"]:
            alerts.append({
                "type": "WARNING",
                "level": methane_level,
                "threshold": threshold["warning_level"],
                "message": f"WARNING: Elevated methane level: {methane_level} ppm (threshold: {threshold['warning_level']} ppm)"
            })

        # Check CO2 levels
        if co2_level >= threshold["co2_danger"]:
            alerts.append({
                "type": "CO2_DANGER",
                "level": co2_level,
                "threshold": threshold["co2_danger"],
                "message": f"High CO2 level: {co2_level} ppm"
            })
        elif co2_level >= threshold["co2_warning"]:
            alerts.append({
                "type": "CO2_WARNING",
                "level": co2_level,
                "threshold": threshold["co2_warning"],
                "message": f"Elevated CO2 level: {co2_level} ppm"
            })

        # Check temperature
        if temp_level >= threshold["temp_danger"]:
            alerts.append({
                "type": "TEMP_DANGER",
                "level": temp_level,
                "threshold": threshold["temp_danger"],
                "message": f"High temperature: {temp_level} C"
            })
        elif temp_level >= threshold["temp_warning"]:
            alerts.append({
                "type": "TEMP_WARNING",
                "level": temp_level,
                "threshold": threshold["temp_warning"],
                "message": f"Elevated temperature: {temp_level} C"
            })

        # Check humidity
        if humidity_level >= threshold["humidity_danger"]:
            alerts.append({
                "type": "HUMIDITY_DANGER",
                "level": humidity_level,
                "threshold": threshold["humidity_danger"],
                "message": f"High humidity: {humidity_level}%"
            })
        elif humidity_level >= threshold["humidity_warning"]:
            alerts.append({
                "type": "HUMIDITY_WARNING",
                "level": humidity_level,
                "threshold": threshold["humidity_warning"],
                "message": f"Elevated humidity: {humidity_level}%"
            })

        return alerts

    except Exception as e:
        logger.error(f"Check threshold alert error: {e}")
        return []

# ============================================================
# WEBSOCKET CLIENTS
# ============================================================

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

async def broadcast(data):
    for client in clients:
        try:
            await client.send_json(data)
        except:
            pass

# ============================================================
# API ENDPOINTS - SENSOR DATA
# ============================================================

@app.post("/api/sensor/insert")
async def insert_sensor(data: SensorInput):
    try:
        payload = data.dict()
        sensor_id = payload["sensor_id"]
        timestamp_key = current_ph_timestamp()
        payload["timestamp"] = readable_time()

        # Fuzzy risk calculation
        risk = fuzzy_system.calculate_risk(
            payload["methane"],
            payload["co2"],
            payload["temperature"],
            payload["humidity"]
        )
        payload["risk"] = risk
        payload["recommendation"] = generate_recommendation(risk, payload["methane"])

        # Check thresholds and generate alerts
        alerts = check_threshold_alert(
            sensor_id,
            payload["methane"],
            payload["co2"],
            payload["temperature"],
            payload["humidity"]
        )

        # Save to Firebase
        if firebase_db:
            firebase_db.child(f"sensorReadings/latest/{sensor_id}").set(payload)
            firebase_db.child(f"sensorReadings/history/{sensor_id}/{timestamp_key}").set(payload)

            # Log alerts
            for alert in alerts:
                alert_log = {
                    "sensor_id": sensor_id,
                    "alert_type": alert["type"],
                    "methane_level": alert["level"],
                    "threshold_value": alert["threshold"],
                    "message": alert["message"],
                    "timestamp": readable_time()
                }
                firebase_db.child(f"alerts/{sensor_id}/{current_ph_timestamp()}").set(alert_log)

        # Broadcast via WebSocket
        await broadcast(payload)

        return {
            "success": True,
            "data": payload,
            "alerts": alerts if alerts else None
        }

    except Exception as e:
        logger.error(f"Insert Error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/sensor/insert-gsm")
async def insert_sensor_gsm(data: SensorInput):
    return await insert_sensor(data)

# ============================================================
# API ENDPOINTS - THRESHOLD MANAGEMENT
# ============================================================

@app.get("/api/threshold/{sensor_id}")
def get_threshold(sensor_id: str):
    try:
        if not firebase_db:
            return {"success": False, "message": "Firebase not connected"}

        threshold = get_threshold_from_firebase(sensor_id)
        return {"success": True, "threshold": threshold}

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/threshold/{sensor_id}")
def update_threshold(sensor_id: str, config: ThresholdConfig):
    try:
        if not firebase_db:
            return {"success": False, "message": "Firebase not connected"}

        config.last_modified = readable_time()

        firebase_db.child(f"thresholds/{sensor_id}").set(config.dict())

        # Log the change
        firebase_db.child(f"thresholdLogs/{sensor_id}/{current_ph_timestamp()}").set({
            "action": "UPDATE",
            "config": config.dict(),
            "modified_by": "api",
            "timestamp": readable_time()
        })

        return {
            "success": True,
            "message": "Threshold updated successfully",
            "threshold": config.dict()
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/threshold/list")
def list_thresholds():
    try:
        if not firebase_db:
            return {"success": False, "message": "Firebase not connected"}

        thresholds = safe_get(firebase_db.child("thresholds"), {})

        result = []
        for sensor_id, config in thresholds.items():
            config["sensor_id"] = sensor_id
            result.append(config)

        return {"success": True, "thresholds": result}

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/threshold/{sensor_id}/reset")
def reset_threshold(sensor_id: str):
    try:
        default_config = get_default_threshold(sensor_id)

        if firebase_db:
            firebase_db.child(f"thresholds/{sensor_id}").set(default_config)

        return {
            "success": True,
            "message": "Threshold reset to default",
            "threshold": default_config
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/threshold/bulk")
def bulk_update_thresholds(bulk_data: BulkThresholdUpdate):
    try:
        if not firebase_db:
            return {"success": False, "message": "Firebase not connected"}

        results = []
        for config in bulk_data.thresholds:
            config.last_modified = readable_time()
            firebase_db.child(f"thresholds/{config.sensor_id}").set(config.dict())
            results.append({"sensor_id": config.sensor_id, "status": "updated"})

        return {
            "success": True,
            "message": f"Updated {len(results)} thresholds",
            "results": results
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================================
# API ENDPOINTS - ALERTS
# ============================================================

@app.post("/api/alert/log")
def log_alert(alert: AlertLog):
    try:
        if not firebase_db:
            return {"success": False}

        firebase_db.child(f"alerts/{alert.sensor_id}/{current_ph_timestamp()}").set(alert.dict())

        return {"success": True}

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/alerts/{sensor_id}")
def get_alerts(sensor_id: str, limit: int = 50):
    try:
        alerts = safe_get(
            firebase_db.child(f"alerts/{sensor_id}").order_by_key().limit_to_last(limit),
            {}
        )

        return {"success": True, "alerts": alerts}

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/alerts/summary/{sensor_id}")
def get_alert_summary(sensor_id: str):
    try:
        alerts = safe_get(firebase_db.child(f"alerts/{sensor_id}"), {})

        summary = {
            "total": len(alerts),
            "explosive": 0,
            "danger": 0,
            "warning": 0,
            "last_24h": 0
        }

        current_time = datetime.now(PH_TZ)

        for alert_id, alert in alerts.items():
            if alert.get("alert_type") == "EXPLOSIVE":
                summary["explosive"] += 1
            elif alert.get("alert_type") == "DANGER":
                summary["danger"] += 1
            elif alert.get("alert_type") == "WARNING":
                summary["warning"] += 1

            # Check last 24 hours
            alert_time_str = alert.get("timestamp", "")
            if alert_time_str:
                try:
                    alert_time = datetime.strptime(alert_time_str, "%Y-%m-%d %H:%M:%S")
                    alert_time = PH_TZ.localize(alert_time)
                    delta = current_time - alert_time
                    if delta.total_seconds() < 86400:
                        summary["last_24h"] += 1
                except:
                    pass

        return {"success": True, "summary": summary}

    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================================
# API ENDPOINTS - MACHINE LEARNING
# ============================================================

@app.post("/api/ml/train/{sensor_id}")
def train_model(sensor_id: str):
    try:
        history = get_history(sensor_id, 500)

        if len(history) < 20:
            return {"success": False, "message": "Need at least 20 records"}

        ai = AdvancedMethaneAI()
        result = ai.train(history)

        if result["success"]:
            save_model_to_firebase(sensor_id, ai)

        return {"success": True, "training": result}

    except Exception as e:
        return {"success": False, "error": str(e)}

from starlette.concurrency import run_in_threadpool

@app.get("/api/ml/predict/{sensor_id}")
async def predict(sensor_id: str):
    try:
        history = await run_in_threadpool(get_history, sensor_id, 100)

        if len(history) < 10:
            return {"success": False, "message": "Insufficient history"}
        model = await run_in_threadpool(load_model_from_firebase, sensor_id)

        if not model:
            return {"success": False, "message": "Model not trained"}

        prediction = model.predict(history[-3:])
        latest = history[-1]

        risk = fuzzy_system.calculate_risk(
            prediction,
            latest["co2"],
            latest["temperature"],
            latest["humidity"]
        )

        recommendation = generate_recommendation(risk, prediction)

        trend = (
            "INCREASING" if prediction > latest["methane"]
            else "DECREASING" if prediction < latest["methane"]
            else "STABLE"
        )
        forecast_series = []

        base = latest["methane"]
        pred = prediction

        for i in range(5):
            step = base + (pred - base) * ((i + 1) / 5)
            forecast_series.append(round(step, 2))
        warning = early_warning_engine(history, forecast_series)

        return {
            "success": True,
            "forecast": [
                prediction,
                prediction + 2,
                prediction + 4,
                prediction + 6
            ],
            "current": latest["methane"],
            "risk": risk,
            "recommendation": recommendation,
            "early_warning": warning,
            "confidence": round(model.training_accuracy * 100, 2)
        }

    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return {"success": False, "error": str(e)}

def early_warning_engine(history, forecast_series):
    if len(history) < 5 or not forecast_series:
        return {"status": "INSUFFICIENT_DATA"}

    current = history[-1]["methane"]
    future_peak = max(forecast_series)

    increase_rate = (future_peak - current) / max(current, 1)

    if increase_rate > 0.5:
        level = "CRITICAL_WARNING"
    elif increase_rate > 0.25:
        level = "HIGH_WARNING"
    elif increase_rate > 0.1:
        level = "CAUTION"
    else:
        level = "STABLE"

    return {
        "level": level,
        "increase_rate": round(increase_rate, 3),
        "current": current,
        "future_peak": future_peak
    }

@app.get("/api/ml/realtime-forecast/{sensor_id}")
def realtime_forecast(sensor_id: str):
    history = get_history(sensor_id, 20)
    model = load_model_from_firebase(sensor_id)

    if not model:
        return {"success": False}

    forecast = model.predict_sequence(history[-5:], steps=5)

    return {
        "success": True,
        "forecast": forecast,
        "timestamp": readable_time()
    }

@app.get("/api/ml/status/{sensor_id}")
def training_status(sensor_id: str):
    model = load_model_from_firebase(sensor_id)

    if not model:
        return {"success": False, "trained": False}

    return {
        "success": True,
        "trained": model.is_trained,
        "accuracy": round(model.training_accuracy * 100, 2),
        "last_training": model.last_training
    }

@app.get("/api/ml/retrain-check/{sensor_id}")
def retrain_check(sensor_id: str):
    history = get_history(sensor_id, 500)
    model = load_model_from_firebase(sensor_id)

    if not model:
        return {"retrain_needed": True, "reason": "No trained model"}

    if len(history) > 100:
        return {"retrain_needed": True, "reason": "Large new dataset available"}

    return {"retrain_needed": False}

@app.get("/api/model/metrics/{sensor_id}")
def model_metrics(sensor_id: str):
    model = load_model_from_firebase(sensor_id)

    if not model:
        return {"success": False, "message": "Model not trained"}

    return {
        "success": True,
        "accuracy": round(model.training_accuracy * 100, 2),
        "rmse": round(model.rmse, 2),
        "mae": round(model.mae, 2),
        "r2": round(model.r2, 2),
        "last_training": model.last_training
    }

# ============================================================
# API ENDPOINTS - DATA RETRIEVAL
# ============================================================

@app.get("/api/sensors")
def sensors():
    data = safe_get(firebase_db.child("sensorReadings/latest"), {})
    return list(data.keys()) if data else []

@app.get("/api/sensor/summary/{sensor_id}")
def summary(sensor_id: str):
    return safe_get(firebase_db.child(f"sensorReadings/latest/{sensor_id}"), {})

@app.get("/api/history/{sensor_id}")
def sensor_history(sensor_id: str, limit: int = 100):
    history = get_history(sensor_id, limit)
    return {"success": True, "records": history, "total": len(history)}

@app.get("/api/dashboard/{sensor_id}")
def dashboard_summary(sensor_id: str):
    try:
        latest = safe_get(firebase_db.child(f"sensorReadings/latest/{sensor_id}"), {})

        if not latest:
            return {"success": False, "message": "Sensor not found"}

        model = load_model_from_firebase(sensor_id)
        prediction = None

        if model:
            history = get_history(sensor_id, 20)
            if len(history) >= 5:
                prediction = model.predict(history[-3:])

        risk = fuzzy_system.calculate_risk(
            float(latest["methane"]),
            float(latest["co2"]),
            float(latest["temperature"]),
            float(latest["humidity"])
        )

        threshold = get_threshold_from_firebase(sensor_id)

        return {
            "success": True,
            "sensor_id": sensor_id,
            "latest": latest,
            "risk": risk,
            "forecast": prediction,
            "recommendation": generate_recommendation(risk, float(latest["methane"])),
            "thresholds": threshold,
            "server_time": readable_time()
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================================
# API ENDPOINTS - FUZZY LOGIC
# ============================================================

@app.get("/api/fuzzy/{sensor_id}")
def fuzzy_analysis(sensor_id: str):
    try:
        latest = safe_get(firebase_db.child(f"sensorReadings/latest/{sensor_id}"), {})

        if not latest:
            return {"success": False, "message": "Sensor not found"}

        methane = float(latest.get("methane", 0))
        co2 = float(latest.get("co2", 0))
        temperature = float(latest.get("temperature", 25))
        humidity = float(latest.get("humidity", 50))

        risk = fuzzy_system.calculate_risk(methane, co2, temperature, humidity)

        return {
            "success": True,
            "sensor_id": sensor_id,
            "risk": risk,
            "recommendation": generate_recommendation(risk, methane)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/fuzzy/config")
def fuzzy_config():
    return {
        "success": True,
        "version": "2.0",
        "inputs": ["methane", "co2", "temperature", "humidity"],
        "outputs": ["risk"]
    }

# ============================================================
# API ENDPOINTS - VISUALIZATION
# ============================================================

@app.get("/api/visualization/chart/{sensor_id}")
def chart_data(sensor_id: str, limit: int = 50, offset: int = 0):
    try:
        history = get_history(sensor_id, limit=500)
        history = history[::-1]
        paginated = history[offset:offset + limit]

        return {
            "success": True,
            "timestamps": [h["timestamp"] for h in paginated],
            "methane": [h["methane"] for h in paginated],
            "co2": [h["co2"] for h in paginated],
            "temperature": [h["temperature"] for h in paginated],
            "humidity": [h["humidity"] for h in paginated],
            "total": len(history)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health_check():
    return {
        "success": True,
        "status": "ONLINE",
        "server_time": readable_time(),
        "firebase_connected": True if firebase_db else False
    }

# ============================================================
# ROOT ENDPOINT
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
            "Base64 Model Storage",
            "Threshold Management",
            "Alert Logging"
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
            "GET /api/threshold/{sensor_id}",
            "POST /api/threshold/{sensor_id}",
            "GET /api/threshold/list",
            "POST /api/threshold/{sensor_id}/reset",
            "POST /api/threshold/bulk",
            "POST /api/alert/log",
            "GET /api/alerts/{sensor_id}",
            "GET /api/alerts/summary/{sensor_id}",
            "WS /ws"
        ]
    }

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run("main:app", host="0.0.0.0", port=port)