# ============================================================
# METHANE GAS MONITORING SYSTEM
# ============================================================
# FEATURES
# ------------------------------------------------------------
# ✔ FASTAPI API SERVER
# ✔ FIREBASE REALTIME DATABASE
# ✔ FUZZY LOGIC RISK ANALYSIS
# ✔ MACHINE LEARNING FORECASTING
# ✔ RANDOM FOREST + RIDGE HYBRID AI
# ✔ MANUAL TRAINING
# ✔ BASE64 MODEL STORAGE
# ✔ FORECAST HISTORY
# ✔ REALTIME WEBSOCKET
# ✔ REACTJS READY
# ✔ RENDER FREE TIER OPTIMIZED
# ✔ GSM/ESP32 READY
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

# ============================================================
# FASTAPI
# ============================================================

from fastapi import FastAPI
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from pydantic import BaseModel

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

    description="""
    Methane Gas Monitoring API
    with Fuzzy Logic and AI Forecasting
    """,

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

    return datetime.now(
        PH_TZ
    ).strftime("%Y%m%d_%H%M%S")


def readable_time():

    return datetime.now(
        PH_TZ
    ).strftime("%Y-%m-%d %H:%M:%S")

# ============================================================
# FIREBASE INIT
# ============================================================

firebase_db = None

def init_firebase():

    global firebase_db

    try:

        # Prevent duplicate initialization
        if firebase_admin._apps:

            firebase_db = db.reference()

            return True

        cred_json = os.environ.get(
            "FIREBASE_SERVICE_ACCOUNT"
        )

        if not cred_json:

            logger.warning(
                "Firebase credentials missing"
            )

            return False

        cred = credentials.Certificate(
            json.loads(cred_json)
        )

        firebase_admin.initialize_app(

            cred,

            {
                "databaseURL": os.environ.get(
                    "FIREBASE_DB_URL"
                )
            }
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
# PYDANTIC MODEL
# ============================================================

class SensorInput(BaseModel):

    sensor_id: str
    methane: float
    co2: float
    temperature: float
    humidity: float

# ============================================================
# FUZZY LOGIC SYSTEM
# ============================================================

class FuzzyLogicSystem:

    def __init__(self):

        # ====================================================
        # INPUT UNIVERSES
        # ====================================================

        self.methane = ctrl.Antecedent(
            np.arange(0, 1001, 1),
            "methane"
        )

        self.co2 = ctrl.Antecedent(
            np.arange(0, 5001, 1),
            "co2"
        )

        self.temperature = ctrl.Antecedent(
            np.arange(0, 61, 1),
            "temperature"
        )

        self.humidity = ctrl.Antecedent(
            np.arange(0, 101, 1),
            "humidity"
        )

        self.risk = ctrl.Consequent(
            np.arange(0, 101, 1),
            "risk"
        )

        # ====================================================
        # MEMBERSHIP FUNCTIONS
        # ====================================================

        self.methane["low"] = fuzz.trimf(
            self.methane.universe,
            [0, 0, 300]
        )

        self.methane["medium"] = fuzz.trimf(
            self.methane.universe,
            [200, 450, 700]
        )

        self.methane["high"] = fuzz.trimf(
            self.methane.universe,
            [600, 850, 1000]
        )

        self.co2["normal"] = fuzz.trimf(
            self.co2.universe,
            [0, 400, 800]
        )

        self.co2["elevated"] = fuzz.trimf(
            self.co2.universe,
            [600, 1500, 2500]
        )

        self.co2["danger"] = fuzz.trimf(
            self.co2.universe,
            [2000, 3500, 5000]
        )

        self.temperature["normal"] = fuzz.trimf(
            self.temperature.universe,
            [15, 25, 35]
        )

        self.temperature["hot"] = fuzz.trimf(
            self.temperature.universe,
            [30, 45, 60]
        )

        self.humidity["normal"] = fuzz.trimf(
            self.humidity.universe,
            [40, 60, 80]
        )

        self.humidity["wet"] = fuzz.trimf(
            self.humidity.universe,
            [70, 90, 100]
        )

        self.risk["safe"] = fuzz.trimf(
            self.risk.universe,
            [0, 10, 25]
        )

        self.risk["low"] = fuzz.trimf(
            self.risk.universe,
            [20, 35, 50]
        )

        self.risk["medium"] = fuzz.trimf(
            self.risk.universe,
            [45, 60, 75]
        )

        self.risk["high"] = fuzz.trimf(
            self.risk.universe,
            [70, 85, 100]
        )

        # ====================================================
        # RULES
        # ====================================================

        rules = [

            ctrl.Rule(
                self.methane["high"],
                self.risk["high"]
            ),

            ctrl.Rule(
                self.methane["medium"] &
                self.co2["elevated"],
                self.risk["medium"]
            ),

            ctrl.Rule(
                self.methane["high"] &
                self.temperature["hot"],
                self.risk["high"]
            ),

            ctrl.Rule(
                self.humidity["wet"] &
                self.methane["medium"],
                self.risk["medium"]
            ),

            ctrl.Rule(
                self.methane["low"] &
                self.co2["normal"],
                self.risk["safe"]
            )
        ]

        self.risk_ctrl = ctrl.ControlSystem(
            rules
        )

    # ========================================================
    # CALCULATE RISK
    # ========================================================

    def calculate_risk(
        self,
        methane,
        co2,
        temperature,
        humidity
    ):

        try:

            # IMPORTANT:
            # Create NEW simulator every request
            # for async safety on FastAPI

            simulator = ctrl.ControlSystemSimulation(
                self.risk_ctrl
            )

            simulator.input["methane"] = methane

            simulator.input["co2"] = co2

            simulator.input["temperature"] = temperature

            simulator.input["humidity"] = humidity

            simulator.compute()

            score = float(
                simulator.output["risk"]
            )

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

            return {

                "level": level,

                "score": round(score, 2)
            }

        except Exception as e:

            logger.error(f"Fuzzy Error: {e}")

            return {

                "level": "UNKNOWN",

                "score": 0
            }

# ============================================================
# FUZZY INIT
# ============================================================

fuzzy_system = FuzzyLogicSystem()

# ============================================================
# MACHINE LEARNING ENGINE
# ============================================================

class AdvancedMethaneAI:

    def __init__(self):

        self.rf_model = RandomForestRegressor(

            n_estimators=150,

            random_state=42
        )

        self.ridge_model = Ridge(
            alpha=1.0
        )

        self.scaler = StandardScaler()

        self.is_trained = False

        self.training_accuracy = 0

        self.rmse = 0

        self.mae = 0

        self.r2 = 0

        self.last_training = None

    # ========================================================
    # PREPARE DATASET
    # ========================================================

    def prepare_dataset(self, history):

        rows = []

        for i in range(3, len(history)):

            prev1 = history[i - 1]
            prev2 = history[i - 2]
            prev3 = history[i - 3]

            current = history[i]

            rows.append({

                "methane_prev1":
                prev1["methane"],

                "methane_prev2":
                prev2["methane"],

                "methane_prev3":
                prev3["methane"],

                "co2_prev1":
                prev1["co2"],

                "temperature_prev1":
                prev1["temperature"],

                "humidity_prev1":
                prev1["humidity"],

                "target":
                current["methane"]
            })

        df = pd.DataFrame(rows)

        X = df.drop(columns=["target"])

        y = df["target"]

        return X, y

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    def train(self, history):

        if len(history) < 20:

            return {

                "success": False,

                "message": "Need at least 20 records"
            }

        X, y = self.prepare_dataset(history)

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=0.2,

            random_state=42
        )

        # ====================================================
        # SCALING
        # ====================================================

        X_train_scaled = self.scaler.fit_transform(
            X_train
        )

        X_test_scaled = self.scaler.transform(
            X_test
        )

        # ====================================================
        # TRAIN MODELS
        # ====================================================

        self.rf_model.fit(
            X_train_scaled,
            y_train
        )

        self.ridge_model.fit(
            X_train_scaled,
            y_train
        )

        # ====================================================
        # PREDICTION
        # ====================================================

        rf_pred = self.rf_model.predict(
            X_test_scaled
        )

        ridge_pred = self.ridge_model.predict(
            X_test_scaled
        )

        final_pred = (

            rf_pred * 0.7

            +

            ridge_pred * 0.3
        )

        # ====================================================
        # METRICS
        # ====================================================

        self.rmse = np.sqrt(
            mean_squared_error(
                y_test,
                final_pred
            )
        )

        self.mae = mean_absolute_error(
            y_test,
            final_pred
        )

        self.r2 = r2_score(
            y_test,
            final_pred
        )

        self.training_accuracy = self.r2

        self.is_trained = True

        self.last_training = readable_time()

        return {

            "success": True,

            "accuracy":
            round(self.r2 * 100, 2),

            "rmse":
            round(self.rmse, 2),

            "mae":
            round(self.mae, 2),

            "samples":
            len(history)
        }

    # ========================================================
    # PREDICT
    # ========================================================

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

        features_scaled = self.scaler.transform(
            features
        )

        rf = self.rf_model.predict(
            features_scaled
        )[0]

        ridge = self.ridge_model.predict(
            features_scaled
        )[0]

        prediction = (

            rf * 0.7

            +

            ridge * 0.3
        )

        prediction = max(0, prediction)

        return round(float(prediction), 2)

# ============================================================
# SAVE MODEL TO FIREBASE
# ============================================================

def save_model_to_firebase(

    sensor_id,

    model
):

    try:

        serialized = pickle.dumps(model)

        encoded = base64.b64encode(
            serialized
        ).decode("utf-8")

        if firebase_db:

            firebase_db.child(
                f"mlModels/{sensor_id}"
            ).set({

                "model": encoded,

                "version": "2.0",

                "algorithm":
                "RandomForest + Ridge Hybrid",

                "accuracy":
                model.training_accuracy,

                "updated_at":
                readable_time()
            })

        return True

    except Exception as e:

        logger.error(f"Save Model Error: {e}")

        return False

# ============================================================
# LOAD MODEL
# ============================================================

def load_model_from_firebase(sensor_id):

    try:

        data = safe_get(

            firebase_db.child(
                f"mlModels/{sensor_id}"
            )
        )

        if not data:
            return None

        encoded = data["model"]

        decoded = base64.b64decode(
            encoded
        )

        model = pickle.loads(decoded)

        return model

    except Exception as e:

        logger.error(f"Load Model Error: {e}")

        return None

# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def generate_recommendation(
    risk,
    methane
):

    level = risk["level"]

    if level == "CRITICAL":

        return (
            "Critical methane concentration detected. "
            "Immediate evacuation and emergency "
            "inspection required."
        )

    elif level == "HIGH":

        return (
            "High methane level detected. "
            "Increase ventilation immediately."
        )

    elif level == "MEDIUM":

        return (
            "Methane level increasing. "
            "Continuous monitoring recommended."
        )

    elif level == "LOW":

        return (
            "Methane level manageable but "
            "continue monitoring."
        )

    return (
        "Environment stable and safe."
    )

# ============================================================
# HISTORY HELPER
# ============================================================

def get_history(
    sensor_id,
    limit=200
):

    data = safe_get(

        firebase_db.child(
            f"sensorReadings/history/{sensor_id}"
        )
        .order_by_key()
        .limit_to_last(limit),

        {}
    )

    history = []

    for _, value in data.items():

        history.append({

            "methane":
            float(value.get("methane", 0)),

            "co2":
            float(value.get("co2", 0)),

            "temperature":
            float(value.get("temperature", 25)),

            "humidity":
            float(value.get("humidity", 50)),

            "timestamp":
            value.get("timestamp")
        })

    return history

# ============================================================
# WEBSOCKET CLIENTS
# ============================================================

clients: List[WebSocket] = []

# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket("/ws")
async def websocket(ws: WebSocket):

    await ws.accept()

    clients.append(ws)

    try:

        while True:

            await ws.receive_text()

    except WebSocketDisconnect:

        clients.remove(ws)

# ============================================================
# BROADCAST
# ============================================================

async def broadcast(data):

    for client in clients:

        try:

            await client.send_json(data)

        except:
            pass

# ============================================================
# INSERT SENSOR DATA
# ============================================================

@app.post("/api/sensor/insert")
async def insert_sensor(data: SensorInput):

    try:

        payload = data.dict()

        sensor_id = payload["sensor_id"]

        timestamp_key = current_ph_timestamp()

        payload["timestamp"] = readable_time()

        # ====================================================
        # FUZZY RISK
        # ====================================================

        risk = fuzzy_system.calculate_risk(

            payload["methane"],

            payload["co2"],

            payload["temperature"],

            payload["humidity"]
        )

        payload["risk"] = risk

        payload["recommendation"] = (
            generate_recommendation(
                risk,
                payload["methane"]
            )
        )

        # ====================================================
        # FIREBASE SAVE
        # ====================================================

        if firebase_db:

            firebase_db.child(

                f"sensorReadings/latest/{sensor_id}"

            ).set(payload)

            firebase_db.child(

                f"sensorReadings/history/{sensor_id}/{timestamp_key}"

            ).set(payload)

        # ====================================================
        # WEBSOCKET BROADCAST
        # ====================================================

        await broadcast(payload)

        return {

            "success": True,

            "data": payload
        }

    except Exception as e:

        logger.error(f"Insert Error: {e}")

        return {

            "success": False,

            "error": str(e)
        }

# ============================================================
# GSM INSERT
# ============================================================

@app.post("/api/sensor/insert-gsm")
async def insert_sensor_gsm(data: SensorInput):

    return await insert_sensor(data)

# ============================================================
# MANUAL TRAINING
# ============================================================
@app.post("/api/ml/train/{sensor_id}")
def train_model(sensor_id: str):

    try:
        history = get_history(sensor_id, 500)

        if len(history) < 20:
            return {
                "success": False,
                "message": "Need at least 20 records"
            }

        ai = AdvancedMethaneAI()
        result = ai.train(history)

        if result["success"]:
            save_model_to_firebase(sensor_id, ai)

        return {
            "success": True,
            "training": result
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

    try:
        history = get_history(sensor_id, 500)

        if len(history) < 20:
            return {
                "success": False,
                "message": "Need at least 20 records"
            }

        ai = AdvancedMethaneAI()
        result = ai.train(history)

        if result["success"]:
            save_model_to_firebase(sensor_id, ai)

        return {
            "success": True,
            "training": result
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
# ============================================================
# AI PREDICTION
# ============================================================

@app.get("/api/ml/predict/{sensor_id}")
def predict(sensor_id: str):

    try:

        history = get_history(sensor_id, 100)

        if len(history) < 10:

            return {

                "success": False,

                "message":
                "Insufficient history"
            }

        model = load_model_from_firebase(
            sensor_id
        )

        if not model:

            return {

                "success": False,

                "message":
                "Model not trained"
            }

        prediction = model.predict(
            history[-3:]
        )

        latest = history[-1]

        # ====================================================
        # FORECAST RISK
        # ====================================================

        risk = fuzzy_system.calculate_risk(

            prediction,

            latest["co2"],

            latest["temperature"],

            latest["humidity"]
        )

        recommendation = generate_recommendation(
            risk,
            prediction
        )

        # ====================================================
        # TREND
        # ====================================================

        trend = "STABLE"

        if prediction > latest["methane"]:
            trend = "INCREASING"

        elif prediction < latest["methane"]:
            trend = "DECREASING"

        # ====================================================
        # SAVE FORECAST HISTORY
        # ====================================================

        if firebase_db:

            firebase_db.child(

                f"forecastHistory/{sensor_id}/{current_ph_timestamp()}"

            ).set({

                "current_methane":
                latest["methane"],

                "forecast_methane":
                prediction,

                "risk":
                risk,

                "recommendation":
                recommendation,

                "generated_at":
                readable_time()
            })

        return {

            "success": True,

            "sensor_id": sensor_id,

            "current_methane":
            latest["methane"],

            "forecast_methane":
            prediction,

            "trend":
            trend,

            "forecast_risk":
            risk,

            "recommendation":
            recommendation,

            "confidence":
            round(
                model.training_accuracy * 100,
                2
            ),

            "rmse":
            round(model.rmse, 2),

            "mae":
            round(model.mae, 2),

            "r2":
            round(model.r2, 2),

            "generated_at":
            readable_time()
        }

    except Exception as e:

        logger.error(f"Prediction Error: {e}")

        return {

            "success": False,

            "error": str(e)
        }

# ============================================================
# LIST SENSORS
# ============================================================

@app.get("/api/sensors")
def sensors():

    data = safe_get(

        firebase_db.child(
            "sensorReadings/latest"
        ),

        {}
    )

    return list(data.keys()) if data else []

# ============================================================
# SENSOR SUMMARY
# ============================================================

@app.get("/api/sensor/summary/{sensor_id}")
def summary(sensor_id: str):

    return safe_get(

        firebase_db.child(

            f"sensorReadings/latest/{sensor_id}"

        ),

        {}
    )

# ============================================================
# FUZZY ANALYSIS
# ============================================================

@app.get("/api/fuzzy/{sensor_id}")
def fuzzy_analysis(sensor_id: str):

    try:

        latest = safe_get(

            firebase_db.child(

                f"sensorReadings/latest/{sensor_id}"

            ),

            {}
        )

        if not latest:

            return {

                "success": False,

                "message":
                "Sensor not found"
            }

        methane = float(
            latest.get("methane", 0)
        )

        co2 = float(
            latest.get("co2", 0)
        )

        temperature = float(
            latest.get("temperature", 25)
        )

        humidity = float(
            latest.get("humidity", 50)
        )

        risk = fuzzy_system.calculate_risk(

            methane,

            co2,

            temperature,

            humidity
        )

        return {

            "success": True,

            "sensor_id": sensor_id,

            "risk": risk,

            "recommendation":
            generate_recommendation(
                risk,
                methane
            )
        }

    except Exception as e:

        return {

            "success": False,

            "error": str(e)
        }

# ============================================================
# FUZZY CONFIG
# ============================================================

@app.get("/api/fuzzy/config")
def fuzzy_config():

    return {

        "success": True,

        "version": "2.0",

        "inputs": [

            "methane",

            "co2",

            "temperature",

            "humidity"
        ],

        "outputs": [

            "risk"
        ]
    }

# ============================================================
# MODEL METRICS
# ============================================================

@app.get("/api/model/metrics/{sensor_id}")
def model_metrics(sensor_id: str):

    model = load_model_from_firebase(
        sensor_id
    )

    if not model:

        return {

            "success": False,

            "message":
            "Model not trained"
        }

    return {

        "success": True,

        "accuracy":
        round(
            model.training_accuracy * 100,
            2
        ),

        "rmse":
        round(model.rmse, 2),

        "mae":
        round(model.mae, 2),

        "r2":
        round(model.r2, 2),

        "last_training":
        model.last_training
    }

# ============================================================
# VISUALIZATION CHART
# ============================================================

@app.get("/api/visualization/chart/{sensor_id}")
def chart_data(sensor_id: str, limit: int = 50, offset: int = 0):

    try:
        history = get_history(sensor_id, limit=500)

        # reverse for newest-first UI
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
# LEGACY PREDICT
# ============================================================

@app.get("/api/predict/{sensor_id}")
def predict(sensor_id: str):

    try:
        history = get_history(sensor_id, 100)

        if len(history) < 10:
            return {
                "success": False,
                "message": "Insufficient history"
            }

        latest = history[-1]
        model = load_model_from_firebase(sensor_id)

        # =========================
        # DEFAULT VALUES
        # =========================
        forecast_value = float(latest["methane"])
        confidence = 0.0
        trend = "STABLE"

        # =========================
        # ML PREDICTION
        # =========================
        if model:
            pred = model.predict(history[-5:])
            forecast_value = float(pred)

            confidence = float(getattr(model, "r2", 0)) * 100

        # =========================
        # TREND ANALYSIS
        # =========================
        if forecast_value > latest["methane"]:
            trend = "INCREASING"
        elif forecast_value < latest["methane"]:
            trend = "DECREASING"

        # =========================
        # RISK (FUZZY)
        # =========================
        risk = fuzzy_system.calculate_risk(
            forecast_value,
            latest["co2"],
            latest["temperature"],
            latest["humidity"]
        )

        # =========================
        # RECOMMENDATION ENGINE
        # =========================
        recommendation = generate_recommendation(
            risk,
            forecast_value
        )

        return {
            "success": True,

            # CURRENT DATA
            "current": {
                "methane": latest["methane"],
                "co2": latest["co2"],
                "temperature": latest["temperature"],
                "humidity": latest["humidity"]
            },

            # FORECAST DATA
            "forecast": {
                "methane": round(forecast_value, 2),
                "trend": trend,
                "confidence": round(confidence, 2)
            },

            # RISK ANALYSIS
            "risk": risk,

            # AI RECOMMENDATION (IMPORTANT FIX)
            "recommendation": {
                "message": recommendation,
                "level": risk["level"]
            },

            "generated_at": readable_time()
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
# ============================================================
# ML STATUS
# ============================================================

@app.get("/api/ml/status/{sensor_id}")
def training_status(sensor_id: str):

    model = load_model_from_firebase(
        sensor_id
    )

    if not model:

        return {

            "success": False,

            "trained": False
        }

    return {

        "success": True,

        "trained":
        model.is_trained,

        "accuracy":
        round(
            model.training_accuracy * 100,
            2
        ),

        "last_training":
        model.last_training
    }

# ============================================================
# RETRAIN CHECK
# ============================================================

@app.get("/api/ml/retrain-check/{sensor_id}")
def retrain_check(sensor_id: str):

    history = get_history(sensor_id, 500)

    model = load_model_from_firebase(
        sensor_id
    )

    if not model:

        return {

            "retrain_needed": True,
            "reason":
            "No trained model"
        }

    if len(history) > 100:

        return {

            "retrain_needed": True,
            "reason":
            "Large new dataset available"
        }

    return {

        "retrain_needed": False
    }

# ============================================================
# SENSOR HISTORY
# ============================================================

@app.get("/api/history/{sensor_id}")
def sensor_history(
    sensor_id: str,
    limit: int = 100
):

    history = get_history(
        sensor_id,
        limit
    )

    return {
        "success": True,
        "records": history,
        "total":
        len(history)
    }

# ============================================================
# DASHBOARD SUMMARY
# ============================================================

@app.get("/api/dashboard/{sensor_id}")
def dashboard_summary(sensor_id: str):

    try:
        latest = safe_get(
            firebase_db.child(
                f"sensorReadings/latest/{sensor_id}"
            ),

            {}
        )

        if not latest:

            return {
                "success": False,
                "message":
                "Sensor not found"
            }

        model = load_model_from_firebase(
            sensor_id
        )

        prediction = None

        if model:

            history = get_history(
                sensor_id,
                20
            )

            if len(history) >= 5:

                prediction = model.predict(
                    history[-3:]
                )

        risk = fuzzy_system.calculate_risk(

            float(latest["methane"]),
            float(latest["co2"]),
            float(latest["temperature"]),
            float(latest["humidity"])
        )

        return {

            "success": True,

            "sensor_id":
            sensor_id,

            "latest":
            latest,

            "risk":
            risk,

            "forecast":
            prediction,

            "recommendation":
            generate_recommendation(
                risk,
                float(latest["methane"])
            ),

            "server_time":
            readable_time()
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health_check():

    return {
        "success": True,
        "status": "ONLINE",
        "server_time":
        readable_time(),

        "firebase_connected":
        True if firebase_db else False
    }

# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "status":
        "Methane AI API Running",

        "version":
        "2.0",

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
# MAIN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get("PORT", 10000)
    )

    uvicorn.run(

        "main:app",

        host="0.0.0.0",

        port=port
    )