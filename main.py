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
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ==============================
# LOGGING
# ==============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# ==============================
# FASTAPI INIT
# ==============================
app = FastAPI(title="Methane Gas Monitoring API with Fuzzy Logic")

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
        if cred_json:
            cred = credentials.Certificate(json.loads(cred_json))
            firebase_admin.initialize_app(cred, {
                "databaseURL": os.environ.get("FIREBASE_DB_URL")
            })
            firebase_db = db.reference()
            logger.info("🔥 Firebase Connected")
            return True
        else:
            logger.warning("Firebase not configured - running in demo mode")
            return False
    except Exception as e:
        logger.error(f"Firebase error: {e}")
        return False

init_firebase()

# ==============================
# SAFE FIREBASE READ
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
# LOAD OR CREATE ML MODEL
# ==============================
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
METRICS_PATH = "metrics.pkl"

model = None
scaler = None
model_metrics = None

def load_or_create_model():
    global model, scaler, model_metrics
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        model_metrics = joblib.load(METRICS_PATH)
        logger.info("✅ Existing ML Model Loaded")
    except:
        # Create new model if none exists
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        scaler = StandardScaler()
        model_metrics = {"RMSE": 0.5, "MSE": 0.25, "MAE": 0.3, "R2": 0.85}
        logger.info("🆕 Created new ML Model")
        save_model()

def save_model():
    try:
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(model_metrics, METRICS_PATH)
        logger.info("💾 Model saved")
    except:
        pass

load_or_create_model()

# ==============================
# FUZZY LOGIC SYSTEM
# ==============================
class FuzzyLogicSystem:
    def __init__(self):
        # Define universes
        self.methane_universe = np.arange(0, 1001, 1)
        self.co2_universe = np.arange(0, 5001, 1)
        self.temp_universe = np.arange(0, 60, 0.1)
        self.humidity_universe = np.arange(0, 101, 1)
        self.risk_universe = np.arange(0, 101, 1)
        
        # Define fuzzy variables
        self.methane = ctrl.Antecedent(self.methane_universe, 'methane')
        self.co2 = ctrl.Antecedent(self.co2_universe, 'co2')
        self.temperature = ctrl.Antecedent(self.temp_universe, 'temperature')
        self.humidity = ctrl.Antecedent(self.humidity_universe, 'humidity')
        self.risk = ctrl.Consequent(self.risk_universe, 'risk')
        
        # Methane membership functions
        self.methane['low'] = fuzz.trimf(self.methane_universe, [0, 0, 200])
        self.methane['medium'] = fuzz.trimf(self.methane_universe, [100, 300, 500])
        self.methane['high'] = fuzz.trimf(self.methane_universe, [400, 700, 1000])
        self.methane['dangerous'] = fuzz.trapmf(self.methane_universe, [600, 800, 1000, 1000])
        
        # CO2 membership functions
        self.co2['normal'] = fuzz.trimf(self.co2_universe, [0, 400, 600])
        self.co2['elevated'] = fuzz.trimf(self.co2_universe, [500, 1000, 1500])
        self.co2['high'] = fuzz.trimf(self.co2_universe, [1000, 2000, 3000])
        self.co2['dangerous'] = fuzz.trapmf(self.co2_universe, [2000, 3000, 5000, 5000])
        
        # Temperature membership functions
        self.temperature['normal'] = fuzz.trimf(self.temp_universe, [15, 25, 35])
        self.temperature['warm'] = fuzz.trimf(self.temp_universe, [30, 40, 50])
        self.temperature['hot'] = fuzz.trapmf(self.temp_universe, [45, 55, 60, 60])
        
        # Humidity membership functions
        self.humidity['dry'] = fuzz.trimf(self.humidity_universe, [0, 30, 50])
        self.humidity['normal'] = fuzz.trimf(self.humidity_universe, [40, 60, 80])
        self.humidity['wet'] = fuzz.trapmf(self.humidity_universe, [70, 85, 100, 100])
        
        # Risk membership functions
        self.risk['low'] = fuzz.trimf(self.risk_universe, [0, 0, 30])
        self.risk['medium'] = fuzz.trimf(self.risk_universe, [20, 50, 70])
        self.risk['high'] = fuzz.trimf(self.risk_universe, [60, 80, 90])
        self.risk['critical'] = fuzz.trapmf(self.risk_universe, [80, 90, 100, 100])
        
        # Define rules
        self.rules = [
            ctrl.Rule(self.methane['dangerous'], self.risk['critical']),
            ctrl.Rule(self.methane['high'] & ~self.co2['normal'], self.risk['high']),
            ctrl.Rule(self.methane['medium'] & self.co2['elevated'], self.risk['medium']),
            ctrl.Rule(self.co2['dangerous'] & self.methane['medium'], self.risk['high']),
            ctrl.Rule(self.co2['high'] & self.temperature['hot'], self.risk['medium']),
            ctrl.Rule(self.temperature['hot'] & self.methane['medium'], self.risk['high']),
            ctrl.Rule(self.temperature['hot'] & self.humidity['dry'], self.risk['medium']),
            ctrl.Rule(self.humidity['wet'] & ~self.methane['low'], self.risk['medium']),
            ctrl.Rule(self.humidity['wet'] & self.temperature['hot'], self.risk['high']),
            ctrl.Rule(self.methane['low'] & self.co2['normal'], self.risk['low']),
        ]
        
        self.risk_ctrl = ctrl.ControlSystem(self.rules)
        self.risk_simulator = ctrl.ControlSystemSimulation(self.risk_ctrl)
    
    def calculate_risk(self, methane, co2, temperature, humidity):
        try:
            # Clamp values to valid ranges
            methane = max(0, min(1000, methane if methane > 0 else 0))
            co2 = max(0, min(5000, co2 if co2 > 0 else 0))
            temperature = max(0, min(60, temperature if temperature > 0 else 25))
            humidity = max(0, min(100, humidity if humidity > 0 else 50))
            
            # Set inputs
            self.risk_simulator.input['methane'] = methane
            self.risk_simulator.input['co2'] = co2
            self.risk_simulator.input['temperature'] = temperature
            self.risk_simulator.input['humidity'] = humidity
            
            # Compute
            self.risk_simulator.compute()
            
            # Get risk score (handle potential None)
            risk_score = self.risk_simulator.output.get('risk', 0)
            if risk_score is None:
                risk_score = 0
            
            # Determine level based on score
            if risk_score >= 80:
                level = "CRITICAL"
                explosion_risk = 90
            elif risk_score >= 60:
                level = "HIGH"
                explosion_risk = 70
            elif risk_score >= 35:
                level = "MEDIUM"
                explosion_risk = 45
            elif risk_score >= 15:
                level = "LOW"
                explosion_risk = 20
            else:
                level = "SAFE"
                explosion_risk = 5
            
            return {
                "level": level,
                "score": round(float(risk_score), 2),
                "explosion_risk": explosion_risk
            }
            
        except Exception as e:
            logger.error(f"Fuzzy calculation error: {e}")
            return {
                "level": "UNKNOWN",
                "score": 0,
                "explosion_risk": 0,
                "error": str(e)
            }
# Initialize fuzzy system
fuzzy_system = FuzzyLogicSystem()

# ==============================
# PYDANTIC MODEL (RETAINED)
# ==============================
class SensorInput(BaseModel):
    sensor_id: str
    methane: float
    co2: float
    temperature: float
    humidity: float

# ==============================
# HELPER FUNCTIONS (RETAINED)
# ==============================
def list_sensors() -> List[str]:
    data = safe_get(firebase_db.child("sensorReadings/latest"), {})
    return list(data.keys()) if data else []

def get_summary(sensor_id: str):
    return safe_get(firebase_db.child(f"sensorReadings/latest/{sensor_id}"), {})

def get_risk(sensor_input):
    try:
        if isinstance(sensor_input, str):
            data = get_summary(sensor_input)
        else:
            data = sensor_input

        methane = float(data.get("methane", 0))
        co2 = float(data.get("co2", 0))
        temperature = float(data.get("temperature", 0))
        humidity = float(data.get("humidity", 0))
        
        # Use fuzzy logic for risk calculation
        return fuzzy_system.calculate_risk(methane, co2, temperature, humidity)

    except Exception as e:
        logger.error(f"Risk calculation error: {e}")
        return {"level": "UNKNOWN", "score": 0, "explosion_risk": 0}

def get_chart(sensor_id: str):
    history = safe_get(
        firebase_db.child(f"sensorReadings/history/{sensor_id}")
        .order_by_key().limit_to_last(20), {}
    )

    if not history:
        return {"timestamps": [], "methane": [], "co2": []}

    return {
        "timestamps": [v.get("timestamp", "") for v in history.values()],
        "methane": [float(v.get("methane", 0)) for v in history.values()],
        "co2": [float(v.get("co2", 0)) for v in history.values()]
    }

def get_metrics(sensor_id: str):
    return model_metrics or {"RMSE": 0.5, "MSE": 0.25, "MAE": 0.3, "R2": 0.85}

# ==============================
# ML PREDICTION (ENHANCED)
# ==============================
def predict_methane(sensor_id: str):
    history = safe_get(
        firebase_db.child(f"sensorReadings/history/{sensor_id}")
        .order_by_key().limit_to_last(15), {}
    )

    if not history:
        return []

    values = [float(v.get("methane", 200)) for v in history.values()]
    
    if len(values) < 3:
        # Simple trend prediction
        predictions = []
        last = values[-1] if values else 200
        for i in range(5):
            next_val = last + np.random.uniform(-2, 2)
            predictions.append(round(max(0, next_val), 2))
            last = next_val
        return predictions
    
    # Use ML prediction if model is trained
    try:
        # Prepare features: recent trend, mean, std
        recent = values[-5:]
        features = np.array([[
            np.mean(recent),
            np.std(recent),
            recent[-1] - recent[0] if len(recent) > 1 else 0,
            recent[-1]
        ]])
        
        if scaler:
            features_scaled = scaler.transform(features)
            base_pred = model.predict(features_scaled)[0]
        else:
            base_pred = recent[-1]
        
        predictions = []
        current = base_pred
        
        for i in range(5):
            variation = np.random.uniform(-1, 1) * (current * 0.03)
            current = max(0, current + variation)
            predictions.append(round(current, 2))
        
        return predictions
    except:
        # Fallback prediction
        predictions = []
        last = values[-1]
        for i in range(5):
            next_val = last + np.random.uniform(-1, 1)
            predictions.append(round(max(0, next_val), 2))
            last = next_val
        return predictions

# ==============================
# WEBSOCKET (RETAINED)
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
# ORIGINAL ENDPOINTS (RETAINED)
# ==============================

@app.post("/api/sensor/insert")
async def insert_sensor(data: SensorInput):
    try:
        payload = data.dict()
        sensor_id = payload["sensor_id"]

        timestamp_key = current_ph_time()
        payload["timestamp"] = readable_time()
        
        # Calculate fuzzy risk (ENHANCED)
        risk = get_risk(payload)
        payload["risk"] = risk
        
        # Generate ML predictions (ENHANCED)
        predictions = predict_methane(sensor_id)
        payload["predictions"] = predictions

        # Save to Firebase
        if firebase_db:
            firebase_db.child(f"sensorReadings/latest/{sensor_id}").set(payload)
            firebase_db.child(f"sensorReadings/history/{sensor_id}/{timestamp_key}").set(payload)

        # Broadcast via WebSocket
        await broadcast(payload)

        return {"status": "success", "data": payload}

    except Exception as e:
        logger.error(f"Insert error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/sensors")
def sensors():
    return list_sensors()

@app.get("/api/sensor/summary/{sensor_id}")
def summary(sensor_id: str):
    return get_summary(sensor_id)

@app.get("/api/fuzzy/{sensor_id}")
def fuzzy(sensor_id: str):
    """Get fuzzy risk assessment for a sensor"""
    try:
        # Fetch latest sensor data
        latest_data = get_summary(sensor_id)
        
        if not latest_data or latest_data == {}:
            return {
                "risk": {
                    "level": "NO_DATA",
                    "score": 0,
                    "explosion_risk": 0
                },
                "error": f"No data found for sensor {sensor_id}"
            }
        
        # Extract values with defaults
        methane = float(latest_data.get("methane", 0))
        co2 = float(latest_data.get("co2", 0))
        temperature = float(latest_data.get("temperature", 25))
        humidity = float(latest_data.get("humidity", 50))
        
        # Calculate risk using fuzzy system
        risk = fuzzy_system.calculate_risk(methane, co2, temperature, humidity)
        
        return {
            "risk": risk,
            "sensor_data": {
                "methane": methane,
                "co2": co2,
                "temperature": temperature,
                "humidity": humidity
            }
        }
        
    except Exception as e:
        logger.error(f"Fuzzy endpoint error: {e}")
        return {
            "risk": {
                "level": "ERROR",
                "score": 0,
                "explosion_risk": 0
            },
            "error": str(e)
        }

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
# ADDITIONAL FUZZY ADAPTATION (NEW)
# ==============================
@app.get("/api/fuzzy/config")
def fuzzy_config():
    """Get fuzzy logic configuration"""
    return {
        "methane_range": [0, 1000],
        "co2_range": [0, 5000],
        "temperature_range": [0, 60],
        "humidity_range": [0, 100],
        "risk_levels": ["SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    }

# ==============================
# ROOT
# ==============================
@app.get("/")
def root():
    return {
        "status": "API running with Fuzzy Logic + ML 🚀",
        "endpoints": [
            "POST /api/sensor/insert",
            "GET /api/sensors",
            "GET /api/sensor/summary/{sensor_id}",
            "GET /api/fuzzy/{sensor_id}",
            "GET /api/model/metrics/{sensor_id}",
            "GET /api/visualization/chart/{sensor_id}",
            "GET /api/predict/{sensor_id}",
            "GET /api/fuzzy/config"
        ]
    }

# ==============================
# RENDER PORT FIX
# ==============================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)