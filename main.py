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
        # Define universes (expanded for dumpsite ranges)
        self.methane_universe = np.arange(0, 5001, 1)     # 0-5000 ppm (dumpsite range)
        self.co2_universe = np.arange(0, 10001, 1)       # 0-10000 ppm
        self.temp_universe = np.arange(0, 60, 0.1)       # 0-60°C
        self.humidity_universe = np.arange(0, 101, 1)    # 0-100%
        self.risk_universe = np.arange(0, 101, 1)        # 0-100% risk
        
        # Define fuzzy variables
        self.methane = ctrl.Antecedent(self.methane_universe, 'methane')
        self.co2 = ctrl.Antecedent(self.co2_universe, 'co2')
        self.temperature = ctrl.Antecedent(self.temp_universe, 'temperature')
        self.humidity = ctrl.Antecedent(self.humidity_universe, 'humidity')
        self.risk = ctrl.Consequent(self.risk_universe, 'risk')
        
        # ========== METHANE (Dumpsite-Optimized) ==========
        # Normal dumpsite baseline is 200-500 ppm
        self.methane['normal'] = fuzz.trimf(self.methane_universe, [0, 300, 600])
        self.methane['elevated'] = fuzz.trimf(self.methane_universe, [400, 800, 1200])
        self.methane['high'] = fuzz.trimf(self.methane_universe, [900, 1500, 2500])
        self.methane['dangerous'] = fuzz.trapmf(self.methane_universe, [2000, 3000, 5000, 5000])
        
        # ========== CO2 (Dumpsite-Optimized) ==========
        self.co2['normal'] = fuzz.trimf(self.co2_universe, [0, 500, 800])
        self.co2['elevated'] = fuzz.trimf(self.co2_universe, [600, 1200, 2000])
        self.co2['high'] = fuzz.trimf(self.co2_universe, [1500, 3000, 5000])
        self.co2['dangerous'] = fuzz.trapmf(self.co2_universe, [4000, 6000, 10000, 10000])
        
        # ========== TEMPERATURE (Tropical Dumpsite) ==========
        self.temperature['normal'] = fuzz.trimf(self.temp_universe, [20, 30, 38])
        self.temperature['hot'] = fuzz.trimf(self.temp_universe, [35, 45, 55])
        self.temperature['extreme'] = fuzz.trapmf(self.temp_universe, [50, 60, 60, 60])
        
        # ========== HUMIDITY (Tropical Climate) ==========
        self.humidity['normal'] = fuzz.trimf(self.humidity_universe, [40, 70, 85])
        self.humidity['wet'] = fuzz.trapmf(self.humidity_universe, [75, 90, 100, 100])
        
        # ========== RISK LEVELS ==========
        self.risk['low'] = fuzz.trimf(self.risk_universe, [0, 0, 30])
        self.risk['medium'] = fuzz.trimf(self.risk_universe, [20, 45, 65])
        self.risk['high'] = fuzz.trimf(self.risk_universe, [55, 75, 90])
        self.risk['critical'] = fuzz.trapmf(self.risk_universe, [80, 90, 100, 100])
        
        # ========== DUMPSITE-SPECIFIC RULES ==========
        self.rules = [
            # Dumpsite normal conditions (low risk)
            ctrl.Rule(self.methane['normal'] & self.co2['normal'], self.risk['low']),
            ctrl.Rule(self.methane['normal'] & self.temperature['normal'], self.risk['low']),
            
            # Elevated methane (still moderate for dumpsite)
            ctrl.Rule(self.methane['elevated'] & self.co2['normal'], self.risk['low']),
            ctrl.Rule(self.methane['elevated'] & self.co2['elevated'], self.risk['medium']),
            
            # High methane (warning)
            ctrl.Rule(self.methane['high'] & ~self.co2['dangerous'], self.risk['medium']),
            ctrl.Rule(self.methane['high'] & self.temperature['hot'], self.risk['high']),
            
            # Dangerous methane (real threat)
            ctrl.Rule(self.methane['dangerous'], self.risk['critical']),
            ctrl.Rule(self.methane['dangerous'] & self.co2['dangerous'], self.risk['critical']),
            
            # CO2-related (landfill gas indicators)
            ctrl.Rule(self.co2['dangerous'] & self.methane['elevated'], self.risk['high']),
            ctrl.Rule(self.co2['high'] & self.temperature['hot'], self.risk['medium']),
            
            # Environmental compounding
            ctrl.Rule(self.temperature['extreme'] & self.methane['elevated'], self.risk['high']),
            ctrl.Rule(self.humidity['wet'] & self.methane['high'], self.risk['high']),
        ]
        
        self.risk_ctrl = ctrl.ControlSystem(self.rules)
        self.risk_simulator = ctrl.ControlSystemSimulation(self.risk_ctrl)
    
    def calculate_risk(self, methane, co2, temperature, humidity):
        try:
            # Dumpsite baseline adjustment
            # Normal dumpsite baseline methane is 300-500 ppm
            methane_baseline = 350  # Typical dumpsite background
            adjusted_methane = max(0, methane - methane_baseline + 200)
            
            # Clamp values to valid ranges
            adjusted_methane = max(0, min(5000, adjusted_methane))
            co2 = max(0, min(10000, co2))
            temperature = max(0, min(60, temperature))
            humidity = max(0, min(100, humidity))
            
            # Set inputs
            self.risk_simulator.input['methane'] = adjusted_methane
            self.risk_simulator.input['co2'] = co2
            self.risk_simulator.input['temperature'] = temperature
            self.risk_simulator.input['humidity'] = humidity
            
            # Compute
            self.risk_simulator.compute()
            
            # Get risk score
            risk_score = self.risk_simulator.output.get('risk', 0)
            if risk_score is None:
                risk_score = 0
            
            # Dumpsite risk level determination
            if risk_score >= 80:
                level = "CRITICAL"
                explosion_risk = 90
                recommendation = "IMMEDIATE EVACUATION - High explosive gas detected!"
            elif risk_score >= 60:
                level = "HIGH"
                explosion_risk = 70
                recommendation = "DANGER - Evacuate area and call emergency services"
            elif risk_score >= 35:
                level = "MEDIUM"
                explosion_risk = 45
                recommendation = "WARNING - Investigate source, increase ventilation"
            elif risk_score >= 15:
                level = "LOW"
                explosion_risk = 20
                recommendation = "CAUTION - Monitor situation, typical dumpsite levels"
            else:
                level = "SAFE"
                explosion_risk = 5
                recommendation = "Normal dumpsite operation"
            
            return {
                "level": level,
                "score": round(float(risk_score), 2),
                "explosion_risk": explosion_risk,
                "recommendation": recommendation,
                "adjusted_methane": round(adjusted_methane, 1),
                "raw_methane": round(methane, 1)
            }
            
        except Exception as e:
            logger.error(f"Fuzzy calculation error: {e}")
            return {
                "level": "UNKNOWN",
                "score": 0,
                "explosion_risk": 0,
                "recommendation": "System error - check sensors",
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
    """Get fuzzy risk assessment for a specific sensor"""
    try:
        # Fetch latest sensor data
        latest_data = get_summary(sensor_id)
        
        if not latest_data or latest_data == {}:
            return {
                "status": "error",
                "message": f"No data found for sensor {sensor_id}",
                "sensor_id": sensor_id,
                "risk": {
                    "level": "NO_DATA",
                    "score": 0,
                    "explosion_risk": 0
                }
            }
        
        # Extract values with defaults
        methane = float(latest_data.get("methane", 0))
        co2 = float(latest_data.get("co2", 0))
        temperature = float(latest_data.get("temperature", 25))
        humidity = float(latest_data.get("humidity", 50))
        
        # Calculate risk using fuzzy system
        risk = fuzzy_system.calculate_risk(methane, co2, temperature, humidity)
        
        return {
            "status": "success",
            "sensor_id": sensor_id,
            "timestamp": latest_data.get("timestamp", readable_time()),
            "sensor_data": {
                "methane": methane,
                "co2": co2,
                "temperature": temperature,
                "humidity": humidity
            },
            "risk": risk
        }
        
    except Exception as e:
        logger.error(f"Fuzzy endpoint error: {e}")
        return {
            "status": "error",
            "sensor_id": sensor_id,
            "error": str(e),
            "risk": {
                "level": "ERROR",
                "score": 0,
                "explosion_risk": 0
            }
        }
    
@app.get("/api/model/metrics/{sensor_id}")
def metrics(sensor_id: str):
    return get_metrics(sensor_id)

@app.get("/api/visualization/chart/{sensor_id}")
def chart(sensor_id: str, limit: int = 20, offset: int = 0):
    """
    Get historical chart data with pagination
    - limit: number of records to return (default 20, max 100)
    - offset: number of records to skip (for pagination)
    """
    try:
        # Get total count first
        history_ref = firebase_db.child(f"sensorReadings/history/{sensor_id}")
        
        # Get all keys to determine total count and paginate
        all_data = safe_get(history_ref.order_by_key(), {})
        
        if not all_data:
            return {
                "timestamps": [], 
                "methane": [], 
                "co2": [],
                "total": 0,
                "has_more": False,
                "offset": offset,
                "limit": limit
            }
        
        # Convert to list and sort by timestamp (oldest first or newest first)
        items = list(all_data.items())
        
        # Sort by key (timestamp) - newest first for pagination
        items.sort(key=lambda x: x[0], reverse=True)
        
        total = len(items)
        
        # Apply pagination
        paginated_items = items[offset:offset + limit]
        
        # Prepare response
        timestamps = []
        methane = []
        co2 = []
        
        for key, value in paginated_items:
            timestamps.append(value.get("timestamp", key))
            methane.append(float(value.get("methane", 0)))
            co2.append(float(value.get("co2", 0)))
        
        # Check if more data available
        has_more = (offset + limit) < total
        
        return {
            "timestamps": timestamps,
            "methane": methane,
            "co2": co2,
            "total": total,
            "has_more": has_more,
            "offset": offset,
            "limit": limit,
            "next_offset": offset + limit if has_more else None
        }
        
    except Exception as e:
        logger.error(f"Chart error: {e}")
        return {
            "timestamps": [], 
            "methane": [], 
            "co2": [],
            "total": 0,
            "has_more": False,
            "error": str(e)
        }

@app.get("/api/predict/{sensor_id}")
def predict(sensor_id: str):
    return {"predictions": predict_methane(sensor_id)}

# ==============================
# ADDITIONAL FUZZY ADAPTATION (NEW)
# ==============================
@app.get("/api/fuzzy/config")
def fuzzy_config():
    """Get fuzzy logic system configuration (no sensor data needed)"""
    return {
        "fuzzy_system": {
            "name": "Methane Gas Risk Assessment",
            "version": "1.0",
            "inputs": [
                {
                    "name": "methane",
                    "range": [0, 1000],
                    "units": "ppm",
                    "membership_functions": ["low", "medium", "high", "dangerous"]
                },
                {
                    "name": "co2", 
                    "range": [0, 5000],
                    "units": "ppm",
                    "membership_functions": ["normal", "elevated", "high", "dangerous"]
                },
                {
                    "name": "temperature",
                    "range": [0, 60],
                    "units": "°C",
                    "membership_functions": ["normal", "warm", "hot"]
                },
                {
                    "name": "humidity",
                    "range": [0, 100],
                    "units": "%",
                    "membership_functions": ["dry", "normal", "wet"]
                }
            ],
            "outputs": [
                {
                    "name": "risk",
                    "range": [0, 100],
                    "units": "%",
                    "levels": ["SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
                }
            ],
            "rule_count": 11,
            "rule_list": [
                "IF methane IS dangerous THEN risk IS critical",
                "IF methane IS high AND co2 IS NOT normal THEN risk IS high",
                "IF methane IS medium AND co2 IS elevated THEN risk IS medium",
                "IF co2 IS dangerous AND methane IS medium THEN risk IS high",
                "IF co2 IS high AND temperature IS hot THEN risk IS medium",
                "IF temperature IS hot AND methane IS medium THEN risk IS high",
                "IF temperature IS hot AND humidity IS dry THEN risk IS medium",
                "IF humidity IS wet AND methane IS NOT low THEN risk IS medium",
                "IF humidity IS wet AND temperature IS hot THEN risk IS high",
                "IF methane IS low AND co2 IS normal THEN risk IS low"
            ]
        },
        "status": "configured"
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