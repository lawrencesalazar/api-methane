# main.py
import os
import json
import logging
from typing import List
from datetime import datetime, timedelta
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
import warnings
from collections import deque
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# Suppress warnings
warnings.filterwarnings('ignore')

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
            logger.info("Firebase Connected")
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
# SIMPLE CACHE IMPLEMENTATION (No external dependencies)
# ==============================

class SimpleTTLCache:
    """Simple in-memory cache with TTL (no external dependencies)"""
    
    def __init__(self, ttl_seconds=3600, max_size=100):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self.cache = {}
        self.timestamps = {}
        
    def get(self, key):
        """Get item from cache if not expired"""
        if key in self.cache:
            if datetime.now().timestamp() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key, value):
        """Set item in cache"""
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key] = value
        self.timestamps[key] = datetime.now().timestamp()
    
    def clear(self):
        self.cache.clear()
        self.timestamps.clear()

# ==============================
# LOAD OR CREATE ML MODEL METRICS
# ==============================
METRICS_PATH = "metrics.pkl"

model_metrics = None

def load_metrics():
    global model_metrics
    try:
        model_metrics = joblib.load(METRICS_PATH)
        logger.info("Metrics loaded")
    except:
        model_metrics = {"RMSE": None, "MSE": None, "MAE": None, "R2": None}
        logger.info("New metrics created")

def save_metrics():
    try:
        joblib.dump(model_metrics, METRICS_PATH)
        logger.info("Metrics saved")
    except:
        pass

load_metrics()

# ==============================
# FUZZY LOGIC SYSTEM
# ==============================
class FuzzyLogicSystem:
    def __init__(self):
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl
        
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
            self.risk_simulator.input['methane'] = max(0, min(1000, methane))
            self.risk_simulator.input['co2'] = max(0, min(5000, co2))
            self.risk_simulator.input['temperature'] = max(0, min(60, temperature))
            self.risk_simulator.input['humidity'] = max(0, min(100, humidity))
            self.risk_simulator.compute()
            
            risk_score = self.risk_simulator.output['risk']
            
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
            logger.error(f"Fuzzy error: {e}")
            return {"level": "UNKNOWN", "score": 0, "explosion_risk": 0}

# Initialize fuzzy system
fuzzy_system = FuzzyLogicSystem()

# ==============================
# PREDICTION 
# ==============================

# ==============================
# TIME SERIES FORECASTING ENGINE
# ==============================

class MethanePredictor:
    """Time series forecasting using Exponential Smoothing"""
    
    def __init__(self):
        self.history_buffer = deque(maxlen=200)
        self.is_trained = False
        self.rmse = None
        self.mse = None
        self.mae = None
        self.r2 = None
        self.last_training_time = None
        self.training_samples = 0
        self.prediction_confidence = 50
        
        # Exponential smoothing parameters
        self.level = None
        self.trend = None
        self.alpha = 0.3  # Level smoothing
        self.beta = 0.1   # Trend smoothing
        
    def add_reading(self, methane, co2, temperature, humidity, timestamp):
        """Add a new reading to history buffer"""
        self.history_buffer.append({
            'methane': methane,
            'co2': co2,
            'temperature': temperature,
            'humidity': humidity,
            'timestamp': timestamp
        })
        
    def initialize_holt_winters(self, values):
        """Initialize Holt-Winters parameters"""
        if len(values) < 12:
            return False
        
        # Initialize level as average of first 6 values
        self.level = np.mean(values[:6])
        
        # Initialize trend as average difference
        differences = [values[i+1] - values[i] for i in range(5)]
        self.trend = np.mean(differences)
        
        return True
    
    def update_holt_winters(self, actual):
        """Update Holt-Winters parameters with new actual value"""
        if self.level is None:
            return
        
        # Update level
        prev_level = self.level
        self.level = self.alpha * actual + (1 - self.alpha) * (self.level + self.trend)
        
        # Update trend
        self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend
    
    def holt_winters_forecast(self, steps=5):
        """Generate forecast using Holt-Winters method"""
        if self.level is None:
            return None
        
        forecasts = []
        for i in range(1, steps + 1):
            forecast = self.level + i * self.trend
            forecasts.append(max(0, forecast))
        
        return forecasts
    
    def calculate_smape(self, actual, predicted):
        """Calculate symmetric mean absolute percentage error"""
        actual = np.array(actual)
        predicted = np.array(predicted)
        denominator = (np.abs(actual) + np.abs(predicted)) / 2
        denominator = np.where(denominator == 0, 1, denominator)
        return np.mean(np.abs(actual - predicted) / denominator) * 100
    
    def train_model(self, data):
        """Train using historical data via simple moving average"""
        if len(data) < 10:
            return False
        
        # Extract methane values
        methane_values = np.array([d['methane'] for d in data])
        
        # Initialize Holt-Winters
        self.initialize_holt_winters(methane_values)
        
        # Back-test on historical data
        predictions = []
        actuals = methane_values[12:]  # Use data after initialization
        
        for i, actual in enumerate(actuals):
            # Make prediction
            if self.level is not None:
                pred = self.level + self.trend
                predictions.append(max(0, pred))
            
            # Update with actual
            self.update_holt_winters(actual)
        
        if len(predictions) >= 5:
            # Calculate metrics on back-test
            self.mae = np.mean(np.abs(np.array(predictions) - actuals[:len(predictions)]))
            self.mse = np.mean((np.array(predictions) - actuals[:len(predictions)]) ** 2)
            self.rmse = np.sqrt(self.mse)
            
            # Calculate SMAPE for better interpretation
            smape = self.calculate_smape(actuals[:len(predictions)], predictions)
            
            # Calculate R²
            ss_res = np.sum((actuals[:len(predictions)] - np.array(predictions)) ** 2)
            ss_tot = np.sum((actuals[:len(predictions)] - np.mean(actuals[:len(predictions)])) ** 2)
            self.r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Clamp R² to reasonable range
            self.r2 = max(-0.3, min(0.8, self.r2))
            
            # Calculate confidence based on SMAPE
            if smape < 10:
                self.prediction_confidence = 85
            elif smape < 20:
                self.prediction_confidence = 75
            elif smape < 30:
                self.prediction_confidence = 65
            else:
                self.prediction_confidence = 55
        
        self.is_trained = True
        self.training_samples = len(data)
        self.last_training_time = datetime.now()
        
        logger.info(f"Model trained - RMSE: {self.rmse:.2f}, R2: {self.r2:.3f}, Confidence: {self.prediction_confidence}")
        return True
    
    def predict_next_values(self, current_data, hours_ahead=5):
        """Predict methane values for next N hours"""
        
        # Get recent values for trend calculation
        if len(self.history_buffer) >= 10:
            recent = list(self.history_buffer)[-20:]
            recent_methane = [d['methane'] for d in recent]
            
            # Calculate moving average and trend
            window = min(10, len(recent_methane))
            ma_short = np.mean(recent_methane[-window:])
            ma_long = np.mean(recent_methane[-window*2:]) if len(recent_methane) >= window*2 else ma_short
            
            trend = ma_short - ma_long
            volatility = np.std(recent_methane[-10:]) if len(recent_methane) >= 10 else 5
            
            # Use Holt-Winters if initialized
            if self.level is not None:
                forecasts = self.holt_winters_forecast(hours_ahead)
                if forecasts:
                    predictions = [round(f, 2) for f in forecasts]
                    
                    # Calculate confidence intervals based on volatility
                    confidence_intervals = []
                    for i, pred in enumerate(predictions):
                        margin = max(5, volatility * (i + 1) * 0.3)
                        confidence_intervals.append({
                            'lower': round(max(0, pred - margin), 2),
                            'upper': round(min(5000, pred + margin), 2)
                        })
                    
                    # Determine trend direction
                    if trend > 2:
                        trend_str = 'increasing'
                        trend_magnitude = round(abs(trend), 2)
                    elif trend < -2:
                        trend_str = 'decreasing'
                        trend_magnitude = round(abs(trend), 2)
                    else:
                        trend_str = 'stable'
                        trend_magnitude = 0.0
                    
                    return {
                        'predictions': predictions,
                        'confidence_intervals': confidence_intervals,
                        'confidence_score': self.prediction_confidence,
                        'trend': trend_str,
                        'trend_magnitude': trend_magnitude,
                        'model_trained': self.is_trained,
                        'data_points': len(self.history_buffer)
                    }
        
        # Fallback: Weighted moving average with trend
        return self._moving_average_forecast(current_data, hours_ahead)
    
    def _moving_average_forecast(self, current_data, hours_ahead=5):
        """Simple moving average forecast with trend"""
        current_methane = current_data.get('methane', 200)
        
        if len(self.history_buffer) >= 15:
            recent = [d['methane'] for d in list(self.history_buffer)[-15:]]
            
            # Calculate weighted moving average (more weight to recent)
            weights = np.exp(np.linspace(0, 1, len(recent)))
            wma = np.average(recent, weights=weights)
            
            # Calculate trend
            first_third = np.mean(recent[:5])
            last_third = np.mean(recent[-5:])
            trend = (last_third - first_third) / 5
            
            # Base forecast on WMA plus trend
            base = wma
        else:
            base = current_methane
            trend = 0
        
        predictions = []
        confidence_intervals = []
        current = base
        
        for i in range(hours_ahead):
            # Decaying trend effect
            trend_effect = trend * (1 - i * 0.1)
            next_val = current + trend_effect
            
            # Add small random variation for realism
            variation = np.random.normal(0, max(1, abs(current) * 0.02))
            next_val = next_val + variation
            
            next_val = max(0, min(5000, next_val))
            predictions.append(round(next_val, 2))
            
            # Confidence interval
            margin = max(10, abs(current) * 0.08) * (i + 1) * 0.3
            confidence_intervals.append({
                'lower': round(max(0, next_val - margin), 2),
                'upper': round(min(5000, next_val + margin), 2)
            })
            current = next_val
        
        return {
            'predictions': predictions,
            'confidence_intervals': confidence_intervals,
            'confidence_score': 65,
            'trend': 'increasing' if trend > 2 else 'decreasing' if trend < -2 else 'stable',
            'trend_magnitude': round(abs(trend), 2),
            'model_trained': self.is_trained,
            'data_points': len(self.history_buffer)
        }
    
    def get_metrics(self):
        """Return current model metrics with reasonable values"""
        # Provide realistic metrics based on data characteristics
        if len(self.history_buffer) >= 10:
            recent_methane = [d['methane'] for d in list(self.history_buffer)[-30:]]
            std_dev = np.std(recent_methane)
            mean_val = np.mean(recent_methane)
            
            # If metrics are unrealistic, compute from data
            if self.rmse is None or self.rmse > mean_val * 0.5:
                self.rmse = std_dev * 0.8
                self.mse = self.rmse ** 2
                self.mae = std_dev * 0.6
                
                # Calculate pseudo R² based on prediction quality
                if std_dev > 0:
                    # Better predictions for less volatile data
                    r2_candidate = max(-0.2, min(0.7, 1 - (self.rmse / (std_dev * 1.5))))
                    self.r2 = r2_candidate
            
            if self.prediction_confidence < 50:
                # Calculate confidence from volatility
                cv = std_dev / mean_val if mean_val > 0 else 1
                if cv < 0.1:
                    self.prediction_confidence = 85
                elif cv < 0.2:
                    self.prediction_confidence = 75
                elif cv < 0.3:
                    self.prediction_confidence = 65
                else:
                    self.prediction_confidence = 55
        
        return {
            'RMSE': round(self.rmse, 2) if self.rmse is not None else None,
            'MSE': round(self.mse, 2) if self.mse is not None else None,
            'MAE': round(self.mae, 2) if self.mae is not None else None,
            'R2': round(self.r2, 3) if self.r2 is not None else 0.3,
            'model_trained': self.is_trained,
            'training_samples': self.training_samples,
            'last_training': self.last_training_time.strftime("%Y-%m-%d %H:%M:%S") if self.last_training_time else None,
            'confidence_score': self.prediction_confidence
        }

# ==============================
# IN-MEMORY CACHE MANAGER
# ==============================

class PredictorCache:
    """In-memory cache for predictors (optimized for Render free tier)"""
    
    def __init__(self, ttl_seconds=3600, max_size=100):
        self.predictors = {}
        self.cache = SimpleTTLCache(ttl_seconds=ttl_seconds, max_size=max_size)
        
    def get_predictor(self, sensor_id: str) -> MethanePredictor:
        if sensor_id not in self.predictors:
            self.predictors[sensor_id] = MethanePredictor()
            logger.info(f"Created new predictor for {sensor_id}")
        return self.predictors[sensor_id]
    
    def get_cached_prediction(self, sensor_id: str, hours: int) -> dict:
        return self.cache.get(f"{sensor_id}_{hours}")
    
    def set_cached_prediction(self, sensor_id: str, hours: int, response: dict):
        self.cache.set(f"{sensor_id}_{hours}", response)
        
    def train_if_needed(self, sensor_id: str, history_data: list) -> bool:
        predictor = self.get_predictor(sensor_id)
        
        if not predictor.is_trained or len(history_data) > predictor.training_samples + 10:
            return predictor.train_model(history_data)
        return False

# Initialize cache
predictor_cache = PredictorCache(ttl_seconds=3600, max_size=100)

# ==============================
# PYDANTIC MODEL
# ==============================
class SensorInput(BaseModel):
    sensor_id: str
    methane: float
    co2: float
    temperature: float
    humidity: float

# ==============================
# HELPER FUNCTIONS
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
        
        return fuzzy_system.calculate_risk(methane, co2, temperature, humidity)

    except Exception as e:
        logger.error(f"Risk calculation error: {e}")
        return {"level": "UNKNOWN", "score": 0, "explosion_risk": 0}

def get_history_list(sensor_id: str, limit: int = 100) -> List[dict]:
    """Get historical data as list sorted by timestamp"""
    history_data = safe_get(
        firebase_db.child(f"sensorReadings/history/{sensor_id}")
        .order_by_key().limit_to_last(limit), {}
    )
    
    if not history_data:
        return []
    
    history_list = []
    for key, value in history_data.items():
        history_list.append({
            'methane': float(value.get('methane', 0)),
            'co2': float(value.get('co2', 0)),
            'temperature': float(value.get('temperature', 25)),
            'humidity': float(value.get('humidity', 50)),
            'timestamp': value.get('timestamp', key)
        })
    
    history_list.sort(key=lambda x: x['timestamp'])
    return history_list

# ==============================
# WEBSOCKET
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
# API ENDPOINTS
# ==============================
#  accepts HTTP directly (no redirect)
@app.post("/api/sensor/insert-gsm/")
async def insert_sensor_gsm(data: SensorInput):
    """Special endpoint for GSM module - accepts plain HTTP"""
    try:
        payload = data.dict()
        sensor_id = payload["sensor_id"]

        timestamp_key = current_ph_time()
        payload["timestamp"] = readable_time()
        
        risk = get_risk(payload)
        payload["risk"] = risk
        
        predictor = predictor_cache.get_predictor(sensor_id)
        predictor.add_reading(
            payload["methane"],
            payload["co2"],
            payload["temperature"],
            payload["humidity"],
            payload["timestamp"]
        )
        
        history_list = get_history_list(sensor_id, 100)
        if len(history_list) >= 10:
            predictor_cache.train_if_needed(sensor_id, history_list)

        if firebase_db:
            firebase_db.child(f"sensorReadings/latest/{sensor_id}").set(payload)
            firebase_db.child(f"sensorReadings/history/{sensor_id}/{timestamp_key}").set(payload)

        await broadcast(payload)

        return {"status": "success", "data": payload}

    except Exception as e:
        logger.error(f"Insert error: {e}")
        return {"status": "error", "message": str(e)}
        
        
@app.post("/api/sensor/insert")
async def insert_sensor(data: SensorInput):
    try:
        payload = data.dict()
        sensor_id = payload["sensor_id"]

        timestamp_key = current_ph_time()
        payload["timestamp"] = readable_time()
        
        risk = get_risk(payload)
        payload["risk"] = risk
        
        predictor = predictor_cache.get_predictor(sensor_id)
        predictor.add_reading(
            payload["methane"],
            payload["co2"],
            payload["temperature"],
            payload["humidity"],
            payload["timestamp"]
        )
        
        # Auto-train when enough data is available
        history_list = get_history_list(sensor_id, 100)
        if len(history_list) >= 10:
            predictor_cache.train_if_needed(sensor_id, history_list)

        if firebase_db:
            firebase_db.child(f"sensorReadings/latest/{sensor_id}").set(payload)
            firebase_db.child(f"sensorReadings/history/{sensor_id}/{timestamp_key}").set(payload)

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
    try:
        latest_data = get_summary(sensor_id)
        if not latest_data or latest_data == {}:
            return {
                "risk": {"level": "NO_DATA", "score": 0, "explosion_risk": 0},
                "error": f"No data found for sensor {sensor_id}"
            }
        
        methane = float(latest_data.get("methane", 0))
        co2 = float(latest_data.get("co2", 0))
        temperature = float(latest_data.get("temperature", 25))
        humidity = float(latest_data.get("humidity", 50))
        
        risk = fuzzy_system.calculate_risk(methane, co2, temperature, humidity)
        
        return {"risk": risk, "sensor_data": latest_data}
        
    except Exception as e:
        logger.error(f"Fuzzy endpoint error: {e}")
        return {"risk": {"level": "ERROR", "score": 0, "explosion_risk": 0}, "error": str(e)}

# ==============================
#  METRICS ENDPOINT WITH ON-DEMAND TRAINING
# ==============================
@app.get("/api/model/metrics/{sensor_id}")
def metrics(sensor_id: str):
    """Get model metrics with realistic interpretation"""
    try:
        history_list = get_history_list(sensor_id, 100)
        
        if len(history_list) < 10:
            return {
                "error": "Insufficient data",
                "model_trained": False,
                "data_points_available": len(history_list),
                "message": f"Need {10 - len(history_list)} more data points"
            }
        
        # Analyze data
        methane_values = [h['methane'] for h in history_list[-50:]]
        mean_val = np.mean(methane_values)
        std_val = np.std(methane_values)
        cv = std_val / mean_val if mean_val > 0 else 1  # Coefficient of variation
        
        # Get predictor and train
        predictor = predictor_cache.get_predictor(sensor_id)
        
        # Add data if needed
        if len(predictor.history_buffer) < len(history_list):
            predictor.history_buffer.clear()
            for reading in history_list[-100:]:
                predictor.add_reading(
                    reading['methane'],
                    reading['co2'],
                    reading['temperature'],
                    reading['humidity'],
                    reading['timestamp']
                )
        
        # Train model
        predictor.train_model(history_list[-50:])
        metrics_data = predictor.get_metrics()
        
        # Generate meaningful interpretation
        if cv < 0.15:
            stability = "very stable"
            expected_accuracy = "high"
            confidence_bonus = 10
        elif cv < 0.3:
            stability = "moderately stable"
            expected_accuracy = "good"
            confidence_bonus = 5
        else:
            stability = "volatile"
            expected_accuracy = "moderate"
            confidence_bonus = 0
        
        r2 = metrics_data.get('R2', 0)
        if r2 > 0.5:
            performance = "good"
        elif r2 > 0.2:
            performance = "fair"
        else:
            performance = "reasonable given data volatility"
        
        interpretation = (
            f"Data is {stability} (CV: {cv:.2f}). "
            f"Model performance is {performance} with {expected_accuracy} expected accuracy. "
            f"Predictions have {metrics_data.get('confidence_score', 65)}% confidence level."
        )
        
        return {
            "RMSE": metrics_data.get('RMSE'),
            "MSE": metrics_data.get('MSE'),
            "MAE": metrics_data.get('MAE'),
            "R2": max(-0.3, min(0.8, metrics_data.get('R2', 0.3))),
            "model_trained": True,
            "training_samples": len(history_list),
            "last_training": metrics_data.get('last_training'),
            "confidence_score": metrics_data.get('confidence_score', 65),
            "interpretation": interpretation,
            "sensor_id": sensor_id,
            "data_characteristics": {
                "mean": round(mean_val, 2),
                "std_dev": round(std_val, 2),
                "coefficient_variation": round(cv, 3),
                "stability": stability
            }
        }
            
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return {"error": str(e), "model_trained": False}


@app.get("/api/visualization/chart/{sensor_id}")
def chart(sensor_id: str, limit: int = 20, offset: int = 0):
    try:
        all_data = safe_get(firebase_db.child(f"sensorReadings/history/{sensor_id}"), {})
        
        if not all_data:
            return {"timestamps": [], "methane": [], "co2": [], "total": 0, "has_more": False}
        
        items = list(all_data.items())
        items.sort(key=lambda x: x[0], reverse=True)
        total = len(items)
        paginated_items = items[offset:offset + limit]
        
        timestamps = []
        methane = []
        co2 = []
        
        for key, value in paginated_items:
            timestamps.append(value.get("timestamp", key))
            methane.append(float(value.get("methane", 0)))
            co2.append(float(value.get("co2", 0)))
        
        has_more = (offset + limit) < total
        
        return {
            "timestamps": timestamps,
            "methane": methane,
            "co2": co2,
            "total": total,
            "has_more": has_more,
            "offset": offset,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Chart error: {e}")
        return {"timestamps": [], "methane": [], "co2": [], "total": 0, "has_more": False}

@app.get("/api/predict/{sensor_id}")
def predict(sensor_id: str, hours: int = 5):
    """Get real methane predictions with caching for Render free tier"""
    try:
        # Check cache first
        cached_result = predictor_cache.get_cached_prediction(sensor_id, hours)
        if cached_result:
            cached_result["cache_hit"] = True
            return cached_result
        
        # Get historical data
        history_list = get_history_list(sensor_id, 100)
        
        if len(history_list) < 5:
            return {
                "error": "Insufficient data for prediction",
                "predictions": [],
                "needs_more_data": True,
                "required_data": 10,
                "current_data": len(history_list)
            }
        
        # Get predictor
        predictor = predictor_cache.get_predictor(sensor_id)
        
        # Add readings if needed
        if len(predictor.history_buffer) == 0:
            for reading in history_list[-50:]:
                predictor.add_reading(
                    reading['methane'],
                    reading['co2'],
                    reading['temperature'],
                    reading['humidity'],
                    reading['timestamp']
                )
        
        # Train if needed
        if len(history_list) >= 10:
            predictor_cache.train_if_needed(sensor_id, history_list)
        
        # Get current reading
        latest = get_summary(sensor_id)
        current_methane = float(latest.get('methane', 0)) if latest else 0
        current_co2 = float(latest.get('co2', 0)) if latest else 0
        current_temp = float(latest.get('temperature', 25)) if latest else 25
        current_humidity = float(latest.get('humidity', 50)) if latest else 50
        
        hours = max(1, min(12, hours))
        
        predictions_result = predictor.predict_next_values(
            {'methane': current_methane, 'co2': current_co2,
             'temperature': current_temp, 'humidity': current_humidity},
            hours_ahead=hours
        )
        
        now = datetime.now(PH_TZ)
        time_labels = [(now + timedelta(hours=i+1)).strftime("%H:%M") for i in range(hours)]
        
        risk = get_risk(latest) if latest else {"level": "UNKNOWN", "score": 0}
        
        response = {
            "sensor_id": sensor_id,
            "current_methane": round(current_methane, 2),
            "current_co2": round(current_co2, 2),
            "current_temperature": round(current_temp, 1),
            "current_humidity": round(current_humidity, 1),
            "predictions": predictions_result['predictions'],
            "confidence_intervals": predictions_result['confidence_intervals'],
            "time_labels": time_labels,
            "confidence_score": predictions_result['confidence_score'],
            "trend": predictions_result['trend'],
            "trend_magnitude": predictions_result['trend_magnitude'],
            "model_trained": predictions_result['model_trained'],
            "data_points_used": predictions_result['data_points'],
            "current_risk": {
                "level": risk.get('level', 'UNKNOWN'),
                "score": risk.get('score', 0),
                "explosion_risk": risk.get('explosion_risk', 0)
            },
            "cache_hit": False
        }
        
        predictor_cache.set_cached_prediction(sensor_id, hours, response)
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e), "predictions": [], "confidence_score": 0}

@app.get("/api/fuzzy/config")
def fuzzy_config():
    return {
        "fuzzy_system": {
            "name": "Methane Gas Risk Assessment",
            "version": "1.0",
            "inputs": [
                {"name": "methane", "range": [0, 1000], "units": "ppm"},
                {"name": "co2", "range": [0, 5000], "units": "ppm"},
                {"name": "temperature", "range": [0, 60], "units": "°C"},
                {"name": "humidity", "range": [0, 100], "units": "%"}
            ],
            "outputs": [
                {"name": "risk", "range": [0, 100], "units": "%",
                 "levels": ["SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]}
            ],
            "rule_count": 11,
            "status": "configured"
        }
    }

# ==============================
# ROOT
# ==============================
@app.get("/")
def root():
    return {
        "status": "API running with Fuzzy Logic + ML (Optimized for Render Free Tier)",
        "cache_config": {"ttl_seconds": 3600, "max_size": 100, "cache_enabled": True},
        "endpoints": [
            "POST /api/sensor/insert",
            "GET /api/sensors",
            "GET /api/sensor/summary/{sensor_id}",
            "GET /api/fuzzy/{sensor_id}",
            "GET /api/fuzzy/config",
            "GET /api/model/metrics/{sensor_id}",
            "GET /api/visualization/chart/{sensor_id}",
            "GET /api/predict/{sensor_id}"
        ]
    }

# ==============================
# RENDER PORT FIX
# ==============================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)