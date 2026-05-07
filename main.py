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
from cachetools import TTLCache

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
# PREDICTION ENGINE WITH IN-MEMORY CACHE
# ==============================

class MethanePredictor:
    """Real-time methane prediction using ML and fuzzy logic"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.poly_features = PolynomialFeatures(degree=2)
        self.is_trained = False
        self.history_buffer = deque(maxlen=100)
        self.prediction_confidence = 0.0
        
        # Metrics storage
        self.rmse = None
        self.mse = None
        self.mae = None
        self.r2 = None
        self.last_training_time = None
        self.training_samples = 0
        
    def add_reading(self, methane, co2, temperature, humidity, timestamp):
        """Add a new reading to history buffer"""
        self.history_buffer.append({
            'methane': methane,
            'co2': co2,
            'temperature': temperature,
            'humidity': humidity,
            'timestamp': timestamp
        })
        
    def prepare_features(self, data):
        """Prepare features for ML model"""
        if len(data) < 5:
            return None
            
        methane_values = [d['methane'] for d in data]
        co2_values = [d['co2'] for d in data]
        temp_values = [d['temperature'] for d in data]
        
        methane_trend = methane_values[-1] - methane_values[0] if len(methane_values) > 1 else 0
        co2_trend = co2_values[-1] - co2_values[0] if len(co2_values) > 1 else 0
        
        methane_mean = np.mean(methane_values[-5:]) if len(methane_values) >= 5 else methane_values[-1]
        methane_std = np.std(methane_values[-5:]) if len(methane_values) >= 5 else 0
        methane_max = max(methane_values[-10:]) if methane_values else 0
        methane_min = min(methane_values[-10:]) if methane_values else 0
        
        if len(methane_values) >= 3:
            x = np.arange(len(methane_values[-5:]))
            y = np.array(methane_values[-5:])
            slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
        else:
            slope = 0
        
        features = [
            methane_values[-1], methane_mean, methane_std, slope,
            methane_trend, co2_values[-1] if co2_values else 0,
            co2_trend, temp_values[-1] if temp_values else 25,
            methane_max, methane_min
        ]
        
        return np.array(features).reshape(1, -1)
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        try:
            self.mse = mean_squared_error(y_true, y_pred)
            self.rmse = np.sqrt(self.mse)
            self.mae = mean_absolute_error(y_true, y_pred)
            self.r2 = r2_score(y_true, y_pred)
            
            return {
                'RMSE': round(self.rmse, 4),
                'MSE': round(self.mse, 4),
                'MAE': round(self.mae, 4),
                'R2': round(self.r2, 4)
            }
        except Exception as e:
            logger.error(f"Metrics calculation error: {e}")
            return {'RMSE': None, 'MSE': None, 'MAE': None, 'R2': None}
    
    def train_model(self, data):
        """Train ML model on historical data and calculate metrics"""
        if len(data) < 10:
            return False
            
        X_list = []
        y_list = []
        
        for i in range(5, len(data) - 1):
            window = data[i-5:i+1]
            features = self.prepare_features(window)
            if features is not None:
                X_list.append(features.flatten())
                y_list.append(data[i+1]['methane'] if i+1 < len(data) else data[i]['methane'])
        
        if len(X_list) > 5:
            X = np.array(X_list)
            y = np.array(y_list)
            X_poly = self.poly_features.fit_transform(X)
            self.model.fit(X_poly, y)
            
            y_pred = self.model.predict(X_poly)
            metrics = self.calculate_metrics(y, y_pred)
            
            self.rmse = metrics['RMSE']
            self.mse = metrics['MSE']
            self.mae = metrics['MAE']
            self.r2 = metrics['R2']
            
            self.is_trained = True
            self.prediction_confidence = (self.r2 if self.r2 else 0.5) * 100
            self.last_training_time = datetime.now()
            self.training_samples = len(X_list)
            
            logger.info(f"Model trained: RMSE={self.rmse}, R2={self.r2}, samples={self.training_samples}")
            return True
        return False
    
    def predict_next_values(self, current_data, hours_ahead=5):
        """Predict methane values for next N hours"""
        if len(self.history_buffer) < 10:
            return self._simple_prediction(current_data, hours_ahead)
        
        try:
            data_list = list(self.history_buffer)
            features = self.prepare_features(data_list)
            
            if features is None:
                return self._simple_prediction(current_data, hours_ahead)
            
            predictions = []
            confidence_intervals = []
            
            current_methane = current_data.get('methane', 0)
            
            # Calculate trend
            recent_methane = [d['methane'] for d in data_list[-10:]]
            if len(recent_methane) > 1:
                x = np.arange(len(recent_methane))
                slope = np.polyfit(x, recent_methane, 1)[0]
                trend = slope
            else:
                trend = 0
            
            for i in range(hours_ahead):
                if self.is_trained:
                    features_poly = self.poly_features.transform(features)
                    base_pred = self.model.predict(features_poly)[0]
                else:
                    base_pred = current_methane
                
                decay = 1.0 - (i * 0.05)
                
                if i == 0:
                    predicted = base_pred
                else:
                    trend_effect = trend * (1 + i * 0.1)
                    predicted = predictions[-1] + (trend_effect * 0.5) * decay
                
                predicted = max(0, min(5000, predicted))
                predictions.append(round(predicted, 2))
                
                margin = max(5, predicted * 0.1) * (i + 1) * 0.5
                confidence_intervals.append({
                    'lower': round(max(0, predicted - margin), 2),
                    'upper': round(min(5000, predicted + margin), 2)
                })
                
                current_methane = predicted
                
                new_features = features.copy()
                new_features[0][0] = predicted
                features = new_features
            
            return {
                'predictions': predictions,
                'confidence_intervals': confidence_intervals,
                'confidence_score': round(self.prediction_confidence, 1),
                'trend': 'increasing' if trend > 2 else 'decreasing' if trend < -2 else 'stable',
                'trend_magnitude': round(abs(trend), 2),
                'model_trained': self.is_trained,
                'data_points': len(self.history_buffer)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._simple_prediction(current_data, hours_ahead)
    
    def _simple_prediction(self, current_data, hours_ahead=5):
        """Fallback simple prediction"""
        current_methane = current_data.get('methane', 200)
        
        if len(self.history_buffer) >= 3:
            recent = [d['methane'] for d in list(self.history_buffer)[-5:]]
            if len(recent) > 1:
                trend = (recent[-1] - recent[0]) / len(recent)
            else:
                trend = 0
        else:
            trend = 0
        
        predictions = []
        confidence_intervals = []
        current = current_methane
        
        for i in range(hours_ahead):
            trend_factor = trend * (1 - i * 0.1)
            variation = np.random.normal(0, max(1, current * 0.03))
            next_val = current + trend_factor + variation
            next_val = max(0, min(5000, next_val))
            predictions.append(round(next_val, 2))
            
            margin = max(10, next_val * 0.15) * (i + 1) * 0.3
            confidence_intervals.append({
                'lower': round(max(0, next_val - margin), 2),
                'upper': round(min(5000, next_val + margin), 2)
            })
            current = next_val
        
        return {
            'predictions': predictions,
            'confidence_intervals': confidence_intervals,
            'confidence_score': 50.0,
            'trend': 'increasing' if trend > 1 else 'decreasing' if trend < -1 else 'stable',
            'trend_magnitude': round(abs(trend), 2),
            'model_trained': False,
            'data_points': len(self.history_buffer)
        }
    
    def get_metrics(self):
        """Return current model metrics"""
        return {
            'RMSE': self.rmse if self.rmse is not None else None,
            'MSE': self.mse if self.mse is not None else None,
            'MAE': self.mae if self.mae is not None else None,
            'R2': self.r2 if self.r2 is not None else None,
            'model_trained': self.is_trained,
            'training_samples': self.training_samples,
            'last_training': self.last_training_time.strftime("%Y-%m-%d %H:%M:%S") if self.last_training_time else None,
            'confidence_score': round(self.prediction_confidence, 1)
        }


# ==============================
# IN-MEMORY CACHE MANAGER
# ==============================

class PredictorCache:
    """In-memory cache for predictors (optimized for Render free tier)"""
    
    def __init__(self, ttl_seconds=3600, max_size=100):
        self.predictors = {}  # sensor_id -> predictor instance
        self.cache = TTLCache(maxsize=max_size, ttl=ttl_seconds)  # Response cache
        self.ttl = ttl_seconds
        
    def get_predictor(self, sensor_id: str) -> MethanePredictor:
        """Get or create predictor for sensor"""
        if sensor_id not in self.predictors:
            self.predictors[sensor_id] = MethanePredictor()
            logger.info(f"Created new predictor for {sensor_id}")
        return self.predictors[sensor_id]
    
    def get_cached_prediction(self, sensor_id: str, hours: int) -> dict:
        """Get cached prediction if available"""
        cache_key = f"{sensor_id}_{hours}"
        return self.cache.get(cache_key)
    
    def set_cached_prediction(self, sensor_id: str, hours: int, response: dict):
        """Cache prediction response"""
        cache_key = f"{sensor_id}_{hours}"
        self.cache[cache_key] = response
        
    def train_if_needed(self, sensor_id: str, history_data: list) -> bool:
        """Train only if not trained or significant new data"""
        predictor = self.get_predictor(sensor_id)
        
        if not predictor.is_trained or len(history_data) > predictor.training_samples + 10:
            result = predictor.train_model(history_data)
            if result:
                logger.info(f"Trained model for {sensor_id} with {len(history_data)} samples")
                # Update global metrics
                global model_metrics
                model_metrics = predictor.get_metrics()
                save_metrics()
            return result
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

@app.post("/api/sensor/insert")
async def insert_sensor(data: SensorInput):
    try:
        payload = data.dict()
        sensor_id = payload["sensor_id"]

        timestamp_key = current_ph_time()
        payload["timestamp"] = readable_time()
        
        # Calculate fuzzy risk
        risk = get_risk(payload)
        payload["risk"] = risk
        
        # Get predictor and add reading
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

        # Save to Firebase
        if firebase_db:
            firebase_db.child(f"sensorReadings/latest/{sensor_id}").set(payload)
            firebase_db.child(f"sensorReadings/history/{sensor_id}/{timestamp_key}").set(payload)

        # Broadcast to WebSocket clients
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

@app.get("/api/model/metrics/{sensor_id}")
def metrics(sensor_id: str):
    """Get actual ML model metrics calculated from historical data"""
    try:
        predictor = predictor_cache.get_predictor(sensor_id)
        
        if predictor.is_trained:
            metrics_data = predictor.get_metrics()
            
            # Add interpretation
            if metrics_data['R2'] and metrics_data['R2'] > 0.7:
                interpretation = "Excellent model performance. Predictions are reliable."
            elif metrics_data['R2'] and metrics_data['R2'] > 0.5:
                interpretation = "Good model performance. Predictions are reasonably accurate."
            elif metrics_data['R2'] and metrics_data['R2'] > 0.3:
                interpretation = "Fair model performance. Predictions should be used with caution."
            else:
                interpretation = "Poor model performance. More data needed for reliable predictions."
            
            return {
                **metrics_data,
                "interpretation": interpretation,
                "sensor_id": sensor_id
            }
        else:
            history_list = get_history_list(sensor_id, 100)
            return {
                "error": "Model not trained yet",
                "model_trained": False,
                "data_points_available": len(history_list),
                "data_points_needed": 10,
                "message": f"Need {max(0, 10 - len(history_list))} more data points for reliable metrics"
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
            logger.info(f"Returning cached prediction for {sensor_id}")
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
                "current_data": len(history_list),
                "message": f"Need {10 - len(history_list)} more readings for reliable prediction"
            }
        
        # Get predictor and add historical readings
        predictor = predictor_cache.get_predictor(sensor_id)
        
        # Add readings if predictor is empty
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
        
        # Get current/latest reading
        latest = get_summary(sensor_id)
        current_methane = float(latest.get('methane', 0)) if latest else 0
        current_co2 = float(latest.get('co2', 0)) if latest else 0
        current_temp = float(latest.get('temperature', 25)) if latest else 25
        current_humidity = float(latest.get('humidity', 50)) if latest else 50
        
        # Limit hours
        hours = max(1, min(12, hours))
        
        # Get predictions
        predictions_result = predictor.predict_next_values(
            {
                'methane': current_methane,
                'co2': current_co2,
                'temperature': current_temp,
                'humidity': current_humidity
            },
            hours_ahead=hours
        )
        
        # Generate time labels
        now = datetime.now(PH_TZ)
        time_labels = [(now + timedelta(hours=i+1)).strftime("%H:%M") for i in range(hours)]
        
        # Get current risk
        risk = get_risk(latest) if latest else {"level": "UNKNOWN", "score": 0}
        
        # Generate interpretation
        if predictions_result['predictions']:
            final_pred = predictions_result['predictions'][-1]
            change = final_pred - current_methane
            
            if risk.get('level') in ["CRITICAL", "HIGH"]:
                interpretation = f"⚠️ CRITICAL: Methane predicted to {'increase' if change > 0 else 'decrease'} from {current_methane:.0f} to {final_pred:.0f} ppm. Immediate action required."
            elif risk.get('level') == "MEDIUM":
                if change > 50:
                    interpretation = f"⚠️ WARNING: Methane expected to rise significantly to {final_pred:.0f} ppm. Increase monitoring."
                else:
                    interpretation = f"📊 Methane predicted to remain around {final_pred:.0f} ppm. Continue monitoring."
            else:
                if change > 30:
                    interpretation = f"📈 Methane predicted to rise to {final_pred:.0f} ppm. Monitor trend closely."
                else:
                    interpretation = f"✅ Methane expected to remain stable at approximately {final_pred:.0f} ppm."
        else:
            interpretation = "Insufficient data for prediction interpretation."
        
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
            "interpretation": interpretation,
            "cache_hit": False
        }
        
        # Cache the response
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
        "cache_config": {
            "ttl_seconds": 3600,
            "max_size": 100,
            "cache_enabled": True
        },
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