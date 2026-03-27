# main.py
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import firebase_admin
from firebase_admin import credentials, db

import joblib
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

import asyncio
import random

# =========================================================
# CONFIG
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gas-monitoring")
PH_TZ = ZoneInfo("Asia/Manila")
firebase_db = None

# =========================================================
# FIREBASE INIT
# =========================================================
def initialize_firebase():
    global firebase_db
    try:
        if firebase_db:
            return True
        service_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
        cred = credentials.Certificate(json.loads(service_json))
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                "databaseURL": os.environ.get("FIREBASE_DB_URL")
            })
        firebase_db = db.reference()
        logger.info("🔥 Firebase Connected")
        return True
    except Exception as e:
        logger.error(f"Firebase error: {e}")
        return False

def is_firebase_connected():
    try:
        if not firebase_admin._apps:
            return False
        db.reference("health_check").get()
        return True
    except:
        return False

# =========================================================
# LIFESPAN
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_firebase()
    yield

app = FastAPI(lifespan=lifespan)

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

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)

    async def broadcast(self, data):
        for ws in self.active_connections:
            try:
                await ws.send_json(data)
            except:
                pass

manager = ConnectionManager()

# =========================================================
# DEFAULT THRESHOLDS
# =========================================================
DEFAULT_THRESHOLDS = {
    "methane_low": 5,
    "methane_high": 10,
    "co2_high": 1200,
    "ammonia_high": 25
}

# =========================================================
# ADAPTIVE THRESHOLDS
# =========================================================
def get_adaptive_thresholds(sensor_id):
    try:
        history = db.reference(f"sensorReadings/history/{sensor_id}").get()
        if not history:
            return DEFAULT_THRESHOLDS
        methane_vals = [float(history[k]["methane"]) for k in history if "methane" in history[k]]
        if len(methane_vals) < 5:
            return DEFAULT_THRESHOLDS
        avg = sum(methane_vals)/len(methane_vals)
        mx = max(methane_vals)
        return {
            **DEFAULT_THRESHOLDS,
            "methane_low": avg * 0.7,
            "methane_high": mx * 0.9
        }
    except:
        return DEFAULT_THRESHOLDS

# =========================================================
# FUZZY LOGIC
# =========================================================
def fuzzify(v, low, high):
    if v <= low: return 0
    if v >= high: return 1
    return (v-low)/(high-low)

def evaluate_risk(data):
    t = get_adaptive_thresholds(data["sensor_id"])
    m = fuzzify(data.get("methane",0), t["methane_low"], t["methane_high"])
    c = fuzzify(data.get("co2",0), 800, t["co2_high"])
    a = fuzzify(data.get("ammonia",0), 10, t["ammonia_high"])
    score = (m*0.4 + c*0.3 + a*0.3)*100
    if score >= 75:
        return {"level":"HIGH","score":round(score,2)}
    elif score >= 40:
        return {"level":"MEDIUM","score":round(score,2)}
    return {"level":"LOW","score":round(score,2)}
# =========================================================
# FUZZY LOGIC ENDPOINT
# =========================================================
@app.get("/api/fuzzy/{sensor_id}")
def api_fuzzy(sensor_id: str):
    """
    Returns the fuzzy logic evaluation for the given sensor.
    Includes human-readable description for dashboard display.
    """
    # Get latest sensor data
    data = db.reference(f"sensorReadings/latest/{sensor_id}").get()
    if not data:
        return {"sensor_id": sensor_id, "error": "No data available"}

    # Compute fuzzy risk
    risk = evaluate_risk({"sensor_id": sensor_id, **data})

    # Prepare human-readable descriptions
    descriptions = {
        "methane": f"Methane level is {data.get('methane',0)} ppm, considered {risk['level']} risk",
        "co2": f"CO2 level is {data.get('co2',0)} ppm, considered {risk['level']} risk",
        "ammonia": f"Ammonia level is {data.get('ammonia',0)} ppm, considered {risk['level']} risk",
        "overall": f"Overall environmental risk level: {risk['level']} with a score of {risk['score']}"
    }

    return {
        "sensor_id": sensor_id,
        "methane": {"value": data.get("methane",0), "risk": risk["level"]},
        "co2": {"value": data.get("co2",0), "risk": risk["level"]},
        "ammonia": {"value": data.get("ammonia",0), "risk": risk["level"]},
        "score": risk["score"],
        "descriptions": descriptions
    }

# =========================================================
# PREDICTIVE ALERT
# =========================================================
def predict_trend(sensor_id):
    try:
        history = db.reference(f"sensorReadings/history/{sensor_id}").get()
        if not history:
            return None
        values = [float(history[k]["methane"]) for k in sorted(history.keys())[-10:]]
        if len(values) < 5:
            return None
        slope = values[-1] - values[0]
        if slope > 2:
            return {"type":"PREDICTIVE","message":"Methane rising fast","trend":round(slope,2)}
        return None
    except:
        return None

# =========================================================
# GAS STATUS
# =========================================================
def gas_status(value, low, high):
    if value < low: return "LOW"
    elif value > high: return "HIGH"
    return "NORMAL"

# =========================================================
# LOAD AI MODEL
# =========================================================
model = None
scaler = None
def load_ai_model():
    global model, scaler
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        logger.info("✅ AI model & scaler loaded")
    except Exception as e:
        logger.warning(f"⚠️ AI model load failed: {e}")
load_ai_model()

# =========================================================
# INSERT SENSOR
# =========================================================
async def insert_sensor(data):
    now = datetime.now(PH_TZ)
    key = now.strftime('%Y%m%d_%H%M%S')
    data["timestamp"] = now.strftime('%Y-%m-%d %H:%M:%S')
    sid = data["sensor_id"]
    root = db.reference()
    root.child(f"sensorReadings/latest/{sid}").set(data)
    root.child(f"sensorReadings/history/{sid}/{key}").set(data)

    # Fuzzy risk
    alert = evaluate_risk(data)
    if alert:
        root.child(f"sensorReadings/alerts/{sid}/{key}").set(alert)

    # Predictive trend
    trend = predict_trend(sid)
    if trend:
        root.child(f"sensorReadings/alerts/{sid}/{key}").set(trend)

    # AI Prediction
    try:
        features = [[float(data.get("co2",0)), float(data.get("ammonia",0)), float(data.get("temperature",0)), float(data.get("humidity",0))]]
        if scaler:
            features = scaler.transform(features)
        prediction = model.predict(features)[0]
        data["ai_prediction"] = round(prediction,2)
        root.child(f"sensorReadings/predictions/{sid}/{key}").set({
            "predicted_methane": round(prediction,2),
            "timestamp": data["timestamp"]
        })
    except:
        data["ai_prediction"] = None

    # Broadcast
    await manager.broadcast(data)

# =========================================================
# API ENDPOINTS
# =========================================================
@app.post("/api/sensor/insert")
async def api_insert(data: Dict[str, Any], bg: BackgroundTasks):
    bg.add_task(insert_sensor, data)
    return {"status":"ok"}

@app.get("/api/sensors")
def get_sensors():
    try:
        data = db.reference("sensorReadings/latest").get()
        return list(data.keys()) if data else []
    except Exception as e:
        logger.error(f"Failed to fetch sensors: {e}")
        return []

@app.get("/api/sensor/summary/{sensor_id}")
def sensor_summary(sensor_id: str):
    data = db.reference(f"sensorReadings/latest/{sensor_id}").get()
    if not data:
        return {"sensor_id": sensor_id, "methane":0,"co2":0,"ammonia":0,"temperature":0,"humidity":0,"ai_prediction":None}
    t = get_adaptive_thresholds(sensor_id)
    return {
        "sensor_id": sensor_id,
        "timestamp": data["timestamp"],
        "methane": {"value":data.get("methane",0), "status":gas_status(data.get("methane",0), t["methane_low"], t["methane_high"])},
        "co2": {"value":data.get("co2",0), "status":gas_status(data.get("co2",0),800,t["co2_high"])},
        "ammonia":{"value":data.get("ammonia",0), "status":gas_status(data.get("ammonia",0),10,t["ammonia_high"])},
        "temperature": data.get("temperature",0),
        "humidity": data.get("humidity",0),
        "ai_prediction": data.get("ai_prediction")
    }

@app.get("/api/forecast/{sensor_id}")
def forecast(sensor_id):
    data = db.reference(f"sensorReadings/history/{sensor_id}").get()
    if not data:
        return {"forecast":[]}
    values = [float(data[k].get("methane",0)) for k in sorted(data.keys())[-10:]]
    if not values:
        return {"forecast":[]}
    slope = (values[-1]-values[0])/len(values)
    future = []
    last = values[-1]
    for _ in range(5):
        last += slope
        future.append(round(last,2))
    return {"forecast":future}

@app.get("/api/fuzzy/heatmap/{sensor_id}")
def fuzzy_heatmap(sensor_id: str):
    data = db.reference(f"sensorReadings/latest/{sensor_id}").get()
    if not data:
        return {"sensor_id": sensor_id, "heatmap":[]}
    t = get_adaptive_thresholds(sensor_id)
    methane = float(data.get("methane",0))
    co2 = float(data.get("co2",0))
    ammonia = float(data.get("ammonia",0))
    heatmap = [
        ["Methane", round(fuzzify(methane,t["methane_low"],t["methane_high"]),2)],
        ["CO2", round(fuzzify(co2,800,t["co2_high"]),2)],
        ["Ammonia", round(fuzzify(ammonia,10,t["ammonia_high"]),2)]
    ]
    return {"sensor_id":sensor_id,"heatmap":heatmap}

@app.post("/api/predict")
def predict_endpoint(data: Dict[str,Any]):
    try:
        features = [[float(data.get("co2",0)), float(data.get("ammonia",0)), float(data.get("temperature",0)), float(data.get("humidity",0))]]
        if scaler:
            features = scaler.transform(features)
        prediction = model.predict(features)[0]
        return {"predicted_methane": round(prediction,2), "input": data}
    except Exception as e:
        raise HTTPException(500,str(e))

@app.get("/api/model/explain/{sensor_id}")
def model_explain(sensor_id: str):
    latest = db.reference(f"sensorReadings/latest/{sensor_id}").get()
    if not latest:
        return {"shap_values":[], "base_value":0, "features":["co2","ammonia","temperature","humidity"]}
    features = np.array([[latest.get("co2",0), latest.get("ammonia",0), latest.get("temperature",0), latest.get("humidity",0)]])
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(features)
        return JSONResponse({
            "shap_values": shap_values.values.tolist(),
            "base_value": shap_values.base_values.tolist(),
            "features": ["co2","ammonia","temperature","humidity"]
        })
    except:
        return {"shap_values":[], "base_value":0, "features":["co2","ammonia","temperature","humidity"]}

@app.get("/api/sensor/alerts/{sensor_id}")
def get_alerts(sensor_id: str):
    alerts = db.reference(f"sensorReadings/alerts/{sensor_id}").get()
    return list(alerts.values()) if alerts else []

@app.get("/api/report/{sensor_id}")
def generate_report(sensor_id: str):
    try:
        filename = f"{sensor_id}_report.pdf"
        filepath = f"/tmp/{filename}"
        doc = SimpleDocTemplate(filepath)
        styles = getSampleStyleSheet()
        elements = []

        latest = db.reference(f"sensorReadings/latest/{sensor_id}").get()
        forecast_data = forecast(sensor_id)
        if not latest:
            return {"status":"error","message":"No data"}

        # TITLE
        elements.append(Paragraph("Methane Monitoring Report", styles["Title"]))
        elements.append(Spacer(1,10))

        # SUMMARY
        elements.append(Paragraph(f"Sensor ID: {sensor_id}", styles["Normal"]))
        elements.append(Paragraph(f"Timestamp: {latest.get('timestamp','')}", styles["Normal"]))
        elements.append(Paragraph(f"Methane: {latest.get('methane',0)} ppm", styles["Normal"]))
        elements.append(Paragraph(f"CO2: {latest.get('co2',0)} ppm", styles["Normal"]))
        elements.append(Paragraph(f"Ammonia: {latest.get('ammonia',0)} ppm", styles["Normal"]))
        elements.append(Paragraph(f"Temperature: {latest.get('temperature',0)} °C", styles["Normal"]))
        elements.append(Paragraph(f"Humidity: {latest.get('humidity',0)} %", styles["Normal"]))
        elements.append(Spacer(1,10))

        # RISK
        risk = evaluate_risk({"sensor_id":sensor_id, **latest})
        elements.append(Paragraph(f"Risk Level: {risk['level']}", styles["Normal"]))
        elements.append(Paragraph(f"Risk Score: {risk['score']}", styles["Normal"]))
        elements.append(Spacer(1,10))

        # FORECAST
        elements.append(Paragraph("Methane Forecast (Next Readings):", styles["Heading2"]))
        for f in forecast_data.get("forecast",[]):
            elements.append(Paragraph(f"{f} ppm", styles["Normal"]))

        doc.build(elements)
        return {"status":"success", "download_url":f"/api/report/download/{sensor_id}"}
    except Exception as e:
        return {"status":"error","message":str(e)}
# =========================================================
# VISUALIZATION CHART
# =========================================================
@app.get("/api/visualization/chart/{sensor_id}")
def visualization_chart(sensor_id: str):
    try:
        history = db.reference(f"sensorReadings/history/{sensor_id}").get()
        if not history:
            return {"timestamps": [], "methane": [], "co2": [], "ammonia": []}
        keys = sorted(history.keys())
        timestamps = [history[k]["timestamp"] for k in keys]
        methane = [float(history[k].get("methane", 0)) for k in keys]
        co2 = [float(history[k].get("co2", 0)) for k in keys]
        ammonia = [float(history[k].get("ammonia", 0)) for k in keys]
        return {"timestamps": timestamps, "methane": methane, "co2": co2, "ammonia": ammonia}
    except Exception as e:
        raise HTTPException(500, f"Visualization chart error: {e}")


# =========================================================
# THRESHOLDS ENDPOINT
# =========================================================
@app.get("/api/thresholds/")
def get_all_thresholds():
    try:
        sensors = db.reference("sensorReadings/latest").get()
        result = {}
        if not sensors:
            return DEFAULT_THRESHOLDS
        for sid in sensors:
            result[sid] = get_adaptive_thresholds(sid)
        return result
    except Exception as e:
        raise HTTPException(500, f"Thresholds fetch error: {e}")


# =========================================================
# SENSOR ALERTS
# =========================================================
@app.get("/api/sensor/alerts/{sensor_id}")
def sensor_alerts(sensor_id: str):
    try:
        alerts = db.reference(f"sensorReadings/alerts/{sensor_id}").get()
        if not alerts:
            return []
        # sort by timestamp
        sorted_alerts = [alerts[k] for k in sorted(alerts.keys())]
        return sorted_alerts
    except Exception as e:
        raise HTTPException(500, f"Alerts fetch error: {e}")
    
@app.get("/api/report/download/{sensor_id}")
def download_report(sensor_id: str):
    filepath = f"/tmp/{sensor_id}_report.pdf"
    return FileResponse(filepath, media_type='application/pdf', filename=f"{sensor_id}_report.pdf")
# =========================================================
# MODEL METRICS (RMSE / MAE)
# =========================================================
@app.get("/api/model/metrics/{sensor_id}")
def model_metrics(sensor_id: str):
    try:
        history = db.reference(f"sensorReadings/history/{sensor_id}").get()
        if not history:
            raise HTTPException(404, "No history data for this sensor")

        # Take last 20 readings for metrics
        keys = sorted(history.keys())[-20:]
        y_true = [float(history[k]["methane"]) for k in keys]
        X = np.array([
            [
                float(history[k].get("co2", 0)),
                float(history[k].get("ammonia", 0)),
                float(history[k].get("temperature", 0)),
                float(history[k].get("humidity", 0))
            ] for k in keys
        ])

        if scaler:
            X = scaler.transform(X)

        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        return {
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "y_true": y_true,
            "y_pred": y_pred.tolist()
        }
    except Exception as e:
        raise HTTPException(500, f"Metrics computation failed: {e}")
# =========================================================
# WEBSOCKET
# =========================================================
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back latest data for demo
            await manager.broadcast({"message": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)