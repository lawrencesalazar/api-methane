import os
import json
import logging
from datetime import datetime
from typing import Dict, Any
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

import firebase_admin
from firebase_admin import credentials, db

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

        methane_vals = [
            float(history[k]["methane"])
            for k in history if "methane" in history[k]
        ]

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
    m = fuzzify(data["methane"], t["methane_low"], t["methane_high"])
    c = fuzzify(data["co2"], 800, t["co2_high"])
    a = fuzzify(data["ammonia"], 10, t["ammonia_high"])

    score = (m*0.4 + c*0.3 + a*0.3)*100

    if score >= 75:
        return {"level":"HIGH","score":round(score,2)}
    elif score >= 40:
        return {"level":"MEDIUM","score":round(score,2)}
    return None

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

    except:
        return None

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

    alert = evaluate_risk(data)
    if alert:
        root.child(f"sensorReadings/alerts/{sid}/{key}").set(alert)

    predict = predict_trend(sid)
    if predict:
        root.child(f"sensorReadings/alerts/{sid}/{key}").set(predict)

    await manager.broadcast(data)

# =========================================================
# APIs
# =========================================================
@app.post("/api/sensor/insert")
async def api_insert(data: Dict[str, Any], bg: BackgroundTasks):
    bg.add_task(insert_sensor, data)
    return {"status":"ok"}

@app.get("/api/sensors")
def sensors():
    data = db.reference("sensorReadings/latest").get()
    return list(data.keys()) if data else []

@app.get("/api/visualization/chart/{sensor_id}")
def chart(sensor_id):
    data = db.reference(f"sensorReadings/history/{sensor_id}").get() or {}
    labels, m, c, a = [], [], [], []

    for k in sorted(data.keys()):
        r = data[k]
        labels.append(r["timestamp"])
        m.append(r["methane"])
        c.append(r["co2"])
        a.append(r["ammonia"])

    return {
        "labels": labels,
        "datasets": [
            {"label":"Methane","data":m},
            {"label":"CO2","data":c},
            {"label":"Ammonia","data":a},
        ]
    }

@app.get("/api/forecast/{sensor_id}")
def forecast(sensor_id):
    data = db.reference(f"sensorReadings/history/{sensor_id}").get()
    values = [float(data[k]["methane"]) for k in sorted(data.keys())[-10:]]

    slope = (values[-1]-values[0])/len(values)
    future = []
    last = values[-1]

    for i in range(5):
        last += slope
        future.append(round(last,2))

    return {"forecast":future}
# =========================================================
# THRESHOLD APIs
# =========================================================
@app.post("/api/thresholds/update")
def update_thresholds(data: Dict[str, Any]):
    sid = data.get("sensor_id")
    if not sid:
        raise HTTPException(400, "sensor_id required")

    db.reference(f"sensorReadings/thresholds/{sid}").set(data)
    return {"status": "updated", "data": data}


@app.get("/api/thresholds/{sensor_id}")
def get_thresholds(sensor_id: str):
    data = db.reference(f"sensorReadings/thresholds/{sensor_id}").get()
    return data if data else DEFAULT_THRESHOLDS
# =========================================================
# GAS STATUS INTERPRETATION
# =========================================================
def gas_status(value, low, high):
    if value < low:
        return "LOW"
    elif value > high:
        return "HIGH"
    return "NORMAL"
@app.get("/api/sensor/summary/{sensor_id}")
def sensor_summary(sensor_id: str):
    data = db.reference(f"sensorReadings/latest/{sensor_id}").get()

    if not data:
        raise HTTPException(404, "No data")

    t = get_adaptive_thresholds(sensor_id)

    return {
        "sensor_id": sensor_id,
        "timestamp": data["timestamp"],

        "methane": {
            "value": data["methane"],
            "status": gas_status(data["methane"], t["methane_low"], t["methane_high"])
        },
        "co2": {
            "value": data["co2"],
            "status": gas_status(data["co2"], 800, t["co2_high"])
        },
        "ammonia": {
            "value": data["ammonia"],
            "status": gas_status(data["ammonia"], 10, t["ammonia_high"])
        },
        "temperature": data.get("temperature", 0),
        "humidity": data.get("humidity", 0)
    }

# =========================================================
# PDF REPORT GENERATOR (CHAPTER 4)
# =========================================================
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

@app.get("/api/report/{sensor_id}")
def generate_report(sensor_id: str):
    try:
        filename = f"{sensor_id}_report.pdf"
        filepath = f"/tmp/{filename}"

        doc = SimpleDocTemplate(filepath)
        styles = getSampleStyleSheet()

        elements = []

        # TITLE
        elements.append(Paragraph("Methane Monitoring System Report", styles["Title"]))
        elements.append(Spacer(1, 10))

        # FETCH DATA
        latest = db.reference(f"sensorReadings/latest/{sensor_id}").get()
        forecast_data = forecast(sensor_id)

        if not latest:
            raise HTTPException(404, "No data available")

        # SUMMARY
        elements.append(Paragraph(f"Sensor ID: {sensor_id}", styles["Normal"]))
        elements.append(Paragraph(f"Timestamp: {latest['timestamp']}", styles["Normal"]))
        elements.append(Spacer(1, 10))

        elements.append(Paragraph(f"Methane: {latest['methane']} ppm", styles["Normal"]))
        elements.append(Paragraph(f"CO2: {latest['co2']} ppm", styles["Normal"]))
        elements.append(Paragraph(f"Ammonia: {latest['ammonia']} ppm", styles["Normal"]))
        elements.append(Paragraph(f"Temperature: {latest.get('temperature',0)} °C", styles["Normal"]))
        elements.append(Paragraph(f"Humidity: {latest.get('humidity',0)} %", styles["Normal"]))

        elements.append(Spacer(1, 15))

        # RISK
        risk = evaluate_risk(latest)
        if risk:
            elements.append(Paragraph(f"Risk Level: {risk['level']}", styles["Normal"]))
            elements.append(Paragraph(f"Risk Score: {risk['score']}", styles["Normal"]))
        else:
            elements.append(Paragraph("Risk Level: LOW", styles["Normal"]))

        elements.append(Spacer(1, 15))

        # FORECAST
        elements.append(Paragraph("Methane Forecast (Next Readings):", styles["Heading2"]))
        for f in forecast_data["forecast"]:
            elements.append(Paragraph(f"{f} ppm", styles["Normal"]))

        # BUILD PDF
        doc.build(elements)

        return {
            "status": "success",
            "download_url": f"/api/report/download/{sensor_id}"
        }

    except Exception as e:
        raise HTTPException(500, str(e))


# DOWNLOAD ENDPOINT
from fastapi.responses import FileResponse

@app.get("/api/report/download/{sensor_id}")
def download_report(sensor_id: str):
    filepath = f"/tmp/{sensor_id}_report.pdf"
    return FileResponse(filepath, media_type='application/pdf', filename=f"{sensor_id}_report.pdf")

import joblib

model = None

def load_model():
    global model
    try:
        model = joblib.load("model.pkl")
        logger.info("✅ AI model loaded")
    except:
        logger.warning("⚠️ No model found")

load_model()

@app.post("/api/predict")
def predict(data: Dict[str, Any]):
    try:
        if not model:
            raise HTTPException(500, "Model not loaded")

        features = [[
            float(data["co2"]),
            float(data["ammonia"]),
            float(data["temperature"]),
            float(data["humidity"])
        ]]

        prediction = model.predict(features)[0]

        return {
            "predicted_methane": round(prediction, 2),
            "input": data
        }

    except Exception as e:
        raise HTTPException(500, str(e))
# =========================================================
# 🔥 FUZZY HEATMAP API (ADD THIS)
# =========================================================
@app.get("/api/fuzzy/heatmap/{sensor_id}")
def fuzzy_heatmap(sensor_id: str):
    try:
        data = db.reference(f"sensorReadings/latest/{sensor_id}").get()

        if not data:
            return {
                "sensor_id": sensor_id,
                "heatmap": []
            }

        t = get_adaptive_thresholds(sensor_id)

        def safe_float(v):
            try:
                return float(v)
            except:
                return 0

        def fuzz(v, low, high):
            if v <= low: return 0
            if v >= high: return 1
            return (v - low) / (high - low)

        methane = safe_float(data.get("methane", 0))
        co2 = safe_float(data.get("co2", 0))
        ammonia = safe_float(data.get("ammonia", 0))

        heatmap = [
            ["Methane", round(fuzz(methane, t["methane_low"], t["methane_high"]), 2)],
            ["CO2", round(fuzz(co2, 800, t["co2_high"]), 2)],
            ["Ammonia", round(fuzz(ammonia, 10, t["ammonia_high"]), 2)]
        ]

        return {
            "sensor_id": sensor_id,
            "heatmap": heatmap
        }

    except Exception as e:
        logger.error(f"Heatmap error: {e}")
        return {
            "sensor_id": sensor_id,
            "heatmap": [],
            "error": str(e)
        }    

# =========================================================
# LIST SENSORS
# =========================================================
@app.get("/api/sensors")
def get_sensors():
    """
    Returns a list of available sensor IDs from Firebase latest readings.
    """
    try:
        data = db.reference("sensorReadings/latest").get()
        if not data:
            return []
        return list(data.keys())
    except Exception as e:
        logger.error(f"Failed to fetch sensors: {e}")
        return []



from fastapi.responses import JSONResponse
import numpy as np
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error

@app.get("/api/model/explain/{sensor_id}")
def model_explain(sensor_id: str):
    """
    Returns SHAP values for latest readings of a sensor.
    """
    try:
        if model is None:
            raise HTTPException(500, "Model not loaded")
        latest = db.reference(f"sensorReadings/latest/{sensor_id}").get()
        if not latest:
            raise HTTPException(404, "No data")
        
        features = np.array([[latest["co2"], latest["ammonia"], latest.get("temperature",0), latest.get("humidity",0)]])
        explainer = shap.Explainer(model)
        shap_values = explainer(features)
        return JSONResponse({"shap_values": shap_values.values.tolist(), "base_value": shap_values.base_values.tolist()})
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/model/metrics/{sensor_id}")
def model_metrics(sensor_id: str):
    """
    Evaluate model accuracy (RMSE, MAE) on latest 10 readings.
    """
    try:
        if model is None:
            raise HTTPException(500, "Model not loaded")
        history = db.reference(f"sensorReadings/history/{sensor_id}").get()
        if not history:
            raise HTTPException(404, "No data")
        keys = sorted(history.keys())[-10:]
        y_true = [float(history[k]["methane"]) for k in keys]
        X = np.array([[history[k]["co2"], history[k]["ammonia"], history[k].get("temperature",0), history[k].get("humidity",0)] for k in keys])
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return {"RMSE": round(rmse,2), "MAE": round(mae,2), "y_true": y_true, "y_pred": y_pred.tolist()}
    except Exception as e:
        raise HTTPException(500, str(e))
# =========================================================
# WEBSOCKET
# =========================================================
@app.websocket("/ws")
async def ws(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)