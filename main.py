from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from collections import defaultdict, deque
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# STORAGE (Replace with Firebase later)
# =========================================
latest_data = {}
history_data = defaultdict(lambda: deque(maxlen=500))
alerts_data = defaultdict(list)

# =========================================
# DEFAULT THRESHOLDS
# =========================================
DEFAULT_THRESHOLDS = {
    "methane_low": 5,
    "methane_high": 10,
    "co2_high": 1200,
    "ammonia_high": 25
}

# =========================================
# ADAPTIVE THRESHOLDS
# =========================================
def get_thresholds(sensor_id):
    history = history_data[sensor_id]
    if len(history) < 5:
        return DEFAULT_THRESHOLDS

    methane_vals = [h["methane"] for h in history]
    avg = np.mean(methane_vals)
    mx = np.max(methane_vals)

    return {
        **DEFAULT_THRESHOLDS,
        "methane_low": round(avg * 0.7, 2),
        "methane_high": round(mx * 0.9, 2)
    }

# =========================================
# FUZZY LOGIC
# =========================================
def fuzzify(v, low, high):
    if v <= low: return 0
    if v >= high: return 1
    return (v - low) / (high - low)

def evaluate_risk(sensor_id, data):
    t = get_thresholds(sensor_id)

    m = fuzzify(data["methane"], t["methane_low"], t["methane_high"])
    c = fuzzify(data["co2"], 800, t["co2_high"])
    a = fuzzify(data.get("ammonia", 0), 10, t["ammonia_high"])

    score = (m*0.4 + c*0.3 + a*0.3) * 100

    if score >= 75:
        level = "HIGH"
        desc = "Danger! Immediate action required"
    elif score >= 40:
        level = "MEDIUM"
        desc = "Moderate risk"
    else:
        level = "LOW"
        desc = "Safe"

    # Methane explosion risk (5–15%)
    methane = data["methane"]
    explosion = 0
    if 5 <= methane <= 15:
        explosion = ((methane - 5) / 10) * 100

    return {
        "level": level,
        "score": round(score, 2),
        "description": desc,
        "methane_explosion_risk": round(explosion, 2)
    }

# =========================================
# INSERT SENSOR
# =========================================
@app.post("/api/sensor/insert")
def insert_sensor(data: dict):
    try:
        sid = data["sensor_id"]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        record = {
            "timestamp": now,
            "methane": float(data.get("methane", 0)),
            "co2": float(data.get("co2", 0)),
            "ammonia": float(data.get("ammonia", 0)),
            "temperature": float(data.get("temperature", 0)),
            "humidity": float(data.get("humidity", 0)),
        }

        latest_data[sid] = record
        history_data[sid].append(record)

        # Fuzzy alert
        risk = evaluate_risk(sid, record)
        if risk["level"] == "HIGH":
            alerts_data[sid].append({
                "type": "RISK",
                **risk,
                "timestamp": now
            })

        return {"status": "ok"}

    except Exception as e:
        raise HTTPException(500, str(e))

# =========================================
# SUMMARY
# =========================================
@app.get("/api/sensor/summary/{sensor_id}")
def summary(sensor_id: str):
    if sensor_id not in latest_data:
        return {"sensor_id": sensor_id}

    data = latest_data[sensor_id]
    t = get_thresholds(sensor_id)
    fuzzy = evaluate_risk(sensor_id, data)

    def status(v, low, high):
        if v < low: return "LOW"
        if v > high: return "HIGH"
        return "NORMAL"

    return {
        "sensor_id": sensor_id,
        "timestamp": data["timestamp"],
        "methane": {
            "value": data["methane"],
            "status": status(data["methane"], t["methane_low"], t["methane_high"])
        },
        "co2": {
            "value": data["co2"],
            "status": status(data["co2"], 800, t["co2_high"])
        },
        "ammonia": {
            "value": data["ammonia"],
            "status": status(data["ammonia"], 10, t["ammonia_high"])
        },
        "temperature": data["temperature"],
        "humidity": data["humidity"],
        "fuzzy_risk": fuzzy
    }

# =========================================
# CHART DATA
# =========================================
@app.get("/api/visualization/chart/{sensor_id}")
def chart(sensor_id: str):
    history = history_data[sensor_id]
    return {
        "timestamps": [h["timestamp"] for h in history],
        "methane": [h["methane"] for h in history],
        "co2": [h["co2"] for h in history],
        "ammonia": [h["ammonia"] for h in history],
    }

# =========================================
# FORECAST
# =========================================
@app.get("/api/forecast/{sensor_id}")
def forecast(sensor_id: str):
    history = history_data[sensor_id]
    if len(history) < 5:
        return {"forecast": []}

    values = [h["methane"] for h in history][-10:]
    slope = (values[-1] - values[0]) / len(values)

    future = []
    last = values[-1]
    for _ in range(5):
        last += slope
        future.append(round(last, 2))

    return {"forecast": future}

# =========================================
# ALERTS
# =========================================
@app.get("/api/sensor/alerts/{sensor_id}")
def alerts(sensor_id: str):
    return alerts_data[sensor_id]

# =========================================
# METRICS
# =========================================
@app.get("/api/model/metrics/{sensor_id}")
def metrics(sensor_id: str):
    history = history_data[sensor_id]
    if len(history) < 10:
        return {"RMSE": 0, "MAE": 0}

    y = np.array([h["methane"] for h in history][-10:])
    y_pred = np.roll(y, 1)

    rmse = np.sqrt(np.mean((y - y_pred)**2))
    mae = np.mean(np.abs(y - y_pred))

    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2)
    }

# =========================================
# HEATMAP
# =========================================
@app.get("/api/fuzzy/heatmap/{sensor_id}")
def heatmap(sensor_id: str):
    if sensor_id not in latest_data:
        return {"heatmap": []}

    data = latest_data[sensor_id]
    t = get_thresholds(sensor_id)

    return {
        "heatmap": [
            ["Methane", fuzzify(data["methane"], t["methane_low"], t["methane_high"])],
            ["CO2", fuzzify(data["co2"], 800, t["co2_high"])],
            ["Ammonia", fuzzify(data["ammonia"], 10, t["ammonia_high"])]
        ]
    }

# =========================================
# SENSORS LIST
# =========================================
@app.get("/api/sensors")
def sensors():
    return list(latest_data.keys())