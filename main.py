import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import firebase_admin
from firebase_admin import credentials, db

import matplotlib.pyplot as plt
import base64
from io import BytesIO
from contextlib import asynccontextmanager

# =========================================================
# CONFIG
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gas-monitoring")

app = FastAPI(title="Gas Monitoring API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

firebase_db = None

# =========================================================
# DEFAULT THRESHOLDS
# =========================================================
DEFAULT_THRESHOLDS = {
    "methane_low": 5,
    "methane_high": 10,
    "co2_low": 800,
    "co2_high": 1200,
    "ammonia_low": 10,
    "ammonia_high": 25,
    "temp_high": 35,
    "humidity_high": 85
}

# =========================================================
# FIREBASE INIT
# =========================================================
def initialize_firebase():
    """Initialize Firebase Realtime Database with better error handling"""
    global firebase_db
    try:
        # Check if Firebase is already initialized
        if firebase_db is not None:
            logger.info("Firebase already initialized")
            return True

        # Method 1: Check for service account JSON string in environment
        firebase_config_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
        if firebase_config_json:
            try:
                service_account_info = json.loads(firebase_config_json)
                cred = credentials.Certificate(service_account_info)
                logger.info("Using Firebase config from FIREBASE_SERVICE_ACCOUNT environment variable")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in FIREBASE_SERVICE_ACCOUNT: {e}")
                return False
        else:
            # Method 2: Individual environment variables
            private_key = os.getenv("FIREBASE_PRIVATE_KEY", "")
            if private_key:
                # Handle newline characters in private key
                private_key = private_key.replace('\\n', '\n')
            
            service_account_info = {
                "type": "service_account",
                "project_id": os.getenv("FIREBASE_PROJECT_ID"),
                "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
                "private_key": private_key,
                "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
                "client_id": os.getenv("FIREBASE_CLIENT_ID"),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL")
            }
            
            # Validate required fields
            required_fields = ["project_id", "private_key", "client_email"]
            for field in required_fields:
                if not service_account_info.get(field):
                    logger.error(f"Missing required Firebase config field: {field}")
                    return False
            
            cred = credentials.Certificate(service_account_info)
            logger.info("Using Firebase config from individual environment variables")

        database_url = os.getenv(
            "FIREBASE_DB_URL",
            "https://gasmonitoring-ec511-default-rtdb.asia-southeast1.firebasedatabase.app"
        )

        # Initialize Firebase app
        if not firebase_admin._apps:
            firebase_app = firebase_admin.initialize_app(cred, {
                "databaseURL": database_url
            })
            logger.info(f"Firebase app initialized: {firebase_app.name}")
        else:
            logger.info("Using existing Firebase app")
        
        # Get database reference
        firebase_db = db.reference()
        logger.info("Firebase Realtime Database initialized successfully")
        logger.info(f"Database URL: {database_url}")
        
        return True
        
    except Exception as e:
        logger.error(f"Firebase initialization failed: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Log environment variables (without sensitive values)
        env_info = {
            "FIREBASE_PROJECT_ID": bool(os.getenv("FIREBASE_PROJECT_ID")),
            "FIREBASE_CLIENT_EMAIL": bool(os.getenv("FIREBASE_CLIENT_EMAIL")),
            "FIREBASE_PRIVATE_KEY": bool(os.getenv("FIREBASE_PRIVATE_KEY")),
            "FIREBASE_PRIVATE_KEY_ID": bool(os.getenv("FIREBASE_PRIVATE_KEY_ID")),
            "FIREBASE_SERVICE_ACCOUNT": bool(os.getenv("FIREBASE_SERVICE_ACCOUNT")),
        }
        logger.info(f"Firebase environment variables: {env_info}")
        
        firebase_db = None
        return False

# =========================================================
# GET THRESHOLDS
# =========================================================
def get_thresholds(sensor_id: str) -> Dict[str, float]:
    try:
        ref = db.reference(f"sensorReadings/thresholds/{sensor_id}")
        data = ref.get()
        if not data:
            return DEFAULT_THRESHOLDS
        return {**DEFAULT_THRESHOLDS, **data}
    except:
        return DEFAULT_THRESHOLDS

# =========================================================
# FUZZY MEMBERSHIP
# =========================================================
def fuzzify(value, low, high):
    if value <= low:
        return 0
    elif value >= high:
        return 1
    return (value - low) / (high - low)

# =========================================================
# RISK EVALUATION
# =========================================================
def evaluate_risk_fuzzy(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sensor_id = data["sensor_id"]
    t = get_thresholds(sensor_id)

    methane = float(data["methane"])
    co2 = float(data["co2"])
    ammonia = float(data["ammonia"])
    temp = float(data["temperature"])
    hum = float(data["humidity"])

    # FUZZIFICATION
    m_score = fuzzify(methane, t["methane_low"], t["methane_high"])
    c_score = fuzzify(co2, t["co2_low"], t["co2_high"])
    a_score = fuzzify(ammonia, t["ammonia_low"], t["ammonia_high"])
    t_score = fuzzify(temp, 25, t["temp_high"])
    h_score = fuzzify(hum, 60, t["humidity_high"])

    # AGGREGATION (weighted)
    risk_score = ((m_score*0.30) + (c_score*0.25) + (a_score*0.25) + (t_score*0.10) + (h_score*0.10))*100

    # DEFUZZIFICATION
    if risk_score >= 75:
        level = "HIGH"
        msg = "🚨 Dangerous gas levels detected!"
    elif risk_score >= 40:
        level = "MEDIUM"
        msg = "⚠️ Air quality is unhealthy"
    else:
        return None

    return {
        "risk_level": level,
        "risk_score": round(risk_score,2),
        "message": msg,
        "timestamp": data["timestamp"],
        "values": {
            "methane": methane,
            "co2": co2,
            "ammonia": ammonia,
            "temperature": temp,
            "humidity": hum
        },
        "thresholds_used": t
    }

# =========================================================
# INSERT SENSOR DATA
# =========================================================
def insert_sensor_reading(data: Dict[str, Any]):
    try:
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        key = now.strftime('%Y%m%d_%H%M%S')

        payload = {**data, "timestamp": timestamp}
        root = db.reference()

        # latest
        root.child(f"sensorReadings/latest/{data['sensor_id']}").set(payload)
        # history
        root.child(f"sensorReadings/history/{data['sensor_id']}/{key}").set(payload)
        # alerts
        alert = evaluate_risk_fuzzy(payload)
        if alert:
            root.child(f"sensorReadings/alerts/{data['sensor_id']}/{key}").set(alert)

        return {"status":"success"}

    except Exception as e:
        logger.error(e)
        return {"status":"error","message":str(e)}

# =========================================================
# API ROUTES
# =========================================================
@app.post("/api/sensor/insert")
async def api_insert(data: Dict[str, Any]):
    required = ["sensor_id","methane","co2","ammonia","temperature","humidity"]
    for r in required:
        if r not in data:
            raise HTTPException(400,f"Missing {r}")
    result = insert_sensor_reading(data)
    if result["status"]=="error":
        raise HTTPException(500,result["message"])
    return result

@app.get("/api/sensor/latest/{sensor_id}")
async def get_latest(sensor_id: str):
    data = db.reference(f"sensorReadings/latest/{sensor_id}").get()
    if not data:
        raise HTTPException(404,"No data")
    return data

@app.get("/api/sensor/history/{sensor_id}")
async def get_history(sensor_id: str):
    return db.reference(f"sensorReadings/history/{sensor_id}").get()

@app.get("/api/sensor/alerts/{sensor_id}")
async def get_alerts(sensor_id: str):
    return db.reference(f"sensorReadings/alerts/{sensor_id}").get()

# =========================================================
# THRESHOLDS API
# =========================================================
@app.post("/api/thresholds/update")
async def update_thresholds(data: Dict[str, Any]):
    sensor_id = data.get("sensor_id")
    if not sensor_id:
        raise HTTPException(400,"sensor_id required")
    thresholds = {**DEFAULT_THRESHOLDS, **data}
    thresholds["updated_at"]=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    db.reference(f"sensorReadings/thresholds/{sensor_id}").set(thresholds)
    return {"status":"updated","thresholds":thresholds}

@app.get("/api/thresholds/{sensor_id}")
async def get_threshold(sensor_id: str):
    return get_thresholds(sensor_id)

# =========================================================
# METHANE VISUALIZATION
# =========================================================
@app.get("/api/visualization/methane/{sensor_id}")
async def methane_visualization(sensor_id: str):
    try:
        ref = db.reference(f"sensorReadings/history/{sensor_id}")
        data = ref.get()
        if not data:
            raise HTTPException(404,"No history data")

        # sort by timestamp key
        sorted_data = sorted(data.items(), key=lambda x: x[0])
        timestamps = []
        methane_values = []
        for key,val in sorted_data:
            timestamps.append(val["timestamp"])
            methane_values.append(val["methane"])

        # thresholds
        t = get_thresholds(sensor_id)

        # PLOT
        plt.figure(figsize=(10,5))
        plt.plot(timestamps,methane_values,marker='o',label="Methane")
        plt.axhline(y=t["methane_low"],linestyle='--',color='green',label='Low Threshold')
        plt.axhline(y=t["methane_high"],linestyle='--',color='red',label='High Threshold')

        # Highlight risk points
        colors = ["green" if v<t["methane_low"] else "orange" if v<t["methane_high"] else "red" for v in methane_values]
        plt.scatter(timestamps,methane_values,c=colors,s=50)
        plt.xticks(rotation=45)
        plt.title(f"Methane Levels - {sensor_id}")
        plt.ylabel("Methane (ppm)")
        plt.xlabel("Timestamp")
        plt.legend()
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf,format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return {"sensor_id":sensor_id,"image":img_base64}

    except Exception as e:
        raise HTTPException(500,str(e))

# =========================================================
# STARTUP
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🔥 STARTUP LOGIC
    if initialize_firebase():
        logger.info("🔥 Firebase Connected")
    else:
        logger.error("❌ Firebase Failed")

    yield  # <-- app runs here

    # 🔥 SHUTDOWN LOGIC (optional)
    logger.info("Shutting down application")

# Attach lifespan to FastAPI
app = FastAPI(
    title="Gas Monitoring API",
    version="3.0.0",
    lifespan=lifespan
)

# =========================================================
# RUN
# =========================================================
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)