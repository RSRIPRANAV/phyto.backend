import os
import bcrypt
import joblib
import pandas as pd
import json
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient


app = FastAPI(title="PhytoSphere Pro AI Engine")

# --- CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MONGODB ATLAS CONNECTION ---
DEFAULT_URI = "mongodb+srv://Tirthesh:TIRTHESH@phytosensor.ohncgpf.mongodb.net/?appName=Phytosensor"
MONGO_URI = os.getenv("MONGO_URI", DEFAULT_URI)

db = None
users_col = None
logs_col = None

try:
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
        tls=True,
        tlsAllowInvalidCertificates=True
    )
    client.admin.command('ping')

    db = client["phytosphere_db"]
    users_col = db["users"]
    logs_col = db["logs"]
    print("✅ Connected to MongoDB Atlas")
except Exception as e:
    print(f"❌ MongoDB Error: {e}")

# --- LOAD MODEL (LAZY) ---
model = None

def get_model():
    global model
    if model is None:
        try:
            print("📦 Loading model...")
            model = joblib.load("phytorem_rf_model.pkl")
            print("✅ Model loaded")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model load failed: {str(e)}")
    return model

# --- LOAD PLANT MAP ---
try:
    with open("plant_map.json", "r") as f:
        plant_map = json.load(f)
    print("🌱 Plant map loaded")
except Exception as e:
    print(f"❌ Plant map load error: {e}")
    plant_map = {}

# --- DATA MODELS ---
class UserAuth(BaseModel):
    email: str
    password: str
    deviceId: str

class SensorPayload(BaseModel):
    cu: float
    cd: float
    pb: float
    deviceId: str

# --- AUTH ROUTES ---
@app.post("/register")
async def register(user: UserAuth):
    if users_col is None:
        raise HTTPException(status_code=503, detail="Database not connected.")

    if users_col.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered.")

    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())

    users_col.insert_one({
        "email": user.email,
        "password": hashed_password,
        "deviceId": user.deviceId,
        "created_at": datetime.utcnow()
    })

    return {"status": "success", "message": "Registration successful"}

@app.post("/login")
async def login(user: UserAuth):
    if users_col is None:
        raise HTTPException(status_code=503, detail="Database not connected.")

    db_user = users_col.find_one({"email": user.email, "deviceId": user.deviceId})

    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if bcrypt.checkpw(user.password.encode('utf-8'), db_user["password"]):
        return {
            "status": "success",
            "user": {
                "email": db_user["email"],
                "deviceId": db_user["deviceId"]
            }
        }

    raise HTTPException(status_code=401, detail="Incorrect password")

# --- PREDICT ---
@app.post("/predict")
async def predict(data: SensorPayload):
    if logs_col is None:
        raise HTTPException(status_code=503, detail="Database not connected.")

    try:
        model_instance = get_model()

        input_df = pd.DataFrame(
            [[data.cu, data.cd, data.pb]],
            columns=['Copper', 'Cadmium', 'Lead']
        )

        pred = model_instance.predict(input_df)[0]
        probs = model_instance.predict_proba(input_df)[0]

        confidence = float(max(probs) * 100)

        # Map contaminant → plants
        recommended_plants = plant_map.get(pred, [])[:5]

        # Save log
        logs_col.insert_one({
            "Id": data.deviceId,
            "metals_detected": {
                "Lead": data.pb,
                "Copper": data.cu,
                "Cadmium": data.cd
            },
            "timestamp": datetime.utcnow(),
            "prediction": pred,
            "confidence": confidence
        })

        return {
            "status": "success",
            "contaminant": pred,
            "confidence": round(confidence, 2),
            "recommended_plants": recommended_plants
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- MAIN ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)