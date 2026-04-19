from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from deepface import DeepFace

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Emotion API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = DeepFace.analyze(
        img,
        actions=['emotion'],
        enforce_detection=False
    )

    if isinstance(result, list):
        result = result[0]

    emotion = result["dominant_emotion"]
    confidence = max(result["emotion"].values())

    return {
        "emotion": emotion,
        "confidence": float(confidence)
    }
