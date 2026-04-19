import cv2
import numpy as np
import os
from collections import deque, Counter
from tensorflow.keras.models import load_model

# ==============================
# FILE PATHS
# ==============================
MODEL_PATH = "emotion_model.keras"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# ==============================
# CHECK FILES
# ==============================
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found: {MODEL_PATH}")
    print("Train the model first using train_model.py")
    exit()

if not os.path.exists(CASCADE_PATH):
    print(f"❌ Haarcascade file not found: {CASCADE_PATH}")
    exit()

# ==============================
# LOAD MODEL
# ==============================
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# Emotion labels must match folder names used in training
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ==============================
# LOAD FACE DETECTOR
# ==============================
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    print("❌ Failed to load Haarcascade classifier")
    exit()

# ==============================
# PREDICTION SETTINGS
# ==============================
# Keep recent predictions for smoothing
emotion_history = deque(maxlen=12)

# Ignore very weak predictions
CONFIDENCE_THRESHOLD = 0.35

# ==============================
# START CAMERA
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera error. Try changing cv2.VideoCapture(0) to cv2.VideoCapture(1)")
    exit()

print("🎥 Starting Emotion Detection... Press 'q' to quit")

# ==============================
# REAL-TIME DETECTION LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame from camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Better face detection settings
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        # Add small margin around face for better crop
        margin = int(0.15 * w)
        x1 = max(x - margin, 0)
        y1 = max(y - margin, 0)
        x2 = min(x + w + margin, gray.shape[1])
        y2 = min(y + h + margin, gray.shape[0])

        face = gray[y1:y2, x1:x2]

        if face.size == 0:
            continue

        try:
            face = cv2.resize(face, (48, 48))
        except:
            continue

        # Improve contrast slightly
        face = cv2.equalizeHist(face)

        # Normalize
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)   # shape: (48,48,1)
        face = np.expand_dims(face, axis=0)    # shape: (1,48,48,1)

        # Predict
        prediction = model.predict(face, verbose=0)[0]
        confidence = float(np.max(prediction))
        predicted_index = int(np.argmax(prediction))
        predicted_emotion = emotion_labels[predicted_index]

        # Apply confidence threshold
        if confidence >= CONFIDENCE_THRESHOLD:
            emotion_history.append(predicted_emotion)

        # Smoothed emotion
        if len(emotion_history) > 0:
            emotion = Counter(emotion_history).most_common(1)[0][0]
        else:
            emotion = "uncertain"

        # Draw face box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Show label
        label = f"{emotion} ({confidence*100:.1f}%)"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()