import streamlit as st
import cv2
import pandas as pd
import time
from collections import deque, Counter
from deepface import DeepFace

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Emotion AI Dashboard",
    page_icon="😊",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
body { background-color: #0e1117; }
.block-container { padding: 2rem 3rem; }

h1, h2, h3 { color: #00ffcc; }

.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}

.stButton > button {
    background-color: #00ffcc;
    color: black;
    border-radius: 10px;
    padding: 8px 16px;
}

hr { border: 1px solid #2a2d34; }
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.title("😊 Real-Time Emotion Detection Dashboard")
st.markdown("---")

# ------------------ SESSION STATE ------------------
if "run" not in st.session_state:
    st.session_state.run = False

if "emotion_data" not in st.session_state:
    st.session_state.emotion_data = []

if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = deque(maxlen=5)

# ------------------ BUTTONS ------------------
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("▶ Start Camera"):
        st.session_state.run = True
        st.session_state.emotion_history.clear()

with col_btn2:
    if st.button("⏹ Stop Camera"):
        st.session_state.run = False

# ------------------ LAYOUT ------------------
left, right = st.columns([2, 1])

frame_placeholder = left.empty()

# ------------------ EMOJI MAP ------------------
emoji_dict = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😄", "neutral": "😐",
    "sad": "😢", "surprise": "😲"
}

# ------------------ FACE DETECTOR ------------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if face_cascade.empty():
    st.error("Cascade file not found")
    st.stop()

# ------------------ CAMERA ------------------
current_emotion = "neutral"
confidence = 0

if st.session_state.run:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Camera not working")
        st.stop()

    prev_time = 0
    frame_count = 0
    cached_emotion = "neutral"
    cached_confidence = 0

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)

            face_roi = frame[y:y+h, x:x+w]

            if frame_count % 3 == 0:
                try:
                    result = DeepFace.analyze(
                        face_roi,
                        actions=['emotion'],
                        enforce_detection=False
                    )

                    if isinstance(result, list):
                        result = result[0]

                    emotion = result["dominant_emotion"]
                    conf = max(result["emotion"].values())

                    st.session_state.emotion_history.append(emotion)

                    if conf > 80:
                        cached_emotion = emotion
                    else:
                        cached_emotion = Counter(st.session_state.emotion_history).most_common(1)[0][0]

                    cached_confidence = conf

                except:
                    pass

            current_emotion = cached_emotion
            confidence = cached_confidence

            cv2.putText(frame, f"{current_emotion} ({confidence:.1f})",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            st.session_state.emotion_data.append({
                "time": time.strftime("%H:%M:%S"),
                "emotion": current_emotion,
                "confidence": confidence
            })

            if len(st.session_state.emotion_data) > 100:
                st.session_state.emotion_data.pop(0)

        frame_count += 1

        # FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time

        cv2.putText(frame, f"FPS: {int(fps)}", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        frame_placeholder.image(frame, channels="BGR")
        time.sleep(0.03)

    cap.release()

# ------------------ ANALYTICS ------------------
if len(st.session_state.emotion_data) > 0:
    df = pd.DataFrame(st.session_state.emotion_data)

    dominant = df["emotion"].mode()[0]

    # TOP METRICS
    col1, col2, col3 = st.columns(3)

    col1.metric("Current Emotion", f"{current_emotion} {emoji_dict.get(current_emotion,'')}")
    col2.metric("Confidence", f"{confidence:.1f}%")
    col3.metric("Dominant Emotion", f"{dominant} {emoji_dict.get(dominant,'')}")

    st.markdown("---")

    # LEFT SIDE
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Emotion Distribution")
        st.bar_chart(df["emotion"].value_counts())
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Emotion Timeline")
        st.line_chart(df["confidence"])
        st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT SIDE
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💡 AI Suggestion")

        if dominant in ["sad", "fear"]:
            st.warning("You seem stressed 😟")
        elif dominant == "angry":
            st.warning("You look angry 😡")
        elif dominant == "happy":
            st.success("You look happy 😄")
        else:
            st.info("You look neutral 😐")

        st.markdown("</div>", unsafe_allow_html=True)