import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from deepface import DeepFace
from PIL import Image

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
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.title("😊 Real-Time Emotion Detection Dashboard")
st.markdown("---")

# ------------------ SESSION ------------------
if "emotion_data" not in st.session_state:
    st.session_state.emotion_data = []

# ------------------ CAMERA INPUT ------------------
st.subheader("📷 Capture Image")

img_file = st.camera_input("Take a picture")

# ------------------ PROCESS IMAGE ------------------
current_emotion = "N/A"
confidence = 0

if img_file is not None:
    image = Image.open(img_file)
    frame = np.array(image)

    with st.spinner("Analyzing Emotion..."):
        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False
            )

            if isinstance(result, list):
                result = result[0]

            current_emotion = result["dominant_emotion"]
            confidence = max(result["emotion"].values())

            # Save data
            st.session_state.emotion_data.append({
                "time": time.strftime("%H:%M:%S"),
                "emotion": current_emotion,
                "confidence": confidence
            })

        except Exception as e:
            st.error(f"Error: {e}")

    # Show image
    st.image(frame, caption=f"{current_emotion} ({confidence:.1f}%)")

# ------------------ ANALYTICS ------------------
if len(st.session_state.emotion_data) > 0:
    df = pd.DataFrame(st.session_state.emotion_data)

    dominant = df["emotion"].mode()[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Current Emotion", current_emotion)
    col2.metric("Confidence", f"{confidence:.1f}%")
    col3.metric("Dominant Emotion", dominant)

    st.markdown("---")

    # Charts
    left, right = st.columns([2, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Emotion Distribution")
        st.bar_chart(df["emotion"].value_counts())
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Confidence Timeline")
        st.line_chart(df["confidence"])
        st.markdown('</div>', unsafe_allow_html=True)

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

        st.markdown('</div>', unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("Emotion AI System • Live Web App")
