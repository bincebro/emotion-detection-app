import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ------------------ PAGE ------------------
st.set_page_config(page_title="Emotion AI Live", layout="wide")
st.title("🎥 Live Emotion Detection")

# ------------------ SESSION ------------------
if "emotion_data" not in st.session_state:
    st.session_state.emotion_data = []

# ------------------ VIDEO PROCESSOR ------------------
class EmotionProcessor(VideoProcessorBase):

    def __init__(self):
        self.last_emotion = "neutral"
        self.last_conf = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        try:
            result = DeepFace.analyze(
                img,
                actions=['emotion'],
                enforce_detection=False
            )

            if isinstance(result, list):
                result = result[0]

            emotion = result["dominant_emotion"]
            confidence = max(result["emotion"].values())

            self.last_emotion = emotion
            self.last_conf = confidence

            # Store data
            st.session_state.emotion_data.append({
                "time": time.strftime("%H:%M:%S"),
                "emotion": emotion,
                "confidence": confidence
            })

        except:
            pass

        # Display text
        cv2.putText(img, f"{self.last_emotion} ({self.last_conf:.1f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ------------------ START STREAM ------------------
webrtc_ctx = webrtc_streamer(
    key="emotion",
    video_processor_factory=EmotionProcessor
)

# ------------------ ANALYTICS ------------------
if len(st.session_state.emotion_data) > 0:
    df = pd.DataFrame(st.session_state.emotion_data)

    dominant = df["emotion"].mode()[0]

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("Current Emotion", dominant)
    col2.metric("Samples", len(df))
    col3.metric("Dominant Emotion", dominant)

    st.markdown("---")

    left, right = st.columns([2, 1])

    with left:
        st.subheader("📊 Distribution")
        st.bar_chart(df["emotion"].value_counts())

        st.subheader("📈 Confidence")
        st.line_chart(df["confidence"])

    with right:
        st.subheader("💡 Suggestion")

        if dominant in ["sad", "fear"]:
            st.warning("You seem stressed 😟")
        elif dominant == "angry":
            st.warning("You look angry 😡")
        elif dominant == "happy":
            st.success("You look happy 😄")
        else:
            st.info("You look neutral 😐")
