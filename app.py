import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Emotion AI",
    page_icon="🎭",
    layout="wide"
)

st.title("🎭 Emotion Detection AI")

# ==============================
# DEVICE DETECTION
# ==============================
user_agent = st.request.headers.get("User-Agent", "").lower()

is_mobile = any(x in user_agent for x in ["iphone", "android", "ipad"])

# ==============================
# MODE SELECTION
# ==============================
if is_mobile:
    st.info("📱 Mobile detected → Using Image Upload Mode")
    mode = "Upload Image"
else:
    mode = st.radio("Select Mode", ["Live Camera", "Upload Image"])

# ==============================
# IMAGE UPLOAD MODE
# ==============================
if mode == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing emotion..."):
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

                st.success(f"Emotion: {emotion} ({confidence:.1f}%)")

            except Exception:
                st.error("❌ Could not detect face clearly. Try another image.")

# ==============================
# LIVE CAMERA MODE
# ==============================
elif mode == "Live Camera":

    st.info("💻 Desktop mode → Live Camera Active")

    class EmotionDetector(VideoTransformerBase):
        def __init__(self):
            self.frame_count = 0
            self.last_emotion = ""
            self.last_confidence = 0

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            # 🔥 Analyze every 5th frame (performance boost)
            if self.frame_count % 5 == 0:
                try:
                    small_img = cv2.resize(img, (224, 224))

                    result = DeepFace.analyze(
                        small_img,
                        actions=['emotion'],
                        enforce_detection=False
                    )

                    if isinstance(result, list):
                        result = result[0]

                    self.last_emotion = result["dominant_emotion"]
                    self.last_confidence = max(result["emotion"].values())

                except Exception:
                    pass

            # Draw last known result
            if self.last_emotion:
                cv2.putText(
                    img,
                    f"{self.last_emotion} ({self.last_confidence:.1f}%)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            return img

    webrtc_streamer(
        key="emotion-detection",
        video_transformer_factory=EmotionDetector
    )
