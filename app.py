import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
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
body {
    background-color: #0e1117;
}
.block-container {
    padding: 2rem 3rem;
}
h1, h2, h3 {
    color: #00ffcc;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    margin-bottom: 1rem;
}
.stButton > button {
    background-color: #00ffcc;
    color: black;
    border-radius: 10px;
    padding: 8px 16px;
}
hr {
    border: 1px solid #2a2d34;
}
.small-text {
    color: #aab3c5;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------ SESSION STATE ------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ LABELS / EMOJIS ------------------
emoji_dict = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲"
}

# ------------------ HELPERS ------------------
def analyze_emotion(pil_image: Image.Image):
    img = np.array(pil_image.convert("RGB"))

    result = DeepFace.analyze(
        img_path=img,
        actions=["emotion"],
        enforce_detection=False
    )

    if isinstance(result, list):
        result = result[0]

    dominant_emotion = result["dominant_emotion"]
    emotion_scores = result["emotion"]
    confidence = float(max(emotion_scores.values()))

    return dominant_emotion, confidence, emotion_scores


def add_to_history(emotion: str, confidence: float):
    st.session_state.history.append(
        {
            "emotion": emotion,
            "confidence": round(confidence, 2)
        }
    )
    if len(st.session_state.history) > 50:
        st.session_state.history.pop(0)


# ------------------ HEADER ------------------
st.title("😊 Emotion Detection AI Dashboard")
st.markdown("### Clean website version: mobile-friendly and browser-friendly")
st.markdown("---")

# ------------------ INPUT MODE ------------------
mode = st.radio(
    "Choose Input Method",
    ["Upload Image", "Take Photo"],
    horizontal=True
)

image_source = None

if mode == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image_source = Image.open(uploaded_file)

elif mode == "Take Photo":
    captured_file = st.camera_input("Take a photo")
    if captured_file is not None:
        image_source = Image.open(captured_file)

# ------------------ MAIN LAYOUT ------------------
left, right = st.columns([2, 1], gap="large")

current_emotion = None
current_confidence = None
emotion_scores = None

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📷 Input Preview")

    if image_source is not None:
        st.image(image_source, caption="Selected Image", use_container_width=True)

        with st.spinner("Analyzing emotion..."):
            try:
                current_emotion, current_confidence, emotion_scores = analyze_emotion(image_source)
                add_to_history(current_emotion, current_confidence)
                st.success(
                    f"Detected Emotion: {current_emotion.upper()} {emoji_dict.get(current_emotion, '')} "
                    f"({current_confidence:.1f}%)"
                )
            except Exception:
                st.error("Could not analyze the image clearly. Try another face image.")
    else:
        st.info("Upload an image or take a photo to begin.")

    st.markdown('</div>', unsafe_allow_html=True)

    if emotion_scores is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Emotion Probability Distribution")

        prob_df = pd.DataFrame(
            {
                "Emotion": list(emotion_scores.keys()),
                "Confidence": list(emotion_scores.values())
            }
        ).sort_values("Confidence", ascending=False)

        st.bar_chart(prob_df.set_index("Emotion"))
        st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Current Result")

    if current_emotion is not None:
        st.metric(
            "Current Emotion",
            f"{current_emotion.capitalize()} {emoji_dict.get(current_emotion, '')}"
        )
        st.metric(
            "Confidence",
            f"{current_confidence:.1f}%"
        )
    else:
        st.markdown('<p class="small-text">No analysis yet.</p>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💡 AI Suggestion")

    if current_emotion in ["sad", "fear"]:
        st.warning("You seem stressed. Take a short break and breathe slowly.")
    elif current_emotion == "angry":
        st.warning("You look tense. Try pausing and relaxing for a moment.")
    elif current_emotion == "happy":
        st.success("You seem happy. Keep smiling! 😄")
    elif current_emotion == "surprise":
        st.info("You look surprised. Stay calm and focused.")
    elif current_emotion == "neutral":
        st.info("You look neutral and composed.")
    elif current_emotion == "disgust":
        st.warning("You seem uncomfortable. Consider adjusting the environment.")
    else:
        st.markdown('<p class="small-text">Suggestions will appear after analysis.</p>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ HISTORY / ANALYTICS ------------------
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    dominant = history_df["emotion"].mode()[0]

    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Dominant Emotion",
        f"{dominant.capitalize()} {emoji_dict.get(dominant, '')}"
    )
    c2.metric(
        "Total Analyses",
        str(len(history_df))
    )
    c3.metric(
        "Average Confidence",
        f"{history_df['confidence'].mean():.1f}%"
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Analysis History")
    st.dataframe(history_df.tail(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📉 Emotion Distribution Over Session")
    st.bar_chart(history_df["emotion"].value_counts())
    st.markdown("</div>", unsafe_allow_html=True)

    csv_data = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Download Results CSV",
        data=csv_data,
        file_name="emotion_results.csv",
        mime="text/csv"
    )
