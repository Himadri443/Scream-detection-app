import streamlit as st
import librosa
import numpy as np
import pickle
import tempfile

# Page config
st.set_page_config(page_title="AI Scream Detection", layout="centered")

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Feature extraction
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3)

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    spec = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)

    return np.hstack([mfcc, chroma, spec, zcr])

# UI Design
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 40px;
    color: #00FFAA;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
}
.result {
    text-align: center;
    font-size: 25px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🚨 AI Scream Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload audio to detect emergency scream</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    with st.spinner("Analyzing..."):
        features = extract_features(path)
        features = scaler.transform([features])
        prediction = model.predict(features)[0]

    if prediction == 1:
        st.markdown('<div class="result" style="color:red;">🚨 Scream Detected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result" style="color:green;">✅ No Scream</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Developed by Himadri Chandra")
