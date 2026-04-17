# Scream-detection-app
AI-based Scream Detection Web App that classifies emergency scream sounds using machine learning and real-time audio analysis with Streamlit.
# 🚨 AI-Based Scream Detection & Audio Classification Web App

This project is a machine learning-based web application that detects emergency scream sounds from audio files. It uses audio feature extraction techniques and a trained classification model to identify whether a given sound contains a scream or not.

---

## 🔥 Features
- Upload audio files (.wav / .mp3)
- Real-time scream detection
- Clean and interactive web interface
- Fast and lightweight prediction system

---

## 🧠 How It Works
- Extracts audio features like:
  - MFCC (Mel-Frequency Cepstral Coefficients)
  - Chroma features
  - Spectral centroid
  - Zero Crossing Rate (ZCR)
- Uses a trained **Random Forest Classifier**
- Applies feature scaling for better accuracy

---

## 🛠 Tech Stack
- Python
- NumPy
- Librosa
- Scikit-learn
- Streamlit

---

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
