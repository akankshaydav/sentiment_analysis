import streamlit as st 
import pandas as pd
import os
import joblib
from pathlib import Path

# Set the page configuration
st.set_page_config(page_title="Sentiment Analysis App", page_icon="🧠", layout="centered")

# Main app container
with st.container():
    st.markdown(
        "<h1 style='text-align: center; color: #4B8BBE;'>🧠 Sentiment Analysis App</h1>", 
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center;'>Analyze sentiment from various data sources using trained ML models</p>", 
        unsafe_allow_html=True
    )

# Path setup
models_dir = Path('src/models')
if not models_dir.exists():
    st.error(f"Models directory not found at {models_dir.resolve()}. Please ensure the path is correct.")
    st.stop()

# Load available sources
try:
    sources = [file.stem.split('_')[0] for file in models_dir.glob('*_model.joblib')]
    if not sources:
        st.error("No models found in the models directory. Please add model files with the naming convention '<source>_model.joblib'.")
        st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# UI layout with two columns
col1, col2 = st.columns([1, 2])

with col1:
    selected_source = st.selectbox("📌 Select a source:", sources)

with col2:
    user_text = st.text_area("✍️ Enter your text here:")

# Helper functions
def load_model_and_vectorizer(source):
    try:
        model_path = f'src/models/{source}_model.joblib'
        vectorizer_path = f'src/models/{source}_vectorizer.joblib'
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        return None, None

def predict_sentiment(text, source):
    model, vectorizer = load_model_and_vectorizer(source)
    if model is None or vectorizer is None:
        return "Error"
    try:
        transformed_text = vectorizer.transform([text])
        prediction = model.predict(transformed_text)[0]
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Error"

# Prediction trigger
st.markdown("---")
if st.button("🚀 Predict Sentiment"):
    if user_text.strip():
        sentiment = predict_sentiment(user_text, selected_source)
        st.subheader("🎯 Prediction Result:")
        if sentiment == "Positive":
            st.success(f"The sentiment is **{sentiment}** 😊")
        elif sentiment == "Negative":
            st.error(f"The sentiment is **{sentiment}** 😞")
        elif sentiment == "Neutral":
            st.info(f"The sentiment is **{sentiment}** 😐")
        else:
            st.warning(f"Received unexpected sentiment: **{sentiment}**")
    else:
        st.warning("⚠️ Please enter some text for prediction.")

# Divider
st.markdown("---")

# Sample texts section
st.subheader("📄 Sample Texts from Dataset")
sample_data_path = Path("data/twitter_validation.csv")
if not sample_data_path.exists():
    st.error(f"Sample data file not found at {sample_data_path.resolve()}. Please ensure the file exists.")
else:
    try:
        samples = pd.read_csv(sample_data_path, header=None, names=["serial_number", "Source", "Sentiment", "Text"])
        if samples.empty:
            st.warning("Sample data file is empty.")
        else:
            st.dataframe(samples[['Source', 'Sentiment', 'Text']].sample(min(5, len(samples))), use_container_width=True)
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
