import streamlit as st 
import pandas as pd
import os
import joblib
from pathlib import Path


# Set the page config with a wider layout and custom styling
st.set_page_config(
    page_title="Sentiment Analysis Pro", 
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .positive-result {
        background: linear-gradient(135deg, #a8e6cf 0%, #7fcdcd 100%);
        color: #2d5a27;
    }
    
    .negative-result {
        background: linear-gradient(135deg, #ffc3a0 0%, #ffb3ba 100%);
        color: #8b2635;
    }
    
    .neutral-result {
        background: linear-gradient(135deg, #a8c8ec 0%, #7fb3d3 100%);
        color: #2c5282;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🎯 Sentiment Analysis Pro</h1>
    <p>Advanced AI-powered sentiment detection across multiple data sources</p>
</div>
""", unsafe_allow_html=True)

# Define the path to the models directory using pathlib for cross-platform compatibility
models_dir = Path('src/models')

# Check if the models directory exists
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

# Sidebar for controls
with st.sidebar:
    st.markdown("## 🎛️ Analysis Controls")
    
    # Source selection with enhanced styling
    st.markdown("### Select Data Source")
    selected_source = st.selectbox(
        "Choose your model source:",
        sources,
        help="Select the trained model source for sentiment analysis"
    )
    
    # Model info
    st.markdown(f"""
    <div class="sidebar-content">
        <h4>📊 Model Info</h4>
        <p><strong>Selected Source:</strong> {selected_source.title()}</p>
        <p><strong>Available Models:</strong> {len(sources)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Instructions
    st.markdown("""
    <div class="sidebar-content">
        <h4>📝 Instructions</h4>
        <ol>
            <li>Select your preferred model source</li>
            <li>Enter text in the main panel</li>
            <li>Click 'Analyze Sentiment' to get results</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📝 Text Input")
    
    # Text input with enhanced styling
    user_text = st.text_area(
        "Enter your text for sentiment analysis:",
        height=150,
        placeholder="Type or paste your text here... (e.g., 'I love this product!' or 'This service was disappointing.')",
        help="Enter any text you'd like to analyze for sentiment"
    )
    
    # Predict button
    predict_button = st.button("🔍 Analyze Sentiment", type="primary")

with col2:
    st.markdown("## 💡 Quick Tips")
    
    st.markdown("""
    <div class="feature-card">
        <h4>🎯 Best Practices</h4>
        <ul>
            <li>Use clear, complete sentences</li>
            <li>Include context when possible</li>
            <li>Avoid excessive punctuation</li>
            <li>Try different sources for comparison</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Function definitions (unchanged)
def load_model_and_vectorizer(source):
    try:
        # Load the trained model
        model_path = f'src/models/{source}_model.joblib'
        vectorizer_path = f'src/models/{source}_vectorizer.joblib'
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)  # Load the vectorizer used for training
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        return None, None

def predict_sentiment(text, source):
    model, vectorizer = load_model_and_vectorizer(source)
    if model is None or vectorizer is None:
        return "Error"
    try:
        # Transform the input text to the numeric format using the vectorizer
        transformed_text = vectorizer.transform([text])
        
        # Make the prediction using the transformed text
        prediction = model.predict(transformed_text)[0]
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Error"

# Prediction logic
if predict_button:
    if user_text.strip():
        with st.spinner("🤖 Analyzing sentiment..."):
            # Perform sentiment prediction
            sentiment = predict_sentiment(user_text, selected_source)
            
            # Display result with enhanced styling
            st.markdown("## 📊 Analysis Result")
            
            if sentiment == "Positive":
                st.markdown(f"""
                <div class="result-card positive-result">
                    <h2>😊 Positive Sentiment</h2>
                    <p>The text expresses a positive sentiment!</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
                
            elif sentiment == "Negative":
                st.markdown(f"""
                <div class="result-card negative-result">
                    <h2>😞 Negative Sentiment</h2>
                    <p>The text expresses a negative sentiment.</p>
                </div>
                """, unsafe_allow_html=True)
                
            elif sentiment == "Neutral":
                st.markdown(f"""
                <div class="result-card neutral-result">
                    <h2>😐 Neutral Sentiment</h2>
                    <p>The text expresses a neutral sentiment.</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.warning(f"Received unexpected sentiment: {sentiment}")
                
            # Additional analysis info
            st.markdown(f"""
            <div class="feature-card">
                <h4>📈 Analysis Details</h4>
                <p><strong>Text Length:</strong> {len(user_text)} characters</p>
                <p><strong>Word Count:</strong> {len(user_text.split())} words</p>
                <p><strong>Model Source:</strong> {selected_source.title()}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text for analysis.")

# Sample data section with enhanced styling
st.markdown("---")
st.markdown("## 📋 Sample Data Preview")

sample_data_path = Path("data/twitter_validation.csv")

# Check if the sample data file exists
if not sample_data_path.exists():
    st.error(f"Sample data file not found at {sample_data_path.resolve()}. Please ensure the file exists.")
else:
    try:
        samples = pd.read_csv(sample_data_path, header=None, names=["serial_number", "Source", "Sentiment", "Text"])
        if samples.empty:
            st.warning("Sample data file is empty.")
        else:
            st.markdown("### Random Sample Texts")
            
            # Display samples in a more organized way
            sample_df = samples[['Source', 'Sentiment', 'Text']].sample(min(5, len(samples)))
            
            # Create columns for better layout
            for idx, row in sample_df.iterrows():
                with st.expander(f"📝 {row['Source']} - {row['Sentiment']} Sentiment"):
                    st.write(f"**Text:** {row['Text']}")
                    st.write(f"**Source:** {row['Source']}")
                    st.write(f"**Sentiment:** {row['Sentiment']}")
                    
    except Exception as e:
        st.error(f"Error loading sample data: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🚀 Powered by Advanced Machine Learning | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)