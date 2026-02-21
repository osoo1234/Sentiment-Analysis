import streamlit as st
import joblib
import os
from src.preprocess import TextPreprocessor
from src.aspect import AspectExtractor
import pandas as pd

# Set Page Config
st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4a90e2;
        color: white;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper function to load models
@st.cache_resource
def load_resources(lang):
    try:
        model = joblib.load(f'models/{lang}_logistic_model.joblib')
        vectorizer = joblib.load(f'models/{lang}_logistic_vectorizer.joblib')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading {lang} models: {e}")
        return None, None

def main():
    st.title("🤖 Sentiment & Aspect Analysis")
    st.markdown("---")
    
    # Sidebar for project info
    st.sidebar.title("Project Details")
    st.sidebar.info("""
    **Goal:** Analyze sentiment and detect key aspects in text.
    **Languages:** English & Arabic.
    **Models:** Logistic Regression & MLP.
    **Aspects:** Customer Service, Flight, Pricing, Baggage.
    """)
    
    # Language Selection
    lang_option = st.selectbox("Choose Language / اختر اللغة", ["English", "Arabic"])
    lang_code = "en" if lang_option == "English" else "ar"
    
    # Load resources
    model, vectorizer = load_resources(lang_code)
    
    if model and vectorizer:
        # Input Section
        label = "Enter Text" if lang_code == "en" else "أدخل النص هنا"
        text_input = st.text_area(label, height=150)
        
        if st.button("Analysis" if lang_code == "en" else "تحليل"):
            if text_input.strip():
                # Preprocessing
                preprocessor = TextPreprocessor(lang=lang_code)
                cleaned_text = preprocessor.preprocess(text_input)
                
                # Prediction
                vec = vectorizer.transform([cleaned_text])
                sentiment = model.predict(vec)[0]
                
                # Aspect Extraction
                aspect_extractor = AspectExtractor(language=lang_code)
                aspect = aspect_extractor.detect_aspect(text_input)
                
                # Display Results
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sentiment Result" if lang_code == "en" else "نتيجة المشاعر")
                    color = "green" if sentiment == "positive" else "red" if sentiment == "negative" else "orange"
                    st.markdown(f"<h2 style='color: {color};'>{sentiment.capitalize()}</h2>", unsafe_allow_html=True)
                
                with col2:
                    st.subheader("Detected Aspect" if lang_code == "en" else "الجانب المكتشف")
                    st.markdown(f"<h2>{aspect}</h2>", unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show Cleaned Text (Subtle)
                with st.expander("Show Internal Process"):
                    st.write(f"**Cleaned Text:** {cleaned_text}")
            else:
                st.warning("Please enter some text first!" if lang_code == "en" else "يرجى إدخال نص أولاً!")

    st.markdown("---")
    st.caption("Developed with ❤️ for Rivoo Sentiment Analysis Project")

if __name__ == "__main__":
    main()
