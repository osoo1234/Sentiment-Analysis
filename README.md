# Sentiment & Aspect Analysis System

A professional, modular sentiment analysis system supporting English and Arabic, featuring automated preprocessing, EDA, model training, and a real-time Streamlit interface.

## 🚀 Features
- **Multi-language Support:** English & Arabic text processing.
- **Aspect Extraction:** Automatically classifies text into categories (Customer Service, Flight, Pricing, Baggage).
- **Multiple Models:** Includes Logistic Regression (Classical) and Multi-Layer Perceptron (Neural Network).
- **Automated EDA:** Generates word clouds, bar charts, and pie charts for data distribution.
- **Ready-to-use API:** Built with FastAPI for integration into other systems.
- **Streamlit Web UI:** A simple, elegant interface for real-time testing.

## 📁 Structure
- `src/`: Core logic (preprocessing, models, aspect extraction).
- `data/`: Raw and processed datasets.
- `models/`: Pre-trained model checkpoints (.joblib).
- `reports/`: Visualization reports and metrics.
- `outputs/`: Final analysis results in CSV/Excel.
- `Project.py`: Main Streamlit application.
- `main.py` / `main_ar.py`: Training pipelines.

## 🛠️ Performance
- **English Model:** ~80% Accuracy on Airline Tweets.
- **Arabic Model:** Prototype trained on small sample datasets (scalable with more data).

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit Interface:
   ```bash
   streamlit run Project.py
   ```

Developed as a professional AI solution.
