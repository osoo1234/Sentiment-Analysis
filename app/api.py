from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from src.preprocess import TextPreprocessor
from src.aspect import AspectExtractor

app = FastAPI(title="Sentiment & Aspect Analysis API")

# Load Models
models = {}
vectorizers = {}

try:
    models['en'] = joblib.load('models/en_logistic_model.joblib')
    vectorizers['en'] = joblib.load('models/en_logistic_vectorizer.joblib')
    models['ar'] = joblib.load('models/ar_logistic_model.joblib')
    vectorizers['ar'] = joblib.load('models/ar_logistic_vectorizer.joblib')
except:
    print("Models not fully loaded yet. Run pipelines first.")

class AnalysisRequest(BaseModel):
    text: str
    lang: str = "en"

class AnalysisResponse(BaseModel):
    sentiment: str
    aspect: str

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    lang = request.lang
    if lang not in ['en', 'ar']:
        raise HTTPException(status_code=400, detail="Language not supported. Use 'en' or 'ar'.")
    
    if lang not in models:
        raise HTTPException(status_code=500, detail="Model for this language not loaded.")
    
    preprocessor = TextPreprocessor(lang=lang)
    aspect_extractor = AspectExtractor(language=lang)
    
    cleaned_v = preprocessor.preprocess(request.text)
    vec = vectorizers[lang].transform([cleaned_v])
    sentiment = models[lang].predict(vec)[0]
    aspect = aspect_extractor.detect_aspect(request.text)
    
    return AnalysisResponse(sentiment=sentiment, aspect=aspect)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
