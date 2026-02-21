from src.preprocess import TextPreprocessor
from src.aspect import AspectExtractor
import joblib
import pandas as pd

def test_inference():
    print("Testing inference...")
    # Load model
    model = joblib.load('models/en_logistic_model.joblib')
    vectorizer = joblib.load('models/en_logistic_vectorizer.joblib')
    
    test_text = "The flight was delayed and the staff were rude."
    
    preprocessor = TextPreprocessor(lang='en')
    cleaned = preprocessor.preprocess(test_text)
    vec = vectorizer.transform([cleaned])
    sentiment = model.predict(vec)[0]
    
    aspect_extractor = AspectExtractor(language='en')
    aspect = aspect_extractor.detect_aspect(test_text)
    
    print(f"Text: {test_text}")
    print(f"Sentiment: {sentiment}")
    print(f"Aspect: {aspect}")
    
    # Test Arabic
    print("\nTesting Arabic inference...")
    ar_model = joblib.load('models/ar_logistic_model.joblib')
    ar_vec = joblib.load('models/ar_vectorizer.joblib') # Wait, I saved it as ar_logistic_vectorizer.joblib? No, check main_ar.py
    
    # In main_ar.py I used: model.save(f'models/ar_{m_type}_model.joblib', f'models/ar_{m_type}_vectorizer.joblib')
    ar_vec = joblib.load('models/ar_logistic_vectorizer.joblib')
    
    test_text_ar = "الرحلة كانت جميلة والخدمة ممتازة"
    preprocessor_ar = TextPreprocessor(lang='ar')
    cleaned_ar = preprocessor_ar.preprocess(test_text_ar)
    vec_ar = ar_vec.transform([cleaned_ar])
    sentiment_ar = ar_model.predict(vec_ar)[0]
    aspect_ar = AspectExtractor(language='ar').detect_aspect(test_text_ar)
    
    print(f"Text: {test_text_ar}")
    print(f"Sentiment: {sentiment_ar}")
    print(f"Aspect: {aspect_ar}")

if __name__ == "__main__":
    test_inference()
