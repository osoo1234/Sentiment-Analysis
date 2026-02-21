import pandas as pd
import os
from src.preprocess import TextPreprocessor
from src.eda import generate_eda_reports
from src.models import SentimentModel, plot_confusion_matrix
from src.aspect import AspectExtractor
import json

def main():
    # 1. Load Data
    print("Loading Arabic data...")
    if not os.path.exists('data/raw/arabic_samples.csv'):
        print("Arabic sample file missing.")
        return
        
    raw_df = pd.read_csv('data/raw/arabic_samples.csv')
    
    # 2. Cleaning & Preprocessing
    print("Preprocessing Arabic data...")
    preprocessor = TextPreprocessor(lang='ar')
    raw_df['cleaned_text'] = raw_df['text'].apply(preprocessor.preprocess)
    
    # 3. EDA
    print("Generating Arabic EDA reports...")
    generate_eda_reports(raw_df, 'cleaned_text', 'sentiment', 'reports/ar', lang='ar')
    
    # 4. Aspect Extraction
    print("Extracting Arabic aspects...")
    aspect_extractor = AspectExtractor(language='ar')
    raw_df['aspect'] = raw_df['text'].apply(aspect_extractor.detect_aspect)
    
    # 5. Model Training (Multiple Models)
    print("Training Arabic Models...")
    model_types = ['logistic', 'mlp']
    for m_type in model_types:
        print(f"Training {m_type} model...")
        model = SentimentModel(model_type=m_type)
        metrics, y_test, y_pred = model.train(raw_df['cleaned_text'], raw_df['sentiment'])
        print(f"{m_type} Accuracy: {metrics['accuracy']:.4f}")
        
        with open(f'reports/ar/metrics_{m_type}.json', 'w') as f:
            json.dump(metrics['report'], f, indent=4)
        
        model.save(f'models/ar_{m_type}_model.joblib', f'models/ar_{m_type}_vectorizer.joblib')
        
    labels = sorted(raw_df['sentiment'].unique())
    #plot_confusion_matrix(metrics['confusion_matrix'], labels, 'reports/ar/confusion_matrix.png')
    
    # 7. Model Selection & Saving
    model.save('models/ar_model.joblib', 'models/ar_vectorizer.joblib')
    
    # 8. Output Final CSV
    print("Saving Arabic final output...")
    output_df = raw_df[['text', 'sentiment', 'aspect']]
    output_df.to_csv('outputs/arabic_analysis.csv', index=False)
    
    print("Arabic Pipeline completed successfully.")

if __name__ == "__main__":
    if not os.path.exists('reports/ar'): os.makedirs('reports/ar')
    if not os.path.exists('models'): os.makedirs('models')
    if not os.path.exists('outputs'): os.makedirs('outputs')
    main()
