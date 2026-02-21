import pandas as pd
import os
from src.preprocess import TextPreprocessor
from src.eda import generate_eda_reports
from src.models import SentimentModel, plot_confusion_matrix
from src.aspect import AspectExtractor
import json

def main():
    # 1. Load Data
    print("Loading data...")
    raw_df = pd.read_csv('data/raw/tweets.csv')
    
    # 2. Cleaning & Preprocessing
    print("Preprocessing data...")
    preprocessor = TextPreprocessor(lang='en')
    raw_df['cleaned_text'] = raw_df['text'].apply(preprocessor.preprocess)
    
    # 3. EDA
    print("Generating EDA reports...")
    generate_eda_reports(raw_df, 'cleaned_text', 'airline_sentiment', 'reports/en')
    
    # 4. Aspect Extraction
    print("Extracting aspects...")
    aspect_extractor = AspectExtractor(language='en')
    raw_df['aspect'] = raw_df['text'].apply(aspect_extractor.detect_aspect)
    
    # 5. Model Training (Multiple Models)
    print("Training Models...")
    model_types = ['logistic', 'mlp'] # Logistic (Classical) and MLP (Deep Learning)
    for m_type in model_types:
        print(f"Training {m_type} model...")
        model = SentimentModel(model_type=m_type)
        metrics, y_test, y_pred = model.train(raw_df['cleaned_text'], raw_df['airline_sentiment'])
        print(f"{m_type} Accuracy: {metrics['accuracy']:.4f}")
        
        # Save metrics for the best one or just save both
        with open(f'reports/en/metrics_{m_type}.json', 'w') as f:
            json.dump(metrics['report'], f, indent=4)
        
        labels = sorted(raw_df['airline_sentiment'].unique())
        plot_confusion_matrix(metrics['confusion_matrix'], labels, f'reports/en/confusion_matrix_{m_type}.png')
        
        model.save(f'models/en_{m_type}_model.joblib', f'models/en_{m_type}_vectorizer.joblib')
    
    # 7. Model Selection & Saving
    model.save('models/en_model.joblib', 'models/en_vectorizer.joblib')
    
    # 8. Output Final CSV
    print("Saving final output...")
    output_df = raw_df[['text', 'airline_sentiment', 'aspect']]
    output_df.to_csv('outputs/sentiment_aspect_analysis.csv', index=False)
    output_df.head(20).to_excel('outputs/analysis_sample.xlsx', index=False)
    
    print("Pipeline completed successfully for English data.")

if __name__ == "__main__":
    if not os.path.exists('reports/en'): os.makedirs('reports/en')
    if not os.path.exists('outputs'): os.makedirs('outputs')
    if not os.path.exists('models'): os.makedirs('models')
    main()
