import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    if not isinstance(text, str) or text == "":
        return "neutral", 0.0
    
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    
    if compound >= 0.05:
        return "positive", compound
    elif compound <= -0.05:
        return "negative", compound
    else:
        return "neutral", compound

def process_sentiment(file_path, output_path):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    # We use 'cleaned_text' for analysis
    print("Analyzing sentiment...")
    results = df['cleaned_text'].apply(get_vader_sentiment)
    
    df['sentiment_vader'] = [r[0] for r in results]
    df['sentiment_score'] = [r[1] for r in results]
    
    # Confidence score (normalized to 0-1)
    df['confidence_score'] = df['sentiment_score'].abs()
    
    # Map to final sentiment label
    df['final_sentiment'] = df['sentiment_vader']
    
    print(f"Saving results to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    base_path = r"d:\Desktop\Rivoo\Sentiment Analysis\data"
    
    # Process Tweets
    tweets_clean = os.path.join(base_path, "tweets_cleaned.csv")
    if os.path.exists(tweets_clean):
        process_sentiment(tweets_clean, os.path.join(base_path, "tweets_sentiment.csv"))
    
    # Process Arabic (Fallback for demo)
    arabic_clean = os.path.join(base_path, "arabic_cleaned.csv")
    if os.path.exists(arabic_clean):
        process_sentiment(arabic_clean, os.path.join(base_path, "arabic_sentiment.csv"))
