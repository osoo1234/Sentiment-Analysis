import pandas as pd
import os

def analyze_trends(file_path):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert 'tweet_created' to datetime
    if 'tweet_created' in df.columns:
        df['tweet_created'] = pd.to_datetime(df['tweet_created'], errors='coerce')
        df = df.dropna(subset=['tweet_created'])
        
        # Trend by Day
        df['date'] = df['tweet_created'].dt.date
        daily_sentiment = df.groupby(['date', 'final_sentiment']).size().unstack(fill_value=0)
        
        print("\n[Sentiment Trends over Time (Daily)]")
        print(daily_sentiment.head(10))
        
        # Peak Negativity Detection
        if 'negative' in daily_sentiment.columns:
            peak_neg_day = daily_sentiment['negative'].idxmax()
            print(f"\nPeak negativity detected on: {peak_neg_day} with {daily_sentiment.loc[peak_neg_day, 'negative']} negative tweets.")

    # Source Analysis (by Airline in this dataset)
    if 'airline' in df.columns:
        airline_sentiment = df.groupby(['airline', 'final_sentiment']).size().unstack(fill_value=0)
        print("\n[Sentiment by airline (Source Analysis)]")
        print(airline_sentiment)

if __name__ == "__main__":
    final_data = r"d:\Desktop\Rivoo\Sentiment Analysis\data\tweets_final.csv"
    if os.path.exists(final_data):
        analyze_trends(final_data)
