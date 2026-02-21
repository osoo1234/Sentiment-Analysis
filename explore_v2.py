import pandas as pd

def analyze_dataset(path):
    df = pd.read_csv(path)
    print("Columns:", df.columns.tolist())
    print("\nShape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    
    # Identify the text column
    text_col = 'text' if 'text' in df.columns else None
    if not text_col:
        text_col = [c for c in df.columns if 'tweet' in c.lower() or 'content' in c.lower()][0]
    
    print(f"\nText Column: {text_col}")
    print("\nSample Texts:")
    for t in df[text_col].head(10):
        print(f"- {t}")

    # Sentiment distribution if exists
    if 'airline_sentiment' in df.columns:
        print("\nSentiment Distribution:")
        print(df['airline_sentiment'].value_counts())

if __name__ == "__main__":
    analyze_dataset(r"d:\Desktop\Rivoo\Sentiment Analysis\Tweets.csv")
