import pandas as pd
import os

def analyze_data(file_path):
    print(f"--- Analyzing: {file_path} ---")
    
    # Load dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Basic Structure
    print("\n[Structure]")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\n[Column Data Types]")
    print(df.dtypes)
    
    # Missing Values
    print("\n[Missing Values]")
    print(df.isnull().sum())
    
    # Preview
    print("\n[First 5 Rows]")
    print(df.head())

    # Check for empty/short texts
    # Assuming there's a column for text, let's find it.
    text_col = None
    for col in df.columns:
        if 'text' in col.lower() or 'comment' in col.lower() or 'tweet' in col.lower():
            text_col = col
            break
    
    if text_col:
        print(f"\n[Text Column Identified: {text_col}]")
        df['text_len'] = df[text_col].astype(str).apply(len)
        print(f"Average text length: {df['text_len'].mean():.2f}")
        print(f"Empty/NaN texts: {df[text_col].isnull().sum()}")
        print(f"Texts with < 5 characters: {len(df[df['text_len'] < 5])}")
    else:
        print("\n[Warning] No text column automatically identified.")

if __name__ == "__main__":
    analyze_data(r'd:\Desktop\Rivoo\Sentiment Analysis\Tweets.csv')
