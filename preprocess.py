import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase (for English)
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Handle Mentions (remove @name)
    text = re.sub(r'@\w+', '', text)
    
    # 4. Handle Hashtags (keep the word, remove #)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # 5. Handle Emojis (convert to text or remove)
    # We'll convert them to space-separated text descriptions if useful, 
    # but for simplicity here we'll just ensure they are handled.
    text = emoji.demojize(text, delimiters=(" ", " "))
    
    # 6. Basic Arabic Normalization (Even if dataset is English, logic is required)
    # Normalize Alif, Ta Marbuta, Ya
    text = re.sub(r'[أإآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[ىي]', 'ي', text)
    # Remove diacritics
    tashkeel = re.compile(r'[\u064B-\u0652]')
    text = re.sub(tashkeel, '', text)
    # Remove elongations
    text = re.sub(r'(.)\1+', r'\1\1', text) # Simple elongation reduction
    
    # 7. Remove Punctuation (except what we need)
    text = text.translate(str.maketrans('', '', string.punctuation.replace('_', '')))
    
    # 8. Tokenization
    tokens = nltk.word_tokenize(text)
    
    # 9. Stopwords removal (English & Arabic)
    stop_words = set(stopwords.words('english')).union(set(stopwords.words('arabic')))
    tokens = [w for w in tokens if w not in stop_words]
    
    # 10. Lemmatization (English)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return " ".join(tokens)

def run_preprocessing(input_path, output_path):
    try:
        print(f"Loading data from {input_path}...")
        df = pd.read_csv(input_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 decode failed, trying ISO-8859-1")
        df = pd.read_csv(input_path, encoding='ISO-8859-1')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    try:
        print("Preprocessing text...")
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        # Remove rows where cleaned_text is empty or too short
        initial_len = len(df)
        df = df[df['cleaned_text'].str.strip().apply(len) > 2]
        print(f"Removed {initial_len - len(df)} rows due to empty/short cleaned text.")
        
        print(f"Saving cleaned data to {output_path}...")
        df.to_csv(output_path, index=False, encoding='utf-8')
        print("Done.")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import os
    base_path = r"d:\Desktop\Rivoo\Sentiment Analysis\data\raw"
    output_base = r"d:\Desktop\Rivoo\Sentiment Analysis\data"
    os.makedirs(output_base, exist_ok=True)
    
    # Process Tweets
    tweets_path = os.path.join(base_path, "tweets.csv")
    if os.path.exists(tweets_path):
        run_preprocessing(tweets_path, os.path.join(output_base, "tweets_cleaned.csv"))
    
    # Process Arabic Samples
    arabic_path = os.path.join(base_path, "arabic_samples.csv")
    if os.path.exists(arabic_path):
        run_preprocessing(arabic_path, os.path.join(output_base, "arabic_cleaned.csv"))

