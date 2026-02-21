import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self, lang='en'):
        self.lang = lang
        if lang == 'en':
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
        elif lang == 'ar':
            # Basic Arabic stop words (can be expanded)
            self.stop_words = set(stopwords.words('arabic'))
    
    def clean_english(self, text):
        if not isinstance(text, str):
            return ""
        # Remove mentions, urls, and special characters
        text = re.sub(r'@\w+|http\S+|[^\w\s]', '', text).lower()
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Tokenize and remove stopwords + stemming
        words = text.split()
        cleaned_words = [self.stemmer.stem(w) for w in words if w not in self.stop_words]
        return " ".join(cleaned_words)

    def clean_arabic(self, text):
        if not isinstance(text, str):
            return ""
        # Remove diacritics
        arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ    | # Tatwil/Kashida
                         """, re.VERBOSE)
        text = re.sub(arabic_diacritics, '', text)
        
        # Normalize
        text = re.sub("[إأآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ؤ", "ء", text)
        text = re.sub("ئ", "ء", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("گ", "ك", text)
        
        # Remove punctuation, numbers, and latin chars
        text = re.sub(r'[^\u0600-\u06ff\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove repeated characters (e.g., هههههه -> هه)
        text = re.sub(r'(.)\1+', r'\1\1', text)
        
        # Tokenize and remove stopwords
        words = text.split()
        cleaned_words = [w for w in words if w not in self.stop_words]
        return " ".join(cleaned_words)

    def preprocess(self, text):
        if self.lang == 'en':
            return self.clean_english(text)
        elif self.lang == 'ar':
            return self.clean_arabic(text)
        return text

def load_and_preprocess(filepath, text_col, target_col, lang='en'):
    df = pd.read_csv(filepath)
    preprocessor = TextPreprocessor(lang=lang)
    df['cleaned_text'] = df[text_col].apply(preprocessor.preprocess)
    return df
