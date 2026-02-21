import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from arabic_reshaper import reshape
from bidi.algorithm import get_display

def generate_eda_reports(df, text_col, target_col, output_dir='reports', lang='en'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Sentiment Distribution (Bar)
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_col, data=df, palette='viridis')
    plt.title(f'Sentiment Class Distribution ({lang})')
    plt.savefig(f'{output_dir}/sentiment_distribution_bar.png')
    plt.close()

    # Sentiment Distribution (Pie)
    plt.figure(figsize=(8, 8))
    df[target_col].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('viridis'))
    plt.title(f'Sentiment Class Pie Chart ({lang})')
    plt.savefig(f'{output_dir}/sentiment_distribution_pie.png')
    plt.close()
    
    # Word Clouds per sentiment
    sentiments = df[target_col].unique()
    for sentiment in sentiments:
        subset = df[df[target_col] == sentiment]
        text = " ".join(subset[text_col].astype(str))
        
        if text.strip():
            if lang == 'ar':
                text = reshape(text)
                text = get_display(text)
                
            font_path = 'C:/Windows/Fonts/arial.ttf' if os.name == 'nt' else None
            wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud - {sentiment}')
            plt.savefig(f'{output_dir}/wordcloud_{sentiment}.png')
            plt.close()

def identify_frequent_words(df, text_col, target_col, top_n=20):
    frequent_words = {}
    sentiments = df[target_col].unique()
    for sentiment in sentiments:
        subset = df[df[target_col] == sentiment]
        words = " ".join(subset[text_col].astype(str)).split()
        word_freq = pd.Series(words).value_counts().head(top_n)
        frequent_words[sentiment] = word_freq
    return frequent_words
